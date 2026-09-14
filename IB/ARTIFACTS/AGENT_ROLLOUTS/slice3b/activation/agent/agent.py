"""
The agent turn loop: prompt tokens -> one engine request -> tool calls -> tool results -> next turn,
until the answer is submitted or a budget runs out. One Agent owns one AgentRunResult; the rollout
manager runs many agents on threads and the engine batches their requests.

The conversation is kept twice: as messages (the readable record for reports and the cache) and as
the token prefix the engine actually reads. The prefix is templated once for the first turn; after
that the agent appends its own sampled tokens verbatim and the dialect's wrapper for the tool results
or the nudge, so the model sees exactly what it wrote, every step records its token segment, and one
trajectory is one training sequence (agent_training).

Activation context (slice 3b): messages may carry activation_context parts (a compaction tree, the
parent's segment for a subagent, a large tool output, extra search passages). A part renders as a run
of placeholder tokens in the prefix; its rows come from the AC model once (the rollout queue and cache
batch and deduplicate them), are cast to the target dtype and kept for the run, and every request
ships them as prompt embeddings next to the token ids. A run is a sequence of segments: the soft
threshold makes compaction due, the model calls `compact(summary)` alone, the finished segment is
recorded under `compactions` and the next one starts from its tree (or the summary alone without an
AC model).
"""
from __future__ import annotations

import threading
import time
import traceback
import typing as t
import uuid
from copy import deepcopy
from dataclasses import asdict

import torch

from activation.common.ac_parts import COMPACTION_INSTRUCTIONS, ac_part, direct_parts

from .agent_config import AgentConfig, AgentRunResult, TrajectoryStep
from .agent_env import AgentEnv
from .agent_tools import DEFAULT_TOOLS, AgentTool, CompactionTool, ToolCallResult, tool_content, truncate_output
from .agent_utils import PARSE_ERROR_TOOL, ModelDialect, parameter_types_of

MAX_CONSECUTIVE_NO_TOOL_TURNS = 3   # nudged after each; the run ends with no_tool_call at the third in a row
MAX_COMPACTION_REFUSALS = 5         # turns that ignore a due compaction before the run ends with compaction_refused

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime
    from .rollout_reporter import RolloutReporter


class Agent:
    COMPACTION_DEMAND = "The context threshold is reached: call compact(summary) now, alone, before any other tool."

    def __init__(
        self,
        harness: "HarnessRuntime",
        agent_config: AgentConfig,
        agent_env: AgentEnv | None = None,
        parent_agent: "Agent | None" = None,
        reporter: "RolloutReporter | None" = None,
        seed: int = 0,
    ):
        self.harness = harness
        self.agent_config = agent_config
        self.parent_agent = parent_agent
        self.reporter = reporter
        self.dialect = ModelDialect()   # replaced by the model's own dialect when begin() runs
        self.seed = seed
        self.owns_env = agent_env is None
        self._env = agent_env
        self.run_results = AgentRunResult(agent_config=agent_config, seed=seed)
        self.agent_id = uuid.uuid4().hex  # Pins the agent to one engine replica, so its prefix cache serves the next turn.
        self.lock = threading.Lock()      # Parallel subagent result appends.
        self.messages: list[dict] = []    # dialect messages of the current segment, parts inline
        self.prefix: list[int] = []       # the token prompt of the next turn (placeholder ids at part positions)
        self.spans: list[tuple[int, int]] = []   # (start, end) of every part in the prefix, in order
        self.rows: list[torch.Tensor] = []       # the rows of those parts, target dtype on CPU, kept for the run (identical bytes each turn)
        self.segment_start = 0            # len(prompt_token_ids) of the current segment: the compaction delta counts from here
        self.compaction_due = False
        self.compaction_refusals = 0
        self.no_tool_turns = 0
        self.errors = 0
        self.last_sampled: list[int] = [] # the previous assistant turn's tokens, for join_continuation
        self.step_mode = False            # simulation: subagents do not run, nothing is submitted
        self.started = False
        self.tools: dict[str, AgentTool] = {}
        self.tool_definitions: list[dict] = []
        self.parameter_types: dict = {}
        self.template_tools: list[dict] | None = None
        self.template_kwargs: dict = {}
        self.finished = False
        self._initialize_tools()

    # ----------------------------------------------------------------------------- setup
    @property
    def agent_env(self) -> AgentEnv:
        """Created on first use, so a cached or failed-early run never starts a container."""
        if self._env is None:
            self._env = AgentEnv(self.agent_config.env_dockerfile_path, self.agent_config.env_args,
                                 default_image=self.harness.harness_config.agent_env_default_image,
                                 memory_limit_mb=self.agent_config.env_memory_limit_mb
                                 if self.agent_config.env_memory_limit_mb is not None
                                 else self.harness.harness_config.agent_env_memory_limit_mb)
        return self._env

    def _initialize_tools(self):
        specs = dict(DEFAULT_TOOLS) | dict(self.agent_config.tools)
        for name, (tool_class, kwargs) in specs.items():
            tool = tool_class(self.harness, self, **kwargs)
            if not tool.name:
                tool.name = name
            self.tools[name] = tool

    def _tool_definitions(self) -> list[dict]:
        return [tool.tool_definition() for tool in self.tools.values()]

    @property
    def loaded_model(self):
        config = self.agent_config
        assert config.model_name in self.harness.loaded_models, f"unknown model {config.model_name!r}"
        return self.harness.loaded_models[config.model_name]

    # ----------------------------------------------------------------------------- activation context
    @property
    def ac_model(self):
        name = self.agent_config.ac_model_name
        return self.harness.module_manager.get_ac_model(name) if name else None

    @property
    def row_dtype(self) -> torch.dtype:
        dtype = self.loaded_model.model_config.dtype
        return dtype if isinstance(dtype, torch.dtype) else torch.bfloat16

    def segment_messages(self) -> list[dict]:
        """The current segment without the system message: what a part sees as 'the parent so far'."""
        return [message for message in self.messages if message.get("role") != "system"]

    def context_part(self, ratio: float) -> dict:
        """The current segment as an activation_context part (task first; a compaction tree rides inside its first user message)."""
        return ac_part(deepcopy(self.segment_messages()), self.agent_config.ac_model_name, ratio, kind="parent_context")

    def _pad_id(self) -> int:
        tokenizer = self.loaded_model.tokenizer
        return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    def _part_lengths(self, messages: list[dict]) -> list[int]:
        parts = direct_parts(messages)
        if not parts:
            return []
        ac_model = self.ac_model
        assert ac_model is not None, "activation_context parts need agent_config.ac_model_name"
        return [ac_model.part_view_rows(part["messages"], part.get("compression_target")) for part in parts]

    def _encode_parts(self, messages: list[dict]) -> list[torch.Tensor]:
        """Rows of the parts in `messages`, in order, through the rollout queue (children first, cached), cast once to the target dtype."""
        parts = direct_parts(messages)
        if not parts:
            return []
        ac_model = self.ac_model
        assert ac_model is not None, "activation_context parts need agent_config.ac_model_name"
        futures = [ac_model.encode_async(part["messages"], part.get("compression_target")) for part in parts]
        rows = [future.result().detach().to(dtype=self.row_dtype).cpu().contiguous() for future in futures]
        self.run_results.num_ac_parts += len(rows)
        self.run_results.num_ac_rows += sum(int(r.shape[0]) for r in rows)
        return rows

    # ----------------------------------------------------------------------------- segments
    def first_user_messages(self) -> list[dict]:
        """messages_input with the task appended as a trailing text part of the last user message (today's prompt when empty)."""
        config = self.agent_config
        messages = deepcopy(config.messages_input)
        if not messages or messages[-1].get("role") != "user":
            messages.append({"role": "user", "content": config.user_prompt})
        elif config.user_prompt:
            content = messages[-1]["content"]
            content = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
            messages[-1]["content"] = content + [{"type": "text", "text": config.user_prompt}]
        return messages

    def begin(self) -> None:
        """Dialect, tool schemas and the first segment; templated once. Safe without an engine (tokenizer only)."""
        if self.started:
            return
        config, results = self.agent_config, self.run_results
        self.dialect = ModelDialect.for_tokenizer(self.loaded_model.tokenizer)
        self._prepare_tools()
        results.ac_model_name = config.ac_model_name
        results.ac_model_version = self.ac_model.version if self.ac_model is not None else None
        system = self.dialect.rendering.system_prompt(config.system_prompt or "", self.tool_definitions)
        self._begin_segment(([{"role": "system", "content": system}] if system else []) + self.first_user_messages())
        self.started = True

    def _prepare_tools(self) -> None:
        self.tool_definitions = self._tool_definitions()
        self.parameter_types = parameter_types_of(self.tool_definitions)
        self.template_tools = self.dialect.rendering.template_tools(self.tool_definitions)
        self.template_kwargs = self.loaded_model.chat_template_kwargs(self.agent_config.call_kwargs)

    def _begin_segment(self, messages: list[dict]) -> None:
        """A fresh prompt for a new segment (run start, or after a compaction): rows for its parts, a new start length."""
        results = self.run_results
        lengths = self._part_lengths(messages)
        ids, spans = self.dialect.prompt_tokens(self.loaded_model.tokenizer, messages, self.template_tools, self.template_kwargs,
                                                lengths, self._pad_id())
        rows = self._encode_parts(messages)
        assert [int(r.shape[0]) for r in rows] == [end - start for start, end in spans], "placeholder runs differ from the encoded rows"
        self.messages = list(messages)
        self.prefix, self.spans, self.rows = list(ids), list(spans), rows
        self.segment_start = len(ids)
        self.last_sampled = []
        results.prompt_token_ids, results.prompt_messages = list(ids), deepcopy(messages)
        results.prompt_ac_spans = [{"start": start, "length": end - start} for start, end in spans]
        self._after_append()

    def compact(self, summary: str) -> None:
        """Close the current segment into run_results.compactions and start the next one from its tree (or the summary alone)."""
        config, results = self.agent_config, self.run_results
        first_user = next(i for i, message in enumerate(self.messages) if message.get("role") == "user")
        inner, segment = self.messages[first_user], self.messages[first_user + 1:]          # verbatim; through the compact call
        record = AgentRunResult(
            agent_config=config, seed=self.seed, source=results.source, lora_name=results.lora_name, finish_reason="compacted",
            trajectory=results.trajectory, prompt_token_ids=results.prompt_token_ids, prompt_messages=results.prompt_messages,
            prompt_ac_spans=results.prompt_ac_spans, ac_model_name=results.ac_model_name, ac_model_version=results.ac_model_version,
            num_turns=sum(1 for step in results.trajectory if step["role"] == "assistant"),
        )
        results.compactions.append(record)
        results.trajectory = []
        text = COMPACTION_INSTRUCTIONS + (f"\n\nSummary written before compaction:\n{summary}" if summary.strip() else "")
        if self.ac_model is not None:
            tree = ac_part(deepcopy([inner] + segment), config.ac_model_name, config.ac_compaction_ratio, kind="compaction")
            after = {"role": "user", "content": [tree, {"type": "text", "text": "\n\n" + text}]}
        else:
            after = {"role": "user", "content": "The conversation so far was compacted.\n\n" + text}
        system = [message for message in self.messages if message.get("role") == "system"]
        self._begin_segment(system + self.first_user_messages() + [after])
        self.compaction_due, self.compaction_refusals = False, 0

    # ----------------------------------------------------------------------------- appending steps
    def record_assistant_turn(self, content: str, calls: list[dict], token_ids: list[int], logprobs: list[float]) -> None:
        message = self.dialect.rendering.assistant_message(content, calls)
        step = TrajectoryStep(role="assistant", content=content, tool_calls=calls, messages=[message], token_ids=list(token_ids),
                              logprobs=list(logprobs))
        self.messages.append(message)
        self.prefix += list(token_ids)
        self.last_sampled = list(token_ids)
        self.run_results.trajectory.append(asdict(step))

    def append_user_message(self, text: str) -> None:
        message = {"role": "user", "content": text}
        wrapper, spans = self._continuation([message])
        self._append_step(TrajectoryStep(role="user", content=text, messages=[message], token_ids=wrapper), [message], spans, [])

    def append_tool_results(self, calls: list[dict], results: list[ToolCallResult]) -> None:
        tool_messages = self.dialect.rendering.tool_messages(calls, results)          # content lists with parts, in order
        wrapper, spans = self._continuation(tool_messages)
        rows = self._encode_parts(tool_messages)
        step = TrajectoryStep(role="tool", content="\n\n".join(result.output for result in results), tool_calls=calls,
                              tool_call_results=[result.output for result in results], messages=tool_messages, token_ids=wrapper)
        self._append_step(step, tool_messages, spans, rows)

    def _continuation(self, new_messages: list[dict]) -> tuple[list[int], list[tuple[int, int]]]:
        """The template's wrapper between the sampled turn and the next generation prompt, with placeholder runs for the new parts."""
        lengths = self._part_lengths(new_messages)
        wrapper, spans = self.dialect.continuation_tokens(self.loaded_model.tokenizer, self.messages, new_messages, self.template_tools,
                                                          self.template_kwargs, lengths, self._pad_id())
        joined = self.dialect.join_continuation(self.last_sampled, wrapper)
        shift = len(wrapper) - len(joined)                                            # a dropped leading end-of-turn token shifts the spans
        return joined, [(start - shift, end - shift) for start, end in spans]

    def _append_step(self, step: TrajectoryStep, messages: list[dict], spans: list[tuple[int, int]], rows: list[torch.Tensor]) -> None:
        offset = len(self.prefix)
        assert [int(r.shape[0]) for r in rows] == [end - start for start, end in spans], "placeholder runs differ from the encoded rows"
        step.ac_spans = [{"start": start, "length": end - start} for start, end in spans]
        self.messages.extend(messages)
        self.prefix += step.token_ids
        self.spans += [(offset + start, offset + end) for start, end in spans]
        self.rows += rows
        self.run_results.trajectory.append(asdict(step))
        self._after_append()

    def _after_append(self) -> None:
        """Budgets that depend on the prefix length: the absolute cap ends the run, the soft delta makes compaction due."""
        config, results = self.agent_config, self.run_results
        if len(self.prefix) >= config.absolute_trajectory_cap and not results.finish_reason:
            results.finish_reason = "trajectory_cap"
        if len(self.prefix) - self.segment_start >= config.compaction_threshold_tokens:
            self.compaction_due = True

    # ----------------------------------------------------------------------------- requests
    def prepare_request(self) -> tuple[list[int], torch.Tensor | None, list[bool] | None]:
        """The next engine request: the prefix, and with parts the full-length tensor (zero rows at token positions) plus its mask."""
        if not self.spans:
            return list(self.prefix), None, None
        embeds = torch.zeros(len(self.prefix), self.loaded_model.model_config.model_description.d_model, dtype=self.row_dtype)
        mask = [True] * len(self.prefix)
        for (start, end), rows in zip(self.spans, self.rows):
            embeds[start:end] = rows
            mask[start:end] = [False] * (end - start)
        return list(self.prefix), embeds, mask

    def _submit(self, chat_kwargs: dict | None = None):
        config = self.agent_config
        prefix, embeds, mask = self.prepare_request()
        return self.loaded_model.engine_submit_tokens(
            prefix, seed=self.seed + self.run_results.num_turns, agent_id=self.agent_id, lora_name=config.lora_name,
            chat_kwargs=config.call_kwargs if chat_kwargs is None else chat_kwargs, record_sampling=config.record_sampling,
            prompt_embeds=embeds, prompt_is_token_ids=mask,
        )

    def submit_probe(self, max_tokens: int | None = None):
        """Submit the prepared request once without recording anything: the engine-interface check of the AC agent test. `max_tokens` overrides the sampling length."""
        chat_kwargs = None
        if max_tokens is not None:
            chat_kwargs = dict(self.agent_config.call_kwargs or {})
            chat_kwargs["sampling_params"] = dict(chat_kwargs.get("sampling_params") or {}) | {"max_tokens": max_tokens}
        return self._submit(chat_kwargs)

    def _check_engine_context(self) -> None:
        """The cap plus one turn's output must fit the engine: otherwise the cap could never be the binding budget."""
        engine_kwargs = self.loaded_model.model_config.engine_kwargs or {}
        max_len = engine_kwargs.get("max_model_len")
        if max_len is None:
            return
        max_tokens = self.loaded_model._merged_chat_kwargs(self.agent_config.call_kwargs)[0].max_tokens or 0
        assert self.agent_config.absolute_trajectory_cap + max_tokens <= max_len, (
            f"absolute_trajectory_cap {self.agent_config.absolute_trajectory_cap} + max_tokens {max_tokens} exceeds max_model_len {max_len}")

    # ----------------------------------------------------------------------------- run
    def run(self) -> AgentRunResult:
        """
        Simple in/out run. Populates run_results. Never raises for a model or tool failure: the run
        ends with finish_reason "error" and the traceback in score_feedback. Compaction restarts the
        segment in place; the earlier segments are in run_results.compactions.
        """
        config, results = self.agent_config, self.run_results
        start = time.time()
        try:
            self.begin()
            if self.reporter is not None:
                self.reporter.report_agent_start(self)
            self._check_engine_context()
            while True:
                if results.finish_reason:                                                        # trajectory_cap, set by an append
                    break
                if results.num_turns >= config.max_turns:
                    results.finish_reason = "max_turns"
                    break
                if time.time() - start > config.max_duration:
                    results.finish_reason = "max_duration"
                    break
                output = self._submit()
                if results.num_turns == 0:
                    results.lora_name = output.lora_name
                results.num_turns += 1
                results.num_input_tokens += output.prompt_token_count
                results.num_cached_input_tokens += output.cached_prompt_token_count
                results.num_output_tokens += output.output_token_count
                content, calls = self.dialect.parse(output.text, self.parameter_types)
                self.record_assistant_turn(content, calls, list(output.token_ids), list(output.logprobs or []))
                if not calls:
                    # A turn without a call gets the nudge (the compaction demand once due); several in a row end the run.
                    self.no_tool_turns += 1
                    if self.no_tool_turns >= MAX_CONSECUTIVE_NO_TOOL_TURNS:
                        results.finish_reason = "no_tool_call"
                        break
                    self.append_user_message(self.COMPACTION_DEMAND if self.compaction_due else self.dialect.nudge_message)
                    if self.compaction_due and self._refused():
                        break
                    self._report_step()
                    continue
                self.no_tool_turns = 0
                if self.compaction_due and not self._is_lone_compact(calls):
                    self.append_tool_results(calls, [self._refusal(call) for call in calls])     # nothing runs until compact
                    if self._refused():
                        break
                    self._report_step()
                    continue
                call_results = self._execute_tool_calls(calls)
                self.errors += sum(1 for result in call_results if result.is_error)
                if any(result.compacted for result in call_results):
                    self._report_step()                                                          # compact() began the new segment; no tool message follows
                    continue
                self.append_tool_results(calls, call_results)
                self._report_step()
                if any(result.is_final for result in call_results):
                    results.finish_reason = "submitted"
                    break
                if self.errors > config.max_tool_errors:
                    results.finish_reason = "max_tool_errors"
                    break
        except Exception as error:
            if _is_context_overflow(error):
                # The conversation outgrew the model's context: a budget, not a bug.
                results.finish_reason = "context_exceeded"
                results.score_feedback = str(error)[:500]
            else:
                results.finish_reason = "error"
                results.score_feedback = traceback.format_exc()
                print(f"Agent {self.agent_id[:8]} - error:\n{results.score_feedback}", flush=True)
        finally:
            results.duration = time.time() - start
            self.finished = True
            if self.reporter is not None:
                self.reporter.report_agent_finish(self)
        return results

    @staticmethod
    def _is_lone_compact(calls: list[dict]) -> bool:
        return len(calls) == 1 and calls[0]["name"] == CompactionTool.name

    def _refusal(self, call: dict) -> ToolCallResult:
        return ToolCallResult(output=f"Refused {call['name']}: {self.COMPACTION_DEMAND}")

    def _refused(self) -> bool:
        """Counts a turn that ignored a due compaction; True when the run must end."""
        self.compaction_refusals += 1
        if self.compaction_refusals >= MAX_COMPACTION_REFUSALS:
            self.run_results.finish_reason = "compaction_refused"
            return True
        return False

    def _report_step(self):
        if self.reporter is not None:
            self.reporter.report_agent_step(self)

    # ----------------------------------------------------------------------------- simulation and resume
    def simulated_run(self) -> AgentRunResult:
        """Step mode: the prompt is built (parts encoded), nothing is sampled, the answer is a dummy submission."""
        results = self.run_results
        self.step_mode = True
        self.begin()
        results.answer = f"Agent {self.agent_id[:8]} simulated run"
        results.finish_reason = "simulated"
        self.finished = True
        return results

    def simulate_step(self) -> dict:
        """
        One step without the engine. Executes the last assistant turn's pending calls (compaction, large
        outputs and search for real; subagents in step mode; refusals when compaction is due), appends
        their results, and prepares the next request. Returns what a test inspects; nothing is submitted.
        """
        self.step_mode = True
        self.begin()
        results = self.run_results
        executed = None
        last = results.trajectory[-1] if results.trajectory else None
        if last is not None and last["role"] == "assistant" and last["tool_calls"] and not results.finish_reason:
            calls = last["tool_calls"]
            if self.compaction_due and not self._is_lone_compact(calls):
                executed = [self._refusal(call) for call in calls]
                self.append_tool_results(calls, executed)
                self._refused()
            else:
                executed = self._execute_tool_calls(calls)
                if not any(result.compacted for result in executed):
                    self.append_tool_results(calls, executed)
        prefix, embeds, mask = self.prepare_request()
        return {
            "prefix_tokens": len(prefix), "spans": list(self.spans), "rows": [tuple(r.shape) for r in self.rows],
            "embeds_shape": None if embeds is None else tuple(embeds.shape), "row_positions": None if mask is None else mask.count(False),
            "compaction_due": self.compaction_due, "compactions": len(results.compactions), "finish_reason": results.finish_reason,
            "results": executed, "messages": len(self.messages),
        }

    @staticmethod
    def resume_from_run_result(harness: "HarnessRuntime", run_result: AgentRunResult, reporter: "RolloutReporter | None" = None) -> "Agent":
        """
        An agent positioned exactly after the recorded steps of `run_result`'s current segment: messages,
        prefix and spans from the record, rows re-encoded from the recorded parts (the record never
        stores rows). Counters continue from the record.
        """
        agent = Agent(harness, run_result.agent_config, reporter=reporter, seed=run_result.seed)
        agent.run_results = run_result
        agent.dialect = ModelDialect.for_tokenizer(agent.loaded_model.tokenizer)
        agent._prepare_tools()
        if agent.ac_model is not None and run_result.ac_model_version is not None and agent.ac_model.version != run_result.ac_model_version:
            print(f"Agent {agent.agent_id[:8]} - resuming rows with AC version {agent.ac_model.version}, recorded {run_result.ac_model_version}", flush=True)
        agent.messages = deepcopy(run_result.prompt_messages)
        agent.prefix = list(run_result.prompt_token_ids)
        agent.segment_start = len(agent.prefix)
        spans = [(span["start"], span["start"] + span["length"]) for span in run_result.prompt_ac_spans]
        parts_messages: list[dict] = list(run_result.prompt_messages)
        for step in run_result.trajectory:
            offset = len(agent.prefix)
            agent.messages.extend(deepcopy(step["messages"]))
            agent.prefix += list(step["token_ids"])
            spans += [(offset + span["start"], offset + span["start"] + span["length"]) for span in step["ac_spans"]]
            parts_messages += step["messages"]
            if step["role"] == "assistant":
                agent.last_sampled = list(step["token_ids"])
        agent.spans = spans
        agent.rows = agent._encode_parts(parts_messages)
        assert [int(r.shape[0]) for r in agent.rows] == [end - start for start, end in spans], "recorded spans differ from the re-encoded rows"
        agent.started = True
        agent._after_append()
        return agent

    # ----------------------------------------------------------------------------- tools
    def execute_tool_call(self, call: dict) -> ToolCallResult:
        """One call -> one truncated result. Unknown tools and bad arguments are errors the model reads."""
        name, arguments = call["name"], call.get("arguments") or {}
        if name == PARSE_ERROR_TOOL:
            result = ToolCallResult(output=self.dialect.parse_error_message(arguments), is_error=True)
        elif name not in self.tools:
            result = ToolCallResult(output=f"Unknown tool {name!r}. Available: {', '.join(self.tools)}.", is_error=True)
        else:
            try:
                result = self.tools[name].execute(**arguments)
            except TypeError as error:
                result = ToolCallResult(output=f"Bad arguments for {name}: {error}", is_error=True)
            except Exception as error:
                result = ToolCallResult(output=f"{name} failed: {type(error).__name__}: {error}", is_error=True)
        return self._finalize_result(call, result)

    def _finalize_result(self, call: dict, result: ToolCallResult) -> ToolCallResult:
        """Truncation and its part, applied to any tool's result (the one text part of a content list equals the visible output)."""
        visible, part = truncate_output(self._env, result.output, call["id"], agent=self if not result.compacted else None)
        if visible != result.output:
            content = [dict(item) for item in tool_content(result)]
            for item in content:
                if item.get("type") == "text" and item.get("text") == result.output:
                    item["text"] = visible
            result.content = ([part] if part is not None else []) + content
            result.output = visible
        return result

    def _execute_tool_calls(self, calls: list[dict]) -> list[ToolCallResult]:
        return [self.execute_tool_call(call) for call in calls]

    # ----------------------------------------------------------------------------- scoring and teardown
    def score(self) -> float:
        task = self.agent_config.dataset_task
        if task is None:
            return 0.0
        try:
            self.run_results.score = float(task.score(self.run_results))
        except Exception as error:
            self.run_results.score = 0.0
            self.run_results.score_feedback = f"scoring failed: {type(error).__name__}: {error}"
        return self.run_results.score

    def shutdown(self):
        if not self.owns_env or self._env is None:
            return
        self._env.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


def _is_context_overflow(error: BaseException) -> bool:
    """vLLM's validation error for a prompt over max_model_len (its class differs across versions)."""
    text = str(error)
    return "maximum context length" in text or "max_model_len" in text or "longer than the maximum" in text
