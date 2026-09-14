"""
The agent turn loop: messages -> one engine request -> tool calls -> tool results -> next turn, until
the answer is submitted or a budget runs out. One Agent owns one AgentRunResult; the rollout manager
runs many agents on threads and the engine batches their requests.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import typing as t
import uuid
from dataclasses import asdict

from .agent_config import AgentConfig, AgentRunResult, TrajectoryStep
from .agent_env import AgentEnv
from .agent_tools import DEFAULT_TOOLS, AgentTool, ToolCallResult, truncate_output
from .agent_utils import PARSE_ERROR_TOOL, ModelDialect, parameter_types_of

MAX_CONSECUTIVE_NO_TOOL_TURNS = 3   # nudged after each; the run ends with no_tool_call at the third in a row

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime
    from .rollout_reporter import RolloutReporter



class Agent:
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
        self.dialect = ModelDialect()   # replaced by the model's own dialect when run() starts
        self.seed = seed
        self.owns_env = agent_env is None
        self._env = agent_env
        self.run_results = AgentRunResult(agent_config=agent_config, seed=seed)
        self.agent_id = uuid.uuid4().hex  # Pins the agent to one engine replica, so its prefix cache serves the next turn.
        self.lock = threading.Lock()      # Parallel subagent result appends.
        self.messages: list[dict] = []
        self.tools: dict[str, AgentTool] = {}
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

    # ----------------------------------------------------------------------------- run
    def run(self) -> AgentRunResult:
        """
        Simple in/out run. Populates run_results. Never raises for a model or tool failure: the run
        ends with finish_reason "error" and the traceback in score_feedback. Compaction is for later.
        """
        config, results = self.agent_config, self.run_results
        start = time.time()
        self.dialect = ModelDialect.for_tokenizer(self.loaded_model.tokenizer)
        rendering = self.dialect.rendering
        tool_definitions = self._tool_definitions()
        parameter_types = parameter_types_of(tool_definitions)
        template_tools = rendering.template_tools(tool_definitions)
        self.messages = []
        system_prompt = rendering.system_prompt(config.system_prompt or "", tool_definitions)
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        self.messages.append({"role": "user", "content": config.user_prompt})
        self._count_ac_inputs()
        if self.reporter is not None:
            self.reporter.report_agent_start(self)
        no_tool_turns = 0
        errors = 0
        try:
            while True:
                if results.num_turns >= config.max_turns:
                    results.finish_reason = "max_turns"
                    break
                if time.time() - start > config.max_duration:
                    results.finish_reason = "max_duration"
                    break
                output = self.loaded_model.engine_submit(
                    self.messages, tools=template_tools, seed=self.seed + results.num_turns,
                    agent_id=self.agent_id, lora_name=config.lora_name, chat_kwargs=config.call_kwargs,
                )
                results.num_turns += 1
                results.num_input_tokens += output.prompt_token_count
                results.num_cached_input_tokens += output.cached_prompt_token_count
                results.num_output_tokens += output.output_token_count
                content, calls = self.dialect.parse(output.text, parameter_types)
                self.messages.append(rendering.assistant_message(content, calls))
                step = TrajectoryStep(role="assistant", content=content, tool_calls=calls)
                if not calls:
                    # A turn without a call gets the nudge; several in a row end the run.
                    results.trajectory.append(asdict(step))
                    no_tool_turns += 1
                    if no_tool_turns >= MAX_CONSECUTIVE_NO_TOOL_TURNS:
                        results.finish_reason = "no_tool_call"
                        break
                    self.messages.append({"role": "user", "content": self.dialect.nudge_message})
                    self._report_step()
                    continue
                no_tool_turns = 0
                call_results = self._execute_tool_calls(calls)
                errors += sum(1 for result in call_results if result.is_error)
                tool_step = TrajectoryStep(
                    role="tool",
                    content="\n\n".join(result.output for result in call_results),
                    tool_calls=calls,
                    tool_call_results=[result.output for result in call_results],
                    activations={call["id"]: result.ac_outputs for call, result in zip(calls, call_results) if result.ac_outputs},
                )
                results.trajectory.extend([asdict(step), asdict(tool_step)])
                self.messages.extend(rendering.tool_messages(calls, call_results))
                self._report_step()
                if any(result.is_final for result in call_results):
                    results.finish_reason = "submitted"
                    break
                if errors > config.max_tool_errors:
                    results.finish_reason = "max_tool_errors"
                    break
        except Exception as error:
            if _is_context_overflow(error):
                # The conversation outgrew the model's context (no compaction in slice 1): a budget, not a bug.
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

    def _report_step(self):
        if self.reporter is not None:
            self.reporter.report_agent_step(self)

    def _count_ac_inputs(self):
        ac_inputs = self.agent_config.ac_inputs
        if not ac_inputs:
            return
        text = json.dumps(ac_inputs, default=repr)
        self.run_results.num_ac_input_bytes = len(text.encode("utf-8"))
        try:
            self.run_results.num_ac_input_tokens = len(self.loaded_model.tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:
            self.run_results.num_ac_input_tokens = len(text) // 4

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
        output, ac_outputs = truncate_output(self._env if self._env is not None else None, result.output, call["id"])
        result.output = output
        if ac_outputs:
            result.ac_outputs = dict(result.ac_outputs) | ac_outputs
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
