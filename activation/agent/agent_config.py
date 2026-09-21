"""
Agent configuration, trajectory steps and run results: plain dataclasses that serialize to JSON so a
rollout can be cached and read back without the engine.
"""
from __future__ import annotations

import json
import typing as t
from dataclasses import dataclass, field, replace
from pathlib import Path

from activation.common.utils import class_spec_name, resolve_class_name

if t.TYPE_CHECKING:
    from .agent_tools import AgentTool
    from .agent_env import AgentEnvSetup
    from activation.harness import HarnessRuntime
    from activation.dataset import DatasetTask


@dataclass
class AgentConfig:
    # Inputs
    system_prompt: str = ""                                   # System prompt.
    user_prompt: str = ""                                     # User prompt (the task).
    messages_input: list[dict] = field(default_factory=list)  # Dialect messages ahead of the task: content may be ordered text and
                                                              # activation_context parts. The task (user_prompt) is appended as a trailing
                                                              # text part of the last user message; empty = today's prompt.
    dataset_task: "DatasetTask | None" = None                 # Set if this corresponds to a specific dataset task.
    agent_name: str = "main" # Used to tag trajectories with their subagent.
    metadata: dict[str, str] = field(default_factory=dict)   # Labels for reports (benchmark, task kind, teacher kind, template); not part of the cache key.

    # Model calls.
    model_name: str | None = None                             # A harness model name; its engine serves the rollout.
    lora_name: str | None = None                              # Adapter on the serving engine (plumbed, untested).
    ac_model_name: str | None = None # the activation context model to use.
    call_kwargs: dict | None = None                           # Engine chat kwargs (sampling_params inside), merged like engine_chat_many.
    record_sampling: bool = True                              # Ask the engine for the sampled token ids and their log-probs (training needs them).

    # Environment. Every agent gets the shell / python / parallel_tool_call / submit_answer tools.
    env_dockerfile_path: str | None = None                    # None: the harness default image.
    env_args: dict[str, str] | None = None                    # Extra `podman run` flags, e.g. {"--env": "PYTHONHASHSEED=0"}.
    env_memory_limit_mb: int | None = None                    # Sandbox memory; None = the harness default (agent_env_memory_limit_mb). Hard container limit where cgroups exist, ulimit -v at 85% per process always.
    tools: dict[str, tuple[type["AgentTool"], dict] | None] = field(default_factory=dict)  # name -> (class, constructor kwargs), added to the
                                                              # defaults; None removes a default tool (subagents lose `subagent` this way).
    env_setups: dict[str, tuple[type["AgentEnvSetup"], dict]] = field(default_factory=dict)  # name -> (setup class, constructor kwargs). Run once,
    agentic_program: tuple[type["AgenticProgram"], dict] | None = None   # the program that owns the rollout (Agent.run_program); None: the model loop
                                                              # in order, when this agent's env is created; never for subagents (they share the env).

    # Budgets
    max_turns: int = 20                                       # total over the run, compaction segments included
    max_tool_errors: int = 5
    max_duration: float = 600                                 # seconds
    compaction_threshold_tokens: int = 32768                  # soft delta over the segment's start length (system, first messages, tree); exceeded by at most one turn, then compaction is due
    absolute_trajectory_cap: int = 50_000                     # prefix length that ends the run with finish_reason "trajectory_cap"
    # Activation-context ratios (rows per side token) of the four channels.
    ac_compaction_ratio: float = 1.0 / 10.0                   # the finished segment as the tree part of the next prompt
    ac_subagent_ratio: float = 1.0 / 20.0                     # parent -> child prompt, child -> parent result, and the nested parent context of the parts below
    ac_tool_output_ratio: float = 1.0 / 40.0                  # a truncated tool output's full text
    ac_search_ratio: float = 1.0 / 20.0                       # semantic search passages beyond top_k

    def serialize(self) -> dict:
        data = {key: value for key, value in self.__dict__.items() if key not in ("dataset_task", "tools", "env_setups", "agentic_program")}
        data["tools"] = {name: None if spec is None else _class_spec(*spec) for name, spec in self.tools.items()}
        data["env_setups"] = {name: _class_spec(cls, kwargs) for name, (cls, kwargs) in self.env_setups.items()}
        data["agentic_program"] = None if self.agentic_program is None else _class_spec(*self.agentic_program)
        data["messages_input"] = _jsonable(self.messages_input)
        task = self.dataset_task
        data["dataset_task"] = None if task is None else {
            "task_id": task.task_id, "dataset_id": task.dataset_id, "reference_metrics_kind": str(task.reference_metrics_kind),
            "gold_answer": task.gold_answer, "gold_answer_aliases": list(task.gold_answer_aliases), "agent_prompt": task.agent_prompt,
        }
        return data

    @staticmethod
    def deserialize(data: dict, harness: "HarnessRuntime | None" = None, strict: bool = False) -> "AgentConfig":
        """
        The dataset task is looked up in the harness when it holds the dataset, else rebuilt from the stored fields.
        Class specs (tools, env setups, the program) resolve leniently by default: a class that is not importable here
        becomes a MissingClass placeholder that keeps the spec and errs on construction, so records can be read for
        training without the classes present. Callers that will run the config pass `strict=True`.
        """
        from activation.dataset import DatasetTask, DatasetTaskMetricsKind
        data = dict(data)
        data.pop("ac_inputs", None)                                                   # rows written before slice 3b
        data.pop("enable_ac_communication", None)                                     # rows written before P4: an agent with an AC model renders every channel
        resolve = lambda spec: _resolve_class_spec(spec, strict=strict)
        tools = {name: None if spec is None else resolve(spec) for name, spec in (data.pop("tools", None) or {}).items()}
        setups = {name: resolve(spec) for name, spec in (data.pop("env_setups", None) or {}).items()}
        program_spec = data.pop("agentic_program", None)
        program = None if program_spec is None else resolve(program_spec)
        task_data = data.pop("dataset_task", None)
        task = None
        if task_data is not None:
            loaded = harness.dataset_manager.loaded_datasets.get(task_data["dataset_id"]) if harness is not None else None
            task = loaded.scorable_tasks.get(task_data["task_id"]) if loaded is not None else None
            if task is None:
                task = DatasetTask(
                    task_id=task_data["task_id"], dataset_id=task_data["dataset_id"], task_datum={},
                    reference_metrics_kind=DatasetTaskMetricsKind(task_data["reference_metrics_kind"]),
                    gold_answer=task_data["gold_answer"], gold_answer_aliases=list(task_data["gold_answer_aliases"]),
                    agent_prompt=task_data.get("agent_prompt", ""),
                )
        return AgentConfig(tools=tools, env_setups=setups, agentic_program=program, dataset_task=task, **data)


def _class_spec(cls: type, kwargs: dict) -> dict:
    """{"class": "module:Qualname", "kwargs": {...}}; a MissingClass placeholder re-serializes to the names it was read with."""
    return {"class": class_spec_name(cls), "kwargs": _jsonable(kwargs)}


def _resolve_class_spec(spec: dict, strict: bool = True) -> tuple[type, dict]:
    """
    {"class": "module:Qualname", "kwargs": {...}} back to (class, kwargs); the loaders write task setups in this form.
    Without `strict`, an unimportable class becomes a MissingClass placeholder (see activation.common.utils).
    """
    return resolve_class_name(spec["class"], strict=strict), dict(spec.get("kwargs") or {})


@dataclass
class TrajectoryStep:
    """
    Contains simple formats the model expects (e.g., raw dicts rather than our objects).
    """
    role: str                                                  # "assistant" | "tool" | "user" (the nudge)
    content: str                                               # assistant text (tool-call blocks removed) or the joined tool outputs
    tool_calls: list[dict] = field(default_factory=list)       # [{"id", "name", "arguments"}]
    tool_call_results: list[str] = field(default_factory=list) # truncated outputs, same order as tool_calls
    tool_results: list[dict] = field(default_factory=list)     # serialized ToolCallResults, same order: output, content, activation_content
                                                               # (recorded whether or not it was rendered), subagent_index. Empty on old records.
    messages: list[dict] = field(default_factory=list)         # the dialect messages this step appended, activation_context parts inline and in
                                                               # order: one assistant message, the tool messages, or the nudge
    ac_spans: list[dict] = field(default_factory=list)         # {start, length, rows (CPU) | row_path}, relative to this step's token_ids
    # Token segment (slice 2): the prompt the model read is AgentRunResult.prompt_token_ids + every step's token_ids in order.
    token_ids: list[int] = field(default_factory=list)         # assistant: the sampled tokens verbatim; tool/user: the template's wrapper up to the next generation prompt
    logprobs: list[float] = field(default_factory=list)        # assistant only: the engine's log-prob of each sampled token (pi_old); empty when not recorded
    think_end: int | None = None                               # assistant only: index one past the </think> token inside token_ids (the opening <think> is in the
                                                               # generation prompt before this step); None when the turn carried no reasoning block
    timing: dict | None = None                                 # assistant: turn_seconds, prompt/cached/output tokens, replica; tool: tool_seconds


@dataclass
class AgentRunResult:
    agent_config: AgentConfig
    answer: t.Any = None                                       # the submit_answer argument; None when never submitted
    num_turns: int = 0
    num_input_tokens: int = 0
    num_cached_input_tokens: int = 0
    num_output_tokens: int = 0
    num_ac_parts: int = 0                                      # parts encoded for this agent's prompts (nested children not counted)
    num_ac_rows: int = 0                                       # rows those parts occupy in the prompts
    duration: float = 0.0
    trajectory: list[dict] = field(default_factory=list)       # TrajectoryStep dicts of the current segment, in order
    score: float | None = None                           # None: unscored (no programmatic metric, UNSCORED datasets, never scored)
    score_feedback: str | None = None
    subagent_results: list["AgentRunResult"] = field(default_factory=list)
    compactions: list["AgentRunResult"] = field(default_factory=list)   # this agent's earlier segments, oldest first (finish_reason "compacted")
    injected_input: list[dict] = field(default_factory=list)   # serialized ToolCallResults placed ahead of the task in the first user message
                                                               # (a program's augment_context; the parent's context for a subagent)
    finish_reason: str = ""                                    # submitted | max_turns | max_tool_errors | max_duration | no_tool_call | context_exceeded
                                                               # | trajectory_cap | compaction_refused | compacted (a segment) | simulated | error
    seed: int = 0
    prompt_token_ids: list[int] = field(default_factory=list)  # the segment's first prompt (system + first messages + tools, templated once)
    prompt_messages: list[dict] = field(default_factory=list)  # the dialect messages of that prompt, parts inline (a compaction tree lives here)
    prompt_ac_spans: list[dict] = field(default_factory=list)  # {start, length, rows (CPU) | row_path}, relative to prompt_token_ids
    ac_model_name: str | None = None                           # the encoder that produced the rows, and its version at run time
    ac_model_version: int | None = None
    source: str = "policy"                                     # "policy" | "hinted:<model>" | "oracle:<model>": who produced this run
    lora_name: str | None = None                               # the adapter the engine actually applied (None: the base model)

    @property
    def num_tool_calls(self) -> int:
        # Assistant steps only: the tool step that follows repeats the same calls next to their results.
        return sum(len(step.get("tool_calls", [])) for step in self.trajectory if step.get("role") == "assistant")

    def tool_call_counts(self, include_segments: bool = True) -> dict[str, int]:
        """Calls by tool name over this agent's own turns (earlier compacted segments included); parallel_tool_call's members count under their own names too."""
        counts: dict[str, int] = {}
        segments = [*self.compactions, self] if include_segments else [self]
        for segment in segments:
            for step in segment.trajectory:
                if step.get("role") != "assistant":
                    continue
                for call in step.get("tool_calls", []):
                    counts[call["name"]] = counts.get(call["name"], 0) + 1
                    if call["name"] == "parallel_tool_call":
                        for member in (call.get("arguments") or {}).get("calls") or []:
                            if isinstance(member, dict) and member.get("name"):
                                counts[str(member["name"])] = counts.get(str(member["name"]), 0) + 1
        return counts

    def total_turns(self) -> int:
        """Turns of this agent over every compaction segment (num_turns counts the current segment only)."""
        return self.num_turns + sum(segment.num_turns for segment in self.compactions)

    def totals(self) -> dict[str, int]:
        """Turns and tokens of this agent and every subagent below it (segments included in the turns)."""
        total = {"turns": self.total_turns(), "input_tokens": self.num_input_tokens, "cached_input_tokens": self.num_cached_input_tokens,
                 "output_tokens": self.num_output_tokens, "subagents": len(self.subagent_results), "compactions": len(self.compactions)}
        for child in self.subagent_results:
            for key, value in child.totals().items():
                total[key] = total.get(key, 0) + value
        return total

    def serialize(self, *, base_dir: Path | None = None) -> dict:
        from .agent_utils import serialize_ac_rows
        data = dict(self.__dict__)
        data["agent_config"] = self.agent_config.serialize()
        return _jsonable(serialize_ac_rows(data, base_dir=base_dir))

    @staticmethod
    def deserialize(data: dict, harness: "HarnessRuntime | None" = None, agent_config: AgentConfig | None = None,
                    strict: bool = False, *, base_dir: Path | None = None) -> "AgentRunResult":
        """With agent_config given (the caller's live config), the stored config is not rebuilt. `strict`: see AgentConfig.deserialize."""
        from .agent_utils import deserialize_ac_rows
        data = deserialize_ac_rows(data, base_dir=base_dir)
        fields = {field_.name for field_ in AgentRunResult.__dataclass_fields__.values()}
        data = {key: value for key, value in data.items() if key in fields}          # cache rows carry extra keys
        config_data = data.pop("agent_config")
        config = agent_config if agent_config is not None else AgentConfig.deserialize(config_data, harness, strict=strict)
        children = [AgentRunResult.deserialize(child, harness, strict=strict) for child in data.pop("subagent_results", [])]
        segments = [AgentRunResult.deserialize(segment, harness, agent_config=config, strict=strict) for segment in data.pop("compactions", [])]
        return AgentRunResult(agent_config=config, subagent_results=children, compactions=segments, **data)


def _jsonable(value):
    """Values that json.dumps accepts; anything else becomes its repr."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        return repr(value)
