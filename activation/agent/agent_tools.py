"""
Defines tool calls.
All tool results are truncated to ~20000 chars as head[:10000] ... [truncated and written to
/tmp/agent_outputs/<id>.txt] ... tail[-10000:] (the env writes these files). A result's content is an
ordered list of text and activation_context parts: with an AC model a truncated output's full text
rides as a part before the visible text, a subagent's final segment before its answer line, and the
semantic-search passages beyond top_k after the visible ones. Every such part nests the parent's
current segment as its own part, so the encoder compresses the content in the parent's context.
"""
from __future__ import annotations

import re
import typing as t
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field

from activation.common.ac_parts import COMPACTION_INSTRUCTIONS, ac_part

if t.TYPE_CHECKING:
    from .agent_config import AgentConfig, AgentRunResult
    from .agent_env import AgentEnv
    from activation.harness import HarnessRuntime
    from .agent import Agent

TOOL_OUTPUT_LIMIT_CHARS = 20_000
AC_OUTPUT_LIMIT_CHARS = 4 * TOOL_OUTPUT_LIMIT_CHARS   # kept as activation context (and written to the env file): 4x what the model sees, head and tail beyond
TOOL_OUTPUT_DIR = "/tmp/agent_outputs"
SEARCH_EXTRA_MAX = 20                                 # passages beyond top_k that ride as activation context: min(this, 2 * top_k)
SUBAGENT_INTRO = "A parent agent spawned you. Its conversation so far is provided as activation context; your task follows.\n"


@dataclass
class ToolCallResult:
    output: str                                          # what the model sees as text (already truncated)
    is_error: bool = False
    is_final: bool = False                               # submit_answer sets it
    content: list[dict] | None = None                    # ordered content parts (text and activation_context) when the result carries
                                                         # activation context; None means [text(output)]. Exactly one text part equals output.
    compacted: bool = False                              # the compaction tool restarted the segment: no tool message follows this result


def tool_content(result: ToolCallResult) -> list[dict]:
    return result.content if result.content is not None else [{"type": "text", "text": result.output}]


def truncate_output(env: "AgentEnv | None", text: str, call_id: str, limit: int = TOOL_OUTPUT_LIMIT_CHARS,
                    agent: "Agent | None" = None) -> tuple[str, dict | None]:
    """
    Head + tail of a long output; the full text goes to a file in the env and, with an AC model on the
    agent, into an activation_context part (the full text as a tool message, nested with the parent's
    current segment). Returns (visible text, part or None).
    """
    if len(text) <= limit:
        return text, None
    if len(text) > AC_OUTPUT_LIMIT_CHARS:
        keep = AC_OUTPUT_LIMIT_CHARS // 2
        text = f"{text[:keep]}\n... [{len(text) - AC_OUTPUT_LIMIT_CHARS} chars dropped] ...\n{text[-keep:]}"
    path = f"{TOOL_OUTPUT_DIR}/{call_id}.txt"
    written = env.write_file(path, text) if env is not None else False
    half = limit // 2
    note = f"full output written to {path}" if written else "full output kept as activation context"
    truncated = f"{text[:half]}\n... [truncated {len(text) - limit} chars; {note}] ...\n{text[-half:]}"
    if agent is None or agent.ac_model is None:
        return truncated, None
    config = agent.agent_config
    part = ac_part([{"role": "user", "content": [agent.context_part(config.ac_subagent_ratio)]}, {"role": "tool", "content": text}],
                   config.ac_model_name, config.ac_tool_output_ratio, kind="tool_output")
    return truncated, part


def segment_messages_of(result: "AgentRunResult") -> list[dict]:
    """A run's final segment as dialect messages without the system message: prompt messages, then each step's messages."""
    messages = [message for message in result.prompt_messages if message.get("role") != "system"]
    for step in result.trajectory:
        messages.extend(step.get("messages") or [])
    return deepcopy(messages)


class AgentTool:
    name: str = ""
    description: str = ""
    parameters: dict = {"type": "object", "properties": {}}

    def __init__(self, harness: "HarnessRuntime", agent: "Agent", **kwargs):
        self.harness = harness
        self.agent = agent
        if kwargs:
            raise TypeError(f"{type(self).__name__} got unexpected kwargs {sorted(kwargs)}")

    def tool_definition(self) -> dict:
        """The function schema handed to the chat template's `tools`."""
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def execute(self, **arguments) -> ToolCallResult:
        raise NotImplementedError


class ShellTool(AgentTool):
    name = "shell"
    description = "Run a bash script in your sandbox (no network). Returns combined stdout and stderr."
    parameters = {
        "type": "object",
        "properties": {
            "script": {"type": "string", "description": "The bash script to run."},
            "timeout": {"type": "integer", "description": "Seconds before the script is killed (default 120)."},
        },
        "required": ["script"],
    }

    def execute(self, script: str, timeout: int = 120) -> ToolCallResult:
        output, ok = self.agent.agent_env.run_shell(script, timeout=int(timeout))
        return ToolCallResult(output=output or "(no output)", is_error=not ok)


class PythonTool(AgentTool):
    name = "python"
    description = "Run a Python 3 program in your sandbox (no network, no state kept between calls). Print what you need to see."
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python source to run."},
            "timeout": {"type": "integer", "description": "Seconds before the program is killed (default 120)."},
        },
        "required": ["code"],
    }

    def execute(self, code: str, timeout: int = 120) -> ToolCallResult:
        output, ok = self.agent.agent_env.run_python_code(code, timeout=int(timeout))
        return ToolCallResult(output=output or "(no output)", is_error=not ok)


class SubmitAnswerTool(AgentTool):
    name = "submit_answer"
    description = "Submit your final answer and end the task. Call it exactly once, when you are done."
    parameters = {
        "type": "object",
        "properties": {"answer": {"type": "string", "description": "The final answer, as concise as possible."}},
        "required": ["answer"],
    }

    def execute(self, answer: t.Any = None) -> ToolCallResult:
        self.agent.run_results.answer = answer
        return ToolCallResult(output=f"Answer submitted: {answer}", is_final=True)


class ParallelCallTool(AgentTool):
    name = "parallel_tool_call"
    description = "Run several tool calls at once. Each call names a tool and its arguments; the results come back in the same order."
    parameters = {
        "type": "object",
        "properties": {
            "calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
                    "required": ["name", "arguments"],
                },
            },
        },
        "required": ["calls"],
    }

    def execute(self, calls: list[dict]) -> ToolCallResult:
        if not isinstance(calls, list) or not calls:
            return ToolCallResult(output="parallel_tool_call needs a non-empty list of {name, arguments}.", is_error=True)
        nested = [{"id": uuid.uuid4().hex[:8], "name": str(call.get("name")), "arguments": call.get("arguments") or {}} for call in calls]
        with ThreadPoolExecutor(max_workers=min(8, len(nested))) as pool:
            results = list(pool.map(self.agent.execute_tool_call, nested))
        outputs = [f"[{call['name']}] {result.output}" for call, result in zip(nested, results)]
        content: list[dict] = []
        for call, result, labeled in zip(nested, results, outputs):
            for item in tool_content(result):                                            # each call's parts in its own order
                content.append({"type": "text", "text": labeled + "\n\n"} if item.get("type") == "text" else item)
        if content and content[-1].get("type") == "text":
            content[-1] = {"type": "text", "text": content[-1]["text"].rstrip("\n")}
        has_parts = any(result.content is not None for result in results)
        return ToolCallResult(
            output="\n\n".join(outputs), is_error=any(result.is_error for result in results),
            is_final=any(result.is_final for result in results), compacted=any(result.compacted for result in results),
            content=content if has_parts else None,
        )


class SemanticSearchTool(AgentTool):
    """Search over a registered dataset's chunks. BM25 for now; the dense path returns with the retrieval rethink."""
    name = "semantic_search"
    description = "Search the task's corpus for passages relevant to a query. Returns the best matching passages with their ids."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "top_k": {"type": "integer", "description": "How many passages to return."},
        },
        "required": ["query"],
    }

    def __init__(self, harness, agent, dataset_id: str | None = None, top_k: int = 5, max_chars: int = 2000):
        super().__init__(harness, agent)
        task = agent.agent_config.dataset_task
        self.dataset_id = dataset_id or (task.dataset_id if task is not None else None)
        self.top_k = top_k
        self.max_chars = max_chars

    def execute(self, query: str, top_k: int | None = None) -> ToolCallResult:
        if self.dataset_id is None or self.dataset_id not in self.harness.dataset_manager.loaded_datasets:
            return ToolCallResult(output=f"No searchable corpus is registered for this task ({self.dataset_id}).", is_error=True)
        index = self.harness.dataset_manager._get_or_create_index(self.dataset_id)
        index.build_bm25_index()
        k = int(top_k or self.top_k)
        config = self.agent.agent_config
        extra = min(SEARCH_EXTRA_MAX, 2 * k) if self.agent.ac_model is not None else 0    # the model sees top_k; the next `extra` ride as rows
        chunks = index.bm25_query_many_frozen([query], top_k=k + extra)[0]
        if not chunks:
            return ToolCallResult(output="No passage matched the query.")

        def blocks(subset) -> str:
            return "\n\n".join(f"[{chunk.chunk_id}]\n{index.get_chunk_section(chunk)[:self.max_chars]}" for chunk in subset)

        output = blocks(chunks[:k])
        content = None
        if chunks[k:]:
            part = ac_part([{"role": "user", "content": [self.agent.context_part(config.ac_subagent_ratio)]},
                            {"role": "tool", "content": blocks(chunks[k:])}], config.ac_model_name, config.ac_search_ratio, kind="search")
            content = [{"type": "text", "text": output}, part]                                # text first, the extra passages after
        return ToolCallResult(output=output, content=content)


class SubagentTool(AgentTool):
    """
    Simple in/out without a persistent handle. Runs in the same env as the caller. With AC
    communication the child's first user message carries the parent's current segment as a part
    (intro text, part, task) and the child's final segment comes back as a part before the answer
    line; the child's own compaction tree rides inside that segment. The result is registered in the
    caller's subagent results. Some subagents are general-purpose, others specific: all are registered
    as tools under their own name, with `base_config` and `extra_description` as constructor kwargs.
    In step mode (simulation) the child is created with a dummy submitted answer and does not run.
    """
    parameters = {
        "type": "object",
        "properties": {"task": {"type": "string", "description": "A self-contained description of what the subagent should do and return."}},
        "required": ["task"],
    }
    SHARED_FIELDS = ("ac_model_name", "enable_ac_communication", "ac_compaction_ratio", "ac_subagent_ratio", "ac_tool_output_ratio", "ac_search_ratio")

    def __init__(self, harness, agent, base_config: "AgentConfig", extra_description: str = ""):
        super().__init__(harness, agent)
        self.base_config = base_config
        self.description = ("Delegate a self-contained task to a subagent and returns its answer. "
                            + extra_description).strip()

    def execute(self, task: str) -> ToolCallResult:
        from .agent import Agent
        parent = self.agent
        parent_config = parent.agent_config
        subagent_config = deepcopy(self.base_config)
        if subagent_config.agent_name == "main":
            subagent_config.agent_name = "general_subagent"
        subagent_config.user_prompt = task
        share_ac = parent_config.enable_ac_communication and parent.ac_model is not None
        if share_ac:
            for name in self.SHARED_FIELDS:
                setattr(subagent_config, name, getattr(parent_config, name))
            subagent_config.messages_input = [{"role": "user", "content": [
                {"type": "text", "text": SUBAGENT_INTRO},
                ac_part(deepcopy(parent.segment_messages()), parent_config.ac_model_name, parent_config.ac_subagent_ratio, kind="subagent_prompt"),   # the parent's segment up to and including this delegating turn
            ]}]                                                                # the child appends the task as the trailing text part
        subagent = Agent(
            harness=self.harness,
            agent_config=subagent_config,
            agent_env=parent._env if parent.step_mode else parent.agent_env,   # step mode never starts a sandbox
            parent_agent=parent,
            reporter=parent.reporter,
            seed=parent.seed,
        )
        subagent.step_mode = parent.step_mode
        result = subagent.simulated_run() if parent.step_mode else subagent.run()
        with parent.lock:
            parent.run_results.subagent_results.append(result)
        output = (f"Subagent finished ({result.finish_reason}, {result.num_turns} turns). "
                  f"Answer: {result.answer if result.answer is not None else '(none)'}")
        content = None
        if share_ac:
            final_segment = segment_messages_of(result)                         # its first user messages (tree inside, if any) + every step
            content = [ac_part(final_segment, parent_config.ac_model_name, parent_config.ac_subagent_ratio, kind="subagent_return"),
                       {"type": "text", "text": output}]
        return ToolCallResult(output=output, is_error=result.finish_reason == "error", content=content)


class CompactionTool(AgentTool):
    """
    Compacts the conversation into a new segment. With an AC model the finished segment becomes the
    recursive tree part of the new prompt and the summary follows it as text; without one only the
    summary carries over. The finished segment is recorded under the run's `compactions`
    (finish_reason "compacted"). The loop demands a lone `compact` call once the segment's soft
    threshold is reached (Agent.COMPACTION_DEMAND) and refuses every other call until then.
    """
    name = "compact"
    description = ("Compact the conversation so far: write a summary of what was done, what was learned and what remains, "
                   "then continue from it. Call it alone, without other tools, when asked to compact.")
    parameters = {
        "type": "object",
        "properties": {"summary": {"type": "string", "description": "A complete summary of the work so far for your future self."}},
        "required": ["summary"],
    }

    def execute(self, summary: str = "") -> ToolCallResult:
        self.agent.compact(str(summary or ""))
        return ToolCallResult(output="Context compacted.", compacted=True)


DEFAULT_TOOLS: dict[str, tuple[type[AgentTool], dict]] = {
    ShellTool.name: (ShellTool, {}),
    PythonTool.name: (PythonTool, {}),
    ParallelCallTool.name: (ParallelCallTool, {}),
    SubmitAnswerTool.name: (SubmitAnswerTool, {}),
    CompactionTool.name: (CompactionTool, {}),
}
