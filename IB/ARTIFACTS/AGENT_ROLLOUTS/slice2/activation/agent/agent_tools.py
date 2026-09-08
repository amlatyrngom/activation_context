"""
Defines tool calls.
All tool results are truncated to ~20000 chars as head[:10000] ... [truncated and written to
/tmp/agent_outputs/<id>.txt] ... tail[-10000:] (the env writes these files). When truncated, the
full output travels as an activation-context output (`ac_outputs["full_output"]`).
"""
from __future__ import annotations

import re
import typing as t
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field

if t.TYPE_CHECKING:
    from .agent_config import AgentConfig
    from .agent_env import AgentEnv
    from activation.harness import HarnessRuntime
    from .agent import Agent

TOOL_OUTPUT_LIMIT_CHARS = 20_000
AC_OUTPUT_LIMIT_CHARS = 4 * TOOL_OUTPUT_LIMIT_CHARS   # kept as activation context (and written to the env file): 4x what the model sees, head and tail beyond
TOOL_OUTPUT_DIR = "/tmp/agent_outputs"
@dataclass
class ToolCallResult:
    output: str                                          # what the model sees (already truncated)
    is_error: bool = False
    ac_outputs: dict[str, t.Any] = field(default_factory=dict)  # activation context outputs (ac name -> ac input)
    is_final: bool = False                               # submit_answer sets it


def truncate_output(env: "AgentEnv | None", text: str, call_id: str, limit: int = TOOL_OUTPUT_LIMIT_CHARS) -> tuple[str, dict]:
    """Head + tail of a long output; the full text goes to a file in the env and out as activation context."""
    if len(text) <= limit:
        return text, {}
    if len(text) > AC_OUTPUT_LIMIT_CHARS:
        keep = AC_OUTPUT_LIMIT_CHARS // 2
        text = f"{text[:keep]}\n... [{len(text) - AC_OUTPUT_LIMIT_CHARS} chars dropped] ...\n{text[-keep:]}"
    path = f"{TOOL_OUTPUT_DIR}/{call_id}.txt"
    written = env.write_file(path, text) if env is not None else False
    half = limit // 2
    note = f"full output written to {path}" if written else "full output kept as activation context"
    truncated = f"{text[:half]}\n... [truncated {len(text) - limit} chars; {note}] ...\n{text[-half:]}"
    return truncated, {"full_output": text}


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
        ac_outputs = {call["id"]: result.ac_outputs for call, result in zip(nested, results) if result.ac_outputs}
        return ToolCallResult(
            output="\n\n".join(outputs), is_error=any(result.is_error for result in results),
            ac_outputs=ac_outputs, is_final=any(result.is_final for result in results),
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
        chunks = index.bm25_query_many_frozen([query], top_k=int(top_k or self.top_k))[0]
        if not chunks:
            return ToolCallResult(output="No passage matched the query.")
        blocks = [f"[{chunk.chunk_id}]\n{index.get_chunk_section(chunk)[:self.max_chars]}" for chunk in chunks]
        return ToolCallResult(output="\n\n".join(blocks), ac_outputs={"chunk_ids": [chunk.chunk_id for chunk in chunks]})


class SubagentTool(AgentTool):
    """
    Simple in/out without a persistent handle. Runs in the same env as the caller. Gets the caller's
    current trajectory as activation context and returns its own trajectory as activation context
    (unless AC communication is disabled); its result is registered in the caller's subagent results.
    Some subagents are general-purpose, others specific: all are registered as tools under their own
    name, with `base_config` and `extra_description` as constructor kwargs.
    """
    parameters = {
        "type": "object",
        "properties": {"task": {"type": "string", "description": "A self-contained description of what the subagent should do and return."}},
        "required": ["task"],
    }

    def __init__(self, harness, agent, base_config: "AgentConfig", extra_description: str = ""):
        super().__init__(harness, agent)
        self.base_config = base_config
        self.description = ("Delegate a self-contained task to a subagent and returns its answer. "
                            + extra_description).strip()

    def execute(self, task: str) -> ToolCallResult:
        from .agent import Agent
        parent = self.agent
        subagent_config = deepcopy(self.base_config)
        if subagent_config.agent_name == "main":
            subagent_config.agent_name = "general_subagent"
        subagent_config.user_prompt = task
        share_ac = parent.agent_config.enable_ac_communication
        
        if share_ac:
            subagent_config.ac_inputs = dict(subagent_config.ac_inputs) | {"caller_trajectory": deepcopy(parent.run_results.trajectory)}
        subagent = Agent(
            harness=self.harness,
            agent_config=subagent_config,
            agent_env=parent.agent_env,
            parent_agent=parent,
            reporter=parent.reporter,
            seed=parent.seed,
        )
        result = subagent.run()
        with parent.lock:
            parent.run_results.subagent_results.append(result)
        output = (f"Subagent finished ({result.finish_reason}, {result.num_turns} turns). "
                  f"Answer: {result.answer if result.answer is not None else '(none)'}")
        ac_outputs = {"trajectory": result.trajectory} if share_ac else {}
        return ToolCallResult(output=output, is_error=result.finish_reason == "error", ac_outputs=ac_outputs)


DEFAULT_TOOLS: dict[str, tuple[type[AgentTool], dict]] = {
    ShellTool.name: (ShellTool, {}),
    PythonTool.name: (PythonTool, {}),
    ParallelCallTool.name: (ParallelCallTool, {}),
    SubmitAnswerTool.name: (SubmitAnswerTool, {}),
}
