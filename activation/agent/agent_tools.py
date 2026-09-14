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
from dataclasses import dataclass, field, replace

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
SUBAGENT_TASK_FORMAT = (
    "A self-contained brief for the subagent, which sees none of your context. Write it as:\n"
    "TASK:\n- what to do, with every fact, path and constraint it needs\n"
    "CONTEXT:\n- the surrounding situation and current state it should know; point to shared files in the sandbox for details\n"
    "TO PRODUCE:\n- the answer format, or the files / report to write and where\n"
    "DON'T DO:\n- scope to avoid, files not to touch, assumptions not to make"
)
SUBAGENT_GUIDE = (
    " The subagent works in your sandbox with the same tools (except delegation) and returns its final answer. "
    "Patterns that work: parallel research (several subagents investigate different questions at once through "
    "parallel_tool_call, you synthesize); parallel solve and synthesize (independent attempts, you compare and pick); "
    "sequential sub-solves (each brief includes the previous result); plan, solve, validate (one subagent solves, "
    "another checks). Delegate self-contained work; keep integration and the final answer yourself."
)
SUBAGENT_SYSTEM_NOTE = ("\n\nYou are a focused subagent spawned by another agent. Do exactly the brief you were given, using the "
                        "tools, and return only what it asks for through submit_answer. You cannot delegate further.")


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

    aliases: dict[str, str] = {}                              # argument names models use for a parameter -> the parameter

    def tool_definition(self) -> dict:
        """The function schema handed to the chat template's `tools`."""
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def normalize_arguments(self, arguments: dict) -> dict:
        """
        Argument names the schema does not know are mapped onto the parameters: by the tool's alias table, then a
        lone unknown key onto the lone missing required parameter (`{"cmd": ...}` for a tool with one required
        argument). Known names always win; the model's original call stays in the trajectory unchanged.
        """
        properties = self.parameters.get("properties") or {}
        normalized = dict(arguments)
        for key in list(normalized):
            if key in properties:
                continue
            target = self.aliases.get(key)
            if target is not None and target not in normalized:
                normalized[target] = normalized.pop(key)
        unknown = [key for key in normalized if key not in properties]
        missing = [key for key in self.parameters.get("required") or [] if key not in normalized]
        if len(unknown) == 1 and len(missing) == 1:
            normalized[missing[0]] = normalized.pop(unknown[0])
        return normalized

    def execute(self, **arguments) -> ToolCallResult:
        raise NotImplementedError


class ShellTool(AgentTool):
    aliases = {'command': 'script', 'cmd': 'script', 'bash': 'script', 'sh': 'script', 'shell': 'script', 'code': 'script'}
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
    aliases = {'script': 'code', 'source': 'code', 'program': 'code', 'python': 'code'}
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
    aliases = {'final_answer': 'answer', 'response': 'answer', 'result': 'answer', 'value': 'answer', 'text': 'answer'}
    name = "submit_answer"
    description = "Submit your final answer and end the task. Call it exactly once, when you are done."
    parameters = {
        "type": "object",
        "properties": {"answer": {"type": "string", "description": "The final answer, as concise as possible."}},
        "required": ["answer"],
    }

    def execute(self, answer: t.Any = None) -> ToolCallResult:
        if answer is None or (isinstance(answer, str) and not answer.strip()):
            return ToolCallResult(output="Nothing submitted: submit_answer needs the answer as its `answer` argument.", is_error=True)
        self.agent.run_results.answer = answer
        return ToolCallResult(output=f"Answer submitted: {answer}", is_final=True)


class ParallelCallTool(AgentTool):
    aliases = {'tool_calls': 'calls', 'requests': 'calls'}
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

    @staticmethod
    def _call_arguments(call: dict) -> dict:
        """
        The call's arguments as the model meant them: the `arguments` object (also `args` / `parameters`), plus any
        other key written directly on the call (models flatten `{"name": "subagent", "task": ...}` or send both, with
        the object's value null); a null never shadows a value.
        """
        if not isinstance(call, dict):
            return {}
        arguments = {}
        for key in ("arguments", "args", "parameters", "input"):
            if isinstance(call.get(key), dict):
                arguments.update(call[key])
        for key, value in call.items():
            if key in ("name", "tool", "id", "arguments", "args", "parameters", "input"):
                continue
            if arguments.get(key) is None:
                arguments[key] = value
        return {key: value for key, value in arguments.items() if value is not None}

    def execute(self, calls: list[dict]) -> ToolCallResult:
        if not isinstance(calls, list) or not calls:
            return ToolCallResult(output="parallel_tool_call needs a non-empty list of {name, arguments}.", is_error=True)
        nested = [{"id": uuid.uuid4().hex[:8], "name": str(call.get("name") or call.get("tool")), "arguments": self._call_arguments(call)} for call in calls]
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
    aliases = {'q': 'query', 'question': 'query', 'text': 'query', 'search': 'query', 'k': 'top_k', 'n': 'top_k', 'limit': 'top_k'}
    """Search over a registered dataset's chunks. BM25 for now; the dense path returns with the retrieval rethink."""
    name = "semantic_search"
    description = ("Search the task's corpus for passages relevant to a query: the fastest way to find relevant text in it. "
                   "Returns the best matching passages with their ids.")
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


def subagent_config_of(parent: "Agent", task: str = "", agent_name: str = "general_subagent") -> "AgentConfig":
    """
    The caller's config as a focused subagent: same model, tools, env image, budgets and dataset task; the subagent note
    appended to the system prompt; every delegation tool removed and no agentic program (no recursion); no env setups
    (the env is the parent's, already prepared). Shared by the subagent tool and Agent.run_subagent.
    """
    parent_config = parent.agent_config
    delegation = {name: None for name, tool in parent.tools.items() if isinstance(tool, SubagentTool)}
    return replace(parent_config, agent_name=agent_name, user_prompt=task, messages_input=[], agentic_program=None,
                   system_prompt=(parent_config.system_prompt or "") + SUBAGENT_SYSTEM_NOTE,
                   tools={**parent_config.tools, **delegation}, env_setups={})


class SubagentTool(AgentTool):
    aliases = {'prompt': 'task', 'brief': 'task', 'instructions': 'task', 'description': 'task', 'query': 'task'}
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

    def __init__(self, harness, agent, base_config: "AgentConfig | None" = None, extra_description: str = "",
                 include_full_subagent_guide: bool = True):
        """
        base_config None: the general subagent, which inherits the caller's configuration (`_inherit_general_config`).
        With the guide, the description explains the multi-agent patterns and the task parameter asks for the
        TASK / CONTEXT / TO PRODUCE / DON'T DO brief.
        """
        super().__init__(harness, agent)
        self.base_config = base_config
        self.description = ("Delegate a self-contained task to a subagent and return its answer."
                            + (SUBAGENT_GUIDE if include_full_subagent_guide else "") + " " + extra_description).strip()
        task_description = SUBAGENT_TASK_FORMAT if include_full_subagent_guide else self.parameters["properties"]["task"]["description"]
        self.parameters = {**self.parameters, "properties": {"task": {"type": "string", "description": task_description}}}

    def _inherit_general_config(self) -> "AgentConfig":
        """The caller's config as a focused subagent (`subagent_config_of`); the task and AC fields are filled by execute."""
        return subagent_config_of(self.agent)

    def execute(self, task: str | None = None) -> ToolCallResult:
        from .agent import Agent
        if not isinstance(task, str) or not task.strip():
            return ToolCallResult(output="Error: the subagent needs a brief in `task` (a string that states the task, the context, what to produce and what not to do); nothing was delegated.", is_error=True)
        parent = self.agent
        parent_config = parent.agent_config
        subagent_config = deepcopy(self.base_config) if self.base_config is not None else self._inherit_general_config()
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
    aliases = {'text': 'summary', 'notes': 'summary', 'content': 'summary', 'state': 'summary'}
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
    "subagent": (SubagentTool, {}),       # the general subagent: inherits the caller; removed from the subagents themselves
}
