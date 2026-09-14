"""
Defines tool calls.
All tool results are truncated to ~20000 chars as head[:10000] ... [truncated and written to
/tmp/agent_outputs/<id>.txt] ... tail[-10000:] (the env writes these files). A result has `content`
(the text the model reads) and `activation_content`: parts every tool produces on every call, whether
or not the reader has an AC model (a truncated output's full text, the semantic-search passages beyond
top_k, a subagent's final segment). The agent renders them ahead of the content when it has an AC
model and records them on the tool step either way (`TrajectoryStep.tool_results`), so a base run can
be converted to an AC-bearing one for training. Parts that compress a segment carry its system message
first and its tool definitions; a tool-output or search part nests the parent's segment the same way.
"""
from __future__ import annotations

import re
import typing as t
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field, replace

from activation.common.ac_parts import COMPACTION_INSTRUCTIONS, activation_part

if t.TYPE_CHECKING:
    from .agent_config import AgentConfig, AgentRunResult
    from .agent_env import AgentEnv
    from activation.harness import HarnessRuntime
    from .agent import Agent

TOOL_OUTPUT_LIMIT_CHARS = 20_000
AC_OUTPUT_LIMIT_CHARS = 4 * TOOL_OUTPUT_LIMIT_CHARS   # kept as activation context (and written to the env file): 4x what the model sees, head and tail beyond
TOOL_OUTPUT_DIR = "/tmp/agent_outputs"
SEARCH_EXTRA_MAX = 20                                 # passages beyond top_k that ride as activation context: min(this, 2 * top_k)
SUBAGENT_INTRO = "A parent agent spawned you. Its conversation so far may precede this note as activation context. Your task follows."
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
    content: list[dict] | None = None                    # text parts; None means [text(output)]. Exactly one text part equals output.
    activation_content: list[dict] = field(default_factory=list)   # parts {type, kind, messages, tools} (activation_part): what the reader
                                                         # may see compressed. Produced always; the agent renders them when it has an AC
                                                         # model and records them on the step either way.
    tool: str = ""                                       # the tool that produced it (the agent sets it; "program" / "parent" for injections)
    subagent_index: int | None = None                    # record only: the child's index in subagent_results; never rendered
    compacted: bool = False                              # the compaction tool restarted the segment: no tool message follows this result

    def serialize(self) -> dict:
        return {"tool": self.tool, "output": self.output, "is_error": self.is_error, "is_final": self.is_final,
                "content": deepcopy(self.content), "activation_content": deepcopy(self.activation_content),
                "subagent_index": self.subagent_index, "compacted": self.compacted}

    @classmethod
    def deserialize(cls, data: dict) -> "ToolCallResult":
        return cls(output=data.get("output", ""), is_error=bool(data.get("is_error", False)), is_final=bool(data.get("is_final", False)),
                   content=deepcopy(data.get("content")), activation_content=deepcopy(data.get("activation_content") or []),
                   tool=data.get("tool", ""), subagent_index=data.get("subagent_index"), compacted=bool(data.get("compacted", False)))


def tool_content(result: ToolCallResult) -> list[dict]:
    return result.content if result.content is not None else [{"type": "text", "text": result.output}]


def injected_block_text(result: "ToolCallResult | dict") -> str:
    """The text part of an injected result in the first user message: the tool's name as a label, then its output."""
    tool = result.get("tool") if isinstance(result, dict) else result.tool
    output = result.get("output") if isinstance(result, dict) else result.output
    return f"[{tool or 'program'}]\n{output}\n\n"


def segment_messages_with_parts(run: "AgentRunResult", render: "t.Callable[[list[dict]], list[dict]]") -> list[dict]:
    """
    The run's current segment as messages with its recorded activation content put back through `render`: the first user
    message's injected blocks get their parts from `injected_input`, each tool step's message gets its results' parts from
    `tool_results` (ahead of the text, the placement the agent uses). Parts the record rendered at run time are replaced
    by the re-rendered ones; steps without `tool_results` (old records) stay as recorded. `render` is the identity for a
    raw segment (a part's messages) or the reader's resolver for a prompt. Token ids are not produced here.
    """
    from activation.common.ac_parts import is_ac_part

    def text_items(content) -> list[dict]:
        items = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        return [item for item in items if not is_ac_part(item)]

    messages = deepcopy(run.prompt_messages)
    injected = list(run.injected_input)
    if injected:
        for message in messages:
            if message.get("role") != "user":
                continue
            out, j = [], 0
            for item in text_items(message.get("content")):
                if j < len(injected) and item.get("type") == "text" and item.get("text") == injected_block_text(injected[j]):
                    out.extend(render(list(injected[j].get("activation_content") or [])))
                    j += 1
                out.append(item)
            if j:
                message["content"] = out
                break
    for step in run.trajectory:
        results = step.get("tool_results") or []
        step_messages = deepcopy(step.get("messages") or [])
        if step.get("role") == "tool" and results and step_messages:
            if step_messages[0].get("role") == "tool":                                            # structured: one message per result
                for message, result in zip(step_messages, results):
                    message["content"] = render(list(result.get("activation_content") or [])) + text_items(message.get("content"))
            else:                                                                                  # inline: one user message of <tool_response> blocks
                out, index = [], 0
                for item in text_items(step_messages[0].get("content")):
                    out.append(item)
                    if item.get("type") == "text" and item.get("text", "").lstrip("\n").startswith("<tool_response>") and index < len(results):
                        out.extend(render(list(results[index].get("activation_content") or [])))
                        index += 1
                step_messages[0]["content"] = out
        messages.extend(step_messages)
    return messages


def injected_content(injected: list[dict], render: "t.Callable[[list[dict]], list[dict]]") -> list[dict]:
    """
    The content parts that injected results (`AgentRunResult.injected_input`, serialized) contribute ahead of the task:
    per result, its rendered activation parts (`render` returns [] for a reader without an AC model) then its block text.
    Shared by Agent.first_user_messages and the training-side conversion (activation_messages_of).
    """
    content: list[dict] = []
    for result in injected:
        content.extend(render(list(result.get("activation_content") or [])))
        content.append({"type": "text", "text": injected_block_text(result)})
    return content


def truncate_output(env: "AgentEnv | None", text: str, call_id: str, limit: int = TOOL_OUTPUT_LIMIT_CHARS,
                    agent: "Agent | None" = None) -> tuple[str, dict | None]:
    """
    Head + tail of a long output; the full text goes to a file in the env and, with an agent given, into an
    activation part (kind tool_output: the full text as a tool message, nested with the parent's current
    segment). Returns (visible text, part or None). The part is produced whether or not the agent has an
    AC model: the agent renders it when it has one and records it either way.
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
    if agent is None:
        return truncated, None
    return truncated, activation_part("tool_output", agent.nested_in_context({"role": "tool", "content": text}))


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
        return ToolCallResult(
            output="\n\n".join(outputs), is_error=any(result.is_error for result in results),
            is_final=any(result.is_final for result in results),
            activation_content=[part for result in results for part in result.activation_content],   # members' parts in call order
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
        extra = min(SEARCH_EXTRA_MAX, 2 * k)                  # the model sees top_k; the next `extra` ride as a part (recorded either way)
        chunks = index.bm25_query_many_frozen([query], top_k=k + extra)[0]
        if not chunks:
            return ToolCallResult(output="No passage matched the query.")

        def blocks(subset) -> str:
            return "\n\n".join(f"[{chunk.chunk_id}]\n{index.get_chunk_section(chunk)[:self.max_chars]}" for chunk in subset)

        output = blocks(chunks[:k])
        activation = []
        if chunks[k:]:
            activation = [activation_part("search", self.agent.nested_in_context({"role": "tool", "content": blocks(chunks[k:])}))]
        return ToolCallResult(output=output, activation_content=activation)


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
    Simple in/out without a persistent handle. Runs in the same env as the caller. The parent's current
    segment goes to the child as an injected result (`augment_context`: the intro note, the segment as a
    subagent_prompt part) and the child's final segment comes back as the result's activation content
    (a subagent_return part) before the answer line; either side renders the part when it has an AC
    model and records it regardless. The child's own compaction tree rides inside its segment. The
    result is registered in the caller's subagent results and the result carries the child's index.
    Some subagents are general-purpose, others specific: all are registered as tools under their own
    name, with `base_config` and `extra_description` as constructor kwargs. The child's system prompt
    is its base config's (the general subagent's base config is the caller's config with the subagent
    note appended once). In step mode (simulation) the child is created with a dummy submitted answer
    and does not run.
    """
    parameters = {
        "type": "object",
        "properties": {"task": {"type": "string", "description": "A self-contained description of what the subagent should do and return."}},
        "required": ["task"],
    }
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
        subagent = Agent(
            harness=self.harness,
            agent_config=subagent_config,
            agent_env=parent._env if parent.step_mode else parent.agent_env,   # step mode never starts a sandbox
            parent_agent=parent,
            reporter=parent.reporter,
            seed=parent.seed,
        )
        subagent.step_mode = parent.step_mode
        subagent.augment_context(ToolCallResult(tool="parent", output=SUBAGENT_INTRO,
                                                activation_content=[parent.to_activation_context("subagent_prompt")]))
        result = subagent.simulated_run() if parent.step_mode else subagent.run()
        with parent.lock:
            parent.run_results.subagent_results.append(result)
            index = len(parent.run_results.subagent_results) - 1
        output = (f"Subagent finished ({result.finish_reason}, {result.num_turns} turns). "
                  f"Answer: {result.answer if result.answer is not None else '(none)'}")
        return ToolCallResult(output=output, is_error=result.finish_reason == "error", subagent_index=index,
                              activation_content=[subagent.to_activation_context("subagent_return")])


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
