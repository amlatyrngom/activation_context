"""
Reformatting public agent trajectories into our dialect, so the AC model learns on the shapes our
agents produce ("the compression bias in our favor").

Every loader normalizes its source's quirks (structured or inline tool calls, JSON tool results,
reasoning in a separate field or in <think> blocks, tool definitions as objects, JSON strings or a
<tools> block in the system prompt) into one shared intermediate:

    {"role": "system" | "user" | "assistant" | "tool", "content": str,
     "reasoning": str | None,                       # assistant only
     "tool_calls": [{"name": str, "arguments": dict}]}   # assistant only

and a list of tool definitions in the OpenAI function shape. `reformat_trajectory` then turns that
into our messages: the source system prompt is dropped (a short domain prompt rides in
`trajectory_kwargs["system_prompt"]`), tool names and argument keys are mapped onto our tools
(`shell(script)`, `python(code)`; unknown tools keep their names and definitions), reasoning stays
as plain text ahead of the reply (`reasoning="keep"`, since the math `cot` rows are nothing but
reasoning) or is dropped, calls become structured `tool_calls` (the shape `MessageRendering.assistant_message`
produces), and every tool result is one `tool` message.
"""
from __future__ import annotations

import json
import re
import typing as t

from ...agent.agent_utils import parse_tool_calls

TOOL_MAP: dict[str, tuple[str, dict[str, str]]] = {
    "bash": ("shell", {"command": "script"}),
    "stateful_python_code_exec": ("python", {"code": "code"}),
    "python": ("python", {}),
    "shell": ("shell", {}),
}
"""Source tool name -> (our tool name, {source argument: our argument}). Unknown tools pass through unchanged."""

OUR_TOOL_DESCRIPTIONS = {
    "shell": ("Run a bash script in the sandbox and return its output.", {"script": {"type": "string", "description": "The bash script to run."}}),
    "python": ("Run Python source in the sandbox and return its output.", {"code": {"type": "string", "description": "The Python source to run."}}),
}

THINK_BLOCK = re.compile(r"<think>\s*(.*?)\s*</think>\s*", re.S)
TOOLS_BLOCK = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.S)
TOOL_RESPONSE_BLOCK = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)


# --------------------------------------------------------------------------------------- source helpers
def split_think(text: str) -> tuple[str | None, str]:
    """(reasoning, the rest) of a text with <think>...</think> blocks (joined when several)."""
    blocks = [match.group(1) for match in THINK_BLOCK.finditer(text or "")]
    rest = THINK_BLOCK.sub("", text or "").strip()
    return ("\n\n".join(blocks) if blocks else None), rest


def parse_inline_calls(text: str) -> tuple[str, list[dict]]:
    """Hermes / XML <tool_call> blocks in an assistant text -> (text without them, [{"name", "arguments"}])."""
    content, calls = parse_tool_calls(text or "")
    return content, [{"name": call["name"], "arguments": call["arguments"] if isinstance(call["arguments"], dict) else {"raw": call["arguments"]}}
                     for call in calls]


def structured_calls(tool_calls: list | None) -> list[dict]:
    """Structured tool calls (OpenAI shape, or {"name", "arguments"}) -> [{"name", "arguments": dict}]."""
    out = []
    for call in tool_calls or []:
        function = call.get("function", call) if isinstance(call, dict) else {}
        name = str(function.get("name", ""))
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        if name:
            out.append({"name": name, "arguments": arguments})
    return out


def flatten_tool_output(content) -> str:
    """A tool result as text: JSON {"returncode", "output"} (Open-SWE) becomes the output plus an exit line when non-zero."""
    if isinstance(content, (dict, list)):
        data = content
    else:
        text = str(content or "")
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                return text
        else:
            return text
    if isinstance(data, dict) and "output" in data:
        output = str(data.get("output") or "")
        returncode = data.get("returncode")
        if returncode not in (None, 0, "0"):
            output = f"{output.rstrip()}\n[exit code {returncode}]"
        return output
    return json.dumps(data, ensure_ascii=False)


def tool_responses_from_text(text: str) -> list[str]:
    """The <tool_response> blocks of a user message that carries tool results inline (Qwen-style templates)."""
    return [match.group(1) for match in TOOL_RESPONSE_BLOCK.finditer(text or "")]


def tools_from_system_prompt(text: str) -> list[dict]:
    """Tool definitions listed inside a <tools> block of a system prompt (one JSON object per line, or a JSON array)."""
    # Prompts mention a literal "<tools></tools>" before the real block: take the first non-empty one.
    body = next((match.group(1).strip() for match in TOOLS_BLOCK.finditer(text or "") if match.group(1).strip()), "")
    if not body:
        return []
    try:
        parsed = json.loads(body)
        return list(parsed) if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        pass
    tools = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            tools.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return tools


def parse_tool_definitions(tools) -> list[dict]:
    """Tool definitions as a list, a JSON string of a list, or a list of JSON strings -> list of dicts."""
    if tools is None:
        return []
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return []
    out = []
    for tool in tools if isinstance(tools, list) else [tools]:
        if isinstance(tool, str):
            try:
                tool = json.loads(tool)
            except json.JSONDecodeError:
                continue
        if isinstance(tool, dict):
            out.append(tool)
    return out


# --------------------------------------------------------------------------------------- the shared step
def map_call(call: dict, tool_map: dict) -> dict:
    """One normalized call with our tool name and argument keys."""
    name, arguments = call["name"], dict(call.get("arguments") or {})
    if name in tool_map:
        our_name, argument_map = tool_map[name]
        arguments = {argument_map.get(key, key): value for key, value in arguments.items()}
        name = our_name
    return {"name": name, "arguments": arguments}


def map_definition(definition: dict, tool_map: dict) -> dict:
    """A tool definition in our shape ({"type": "function", "function": {...}}), renamed and re-keyed when mapped."""
    function = dict(definition.get("function", definition))
    name = str(function.get("name", ""))
    if name in tool_map:
        our_name, argument_map = tool_map[name]
        description, our_properties = OUR_TOOL_DESCRIPTIONS.get(our_name, (function.get("description", ""), None))
        if our_properties is None:
            parameters = dict(function.get("parameters") or {})
            properties = {argument_map.get(key, key): value for key, value in (parameters.get("properties") or {}).items()}
            required = [argument_map.get(key, key) for key in parameters.get("required") or []]
            parameters = parameters | {"properties": properties, "required": required}
        else:
            parameters = {"type": "object", "properties": dict(our_properties), "required": list(our_properties)}
        function = {"name": our_name, "description": description, "parameters": parameters}
    else:
        function = {"name": name, "description": function.get("description", ""), "parameters": function.get("parameters") or {"type": "object", "properties": {}}}
    return {"type": "function", "function": function}


def reformat_trajectory(
    normalized: list[dict],
    tools: list[dict] | None,
    *,
    reasoning: str = "keep",
    tool_map: dict | None = None,
) -> tuple[list[dict], dict]:
    """
    Normalized messages (see the module docstring) -> (messages in our dialect, trajectory kwargs).
    The kwargs hold the mapped tool definitions (deduplicated by name, in first-seen order, plus a
    definition for every mapped tool the trajectory calls) and an empty system prompt for the loader to fill.
    """
    assert reasoning in ("keep", "drop"), reasoning
    tool_map = TOOL_MAP if tool_map is None else tool_map
    messages: list[dict] = []
    counter = 0
    used: dict[str, dict] = {}
    for message in normalized:
        role = message.get("role")
        if role == "system":
            continue
        if role == "assistant":
            calls = [map_call(call, tool_map) for call in message.get("tool_calls") or []]
            text = str(message.get("content") or "").strip()
            reasoning_text = (message.get("reasoning") or "").strip()
            if reasoning == "keep" and reasoning_text:
                text = f"{reasoning_text}\n\n{text}".strip() if text else reasoning_text
            out: dict = {"role": "assistant", "content": text}
            if calls:
                out["tool_calls"] = []
                for call in calls:
                    counter += 1
                    out["tool_calls"].append({"id": f"call_{counter:04d}", "type": "function",
                                              "function": {"name": call["name"], "arguments": call["arguments"]}})
                    used.setdefault(call["name"], call)
            messages.append(out)
        elif role == "tool":
            messages.append({"role": "tool", "content": str(message.get("content") or "")})
        else:
            messages.append({"role": "user", "content": str(message.get("content") or "")})
    definitions: dict[str, dict] = {}
    for definition in tools or []:
        mapped = map_definition(definition, tool_map)
        definitions.setdefault(mapped["function"]["name"], mapped)
    for name in used:
        if name not in definitions and name in OUR_TOOL_DESCRIPTIONS:
            description, properties = OUR_TOOL_DESCRIPTIONS[name]
            definitions[name] = {"type": "function", "function": {"name": name, "description": description,
                                 "parameters": {"type": "object", "properties": dict(properties), "required": list(properties)}}}
    return messages, {"tools": list(definitions.values()), "system_prompt": ""}


def submit_answer_definition() -> dict:
    """Use the actual runtime answer schema in representation-training data."""
    from ...agent.agent_tools import SubmitAnswerTool
    return {"type": "function", "function": {"name": SubmitAnswerTool.name,
            "description": SubmitAnswerTool.description, "parameters": SubmitAnswerTool.parameters}}


def observed_terminal_answer(messages: list[dict]) -> str | None:
    if messages and messages[-1].get("role") == "assistant":
        for call in messages[-1].get("tool_calls") or []:
            function = call.get("function", call)
            if function.get("name") == "submit_answer":
                return function["arguments"]["answer"]
    return None


def trajectory_chars(messages: list[dict]) -> int:
    """Characters of content and arguments, the cheap size measure loaders filter on."""
    total = 0
    for message in messages:
        total += len(str(message.get("content") or ""))
        for call in message.get("tool_calls") or []:
            total += len(json.dumps(call.get("function", call).get("arguments", {}), ensure_ascii=False))
    return total


def ngram_windows(texts: t.Iterable[str], n: int = 13) -> set[tuple[str, ...]]:
    """Every n-word window of the normalized texts (lower-cased, punctuation stripped): the decontamination key."""
    windows: set[tuple[str, ...]] = set()
    for text in texts:
        words = re.sub(r"[^0-9a-z]+", " ", str(text).lower()).split()
        for start in range(0, max(0, len(words) - n + 1)):
            windows.add(tuple(words[start:start + n]))
        if 0 < len(words) < n:
            windows.add(tuple(words))
    return windows


def parity_split(doc_id: str, held_out_every: int = 10):
    """A stable held-out split for trajectory corpora: one document in `held_out_every` (by hash) is TEST, the rest TRAIN."""
    import hashlib
    from ..dataset import DataSplit
    digest = int(hashlib.sha1(str(doc_id).encode()).hexdigest()[:8], 16)
    return DataSplit.TEST if digest % held_out_every == 0 else DataSplit.TRAIN
