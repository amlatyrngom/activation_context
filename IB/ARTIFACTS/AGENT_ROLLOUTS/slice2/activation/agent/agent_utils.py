"""
Model-specific details of talking to an agent model, kept out of the agent loop: how tool calls
appear in a reply (parsing), how a turn is rendered back to the model (assistant turns with calls,
tool results, the tool list), and the nudge. `ModelDialect.from_chat_template` reads them off the
model's chat template, so the same loop drives a Qwen3.5 (XML function blocks), a Hermes-style
model (JSON in <tool_call>) or a template without tool support (everything inlined as text).
Parsing is adaptive per block: every registered format is tried, the dialect only decides which
one is shown in error messages and used when rendering inline.

The dialect also owns the token-level view of a conversation (slice 2): the first prompt is templated
once, then the agent appends its own sampled tokens verbatim and only the wrapper the template puts
between a finished assistant turn and the next generation prompt (`continuation_token_ids`). The
model therefore reads exactly the tokens it wrote, one trajectory is one training sequence, and
prefix-cache hits are exact. `continuation_token_ids` renders the wrapper from the chat template with a
sentinel in place of the assistant text, so it needs no assumptions about the template's wording. Note
that templates re-render *history* in ways the model never saw while sampling (Qwen3 drops the empty
`<think>` block of finished turns, Qwen3.5 once a user message follows); the segments keep the
generation-prompt form the model actually read, which is what training must see.
"""
from __future__ import annotations

import json
import re
import typing as t
import uuid
from dataclasses import dataclass, field

if t.TYPE_CHECKING:
    from .agent_tools import ToolCallResult

PARSE_ERROR_TOOL = "parse_error"
ASSISTANT_SENTINEL = "\u241f TRESSOIR_ASSISTANT_TURN \u241f"   # never in real text; marks where the sampled tokens sit in a rendering
TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
TOOL_CALL_OPEN = re.compile(r"<tool_call>(?!.*</tool_call>)(.*)$", re.S)   # an unterminated block (max_tokens hit)
ParameterTypes = dict[str, dict[str, str]]   # tool name -> parameter name -> JSON schema type


# --------------------------------------------------------------------------------------- formats
class ToolCallFormat:
    """One way of writing a tool call inside a <tool_call> block."""
    name: str = ""
    reminder: str = ""   # one line telling the model the shape, used in parse errors

    def matches(self, block: str) -> bool:
        raise NotImplementedError

    def parse(self, block: str, parameter_types: ParameterTypes | None) -> tuple[str, dict]:
        """(tool name, arguments) or raise ValueError."""
        raise NotImplementedError

    def render(self, name: str, arguments: dict) -> str:
        """The block as the model would write it (inline rendering, examples)."""
        raise NotImplementedError


class XmlFunctionFormat(ToolCallFormat):
    """Qwen3.5 / Qwen3-Coder: <function=name><parameter=key>value</parameter></function>."""
    name = "xml_function"
    reminder = ("<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>, "
                "one <parameter> block per argument, values as plain text")
    FUNCTION = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
    PARAMETER = re.compile(r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.S)

    def matches(self, block: str) -> bool:
        return "<function=" in block

    def parse(self, block: str, parameter_types: ParameterTypes | None) -> tuple[str, dict]:
        match = self.FUNCTION.search(block)
        if match is None:
            raise ValueError("no <function=NAME> block")
        name, body = match.group(1).strip(), match.group(2)
        arguments = {}
        for parameter in self.PARAMETER.finditer(body):
            value = parameter.group(2)
            value = value[1:] if value.startswith("\n") else value
            value = value[:-1] if value.endswith("\n") else value
            arguments[parameter.group(1).strip()] = value
        return name, coerce_arguments(name, arguments, parameter_types)

    def render(self, name: str, arguments: dict) -> str:
        parts = [f"<function={name}>"]
        for key, value in arguments.items():
            text = value if isinstance(value, str) else json.dumps(value)
            parts.append(f"<parameter={key}>\n{text}\n</parameter>")
        parts.append("</function>")
        return "<tool_call>\n" + "\n".join(parts) + "\n</tool_call>"


class HermesJsonFormat(ToolCallFormat):
    """Qwen3 / Qwen2.5 / Hermes: one JSON object {"name": ..., "arguments": {...}} per block."""
    name = "hermes_json"
    reminder = '<tool_call>{"name": NAME, "arguments": {KEY: VALUE, ...}}</tool_call>, one JSON object per block'

    def matches(self, block: str) -> bool:
        return block.lstrip().startswith("{")

    def parse(self, block: str, parameter_types: ParameterTypes | None) -> tuple[str, dict]:
        data = json.loads(block)
        arguments = data.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments.strip() else {}
        if not isinstance(data.get("name"), str) or not isinstance(arguments, dict):
            raise ValueError("a tool call needs a string 'name' and an object 'arguments'")
        return data["name"], arguments

    def render(self, name: str, arguments: dict) -> str:
        return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


FORMATS: list[ToolCallFormat] = [XmlFunctionFormat(), HermesJsonFormat()]


def coerce_arguments(tool_name: str, arguments: dict, parameter_types: ParameterTypes | None) -> dict:
    """Text values (XML format) become what the schema declares: integer, number, boolean, object, array."""
    types = (parameter_types or {}).get(tool_name, {})
    out = {}
    for key, value in arguments.items():
        declared = types.get(key)
        if isinstance(value, str) and declared and declared != "string":
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass   # the tool's own TypeError/ValueError reaches the model
        out[key] = value
    return out


def parse_tool_calls(
    text: str, parameter_types: ParameterTypes | None = None, formats: list[ToolCallFormat] | None = None,
    preferred: ToolCallFormat | None = None,
) -> tuple[str, list[dict]]:
    """
    The assistant text without the <tool_call> blocks, and the calls as [{"id", "name", "arguments"}].
    Each block is parsed by the first format that recognises it; a block no format accepts becomes a
    `parse_error` call whose arguments carry the raw text and a reminder of the preferred shape.
    """
    formats = formats or FORMATS
    preferred = preferred or formats[0]
    calls = []
    blocks = [match.group(1) for match in TOOL_CALL_BLOCK.finditer(text)]
    tail = TOOL_CALL_OPEN.search(TOOL_CALL_BLOCK.sub("", text))
    if tail is not None and tail.group(1).strip():
        blocks.append(tail.group(1))   # an unterminated block still gets a readable error
    for block in blocks:
        call_id = uuid.uuid4().hex[:8]
        error = None
        for fmt in [f for f in formats if f.matches(block)] or [preferred]:
            try:
                name, arguments = fmt.parse(block, parameter_types)
                calls.append({"id": call_id, "name": name, "arguments": arguments})
                error = None
                break
            except (json.JSONDecodeError, ValueError, AttributeError) as failure:
                error = f"{fmt.name}: {failure}"
        if error is not None:
            calls.append({"id": call_id, "name": PARSE_ERROR_TOOL,
                          "arguments": {"raw": block[:2000], "error": error, "expected": preferred.reminder}})
    content = TOOL_CALL_OPEN.sub("", TOOL_CALL_BLOCK.sub("", text)).strip()
    return content, calls


def parameter_types_of(tool_definitions: list[dict]) -> ParameterTypes:
    """Tool name -> parameter -> declared type, from the function schemas."""
    out = {}
    for definition in tool_definitions:
        function = definition.get("function", definition)
        properties = (function.get("parameters") or {}).get("properties") or {}
        out[function["name"]] = {key: str(spec.get("type", "")) for key, spec in properties.items() if isinstance(spec, dict)}
    return out


# --------------------------------------------------------------------------------------- rendering
class MessageRendering:
    """How a turn goes back to the model. The chat template gets structured messages by default."""

    def system_prompt(self, system_prompt: str, tool_definitions: list[dict]) -> str:
        return system_prompt

    def template_tools(self, tool_definitions: list[dict]) -> list[dict] | None:
        return tool_definitions

    def assistant_message(self, content: str, calls: list[dict]) -> dict:
        message: dict = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = [
                {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
                for call in calls
            ]
        return message

    def tool_messages(self, calls: list[dict], results: list["ToolCallResult"]) -> list[dict]:
        return [{"role": "tool", "content": result.output} for result in results]


class InlineRendering(MessageRendering):
    """
    For templates without tool support: the tool list goes into the system prompt, calls are written
    into the assistant text in the preferred format and results come back as one user message of
    <tool_response> blocks (the shape Qwen's own templates produce).
    """
    def __init__(self, fmt: ToolCallFormat):
        self.format = fmt

    def system_prompt(self, system_prompt: str, tool_definitions: list[dict]) -> str:
        listing = "\n".join(json.dumps(definition) for definition in tool_definitions)
        return (f"{system_prompt}\n\n# Tools\n\nYou may call these functions:\n<tools>\n{listing}\n</tools>\n\n"
                f"To call one, reply in this exact shape: {self.format.reminder}. Results come back in <tool_response> blocks.")

    def template_tools(self, tool_definitions: list[dict]) -> list[dict] | None:
        return None

    def assistant_message(self, content: str, calls: list[dict]) -> dict:
        blocks = [self.format.render(call["name"], call["arguments"]) for call in calls]
        return {"role": "assistant", "content": "\n\n".join([content] + blocks if content else blocks)}

    def tool_messages(self, calls: list[dict], results: list["ToolCallResult"]) -> list[dict]:
        body = "\n".join(f"<tool_response>\n{result.output}\n</tool_response>" for result in results)
        return [{"role": "user", "content": body}]


# --------------------------------------------------------------------------------------- dialect
@dataclass
class ModelDialect:
    preferred_format: ToolCallFormat = field(default_factory=XmlFunctionFormat)
    formats: list[ToolCallFormat] = field(default_factory=lambda: list(FORMATS))
    rendering: MessageRendering = field(default_factory=MessageRendering)
    nudge_message: str = "Call one of the provided tools or submit_answer when done."

    @classmethod
    def from_chat_template(cls, chat_template: str | None) -> "ModelDialect":
        """
        Read the model's habits off its chat template: the call shape it teaches (XML function blocks
        vs JSON), and whether it renders `tools` / `tool_calls` at all (else everything is inlined).
        """
        template = chat_template or ""
        if "<function=" in template:
            preferred: ToolCallFormat = XmlFunctionFormat()
        elif "<tool_call>" in template or '"arguments"' in template:
            preferred = HermesJsonFormat()
        else:
            preferred = FORMATS[0]
        formats = [preferred] + [fmt for fmt in FORMATS if fmt.name != preferred.name]
        supports_tools = "tools" in template and "tool_calls" in template
        rendering = MessageRendering() if supports_tools else InlineRendering(preferred)
        return cls(preferred_format=preferred, formats=formats, rendering=rendering)

    @classmethod
    def for_tokenizer(cls, tokenizer) -> "ModelDialect":
        return cls.from_chat_template(getattr(tokenizer, "chat_template", None))

    def parse(self, text: str, parameter_types: ParameterTypes | None = None) -> tuple[str, list[dict]]:
        return parse_tool_calls(text, parameter_types, self.formats, self.preferred_format)

    def parse_error_message(self, arguments: dict) -> str:
        return (f"Could not parse the tool call ({arguments.get('error')}). Expected shape: "
                f"{arguments.get('expected') or self.preferred_format.reminder}.")

    # ------------------------------------------------------------------------------- tokens
    def prompt_token_ids(self, tokenizer, messages: list[dict], tools: list[dict] | None = None,
                         chat_template_kwargs: dict | None = None) -> list[int]:
        """The first prompt of a conversation: the chat template, once, with the generation prompt open."""
        return _template_ids(tokenizer, messages, tools, True, chat_template_kwargs)

    def continuation_token_ids(self, tokenizer, messages_before: list[dict], new_messages: list[dict],
                               tools: list[dict] | None = None, chat_template_kwargs: dict | None = None) -> list[int]:
        """
        The tokens the model reads between its own sampled text and its next turn: the end-of-turn
        marker, `new_messages` (tool results, or the nudge) and the next generation prompt, exactly as
        the chat template renders them. Rendered with a sentinel as the assistant content, so the
        wrapper is independent of what the model wrote. Starts with the end-of-turn token(s); the caller
        drops the first one when the sample already ended with it.
        """
        history = list(messages_before) + [{"role": "assistant", "content": ASSISTANT_SENTINEL}] + list(new_messages)
        text = _template_text(tokenizer, history, tools, True, chat_template_kwargs)
        index = text.rfind(ASSISTANT_SENTINEL)
        if index < 0:
            raise ValueError("the chat template did not render the assistant content verbatim; cannot derive the turn wrapper")
        wrapper = text[index + len(ASSISTANT_SENTINEL):]
        return list(tokenizer.encode(wrapper, add_special_tokens=False))

    @staticmethod
    def join_continuation(sampled_token_ids: list[int], wrapper_token_ids: list[int]) -> list[int]:
        """The wrapper without its leading end-of-turn token when the sample already produced it."""
        if sampled_token_ids and wrapper_token_ids and sampled_token_ids[-1] == wrapper_token_ids[0]:
            return wrapper_token_ids[1:]
        return wrapper_token_ids


def _template_text(tokenizer, messages: list[dict], tools, add_generation_prompt: bool, chat_template_kwargs: dict | None) -> str:
    return tokenizer.apply_chat_template(
        _flatten(messages), tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False, **(chat_template_kwargs or {}),
    )


def _template_ids(tokenizer, messages: list[dict], tools, add_generation_prompt: bool, chat_template_kwargs: dict | None) -> list[int]:
    ids = tokenizer.apply_chat_template(
        _flatten(messages), tools=tools, add_generation_prompt=add_generation_prompt, tokenize=True, **(chat_template_kwargs or {}),
    )
    if hasattr(ids, "keys"):                                                      # BatchEncoding / dict
        ids = ids["input_ids"]
    return list(ids)


def _flatten(messages: list[dict]) -> list[dict]:
    """Text-only content parts joined, the shape our templates take."""
    out = []
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        out.append(message)
    return out
