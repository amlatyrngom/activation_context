"""
Activation-context parts as data: the part schema, the sentinel the chat template renders in place of
a part, and the split-and-tokenize step that turns a rendered template into token ids with placeholder
runs. Shared by the AC model (side tokenization) and the agent dialect (target prompts) without either
importing the other.
"""
from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

AC_PART_TYPE = "activation_context"
PART_SENTINEL = "⁣ACPART{index}⁣"     # rendered in place of a part by the chat template, split out before tokenizing
COMPACTION_INSTRUCTIONS = (
    "The conversation so far has been compacted into the activation context above. Continue the task "
    "from exactly where it left off, as if the full conversation were still in front of you."
)


def ac_part(messages: list[dict], ac_name: str, compression_target: float, kind: str | None = None) -> dict:
    """A part; `kind` names the channel that produced it (compaction, subagent_prompt, subagent_return, tool_output, search, parent_context)."""
    part = {"type": AC_PART_TYPE, "ac_name": ac_name, "compression_target": compression_target, "messages": messages}
    if kind is not None:
        part["kind"] = kind
    return part


RATIO_FIELD_BY_KIND = {                    # the AgentConfig field that sets a part's compression target, by channel
    "compaction": "ac_compaction_ratio", "tool_output": "ac_tool_output_ratio", "search": "ac_search_ratio",
    "subagent_prompt": "ac_subagent_ratio", "subagent_return": "ac_subagent_ratio", "parent_context": "ac_subagent_ratio",
}


def activation_part(kind: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
    """
    A part without encoder settings: what a tool or an agent produces (`ToolCallResult.activation_content`). The reader's
    `render_activation` / `resolve_activation_part` fills `ac_name` and `compression_target` by kind. `tools` are the
    definitions the compressed segment ran with; the encoder templates them natively when the side template takes tools.
    """
    part = {"type": AC_PART_TYPE, "kind": kind, "messages": messages}
    if tools:
        part["tools"] = tools
    return part


def resolve_activation_part(part: dict, config, ac_name: str | None = None) -> dict:
    """
    A deep copy of `part` with `ac_name` and `compression_target` filled from `config` (an AgentConfig: its
    ac_model_name and the ratio field of the part's kind), recursively for the parts nested in its messages.
    Parts that already carry both keep them.
    """
    from copy import deepcopy
    ac_name = ac_name or getattr(config, "ac_model_name", None)
    assert ac_name, "resolving a part needs the reader's ac_model_name"

    def resolve(item: dict) -> dict:
        out = deepcopy(item)
        out.setdefault("ac_name", ac_name)
        if out.get("compression_target") is None:
            field = RATIO_FIELD_BY_KIND.get(out.get("kind") or "", "ac_subagent_ratio")
            out["compression_target"] = float(getattr(config, field))
        out["messages"] = [resolve_in_message(message) for message in out.get("messages") or []]
        return out

    def resolve_in_message(message: dict) -> dict:
        content = message.get("content")
        if not isinstance(content, list):
            return message
        return dict(message, content=[resolve(piece) if is_ac_part(piece) else piece for piece in content])

    return resolve(part)


def tools_in_system_text(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """
    For a chat template without tool support: the tool definitions appended to the system message (one is inserted
    when the messages have none). With `tools` empty the messages come back unchanged.
    """
    if not tools:
        return list(messages)
    import json
    listing = "\n".join(json.dumps(definition) for definition in tools)
    block = f"\n\n# Tools\n\nThe functions available in this conversation:\n<tools>\n{listing}\n</tools>"
    out = [dict(message) for message in messages]
    for message in out:
        if message.get("role") == "system":
            content = message.get("content")
            text = content if isinstance(content, str) else "".join(
                piece.get("text", "") if isinstance(piece, dict) else str(piece) for piece in content or [])
            message["content"] = text + block
            return out
    return [{"role": "system", "content": block.strip()}] + out


def template_takes_tools(tokenizer: "PreTrainedTokenizerBase") -> bool:
    """Whether the tokenizer's chat template renders a `tools` argument (Qwen's do)."""
    template = getattr(tokenizer, "chat_template", None)
    return isinstance(template, str) and "tools" in template


def is_ac_part(part: object) -> bool:
    return isinstance(part, dict) and part.get("type") == AC_PART_TYPE


def strip_ac_parts(messages: list[dict]) -> list[dict]:
    """
    The same messages with every activation_context part removed (descendants included): the no-context view of a
    history. Text pieces of a content list are kept in order; a list left with only text becomes one string.
    """
    stripped = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            pieces = [piece for piece in content if not is_ac_part(piece)]
            if all(isinstance(piece, dict) and piece.get("type") == "text" for piece in pieces):
                content = "".join(piece.get("text", "") for piece in pieces)
            else:
                content = pieces
            message = {**message, "content": content}
        stripped.append(message)
    return stripped


def direct_parts(messages: list[dict]) -> list[dict]:
    """The activation_context parts of these messages, in order of appearance (not their descendants)."""
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts.extend(part for part in content if is_ac_part(part))
    return parts


def flatten_with_sentinels(messages: list[dict], parts: str = "sentinel") -> list[dict]:
    """
    Content lists joined into the text the chat template takes: text parts verbatim, activation_context
    parts as a numbered sentinel (`parts="sentinel"`) or dropped (`parts="drop"`, for history that only
    needs to render as some text). Other keys of a message (tool_calls, ...) are kept.
    """
    out = []
    index = 0
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            pieces = []
            for part in content:
                if is_ac_part(part):
                    if parts == "sentinel":
                        pieces.append(PART_SENTINEL.format(index=index))
                        index += 1
                elif isinstance(part, dict):
                    pieces.append(part.get("text", ""))
                else:
                    pieces.append(str(part))
            message["content"] = "".join(pieces)
        out.append(message)
    return out


def encode_with_part_sentinels(tokenizer: "PreTrainedTokenizerBase", text: str, part_lengths: list[int], pad_id: int) -> tuple[list[int], list[tuple[int, int]]]:
    """
    Token ids of a rendered template whose part sentinels become `pad_id` runs of the given lengths, plus
    the (start, end) span of each run. The pieces between sentinels are tokenized without special tokens,
    as the template's own tokenization would; the caller writes rows over the spans.
    """
    ids: list[int] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for part_index, length in enumerate(part_lengths):
        sentinel = PART_SENTINEL.format(index=part_index)
        at = text.find(sentinel, cursor)
        assert at >= 0, "the chat template did not render a part sentinel verbatim"
        ids.extend(tokenizer.encode(text[cursor:at], add_special_tokens=False))
        spans.append((len(ids), len(ids) + length))
        ids.extend([pad_id] * length)
        cursor = at + len(sentinel)
    ids.extend(tokenizer.encode(text[cursor:], add_special_tokens=False))
    return ids, spans
