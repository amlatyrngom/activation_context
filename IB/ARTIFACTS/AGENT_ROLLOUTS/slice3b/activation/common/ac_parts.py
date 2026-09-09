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


def is_ac_part(part: object) -> bool:
    return isinstance(part, dict) and part.get("type") == AC_PART_TYPE


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
