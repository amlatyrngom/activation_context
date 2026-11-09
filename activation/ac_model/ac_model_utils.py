"""
Building blocks of the activation-context model (ac_model.py): part-aware tokenization, the mixer's
blocked local attention, windowed pooling, the row heads, the CPU row cache and the encode queue.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import random
import re
import shutil
import threading
import time
import typing as t
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, field
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

if t.TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from activation.common.ac_parts import (   # the part schema and the sentinel split live in common; re-exported here
    template_takes_tools, tools_in_system_text,
    AC_PART_TYPE, PART_SENTINEL, direct_parts, encode_with_part_sentinels, flatten_with_sentinels, is_ac_part,
)


# --------------------------------------------------------------------------------------- parts and keys
def normalize_messages(messages: list[dict] | str) -> list[dict]:
    """A plain string becomes one user message."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return list(messages)


def qa_encoder_messages(messages, question, ac_name, ratio, *, encoder_sees_question=True,
                        parent_messages=None):
    """Question-aware, general-summary, or compressed-parent conditioning for QA."""
    from ..common.ac_parts import ac_part
    task = [{"role": "user", "content": f"Question: {question}" if encoder_sees_question else
             "Summarize the useful facts and exact details in this context for later use."}]
    if parent_messages is None:
        return deepcopy(messages) + task
    if not encoder_sees_question:
        raise ValueError("Compressed question-bearing parent requires encoder_sees_question=True")
    parent = ac_part(deepcopy(parent_messages) + task, ac_name, ratio)
    # The fresh-evidence encoder only receives the question through its parent's rows.
    evidence = deepcopy(messages)
    insertion = 1 if evidence and evidence[0].get("role") == "system" else 0
    evidence.insert(insertion, {"role": "user", "content": [parent]})
    return evidence


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def row_cache_key(messages: list[dict], compression_ratio: float, is_recursive: bool, version: int, tools: list[dict] | None = None) -> str:
    """sha256 of the canonical part: the same messages, tools, ratio, kind and model version hit the same rows."""
    payload = canonical_json({"messages": messages, "ratio": round(float(compression_ratio), 8), "recursive": bool(is_recursive), "version": int(version),
                              **({"tools": tools} if tools else {})})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tokenize_with_parts(
    tokenizer: "PreTrainedTokenizerBase",
    messages: list[dict],
    part_lengths: list[int],
    pad_id: int,
    *,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = False,
    chat_template_kwargs: dict | None = None,
) -> tuple[list[int], list[tuple[int, int]]]:
    """
    Token ids of the messages under the chat template with every activation_context part replaced by
    `part_lengths[k]` placeholder ids (`pad_id`), plus the (start, end) span of each part. The template
    renders one sentinel string per part; the text is split at the sentinels and the pieces tokenized
    (no special tokens, as the template's own tokenization). The caller writes rows over the spans.
    """
    index = sum(1 for _ in direct_parts(messages))
    assert index == len(part_lengths), f"{index} parts in the messages, {len(part_lengths)} lengths given"
    text = render_with_part_sentinels(tokenizer, messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                      chat_template_kwargs=chat_template_kwargs)
    return encode_with_part_sentinels(tokenizer, text, list(part_lengths), pad_id)


def render_with_part_sentinels(tokenizer: "PreTrainedTokenizerBase", messages: list[dict], *, tools: list[dict] | None = None,
                               add_generation_prompt: bool = False, chat_template_kwargs: dict | None = None) -> str:
    """The chat-template text tokenize_with_parts encodes: one sentinel string per direct part."""
    flat = [_training_template_message(tokenizer, message, reasoning_as_text=True)
            for message in flatten_with_sentinels(messages, parts="sentinel")]
    if tools and not template_takes_tools(tokenizer):
        flat, tools = tools_in_system_text(flat, tools), None                     # the definitions ride in the system text instead
    if not any(message.get("role") == "user" and not str(message.get("content") or "").strip().startswith("<tool_response>") for message in flat):
        flat.insert(0, {"role": "user", "content": ""})                    # Qwen templates refuse a conversation without a user query (a mid-trajectory segment)
    return tokenizer.apply_chat_template(flat, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False,
                                         **(chat_template_kwargs or {}))


def batch_token_counts(tokenizer: "PreTrainedTokenizerBase", texts: list[str], chunk: int = 1024) -> list[int]:
    """len(tokenizer.encode(text, add_special_tokens=False)) of each text, through the tokenizer's batched call with its
    Rust threads on (TOKENIZERS_PARALLELISM=false would run the batch on one thread; the setting is restored after)."""
    previous = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    try:
        return [len(ids) for start in range(0, len(texts), chunk)
                for ids in tokenizer(texts[start:start + chunk], add_special_tokens=False)["input_ids"]]
    finally:
        if previous is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = previous


def _training_template_message(tokenizer, message: dict, *, reasoning_as_text: bool = False) -> dict:
    """Keep separate reasoning and structured calls visible under the selected template."""
    from ..agent.agent_utils import ModelDialect
    dialect = ModelDialect.for_tokenizer(tokenizer)
    message = dict(message)
    if reasoning_as_text and message.get("role") == "assistant" and isinstance(message.get("content"), str):
        # Agent records may carry inline reasoning. Escape its delimiters so templates cannot strip earlier thoughts.
        message["content"] = message["content"].replace("<think>", "[Reasoning]").replace("</think>", "[/Reasoning]")
    reasoning = message.pop("reasoning", None) or message.get("reasoning_content")
    if reasoning:
        if not reasoning_as_text and "reasoning" in (tokenizer.chat_template or ""):
            message["reasoning_content"] = reasoning
        else:
            message.pop("reasoning_content", None)
            message["content"] = str(reasoning) + "\n\n" + str(message.get("content") or "")
    if not template_takes_tools(tokenizer) and message.get("tool_calls"):
        calls = [dict(call.get("function", call), id=call.get("id", f"call_{index}"))
                 for index, call in enumerate(message["tool_calls"])]
        message = dialect.rendering.assistant_message(message.get("content") or "", calls)
    return message


def assistant_target_ids(tokenizer, message: dict, tools: list[dict] | None = None,
                         chat_template_kwargs: dict | None = None) -> list[int]:
    """Render a complete public assistant output once, excluding the supplied generation prompt."""
    message = _training_template_message(tokenizer, message)
    prefix = [{"role": "user", "content": ""}]
    if tools and not template_takes_tools(tokenizer):
        prefix, tools = tools_in_system_text(prefix, tools), None
    kwargs = {"enable_thinking": bool(message.get("reasoning_content"))} if chat_template_kwargs is None else chat_template_kwargs
    opened = tokenizer.apply_chat_template(prefix, tools=tools, tokenize=False, add_generation_prompt=True, **kwargs)
    completed = tokenizer.apply_chat_template(prefix + [message], tools=tools, tokenize=False, add_generation_prompt=False, **kwargs)
    if not completed.startswith(opened):
        raise ValueError("Assistant output does not extend this tokenizer's generation prompt")
    output = completed[len(opened):]
    from ..harness.hf_utils import canonical_eot_token
    eot = canonical_eot_token(tokenizer) or tokenizer.eos_token
    if eot and eot in output and not output.rsplit(eot, 1)[1].strip():
        output = output[:output.rfind(eot) + len(eot)]
    ids = tokenizer.encode(output, add_special_tokens=False)
    if not ids:
        raise ValueError("Assistant output has no target tokens")
    return list(ids)


def without_stop_token(tokenizer, ids: list[int]) -> list[int]:
    """`ids` (an assistant_target_ids output) without its trailing end-of-turn token(s): a target whose stop position is
    unpredictable by construction then carries no loss for it. Refuses a target that does not end with the end-of-turn."""
    from ..harness.hf_utils import canonical_eot_token
    eot = canonical_eot_token(tokenizer) or tokenizer.eos_token
    stop = tokenizer.encode(eot, add_special_tokens=False) if eot else []
    if not stop or len(ids) <= len(stop) or ids[-len(stop):] != stop:
        raise ValueError("The target does not end with the tokenizer's end-of-turn token")
    return list(ids[:-len(stop)])


def cached_vocabulary_size(tokenizer) -> int:
    """len(tokenizer) walks the added-token table every call (~30 ms on Qwen3.5; it was 96 % of example building): computed once per tokenizer object."""
    size = tokenizer.__dict__.get("_activation_vocabulary_size")
    if size is None:
        size = tokenizer.__dict__["_activation_vocabulary_size"] = len(tokenizer)
    return size


def prepare_training_history(tokenizer, messages: list[dict], start: int, assistant_ids: list[list[int]],
                             part_lengths: list[int], *, tools: list[dict] | None = None,
                             chat_template_kwargs: dict | None = None) -> tuple[list[int], list[tuple[int, int]], list[int]]:
    """Incremental runtime-shaped tokens, AC spans and predicting positions for one history."""
    from ..agent.agent_utils import ModelDialect
    if not isinstance(start, int) or not 0 <= start < len(messages):
        raise ValueError("Training history start lies outside its messages")
    selected = [index for index in range(start, len(messages)) if messages[index].get("role") == "assistant"]
    if len(selected) != len(assistant_ids) or not selected:
        raise ValueError("Every retained assistant message needs exactly one target-token sequence")
    vocabulary_size = cached_vocabulary_size(tokenizer)
    for ids in assistant_ids:
        if not ids or any(type(token) is not int or not 0 <= token < vocabulary_size for token in ids):
            raise ValueError("Assistant targets must be nonempty valid tokenizer IDs")
    if any(direct_parts([message]) for message in messages if message.get("role") == "assistant"):
        raise ValueError("Assistant outputs cannot contain AC input parts")
    dialect = ModelDialect.for_tokenizer(tokenizer)
    chat_template_kwargs = {"enable_thinking": False} if chat_template_kwargs is None else chat_template_kwargs
    messages = list(messages)
    if tools and not template_takes_tools(tokenizer):
        previous_count = len(messages)
        messages, tools = tools_in_system_text(messages, tools), None
        start += len(messages) - previous_count
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    cursor = 0
    def lengths(block):
        nonlocal cursor
        count = len(direct_parts(block))
        values = part_lengths[cursor:cursor + count]
        cursor += count
        if len(values) != count:
            raise ValueError("AC part lengths do not match the message history")
        return values
    first = next(index for index in range(start, len(messages)) if messages[index].get("role") == "assistant")
    if first == 0:
        raise ValueError("An assistant target needs preceding prompt context")
    prefix = [_training_template_message(tokenizer, message, reasoning_as_text=True) for message in messages[:first]]
    ids, spans = dialect.prompt_tokens(tokenizer, prefix, tools, chat_template_kwargs, lengths(messages[:first]), pad)
    if not ids:
        raise ValueError("An assistant target needs a nonempty token prefix")
    positions = []
    selected_ids = iter(assistant_ids)
    index = first
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "assistant":
            raise ValueError("Expected an assistant output after a generation prompt")
        output = list(next(selected_ids))
        positions.extend(range(len(ids) - 1, len(ids) + len(output) - 1))
        ids.extend(output)
        end = index + 1
        while end < len(messages) and messages[end].get("role") != "assistant":
            end += 1
        if end > index + 1 or end < len(messages):
            following = messages[index + 1:end]
            wrapper, local_spans = dialect.continuation_tokens(tokenizer, messages[:index + 1], following,
                                tools, chat_template_kwargs, lengths(following), pad)
            joined = dialect.join_continuation(output, wrapper)
            shift = len(wrapper) - len(joined)
            spans.extend((len(ids) + a - shift, len(ids) + b - shift) for a, b in local_spans)
            ids.extend(joined)
        index = end
    if cursor != len(part_lengths):
        raise ValueError("Unused AC part lengths in the message history")
    return ids, spans, positions


def validate_paired_histories(teacher: list[dict], teacher_start: int, student: list[dict], student_start: int) -> None:
    """Suffix roles/calls agree; tool evidence can differ, assistant outputs cannot."""
    if not 0 <= teacher_start < len(teacher) or not 0 <= student_start < len(student):
        raise ValueError("Paired history starts lie outside their messages")
    left, right = teacher[teacher_start:], student[student_start:]
    if len(left) != len(right):
        raise ValueError("Paired histories have different retained message counts")
    for a, b in zip(left, right):
        if a.get("role") != b.get("role"):
            raise ValueError("Paired histories have different retained role order")
        for key in ("tool_calls", "tool_call_id", "name"):
            if a.get(key) != b.get(key):
                raise ValueError("Paired histories have different retained call identity/order")
        if a.get("role") == "assistant" and any(a.get(key) != b.get(key) for key in ("content", "reasoning", "reasoning_content")):
            raise ValueError("Paired histories have different assistant outputs")


def validate_part_capacity(ac_model, request, capacity: int) -> None:
    """Check every recursive side forward: content tokens plus summary rows."""
    children = [ac_model._child_request(part, request.compression_ratio) for part in direct_parts(request.messages)]
    lengths = [ac_model.part_view_rows(child.messages, child.compression_ratio, child.tools) for child in children]
    count = ac_model.side_token_count(request.messages, lengths, request.tools)
    total = count + ac_model.num_view_rows(count, request.compression_ratio)      # the content tokens plus the summary rows
    if total > capacity:
        raise ValueError(f"AC side forward needs {total} positions, exceeding supported capacity {capacity}")
    for child in children:
        validate_part_capacity(ac_model, child, capacity)


def normalize_public_assistants(tokenizer, messages: list[dict], tools: list[dict] | None,
                                max_tokens: int) -> tuple[list[dict], list[dict] | None]:
    """Split only public text/reasoning into complete, budgeted Python continuation turns."""
    from ..dataset.loaders.trajectory_utils import map_definition, TOOL_MAP
    if max_tokens < 1:
        raise ValueError("max_assistant_tokens must be positive")
    messages, tools = deepcopy(messages), deepcopy(tools or [])
    used = {call.get("id") for message in messages for call in message.get("tool_calls") or []}
    output, counter = [], 0
    for message in messages:
        if message.get("role") != "assistant":
            output.append(message)
            continue
        while len(assistant_target_ids(tokenizer, message, tools, {"enable_thinking": True})) > max_tokens:
            field = next((key for key in ("reasoning", "reasoning_content", "content")
                          if isinstance(message.get(key), str) and message[key]), None)
            if field is None:
                raise ValueError("Public assistant has an indivisible call exceeding its token budget")
            counter += 1
            while f"ac_continue_{counter}" in used:
                counter += 1
            call_id = f"ac_continue_{counter}"
            used.add(call_id)
            if not any(tool.get("function", tool).get("name") == "python" for tool in tools):
                tools.append(map_definition({"name": "python"}, TOOL_MAP))
            call = {"id": call_id, "type": "function", "function": {
                "name": "python", "arguments": {"code": f"print('continue {counter}')"}}}
            def turn(text):
                return {"role": "assistant", "content": "", field: text, "tool_calls": [call]}
            text = message[field]
            low, high, best = 1, len(text), 0
            while low <= high:
                middle = (low + high) // 2
                if len(assistant_target_ids(tokenizer, turn(text[:middle]), tools, {"enable_thinking": True})) <= max_tokens:
                    best, low = middle, middle + 1
                else:
                    high = middle - 1
            if not best:
                raise ValueError("Public continuation call cannot fit the assistant token budget")
            output.extend([turn(text[:best]), {"role": "tool", "name": "python", "tool_call_id": call_id,
                                               "content": f"continue {counter}\n"}])
            message[field] = text[best:]
        output.append(message)
    return output, tools or None


def complete_public_turns(messages: list[dict]) -> list[list[dict]]:
    """Assistant outputs with their complete replies and intervening input, without dangling calls."""
    groups = []
    index = 0
    while index < len(messages):
        if messages[index].get("role") != "assistant":
            raise ValueError("Public continuation must begin with an assistant")
        end = index + 1
        while end < len(messages) and messages[end].get("role") != "assistant":
            end += 1
        group = messages[index:end]
        calls = group[0].get("tool_calls") or []
        replies = [message for message in group[1:] if message.get("role") == "tool"]
        terminal_answer = (end == len(messages) and len(group) == 1 and len(calls) == 1
                           and calls[0].get("function", calls[0]).get("name") == "submit_answer")
        if len(calls) != len(replies) and not terminal_answer:
            raise ValueError("Public trajectory has unmatched tool calls or replies")
        for call, reply in zip(calls, replies):
            if reply.get("tool_call_id") and reply["tool_call_id"] != call.get("id"):
                raise ValueError("Public tool reply does not match its call")
        if end == len(messages):
            while len(group) > 1 and group[-1].get("role") in ("user", "system"):
                group = group[:-1]
        groups.append(group)
        index = end
    return groups


def sinusoidal_positions(length: int, d: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[length, d] fixed sinusoidal positions (the mixer is bidirectional and has no positions of its own)."""
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    half = d // 2
    frequency = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(1, half))
    angles = position * frequency[None, :]
    out = torch.zeros(length, d, device=device, dtype=torch.float32)
    out[:, 0:half] = torch.sin(angles)
    out[:, half:2 * half] = torch.cos(angles)
    return out.to(dtype)


# --------------------------------------------------------------------------------------- modules
class FeedForward(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_in, d_hidden)
        self.down = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class BlockedLocalAttentionLayer(nn.Module):
    """
    Pre-norm bidirectional attention where every block of `block` rows attends to itself and its two
    neighbours (each row sees at least `block` rows on each side, at most 2·block), then an FFN x4.
    Runs SDPA on the reshaped [B·blocks, heads, block, 3·block] tensors: the cost is linear in the
    length and no N x N mask exists (a materialized band mask at 32k rows would be 1 GB of booleans).
    """

    def __init__(self, d: int, num_heads: int, block: int) -> None:
        super().__init__()
        assert d % num_heads == 0
        self.d, self.num_heads, self.block = d, num_heads, block
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = FeedForward(d, 4 * d, d)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x [B, N, d], valid [B, N] bool -> [B, N, d]."""
        batch_size, length, d = x.shape
        block, heads, head_dim = self.block, self.num_heads, d // self.num_heads
        padded = -(-length // block) * block
        h = self.norm1(x)
        if padded != length:
            h = F.pad(h, (0, 0, 0, padded - length))
            valid = F.pad(valid, (0, padded - length), value=False)
        num_blocks = padded // block
        q, k, v = self.qkv(h).chunk(3, dim=-1)                                                    # [B, P, d]

        def blocks(tensor: torch.Tensor) -> torch.Tensor:                                          # [B, P, d] -> [B, nb, H, b, hd]
            return tensor.view(batch_size, num_blocks, block, heads, head_dim).transpose(2, 3)

        def windows(tensor: torch.Tensor) -> torch.Tensor:                                         # [B, nb, H, b, hd] -> [B, nb, H, 3b, hd]
            previous = F.pad(tensor[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
            following = F.pad(tensor[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1))
            return torch.cat([previous, tensor, following], dim=3)

        q_blocks = blocks(q).reshape(batch_size * num_blocks, heads, block, head_dim)
        k_windows = windows(blocks(k)).reshape(batch_size * num_blocks, heads, 3 * block, head_dim)
        v_windows = windows(blocks(v)).reshape(batch_size * num_blocks, heads, 3 * block, head_dim)
        valid_blocks = valid.view(batch_size, num_blocks, block)
        key_valid = torch.cat([F.pad(valid_blocks[:, :-1], (0, 0, 1, 0), value=False), valid_blocks,
                               F.pad(valid_blocks[:, 1:], (0, 0, 0, 1), value=False)], dim=2)      # [B, nb, 3b]
        # Padded queries attend everything in their window (their rows are discarded): no all-masked row, no NaN.
        mask = key_valid[:, :, None, :] | ~valid_blocks[:, :, :, None]                             # [B, nb, b, 3b]
        mask = mask.reshape(batch_size * num_blocks, 1, block, 3 * block)
        attended = F.scaled_dot_product_attention(q_blocks, k_windows, v_windows, attn_mask=mask)  # [B·nb, H, b, hd]
        attended = attended.view(batch_size, num_blocks, heads, block, head_dim).transpose(2, 3).reshape(batch_size, padded, d)
        x = x + self.residual_scale * self.proj(attended[:, :length])
        return x + self.residual_scale * self.ffn(self.norm2(x))


class WindowedPooling(nn.Module):
    """
    Self-attention on `window`-wide windows with `stride` stride, pooled to one vector per window,
    then an FFN: the token sequence becomes about 1/stride of its length (the retrieval model's
    byte pooling, on token rows).
    """

    def __init__(self, d: int, num_heads: int, window: int, stride: int) -> None:
        super().__init__()
        self.window, self.stride = window, stride
        self.norm = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, num_heads, dropout=0.0, batch_first=True)
        self.ffn = FeedForward(d, 4 * d, d)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def num_windows(self, length: int) -> int:
        return 1 if length <= self.window else (length - self.window + self.stride - 1) // self.stride + 1

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, L, d], valid [B, L] bool -> pooled [B, W, d], window mask [B, W] bool (exactly the windows a row gets alone)."""
        batch_size, length, d = x.shape
        window, stride = self.window, self.stride
        pad = (-(length - window)) % stride if length > window else window - length
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            valid = F.pad(valid, (0, pad), value=False)
        windows = x.unfold(1, window, stride).permute(0, 1, 3, 2)                                  # [B, W, window, d]
        window_masks = valid.unfold(1, window, stride)                                              # [B, W, window]
        num_windows = windows.shape[1]
        flat = windows.reshape(batch_size * num_windows, window, d)
        flat_masks = window_masks.reshape(batch_size * num_windows, window)
        lengths = valid.sum(dim=1)
        num_valid = torch.where(lengths > window, (lengths - window + stride - 1) // stride + 1, torch.ones_like(lengths))
        window_valid = (torch.arange(num_windows, device=x.device)[None, :] < num_valid[:, None]).reshape(-1)
        key_padding = ~flat_masks
        key_padding[~window_valid] = False                                                          # invalid windows attend harmlessly, masked downstream
        normed = self.norm(flat)
        attended, _ = self.attention(normed, normed, normed, key_padding_mask=key_padding, need_weights=False)
        flat = flat + self.residual_scale * attended
        weights = flat_masks.to(flat.dtype)[..., None]
        pooled = (flat * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = pooled + self.residual_scale * self.ffn(pooled)
        return pooled.view(batch_size, num_windows, d), window_valid.view(batch_size, num_windows)


def unit_rows(rows: torch.Tensor) -> torch.Tensor:
    """Normalize fp32 rows without mixing content across positions."""
    rows = rows.float()
    return rows * rows.square().mean(dim=-1, keepdim=True).clamp_min(1e-8).rsqrt()


DENSE_RE = re.compile(r"[0-9=+*/^_{}\[\]<>|\\#$%&~@`]")


def is_dense_piece(piece: str) -> bool:
    """Whether a token's text is "dense" content: a digit, a code/math symbol (DENSE_RE), or a non-ASCII character that is neither a
    letter nor typographic punctuation / space (the rule of the study's density-weighted draws, shared with the bin weights)."""
    piece = piece.replace("�", "")   # a byte-level token that is a fragment of a multi-byte character decodes to the replacement character
    return bool(DENSE_RE.search(piece)) or (not piece.isascii() and any(
        ord(ch) > 127 and not ch.isalpha() and unicodedata.category(ch) not in ("Pi", "Pf", "Pd", "Po", "Ps", "Pe", "Zs") for ch in piece))


def summary_bins(length: int, num_rows: int, device: torch.device, weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The V ordered token-count bins of N content rows, as [start, end) bounds: floor/ceil bins cover the input; V>N repeats
    nonempty bins, never padding. With `weights` (one positive weight per token) and V < N the bins hold equal cumulative weight
    instead: bin k runs from the token whose cumulative weight first exceeds k·W/V to the one at which it first reaches (k+1)·W/V
    (the floor/ceil rule applied to the weight axis: uniform weights give the token-count bins' ends, bins stay nonempty)."""
    if length == 0 or num_rows < 1:
        raise ValueError("Summary pooling needs nonempty content and a positive row count")
    ordinal = torch.arange(num_rows, device=device)
    if weights is None or num_rows >= length:
        return ordinal * length // num_rows, ((ordinal + 1) * length + num_rows - 1) // num_rows
    if weights.shape != (length,):
        raise ValueError("Bin weights must hold one weight per content token")
    cumulative = weights.double().cumsum(0)                                            # double: k·W/V must not round onto an integer sum (N·V > 2^24)
    targets = cumulative[-1] * ordinal[1:].double() / num_rows                         # the V-1 interior cuts k·W/V
    starts = torch.cat([ordinal[:1], torch.searchsorted(cumulative, targets, right=True)])
    ends = torch.cat([torch.searchsorted(cumulative, targets, right=False) + 1, torch.tensor([length], device=device)])
    return starts, ends


def interleaved_positions(length: int, num_rows: int, device: torch.device, weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoder positions of the N content rows and of the V summaries when each summary follows its pooling bin: summary k
    sits right after its bin's last token and after the k summaries before it; token j after the summaries whose bins end at
    or before it. A permutation of range(N + V). `weights`: the bins' token weights (summary_bins)."""
    _, ends = summary_bins(length, num_rows, device, weights)
    tokens = torch.arange(length, device=device)
    return tokens + torch.searchsorted(ends, tokens, right=True), ends + torch.arange(num_rows, device=device)


class AdaptiveSummaryPooling(nn.Module):
    """V content-derived mean queries, each attending its ordered bin of the content (token-count bins); O(N + V) source rows."""

    def __init__(self, d: int, num_heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, num_heads, dropout=0.0, batch_first=True)
        self.ffn = FeedForward(d, 4 * d, d)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, rows: torch.Tensor, num_rows: int, weights: torch.Tensor | None = None) -> torch.Tensor:
        length = rows.shape[0]
        starts, ends = summary_bins(length, num_rows, rows.device, weights)
        width = (length + num_rows - 1) // num_rows + 1 if weights is None or num_rows >= length else int((ends - starts).max().item())
        indexes = starts[:, None] + torch.arange(width, device=rows.device)[None]
        valid = indexes < ends[:, None]
        bins = rows[indexes.clamp_max(length - 1)]
        weights = valid[..., None].to(rows.dtype)
        means = (bins * weights).sum(dim=1) / weights.sum(dim=1)
        normed = self.norm(bins)
        update, _ = self.attention(self.norm(means)[:, None], normed, normed, key_padding_mask=~valid, need_weights=False)
        summaries = means + self.residual_scale * update[:, 0]
        return summaries + self.residual_scale * self.ffn(self.norm(summaries))


class RowHead(nn.Module):
    """FFN x4 into the destination width, each row RMS-normalized and scaled by a learned scalar (initialized to the destination embedding RMS).
    `skip`: the input added to the FFN's output, which starts at zero (equal widths: the row starts as the normalized input).
    `linear_skip`: a bias-free linear map of the input added instead (unequal widths), zero until its owner fills it with the
    embedding map (ActivationContextModel._ensure_scales); the FFN's output starts at zero too, so the row starts as the mapped input."""

    def __init__(self, d_in: int, d_out: int, initial_scale: float = 1.0, skip: bool = False, linear_skip: bool = False) -> None:
        super().__init__()
        self.ffn = FeedForward(d_in, 4 * d_in, d_out)
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))
        self.skip = skip
        self.linear: nn.Linear | None = None
        if skip and linear_skip:
            raise ValueError("RowHead takes one skip: identity or linear")
        if skip:
            if d_in != d_out:
                raise ValueError("RowHead skip needs equal widths")
            nn.init.zeros_(self.ffn.down.weight); nn.init.zeros_(self.ffn.down.bias)   # identity at init: the row is the normalized input
        if linear_skip:
            with torch.random.fork_rng(devices=[]):                                      # zeroed anyway: the global stream draws as without it
                self.linear = nn.Linear(d_in, d_out, bias=False)
            nn.init.zeros_(self.linear.weight)                                          # the owner fills the map before the first encode
            nn.init.zeros_(self.ffn.down.weight); nn.init.zeros_(self.ffn.down.bias)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        out = self.ffn(rows)
        if self.skip:
            out = out + rows
        if self.linear is not None:
            out = out + self.linear(rows)
        rms = out.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        return out / rms * self.scale


@dataclass
class ACOutput:
    """What the encoder emits for one part: the rows in the reader's input space plus, for a deep model, one residual delta per
    intervention layer ([V, d_target], `layer_inputs`) that the reader adds to that layer's input at the row positions. Recursive
    results carry an empty map and their pre-adapter final states (`states`). A key/value-transfer model (`kv_transfer`) emits
    zero rows (inert placeholders in the reader) and, per reader layer, the keys and values of its V summary tokens."""
    input_embeds: torch.Tensor
    layer_inputs: dict[int, torch.Tensor] = field(default_factory=dict)
    states: torch.Tensor | None = None       # a recursive result's final V side states before the recursive adapter ([V, d_side]): the
                                             # parent's read-outs attend them as extra keys when `recursive_keys` is on
    kv_values: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)   # reader layer -> (keys, values), each [V, kv_heads, head_dim];
                                             # the keys are not rotated yet: the reader rotates them to where the part sits in its prompt
    kv_positions: torch.Tensor | None = None # [V] long: the summary tokens' positions in the encoder's sequence
    kv_length: int = 0                       # the length of the encoder's sequence (the positions the part takes up in the reader)
    kv_overwrite: bool = False               # overwrite mode: the rows are real reader positions (input_embeds: the marker) whose keys/values are replaced
    kv_bias: dict[int, torch.Tensor] = field(default_factory=dict)   # optional reader layer -> [V, kv_heads]: added to the entries' attention
                                             # scores (a compacted cache's log-weights β; harness/kv_transfer.py KV_BIAS)

    def _map(self, move: t.Callable[[torch.Tensor], torch.Tensor], index: t.Callable[[torch.Tensor], torch.Tensor]) -> "ACOutput":
        return ACOutput(move(self.input_embeds), {int(layer): move(rows) for layer, rows in self.layer_inputs.items()},
                        None if self.states is None else move(self.states),
                        {int(layer): (move(keys), move(values)) for layer, (keys, values) in self.kv_values.items()},
                        None if self.kv_positions is None else index(self.kv_positions), self.kv_length, self.kv_overwrite,
                        {int(layer): move(bias) for layer, bias in self.kv_bias.items()})

    def to(self, device: torch.device | str, dtype: torch.dtype) -> "ACOutput":
        return self._map(lambda tensor: tensor.to(device=device, dtype=dtype), lambda tensor: tensor.to(device=device))

    def detach_cpu(self, dtype: torch.dtype = torch.bfloat16) -> "ACOutput":
        return self._map(lambda tensor: tensor.detach().to(device="cpu", dtype=dtype).contiguous(), lambda tensor: tensor.detach().to(device="cpu"))

    @property
    def has_interventions(self) -> bool:
        return bool(self.layer_inputs)

    @property
    def tensors(self) -> list[torch.Tensor]:
        return [self.input_embeds, *self.layer_inputs.values(), *([self.states] if self.states is not None else []),
                *(tensor for pair in self.kv_values.values() for tensor in pair), *([self.kv_positions] if self.kv_positions is not None else []),
                *self.kv_bias.values()]

    @property
    def nbytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors)


class DeltaHead(nn.Module):
    """Bottleneck FFN (d_in -> r -> d_out, SiLU) into a reader layer's residual stream at the row positions. `down` is zero-initialized
    (weight and bias), so the delta is exactly 0 at init - a deep model starts bit-identical to the input-only model - and the head owns
    its magnitude from there (no normalization, no learned scale)."""

    def __init__(self, d_in: int, d_out: int, r: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_in, r)
        self.down = nn.Linear(r, d_out)
        nn.init.zeros_(self.down.weight); nn.init.zeros_(self.down.bias)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(rows.float())))


class PassageAttention(nn.Module):
    """The read-out in front of every delta head: each memory row's final side state queries the passage tokens' final side states
    through bottleneck projections (d -> r; `heads` attention heads of r / heads dims) and adds the projected result back:
    `rows + out(attn)`. `out` is zero-initialized, so at init the block is the identity on the rows (the head behind it reads the
    row's own state) and the attention path opens only as `out` leaves zero. One shared LayerNorm normalizes both inputs."""

    def __init__(self, d: int, r: int, heads: int | None = None) -> None:
        super().__init__()
        self.heads = heads if heads is not None else (4 if r % 4 == 0 and r >= 64 else 1)
        if r % self.heads:
            raise ValueError("the bottleneck width must be a multiple of the head count")
        self.norm = nn.LayerNorm(d)
        self.query = nn.Linear(d, r)
        self.key = nn.Linear(d, r)
        self.value = nn.Linear(d, r)
        self.out = nn.Linear(r, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, rows: torch.Tensor, passage: torch.Tensor) -> torch.Tensor:
        """rows [M, d] (queries), passage [N, d] (keys and values) -> [M, d]."""
        rows, passage = rows.float(), passage.float()
        if passage.shape[0] == 0:
            return rows
        normed = self.norm(passage)
        split = lambda tensor, n: tensor.view(n, self.heads, -1).transpose(0, 1)                   # [heads, n, r / heads]
        q, k, v = split(self.query(self.norm(rows)), rows.shape[0]), split(self.key(normed), passage.shape[0]), split(self.value(normed), passage.shape[0])
        # A batch dimension lets CUDA use memory-efficient attention for these FP32 readouts.
        mixed = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0).transpose(0, 1).reshape(rows.shape[0], -1)
        return rows + self.out(mixed)


def resolve_intervention_layers(num_layers: int, frequency: int, add_last: bool, full_attention_layers: t.Iterable[int] | None = None) -> list[int]:
    """
    Reader layers whose input receives a delta: `frequency` interior points floor(L*i/(frequency+1)), each snapped to a
    full-attention block within two indices (the earlier one on a tie), plus L-1 when asked; 0 excluded, sorted,
    deduplicated. Frequency 0 disables everything, the last-layer flag included. Frequency -1 = every full-attention
    layer of the reader (0 excluded; the last-layer flag and the snapping are moot).
    """
    if frequency == 0:
        return []                              # zero mode is unconditional, whatever the target description says
    if frequency < -1 or num_layers < 1:
        raise ValueError("intervention_frequency must be -1 (every full-attention layer), 0 or positive, and the reader must have layers")
    full = sorted(set(int(index) for index in (full_attention_layers or [])))
    if frequency == -1:
        if not full:
            raise ValueError("intervention_frequency -1 selects every full-attention layer, and the reader describes none")
        return [layer for layer in full if 0 < layer < num_layers]
    chosen = []
    for i in range(1, frequency + 1):
        layer = (num_layers * i) // (frequency + 1)
        if full:
            candidates = [index for index in full if abs(index - layer) <= 2]
            if candidates:
                layer = min(candidates, key=lambda index: (abs(index - layer), index))
        chosen.append(layer)
    if add_last:
        chosen.append(num_layers - 1)
    return sorted({layer for layer in chosen if 0 < layer < num_layers})


# --------------------------------------------------------------------------------------- cache and queue
class RowCache:
    """CPU cache of encoded rows keyed by row_cache_key, LRU by bytes."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = int(max_bytes)
        self.entries: "OrderedDict[str, ACOutput]" = OrderedDict()
        self.total_bytes = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def get(self, key: str) -> ACOutput | None:
        with self.lock:
            rows = self.entries.get(key)
            if rows is None:
                self.misses += 1
                return None
            self.entries.move_to_end(key)
            self.hits += 1
            return rows

    def put(self, key: str, rows: torch.Tensor | ACOutput) -> None:
        """Every tensor of the output (rows and deltas) is kept in bf16 on the CPU and counted."""
        if self.max_bytes <= 0:
            return
        output = (rows if isinstance(rows, ACOutput) else ACOutput(rows)).detach_cpu()
        size = output.nbytes
        with self.lock:
            if key in self.entries:
                self.total_bytes -= self.entries[key].nbytes
            self.entries[key] = output
            self.entries.move_to_end(key)
            self.total_bytes += size
            while self.total_bytes > self.max_bytes and self.entries:
                _, evicted = self.entries.popitem(last=False)
                self.total_bytes -= evicted.nbytes

    def clear(self) -> None:
        with self.lock:
            self.entries.clear()
            self.total_bytes = 0

    def __len__(self) -> int:
        return len(self.entries)


class EncodeRequest(t.NamedTuple):
    messages: list[dict]
    compression_ratio: float
    is_recursive: bool
    tools: list[dict] | None = None            # the definitions the compressed segment ran with (part["tools"]); templated on the side


@t.runtime_checkable
class SpecialInputHandler(t.Protocol):
    """Pluggable answers for chosen inputs, owned outside the main code (ActivationContextModel.special_input_handler).
    Called once per encode_batch with every request; returns one answer per request: an ACOutput that stands for the
    encoder's result for that part (never row-cached; it may carry gradients into `parameters()`), or None (the encoder
    runs). The ACOutput must have the shape the encoder would give: as many rows (placeholders, for key/value transfer) as
    `num_view_rows` of the part, and for key/value transfer `kv_values` at every reader layer, `kv_positions`, `kv_length`."""

    def __call__(self, requests: list[EncodeRequest]) -> list["ACOutput | None"]: ...

    def parameters(self) -> t.Iterable[torch.nn.Parameter]: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> t.Any: ...


def ac_execution_records(ac, examples):
    """Actual recursive encoder work and reader lengths, shared by tuning and partitioning."""
    counts, sequences = {}, {}
    def request_record(request):
        key = hashlib.sha256(canonical_json(request._asdict()).encode()).hexdigest()
        if key not in counts:
            children = [ac._child_request(part, request.compression_ratio) for part in direct_parts(request.messages)]
            keys = [request_record(child) for child in children]
            rows = [ac.num_view_rows(counts[key][0], child.compression_ratio) for key, child in zip(keys, children)]
            ids, _ = tokenize_with_parts(ac.side.tokenizer, request.messages, rows, ac.pad_id, tools=request.tools)
            length = len(ids) + ac.num_view_rows(len(ids), request.compression_ratio)
            counts[key] = (len(ids), length + sum(counts[child][1] for child in keys))
            sequences[key] = [length, *(value for child in keys for value in sequences[child])]
        return key
    records = []
    for example in examples:
        keys = [request_record(request) for request in example.part_requests]
        records.append({"item": example.item.item_id, "request_keys": keys,
            "encoder_tokens": sum(counts[key][1] for key in keys),
            "encoder_sequence_tokens": [value for key in keys for value in sequences[key]],
            "reader_tokens": len(example.student_ids), "target_tokens": len(example.target_ids),
            "penalty_tokens": len(example.penalty.student_ids) if example.penalty else 0,
            "weight": example.item.weight,
            "reader_identity": hashlib.sha256(canonical_json([list(example.student_ids), list(example.student_positions), list(example.target_ids)]).encode()).hexdigest(),
            "reference_identity": list(example.penalty.teacher_cache_key) if example.penalty else None})
    return records


def ac_execution_cost(record):
    # Encoder/reader have different widths. This is a length-based scheduling estimate,
    # not a throughput claim; hardware tuning measures complete updates separately.
    return max(1., record["encoder_tokens"] * .25 + record["reader_tokens"]
               + record["target_tokens"] * 2 + record["penalty_tokens"] * .25)


def ac_execution_costs(ac, examples):
    """Reuse CPU-prepared scheduling estimates; old prepared examples remain readable."""
    missing = [example for example in examples if getattr(example, "execution_cost", None) is None]
    for example, record in zip(missing, ac_execution_records(ac, missing), strict=True):
        example.execution_cost = ac_execution_cost(record)
    return [example.execution_cost for example in examples]


def bucket_training_order(order, costs, batch_size, window, seed):
    """Opt-in similar-cost batches; preserve every item and keep the partial tail last."""
    if not window:
        return order
    if window < batch_size or window % batch_size:
        raise ValueError("cost_bucket_examples must be a positive multiple of the logical batch size")
    batches, tail = [], []
    for start in range(0, len(order), window):
        local = sorted(order[start:start + window], key=lambda index: -costs[index])
        for offset in range(0, len(local), batch_size):
            batch = local[offset:offset + batch_size]
            if len(batch) == batch_size:
                batches.append(batch)
            else:
                tail = batch
    random.Random(seed).shuffle(batches)
    return [index for batch in batches for index in batch] + tail


def part_group_key(example) -> str:
    """What an example's compressed parts are (texts, ratios, tools): equal keys mean one encoding serves both examples."""
    return canonical_json([request._asdict() for request in example.part_requests])


def part_grouped_order(order, keys, group, seed):
    """Opt-in: consecutive runs of `group` examples share their compressed parts (equal `keys`), so one encoding can serve a
    run. The runs are cut from each key's examples in the order given and shuffled as wholes; every example is kept, and the
    examples left over (fewer than `group` per key) come last in the order given."""
    if not group:
        return order
    by_key: dict[str, list[int]] = {}
    for index in order:
        by_key.setdefault(keys[index], []).append(index)
    runs, rest = [], []
    for members in by_key.values():
        full = len(members) - len(members) % group
        runs.extend(members[start:start + group] for start in range(0, full, group))
        rest.extend(members[full:])
    random.Random(seed).shuffle(runs)
    return [index for run in runs for index in run] + rest


def checkpoint_preserved(path):
    """A preservation flag on either a checkpoint or its ancestor protects it."""
    path = Path(path).absolute()
    return any((parent / "PRESERVE").exists() for parent in (path, *path.parents))


def check_checkpoint_space(folders, parameters, optimizer):
    """Reserve a complete new pair and Adam state while keeping the old pair intact."""
    parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
    state_bytes = sum(value.numel() * value.element_size() for state in (optimizer.state.values() if optimizer else [])
                      for value in state.values() if isinstance(value, torch.Tensor))
    required = int((parameter_bytes + state_bytes) * 1.15) + 1024**3
    devices = set()
    for folder in folders:
        parent = Path(folder).parent
        while not parent.exists():
            parent = parent.parent
        device = parent.stat().st_dev
        if device in devices:
            continue
        devices.add(device)
        available = shutil.disk_usage(parent).free
        if available < required:
            raise OSError(f"Checkpoint needs {required / 1024**3:.1f} GiB free at {parent}; "
                          f"{available / 1024**3:.1f} GiB available. Previous recovery pair retained.")
    return required


def training_storage_preflight(output, *, quota_root, quota_limit_bytes, reserve_bytes=32 * 1024**3):
    """Check the entire quota domain on a compute node before placing large models."""
    import subprocess
    output, quota_root = Path(output).resolve(), Path(quota_root).resolve()
    if not output.is_relative_to(quota_root):
        raise ValueError("Training output is outside the supplied quota domain")
    output.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["du", "-sx", "--block-size=1", str(quota_root)],
                            check=True, capture_output=True, text=True, timeout=600)
    used = int(result.stdout.split()[0])
    filesystem_free = shutil.disk_usage(output).free
    quota_free = int(quota_limit_bytes) - used
    if min(filesystem_free, quota_free) < reserve_bytes:
        raise OSError(f"Training needs {reserve_bytes / 1024**3:.1f} GiB headroom; "
                      f"filesystem {filesystem_free / 1024**3:.1f}, quota {quota_free / 1024**3:.1f} GiB free")
    return {"time_unix": time.time(), "quota_root": str(quota_root), "quota_limit_bytes": quota_limit_bytes,
            "used_bytes": used, "quota_free_bytes": quota_free, "filesystem_free_bytes": filesystem_free,
            "required_headroom_bytes": reserve_bytes, "method": "whole quota-domain allocated bytes; conservative explicit site limit"}


@contextmanager
def deterministic_training(enabled: bool):
    """Scope PyTorch's process-wide kernel controls; CUDA workspace is set at process startup."""
    if not enabled:
        yield
        return
    if torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError("Deterministic CUDA training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                         "(or :16:8) before process startup")
    previous = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous[2:]


class DistributedTraining:
    """One existing trainer per GPU; synchronize once per global logical update.

    Local losses already use the global example count, so gradients are SUMmed.
    Globally unused parameters retain grad=None, including during plain-agent SFT.
    The caller launches with torchrun; model ownership and forward paths stay local.
    """
    def __init__(self, enabled: bool = False, bucket_bytes: int = 64 * 1024**2):
        self.rank, self.world_size, self.depth = 0, 1, 0
        self.device = torch.device("cpu")
        self.bucket_bytes = bucket_bytes
        self.owns_group = False
        self.parameters = []
        self.buckets = []
        if not enabled:
            if int(os.environ.get("WORLD_SIZE", "1")) > 1:
                raise ValueError("torchrun requires distributed=True; refusing concurrent checkpoint writers")
            return
        import torch.distributed as dist
        if not dist.is_initialized():
            if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
                raise ValueError("Launch distributed training with torchrun")
            if torch.cuda.is_available():
                torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(minutes=30))
            self.owns_group = True
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        if int(os.environ.get("LOCAL_WORLD_SIZE", str(self.world_size))) != self.world_size:
            raise ValueError("AC distributed training currently supports one GPU node")
        if dist.get_backend() == "nccl":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            self.device = torch.device("cuda", torch.cuda.current_device())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def is_main(self):
        return self.rank == 0

    @property
    def active(self):
        return self.world_size > 1

    def gather(self, value):
        if not self.active:
            return [value]
        import torch.distributed as dist
        result = [None] * self.world_size
        dist.all_gather_object(result, value)
        return result

    def any(self, value: bool) -> bool:
        if not self.active:
            return value
        import torch.distributed as dist
        flag = torch.tensor(int(value), device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    @contextmanager
    def local_work(self):
        """Finish or reject local preparation/backward together, before any update."""
        error = None
        self.depth += 1
        try:
            yield
        except Exception as caught:
            error = caught
        finally:
            self.depth -= 1
        if self.any(error is not None):
            errors = self.gather(error)
            raise next(value for value in errors if value is not None)

    def main(self, function, *args, **kwargs):
        """Run reporting/checkpoint I/O once; share results and exceptions."""
        if not self.active:
            return function(*args, **kwargs)
        import torch.distributed as dist
        result = [None, None]
        if self.is_main:
            self.depth += 1
            try:
                result[0] = function(*args, **kwargs)
            except Exception as error:
                result[1] = error
            finally:
                self.depth -= 1
        dist.broadcast_object_list(result, src=0)
        if result[1] is not None:
            raise result[1]
        return result[0]

    def reporter(self, target):
        """Keep root-only reporting failures on the same control flow as training."""
        if not self.active:
            return target
        if not self.main(lambda: target is not None):
            return None
        parallel = self
        class Reporter:
            def __getattr__(self, name):
                def report(*args, **kwargs):
                    def call():
                        return getattr(target, name)(*args, **kwargs)
                    if parallel.depth:
                        return call() if parallel.is_main else None
                    return parallel.main(call)
                return report
        return Reporter()

    def expired(self, deadline) -> bool:
        expired = deadline is not None and time.monotonic() >= deadline
        return expired if self.depth else self.any(expired)

    def initialize(self, parameters, identity):
        """Check the replica contract and copy the initial trainable weights once."""
        self.parameters = list(parameters)
        if self.parameters:
            self.device = self.parameters[0].device
        self.buckets = []
        if self.bucket_bytes < 1:
            raise ValueError("Gradient bucket size must be positive")
        contract = (identity, [(tuple(p.shape), str(p.dtype)) for p in self.parameters])
        if any(value != contract for value in self.gather(contract)):
            raise ValueError("Distributed replicas have different training data/configuration/parameters")
        bucket, size, dtype = [], 0, None
        for index, parameter in enumerate(self.parameters):
            nbytes = parameter.numel() * parameter.element_size()
            if bucket and (dtype != parameter.dtype or size + nbytes > self.bucket_bytes):
                self.buckets.append(bucket); bucket, size = [], 0
            bucket.append(index); size += nbytes; dtype = parameter.dtype
        if bucket:
            self.buckets.append(bucket)
        if self.active:
            import torch.distributed as dist
            with torch.no_grad():
                for bucket in self.buckets:
                    flat = torch.cat([self.parameters[i].detach().reshape(-1) for i in bucket])
                    dist.broadcast(flat, src=0)
                    offset = 0
                    for index in bucket:
                        parameter = self.parameters[index]; count = parameter.numel()
                        parameter.copy_(flat[offset:offset + count].view_as(parameter)); offset += count

    def identity(self, examples, config):
        """Bind order, content, targets and weights, without retaining serialized data."""
        if not self.active:
            return None
        digest = hashlib.sha256(canonical_json(config).encode())
        for example in examples:
            digest.update(canonical_json((example.item.item_id, example.item.weight,
                list(example.student_ids), list(example.student_positions), list(example.target_ids),
                [request._asdict() for request in example.part_requests],
                getattr(example, "execution_cost", None))).encode())
        return digest.hexdigest()

    def merge_step(self, records, totals, side_tokens, started):
        """Global accounting and slowest-rank wall time, including optimizer work."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        gathered = self.gather((records, totals, side_tokens, time.monotonic() - started))
        merged = {"loss": 0., "agreement": 0., "tokens": 0, "penalty": []}
        for _, values, _, _ in gathered:
            for key in ("loss", "agreement", "tokens"):
                merged[key] += values[key]
            merged["penalty"].extend(values["penalty"])
        records = sorted((record for rows, _, _, _ in gathered for record in rows), key=lambda row: row[0])
        elapsed = time.monotonic() - started
        if self.active:
            import torch.distributed as dist
            timing = torch.tensor(elapsed, device=self.device, dtype=torch.float64)
            dist.all_reduce(timing, op=dist.ReduceOp.MAX)
            elapsed = float(timing.item())
        return records, merged, sum(row[2] for row in gathered), elapsed

    def partition(self, indices, costs):
        """Greedy longest-first assignment without changing global batch membership."""
        if not self.active:
            return list(indices)
        loads = [0.] * self.world_size
        assignments = [[] for _ in loads]
        for index in sorted(indices, key=lambda i: (-costs[i], i)):
            rank = min(range(self.world_size), key=lambda r: (loads[r], len(assignments[r]), r))
            assignments[rank].append(index); loads[rank] += costs[index]
        chosen = set(assignments[self.rank])
        return [index for index in indices if index in chosen]

    def reduce_gradients(self):
        """Bucketed exact-dtype SUM; missing local gradients contribute zero."""
        if not self.active:
            return
        import torch.distributed as dist
        present = torch.tensor([p.grad is not None for p in self.parameters], device=self.device, dtype=torch.int32)
        dist.all_reduce(present, op=dist.ReduceOp.MAX)
        present = present.cpu().tolist()
        for bucket in self.buckets:
            active = [i for i in bucket if present[i]]
            if not active:
                continue
            first = self.parameters[active[0]]
            flat = torch.zeros(sum(self.parameters[i].numel() for i in active), device=first.device, dtype=first.dtype)
            offset = 0
            for index in active:
                parameter = self.parameters[index]; count = parameter.numel()
                if parameter.grad is not None:
                    flat[offset:offset + count].copy_(parameter.grad.reshape(-1))
                offset += count
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            offset = 0
            for index in active:
                parameter = self.parameters[index]; count = parameter.numel()
                if parameter.grad is None:
                    parameter.grad = torch.empty_like(parameter)
                parameter.grad.copy_(flat[offset:offset + count].view_as(parameter)); offset += count
        for parameter, used in zip(self.parameters, present):
            if not used:
                parameter.grad = None

    def checkpoint_state(self):
        states = self.gather({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None})
        return {"world_size": self.world_size, "rng_by_rank": states, "gradient_reduction": "sum", "global_batch_unchanged": True}

    def restore_rng(self, state):
        if state is None:
            return
        if state["world_size"] != self.world_size:
            raise ValueError("Exact distributed RNG resume requires the saved GPU count")
        local = state["rng_by_rank"][self.rank]
        torch.set_rng_state(local["cpu"])
        if local["cuda"] is not None:
            torch.cuda.set_rng_state(local["cuda"], self.device)

    def close(self):
        if self.owns_group:
            import torch.distributed as dist
            dist.destroy_process_group()
            self.owns_group = False
        self.world_size = 1


def validate_execution_policy(policy: dict) -> None:
    """An opt-in measured training policy; never changes the logical update size."""
    allowed = {"graph_examples", "reader_batch", "encode_batch_tokens", "reuse_encodings",
               "side_checkpoint_tokens", "target_checkpoint_tokens", "optimizer_moments"}
    if set(policy) - allowed:
        raise ValueError(f"Unknown AC execution settings: {sorted(set(policy) - allowed)}")
    for name in ("graph_examples", "reader_batch", "encode_batch_tokens"):
        if name in policy and (type(policy[name]) is not int or policy[name] < 1):
            raise ValueError(f"{name} must be a positive integer")
    for name in ("side_checkpoint_tokens", "target_checkpoint_tokens"):
        if name in policy and (type(policy[name]) is not int or policy[name] < 0):
            raise ValueError(f"{name} must be a nonnegative integer")
    if "reuse_encodings" in policy and type(policy["reuse_encodings"]) is not bool:
        raise ValueError("reuse_encodings must be a boolean")
    if policy.get("optimizer_moments", "gpu") not in ("cpu", "gpu"):
        raise ValueError("optimizer_moments must be cpu or gpu")


def execution_groups(examples: t.Sequence, policy: dict) -> list[list[int]]:
    """Bound live graphs inside this logical update; preserve its membership and order."""
    size = policy.get("graph_examples", 1)
    return [list(range(start, min(start + size, len(examples)))) for start in range(0, len(examples), size)]


def execution_batch_loss(trainer, ac_model, examples: t.Sequence, device, penalty_flags: list[bool], policy: dict) -> list[dict]:
    """Reuse complete differentiable encodings only until this group's single backward.

    Reader batching is independent of encoder batching. Penalties are sampled by the
    caller in logical item order, before grouping; every returned loss remains an item
    mean. The caller applies item weights and the logical-update denominator.
    """
    if len(penalty_flags) != len(examples):
        raise ValueError("One penalty decision is required per example")
    requests, indices, known = [], [], {}
    for example in examples:
        positions = []
        for request in example.part_requests:
            # Exact ratio, tools and recursive content are all included. No persistent
            # cache, detached tensors, or payload dictionaries cross a backward call.
            key = canonical_json(request._asdict()) if policy.get("reuse_encodings", False) else None
            if key is None or key not in known:
                index = len(requests)
                requests.append(request)
                if key is not None:
                    known[key] = index
            else:
                index = known[key]
            positions.append(index)
        indices.append(positions)
    previous_budget = ac_model.config.encode_batch_max_tokens
    try:
        ac_model.config.encode_batch_max_tokens = policy.get("encode_batch_tokens", previous_budget)
        outputs = ac_model.encode_batch(requests, with_layers=True) if requests else []
    finally:
        ac_model.config.encode_batch_max_tokens = previous_budget
    results = []
    size = policy.get("reader_batch", 1)
    for start in range(0, len(examples), size):
        stop = min(start + size, len(examples))
        encoded = [outputs[index] for positions in indices[start:stop] for index in positions]
        results.extend(trainer._batch_loss(ac_model, examples[start:stop], device,
            encoded_outputs=encoded, penalty_flags=penalty_flags[start:stop]))
    return results


class AdamMomentPlacement:
    """Park AdamW moments between updates without changing its step or clock tensors."""
    def __init__(self, optimizer):
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("Moment placement supports AdamW only")
        if any(group.get("amsgrad") or group.get("capturable") or group.get("differentiable") for group in optimizer.param_groups):
            raise ValueError("Moment placement requires ordinary non-AMSGrad AdamW")
        self.optimizer = optimizer
        self.handles = [optimizer.register_step_pre_hook(self._before), optimizer.register_step_post_hook(self._after)]

    def _move(self, to_cpu: bool) -> None:
        for parameter, state in self.optimizer.state.items():
            if not state:
                continue
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Unexpected AdamW state")
            destination = torch.device("cpu") if to_cpu else parameter.device
            for key in ("exp_avg", "exp_avg_sq"):
                value = state[key]
                if value.shape != parameter.shape:
                    raise ValueError("AdamW moment/parameter shape mismatch")
                if value.device != destination:
                    state[key] = value.to(device=destination, non_blocking=False)

    def park(self) -> None:
        self._move(True)

    def _before(self, optimizer, args, kwargs):
        self._move(False)

    def _after(self, optimizer, args, kwargs):
        self.park()

    def close(self) -> None:
        try:
            self.park()
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()


class EncodeQueue:
    """
    One worker thread that batches concurrent no-grad encodes: requests submitted from any thread are
    drained every `drain_seconds`, sorted by size and encoded together through `encode_batch`.
    """

    def __init__(self, encode_batch: t.Callable[[list[EncodeRequest]], list[torch.Tensor]], drain_seconds: float = 0.005) -> None:
        self.encode_batch = encode_batch
        self.drain_seconds = drain_seconds
        self.pending: "queue.Queue[tuple[EncodeRequest, Future]]" = queue.Queue()
        self.batches = 0
        self.requests = 0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="ac-encode-queue", daemon=True)
        self.thread.start()

    def submit(self, request: EncodeRequest) -> Future:
        future: Future = Future()
        self.pending.put((request, future))
        return future

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                first = self.pending.get(timeout=0.05)
            except queue.Empty:
                continue
            time.sleep(self.drain_seconds)
            items = [first]
            while True:
                try:
                    items.append(self.pending.get_nowait())
                except queue.Empty:
                    break
            items.sort(key=lambda item: len(canonical_json(item[0].messages)))
            self.batches += 1
            self.requests += len(items)
            try:
                results = self.encode_batch([request for request, _ in items])
                for (_, future), rows in zip(items, results):
                    future.set_result(rows)
            except Exception as error:                                                             # noqa: BLE001 - surfaced through the futures
                for _, future in items:
                    if not future.done():
                        future.set_exception(error)
