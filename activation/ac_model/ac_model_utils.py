"""
Building blocks of the activation-context model (ac_model.py): part-aware tokenization, the mixer's
blocked local attention, windowed pooling, the row heads, the CPU row cache and the encode queue.
"""
from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
import time
import typing as t
from copy import deepcopy
from dataclasses import dataclass, field
from collections import OrderedDict
from concurrent.futures import Future

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
    flat = [_training_template_message(tokenizer, message, reasoning_as_text=True)
            for message in flatten_with_sentinels(messages, parts="sentinel")]
    index = sum(1 for _ in direct_parts(messages))
    assert index == len(part_lengths), f"{index} parts in the messages, {len(part_lengths)} lengths given"
    if tools and not template_takes_tools(tokenizer):
        flat, tools = tools_in_system_text(flat, tools), None                     # the definitions ride in the system text instead
    if not any(message.get("role") == "user" and not str(message.get("content") or "").strip().startswith("<tool_response>") for message in flat):
        flat.insert(0, {"role": "user", "content": ""})                    # Qwen templates refuse a conversation without a user query (a mid-trajectory segment)
    text = tokenizer.apply_chat_template(flat, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False,
                                         **(chat_template_kwargs or {}))
    return encode_with_part_sentinels(tokenizer, text, list(part_lengths), pad_id)


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
    """Check every recursive side forward, including pooled content and summary rows."""
    children = [ac_model._child_request(part, request.compression_ratio) for part in direct_parts(request.messages)]
    lengths = [ac_model.part_view_rows(child.messages, child.compression_ratio, child.tools) for child in children]
    tokenizer = ac_model.side.tokenizer
    ids, _ = tokenize_with_parts(tokenizer, request.messages, lengths, tokenizer.pad_token_id or 0, tools=request.tools)
    content_length = len(ids)
    if ac_model.config.input_pooling_stride:
        window, stride = ac_model.config.input_pooling_window, ac_model.config.input_pooling_stride
        content_length = 1 if len(ids) <= window else -(-(len(ids) - window) // stride) + 1
    total = content_length + ac_model.num_view_rows(len(ids), request.compression_ratio)      # the summary rows (kv_transfer_full's reader run is longer, the side forward is not)
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


def mass_bins(mass: torch.Tensor, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordered bins by cumulative mass: with T = the total mass, bin j spans the tokens holding the mass points [j T / V, (j + 1) T / V].
    `starts[j]` = the number of tokens whose inclusive cumulative mass is <= j T / V; `ends[j]` = one more than the number whose inclusive
    cumulative mass is < (j + 1) T / V, clamped to N. With a uniform mass these are floor(j N / V) and ceil((j + 1) N / V), the token-count
    bins; with a positive mass every bin is nonempty (start < end), consecutive bins overlap by at most one token and their union is every token
    (the float64 cumulative sum decides the exact boundary tokens). O(V log N)."""
    if mass.dim() != 1 or mass.shape[0] == 0 or num_rows < 1:
        raise ValueError("mass bins need one positive mass per token and a positive row count")
    length = mass.shape[0]
    cumulative = mass.double().cumsum(0)
    total = cumulative[-1]
    ordinal = torch.arange(num_rows, device=mass.device, dtype=torch.float64)
    low, high = ordinal * total / num_rows, (ordinal + 1) * total / num_rows
    starts = torch.searchsorted(cumulative, low, right=True)                       # tokens with cumulative mass <= low
    ends = (torch.searchsorted(cumulative, high, right=False) + 1).clamp_max(length)   # tokens with cumulative mass < high, plus one
    return starts, ends


class AdaptiveSummaryPooling(nn.Module):
    """V content-derived mean queries, each attending its ordered adaptive bin; O(N + V) source rows.

    `surprisal` > 0 (pool_surprisal, the writer-side lever of round 8): the bins are drawn by cumulative mass 1 + surprisal x s_t / mean(s) instead of
    token count (s_t = the base model's no-context surprisal of content token t: a row covers an equal share of the passage's entropy plus the token-count
    prior; a bin is at most (1 + surprisal) x the uniform width + 2 tokens; rows per part unchanged), and the pooling attention logits gain the bias
    surprisal x b_h x s_t / mean(s) with b_h a zero-initialised per-head parameter (`surprisal_bias`, registered only when on). At 0 the module and its
    state dict are exactly the token-count pooling."""

    def __init__(self, d: int, num_heads: int, surprisal: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, num_heads, dropout=0.0, batch_first=True)
        self.ffn = FeedForward(d, 4 * d, d)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.surprisal = float(surprisal)
        self.surprisal_bias: nn.Parameter | None = nn.Parameter(torch.zeros(num_heads)) if self.surprisal > 0 else None

    def bins(self, length: int, num_rows: int, surprisal: torch.Tensor | None = None, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor | None]:
        """(starts, ends, the widest bin, the relative surprisal s / mean(s) or None): token-count bins, or mass bins when the module is on and a
        surprisal is given."""
        ordinal = torch.arange(num_rows, device=device if device is not None else (surprisal.device if surprisal is not None else None))
        if surprisal is None or self.surprisal_bias is None:
            starts = ordinal * length // num_rows
            ends = ((ordinal + 1) * length + num_rows - 1) // num_rows
            return starts, ends, (length + num_rows - 1) // num_rows + 1, None
        if surprisal.shape[0] != length:
            raise ValueError(f"one surprisal per content token: {surprisal.shape[0]} for {length}")
        relative = surprisal.float() / surprisal.float().mean().clamp_min(1e-6)          # scale-free: mean 1 over the part
        starts, ends = mass_bins(1.0 + self.surprisal * relative, num_rows)
        return starts, ends, int((ends - starts).max()), relative

    def forward(self, rows: torch.Tensor, num_rows: int, surprisal: torch.Tensor | None = None) -> torch.Tensor:
        # floor/ceil bins cover the input; V>N repeats nonempty bins, never padding.
        length = rows.shape[0]
        if length == 0 or num_rows < 1:
            raise ValueError("Summary pooling needs nonempty content and a positive row count")
        starts, ends, width, relative = self.bins(length, num_rows, surprisal, device=rows.device)
        indexes = starts[:, None] + torch.arange(width, device=rows.device)[None]
        valid = indexes < ends[:, None]
        bins = rows[indexes.clamp_max(length - 1)]
        weights = valid[..., None].to(rows.dtype)
        means = (bins * weights).sum(dim=1) / weights.sum(dim=1)
        normed = self.norm(bins)
        if relative is None:
            update, _ = self.attention(self.norm(means)[:, None], normed, normed,
                                       key_padding_mask=~valid, need_weights=False)
        else:
            # The surprisal bias on the logits: [V, heads, 1, width] -> (V x heads, 1, width), plus the padding as a float mask of the same kind.
            heads = self.attention.num_heads
            bias = self.surprisal * self.surprisal_bias[None, :, None, None] * relative[indexes.clamp_max(length - 1)][:, None, None, :]
            bias = bias.reshape(num_rows * heads, 1, width).to(normed.dtype)
            padding = torch.zeros(valid.shape, dtype=normed.dtype, device=rows.device).masked_fill(~valid, float("-inf"))
            update, _ = self.attention(self.norm(means)[:, None], normed, normed,
                                       key_padding_mask=padding, attn_mask=bias, need_weights=False)
        summaries = means + self.residual_scale * update[:, 0]
        return summaries + self.residual_scale * self.ffn(self.norm(summaries))


class RowHead(nn.Module):
    """FFN x4 into the destination width, each row RMS-normalized and scaled by a learned scalar (initialized to the destination embedding RMS)."""

    def __init__(self, d_in: int, d_out: int, initial_scale: float = 1.0, skip: bool = False) -> None:
        super().__init__()
        self.ffn = FeedForward(d_in, 4 * d_in, d_out)
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))
        self.skip = skip
        if skip:
            if d_in != d_out:
                raise ValueError("RowHead skip needs equal widths")
            nn.init.zeros_(self.ffn.down.weight); nn.init.zeros_(self.ffn.down.bias)   # identity at init: the row is the normalized input

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        out = self.ffn(rows)
        if self.skip:
            out = out + rows
        rms = out.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        return out / rms * self.scale


class RefineAdapter(nn.Module):
    """Iterative encoder (encoder_passes >= 2): a residual bottleneck d -> r -> d, its output projection zero-initialized, mapping the previous
    pass's final row states (unit rows) back into the side input space. Pass k >= 2 reads summaries_1 + refine(rows_{k-1}); at init the
    refinement is exactly 0, so a second pass starts as a plain re-read of the first pass's summaries (bit-identical rows)."""

    def __init__(self, d: int, r: int) -> None:
        super().__init__()
        self.down = nn.Linear(d, r)
        self.up = nn.Linear(r, d)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(rows)))


@dataclass
class ACOutput:
    """What the encoder emits for one part: rows in the target input space plus, for a deep model, per intervention layer either
    one residual delta ([V, d_target], `layer_inputs`), one pre-RoPE K/V delta pair ([V, d_kv] each, `kv_inputs`), or one K/V slot
    tuple (`kv_slots`: the side-computed K and V contributions [V, d_kv] each plus the layer's two 0-d keep factors for the reader's
    own K and V at the rows; the reader writes keep x own + contribution over the row slices; the K/V transfer targets use the same
    entry with contribution = takeover x (the side model's own K/V + correction) - `kv_transfer_full` over the whole passage run, its
    `input_embeds` then being the rows followed by zero fill up to the run length), learned scales included. A
    cross-attention model carries `cross_passage` instead: the side model's final states at every passage position ([N, d_side]),
    read by the reader's per-layer cross-attention blocks. Recursive results carry empty maps."""
    input_embeds: torch.Tensor
    layer_inputs: dict[int, torch.Tensor] = field(default_factory=dict)
    kv_inputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    kv_slots: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    cross_passage: torch.Tensor | None = None

    def to(self, device: torch.device | str, dtype: torch.dtype) -> "ACOutput":
        move = lambda tensor: tensor.to(device=device, dtype=dtype)
        return ACOutput(move(self.input_embeds),
                        {int(layer): move(rows) for layer, rows in self.layer_inputs.items()},
                        {int(layer): (move(dk), move(dv)) for layer, (dk, dv) in self.kv_inputs.items()},
                        {int(layer): tuple(move(tensor) for tensor in entry) for layer, entry in self.kv_slots.items()},
                        None if self.cross_passage is None else move(self.cross_passage))

    def detach_cpu(self, dtype: torch.dtype = torch.bfloat16) -> "ACOutput":
        cpu = lambda tensor: tensor.detach().to(device="cpu", dtype=dtype).contiguous()
        return ACOutput(cpu(self.input_embeds), {int(layer): cpu(rows) for layer, rows in self.layer_inputs.items()},
                        {int(layer): (cpu(dk), cpu(dv)) for layer, (dk, dv) in self.kv_inputs.items()},
                        {int(layer): tuple(cpu(tensor) for tensor in entry) for layer, entry in self.kv_slots.items()},
                        None if self.cross_passage is None else cpu(self.cross_passage))

    @property
    def has_interventions(self) -> bool:
        return bool(self.layer_inputs or self.kv_inputs or self.kv_slots or self.cross_passage is not None)

    @property
    def tensors(self) -> list[torch.Tensor]:
        """The per-row payload tensors (the slot keep factors are per-model scalars, not payload)."""
        return [self.input_embeds, *self.layer_inputs.values(), *(tensor for pair in self.kv_inputs.values() for tensor in pair),
                *(tensor for entry in self.kv_slots.values() for tensor in entry[:2]), *([self.cross_passage] if self.cross_passage is not None else [])]

    @property
    def nbytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors)


HEAD_STYLES = ("scaled", "lora")


class DeltaHead(nn.Module):
    """Bottleneck FFN (d_in -> r -> d_out, SiLU) into a reader layer's residual stream, in one of two styles.

    "scaled": every row RMS-normalized and scaled by `rms x relative`: `rms` is the layer's residual RMS measured by
    `ActivationContextModel.calibrate_interventions` (a buffer), `relative` the learned scalar in units of that RMS (initialized
    to the scale fraction), so one Adam step moves every layer's scale by the same share of its residual whatever the layer's magnitude.
    "lora": `down(silu(up(z)))` with `down` zero-initialized (weight and bias): the delta is exactly 0 at init and the head owns its
    magnitude (no unit-RMS normalization, no learned scale). `relative` and `rms` stay registered (the checkpoint keys are the same)
    but are inert: `relative` never requires gradient and `rms` only feeds the diagnostics."""

    def __init__(self, d_in: int, d_out: int, r: int, relative: float = 0.0, style: str = "scaled") -> None:
        super().__init__()
        if style not in HEAD_STYLES:
            raise ValueError(f"delta head style must be one of {HEAD_STYLES}")
        self.style = style
        self.up = nn.Linear(d_in, r)
        self.down = nn.Linear(r, d_out)
        self.relative = nn.Parameter(torch.tensor(float(relative)), requires_grad=style == "scaled")
        self.register_buffer("rms", torch.tensor(0.0))
        if style == "lora":
            nn.init.zeros_(self.down.weight); nn.init.zeros_(self.down.bias)   # after the draws: the up projection's init is the scaled style's

    @property
    def scale(self) -> torch.Tensor:
        """The applied delta RMS in the scaled style: rms x relative (an inert anchor in the lora style)."""
        return self.rms * self.relative

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        out = self.down(F.silu(self.up(rows.float())))
        if self.style == "lora":
            return out
        return unit_rows(out) * self.scale

    def scale_parameters(self) -> list[nn.Parameter]:
        """The learned scale(s): the scaled style's `relative`; nothing in the lora style (the optimizer gets no scale group)."""
        return [self.relative] if self.style == "scaled" else []


class KVDeltaHead(nn.Module):
    """Bottleneck FFN (d_in -> r -> 2 d_kv, SiLU) into a full-attention reader layer's keys and values at the row positions.
    "scaled": the two halves are RMS-normalized per row and scaled by their own learned scalars, which calibration sets to a share of the
    reader's own pre-RoPE K RMS and V RMS at that layer (the same construction as DeltaHead: random bottleneck, unit rows, calibrated
    scale, so a fresh K/V head starts as a small random perturbation of the row's key and value). "lora": both halves are the raw
    bottleneck output with `down` zero-initialized (exactly 0 at init, the head owns its magnitude); the scales stay registered but inert."""

    def __init__(self, d_in: int, d_kv: int, r: int, relative: float = 0.0, style: str = "scaled") -> None:
        super().__init__()
        if style not in HEAD_STYLES:
            raise ValueError(f"delta head style must be one of {HEAD_STYLES}")
        self.style = style
        self.d_kv = int(d_kv)
        self.up = nn.Linear(d_in, r)
        self.down = nn.Linear(r, 2 * self.d_kv)
        self.relative_k = nn.Parameter(torch.tensor(float(relative)), requires_grad=style == "scaled")
        self.relative_v = nn.Parameter(torch.tensor(float(relative)), requires_grad=style == "scaled")
        self.register_buffer("rms_k", torch.tensor(0.0))
        self.register_buffer("rms_v", torch.tensor(0.0))
        if style == "lora":
            nn.init.zeros_(self.down.weight); nn.init.zeros_(self.down.bias)

    @property
    def scale_k(self) -> torch.Tensor:
        return self.rms_k * self.relative_k

    @property
    def scale_v(self) -> torch.Tensor:
        return self.rms_v * self.relative_v

    def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        both = self.down(F.silu(self.up(rows.float())))
        if self.style == "lora":
            return both[:, :self.d_kv], both[:, self.d_kv:]
        return unit_rows(both[:, :self.d_kv]) * self.scale_k, unit_rows(both[:, self.d_kv:]) * self.scale_v

    def scale_parameters(self) -> list[nn.Parameter]:
        return [self.relative_k, self.relative_v] if self.style == "scaled" else []


class PassageAttention(nn.Module):
    """One head-input module for candidate A2: each memory row's state queries the passage tokens' states (keys and values) through
    bottleneck projections (d -> r; `heads` attention heads of r / heads dims) and adds the projected result back: `rows + out(attn)`.
    `out` is zero-initialized, so at init the module is the identity on the rows (the delta head behind it reads exactly what it
    read before: the summary readout at depth "final", the layer-matched readout at depth "matched") and the attention path opens
    only as `out` leaves zero. Both inputs pass one shared LayerNorm before the projections.

    The same block serves the reader-side cross-attention ceiling (`intervention_target = "cross_attention"`): `d_passage` gives the
    passage its own width and LayerNorm when it differs from the query width, `residual=False` returns the projected attention alone
    (the delta the reader adds at its target positions; exactly 0 at init), and `forward(..., mask)` takes a [M, N] boolean mask
    (True = the query may read that passage token) for the causal visibility of the passage parts."""

    def __init__(self, d: int, r: int, heads: int | None = None, d_passage: int | None = None, residual: bool = True) -> None:
        super().__init__()
        self.heads = heads if heads is not None else (4 if r % 4 == 0 and r >= 64 else 1)
        if r % self.heads:
            raise ValueError("the bottleneck width must be a multiple of the head count")
        self.residual = residual
        d_passage = d if d_passage is None else int(d_passage)
        self.norm = nn.LayerNorm(d)
        self.passage_norm = nn.LayerNorm(d_passage) if d_passage != d else None      # registered only when the widths differ (the A2 keys stay as they were)
        self.query = nn.Linear(d, r)
        self.key = nn.Linear(d_passage, r)
        self.value = nn.Linear(d_passage, r)
        self.out = nn.Linear(r, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, rows: torch.Tensor, passage: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """rows [M, d] (queries), passage [N, d_passage] (keys and values), optional mask [M, N] (True = readable) -> [M, d]."""
        rows, passage = rows.float(), passage.float()
        if passage.shape[0] == 0:
            return rows if self.residual else torch.zeros_like(rows)
        normed = lambda: (self.norm if self.passage_norm is None else self.passage_norm)(passage)    # applied once per consumer: the A2 op sequence (and its gradients) exactly as before
        split = lambda tensor, n: tensor.view(n, self.heads, -1).transpose(0, 1)                   # [heads, n, r / heads]
        q, k, v = split(self.query(self.norm(rows)), rows.shape[0]), split(self.key(normed()), passage.shape[0]), split(self.value(normed()), passage.shape[0])
        mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else mask[None]).transpose(0, 1).reshape(rows.shape[0], -1)
        projected = self.out(mixed)
        return rows + projected if self.residual else projected


class KVSlotHead(nn.Module):
    """The K/V slot head of `intervention_target = "kv_slots"`: a bottleneck FFN (d_in -> r -> 2 d_kv, SiLU) whose two halves are the
    side-computed contributions to a full-attention reader layer's pre-RoPE keys and values at the row positions, and two learned
    takeover scalars (one per half, 0 at init). The reader REPLACES the row slices with `keep x own + contribution`, keep = 1 - takeover.

    Init: `down` is zero-initialized and the takeovers are 0, so the written K/V are exactly the reader's own (bit-identical to no
    intervention) - the one replacement that starts meaningful without any extra state: a pure side-computed replacement would have
    to reproduce the reader's own K/V (a function of the reader's forward) before it could help, and a random one starts by wrecking
    the rows' keys (the silent-phase failure of the scaled heads). From there the head owns its magnitude (no unit-RMS normalisation,
    no scale on the content) and the takeover lets the layer discard the reader's own K/V at the rows entirely (takeover 1 = the slot
    is purely side-computed: what an engine injects as a per-request K/V prefix); an additive head cannot cancel a value it never sees.
    The two takeovers are the head's `scale_parameters()`: they train in the optimizer's scale group at `learning_rate_delta_scales`
    (a scalar under Adam moves by about its rate per step; at the encoder rate, 1e-4, the sum over an arm's schedule would bound the
    takeover near 0.05 and the slots could never leave the reader's-own regime), like the scaled heads' relative scales and unlike the
    lora-style heads, which have none. `rms_k` / `rms_v` are calibration anchors for the diagnostics only (the forward never reads them)."""

    def __init__(self, d_in: int, d_kv: int, r: int) -> None:
        super().__init__()
        self.d_kv = int(d_kv)
        self.up = nn.Linear(d_in, r)
        self.down = nn.Linear(r, 2 * self.d_kv)
        self.takeover_k = nn.Parameter(torch.tensor(0.0))
        self.takeover_v = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("rms_k", torch.tensor(0.0))
        self.register_buffer("rms_v", torch.tensor(0.0))
        nn.init.zeros_(self.down.weight); nn.init.zeros_(self.down.bias)

    def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[M, d_in] -> (K contribution [M, d_kv], V contribution [M, d_kv])."""
        both = self.down(F.silu(self.up(rows.float())))
        return both[:, :self.d_kv], both[:, self.d_kv:]

    def keep(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The factors on the reader's own K and V at the rows: 1 - takeover (0-d tensors, graph-attached)."""
        return 1.0 - self.takeover_k, 1.0 - self.takeover_v

    def scale_parameters(self) -> list[nn.Parameter]:
        """The takeovers: the scale group's members (the content path has no scale, like a lora-style head)."""
        return [self.takeover_k, self.takeover_v]


class KVTransferHead(nn.Module):
    """The per-layer module of `intervention_target = "kv_transfer"` / `"kv_transfer_full"`: the reader's row positions' pre-RoPE K and V at a
    full-attention layer are replaced by the SIDE MODEL'S OWN K and V at the same layer (captured after k_norm / v_proj during the side pass,
    at the same positions), mixed with the reader's own by a learned takeover t per half:
        K[rows] = (1 - t_k) x K_reader[rows] + t_k x (K_side[rows] + corr_k(side state))
    Side and reader share the base model, so the side K/V are in the format the reader's attention heads read natively: the replacement is
    meaningful at init (t = `takeover_init`, 1.0 = a full replacement; NOT an identity) - unlike a learned slot content, which starts at zero
    because the reader's heads do not read a fresh representation. `corr_*` is an optional zero-initialized lora-style correction on the side
    state (`correction`; `down` = 0, so the transferred K/V are exactly the side's at init). The takeovers are the head's `scale_parameters()`
    (the scale group at `learning_rate_delta_scales`); `rms_k` / `rms_v` are calibration anchors for the diagnostics only. The reader-side
    operation and the engine mapping are those of `kv_slots`: a per-request K/V write into the row positions' cache entries."""

    def __init__(self, d_in: int, d_kv: int, r: int, takeover_init: float = 1.0, correction: bool = False) -> None:
        super().__init__()
        self.d_kv = int(d_kv)
        self.correction = bool(correction)
        if self.correction:
            self.up = nn.Linear(d_in, r)
            self.down = nn.Linear(r, 2 * self.d_kv)
            nn.init.zeros_(self.down.weight); nn.init.zeros_(self.down.bias)
        self.takeover_k = nn.Parameter(torch.tensor(float(takeover_init)))
        self.takeover_v = nn.Parameter(torch.tensor(float(takeover_init)))
        self.register_buffer("rms_k", torch.tensor(0.0))
        self.register_buffer("rms_v", torch.tensor(0.0))

    def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """[M, d_in] side states -> (K correction [M, d_kv], V correction [M, d_kv]); None without the correction head."""
        if not self.correction:
            return None
        both = self.down(F.silu(self.up(rows.float())))
        return both[:, :self.d_kv], both[:, self.d_kv:]

    def keep(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The factors on the reader's own K and V at the rows: 1 - takeover (0-d tensors, graph-attached)."""
        return 1.0 - self.takeover_k, 1.0 - self.takeover_v

    def scale_parameters(self) -> list[nn.Parameter]:
        return [self.takeover_k, self.takeover_v]


def resolve_intervention_layers(num_layers: int, frequency: int, add_last: bool, prefer_attention: bool,
                                full_attention_layers: t.Iterable[int] | None = None) -> list[int]:
    """
    Reader layers whose input receives a delta: `frequency` interior points floor(L*i/(frequency+1)), each snapped to a
    full-attention block within two indices when asked (the earlier one on a tie), plus L-1 when asked; 0 excluded,
    sorted, deduplicated. Frequency 0 disables everything, the last-layer flag included. Frequency -1 = every full-attention
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
        if prefer_attention and full:
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
