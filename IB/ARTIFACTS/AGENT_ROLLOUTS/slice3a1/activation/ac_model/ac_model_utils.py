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
from collections import OrderedDict
from concurrent.futures import Future

import torch
import torch.nn as nn
import torch.nn.functional as F

if t.TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

AC_PART_TYPE = "activation_context"
PART_SENTINEL = "⁣ACPART{index}⁣"     # rendered in place of a part by the chat template, split out before tokenizing


# --------------------------------------------------------------------------------------- parts and keys
def normalize_messages(messages: list[dict] | str) -> list[dict]:
    """A plain string becomes one user message."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return list(messages)


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


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def row_cache_key(messages: list[dict], compression_ratio: float, is_recursive: bool, version: int) -> str:
    """sha256 of the canonical part: the same messages, ratio, kind and model version hit the same rows."""
    payload = canonical_json({"messages": messages, "ratio": round(float(compression_ratio), 8), "recursive": bool(is_recursive), "version": int(version)})
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
    flat: list[dict] = []
    index = 0
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            pieces = []
            for part in content:
                if is_ac_part(part):
                    pieces.append(PART_SENTINEL.format(index=index))
                    index += 1
                elif isinstance(part, dict):
                    pieces.append(part.get("text", ""))
                else:
                    pieces.append(str(part))
            message["content"] = "".join(pieces)
        flat.append(message)
    assert index == len(part_lengths), f"{index} parts in the messages, {len(part_lengths)} lengths given"
    if not any(message.get("role") == "user" and not str(message.get("content") or "").strip().startswith("<tool_response>") for message in flat):
        flat.insert(0, {"role": "user", "content": ""})                    # Qwen templates refuse a conversation without a user query (a mid-trajectory segment)
    text = tokenizer.apply_chat_template(flat, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False,
                                         **(chat_template_kwargs or {}))
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


class AdaptiveSummaryPooling(nn.Module):
    """V content-derived mean queries, each attending its ordered adaptive bin; O(N + V) source rows."""

    def __init__(self, d: int, num_heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, num_heads, dropout=0.0, batch_first=True)
        self.ffn = FeedForward(d, 4 * d, d)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, rows: torch.Tensor, num_rows: int) -> torch.Tensor:
        # floor/ceil bins cover the input; V>N repeats nonempty bins, never padding.
        length = rows.shape[0]
        if length == 0 or num_rows < 1:
            raise ValueError("Summary pooling needs nonempty content and a positive row count")
        ordinal = torch.arange(num_rows, device=rows.device)
        starts = ordinal * length // num_rows
        ends = ((ordinal + 1) * length + num_rows - 1) // num_rows
        width = (length + num_rows - 1) // num_rows + 1
        indexes = starts[:, None] + torch.arange(width, device=rows.device)[None]
        valid = indexes < ends[:, None]
        bins = rows[indexes.clamp_max(length - 1)]
        weights = valid[..., None].to(rows.dtype)
        means = (bins * weights).sum(dim=1) / weights.sum(dim=1)
        normed = self.norm(bins)
        update, _ = self.attention(self.norm(means)[:, None], normed, normed,
                                   key_padding_mask=~valid, need_weights=False)
        summaries = means + self.residual_scale * update[:, 0]
        return summaries + self.residual_scale * self.ffn(self.norm(summaries))


class RowHead(nn.Module):
    """FFN x4 into the destination width, each row RMS-normalized and scaled by a learned scalar (initialized to the destination embedding RMS)."""

    def __init__(self, d_in: int, d_out: int, initial_scale: float = 1.0) -> None:
        super().__init__()
        self.ffn = FeedForward(d_in, 4 * d_in, d_out)
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        out = self.ffn(rows)
        rms = out.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        return out / rms * self.scale


# --------------------------------------------------------------------------------------- cache and queue
class RowCache:
    """CPU cache of encoded rows keyed by row_cache_key, LRU by bytes."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = int(max_bytes)
        self.entries: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.total_bytes = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def get(self, key: str) -> torch.Tensor | None:
        with self.lock:
            rows = self.entries.get(key)
            if rows is None:
                self.misses += 1
                return None
            self.entries.move_to_end(key)
            self.hits += 1
            return rows

    def put(self, key: str, rows: torch.Tensor) -> None:
        if self.max_bytes <= 0:
            return
        rows = rows.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        size = rows.numel() * rows.element_size()
        with self.lock:
            if key in self.entries:
                self.total_bytes -= self.entries[key].numel() * self.entries[key].element_size()
            self.entries[key] = rows
            self.entries.move_to_end(key)
            self.total_bytes += size
            while self.total_bytes > self.max_bytes and self.entries:
                _, evicted = self.entries.popitem(last=False)
                self.total_bytes -= evicted.numel() * evicted.element_size()

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
