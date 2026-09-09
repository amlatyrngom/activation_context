"""
The activation-context (AC) model: a side language model (Qwen3.5-0.8B for development, with a
rank-128 LoRA) that reads messages and emits V "view rows" in the target model's input space,
V = clamp(ceil(side tokens x compression ratio), min_view_rows, max_view_rows). The target model reads
those rows in place of the messages (an `activation_context` content part).

Input shape (messages in our dialect; a plain string is one user message):

    [{"role": "user", "content": [
        {"type": "text", "text": "..."},
        {"type": "activation_context", "ac_name": "<this model>", "compression_target": 1/16, "messages": [...]},
        {"type": "text", "text": "..."}]}, ...]

Forward of one part: its activation_context children first (recursively, cached), through the
recursive adapter into the side input space; `tokenize_with_parts` renders the part under the side
chat template with a pad-id placeholder run per child; side input embeddings with the child rows
written over the placeholders + sinusoidal positions -> `mixer_layers` blocked-local-attention layers
-> windowed attention pooling (window 16, stride 4) -> V marker rows appended (marker kind top-level /
recursive + marker index) -> the causal side decoder with its LoRA -> the hidden states at the marker
positions -> `target_head` (into the target width) or, for a recursive child, returned for the
parent's `recursive_adapter`.

Modes: rollout (no grad, CPU row cache keyed by the part's content, an encode queue that batches
concurrent requests) and training (grad, no cache, the caller batches). Trainable AC modules are
fp32; the bases stay bf16. Checkpoints: `AC_MODELS/<name>/round_NNN` (+ `latest`) holding
`ac_modules.pt` and the side LoRA folder; `version` counts the saves and keys the cache.
"""
from __future__ import annotations

import dataclasses
import math
import shutil
import time
import typing as t
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from ..common.data_syncing import resolve_path
from ..harness.hf_utils import SOURCE_DEVICE, TARGET_DEVICE
from .ac_model_utils import (
    AC_PART_TYPE,
    BlockedLocalAttentionLayer,
    EncodeQueue,
    EncodeRequest,
    RowCache,
    RowHead,
    WindowedPooling,
    direct_parts,
    normalize_messages,
    row_cache_key,
    sinusoidal_positions,
    tokenize_with_parts,
)

if t.TYPE_CHECKING:
    from ..harness import HarnessRuntime

CHECKPOINT_FOLDER = "AC_MODELS"
MODULES_FILE = "ac_modules.pt"
SIDE_LORA_FOLDER = "side_lora"
ROLLOUT, TRAINING = "rollout", "training"


@dataclass
class ActivationContextModelConfig:
    ac_model_name: str                       # identifies this model; parts name it in `ac_name`
    base_side_model_name: str                # the side base (a loaded model)
    base_side_model_lora_name: str           # its adapter, registered by ModuleManager.register_ac_model when new
    target_model_name: str                   # whose input space the rows land in
    target_model_lora_name: str              # the target adapter trained alongside (ActivationContextTrainer)
    default_compression_ratio: float = 1.0 / 16.0
    input_pooling_stride: int = 4
    input_pooling_window: int = 16
    mixer_layers: int = 2
    mixer_window: int = 256                  # block size of the blocked local attention (each row sees 256-512 rows per side)
    mixer_heads: int = 8
    min_view_rows: int = 4
    max_view_rows: int = 4096
    side_lora_rank: int = 128
    encode_batch_max_tokens: int = 65_536    # side tokens per padded encode batch (B x longest)
    side_gradient_checkpointing: bool = True # on the side base in training mode
    checkpoint_path: str | None = None       # loaded by register_ac_model when given


@dataclass
class ActivationContextModelStats:
    encodes: int = 0                         # parts encoded through the network (children included)
    cache_hits: int = 0
    cache_misses: int = 0
    side_tokens: int = 0
    view_rows: int = 0
    encode_batches: int = 0
    encode_seconds: list[float] = field(default_factory=list)
    queue_batches: int = 0
    queue_requests: int = 0
    phase_seconds: dict[str, float] = field(default_factory=dict)   # per encode phase, only while `profile_phases` is on

    def summarize(self) -> dict:
        return {
            "encodes": self.encodes, "cache_hits": self.cache_hits, "cache_misses": self.cache_misses,
            "side_tokens": self.side_tokens, "view_rows": self.view_rows, "encode_batches": self.encode_batches,
            "encode_seconds_total": round(sum(self.encode_seconds), 3),
            "encode_seconds_max": round(max(self.encode_seconds), 3) if self.encode_seconds else 0.0,
            "queue_batches": self.queue_batches, "queue_requests": self.queue_requests,
            **({"phase_seconds": {key: round(value, 3) for key, value in self.phase_seconds.items()}} if self.phase_seconds else {}),
        }


class ActivationContextModules(nn.Module):
    """The trainable AC modules besides the side LoRA (fp32)."""

    def __init__(self, config: ActivationContextModelConfig, d_side: int, d_target: int):
        super().__init__()
        self.mixer = nn.ModuleList([BlockedLocalAttentionLayer(d_side, config.mixer_heads, config.mixer_window) for _ in range(config.mixer_layers)])
        self.pooling = WindowedPooling(d_side, config.mixer_heads, config.input_pooling_window, config.input_pooling_stride)
        self.marker_kind = nn.Embedding(2, d_side)                                             # 0 top-level, 1 recursive
        self.marker_index = nn.Embedding(config.max_view_rows, d_side)
        self.recursive_adapter = RowHead(d_side, d_side)
        self.target_head = RowHead(d_side, d_target)
        self.register_buffer("scales_initialized", torch.tensor(False))
        nn.init.normal_(self.marker_kind.weight, std=0.02)
        nn.init.normal_(self.marker_index.weight, std=0.02)


class ActivationContextModel:
    def __init__(self, harness: "HarnessRuntime", config: ActivationContextModelConfig):
        self.harness = harness
        self.config = config
        self.name = config.ac_model_name
        self.side = harness.loaded_models[config.base_side_model_name]
        self.target = harness.loaded_models[config.target_model_name]
        self.d_side = self.side.model_config.model_description.d_model
        self.d_target = self.target.model_config.model_description.d_model
        self.modules = ActivationContextModules(config, self.d_side, self.d_target)
        self.version = 0
        self.mode = ROLLOUT
        self.cache = RowCache(harness.harness_config.ac_row_cache_bytes)
        self.queue: EncodeQueue | None = None
        self.stats = ActivationContextModelStats()
        self.profile_phases = False              # per-phase encode timings into stats.phase_seconds (synchronizes the device)
        self.saved_rounds = 0
        tokenizer = self.side.tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    # ------------------------------------------------------------------------------------------ placement and modes
    def num_view_rows(self, num_side_tokens: int, compression_ratio: float | None = None) -> int:
        ratio = self.config.default_compression_ratio if compression_ratio is None else compression_ratio
        return int(min(self.config.max_view_rows, max(self.config.min_view_rows, math.ceil(num_side_tokens * ratio))))

    def part_view_rows(self, messages: list[dict], compression_ratio: float | None = None) -> int:
        """V of a part before encoding it: its side tokenization (children's rows counted recursively) times the ratio."""
        ratio = self.config.default_compression_ratio if compression_ratio is None else float(compression_ratio)
        child_lengths = [self.part_view_rows(*self._child_request(part, ratio)[:2]) for part in direct_parts(messages)]
        ids, _ = tokenize_with_parts(self.side.tokenizer, messages, child_lengths, self.pad_id)
        return self.num_view_rows(len(ids), ratio)

    def set_mode(self, mode: str) -> None:
        """`training`: gradients, no cache, no queue (the caller batches); `rollout`: no grad, cache, queue."""
        assert mode in (ROLLOUT, TRAINING), mode
        if mode == self.mode:
            return
        if self.queue is not None:
            self.queue.stop()
            self.queue = None
        self.mode = mode
        self.modules.train(mode == TRAINING)

    @property
    def device(self) -> torch.device:
        return next(self.modules.parameters()).device

    def prepare(self, device: str = TARGET_DEVICE) -> torch.device:
        """Side base with its adapter on the device, the AC modules next to it, the head scales initialized."""
        peft_model = self.harness.module_manager.ensure_lora(self.config.base_side_model_lora_name)
        device_obj = next(peft_model.parameters()).device
        if self.device != device_obj:
            self.modules.to(device_obj)
        self._ensure_scales()
        return device_obj

    def release(self) -> None:
        """Modules to host RAM (the bases are the harness's to move); the queue stopped."""
        if self.queue is not None:
            self.queue.stop()
            self.queue = None
        self.modules.to(SOURCE_DEVICE)

    def _ensure_scales(self) -> None:
        """Head scales start at the RMS of the destination embeddings, so rows land at the scale the decoders expect."""
        if bool(self.modules.scales_initialized):
            return
        with torch.no_grad():
            side_rms = self._embedding_rms(self.side)
            target_rms = self._embedding_rms(self.target)
            self.modules.recursive_adapter.scale.fill_(side_rms)
            self.modules.target_head.scale.fill_(target_rms)
            self.modules.scales_initialized.fill_(True)
        print(f"{self.name} - Head scales initialized: side {side_rms:.4f}, target {target_rms:.4f}")

    @staticmethod
    def _embedding_rms(loaded_model) -> float:
        if loaded_model.model is not None:
            weight = loaded_model.model.get_input_embeddings().weight
        else:
            loaded_model.embedding_layer_to_device(SOURCE_DEVICE)
            weight = loaded_model.embedding_layer.weight
        sample = weight[: min(weight.shape[0], 65_536)].float()
        return float(sample.pow(2).mean(dim=-1).sqrt().mean())

    def parameters(self) -> list[nn.Parameter]:
        """The AC modules' parameters (the side LoRA's come from the module manager)."""
        return list(self.modules.parameters())

    def trainable_parameters(self) -> list[nn.Parameter]:
        return self.parameters() + self.harness.module_manager.lora_parameters(self.config.base_side_model_lora_name)

    # ------------------------------------------------------------------------------------------ encoding
    def encode(self, messages: list[dict] | str, compression_ratio: float | None = None, is_recursive: bool = False) -> torch.Tensor:
        """Rows [V, d_target] of one part (or [V, d_side] after the recursive adapter when is_recursive)."""
        ratio = self.config.default_compression_ratio if compression_ratio is None else float(compression_ratio)
        return self.encode_batch([EncodeRequest(normalize_messages(messages), ratio, is_recursive)])[0]

    def encode_async(self, messages: list[dict] | str, compression_ratio: float | None = None, is_recursive: bool = False) -> Future:
        """Rollout mode: the rows through the encode queue (concurrent callers share batches)."""
        assert self.mode == ROLLOUT, "the encode queue serves rollout mode only"
        if self.queue is None:
            self.queue = EncodeQueue(self.encode_batch)
        ratio = self.config.default_compression_ratio if compression_ratio is None else float(compression_ratio)
        return self.queue.submit(EncodeRequest(normalize_messages(messages), ratio, is_recursive))

    def encode_batch(self, requests: list[EncodeRequest]) -> list[torch.Tensor]:
        """The rows of every request, cache first in rollout mode, the rest encoded together (children first)."""
        if not requests:
            return []
        results: list[torch.Tensor | None] = [None] * len(requests)
        pending: list[int] = []
        for index, request in enumerate(requests):
            cached = self.cache.get(row_cache_key(request.messages, request.compression_ratio, request.is_recursive, self.version)) if self.mode == ROLLOUT else None
            if cached is not None:
                self.stats.cache_hits += 1
                results[index] = cached
            else:
                if self.mode == ROLLOUT:
                    self.stats.cache_misses += 1
                pending.append(index)
        if pending:
            if self.mode == ROLLOUT:
                with torch.no_grad():
                    rows = self._encode_requests([requests[index] for index in pending])
            else:
                rows = self._encode_requests([requests[index] for index in pending])
            for index, part_rows in zip(pending, rows):
                results[index] = part_rows
                if self.mode == ROLLOUT:
                    request = requests[index]
                    self.cache.put(row_cache_key(request.messages, request.compression_ratio, request.is_recursive, self.version), part_rows)
        if self.queue is not None:
            self.stats.queue_batches, self.stats.queue_requests = self.queue.batches, self.queue.requests
        device = self.device
        return [rows.to(device=device, dtype=torch.float32) for rows in results]

    def _child_request(self, part: dict, parent_ratio: float) -> EncodeRequest:
        ac_name = part.get("ac_name") or self.name
        assert ac_name == self.name, f"a part of AC model {ac_name!r} inside {self.name!r} (routing between AC models is not supported)"
        ratio = part.get("compression_target")
        return EncodeRequest(normalize_messages(part.get("messages") or []), float(ratio) if ratio is not None else parent_ratio, True)

    def _encode_requests(self, requests: list[EncodeRequest]) -> list[torch.Tensor]:
        """Children of every request (one recursive batch), then the requests in token-budgeted padded batches."""
        device = self.prepare()
        child_requests: list[EncodeRequest] = []
        child_slices: list[tuple[int, int]] = []
        for request in requests:
            parts = direct_parts(request.messages)
            child_slices.append((len(child_requests), len(child_requests) + len(parts)))
            child_requests.extend(self._child_request(part, request.compression_ratio) for part in parts)
        child_rows = self.encode_batch(child_requests) if child_requests else []
        prepared = []
        for request, (start, end) in zip(requests, child_slices):
            rows = child_rows[start:end]
            ids, spans = tokenize_with_parts(self.side.tokenizer, request.messages, [r.shape[0] for r in rows], self.pad_id)
            prepared.append((ids, spans, rows, self.num_view_rows(len(ids), request.compression_ratio), request.is_recursive))
        order = sorted(range(len(prepared)), key=lambda index: len(prepared[index][0]))
        results: list[torch.Tensor | None] = [None] * len(prepared)
        batch: list[int] = []
        longest = 0
        for index in order:
            length = len(prepared[index][0])
            if batch and max(longest, length) * (len(batch) + 1) > self.config.encode_batch_max_tokens:
                self._encode_prepared_batch([prepared[i] for i in batch], batch, results, device)
                batch, longest = [], 0
            batch.append(index)
            longest = max(longest, length)
        if batch:
            self._encode_prepared_batch([prepared[i] for i in batch], batch, results, device)
        return results  # type: ignore[return-value]

    def _phase(self, name: str, started: float, device: torch.device) -> float:
        """Accumulates the phase's seconds (device synchronized) while `profile_phases` is on; returns the next start."""
        if not self.profile_phases:
            return started
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        now = time.time()
        self.stats.phase_seconds[name] = self.stats.phase_seconds.get(name, 0.0) + now - started
        return now

    def _encode_prepared_batch(self, items: list[tuple], indexes: list[int], results: list, device: torch.device) -> None:
        started = time.time()
        phase_started = started
        modules = self.modules
        base = self.side.model
        base_dtype = next(base.parameters()).dtype
        embedding = base.get_input_embeddings()
        batch_size = len(items)
        lengths = [len(ids) for ids, *_ in items]
        max_length = max(lengths)
        x = torch.zeros(batch_size, max_length, self.d_side, device=device, dtype=torch.float32)
        valid = torch.zeros(batch_size, max_length, device=device, dtype=torch.bool)
        for row, (ids, spans, child_rows, _, _) in enumerate(items):
            with torch.no_grad():
                embedded = embedding(torch.tensor(ids, device=device)).float()
            if spans:                                                                              # child rows over their placeholder runs
                pieces, cursor = [], 0
                for (start, end), rows in zip(spans, child_rows):
                    pieces.extend([embedded[cursor:start], rows.to(embedded.dtype)])
                    cursor = end
                pieces.append(embedded[cursor:])
                embedded = torch.cat(pieces, dim=0)
            x[row, :len(ids)] = embedded
            valid[row, :len(ids)] = True
        x = x + sinusoidal_positions(max_length, self.d_side, device)[None]
        phase_started = self._phase("embed", phase_started, device)
        autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
        with autocast:
            for layer in modules.mixer:
                x = layer(x, valid)
            phase_started = self._phase("mixer", phase_started, device)
            pooled, window_valid = modules.pooling(x, valid)                                   # [B, W, d], [B, W]
        pooled = pooled.float()
        phase_started = self._phase("pooling", phase_started, device)
        # Side decoder input: the valid windows then V marker rows, right-padded.
        sequences = []
        for row, (_, _, _, num_rows, is_recursive) in enumerate(items):
            kind = torch.full((num_rows,), int(is_recursive), device=device)
            markers = modules.marker_kind(kind) + modules.marker_index(torch.arange(num_rows, device=device))
            sequences.append(torch.cat([pooled[row][window_valid[row]], markers], dim=0))
        seq_lengths = [int(sequence.shape[0]) for sequence in sequences]
        max_seq = max(seq_lengths)
        inputs_embeds = torch.zeros(batch_size, max_seq, self.d_side, device=device, dtype=torch.float32)
        for row, sequence in enumerate(sequences):
            inputs_embeds[row, :sequence.shape[0]] = sequence
        attention_mask = None
        if batch_size > 1:
            attention_mask = torch.zeros(batch_size, max_seq, device=device, dtype=torch.long)
            for row, length in enumerate(seq_lengths):
                attention_mask[row, :length] = 1
        hidden = self.side.decoder_forward(inputs_embeds.to(base_dtype), attention_mask, lora_name=self.config.base_side_model_lora_name)
        phase_started = self._phase("side_decoder", phase_started, device)
        for row, (ids, _, _, num_rows, is_recursive) in enumerate(items):
            end = seq_lengths[row]
            rows = hidden[row, end - num_rows:end].float()
            rows = modules.recursive_adapter(rows) if is_recursive else modules.target_head(rows)
            results[indexes[row]] = rows
            self.stats.encodes += 1
            self.stats.side_tokens += len(ids)
            self.stats.view_rows += num_rows
        self._phase("heads", phase_started, device)
        self.stats.encode_batches += 1
        self.stats.encode_seconds.append(time.time() - started)

    # ------------------------------------------------------------------------------------------ checkpoints
    def checkpoint_folder(self, round_index: int | None = None) -> Path:
        leaf = "latest" if round_index is None else f"round_{round_index:03d}"
        return resolve_path(f"{CHECKPOINT_FOLDER}/{self.name}/{leaf}", create=False)

    def save(self, checkpoint_path: str | None = None) -> str:
        """Modules + side LoRA under `checkpoint_path` or the next round folder; `latest` refreshed; version bumped (cache keys change)."""
        folder = Path(checkpoint_path) if checkpoint_path else self.checkpoint_folder(self.saved_rounds)
        if checkpoint_path and not folder.is_absolute():
            folder = resolve_path(checkpoint_path, create=False)
        folder.mkdir(parents=True, exist_ok=True)
        self.version += 1
        torch.save({"modules": self.modules.state_dict(), "config": dataclasses.asdict(self.config), "version": self.version,
                    "d_side": self.d_side, "d_target": self.d_target}, folder / MODULES_FILE)
        module_manager = self.harness.module_manager
        peft_model = module_manager.peft_models.get(self.config.base_side_model_name)
        adapter_name = module_manager.get_lora_config(self.config.base_side_model_lora_name).adapter_name
        if peft_model is not None and adapter_name in peft_model.peft_config:
            module_manager.save_lora(self.config.base_side_model_name, self.config.base_side_model_lora_name, str(folder / SIDE_LORA_FOLDER))
        else:
            print(f"{self.name} - Side LoRA not injected, not saved with the checkpoint.")
        if not checkpoint_path:
            latest = self.checkpoint_folder(None)
            if latest.exists():
                shutil.rmtree(latest)
            shutil.copytree(folder, latest)
            self.saved_rounds += 1
        self.cache.clear()
        print(f"{self.name} - Saved checkpoint (version {self.version}) to {folder}")
        return str(folder)

    def load(self, checkpoint_path: str) -> None:
        """Modules from the folder; the side LoRA re-injects from the folder's adapter on next use."""
        folder = Path(checkpoint_path)
        if not folder.is_absolute():
            folder = resolve_path(checkpoint_path, create=False)
        state = torch.load(folder / MODULES_FILE, map_location="cpu", weights_only=False)
        assert state["d_side"] == self.d_side and state["d_target"] == self.d_target, "checkpoint widths differ from the loaded bases"
        self.modules.load_state_dict(state["modules"])
        self.version = int(state.get("version", 0))
        side_lora = folder / SIDE_LORA_FOLDER
        if (side_lora / "adapter_config.json").exists():
            module_manager = self.harness.module_manager
            module_manager.free_lora(self.config.base_side_model_name, self.config.base_side_model_lora_name)
            module_manager.get_lora_config(self.config.base_side_model_lora_name).checkpoint_path = str(side_lora)
        self.cache.clear()
        print(f"{self.name} - Loaded checkpoint (version {self.version}) from {folder}")
