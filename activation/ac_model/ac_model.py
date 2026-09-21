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
written over the placeholders, scale-aligned with positional encoding -> blocked-local mixer
-> V pooled content summaries
with one shared marker and positions -> the causal side decoder with its LoRA -> the final V
positions -> `target_head` (into the target width) or, for a recursive child, returned for the
parent's `recursive_adapter`.

Modes: rollout (no grad, CPU row cache keyed by the part's content, an encode queue that batches
concurrent requests) and training (grad, no cache, the caller batches). Trainable AC modules are
fp32; the bases stay bf16. Checkpoints: `AC_MODELS/<name>/round_NNN` (+ `latest`) holding
`ac_modules.pt` and the side LoRA folder; `version` changes on encoder updates and keys the cache.
"""
from __future__ import annotations

import dataclasses
import json
import math
import shutil
import time
import typing as t
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..common.data_syncing import resolve_path
from ..harness.hf_utils import SOURCE_DEVICE, TARGET_DEVICE
from .ac_model_utils import (
    AC_PART_TYPE,
    ACOutput,
    AdaptiveSummaryPooling,
    BlockedLocalAttentionLayer,
    DeltaHead,
    PassageAttention,
    EncodeQueue,
    EncodeRequest,
    RowCache,
    RowHead,
    WindowedPooling,
    direct_parts,
    normalize_messages,
    resolve_intervention_layers,
    row_cache_key,
    sinusoidal_positions,
    tokenize_with_parts,
    unit_rows,
)

if t.TYPE_CHECKING:
    from ..harness import HarnessRuntime
    from ..harness.loaded_model import LoadedModel

CHECKPOINT_FOLDER = "AC_MODELS"
MODULES_FILE = "ac_modules.pt"
SIDE_LORA_FOLDER = "side_lora"
ROLLOUT, TRAINING = "rollout", "training"
ARCHITECTURE_VERSION = 2
INTERVENTION_INIT_SEED_MASK = 0xDEE9    # the intervention modules draw from a forked stream (initial seed ^ mask): the shared modules and the adapters draw as before
INTERVENTION_MODULE_PREFIXES = ("delta_heads.", "passage_attention.")   # state-dict / parameter-name prefixes of the intervention modules
READOUT_MODULE_PREFIXES = ("row_readout.", "recursive_readout.")
POOLING_MODULE_PREFIX = "pooling."          # the read-outs in front of the row heads: fresh (identity) when a checkpoint predates them
DIAGNOSTIC_CAPACITY = 256        # per layer, the encoder keeps at most this many diagnostic values between trace points
PreparedPart = tuple[list[int], list[tuple[int, int]], list[torch.Tensor], int, bool, list[torch.Tensor]]   # ids, part spans, child rows, V, recursive, child states


class CompletedEpoch(t.TypedDict):
    epoch_number: int
    ac_checkpoint: str
    target_lora_checkpoint: str


@dataclass
class ActivationContextModelConfig:
    """
    The AC model's shape. The encoder is fixed: the side base's frozen embeddings -> `mixer_layers` bidirectional blocked-local
    attention layers -> summary pooling into V rows -> one causal pass of the side base with its LoRA over [content; summaries]
    -> the final V states, each read out through a PassageAttention over the level's key set (the passage's final side states
    and, with `recursive_keys`, the final states its children returned), through the row head (identity-skip FFN, RMS-normalized,
    learned scale) into the reader's input space; a recursive part's rows go through the same kind of read-out and the recursive
    adapter instead. Optional content pooling (`input_pooling_stride`) shortens the decoder's content block to about 1/stride.
    A deep model (`intervention_frequency` != 0 or explicit `intervention_layers`) adds, per intervention layer, a read-out in
    which the row's final side state attends over the passage tokens' final side states (PassageAttention, zero-initialized
    output) and a zero-initialized bottleneck head (DeltaHead) whose output the reader adds to its residual stream at the row
    positions, at the input of that layer. Both are exactly zero at init, so a deep model starts bit-identical to the
    input-only one, and an input-only checkpoint loads into a deep model (the heads fresh). The reader-side operation is a
    per-request scatter-add on the hidden state entering the named layers: the same in HF and in an engine.
    """
    ac_model_name: str                       # identifies this model; parts name it in `ac_name`
    base_side_model_name: str                # the side base (a loaded model)
    base_side_model_lora_name: str           # its adapter, registered by ModuleManager.register_ac_model when new
    target_model_name: str                   # whose input space the rows land in
    target_model_lora_name: str              # the target adapter trained alongside (ActivationContextTrainer)
    default_compression_ratio: float = 1.0 / 16.0
    mixer_layers: int = 2
    mixer_window: int = 256                  # block size of the blocked local attention (each row sees 256-512 rows per side)
    mixer_heads: int = 8
    min_view_rows: int = 4
    max_view_rows: int = 4096
    side_lora_rank: int = 128
    encode_batch_max_tokens: int = 65_536    # side tokens per padded encode batch (B x longest)
    side_gradient_checkpointing: bool = True # on the side base in training mode
    checkpoint_path: str | None = None       # loaded by register_ac_model when given
    intervention_frequency: int = 0          # deep inputs: interior reader layers receiving a residual delta per row (0 = the embedding channel only);
                                             # -1 = every full-attention layer of the reader (0 excluded; the last-layer flag is moot) - the validated setting
    intervention_layers: list[int] | None = None   # explicit reader layers instead of the frequency rule (with intervention_frequency 0; 0 < layer < depth, unique)
    add_last_layer_intervention: bool = True # ... plus the last reader block (inert at frequency 0)
    recursive_keys: bool = False             # the read-outs of a level also attend the final states its children returned (nested parts)
    input_pooling_stride: int = 0            # content pooling before the side decoder: one vector per `input_pooling_window`-wide window every
                                             # `stride` rows (the content block becomes about 1/stride of the tokens); 0 = off, the tokens go through
    input_pooling_window: int = 8

    def __post_init__(self) -> None:
        if self.input_pooling_stride < 0:
            raise ValueError("Content pooling stride must be nonnegative")
        if self.input_pooling_stride and not 0 < self.input_pooling_stride <= self.input_pooling_window:
            raise ValueError("Positive pooling stride must not exceed its positive window")
        if self.intervention_frequency < -1:
            raise ValueError("intervention_frequency must be -1 (every full-attention layer), 0 or positive")
        if self.intervention_layers is not None:
            layers = [int(layer) for layer in self.intervention_layers]
            if self.intervention_frequency != 0:
                raise ValueError("explicit intervention_layers replace the frequency rule: set intervention_frequency 0")
            if not layers or len(set(layers)) != len(layers) or any(layer <= 0 for layer in layers):
                raise ValueError("intervention_layers must be a non-empty list of distinct positive reader layers")
            self.intervention_layers = sorted(layers)
        if not 0 < self.min_view_rows <= self.max_view_rows:
            raise ValueError("Invalid view-row bounds")


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
    decoder_sequences: list[int] = field(default_factory=list)    # per-part lengths, only while profiling
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

    def __init__(self, config: ActivationContextModelConfig, d_side: int, d_target: int, intervention_layers: list[int] = ()) -> None:
        super().__init__()
        self.mixer = nn.ModuleList([BlockedLocalAttentionLayer(d_side, config.mixer_heads, config.mixer_window) for _ in range(config.mixer_layers)])
        self.summary_pooling = AdaptiveSummaryPooling(d_side, config.mixer_heads)
        self.summary_marker = nn.Parameter(torch.empty(d_side))
        self.position_scale = nn.Parameter(torch.tensor(0.1))
        self.input_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("embedding_scale", torch.tensor(1.0))
        self.recursive_adapter = RowHead(d_side, d_side)
        # The rows start as the (normalized) final side state itself when the widths allow the identity skip.
        self.target_head = RowHead(d_side, d_target, skip=d_side == d_target)
        self.register_buffer("scales_initialized", torch.tensor(False))
        nn.init.normal_(self.summary_marker, std=0.02)
        # Deep inputs: one read-out block and one bottleneck head per intervention layer, registered (parameters, state-dict keys)
        # only when the model is deep, so an input-only model's checkpoint is the one it always was. Their initialization draws
        # from forked generators: the mixer, the row heads and the adapters injected later draw exactly the sequence they drew before.
        self.intervention_layers: list[int] = [int(layer) for layer in intervention_layers]
        self.delta_heads: nn.ModuleDict | None = None
        self.passage_attention: nn.ModuleDict | None = None
        if self.intervention_layers:
            r = max(1, min(d_side, d_target) // 4)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed() ^ INTERVENTION_INIT_SEED_MASK)
                self.delta_heads = nn.ModuleDict({str(layer): DeltaHead(d_side, d_target, r) for layer in self.intervention_layers})
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed((torch.initial_seed() ^ INTERVENTION_INIT_SEED_MASK) + 1)
                self.passage_attention = nn.ModuleDict({str(layer): PassageAttention(d_side, r) for layer in self.intervention_layers})
        # The read-outs in front of the row heads (the same block the delta heads have): identity at init (zero output projection), so
        # a checkpoint that predates them loads with them fresh and reproduces its outputs. Forked streams, created last: every module
        # above draws exactly the sequence it drew before.
        r_side = max(1, d_side // 4)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((torch.initial_seed() ^ INTERVENTION_INIT_SEED_MASK) + 2)
            self.row_readout = PassageAttention(d_side, r_side)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((torch.initial_seed() ^ INTERVENTION_INIT_SEED_MASK) + 3)
            self.recursive_readout = PassageAttention(d_side, r_side)
        self.pooling: WindowedPooling | None = None
        if config.input_pooling_stride:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed((torch.initial_seed() ^ INTERVENTION_INIT_SEED_MASK) + 4)
                self.pooling = WindowedPooling(d_side, config.mixer_heads, config.input_pooling_window, config.input_pooling_stride)

    @property
    def is_deep(self) -> bool:
        return bool(self.intervention_layers)

    @property
    def intervention_modules(self) -> list[nn.Module]:
        """The per-layer delta heads in layer order."""
        return [self.delta_heads[str(layer)] for layer in self.intervention_layers] if self.delta_heads is not None else []


class ActivationContextModel:
    def __init__(self, harness: "HarnessRuntime", config: ActivationContextModelConfig) -> None:
        self.harness = harness
        self.config = config
        self.name = config.ac_model_name
        self.side = harness.loaded_models[config.base_side_model_name]
        self.target = harness.loaded_models[config.target_model_name]
        if config.base_side_model_lora_name == config.target_model_lora_name:
            raise ValueError("AC side and reader require distinct LoRA adapters")
        self.d_side = self.side.model_config.model_description.d_model
        self.d_target = self.target.model_config.model_description.d_model
        self.intervention_layers = self._resolve_intervention_layers()
        self.modules = ActivationContextModules(config, self.d_side, self.d_target, self.intervention_layers)
        self.intervention_diagnostics: dict[str, dict[int | str, list[torch.Tensor]]] = {"cosine": {}, "delta_rms": {}}   # per layer, per encoded example: cosine(delta, input row) and the delta RMS; the trainer drains them
        self.version = 0
        self.mode = ROLLOUT
        self.cache = RowCache(harness.harness_config.ac_row_cache_bytes)
        self.queue: EncodeQueue | None = None
        self.stats = ActivationContextModelStats()
        self.profile_phases = False              # per-phase encode timings into stats.phase_seconds (synchronizes the device)
        self.saved_rounds = 0
        self.epoch_number = 0
        tokenizer = self.side.tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    # ------------------------------------------------------------------------------------------ placement and modes
    def num_view_rows(self, num_side_tokens: int, compression_ratio: float | None = None) -> int:
        ratio = self.config.default_compression_ratio if compression_ratio is None else compression_ratio
        return int(min(self.config.max_view_rows, max(self.config.min_view_rows, math.ceil(num_side_tokens * ratio))))

    def part_view_rows(self, messages: list[dict], compression_ratio: float | None = None, tools: list[dict] | None = None) -> int:
        """V of a part before encoding it: its side tokenization (children's rows counted recursively) times the ratio."""
        ratio = self.config.default_compression_ratio if compression_ratio is None else float(compression_ratio)
        child_lengths = [self.part_view_rows(child.messages, child.compression_ratio, child.tools)
                         for child in (self._child_request(part, ratio) for part in direct_parts(messages))]
        ids, _ = tokenize_with_parts(self.side.tokenizer, messages, child_lengths, self.pad_id, tools=tools)
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
        for values in self.intervention_diagnostics.values():
            values.clear()                                    # the training trace drains them; rollout never records them

    @property
    def device(self) -> torch.device:
        return next(self.modules.parameters()).device

    def prepare(self, device: str = TARGET_DEVICE) -> torch.device:
        """Side base with its adapter on the device, the AC modules next to it, the head scales initialized."""
        peft_model = self.harness.module_manager.ensure_lora(self.config.base_side_model_lora_name)
        if self.mode == ROLLOUT:
            peft_model.eval()   # newly injected PEFT dropout modules otherwise default to training
            self.modules.eval()
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
            self.modules.embedding_scale.fill_(side_rms)
            self.modules.input_scale.fill_(side_rms)
            self.modules.recursive_adapter.scale.fill_(side_rms)
            self.modules.target_head.scale.fill_(target_rms)
            self.modules.scales_initialized.fill_(True)
        print(f"{self.name} - Head scales initialized: side {side_rms:.4f}, target {target_rms:.4f}")

    @staticmethod
    def _embedding_rms(loaded_model: "LoadedModel") -> float:
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

    # ------------------------------------------------------------------------------------------ deep inputs
    def _resolve_intervention_layers(self) -> list[int]:
        from ..harness.model_config import LayerType
        layers = self.target.model_config.model_description.layer_descriptions
        full = [layer.layer_idx for layer in layers if layer.layer_type is LayerType.FULL_ATTENTION]
        if self.config.intervention_layers is not None:            # explicit layers: validated against the reader's depth
            wrong = [layer for layer in self.config.intervention_layers if not 0 < layer < len(layers)]
            if wrong:
                raise ValueError(f"intervention_layers {wrong} lie outside the reader's interior layers 1..{len(layers) - 1}")
            return list(self.config.intervention_layers)
        return resolve_intervention_layers(len(layers), self.config.intervention_frequency, self.config.add_last_layer_intervention, full)

    @property
    def is_deep(self) -> bool:
        return bool(self.intervention_layers)

    def intervention_summary(self) -> dict:
        """Layers and module sizes (for reports, checkpoints and result records); empty for an input-only model."""
        if not self.is_deep:
            return {}
        return {"layers": list(self.intervention_layers), "source": "passage_attention", "attention_depth": "final", "target": "residual", "head_style": "lora",
                "readouts": "rows,recursive,deltas", "recursive_keys": bool(self.config.recursive_keys),
                "head_parameters": sum(parameter.numel() for parameter in self.modules.delta_heads.parameters()),
                "attention_parameters": sum(parameter.numel() for parameter in self.modules.passage_attention.parameters())}

    def trainable_parameters(self) -> list[nn.Parameter]:
        """The AC modules' parameters plus the side adapter's."""
        return [parameter for parameter in self.modules.parameters() if parameter.requires_grad] + self.harness.module_manager.lora_parameters(self.config.base_side_model_lora_name)

    # ------------------------------------------------------------------------------------------ encoding
    def encode(self, messages: list[dict] | str, compression_ratio: float | None = None, is_recursive: bool = False,
               tools: list[dict] | None = None) -> torch.Tensor:
        """Rows [V, d_target] of one part (or [V, d_side] after the recursive adapter when is_recursive)."""
        ratio = self.config.default_compression_ratio if compression_ratio is None else float(compression_ratio)
        return self.encode_batch([EncodeRequest(normalize_messages(messages), ratio, is_recursive, tools or None)])[0]

    def encode_async(self, messages: list[dict] | str, compression_ratio: float | None = None, is_recursive: bool = False,
                     tools: list[dict] | None = None) -> Future:
        """Rollout mode: the rows through the encode queue (concurrent callers share batches)."""
        assert self.mode == ROLLOUT, "the encode queue serves rollout mode only"
        if self.queue is None:
            self.queue = EncodeQueue(self.encode_batch)
        ratio = self.config.default_compression_ratio if compression_ratio is None else float(compression_ratio)
        return self.queue.submit(EncodeRequest(normalize_messages(messages), ratio, is_recursive, tools or None))

    def encode_batch(self, requests: list[EncodeRequest], *, with_layers: bool = False) -> list[torch.Tensor] | list[ACOutput]:
        """The rows of every request, cache first in rollout mode, the rest encoded together (children first).
        With `with_layers`, each result is the complete ACOutput (rows plus the deep model's per-layer deltas)."""
        if not requests:
            return []
        if self.is_deep and not with_layers:
            raise RuntimeError(f"{self.name} is a deep AC model: encode_batch(with_layers=True) returns its rows with their deltas; a rows-only consumer would drop them silently")
        results: list[ACOutput | None] = [None] * len(requests)
        pending: list[int] = []
        for index, request in enumerate(requests):
            cached = self.cache.get(row_cache_key(request.messages, request.compression_ratio, request.is_recursive, self.version, request.tools)) if self.mode == ROLLOUT else None
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
                    self.cache.put(row_cache_key(request.messages, request.compression_ratio, request.is_recursive, self.version, request.tools), part_rows)
        if self.queue is not None:
            self.stats.queue_batches, self.stats.queue_requests = self.queue.batches, self.queue.requests
        device = self.device
        outputs = [output.to(device, torch.float32) for output in results]
        return outputs if with_layers else [output.input_embeds for output in outputs]

    def _child_request(self, part: dict, parent_ratio: float) -> EncodeRequest:
        ac_name = part.get("ac_name") or self.name
        assert ac_name == self.name, f"a part of AC model {ac_name!r} inside {self.name!r} (routing between AC models is not supported)"
        ratio = part.get("compression_target")
        return EncodeRequest(normalize_messages(part.get("messages") or []), float(ratio) if ratio is not None else parent_ratio, True,
                             part.get("tools") or None)

    def _encode_requests(self, requests: list[EncodeRequest]) -> list[ACOutput]:
        """Children of every request (one recursive batch), then the requests in token-budgeted padded batches."""
        device = self.prepare()
        child_requests: list[EncodeRequest] = []
        child_slices: list[tuple[int, int]] = []
        for request in requests:
            parts = direct_parts(request.messages)
            child_slices.append((len(child_requests), len(child_requests) + len(parts)))
            child_requests.extend(self._child_request(part, request.compression_ratio) for part in parts)
        child_outputs = self.encode_batch(child_requests, with_layers=True) if child_requests else []
        prepared: list[PreparedPart] = []
        for request, (start, end) in zip(requests, child_slices):
            outputs = child_outputs[start:end]
            rows = [output.input_embeds for output in outputs]
            states: list[torch.Tensor] = []
            if self.config.recursive_keys:
                if any(output.states is None for output in outputs):
                    raise RuntimeError("recursive_keys needs the children's final states (a recursive result without `states`)")
                states = [output.states for output in outputs]
            ids, spans = tokenize_with_parts(self.side.tokenizer, request.messages, [r.shape[0] for r in rows], self.pad_id, tools=request.tools)
            prepared.append((ids, spans, rows, self.num_view_rows(len(ids), request.compression_ratio), request.is_recursive, states))
        order = sorted(range(len(prepared)), key=lambda index: len(prepared[index][0]))
        results: list[ACOutput | None] = [None] * len(prepared)
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

    def _encode_prepared_batch(self, items: list[PreparedPart], indexes: list[int], results: list[ACOutput | None], device: torch.device) -> None:
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
        for row, (ids, spans, child_rows, _, _, _) in enumerate(items):
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
        # Both ordinary embeddings and recursive rows enter at side embedding scale.
        # Work at unit scale so a random pre-norm residual cannot overwhelm the content.
        x = x / modules.embedding_scale.clamp_min(1e-8)
        x = x + modules.position_scale * sinusoidal_positions(max_length, self.d_side, device)[None]
        phase_started = self._phase("embed", phase_started, device)
        autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
        def run(module: nn.Module, *args: torch.Tensor | int) -> torch.Tensor:
            if self.mode == TRAINING and torch.is_grad_enabled() and self.config.side_gradient_checkpointing:
                return checkpoint(module, *args, use_reentrant=False)
            return module(*args)

        with autocast:
            for layer in modules.mixer:
                x = run(layer, x, valid)
            phase_started = self._phase("mixer", phase_started, device)
            if modules.pooling is not None:                                 # content pooling: about 1/stride of the tokens enter the decoder
                pooled, window_valid = run(modules.pooling, x, valid)
                phase_started = self._phase("pooling", phase_started, device)
            sequences = []
            for row, (_, _, _, num_rows, _, _) in enumerate(items):
                mixed = x[row, :lengths[row]]
                content = mixed if modules.pooling is None else pooled[row][window_valid[row]]
                summaries = run(modules.summary_pooling, mixed, num_rows)   # the summaries always pool from the unpooled content
                summaries = (unit_rows(summaries) + modules.summary_marker
                             + modules.position_scale * sinusoidal_positions(num_rows, self.d_side, device))
                sequences.append(torch.cat([unit_rows(content), unit_rows(summaries)], dim=0) * modules.input_scale)
        phase_started = self._phase("summaries", phase_started, device)
        seq_lengths = [int(sequence.shape[0]) for sequence in sequences]
        if self.profile_phases:
            self.stats.decoder_sequences.extend(seq_lengths)
        max_seq = max(seq_lengths)
        inputs_embeds = torch.zeros(batch_size, max_seq, self.d_side, device=device, dtype=torch.float32)
        for row, sequence in enumerate(sequences):
            inputs_embeds[row, :sequence.shape[0]] = sequence
        attention_mask = None
        if batch_size > 1:
            attention_mask = torch.zeros(batch_size, max_seq, device=device, dtype=torch.long)
            for row, length in enumerate(seq_lengths):
                attention_mask[row, :length] = 1
        enabled = (self.mode == TRAINING and torch.is_grad_enabled() and self.config.side_gradient_checkpointing
                   and getattr(self, "_training_gradient_checkpointing", True)
                   and max_seq >= getattr(self, "_side_gradient_checkpointing_min_tokens", 0))
        hidden = self.side.decoder_forward(inputs_embeds.to(base_dtype), attention_mask,
                                           lora_name=self.config.base_side_model_lora_name, gradient_checkpointing=enabled)
        phase_started = self._phase("side_decoder", phase_started, device)
        for row, (ids, _, _, num_rows, is_recursive, child_states) in enumerate(items):
            end = seq_lengths[row]
            rows = hidden[row, end - num_rows:end].float()
            # The level's key set: the passage positions' final side states (one fp32 copy per example, shared by every read-out)
            # plus, with recursive_keys, the final states its children returned (the parent reads a child at the child's granularity).
            keys = hidden[row, :end - num_rows].float()
            if child_states:
                keys = torch.cat([keys, *[states.to(keys.dtype) for states in child_states]], dim=0)
            if is_recursive:
                results[indexes[row]] = ACOutput(modules.recursive_adapter(modules.recursive_readout(rows, keys)), states=rows if self.config.recursive_keys else None)   # states ride the cache only when a parent reads them
            else:
                input_embeds = modules.target_head(modules.row_readout(rows, keys))
                deltas = {}
                if modules.is_deep:
                    for layer in self.intervention_layers:
                        source = modules.passage_attention[str(layer)](rows, keys)
                        deltas[layer] = modules.delta_heads[str(layer)](source)
                        if self.mode == TRAINING:                              # does the delta re-inject the row? cosine with the input row, per layer; and the delta's RMS (0-d tensors, read at the trace point)
                            with torch.no_grad():
                                cosine = F.cosine_similarity(deltas[layer].detach(), input_embeds.detach(), dim=-1).mean()
                            self._record_diagnostic("cosine", layer, cosine)
                            self._record_diagnostic("delta_rms", layer, self._row_rms(deltas[layer]))
                results[indexes[row]] = ACOutput(input_embeds, deltas)
            self.stats.encodes += 1
            self.stats.side_tokens += len(ids)
            self.stats.view_rows += num_rows
        self._phase("heads", phase_started, device)
        self.stats.encode_batches += 1
        self.stats.encode_seconds.append(time.time() - started)

    @staticmethod
    def _row_rms(rows: torch.Tensor) -> torch.Tensor:
        """Mean over rows of the per-row RMS, detached (0-d tensor: no host sync)."""
        with torch.no_grad():
            return rows.detach().float().pow(2).mean(dim=-1).sqrt().mean()

    def _record_diagnostic(self, name: str, key: int | str, value: torch.Tensor) -> None:
        values = self.intervention_diagnostics[name].setdefault(key, [])
        values.append(value)
        del values[:-DIAGNOSTIC_CAPACITY]                     # bounded when nothing drains it (eval-only use of a training-mode model)

    # ------------------------------------------------------------------------------------------ checkpoints
    def invalidate_encoded_rows(self) -> None:
        """An encoder update changes cache identity even when checkpointing is disabled or fails."""
        self.version += 1
        self.cache.clear()

    def checkpoint_folder(self, round_index: int | None = None) -> Path:
        leaf = "latest" if round_index is None else f"round_{round_index:03d}"
        return resolve_path(f"{CHECKPOINT_FOLDER}/{self.name}/{leaf}", create=False)

    def save(self, checkpoint_path: str | None = None, *, epoch_number: int | None = None) -> str:
        """Modules + side LoRA under `checkpoint_path` or the next round folder; `latest` refreshed only for default destinations."""
        folder = Path(checkpoint_path) if checkpoint_path else self.checkpoint_folder(self.saved_rounds)
        if checkpoint_path and not folder.is_absolute():
            folder = resolve_path(checkpoint_path, create=False)
        self.harness.module_manager.ensure_lora(self.config.base_side_model_lora_name)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"modules": self.modules.state_dict(), "config": {**dataclasses.asdict(self.config), "row_head_skip": self.modules.target_head.skip}, "version": self.version,
                    "d_side": self.d_side, "d_target": self.d_target, "architecture_version": ARCHITECTURE_VERSION,
                    "model_ids": [self.side.model_config.model_id, self.target.model_config.model_id],
                    "side_lora_targets": list(self.harness.module_manager.get_lora_config(self.config.base_side_model_lora_name).target_modules),
                    "epoch_number": self.epoch_number if epoch_number is None else epoch_number,
                    **({"interventions": self.intervention_summary()} if self.is_deep else {})}, folder / MODULES_FILE)
        module_manager = self.harness.module_manager
        module_manager.save_lora(self.config.base_side_model_name, self.config.base_side_model_lora_name, str(folder / SIDE_LORA_FOLDER))
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
        """Modules from the folder; the side LoRA re-injects from the folder's adapter on next use. An input-only checkpoint loads
        into a deep model (the shared modules loaded, the intervention modules at their seeded init); a deep checkpoint loads into
        the same intervention layers only."""
        folder = Path(checkpoint_path)
        if not folder.is_absolute():
            folder = resolve_path(checkpoint_path, create=False)
        state = torch.load(folder / MODULES_FILE, map_location="cpu", weights_only=False)
        if state.get("architecture_version") != ARCHITECTURE_VERSION:
            raise ValueError("Checkpoint uses an incompatible AC architecture (legacy learned-marker checkpoints cannot load as content summaries)")
        expected_ids = [self.side.model_config.model_id, self.target.model_config.model_id]
        if state.get("model_ids") != expected_ids or state["d_side"] != self.d_side or state["d_target"] != self.d_target:
            raise ValueError("Checkpoint model identities or widths differ from the loaded bases")
        structural = ("mixer_layers", "mixer_window", "mixer_heads", "min_view_rows", "max_view_rows", "side_lora_rank", "default_compression_ratio")
        differences = [name for name in structural if state["config"].get(name) != getattr(self.config, name)]
        saved_stride = int(state["config"].get("input_pooling_stride", 0) or 0)
        # An unpooled checkpoint into a pooled model is a migration (the pooler is a new module, trained from its init on top of the
        # loaded encoder: that is how the strided arms warm-start). Every other stride change is refused, the reverse direction included.
        pooling_fresh = saved_stride == 0 and self.config.input_pooling_stride > 0
        if saved_stride != self.config.input_pooling_stride and not pooling_fresh:
            differences.append(f"input_pooling_stride (checkpoint {saved_stride}, model {self.config.input_pooling_stride})")
        elif saved_stride and int(state["config"].get("input_pooling_window", 0) or 0) != self.config.input_pooling_window:
            differences.append("input_pooling_window")
        if bool(state["config"].get("row_head_skip", False)) != self.modules.target_head.skip:
            differences.append("row_head_skip")
        if state["config"].get("encoder_passes", 1) != 1 or state["config"].get("pool_surprisal", 0.0):
            differences.append("encoder (iterative passes or surprisal pooling are no longer part of the model)")
        saved_interventions = state.get("interventions") or {}
        saved_layers = [int(layer) for layer in saved_interventions.get("layers", [])]   # absent in input-only checkpoints
        migrate = self.is_deep and not saved_layers          # input-only checkpoint into a deep model: shared modules loaded, intervention modules fresh
        if not migrate and saved_layers != self.intervention_layers:
            raise ValueError(f"Checkpoint intervention layers {saved_layers} differ from the configured {self.intervention_layers} (a deep checkpoint loads into the same layers only)")
        if saved_layers:
            saved_shape = (saved_interventions.get("source", "summary"), saved_interventions.get("attention_depth"),
                           saved_interventions.get("target", "residual"), saved_interventions.get("head_style", "scaled"))
            if saved_shape != ("passage_attention", "final", "residual", "lora"):
                raise ValueError(f"Checkpoint interventions {saved_shape} are not the model's design (passage attention over the final side states, "
                                 "zero-initialized residual deltas): that checkpoint belongs to the full-featured branch")
        targets = list(self.harness.module_manager.get_lora_config(self.config.base_side_model_lora_name).target_modules)
        if differences or state.get("side_lora_targets") != targets:
            raise ValueError(f"Checkpoint AC configuration differs: {differences or ['side_lora_targets']}")
        side_lora = folder / SIDE_LORA_FOLDER
        self._validate_adapter(side_lora, self.config.base_side_model_lora_name)
        modules_state = dict(state["modules"])
        for name in list(modules_state):        # inert calibration state that earlier deep checkpoints carried
            if name in ("intervention_calibration", "interventions_calibrated") or (name.startswith("delta_heads.") and name.split(".")[-1] in ("relative", "rms")):
                modules_state.pop(name)
        # Two kinds of keys may be absent: the intervention modules when an input-only checkpoint migrates into a deep model, and the
        # row read-outs when the checkpoint predates them (identity at init: the outputs are the checkpoint's). Anything else raises.
        deep_keys = {name for name in self.modules.state_dict() if name.startswith(INTERVENTION_MODULE_PREFIXES)} if migrate else set()
        readout_keys = {name for name in self.modules.state_dict() if name.startswith(READOUT_MODULE_PREFIXES)}
        pooling_keys = {name for name in self.modules.state_dict() if name.startswith(POOLING_MODULE_PREFIX)} if pooling_fresh else set()
        expected = set(self.modules.state_dict()); saved = set(modules_state)
        missing, unexpected = expected - saved, saved - expected               # decided before anything is written: a mismatch leaves the modules untouched
        fresh_readouts = missing & readout_keys
        if (missing - deep_keys - readout_keys - pooling_keys) or unexpected or (fresh_readouts and fresh_readouts != readout_keys) \
                or (migrate and not deep_keys <= missing) or (pooling_fresh and not pooling_keys <= missing):
            raise ValueError(f"Checkpoint modules do not match (missing {sorted(missing - deep_keys - readout_keys - pooling_keys)}, unexpected {sorted(unexpected)})")
        self.modules.load_state_dict(modules_state, strict=False)
        if bool(state["config"].get("recursive_keys", False)) != self.config.recursive_keys:
            print(f"{self.name} - Note: the checkpoint was trained with recursive_keys={state['config'].get('recursive_keys', False)}; this model uses {self.config.recursive_keys} (the read-outs see a different key set)")
        if migrate:
            print(f"{self.name} - Migrated an input-only checkpoint into a deep model: shared modules loaded, {len(self.intervention_layers)} intervention layers fresh (seeded init)")
        if fresh_readouts:
            print(f"{self.name} - Row read-outs fresh (identity at init): the checkpoint predates the read-outs in front of the row heads")
        if pooling_fresh:
            print(f"{self.name} - Content pooling fresh (stride {self.config.input_pooling_stride}, window {self.config.input_pooling_window}): "
                  "the checkpoint was trained on unpooled content, the pooler starts at its init")
        self.version = max(self.version, int(state.get("version", 0)))
        self.epoch_number = int(state.get("epoch_number", 0))
        self.invalidate_encoded_rows()
        module_manager = self.harness.module_manager
        module_manager.free_lora(self.config.base_side_model_name, self.config.base_side_model_lora_name)
        module_manager.get_lora_config(self.config.base_side_model_lora_name).checkpoint_path = str(side_lora)
        self.cache.clear()
        print(f"{self.name} - Loaded checkpoint (version {self.version}) from {folder}")

    def _validate_adapter(self, folder: Path, lora_name: str) -> None:
        config = self.harness.module_manager.get_lora_config(lora_name)
        weights = [*folder.glob("*.safetensors"), *folder.glob("*.bin")]
        if not (folder / "adapter_config.json").is_file() or not any(path.stat().st_size for path in weights):
            raise ValueError(f"Incomplete adapter checkpoint: {folder}")
        saved = json.loads((folder / "adapter_config.json").read_text())
        if (saved.get("r") != config.rank or saved.get("lora_alpha") != config.alpha
                or set(saved.get("target_modules", [])) != set(config.target_modules)):
            raise ValueError(f"Adapter configuration differs: {lora_name}")

    def load_completed_epoch(self) -> CompletedEpoch:
        """Load both numbered paths from the committed pair; ignore possibly partial latest aliases.

        Use a fresh trainer after loading: optimizer state is intentionally not resumed across processes.
        """
        record = json.loads((self.checkpoint_folder().parent / "completed_epoch.json").read_text())
        epoch = int(record["epoch_number"])
        expected_ac = f"AC_MODELS/{self.name}/epoch_{epoch:03d}"
        expected_target = f"LORAS/{self.config.target_model_lora_name}/epoch_{epoch:03d}"
        if record["ac_checkpoint"] != expected_ac or record["target_lora_checkpoint"] != expected_target:
            raise ValueError("Completed checkpoint pair belongs to different registered modules")
        target_path = resolve_path(expected_target, create=False)
        self._validate_adapter(target_path, self.config.target_model_lora_name)
        self.load(expected_ac)
        manager = self.harness.module_manager
        manager.free_lora(self.config.target_model_name, self.config.target_model_lora_name)
        manager.get_lora_config(self.config.target_model_lora_name).checkpoint_path = str(target_path)
        return record
