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
-> optional content pooling (stride 0 retains content), plus V separately pooled content summaries
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
import hashlib
import json
import math
import shutil
import time
import typing as t
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..common.data_syncing import resolve_path
from ..harness.hf_utils import SOURCE_DEVICE, TARGET_DEVICE
from contextlib import nullcontext
from .ac_model_utils import (
    AC_PART_TYPE,
    ACOutput,
    AdaptiveSummaryPooling,
    BlockedLocalAttentionLayer,
    DeltaHead,
    KVDeltaHead,
    KVSlotHead,
    KVTransferHead,
    PassageAttention,
    RefineAdapter,
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
INTERVENTION_SCALE_FRACTION = 0.05      # calibrated initial delta magnitude: this share of the destination layer's residual RMS
INTERVENTION_INIT_SEED_MASK = 0xDEE9
INTERVENTION_TARGETS = ("residual", "kv", "kv_slots", "cross_attention", "kv_transfer", "kv_transfer_full")
KV_TARGETS = ("kv", "kv_slots", "kv_transfer", "kv_transfer_full")   # targets that touch the pre-RoPE keys and values of full-attention layers (calibrated against the K and V RMS)
KV_REPLACEMENT_TARGETS = ("kv_slots", "kv_transfer", "kv_transfer_full")   # the row slices are REPLACED (keep x the reader's own + content): one reader path, one engine mapping
KV_TRANSFER_TARGETS = ("kv_transfer", "kv_transfer_full")   # the content is the side model's own K/V at the same layer ("_full": over the whole passage run - a CEILING)
INTERVENTION_MODULE_PREFIXES = ("delta_heads.", "passage_attention.", "cross_attention.")   # state-dict / parameter-name prefixes of the intervention modules
REFINE_ADAPTER_RANK = 256        # iterative encoder: the refine adapter's bottleneck width (min(256, d_side))
DIAGNOSTIC_CAPACITY = 256        # per layer, the encoder keeps at most this many diagnostic values between trace points    # delta heads draw from a forked stream (initial seed ^ mask): the shared modules and the adapters draw as before
PreparedPart = tuple[list[int], list[tuple[int, int]], list[torch.Tensor], int, bool]


class CompletedEpoch(t.TypedDict):
    epoch_number: int
    ac_checkpoint: str
    target_lora_checkpoint: str


@dataclass
class ActivationContextModelConfig:
    ac_model_name: str                       # identifies this model; parts name it in `ac_name`
    base_side_model_name: str                # the side base (a loaded model)
    base_side_model_lora_name: str           # its adapter, registered by ModuleManager.register_ac_model when new
    target_model_name: str                   # whose input space the rows land in
    target_model_lora_name: str              # the target adapter trained alongside (ActivationContextTrainer)
    default_compression_ratio: float = 1.0 / 16.0
    input_pooling_stride: int = 0           # zero bypasses content pooling; summaries always pool
    input_pooling_window: int = 8
    mixer_layers: int = 2
    mixer_window: int = 256                  # block size of the blocked local attention (each row sees 256-512 rows per side)
    mixer_heads: int = 8
    min_view_rows: int = 4
    max_view_rows: int = 4096
    side_lora_rank: int = 128
    encode_batch_max_tokens: int = 65_536    # side tokens per padded encode batch (B x longest)
    side_gradient_checkpointing: bool = True # on the side base in training mode
    checkpoint_path: str | None = None       # loaded by register_ac_model when given
    row_head_skip: bool = False              # target head = RMS-normalized (x + FFN(x)) with the FFN's output zero-initialized: rows start as the side state itself
    row_scale_init: float = 1.0              # target_head.scale starts at this multiple of the target embedding RMS (rows louder than tokens from the start)
    side_lora_target_modules: list[str] | None = None   # None = the manager's default seven projections
    encoder_passes: int = 1                  # iterative encoder: side-decoder passes over [content; summaries]. 1 = today. Pass k >= 2 re-reads the passage with
                                             # summaries_k = summaries_1 + refine_adapter(unit_rows(the final row states of pass k - 1)) (marker and positions as in pass 1);
                                             # the mixer and pooling run once; the rows and any K/V capture come from the LAST pass; the adapter is zero-initialized
                                             # (pass 2 starts as an exact re-read). Write-time cost x passes, reader and inference untouched.
    pool_surprisal: float = 0.0              # surprisal-weighted summary pooling: the summary bins are drawn by cumulative mass 1 + pool_surprisal x s_t / mean(s)
                                             # (s_t = the side BASE model's no-context surprisal of content token t, one extra no-grad side forward per uncached part,
                                             # cached by token ids) instead of token count, and the pooling logits gain a zero-initialised per-head bias
                                             # pool_surprisal x b_h x s_t / mean(s). Rows per part unchanged. 0 = today's token-count bins (bit-identical).
    intervention_frequency: int = 0          # deep inputs: interior reader layers receiving a residual delta per row (0 = the embedding channel only; the product default becomes 4 once the POC passes);
                                             # -1 = every full-attention layer of the reader (0 excluded; the last-layer flag is moot)
    intervention_layers: list[int] | None = None   # explicit reader layers instead of the frequency rule (with intervention_frequency 0; 0 < layer < depth, unique)
    add_last_layer_intervention: bool = True # ... plus the last reader block (inert at frequency 0)
    prefer_attention_interventions: bool = True   # snap interior points to a full-attention block within two indices (earlier on a tie)
    intervention_source: str = "summary"     # what the delta heads read: "summary" = the side model's final state at the row positions (the vector that also
                                             # makes the input row); "side_layers" = the side model's hidden state at the row positions at layer
                                             # intervention_source_layers[i] for intervention layer i (the layer-matched readout); "passage_attention" = per
                                             # layer, the row's state plus a zero-initialized attention of that row over the passage tokens' states (A2)
    intervention_source_layers: list[int] | None = None   # side layers for "side_layers" and "passage_attention"/"matched" (None = the resolved reader layers, s(l) = l)
    intervention_attention_depth: str = "final"   # "passage_attention": "final" = the row's final state queries the passage tokens' final states; "matched" = both from side layer s(l)
    intervention_target: str = "residual"    # where the deltas land: "residual" = the reader layer's input; "kv" = the layer's pre-RoPE keys and values at the
                                             # row positions (full-attention layers only; one head per layer emits dk, dv with separate calibrated scales);
                                             # "kv_slots" = the row positions' pre-RoPE K/V of full-attention layers are REPLACED by keep x the reader's own
                                             # + a side-computed slot content (KVSlotHead: zero-init head, takeover 0 at init, so bit-identical to no
                                             # intervention; the engine mapping is a per-request K/V write into the rows' cache entries); "cross_attention" =
                                             # a MEASUREMENT, not a candidate: every target position adds a zero-init cross-attention over the side model's
                                             # final states of ALL passage tokens (zero compression: the ceiling of any fetch-side design; not engine-compatible);
                                             # "kv_transfer" = the row positions' pre-RoPE K/V of full-attention layers are REPLACED by (1 - t) x the reader's own + t x the
                                             # SIDE MODEL'S OWN K/V at the same layer and positions (captured after k_norm / v_proj in the side pass; the reader's native
                                             # format, so the replacement is meaningful at init, t = intervention_transfer_init; the engine mapping is the kv_slots one);
                                             # "kv_transfer_full" = a CEILING MEASUREMENT: the same splice over the WHOLE passage run (the reader's placeholder run is as
                                             # long as the passage, the rows first and zero fill after; zero compression at those layers)
    intervention_transfer_init: float = 1.0  # kv_transfer: the takeovers' initial value (1.0 = a full replacement by the side K/V at init; 0.5 = an even mix)
    intervention_transfer_correction: bool = False   # kv_transfer: add a zero-initialized lora-style correction head (side state -> dK, dV) to the transferred K/V
    intervention_slots_extra: int = 0        # kv_slots: extra K/V slots per layer beyond the row positions (prefix-style). Designed only: any value but 0 is refused
                                             # (see POC_IMPLEMENTATION_5.md; extra entries must be prompt positions in both HF and vLLM, i.e. more rows)
    intervention_scale_fraction: float = INTERVENTION_SCALE_FRACTION   # initial delta scale as a share of the calibrated residual (or K/V) RMS: the heads' `relative` start
    intervention_heads_frozen: bool = False  # control: the heads' weights stay at their random init; only the relative scales train
    intervention_detach_source: bool = False # control: the heads read a detached source (the encoder never hears the deltas' gradient)
    intervention_head_style: str = "scaled"  # "scaled" = unit-RMS rows x (calibrated RMS x learned relative scale); "lora" = down(silu(up(z))) with a zero-initialized
                                             # down projection: the delta is exactly 0 at init, the head owns its magnitude, the scales are inert and never optimized
    intervention_calibration: str = "no_rows"   # what calibrate_interventions measures: "no_rows" = the median residual (or K/V) RMS over the positions of the stripped
                                             # histories (no row positions exist); "row_positions" = the mean RMS at the row positions of real training sequences with
                                             # the rows present (the quantity `intervention_row_rms` traces). The lora style uses it for diagnostics only.

    @property
    def layer_matched_source(self) -> bool:
        """The heads read side layer s(l) states: 'side_layers', or 'passage_attention' at depth 'matched'."""
        return self.intervention_source == "side_layers" or (self.intervention_source == "passage_attention" and self.intervention_attention_depth == "matched")

    def __post_init__(self) -> None:
        if self.intervention_frequency < -1:
            raise ValueError("intervention_frequency must be -1 (every full-attention layer), 0 or positive")
        if self.intervention_layers is not None:
            layers = [int(layer) for layer in self.intervention_layers]
            if self.intervention_frequency != 0:
                raise ValueError("explicit intervention_layers replace the frequency rule: set intervention_frequency 0")
            if not layers or len(set(layers)) != len(layers) or any(layer <= 0 for layer in layers):
                raise ValueError("intervention_layers must be a non-empty list of distinct positive reader layers")
            self.intervention_layers = sorted(layers)
        if self.intervention_source not in ("summary", "side_layers", "passage_attention"):
            raise ValueError("intervention_source must be 'summary', 'side_layers' or 'passage_attention'")
        if self.intervention_target not in INTERVENTION_TARGETS:
            raise ValueError(f"intervention_target must be one of {INTERVENTION_TARGETS}")
        if self.intervention_slots_extra:
            raise ValueError("intervention_slots_extra is designed only (0): extra K/V slots are extra prompt positions in HF and vLLM alike, i.e. more rows")
        if self.intervention_target == "kv_slots" and self.intervention_head_style != "lora":
            raise ValueError("intervention_target 'kv_slots' needs intervention_head_style 'lora' (its head is zero-initialized around the reader's own K/V; a scaled head would replace the rows' keys with noise)")
        if self.intervention_target == "cross_attention" and (self.intervention_source != "summary" or self.intervention_heads_frozen):
            raise ValueError("intervention_target 'cross_attention' reads the side model's final passage states (intervention_source 'summary') and has no heads to freeze")
        if self.intervention_target in KV_TRANSFER_TARGETS:
            if self.intervention_head_style != "lora":
                raise ValueError(f"intervention_target '{self.intervention_target}' needs intervention_head_style 'lora' (its optional correction is a zero-initialized lora-style head; there is no scaled variant)")
            if self.intervention_source != "side_layers":
                raise ValueError(f"intervention_target '{self.intervention_target}' reads the side model at the reader's own layers: intervention_source 'side_layers'")
            if self.intervention_target == "kv_transfer_full" and self.input_pooling_stride:
                raise ValueError("intervention_target 'kv_transfer_full' splices the side K/V of every passage token: content pooling must be off (input_pooling_stride 0)")
        if self.intervention_attention_depth not in ("final", "matched"):
            raise ValueError("intervention_attention_depth must be 'final' or 'matched'")
        if self.intervention_source_layers is not None and not self.layer_matched_source:
            raise ValueError("intervention_source_layers applies to intervention_source 'side_layers' or 'passage_attention' at depth 'matched' only")
        if self.intervention_attention_depth != "final" and self.intervention_source != "passage_attention":
            raise ValueError("intervention_attention_depth applies to intervention_source 'passage_attention' only")
        if self.intervention_head_style not in ("scaled", "lora"):
            raise ValueError("intervention_head_style must be 'scaled' or 'lora'")
        if self.intervention_calibration not in ("no_rows", "row_positions"):
            raise ValueError("intervention_calibration must be 'no_rows' or 'row_positions'")
        if self.intervention_heads_frozen and self.intervention_head_style == "lora":
            raise ValueError("intervention_heads_frozen needs intervention_head_style 'scaled' (a lora head with frozen weights emits nothing)")
        if not 0 < self.intervention_scale_fraction < 1:
            raise ValueError("intervention_scale_fraction must lie in (0, 1)")
        if self.input_pooling_stride < 0:
            raise ValueError("Content pooling stride must be nonnegative")
        if self.input_pooling_stride and not 0 < self.input_pooling_stride <= self.input_pooling_window:
            raise ValueError("Positive pooling stride must not exceed its positive window")
        if not 0 < self.min_view_rows <= self.max_view_rows:
            raise ValueError("Invalid view-row bounds")
        if type(self.encoder_passes) is not int or self.encoder_passes < 1:
            raise ValueError("encoder_passes must be an integer >= 1 (1 = a single side-decoder pass, today's encoder)")
        if self.encoder_passes > 1 and self.intervention_target in ("kv_transfer_full", "cross_attention"):
            raise ValueError(f"encoder_passes > 1 with intervention_target '{self.intervention_target}' would cost an extra pass for nothing: their spliced content comes "
                             "from the content positions, whose states are pass-invariant under causal attention (only the summary positions change)")
        if not math.isfinite(self.pool_surprisal) or self.pool_surprisal < 0:
            raise ValueError("pool_surprisal must be finite and nonnegative (0 = token-count bins)")


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

    def __init__(self, config: ActivationContextModelConfig, d_side: int, d_target: int, intervention_layers: list[int] = (), d_kv: int | None = None) -> None:
        super().__init__()
        self.mixer = nn.ModuleList([BlockedLocalAttentionLayer(d_side, config.mixer_heads, config.mixer_window) for _ in range(config.mixer_layers)])
        self.pooling = (WindowedPooling(d_side, config.mixer_heads, config.input_pooling_window, config.input_pooling_stride)
                        if config.input_pooling_stride else None)
        self.summary_pooling = AdaptiveSummaryPooling(d_side, config.mixer_heads, surprisal=config.pool_surprisal)
        self.summary_marker = nn.Parameter(torch.empty(d_side))
        self.position_scale = nn.Parameter(torch.tensor(0.1))
        self.input_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("embedding_scale", torch.tensor(1.0))
        self.recursive_adapter = RowHead(d_side, d_side)
        self.target_head = RowHead(d_side, d_target, skip=config.row_head_skip)
        self.register_buffer("scales_initialized", torch.tensor(False))
        nn.init.normal_(self.summary_marker, std=0.02)
        # Deep inputs: one bottleneck head per intervention layer, registered (parameters, buffers, state-dict keys) only when
        # the model is deep, so an input-only model's checkpoint is the one it always was. Their initialization draws from a
        # forked generator: the mixer, heads and the adapters injected later draw exactly the sequence they drew before.
        self.delta_heads: nn.ModuleDict | None = None
        self.cross_attention: nn.ModuleDict | None = None    # "cross_attention": one reader-side block per layer (no delta heads)
        self.intervention_layers: list[int] = [int(layer) for layer in intervention_layers]
        self.intervention_target = config.intervention_target
        if self.intervention_layers:
            r = max(1, min(d_side, d_target) // 4)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed() ^ INTERVENTION_INIT_SEED_MASK)
                if config.intervention_target in KV_TARGETS:
                    if not d_kv:
                        raise ValueError("K/V interventions need the reader's K/V width (num_key_value_heads x head_dim)")
                    if config.intervention_target == "kv":
                        self.delta_heads = nn.ModuleDict({str(layer): KVDeltaHead(d_side, d_kv, r, config.intervention_scale_fraction, config.intervention_head_style) for layer in self.intervention_layers})
                    elif config.intervention_target == "kv_slots":
                        self.delta_heads = nn.ModuleDict({str(layer): KVSlotHead(d_side, d_kv, r) for layer in self.intervention_layers})
                    else:                      # K/V transfer: the takeovers (and the optional correction head) per layer; no draws without the correction
                        self.delta_heads = nn.ModuleDict({str(layer): KVTransferHead(d_side, d_kv, r, config.intervention_transfer_init, config.intervention_transfer_correction)
                                                          for layer in self.intervention_layers})
                    self.register_buffer("intervention_calibration", torch.zeros(len(self.intervention_layers), 2))   # (K RMS, V RMS) per layer at calibration
                elif config.intervention_target == "cross_attention":
                    # Queries from the reader's residual (d_target), keys and values from the side model's final states (d_side); no residual inside the block.
                    self.cross_attention = nn.ModuleDict({str(layer): PassageAttention(d_target, r, d_passage=d_side, residual=False) for layer in self.intervention_layers})
                    self.register_buffer("intervention_calibration", torch.zeros(len(self.intervention_layers)))   # residual RMS per layer at calibration (diagnostic anchor)
                else:
                    self.delta_heads = nn.ModuleDict({str(layer): DeltaHead(d_side, d_target, r, config.intervention_scale_fraction, config.intervention_head_style) for layer in self.intervention_layers})
                    self.register_buffer("intervention_calibration", torch.zeros(len(self.intervention_layers)))   # residual RMS per layer at calibration
            self.register_buffer("interventions_calibrated", torch.tensor(False))
        # A2: one passage-attention block in front of every head, drawn after the heads (their draws stay those of the summary model).
        self.passage_attention: nn.ModuleDict | None = None
        if self.intervention_layers and config.intervention_source == "passage_attention":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed((torch.initial_seed() ^ INTERVENTION_INIT_SEED_MASK) + 1)
                self.passage_attention = nn.ModuleDict({str(layer): PassageAttention(d_side, max(1, min(d_side, d_target) // 4)) for layer in self.intervention_layers})
        # Iterative encoder: the refine adapter exists (parameters, state-dict keys) only with encoder_passes >= 2, drawn from its own forked stream so every
        # other module (and the adapters injected later) draws exactly what it draws at encoder_passes 1; its output projection is zero-initialized.
        self.refine_adapter: RefineAdapter | None = None
        if config.encoder_passes >= 2:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed((torch.initial_seed() ^ INTERVENTION_INIT_SEED_MASK) + 2)
                self.refine_adapter = RefineAdapter(d_side, min(REFINE_ADAPTER_RANK, d_side))

    @property
    def is_deep(self) -> bool:
        return bool(self.intervention_layers)

    @property
    def intervention_modules(self) -> list[nn.Module]:
        """The per-layer intervention modules in layer order (delta heads, or the cross-attention blocks)."""
        owner = self.delta_heads if self.delta_heads is not None else self.cross_attention
        return [owner[str(layer)] for layer in self.intervention_layers] if owner is not None else []



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
        self.intervention_source_layers = self._resolve_source_layers()
        self.modules = ActivationContextModules(config, self.d_side, self.d_target, self.intervention_layers,
                                                self.target.model_config.model_description.d_kv)
        if self.kv_transfer and self.side is not self.target:
            raise ValueError("the K/V transfer targets need the side and the reader to share one base model: the side K/V are read in the reader's native format")
        self.calibration_items: list[str] = []      # what calibrate_interventions measured on (checkpoint provenance)
        self.intervention_diagnostics: dict[str, dict[int | str, list[torch.Tensor]]] = {"cosine": {}, "delta_rms": {}, "encoder": {}}   # per layer (K/V: "l:k" / "l:v"), per encoded example: cosine(delta, target_head(z)) and the delta RMS; the trainer drains them
                                                                                                                                      # "encoder": the round-8 write-side diagnostics per encoded part (refine_rms, pass_cosine, pool_*), drained the same way
        self.surprisal_cache: dict[str, torch.Tensor] = {}   # pool_surprisal: the side base's no-context surprisal per part, keyed by the part's side token ids (the base never changes)
        self._calibrating = False                    # encode_batch may run before calibration inside `uncalibrated_encoding` (the row-position calibration needs the rows)
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
        self._refuse_nested_parts(messages)
        child_lengths = [self.part_view_rows(child.messages, child.compression_ratio, child.tools)
                         for child in (self._child_request(part, ratio) for part in direct_parts(messages))]
        ids, _ = tokenize_with_parts(self.side.tokenizer, messages, child_lengths, self.pad_id, tools=tools)
        return self.run_rows(len(ids), ratio)

    def run_rows(self, num_side_tokens: int, compression_ratio: float | None = None) -> int:
        """The placeholder run a part occupies in the reader: its rows (num_view_rows), or under kv_transfer_full the whole passage run
        max(N, V) - the rows first, zero fill after - so every passage token's side K/V has a reader position to land on."""
        rows = self.num_view_rows(num_side_tokens, compression_ratio)
        return max(int(num_side_tokens), rows) if self.kv_transfer_full else rows

    def _refuse_nested_parts(self, messages: list[dict]) -> None:
        """kv_transfer_full's run is the part's own passage: a part nested inside it (recursive rows) has no side positions of its own to splice."""
        if self.kv_transfer_full and direct_parts(messages):
            raise ValueError("intervention_target 'kv_transfer_full' (a ceiling measurement) does not support nested parts")

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
            self.modules.target_head.scale.fill_(target_rms * self.config.row_scale_init)
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
        if self.config.intervention_layers is not None:            # explicit layers: validated against the reader's depth (K/V targets check full attention below)
            wrong = [layer for layer in self.config.intervention_layers if not 0 < layer < len(layers)]
            if wrong:
                raise ValueError(f"intervention_layers {wrong} lie outside the reader's interior layers 1..{len(layers) - 1}")
            return list(self.config.intervention_layers)
        return resolve_intervention_layers(len(layers), self.config.intervention_frequency, self.config.add_last_layer_intervention,
                                           self.config.prefer_attention_interventions, full)

    def _resolve_source_layers(self) -> list[int]:
        """Side layers read by the heads under intervention_source 'side_layers' (s(l) = l unless configured); [] for 'summary'.
        Also checks that K/V targets are full-attention reader layers."""
        from ..harness.model_config import LayerType
        if not self.intervention_layers:
            return []
        if self.config.intervention_target in KV_TARGETS:
            full = {layer.layer_idx for layer in self.target.model_config.model_description.layer_descriptions if layer.layer_type is LayerType.FULL_ATTENTION}
            wrong = [layer for layer in self.intervention_layers if layer not in full]
            if wrong:
                raise ValueError(f"K/V interventions need full-attention reader layers; {wrong} are not (prefer_attention_interventions snaps to them)")
        if not self.config.layer_matched_source:
            return []
        layers = list(self.config.intervention_source_layers) if self.config.intervention_source_layers is not None else list(self.intervention_layers)
        side_depth = len(self.side.model_config.model_description.layer_descriptions)
        if len(layers) != len(self.intervention_layers) or any(not 0 <= int(layer) < side_depth for layer in layers):
            raise ValueError(f"intervention_source_layers must name one side layer (0..{side_depth - 1}) per intervention layer {self.intervention_layers}: {layers}")
        return [int(layer) for layer in layers]

    @property
    def is_deep(self) -> bool:
        return bool(self.intervention_layers)

    @property
    def kv_interventions(self) -> bool:
        """Additive K/V deltas at the rows (`kv`)."""
        return self.is_deep and self.config.intervention_target == "kv"

    @property
    def kv_targets(self) -> bool:
        """Any target on the pre-RoPE keys and values (`kv` deltas or `kv_slots` replacement): calibrated against the K and V RMS."""
        return self.is_deep and self.config.intervention_target in KV_TARGETS

    @property
    def kv_slots(self) -> bool:
        return self.is_deep and self.config.intervention_target == "kv_slots"

    @property
    def kv_transfer(self) -> bool:
        """The side model's own K/V replace the reader's at the rows (`kv_transfer`) or over the whole passage run (`kv_transfer_full`)."""
        return self.is_deep and self.config.intervention_target in KV_TRANSFER_TARGETS

    @property
    def kv_transfer_full(self) -> bool:
        return self.is_deep and self.config.intervention_target == "kv_transfer_full"

    @property
    def kv_replacement(self) -> bool:
        """Any replacement of the rows' K/V (`kv_slots` payload, takeovers, the kv_slots reader path and engine mapping)."""
        return self.is_deep and self.config.intervention_target in KV_REPLACEMENT_TARGETS

    @property
    def cross_attention(self) -> bool:
        return self.is_deep and self.config.intervention_target == "cross_attention"

    @property
    def interventions_calibrated(self) -> bool:
        return self.is_deep and bool(self.modules.interventions_calibrated)

    def intervention_summary(self) -> dict:
        """Layers, calibration and current scales (for reports and result records); empty for an input-only model."""
        if not self.is_deep:
            return {}
        heads = self.modules.delta_heads
        calibration = self.modules.intervention_calibration.tolist()
        scales, relative, takeover = {}, {}, {}
        if self.kv_targets:
            rms = {layer: {"k": float(k), "v": float(v)} for layer, (k, v) in zip(self.intervention_layers, calibration)}
        else:
            rms = {layer: float(value) for layer, value in zip(self.intervention_layers, calibration)}
        if self.kv_interventions:
            scales = {layer: {"k": float(heads[str(layer)].scale_k.detach()), "v": float(heads[str(layer)].scale_v.detach())} for layer in self.intervention_layers}
            relative = {layer: {"k": float(heads[str(layer)].relative_k.detach()), "v": float(heads[str(layer)].relative_v.detach())} for layer in self.intervention_layers}
        elif self.kv_replacement:      # no scales: the takeover of the reader's own K and V at the rows (0 = the reader's own, 1 = purely side-computed / the side's own)
            takeover = {layer: {"k": float(heads[str(layer)].takeover_k.detach()), "v": float(heads[str(layer)].takeover_v.detach())} for layer in self.intervention_layers}
        elif heads is not None:
            scales = {layer: float(heads[str(layer)].scale.detach()) for layer in self.intervention_layers}
            relative = {layer: float(heads[str(layer)].relative.detach()) for layer in self.intervention_layers}
        blocks = self.modules.passage_attention if self.modules.passage_attention is not None else self.modules.cross_attention
        return {"layers": list(self.intervention_layers), "calibrated": self.interventions_calibrated, "calibration_items": list(self.calibration_items),
                "source": self.config.intervention_source, "source_layers": list(self.intervention_source_layers), "target": self.config.intervention_target,
                "attention_depth": self.config.intervention_attention_depth if self.config.intervention_source == "passage_attention" else None,
                "calibration_rms": rms, "scales": scales, "relative": relative, **({"takeover": takeover} if self.kv_replacement else {}), "scale_fraction": self.config.intervention_scale_fraction,
                **({"transfer": {"init": self.config.intervention_transfer_init, "correction": self.config.intervention_transfer_correction}} if self.kv_transfer else {}),
                "parametrization": "relative" if self.config.intervention_head_style == "scaled" and heads is not None and not self.kv_replacement else "lora",
                "head_style": self.config.intervention_head_style, "calibration": self.config.intervention_calibration,
                "heads_frozen": self.config.intervention_heads_frozen, "detach_source": self.config.intervention_detach_source,
                "head_parameters": sum(parameter.numel() for parameter in heads.parameters()) if heads is not None else 0,
                "attention_parameters": sum(parameter.numel() for parameter in blocks.parameters()) if blocks is not None else 0,
                **({"measurement_only": True} if self.cross_attention or self.kv_transfer_full else {})}      # the ceiling targets only: every other summary keeps its keys

    @contextmanager
    def uncalibrated_encoding(self) -> t.Iterator[None]:
        """Lets encode_batch run on a deep model before calibration (the row-position calibration needs the rows the reader will see;
        the deltas of such an encode are meaningless and must be dropped). The diagnostics recorded meanwhile are discarded."""
        self._calibrating = True
        try:
            yield
        finally:
            self._calibrating = False
            for values in self.intervention_diagnostics.values():
                values.clear()

    def calibrate_interventions(self, sequences: list[list[int]] | None, item_ids: list[str] | None = None, *,
                                row_inputs: list[tuple[torch.Tensor, list[int]]] | None = None) -> dict[int, float]:
        """
        Measure every intervention layer's residual RMS (K/V targets: K and V RMS) by running the reader (its adapter active, no
        deltas, no gradient, one sequence per forward) on a fixed training-only sample and store it in the heads (`rms`).
        `intervention_calibration` picks the measurement: "no_rows" takes `sequences` (token ids of the stripped histories, no
        row positions exist; the median over positions, first token excluded); "row_positions" takes `row_inputs` (per sequence
        the reader's input embeddings with the encoded rows written in, and the row positions; the mean RMS at those positions:
        the quantity the training trace reports as `intervention_row_rms`). In the scaled style the applied scale is rms x
        relative, the learned `relative` starting at the scale fraction; in the lora style the value only feeds the diagnostics.
        Runs once per fresh (or migrated) deep model; checkpoints carry the result and never recalibrate.
        """
        if not self.is_deep:
            raise ValueError(f"{self.name} has no interventions to calibrate")
        on_rows = self.config.intervention_calibration == "row_positions"
        if on_rows and not row_inputs:
            raise ValueError("intervention_calibration 'row_positions' needs row_inputs: at least one reader input with its row positions")
        if not on_rows and (not sequences or row_inputs):
            raise ValueError("intervention_calibration 'no_rows' needs token sequences (and no row_inputs)")
        from ..harness.hf_utils import capture_kv_rms, capture_layer_input_rms
        device = self.prepare()
        self.harness.module_manager.ensure_lora(self.config.target_model_lora_name)
        model = self.target.model
        was_training = model.training
        model.eval()
        embedding = model.get_input_embeddings()
        capture = capture_kv_rms if self.kv_targets else capture_layer_input_rms
        record: dict[int, list] = {layer: [] for layer in self.intervention_layers}
        try:
            with torch.no_grad():
                if on_rows:
                    for embeds, positions in row_inputs:
                        if not positions:
                            raise ValueError("a row-position calibration sequence carries no row positions")
                        with capture(model, self.intervention_layers, positions) as measured:
                            self.target.decoder_forward(embeds.to(device)[None], None, lora_name=self.config.target_model_lora_name, gradient_checkpointing=False)
                        for layer, values in measured.items():
                            record[layer].extend(values)
                else:
                    with capture(model, self.intervention_layers) as measured:
                        for ids in sequences:
                            embeds = embedding(torch.tensor(ids, device=device, dtype=torch.long))[None]
                            self.target.decoder_forward(embeds, None, lora_name=self.config.target_model_lora_name, gradient_checkpointing=False)
                    record = measured
        finally:
            model.train(was_training)
        with torch.no_grad():
            if self.kv_targets:      # per layer (K RMS, V RMS): each scale a share of its own statistic (slots: diagnostic anchors)
                rms = {layer: (sum(k for k, _ in values) / len(values), sum(v for _, v in values) / len(values)) for layer, values in record.items()}
                for index, layer in enumerate(self.intervention_layers):
                    self.modules.intervention_calibration[index, 0] = rms[layer][0]
                    self.modules.intervention_calibration[index, 1] = rms[layer][1]
                    self.modules.delta_heads[str(layer)].rms_k.fill_(rms[layer][0])
                    self.modules.delta_heads[str(layer)].rms_v.fill_(rms[layer][1])
                described = ", ".join(f"layer {layer} k rms {k:.3f} v rms {v:.3f}" for layer, (k, v) in rms.items())
            else:
                rms = {layer: sum(values) / len(values) for layer, values in record.items()}
                for index, layer in enumerate(self.intervention_layers):
                    self.modules.intervention_calibration[index] = rms[layer]
                    if self.modules.delta_heads is not None:
                        self.modules.delta_heads[str(layer)].rms.fill_(rms[layer])
                described = ", ".join(f"layer {layer} rms {value:.3f}" for layer, value in rms.items())
            self.modules.interventions_calibrated.fill_(True)
        self.calibration_items = list(item_ids or [])
        self.invalidate_encoded_rows()
        print(f"{self.name} - Interventions calibrated ({self.config.intervention_calibration}) on {len(row_inputs if on_rows else sequences)} sequences: " + described)
        return rms

    def trainable_parameters(self) -> list[nn.Parameter]:
        """The AC modules' parameters (minus the delta heads' weights when they are frozen) plus the side adapter's."""
        modules = [parameter for name, parameter in self.modules.named_parameters()
                   if parameter.requires_grad       # the lora-style heads' inert scales never train
                   and not (self.config.intervention_heads_frozen and name.startswith(INTERVENTION_MODULE_PREFIXES) and not name.split(".")[-1].startswith("relative"))]
        return modules + self.harness.module_manager.lora_parameters(self.config.base_side_model_lora_name)

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
        if self.is_deep and not self.interventions_calibrated and not self._calibrating:
            raise RuntimeError(f"{self.name} is a deep AC model whose delta scales are not calibrated: call calibrate_interventions (or load a calibrated checkpoint) first")
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
            self._refuse_nested_parts(request.messages)
            parts = direct_parts(request.messages)
            child_slices.append((len(child_requests), len(child_requests) + len(parts)))
            child_requests.extend(self._child_request(part, request.compression_ratio) for part in parts)
        child_rows = [output.input_embeds for output in self.encode_batch(child_requests, with_layers=True)] if child_requests else []
        prepared: list[PreparedPart] = []
        for request, (start, end) in zip(requests, child_slices):
            rows = child_rows[start:end]
            ids, spans = tokenize_with_parts(self.side.tokenizer, request.messages, [r.shape[0] for r in rows], self.pad_id, tools=request.tools)
            prepared.append((ids, spans, rows, self.num_view_rows(len(ids), request.compression_ratio), request.is_recursive))
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
        # Both ordinary embeddings and recursive rows enter at side embedding scale.
        # Work at unit scale so a random pre-norm residual cannot overwhelm the content.
        x = x / modules.embedding_scale.clamp_min(1e-8)
        x = x + modules.position_scale * sinusoidal_positions(max_length, self.d_side, device)[None]
        surprisals = self._content_surprisal(items, device) if self.config.pool_surprisal > 0 else None   # one per item: [N] fp32 on the device
        phase_started = self._phase("embed", phase_started, device)
        autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
        def run(module: nn.Module, *args: torch.Tensor | int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            if self.mode == TRAINING and torch.is_grad_enabled() and self.config.side_gradient_checkpointing:
                return checkpoint(module, *args, use_reentrant=False)
            return module(*args)

        with autocast:
            for layer in modules.mixer:
                x = run(layer, x, valid)
            phase_started = self._phase("mixer", phase_started, device)
            if modules.pooling is not None:
                pooled, window_valid = run(modules.pooling, x, valid)
            phase_started = self._phase("pooling", phase_started, device)
            sequences, content_blocks, summary_blocks = [], [], []
            for row, (_, _, _, num_rows, _) in enumerate(items):
                mixed = x[row, :lengths[row]]
                content = mixed if modules.pooling is None else pooled[row][window_valid[row]]
                if surprisals is None:
                    summaries = run(modules.summary_pooling, mixed, num_rows)
                else:
                    summaries = run(modules.summary_pooling, mixed, num_rows, surprisals[row])
                    if self.mode == TRAINING:
                        self._record_pool_diagnostics(surprisals[row], lengths[row], num_rows)
                summaries = (unit_rows(summaries) + modules.summary_marker
                             + modules.position_scale * sinusoidal_positions(num_rows, self.d_side, device))
                content_blocks.append(unit_rows(content))                   # shared by every pass (the mixer and pooling run once)
                summary_blocks.append(summaries)                             # summaries_1: the base of every later pass's row inputs
                sequences.append(torch.cat([content_blocks[-1], unit_rows(summaries)], dim=0) * modules.input_scale)
        phase_started = self._phase("summaries", phase_started, device)
        seq_lengths = [int(sequence.shape[0]) for sequence in sequences]
        if self.profile_phases:
            self.stats.decoder_sequences.extend(seq_lengths)
        max_seq = max(seq_lengths)
        def padded(sequences: list[torch.Tensor]) -> torch.Tensor:
            inputs_embeds = torch.zeros(batch_size, max_seq, self.d_side, device=device, dtype=torch.float32)
            for row, sequence in enumerate(sequences):
                inputs_embeds[row, :sequence.shape[0]] = sequence
            return inputs_embeds
        inputs_embeds = padded(sequences)
        attention_mask = None
        if batch_size > 1:
            attention_mask = torch.zeros(batch_size, max_seq, device=device, dtype=torch.long)
            for row, length in enumerate(seq_lengths):
                attention_mask[row, :length] = 1
        enabled = (self.mode == TRAINING and torch.is_grad_enabled() and self.config.side_gradient_checkpointing
                   and getattr(self, "_training_gradient_checkpointing", True)
                   and max_seq >= getattr(self, "_side_gradient_checkpointing_min_tokens", 0))
        # Layer-matched readout: the heads read the side model's state at the row positions at their source layers, captured
        # by pre-hooks on exactly those layers (row slices only) during this forward; "summary" reads the final state (`rows`).
        side_layers = modules.is_deep and self.config.layer_matched_source
        attention = modules.passage_attention if modules.is_deep else None
        matched = attention is not None and self.config.intervention_attention_depth == "matched"
        transfer = self.kv_transfer
        captured_kv = None
        if transfer:
            # K/V transfer: the side model's own pre-RoPE K and V at the chosen layers, at the spliced side positions (the summary rows, or the
            # whole passage run for the ceiling), captured by forward hooks on k_norm / v_proj during this forward (graph-attached slices).
            from ..harness.hf_utils import capture_kv_rows
            kv_spans = [self._transfer_span(seq_lengths[row], item[3]) for row, item in enumerate(items)]
            kv_capture = capture_kv_rows(self.side.model, set(self.intervention_layers), kv_spans)
        else:
            kv_capture = nullcontext()
        # Iterative encoder: passes 1 .. encoder_passes - 1 run without any capture (no hook is registered, so nothing fires); the LAST pass runs
        # inside the capture contexts, so the layer-matched states and the transferred K/V are those of the final read, like the rows. Pass k >= 2
        # re-reads the same content with the row inputs summaries_1 + refine_adapter(unit_rows(the final row states of pass k - 1)); the graph runs
        # through every pass (each pass is its own checkpointed decoder call).
        passes = int(getattr(self, "_encoder_passes_override", 0) or self.config.encoder_passes)   # the trainer's passes_off panel forces 1
        first_rows: list[torch.Tensor] = []                                                       # pass-1 final row states (detached), for the pass cosine
        for pass_index in range(1, passes + 1):
            if pass_index > 1:
                with autocast:
                    refined = []
                    for row, (_, _, _, num_rows, _) in enumerate(items):
                        end = seq_lengths[row]
                        previous = unit_rows(hidden[row, end - num_rows:end].float())
                        if pass_index == 2 and self.mode == TRAINING:
                            first_rows.append(previous.detach())
                        refinement = modules.refine_adapter(previous)
                        if pass_index == passes and self.mode == TRAINING:
                            self._record_diagnostic("encoder", "refine_rms", self._row_rms(refinement) / self._row_rms(summary_blocks[row]).clamp_min(1e-8))
                        summaries = summary_blocks[row] + refinement
                        refined.append(torch.cat([content_blocks[row], unit_rows(summaries)], dim=0) * modules.input_scale)
                inputs_embeds = padded(refined)
            last = pass_index == passes
            if side_layers and last:
                from ..harness.hf_utils import capture_layer_input_rows
                # Passage attention at depth "matched" needs the passage tokens' states too: the whole valid sequence is captured (a view, no copy).
                # The transfer's correction reads the side state at the spliced positions.
                row_spans = kv_spans if transfer else [((0 if matched else seq_lengths[row] - item[3]), seq_lengths[row]) for row, item in enumerate(items)]
                with kv_capture as captured_kv, capture_layer_input_rows(self.side.model, set(self.intervention_source_layers), row_spans) as captured:
                    hidden = self.side.decoder_forward(inputs_embeds.to(base_dtype), attention_mask,
                                lora_name=self.config.base_side_model_lora_name, gradient_checkpointing=enabled)
                if any(len(states) != batch_size for states in captured.values()):
                    raise RuntimeError("side-layer capture missed rows (the side decoder did not run the requested layers once)")
            elif last:
                with kv_capture as captured_kv:
                    hidden = self.side.decoder_forward(inputs_embeds.to(base_dtype), attention_mask,
                                lora_name=self.config.base_side_model_lora_name, gradient_checkpointing=enabled)
            else:
                hidden = self.side.decoder_forward(inputs_embeds.to(base_dtype), attention_mask,
                            lora_name=self.config.base_side_model_lora_name, gradient_checkpointing=enabled)
        if transfer and any(len(keys) != batch_size or len(values) != batch_size for keys, values in captured_kv.values()):
            raise RuntimeError("side K/V capture missed rows (the side decoder did not run the requested full-attention layers once)")
        if first_rows:                                                          # pass cosine: the pass-1 rows against the last pass's, per part (unit rows)
            with torch.no_grad():
                for row, (_, _, _, num_rows, _) in enumerate(items):
                    end = seq_lengths[row]
                    final = unit_rows(hidden[row, end - num_rows:end].detach().float())
                    self._record_diagnostic("encoder", "pass_cosine", F.cosine_similarity(first_rows[row], final, dim=-1).mean())
        phase_started = self._phase("side_decoder", phase_started, device)
        for row, (ids, _, _, num_rows, is_recursive) in enumerate(items):
            end = seq_lengths[row]
            rows = hidden[row, end - num_rows:end].float()
            if is_recursive:
                results[indexes[row]] = ACOutput(modules.recursive_adapter(rows))
            else:
                deltas, kv, slots, cross_passage = {}, {}, {}, None
                input_embeds = modules.target_head(rows)
                if self.kv_transfer_full:                                      # the run is the whole passage: the rows first, zero fill after
                    run = self._transfer_span(end, num_rows)[1]
                    if run > num_rows:
                        input_embeds = torch.cat([input_embeds, input_embeds.new_zeros(run - num_rows, input_embeds.shape[1])], dim=0)
                if self.cross_attention:
                    # The measurement target: the side model's final states of every passage position travel to the reader whole (no per-layer head here;
                    # the reader's blocks query them at its target positions). The control detaches them like any other source.
                    cross_passage = hidden[row, :end - num_rows].float()
                    if self.config.intervention_detach_source:
                        cross_passage = cross_passage.detach()
                elif modules.is_deep:
                    final_passage = hidden[row, :end - num_rows].float() if attention is not None and not matched else None   # one fp32 copy per example, shared by the layers
                    for index, layer in enumerate(self.intervention_layers):
                        if matched:                                                # side layer s(l): the row states query that layer's passage states
                            full = captured[self.intervention_source_layers[index]][row].float()
                            source, passage = full[end - num_rows:], full[:end - num_rows]
                        elif side_layers:
                            source, passage = captured[self.intervention_source_layers[index]][row].float(), None
                        else:                                                      # final: the row's z queries the passage tokens' final (normed) states
                            source, passage = rows, final_passage
                        if self.config.intervention_detach_source:
                            source = source.detach()                               # control: the heads learn, the encoder never hears them
                            passage = passage.detach() if passage is not None else None
                        if attention is not None:
                            source = attention[str(layer)](source, passage)
                        if self.kv_interventions:
                            kv[layer] = modules.delta_heads[str(layer)](source)
                            if self.mode == TRAINING:
                                self._record_diagnostic("delta_rms", f"{layer}:k", self._row_rms(kv[layer][0]))
                                self._record_diagnostic("delta_rms", f"{layer}:v", self._row_rms(kv[layer][1]))
                        elif self.kv_slots:                                        # the slot content plus the layer's keep factors (shared by every part)
                            head = modules.delta_heads[str(layer)]
                            slot_k, slot_v = head(source)
                            slots[layer] = (slot_k, slot_v, *head.keep())
                            if self.mode == TRAINING:
                                self._record_diagnostic("delta_rms", f"{layer}:k", self._row_rms(slot_k))
                                self._record_diagnostic("delta_rms", f"{layer}:v", self._row_rms(slot_v))
                        elif transfer:                                             # the side model's own K/V at this layer (+ correction), scaled by the takeover
                            head = modules.delta_heads[str(layer)]
                            side_k, side_v = captured_kv[layer][0][row].float(), captured_kv[layer][1][row].float()
                            if self.config.intervention_detach_source:
                                side_k, side_v = side_k.detach(), side_v.detach()
                            correction = head(source)
                            if correction is not None:
                                side_k, side_v = side_k + correction[0], side_v + correction[1]
                            content_k, content_v = head.takeover_k * side_k, head.takeover_v * side_v
                            slots[layer] = (content_k, content_v, *head.keep())
                            if self.mode == TRAINING:
                                self._record_diagnostic("delta_rms", f"{layer}:k", self._row_rms(content_k))
                                self._record_diagnostic("delta_rms", f"{layer}:v", self._row_rms(content_v))
                        else:
                            deltas[layer] = modules.delta_heads[str(layer)](source)
                            if self.mode == TRAINING:                              # does the delta re-inject the row? cosine with the input row, per layer; and the delta's RMS (0-d tensors, read at the trace point)
                                with torch.no_grad():
                                    cosine = F.cosine_similarity(deltas[layer].detach(), input_embeds.detach(), dim=-1).mean()
                                self._record_diagnostic("cosine", layer, cosine)
                                self._record_diagnostic("delta_rms", layer, self._row_rms(deltas[layer]))
                results[indexes[row]] = ACOutput(input_embeds, deltas, kv, slots, cross_passage)
            self.stats.encodes += 1
            self.stats.side_tokens += len(ids)
            self.stats.view_rows += num_rows
        self._phase("heads", phase_started, device)
        self.stats.encode_batches += 1
        self.stats.encode_seconds.append(time.time() - started)

    def _transfer_span(self, seq_length: int, num_rows: int) -> tuple[int, int]:
        """The side positions whose K/V the transfer splices, for one side sequence of `seq_length` = passage (N) + summary rows (V):
        the rows [N, N + V) for kv_transfer; [0, max(N, V)) for kv_transfer_full (the whole passage run; the summaries only when V > N)."""
        if self.kv_transfer_full:
            return 0, max(seq_length - num_rows, num_rows)
        return seq_length - num_rows, seq_length

    @torch.no_grad()
    def _content_surprisal(self, items: list[PreparedPart], device: torch.device) -> list[torch.Tensor]:
        """pool_surprisal: per item the side BASE model's (no adapter) no-context surprisal of every side token, s_t = -log p_base(id_t | id_<t) in nats,
        s_0 = 0 and the placeholder runs of nested parts set to the mean over the real tokens; one right-padded forward per batch, the gold log-prob read
        chunk by chunk through the LM head. Cached per part by its token ids (the base never changes, so the cache is valid across training)."""
        from ..agent_training.agent_training_utils import head_logits
        results: list[torch.Tensor | None] = [None] * len(items)
        keys, pending = [], []
        for index, (ids, *_) in enumerate(items):
            key = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            keys.append(key)
            cached = self.surprisal_cache.get(key)
            if cached is not None:
                results[index] = cached.to(device)
            else:
                pending.append(index)
        if pending:
            base = self.side.model
            embedding = base.get_input_embeddings()
            head = base.get_output_embeddings()
            lengths = [len(items[index][0]) for index in pending]
            longest = max(lengths)
            ids = torch.full((len(pending), longest), self.pad_id, device=device, dtype=torch.long)
            for row, index in enumerate(pending):
                ids[row, :lengths[row]] = torch.tensor(items[index][0], device=device)
            hidden = self.side.decoder_forward(embedding(ids), None, lora_name=None, gradient_checkpointing=False)   # right padding: a real token never sees a pad
            surprisal = torch.zeros(len(pending), longest, device=device, dtype=torch.float32)
            chunk = 256
            for start in range(0, longest - 1, chunk):
                end = min(start + chunk, longest - 1)
                states = hidden[:, start:end].reshape(-1, hidden.shape[-1])                   # head_logits' CUDA bf16 path is a plain matmul: matrices only
                logp = F.log_softmax(head_logits(states, head), dim=-1).view(len(pending), end - start, -1)
                surprisal[:, start + 1:end + 1] = -logp.gather(-1, ids[:, start + 1:end + 1, None])[..., 0]
            for row, index in enumerate(pending):
                values = surprisal[row, :lengths[row]].clone()
                spans = items[index][1]
                if spans:
                    real = torch.ones(lengths[row], dtype=torch.bool, device=device)
                    for span_start, span_end in spans:
                        real[span_start:span_end] = False
                    values[~real] = values[real].mean() if bool(real.any()) else 0.0
                results[index] = values
                if len(self.surprisal_cache) >= 100_000:
                    self.surprisal_cache.pop(next(iter(self.surprisal_cache)))
                self.surprisal_cache[keys[index]] = values.detach().cpu()
        return results   # type: ignore[return-value]

    def _record_pool_diagnostics(self, surprisal: torch.Tensor, length: int, num_rows: int) -> None:
        """pool_surprisal (training mode): per part, the mean surprisal of the tokens in each bin - its mean over the rows, the first row's and the last
        row's - and the widest bin against the uniform width, as 0-d tensors."""
        with torch.no_grad():
            starts, ends, width, _ = self.modules.summary_pooling.bins(length, num_rows, surprisal)
            row_means = torch.stack([surprisal[int(start):int(end)].mean() for start, end in zip(starts.tolist(), ends.tolist())])
            self._record_diagnostic("encoder", "pool_row_mean", row_means.mean())
            self._record_diagnostic("encoder", "pool_first", row_means[0])
            self._record_diagnostic("encoder", "pool_last", row_means[-1])
            self._record_diagnostic("encoder", "pool_width_max", torch.tensor(float(width)))
            self._record_diagnostic("encoder", "pool_width_uniform", torch.tensor(float((length + num_rows - 1) // num_rows + 1)))

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
        torch.save({"modules": self.modules.state_dict(), "config": dataclasses.asdict(self.config), "version": self.version,
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
        """Modules from the folder; the side LoRA re-injects from the folder's adapter on next use."""
        folder = Path(checkpoint_path)
        if not folder.is_absolute():
            folder = resolve_path(checkpoint_path, create=False)
        state = torch.load(folder / MODULES_FILE, map_location="cpu", weights_only=False)
        if state.get("architecture_version") != ARCHITECTURE_VERSION:
            raise ValueError("Checkpoint uses an incompatible AC architecture (legacy learned-marker checkpoints cannot load as content summaries)")
        expected_ids = [self.side.model_config.model_id, self.target.model_config.model_id]
        if state.get("model_ids") != expected_ids or state["d_side"] != self.d_side or state["d_target"] != self.d_target:
            raise ValueError("Checkpoint model identities or widths differ from the loaded bases")
        structural = ("input_pooling_stride", "input_pooling_window", "mixer_layers", "mixer_window", "mixer_heads", "min_view_rows", "max_view_rows", "side_lora_rank", "default_compression_ratio")
        differences = [name for name in structural if state["config"].get(name) != getattr(self.config, name)]
        if bool(state["config"].get("row_head_skip", False)) != self.config.row_head_skip:
            differences.append("row_head_skip")
        saved_interventions = state.get("interventions") or {}
        saved_layers = [int(layer) for layer in saved_interventions.get("layers", [])]   # absent in older checkpoints: input-only
        migrate = self.is_deep and not saved_layers          # input-only checkpoint into a deep model: shared modules loaded, heads fresh, calibration pending
        if not migrate and saved_layers != self.intervention_layers:
            raise ValueError(f"Checkpoint intervention layers {saved_layers} differ from the configured {self.intervention_layers} (a deep checkpoint loads into the same layers only)")
        if saved_layers:
            saved_shape = (saved_interventions.get("source", "summary"), [int(layer) for layer in saved_interventions.get("source_layers", [])],
                           saved_interventions.get("target", "residual"), saved_interventions.get("attention_depth"))
            configured = (self.config.intervention_source, self.intervention_source_layers, self.config.intervention_target,
                          self.config.intervention_attention_depth if self.config.intervention_source == "passage_attention" else None)
            if saved_shape != configured:
                raise ValueError(f"Checkpoint intervention source/target {saved_shape} differ from the configured {configured}")
            saved_style = (saved_interventions.get("head_style", "scaled"), saved_interventions.get("calibration", "no_rows"))   # absent before the lora style existed
            if saved_style != (self.config.intervention_head_style, self.config.intervention_calibration):
                raise ValueError(f"Checkpoint intervention head style/calibration {saved_style} differ from the configured "
                                 f"{(self.config.intervention_head_style, self.config.intervention_calibration)} (the head weights and the calibration mean different things)")
            if self.kv_transfer and bool((saved_interventions.get("transfer") or {}).get("correction", False)) != self.config.intervention_transfer_correction:
                raise ValueError("Checkpoint K/V transfer correction differs from the configured (the correction head's keys exist in one and not the other)")
        targets = list(self.harness.module_manager.get_lora_config(self.config.base_side_model_lora_name).target_modules)
        if differences or state.get("side_lora_targets") != targets:
            raise ValueError(f"Checkpoint AC configuration differs: {differences or ['side_lora_targets']}")
        side_lora = folder / SIDE_LORA_FOLDER
        self._validate_adapter(side_lora, self.config.base_side_model_lora_name)
        modules_state = dict(state["modules"])
        # Iterative encoder: a checkpoint written at encoder_passes 1 (no refine adapter) loads into a model with more passes, the adapter staying at its
        # zero init (the extra passes start as exact re-reads: a warm start from any single-pass checkpoint). A checkpoint carrying an adapter needs a model with one.
        saved_refine = any(name.startswith("refine_adapter.") for name in modules_state)
        if saved_refine and self.modules.refine_adapter is None:
            raise ValueError(f"Checkpoint carries a refine adapter (encoder_passes {state['config'].get('encoder_passes')}) but the configured model has encoder_passes 1")
        fresh_refine = {name for name in self.modules.state_dict() if name.startswith("refine_adapter.")} if self.modules.refine_adapter is not None and not saved_refine else set()
        # Surprisal pooling: likewise, its zero-initialised per-head bias is fresh when the checkpoint predates it; a checkpoint carrying it needs a model with it.
        saved_bias = "summary_pooling.surprisal_bias" in modules_state
        if saved_bias and self.modules.summary_pooling.surprisal_bias is None:
            raise ValueError(f"Checkpoint carries a surprisal pooling bias (pool_surprisal {state['config'].get('pool_surprisal')}) but the configured model has pool_surprisal 0")
        if self.modules.summary_pooling.surprisal_bias is not None and not saved_bias:
            fresh_refine = fresh_refine | {"summary_pooling.surprisal_bias"}
        if migrate or fresh_refine:
            deep_keys = {name for name in self.modules.state_dict() if name.startswith(INTERVENTION_MODULE_PREFIXES) or name in ("intervention_calibration", "interventions_calibrated")} if migrate else set()
            result = self.modules.load_state_dict(modules_state, strict=False)
            if set(result.missing_keys) != deep_keys | fresh_refine or result.unexpected_keys:
                raise ValueError(f"Checkpoint modules do not match (missing {sorted(set(result.missing_keys) - deep_keys - fresh_refine)}, unexpected {list(result.unexpected_keys)})")
            if migrate:
                with torch.no_grad():
                    self.modules.interventions_calibrated.fill_(False)
                print(f"{self.name} - Migrated an input-only checkpoint into a deep model: shared modules loaded, {len(self.intervention_layers)} delta heads fresh (seeded init), calibration pending")
            if fresh_refine:
                print(f"{self.name} - Fresh (zero-initialised) round-8 modules not in the checkpoint: {sorted(fresh_refine)} (encoder_passes {self.config.encoder_passes}, pool_surprisal {self.config.pool_surprisal:g})")
        else:
            converted = self._convert_absolute_scales(modules_state)
            self.modules.load_state_dict(modules_state)
            if converted:
                print(f"{self.name} - Converted the checkpoint's absolute delta scales to the relative parametrization: " + ", ".join(converted))
        self.calibration_items = list((state.get("interventions") or {}).get("calibration_items", []))
        self.version = max(self.version, int(state.get("version", 0)))
        self.epoch_number = int(state.get("epoch_number", 0))
        self.invalidate_encoded_rows()
        module_manager = self.harness.module_manager
        module_manager.free_lora(self.config.base_side_model_name, self.config.base_side_model_lora_name)
        module_manager.get_lora_config(self.config.base_side_model_lora_name).checkpoint_path = str(side_lora)
        self.cache.clear()
        print(f"{self.name} - Loaded checkpoint (version {self.version}) from {folder}")

    def _convert_absolute_scales(self, modules_state: dict) -> list[str]:
        """Checkpoints written before the relative parametrization stored one absolute `scale` per head (K/V: `scale_k`, `scale_v`) and no
        `rms` buffer; they become relative = scale / calibration RMS with rms = that RMS. Returns what was converted (empty when nothing)."""
        converted = []
        calibration = modules_state.get("intervention_calibration")
        if self.config.intervention_target not in ("residual", "kv"):
            return converted                     # the slot and cross-attention targets never had an absolute scale
        for index, layer in enumerate(self.intervention_layers):
            prefix = f"delta_heads.{layer}."
            pairs = [("scale_k", "relative_k", "rms_k", 0), ("scale_v", "relative_v", "rms_v", 1)] if self.kv_interventions else [("scale", "relative", "rms", None)]
            for old, relative, rms_name, column in pairs:
                if prefix + old not in modules_state:
                    continue
                if calibration is None:
                    raise ValueError(f"Checkpoint stores an absolute delta scale for layer {layer} but no calibration to convert it with")
                rms = float(calibration[index] if column is None else calibration[index, column])
                scale = float(modules_state.pop(prefix + old))
                modules_state[prefix + relative] = torch.tensor(scale / rms if rms > 0 else self.config.intervention_scale_fraction)
                modules_state[prefix + rms_name] = torch.tensor(rms)
                converted.append(f"layer {layer}{'' if column is None else ' ' + old[-1]}: {scale:.4g} -> relative {scale / rms if rms > 0 else float('nan'):.4g}")
        return converted

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
