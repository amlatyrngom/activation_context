"""Whole-history AC self-distillation: current-reader KL or teacher-free supervised loss."""
from __future__ import annotations

import random
import math
import hashlib
import json
import shutil
from fractions import Fraction
from pathlib import Path
import time
import typing as t
from dataclasses import asdict, dataclass, field, replace
from contextlib import ExitStack, nullcontext

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..agent_training.agent_training_utils import (
    disable_dropout, head_logits, optimizer_state_to, persistent_optimizer, set_learning_rates,
)
from ..autotuners import get_ac_training_autotuning_variables, configure_fla_runtime
from ..harness.hf_utils import SOURCE_DEVICE, TARGET_DEVICE, Interventions, generate_with_interventions
from ..common.data_syncing import resolve_path
from .ac_model import ROLLOUT, TRAINING, ActivationContextModel
from ..common.ac_parts import strip_ac_parts
from .ac_model_utils import (ACOutput, EncodeRequest, direct_parts, prepare_training_history,
                             validate_paired_histories, validate_part_capacity)

if t.TYPE_CHECKING:
    from ..harness import HarnessRuntime
    from .ac_model_reporter import ActivationContextTrainingReporter

ReferenceName = t.Literal["no_context", "recent_text", "untrained_ac", "in_context"]


class TrainingExampleRecord(t.TypedDict):
    kind: str
    teacher_tokens: int
    student_tokens: int
    seconds: float
    teacher_cached: bool


ITEM_KINDS = ("compaction", "traj_qa", "rag_qa")


@dataclass
class ActivationContextTrainingItem:
    item_id: str
    kind: str                                  # compaction | traj_qa | rag_qa (or a reference variant's name)
    in_context_messages: list[dict] | None    # full teacher view; unused in SFT
    ac_messages: list[dict]                   # full student view, including all later input parts
    in_context_start: int                     # independent logical starts, in message indexes
    ac_start: int
    assistant_token_ids: list[list[int]]      # one authoritative output per retained assistant message
    tools: list[dict] | None = None            # tool definitions for the template (both sides)
    weight: float = 1.0
    dataset_id: str = ""
    doc_ids: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)   # generator details (depth, ratio, budgets, gold position, ...)


@dataclass
class ActivationContextTrainingConfig:
    """
    The training recipe. The defaults are the ones the reconstruction bench settled on: SFT on the gold target tokens, cosine
    decay with a linear warm-up, one joint gradient-norm clip over the encoder and the reader adapter, the no-context penalty
    (λ x KL(base || adapted reader) on the same history with its AC parts removed, at the target positions, on a random
    quarter of the examples per update and scaled by 1 / fraction) whenever the reader adapter trains, and stored teacher /
    reference targets that always hold the gold token. `learning_rate_target_lora = 0` freezes the reader adapter (it neither
    steps nor carries the penalty): the regime that adapts the encoder alone to a dataset or task.
    """
    loss_kind: t.Literal["kl", "sft"] = "sft"   # "sft": cross-entropy on the target tokens; "kl": the coarsened KL to the stored teacher (the reader as it is at the
                                               # call's start, on the in-context history; its top-k log-probs + the remainder mass are stored before the first update)
    learning_rate_ac: float = 1e-4             # AC modules and the side LoRA
    learning_rate_target_lora: float = 2e-5    # the target adapter (0 = frozen: no steps, no penalty)
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    lr_schedule: str = "cosine"                # "constant" (warm-up only) or "cosine": decay to lr_final_fraction of the base rate over the planned updates
    lr_final_fraction: float = 0.1
    warmup_updates: int = 50                   # linear warm-up of both learning rates over this many updates (counted across calls)
    schedule_total_updates: int | None = None  # planned updates the cosine spans (default: updates done at the call's start + steps x num_epochs, so a resume stays on the curve)
    resume_optimizer: bool = False             # load `optimizer.pt` next to the AC checkpoint the model was loaded from (moments + updates done), once per model
    resume_skip_steps: int = 0                 # steps of the first epoch of this call already trained by a previous attempt (resume from a mid-epoch checkpoint)
    checkpoint_every_updates: int | None = None  # mid-epoch checkpoint (AC_MODELS/<name>/epoch_<n>_step_<s> + optimizer.pt + partial_epoch.json) every this many steps
    checkpoint_every_epoch: bool = True        # AC_MODELS/<name>/epoch_<n> (+ latest) and LORAS/<target lora>/epoch_<n>
    no_context_penalty_weight: float = 1.0     # λ of the no-context penalty (0 = off); applied only while the reader adapter trains
    no_context_penalty_fraction: float = 0.25  # share of training examples carrying the penalty each update; the term is scaled by 1/fraction (unbiased)
    teacher_cache_top_k: int = 32              # stored teacher (KL) and no-context reference (penalty): top-k log-probs + the remainder mass per target position,
                                               # the gold token always among them; the KL is between the k+1-way coarsened distributions (a lower bound of the exact KL)
    deltas_off_panel: bool = False             # deep models: evaluate the reporting items a second time at every panel with the rows kept and the deltas removed ("deltas_off"): held minus deltas_off is the deltas' contribution
    updates_per_epoch: int = 4                 # maximum gradient steps per epoch when examples_per_update is unset
    examples_per_update: int | None = None     # when set, overrides updates_per_epoch: ceil(items / examples_per_update) steps
    micro_batch_examples: int = 1              # SFT: examples per right-padded reader forward (1 = one example at a time)
    max_example_tokens: int = 72_000           # explicit error; no dropping or truncation
    logits_chunk_tokens: int | None = None     # completion positions per lm_head chunk
    fp32_head_matmul: bool = False
    gradient_checkpointing: bool = True        # on both bases
    gradient_checkpointing_min_tokens: int | None = None
    seed: int = 0
    num_epochs: int = 2
    reporting_interval: float = 0.1
    completion_samples: int = 4
    completion_max_new_tokens: int = 64

    def __post_init__(self) -> None:
        if self.loss_kind not in ("kl", "sft") or self.max_example_tokens < 1 or self.teacher_cache_top_k < 1:
            raise ValueError("Invalid loss kind, example capacity, or teacher cache size (the stored targets need a positive top-k)")
        if self.no_context_penalty_weight < 0 or not 0 < self.no_context_penalty_fraction <= 1:
            raise ValueError("no_context_penalty_weight must be nonnegative and no_context_penalty_fraction in (0, 1]")
        if self.learning_rate_target_lora < 0 or self.learning_rate_ac < 0:
            raise ValueError("learning rates must be nonnegative (0 freezes the reader adapter)")
        if self.lr_schedule not in ("constant", "cosine") or not 0 <= self.lr_final_fraction <= 1 or self.warmup_updates < 0:
            raise ValueError("lr_schedule must be 'constant' or 'cosine' with lr_final_fraction in [0, 1] and a nonnegative warm-up")
        if self.num_epochs < 1 or self.updates_per_epoch < 1 or not 0 < self.reporting_interval <= 1:
            raise ValueError("Epochs/updates must be positive and reporting_interval must be in (0, 1]")
        if self.examples_per_update is not None and self.examples_per_update < 1:
            raise ValueError("examples_per_update must be positive")
        if self.micro_batch_examples < 1:
            raise ValueError("micro_batch_examples must be positive")
        if self.micro_batch_examples > 1 and self.loss_kind != "sft":
            raise ValueError("micro_batch_examples > 1 is only implemented for loss_kind='sft'")
        if self.completion_samples < 0 or self.completion_max_new_tokens < 1:
            raise ValueError("Invalid completion inspection budget")

    @property
    def penalty_active(self) -> bool:
        """The no-context penalty trains the reader adapter only: it is on when λ > 0 and the adapter has a learning rate."""
        return self.no_context_penalty_weight > 0 and self.learning_rate_target_lora > 0


@dataclass(frozen=True)
class ActivationContextTrainingProgress:
    epoch: int
    epochs: int
    epoch_number: int
    step: int
    steps: int
    global_step: int
    epoch_elapsed_s: float
    elapsed_s: float


class EvalItemSummary(t.TypedDict, total=False):
    token_accuracy: float                      # teacher-forced argmax accuracy over the target positions
    item_id: str
    kind: str
    loss_kind: str
    loss: float | None
    kl: float | None
    agreement: float | None
    positions: int
    teacher_tokens: int
    student_tokens: int
    gold_nll: float                     # mean negative log-prob of the item's target tokens under the student
    target_logp: list[float]            # the student's log-prob of every target token, in order
    teacher_gold_nll: float | None      # the same under the stored teacher (KL only)
    teacher_target_logp: list[float] | None


class EvalKindSummary(t.TypedDict, total=False):
    token_accuracy: float | None
    loss_kind: str
    loss: float | None
    kl: float | None
    agreement: float | None
    gold_nll: float | None
    teacher_gold_nll: float | None
    items: int


class ActivationContextEvalSummary(t.TypedDict, total=False):
    token_accuracy: float | None
    items: int
    dropped_too_long: int
    loss_kind: str
    loss: float | None
    kl: float | None
    agreement: float | None
    gold_nll: float | None              # item mean of the per-token target NLL under the student
    teacher_gold_nll: float | None      # the same under the stored teacher (KL only)
    by_kind: dict[str, EvalKindSummary]
    per_item: list[EvalItemSummary]


class CompletionSample(t.TypedDict):
    item_id: str
    kind: str
    reference: str
    teacher: str | None
    student: str
    epoch_number: int
    global_step: int
    ac_version: int
    target_lora: str


@dataclass
class ActivationContextEpochStats:
    progress: ActivationContextTrainingProgress
    training_seconds: float = 0.0
    reporting_seconds: float = 0.0
    validation_seconds: float = 0.0
    sample_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    reporting: ActivationContextEvalSummary | None = None
    validation: ActivationContextEvalSummary | None = None
    deltas_off: ActivationContextEvalSummary | None = None       # deep models with deltas_off_panel: the reporting items without their deltas
    checkpoint_path: str | None = None
    target_lora_checkpoint_path: str | None = None
    samples: list[CompletionSample] = field(default_factory=list)


def reporting_boundaries(steps: int, interval: float) -> list[int]:
    fraction = Fraction(str(interval))
    if steps < 1 or not 0 < fraction <= 1:
        raise ValueError("Invalid reporting boundaries")
    count = (fraction.denominator + fraction.numerator - 1) // fraction.numerator
    return sorted({min(steps, (steps * i * fraction.numerator + fraction.denominator - 1) // fraction.denominator)
                   for i in range(1, count + 1)} | {steps})

@dataclass
class ActivationContextTrainingStats:
    loss_kind: str = "kl"
    loss: list[float] = field(default_factory=list)
    loss_by_kind: dict[str, list[float]] = field(default_factory=dict)
    start_epoch: int = 0
    epochs: list[ActivationContextEpochStats] = field(default_factory=list)
    baseline_samples: list[CompletionSample] = field(default_factory=list)
    baseline_seconds: float = 0.0
    placement_seconds: float = 0.0
    cleanup_seconds: float = 0.0
    other_seconds: float = 0.0
    learning_rates: list[list[float]] = field(default_factory=list)
    global_steps: list[int] = field(default_factory=list)
    items: int = 0
    examples: int = 0
    dropped_too_long: int = 0
    steps: int = 0
    kl: list[float] = field(default_factory=list)               # per step, item-weighted mean of each item’s mean KL
    agreement: list[float] = field(default_factory=list)        # per step, top-1 teacher/student agreement
    grad_norm: list[float] = field(default_factory=list)
    grad_norm_ac: list[float] = field(default_factory=list)        # encoder side (AC modules + side LoRA) before clipping
    grad_norm_reader: list[float] = field(default_factory=list)    # reader LoRA before clipping (0 while frozen)
    penalty_counts: list[int] = field(default_factory=list)         # per update: examples that carried the no-context penalty
    gate_trace: list[dict] = field(default_factory=list)           # the scalar gates every 25 updates
    intervention_trace: list[dict] = field(default_factory=list)   # deep models, every 5 updates for the first 200 then every 25: per layer the head's gradient norm, the applied delta RMS relative to the
                                                                    # residual RMS at the row positions, the delta / input-row cosine and the read-out block's output-projection norm
    payload_bytes: int = 0                                          # deep models: delta bytes (bf16) handed to the reader over the call's training forwards
    deltas_off: ActivationContextEvalSummary | None = None          # deep models: the reporting items without their deltas at the last panel
    step_seconds: list[float] = field(default_factory=list)
    step_tokens: list[int] = field(default_factory=list)        # teacher + student tokens per step
    kl_by_kind: dict[str, list[float]] = field(default_factory=dict)   # per example
    agreement_by_kind: dict[str, list[float]] = field(default_factory=dict)
    completion_tokens: int = 0
    teacher_tokens: int = 0
    student_tokens: int = 0
    side_tokens: int = 0
    reporting: ActivationContextEvalSummary | None = None               # reporting items at the last epoch boundary
    validation_baseline: ActivationContextEvalSummary | None = None     # validation items before the first update (when distinct from reporting)
    teacher_targets: dict = field(default_factory=dict)                 # KL: {"examples", "top_k", "coverage"} of the stored teacher targets
    resumed_from: str | None = None                                       # the optimizer.pt this call restored, if any
    no_context_reference: dict = field(default_factory=dict)            # penalty: the same for the base's no-context reference
    penalty: list[float] = field(default_factory=list)                  # per update: mean unweighted no-context KL over the sampled examples (carried forward when none)
    checkpoint_path: str | None = None
    target_lora_checkpoint_path: str | None = None
    peak_memory_bytes: int = 0
    duration_s: float = 0.0
    example_records: list[TrainingExampleRecord] = field(default_factory=list)   # per example: kind, teacher/student tokens, seconds, teacher_cached
    teacher_cache_hits: int = 0
    teacher_cache_bytes: int = 0
    autotuning: dict = field(default_factory=dict)

    def seconds_by_bucket(self) -> dict[str, dict]:
        """Examples, mean seconds and examples per hour by teacher-length bucket (8k, 16k, 32k, 48k, 64k, 72k, more)."""
        edges = (8_192, 16_384, 32_768, 49_152, 65_536, 73_728)
        out: dict[str, list[float]] = {}
        for record in self.example_records:
            label = next((f"<{edge // 1024}k" for edge in edges if record["teacher_tokens"] < edge), f">={edges[-1] // 1024}k")
            out.setdefault(label, []).append(record["seconds"])
        return {label: {"examples": len(values), "mean_s": round(sum(values) / len(values), 2), "per_hour": round(3600 * len(values) / sum(values))}
                for label, values in out.items() if sum(values) > 0}

    def summarize(self) -> dict:
        tokens = sum(self.step_tokens)
        seconds = sum(self.step_seconds)
        return {
            "loss_kind": self.loss_kind, "loss_first": self.loss[0] if self.loss else None,
            "loss_last": self.loss[-1] if self.loss else None,
            "loss_by_kind": {kind: sum(values) / len(values) for kind, values in self.loss_by_kind.items() if values},
            "autotuning": self.autotuning,
            "start_epoch": self.start_epoch, "epochs": [asdict(epoch) for epoch in self.epochs],
            "placement_seconds": self.placement_seconds, "cleanup_seconds": self.cleanup_seconds, "other_seconds": self.other_seconds,
            "baseline_samples": self.baseline_samples, "baseline_seconds": self.baseline_seconds, "global_steps": self.global_steps, "learning_rates": self.learning_rates, "grad_norm_ac": self.grad_norm_ac, "grad_norm_reader": self.grad_norm_reader, "penalty_counts": self.penalty_counts, "gate_trace": self.gate_trace, "intervention_trace": self.intervention_trace, "payload_bytes": self.payload_bytes, "deltas_off": self.deltas_off, "items": self.items, "examples": self.examples, "dropped_too_long": self.dropped_too_long,
            "steps": self.steps, "kl_first": round(self.kl[0], 4) if self.kl else None, "kl_last": round(self.kl[-1], 4) if self.kl else None,
            "agreement_last": round(self.agreement[-1], 4) if self.agreement else None,
            "kl_by_kind": {kind: round(sum(values) / len(values), 4) for kind, values in self.kl_by_kind.items() if values},
            "completion_tokens": self.completion_tokens, "teacher_tokens": self.teacher_tokens, "student_tokens": self.student_tokens,
            "side_tokens": self.side_tokens, "tokens_per_second": round(tokens / seconds, 1) if seconds else None,
            "reporting": self.reporting, "checkpoint": self.checkpoint_path, "target_lora_checkpoint": self.target_lora_checkpoint_path,
            "peak_memory_gb": round(self.peak_memory_bytes / 2**30, 2), "duration_s": round(self.duration_s, 1),
            "seconds_by_bucket": self.seconds_by_bucket(), "teacher_cache_hits": self.teacher_cache_hits,
            "teacher_cache_mb": round(self.teacher_cache_bytes / 2**20, 1), "teacher_targets": self.teacher_targets,
        }


@dataclass
class _Example:
    item: ActivationContextTrainingItem
    teacher_ids: list[int]
    student_ids: list[int]
    teacher_positions: list[int]
    student_positions: list[int]
    target_ids: list[int]
    spans: list[tuple[int, int]]
    part_requests: list[EncodeRequest]
    teacher_cache_key: tuple[str, str, int]
    penalty: "_Example | None" = None          # the same history without its AC parts, scored against the base (no-context penalty)

    @property
    def num_completion(self) -> int:
        return len(self.target_ids)


class ActivationContextTrainer:
    def __init__(self, harness: "HarnessRuntime", config: ActivationContextTrainingConfig | None = None) -> None:
        self.harness = harness
        self.config = config or ActivationContextTrainingConfig()
        self.epochs_done: dict[str, int] = {}
        self.resumed: set[str] = set()                # models whose optimizer.pt was restored in this process
        self.updates_done: dict[str, int] = {}                    # optimizer steps so far per AC model (the warm-up clock)
        self.optimizers: dict[str, torch.optim.AdamW] = {}        # one per AC model; moments survive train() calls
        self.teacher_cache: dict[tuple[str, str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}   # prepared-token identity -> (ids [C, k], log-probs [C, k], log remainder [C]), CPU
        self.teacher_target_logp: dict[tuple[str, str, int], torch.Tensor] = {}   # same identity -> the stored teacher's log-prob of each target token [C], CPU
        self._teacher_texts: dict[str, str] = {}                                   # item id -> the stored teacher's greedy completion, sampled once at baseline
        self._penalty_rng = random.Random(self.config.seed)                        # which training examples carry the no-context penalty
        self._penalty_term: torch.Tensor | None = None                             # set by _example_loss: the weighted penalty of the last example, if sampled
        self._last_hits: torch.Tensor | None = None                                # set by the chunked losses: argmax == target per target position
        self._payload_bytes = 0                                                    # deep models: delta bytes handed to the reader (reset per train call)
        self._row_rms: dict[int, list[torch.Tensor]] = {}                          # deep models: residual RMS at the row positions per layer since the last trace (0-d tensors)

    # ------------------------------------------------------------------------------------------ examples
    def build_example(self, ac_model: ActivationContextModel, item: ActivationContextTrainingItem, reference: bool = False) -> _Example:
        """With `reference`, the example is a no-context reference: its in-context history is scored regardless of the loss kind."""
        if not math.isfinite(item.weight) or item.weight < 0:
            raise ValueError(f"{item.item_id}: weight must be finite and nonnegative")
        tokenizer = ac_model.target.tokenizer
        chat_kwargs = item.info.get("chat_template_kwargs")
        requests = [ac_model._child_request(part, ac_model.config.default_compression_ratio)
                    for part in direct_parts(item.ac_messages)]
        requests = [EncodeRequest(request.messages, request.compression_ratio, False, request.tools) for request in requests]
        lengths = [ac_model.part_view_rows(request.messages, request.compression_ratio, request.tools) for request in requests]
        student_ids, spans, student_positions = prepare_training_history(tokenizer, item.ac_messages, item.ac_start,
                            item.assistant_token_ids, lengths, tools=item.tools, chat_template_kwargs=chat_kwargs)
        teacher_ids, teacher_positions = [], []
        if self.config.loss_kind == "kl" or reference:
            if item.in_context_messages is None:
                raise ValueError(f"{item.item_id}: KL needs an in-context teacher history")
            validate_paired_histories(item.in_context_messages, item.in_context_start, item.ac_messages, item.ac_start)
            teacher_ids, _, teacher_positions = prepare_training_history(tokenizer, item.in_context_messages,
                                item.in_context_start, item.assistant_token_ids, [], tools=item.tools, chat_template_kwargs=chat_kwargs)
        for view, ids in (("teacher", teacher_ids), ("student", student_ids)):
            if len(ids) > self.config.max_example_tokens:
                raise ValueError(f"{item.item_id}: {view} history has {len(ids)} tokens, exceeding max_example_tokens={self.config.max_example_tokens}; regenerate or raise the guard")
        digest = hashlib.sha256(json.dumps([teacher_ids, teacher_positions], separators=(",", ":")).encode()).hexdigest()
        key = (ac_model.name + ("#base" if reference else ""), digest, self.config.teacher_cache_top_k)
        example = _Example(item, teacher_ids, student_ids, teacher_positions, student_positions,
                           [token for output in item.assistant_token_ids for token in output], spans, requests, key)
        if not reference and requests and self.config.penalty_active:
            # The no-context reference: the same history with its AC parts removed, scored against the base with no adapter.
            stripped = strip_ac_parts(item.ac_messages)
            example.penalty = self.build_example(ac_model, replace(item, item_id=item.item_id + ":no_context_reference", in_context_messages=stripped,
                                                                   ac_messages=stripped, in_context_start=item.ac_start), reference=True)
        return example

    def _check_capacity(self, ac_model: ActivationContextModel, examples: t.Iterable[_Example]) -> None:
        def capacity(model):
            config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
            return getattr(config, "max_position_embeddings", None)
        target_limit, side_limit = capacity(ac_model.target.model), capacity(ac_model.side.model)
        for example in examples:
            for view, ids in (("teacher", example.teacher_ids), ("student", example.student_ids)):
                if target_limit and len(ids) > target_limit:
                    raise ValueError(f"{example.item.item_id}: {view} reader history has {len(ids)} tokens, exceeding supported capacity {target_limit}")
            if side_limit:
                for request in example.part_requests:
                    try:
                        validate_part_capacity(ac_model, request, side_limit)
                    except ValueError as error:
                        raise ValueError(f"{example.item.item_id}: {error}") from error

    # ------------------------------------------------------------------------------------------ training
    def train(
        self,
        ac_model_name: str,
        training_data: list[ActivationContextTrainingItem],
        reporting_data: t.Sequence[ActivationContextTrainingItem] = (),
        reporter: "ActivationContextTrainingReporter | None" = None,
        *,
        validation_data: t.Sequence[ActivationContextTrainingItem] = (),
        reset_optimizer: bool = False,
    ) -> ActivationContextTrainingStats:
        """Complete epochs in one residency scope; optimizer/global step persist across calls."""
        self._training_ac_name = ac_model_name
        config = self.config
        if not training_data:
            raise ValueError("No eligible AC training data")
        self.teacher_cache.clear()                 # teacher targets belong to this invocation only: the reader as it is at this call's start
        self.teacher_target_logp.clear()
        self._teacher_texts.clear()
        manager = self.harness.module_manager
        ac_model = manager.get_ac_model(ac_model_name)
        target = ac_model.target
        target_lora = ac_model.config.target_model_lora_name
        start_epoch = self._next_epoch_number(ac_model)
        stats = ActivationContextTrainingStats(loss_kind=config.loss_kind, start_epoch=start_epoch, items=len(training_data))
        started = time.monotonic()
        if reporter is not None:
            reporter.set_status(phase="building training examples", examples=str(len(training_data)))
            reporter.render(force=True)
        torch.manual_seed(config.seed + start_epoch)
        examples = [self.build_example(ac_model, item) for item in training_data]
        reporting_examples = [self.build_example(ac_model, item) for item in reporting_data]
        validation_examples = [self.build_example(ac_model, item) for item in validation_data]
        stats.examples = len(examples)
        stats.completion_tokens = sum(example.num_completion for example in examples)
        stats.teacher_tokens = sum(len(example.teacher_ids) for example in examples)
        stats.student_tokens = sum(len(example.student_ids) for example in examples)
        per_step = config.examples_per_update or -(-len(examples) // min(config.updates_per_epoch, len(examples)))
        steps = -(-len(examples) // per_step)
        boundaries = set(reporting_boundaries(steps, config.reporting_interval))
        fixed_items = list(reporting_data[:config.completion_samples])
        kernel_scope = ExitStack()
        optimizer = None
        device = None
        bases = []
        if reporter is not None:
            reporter.initialize_training(config, stats)
        try:
            placement_started = time.monotonic()
            if reporter is not None:
                reporter.set_status(phase="model placement and kernel preparation")
                reporter.render(force=True)
            if target.vllm_model is not None:
                target.engine_to_device(SOURCE_DEVICE)
            variables = self._autotuning_variables(ac_model)
            stats.autotuning = {**variables["provenance"], "logits_chunk_tokens": variables["logits_chunk_tokens"],
                                "side_gradient_checkpointing_min_tokens": variables["side_gradient_checkpointing_min_tokens"],
                                "target_gradient_checkpointing_min_tokens": variables["target_gradient_checkpointing_min_tokens"]}
            kernel_scope.enter_context(configure_fla_runtime(variables["fla_config"]))
            manager.ensure_lora(target_lora)
            device = ac_model.prepare()
            ac_model.set_mode(TRAINING)
            bases = list(dict.fromkeys([target.model, ac_model.side.model]))
            for base in bases:
                base.train()
                disable_dropout(base)
            self._target_gradient_checkpointing_min_tokens = variables["target_gradient_checkpointing_min_tokens"]
            ac_model._side_gradient_checkpointing_min_tokens = variables["side_gradient_checkpointing_min_tokens"]
            ac_model._training_gradient_checkpointing = config.gradient_checkpointing
            self._check_capacity(ac_model, [*examples, *reporting_examples, *validation_examples])
            self._payload_bytes = 0
            if config.loss_kind == "kl":
                # The teacher of this call is the reader as it is now; nothing after this line recomputes it.
                if reporter is not None:
                    reporter.set_status(phase="teacher targets")
                    reporter.render(force=True)
                stats.teacher_targets = self._store_teacher_targets(ac_model, [*examples, *reporting_examples, *validation_examples], device, reporter)
                if reporter is not None:
                    reporter.report_teacher_targets(stats.teacher_targets)
            penalty_examples = [example.penalty for example in examples if example.penalty is not None]
            if penalty_examples:
                # The penalty's reference is the base with no adapter on the no-context history: fixed for the whole call, and kept next to
                # the checkpoints so a resumed run skips the reference pass.
                if reporter is not None:
                    reporter.set_status(phase="no-context reference")
                    reporter.render(force=True)
                reference_path = resolve_path(f"AC_MODELS/{ac_model.name}/no_context_reference.pt", create=False)
                if reference_path.exists():
                    loaded = self._merge_teacher_targets(reference_path, {example.teacher_cache_key for example in penalty_examples})
                    print(f"AC resume: {loaded} no-context references loaded from {reference_path}", flush=True)
                stats.no_context_reference = self._store_teacher_targets(ac_model, penalty_examples, device, reporter, reporting_name="no-context reference", lora=None)
                if stats.no_context_reference["examples"] or not reference_path.exists():
                    keys = {example.teacher_cache_key for example in penalty_examples}
                    reference_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"top_k": config.teacher_cache_top_k, "targets": {k: v for k, v in self.teacher_cache.items() if k in keys},
                                "target_logp": {k: v for k, v in self.teacher_target_logp.items() if k in keys}}, reference_path)
            ac_parameters = ac_model.trainable_parameters()
            target_parameters = manager.lora_parameters(target_lora)
            assert ac_parameters and target_parameters
            gate_parameters = [parameter for parameter in ac_parameters if parameter.numel() == 1]     # the scalar gates (row scale, input / position / residual scales): traced
            gate_names = {id(parameter): name for name, parameter in ac_model.modules.named_parameters() if parameter.numel() == 1}
            groups = [{"params": ac_parameters, "lr": config.learning_rate_ac}, {"params": target_parameters, "lr": config.learning_rate_target_lora}]
            if reset_optimizer:
                self.optimizers.pop(ac_model_name, None)
            optimizer = persistent_optimizer(self.optimizers, ac_model_name, groups,
                    device, betas=config.adam_betas, eps=config.adam_eps, weight_decay=config.weight_decay)
            if config.resume_optimizer and ac_model_name not in self.resumed and ac_model.config.checkpoint_path:
                saved = Path(ac_model.config.checkpoint_path) / "optimizer.pt"
                if saved.exists():
                    state = torch.load(saved, map_location="cpu", weights_only=False)
                    saved_layout = [len(group["params"]) for group in state["optimizer"]["param_groups"]]
                    current_layout = [len(group["params"]) for group in groups]
                    if saved_layout == current_layout:
                        optimizer.load_state_dict(state["optimizer"])
                        for group, wanted in zip(optimizer.param_groups, groups):   # load_state_dict also restores the saved hyper-parameters: the arm's rates win
                            group["lr"] = group["base_lr"] = wanted["lr"]
                            group["betas"] = tuple(config.adam_betas); group["eps"] = config.adam_eps; group["weight_decay"] = config.weight_decay
                        optimizer_state_to(optimizer, device)
                        stats.resumed_from = str(saved)
                    else:
                        # The group layout changed since the checkpoint (a checkpoint from the full-featured branch with extra groups, a migrated
                        # checkpoint): the moments cannot be mapped, so the optimizer starts fresh and only the update clock resumes. A requeued run never dies here.
                        print(f"AC resume: optimizer state discarded (layout changed: saved groups {saved_layout}, current {current_layout}); "
                              f"fresh moments, update clock resumed at {int(state['updates_done'])}", flush=True)
                        stats.resumed_from = f"{saved} (moments discarded: layout changed)"
                    self.updates_done[ac_model_name] = int(state["updates_done"])
                    self.resumed.add(ac_model_name)
                    print(f"AC resume: optimizer moments and {state['updates_done']} updates restored from {saved}", flush=True)
            if device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True   # fp32 GEMMs (LoRA path, gates) on tensor cores
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            stats.placement_seconds = time.monotonic() - placement_started
            baseline = ActivationContextTrainingProgress(0, config.num_epochs, start_epoch - 1, 0, steps,
                        self.updates_done.get(ac_model_name, 0), 0.0, time.monotonic() - started)
            phase = time.monotonic()
            if reporting_data:
                stats.reporting = self._evaluate(ac_model, list(reporting_data), device, reporter, "reporting")
                if reporter is not None:
                    reporter.report_eval(baseline, "reporting", stats.reporting)
            deltas_off_panel = config.deltas_off_panel and ac_model.is_deep and bool(reporting_data)
            if deltas_off_panel:
                stats.deltas_off = self._evaluate(ac_model, list(reporting_data), device, reporter, "deltas_off", deltas_off=True)
                if reporter is not None:
                    reporter.report_eval(baseline, "deltas_off", stats.deltas_off)
            if validation_data and list(validation_data) != list(reporting_data):
                stats.validation_baseline = self._evaluate(ac_model, list(validation_data), device, reporter, "validation")
                if reporter is not None:
                    reporter.report_eval(baseline, "validation", stats.validation_baseline)
            stats.baseline_samples = self._sample_completions(ac_model, fixed_items, baseline)
            stats.baseline_seconds = time.monotonic() - phase
            if reporter is not None and stats.baseline_samples:
                reporter.report_completions(baseline, stats.baseline_samples)
            planned_total = config.schedule_total_updates or (self.updates_done.get(ac_model_name, 0) + steps * config.num_epochs)
            reader_frozen = config.learning_rate_target_lora == 0.0     # a frozen reader neither steps nor takes part in the clipping norm
            for epoch in range(1, config.num_epochs + 1):
                epoch_number = start_epoch + epoch - 1
                epoch_started = time.monotonic()
                def progress(step: int) -> ActivationContextTrainingProgress:
                    return ActivationContextTrainingProgress(epoch, config.num_epochs, epoch_number, step, steps,
                        self.updates_done.get(ac_model_name, 0), time.monotonic() - epoch_started, time.monotonic() - started)
                record = ActivationContextEpochStats(progress(0))
                order = list(range(len(examples)))
                random.Random(config.seed + epoch_number).shuffle(order)
                for step in range(1, steps + 1):
                    if epoch == 1 and step <= config.resume_skip_steps:
                        continue        # trained by the previous attempt (same seeded order); the mid-epoch checkpoint carries the moments and the update count
                    if reporter is not None:
                        reporter.report_phase(progress(step - 1), "training")
                    step_started = time.monotonic()
                    before_side_tokens = ac_model.stats.side_tokens
                    mini = order[(step - 1) * per_step:step * per_step]
                    optimizer.zero_grad(set_to_none=True)
                    totals = {"loss": 0.0, "agreement": 0.0, "tokens": 0, "penalty": []}
                    batched = config.micro_batch_examples > 1 and config.loss_kind == "sft"
                    if batched:
                        mini = sorted(mini, key=lambda index: len(examples[index].student_ids))   # same examples, less padding per micro-batch
                        groups = [mini[i:i + config.micro_batch_examples] for i in range(0, len(mini), config.micro_batch_examples)]
                    else:
                        groups = [[index] for index in mini]
                    for group in groups:
                        group_examples = [examples[index] for index in group]
                        example_started = time.monotonic()
                        if batched:
                            outcomes = [(result["objective"], None, result["penalty"]) for result in self._batch_loss(ac_model, group_examples, device)]
                        else:
                            objective, agreement, _, _ = self._example_loss(ac_model, group_examples[0], device)
                            outcomes = [(objective, agreement, self._penalty_term)]
                        loss = sum((objective + (penalty_term if penalty_term is not None else 0.0)) * (example.item.weight / len(mini))
                                   for example, (objective, _, penalty_term) in zip(group_examples, outcomes))
                        loss.backward()
                        seconds_each = (time.monotonic() - example_started) / len(group)
                        for example, (objective, agreement, penalty_term) in zip(group_examples, outcomes):
                            cached = config.loss_kind == "kl" and example.teacher_cache_key in self.teacher_cache
                            if penalty_term is not None:
                                totals["penalty"].append(float(penalty_term.detach()) * config.no_context_penalty_fraction / config.no_context_penalty_weight)
                            loss_value = float(objective.detach())
                            stats.example_records.append({"kind": example.item.kind, "teacher_tokens": len(example.teacher_ids),
                                "student_tokens": len(example.student_ids), "seconds": round(seconds_each, 3), "teacher_cached": cached})
                            stats.teacher_cache_hits += int(cached)
                            stats.loss_by_kind.setdefault(example.item.kind, []).append(loss_value)
                            totals["loss"] += loss_value * example.item.weight
                            if config.loss_kind == "kl":
                                stats.kl_by_kind.setdefault(example.item.kind, []).append(loss_value)
                                stats.agreement_by_kind.setdefault(example.item.kind, []).append(agreement)
                                totals["agreement"] += agreement
                            totals["tokens"] += len(example.teacher_ids) + len(example.student_ids)
                    stats.penalty_counts.append(len(totals["penalty"]))
                    if reader_frozen:
                        for parameter in target_parameters:
                            parameter.grad = None
                    norm_ac = self._grad_norm(ac_parameters); norm_reader = self._grad_norm(target_parameters)
                    head_norms = self._head_grad_norms(ac_model) if ac_model.is_deep else None       # before clipping, like norm_ac
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(ac_parameters + ([] if reader_frozen else target_parameters), config.max_grad_norm))
                    set_learning_rates(optimizer, self.updates_done.get(ac_model_name, 0), config.warmup_updates,
                                       total_updates=planned_total, schedule=config.lr_schedule, final_fraction=config.lr_final_fraction)
                    stats.grad_norm_ac.append(norm_ac); stats.grad_norm_reader.append(norm_reader)
                    if self.updates_done.get(ac_model_name, 0) % 25 == 0:
                        stats.gate_trace.append({"update": self.updates_done.get(ac_model_name, 0), **{gate_names.get(id(parameter), str(index)): float(parameter.detach()) for index, parameter in enumerate(gate_parameters)}})
                    if ac_model.is_deep and self._intervention_trace_due(self.updates_done.get(ac_model_name, 0)):
                        stats.intervention_trace.append(self._intervention_trace_entry(ac_model, self.updates_done.get(ac_model_name, 0), head_norms))
                    encoder_has_grad = any(parameter.grad is not None for parameter in ac_parameters)
                    try:
                        optimizer.step()
                    finally:
                        if encoder_has_grad:
                            ac_model.invalidate_encoded_rows()  # also on a partially failed update; independent of saving
                    self.updates_done[ac_model_name] = self.updates_done.get(ac_model_name, 0) + 1
                    seconds = time.monotonic() - step_started
                    record.training_seconds += seconds
                    stats.steps += 1
                    stats.loss.append(totals["loss"] / len(mini))
                    if config.loss_kind == "kl":
                        stats.kl.append(stats.loss[-1])
                        stats.agreement.append(totals["agreement"] / len(mini))
                    if config.penalty_active:
                        stats.penalty.append(sum(totals["penalty"]) / len(totals["penalty"]) if totals["penalty"] else (stats.penalty[-1] if stats.penalty else 0.0))
                    stats.grad_norm.append(grad_norm)
                    stats.step_seconds.append(seconds)
                    stats.step_tokens.append(totals["tokens"])
                    stats.side_tokens += ac_model.stats.side_tokens - before_side_tokens
                    stats.global_steps.append(self.updates_done[ac_model_name])
                    stats.learning_rates.append([float(group["lr"]) for group in optimizer.param_groups])
                    print(f"AC epoch {epoch}/{config.num_epochs} step {step}/{steps} global {stats.global_steps[-1]}: "
                          f"{config.loss_kind} loss {stats.loss[-1]:.4f}, {seconds:.1f}s", flush=True)
                    if reporter is not None:
                        reporter.report_step(progress(step), stats)
                    if config.checkpoint_every_updates and config.checkpoint_every_epoch and step < steps and step % config.checkpoint_every_updates == 0:
                        self._save_partial(ac_model, epoch_number, step, steps)
                    if step in boundaries and reporting_data:
                        phase = time.monotonic()
                        if reporter is not None:
                            reporter.report_phase(progress(step), "reporting")
                        record.reporting = self._evaluate(ac_model, list(reporting_data), device, reporter, "reporting")
                        stats.reporting = record.reporting
                        if reporter is not None:
                            reporter.report_eval(progress(step), "reporting", record.reporting)
                        if deltas_off_panel:
                            record.deltas_off = self._evaluate(ac_model, list(reporting_data), device, reporter, "deltas_off", deltas_off=True)
                            stats.deltas_off = record.deltas_off
                            if reporter is not None:
                                reporter.report_eval(progress(step), "deltas_off", record.deltas_off)
                        record.reporting_seconds += time.monotonic() - phase
                if validation_data:
                    phase = time.monotonic()
                    if reporter is not None:
                        reporter.report_phase(progress(steps), "validation")
                    record.validation = (record.reporting if list(validation_data) == list(reporting_data) and record.reporting is not None
                                         else self._evaluate(ac_model, list(validation_data), device, reporter, "validation"))
                    record.validation_seconds = time.monotonic() - phase
                    if reporter is not None:
                        reporter.report_eval(progress(steps), "validation", record.validation)
                phase = time.monotonic()
                if reporter is not None:
                    reporter.report_phase(progress(steps), "samples")
                record.samples = self._sample_completions(ac_model, fixed_items, progress(steps))
                record.sample_seconds = time.monotonic() - phase
                if reporter is not None and record.samples:
                    reporter.report_completions(progress(steps), record.samples)
                if config.checkpoint_every_epoch:
                    phase = time.monotonic()
                    if reporter is not None:
                        reporter.report_phase(progress(steps), "checkpoint")
                    record.checkpoint_path, record.target_lora_checkpoint_path = self._save_epoch(ac_model, epoch_number)
                    record.checkpoint_seconds = time.monotonic() - phase
                    stats.checkpoint_path = record.checkpoint_path
                    stats.target_lora_checkpoint_path = record.target_lora_checkpoint_path
                record.progress = progress(steps)
                if reporter is not None:
                    reporter.report_epoch(record, stats)
                stats.epochs.append(record)
                self.epochs_done[ac_model_name] = epoch_number
                ac_model.epoch_number = epoch_number
            stats.teacher_cache_bytes = sum(sum(tensor.numel() * tensor.element_size() for tensor in entry) for entry in self.teacher_cache.values())
            stats.payload_bytes = self._payload_bytes
        finally:
            kernel_scope.close()
            cleanup_started = time.monotonic()
            if optimizer is not None:
                optimizer_state_to(optimizer, SOURCE_DEVICE)
            for base in bases:
                base.eval()
                if base.is_gradient_checkpointing:
                    base.gradient_checkpointing_disable()
            ac_model.__dict__.pop("_training_gradient_checkpointing", None)
            ac_model.__dict__.pop("_side_gradient_checkpointing_min_tokens", None)
            ac_model.set_mode(ROLLOUT)
            if device is not None and device.type == "cuda":
                stats.peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
            ac_model.release()
            target.model_to_device(SOURCE_DEVICE)
            if ac_model.side is not target:
                ac_model.side.model_to_device(SOURCE_DEVICE)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        stats.cleanup_seconds = time.monotonic() - cleanup_started
        stats.duration_s = time.monotonic() - started
        measured = stats.placement_seconds + stats.cleanup_seconds + stats.baseline_seconds
        measured += sum(e.training_seconds + e.reporting_seconds + e.validation_seconds + e.sample_seconds + e.checkpoint_seconds for e in stats.epochs)
        stats.other_seconds = max(0.0, stats.duration_s - measured)
        if reporter is not None:
            reporter.report_training(stats)
        return stats

    def _next_epoch_number(self, ac_model: ActivationContextModel) -> int:
        manager = self.harness.module_manager
        roots = [ac_model.checkpoint_folder().parent, manager.checkpoint_folder(ac_model.config.target_model_lora_name).parent]
        numbers = [self.epochs_done.get(ac_model.name, 0), ac_model.epoch_number]
        for root in roots:
            numbers.extend(int(path.name[6:]) for path in root.glob("epoch_*") if path.name[6:].isdigit())
        return max(numbers) + 1

    def _save_epoch(self, ac_model: ActivationContextModel, epoch_number: int) -> tuple[str, str]:
        manager = self.harness.module_manager
        target_lora = ac_model.config.target_model_lora_name
        leaf = f"epoch_{epoch_number:03d}"
        ac_relative = f"AC_MODELS/{ac_model.name}/{leaf}"
        target_relative = f"LORAS/{target_lora}/{leaf}"
        ac_folder, target_folder = resolve_path(ac_relative, create=False), resolve_path(target_relative, create=False)
        if ac_folder.exists() or target_folder.exists():
            raise FileExistsError(f"Epoch {epoch_number} already has checkpoint files")
        ac_path = ac_model.save(str(ac_folder), epoch_number=epoch_number)
        target_path = manager.save_lora(ac_model.target.model_config.model_name, target_lora, str(target_folder))
        if self.teacher_cache and self.config.loss_kind == "kl":
            self.save_teacher_targets(ac_folder / "teacher_targets.pt")     # the reference this run was trained against (an SFT run's cache only holds penalty references)
        for source, latest in ((ac_folder, ac_model.checkpoint_folder()), (target_folder, manager.checkpoint_folder(target_lora))):
            if latest.exists():
                shutil.rmtree(latest)
            shutil.copytree(source, latest)
        optimizer = self.optimizers.get(ac_model.name)
        if optimizer is not None:               # resume material: moments + the update count (the LR schedule and the reader freeze window read it)
            torch.save({"optimizer": optimizer.state_dict(), "updates_done": self.updates_done.get(ac_model.name, 0), "epoch_number": epoch_number},
                       ac_folder / "optimizer.pt")
            for previous in sorted(ac_folder.parent.glob("epoch_*/optimizer.pt")):
                if previous.parent != ac_folder:
                    previous.unlink()           # only the newest epoch keeps its optimizer (about 2x the adapter size)
        for root in (ac_folder.parent, target_folder.parent):
            for partial in root.glob(f"epoch_{epoch_number:03d}_step_*"):
                shutil.rmtree(partial)      # the completed epoch supersedes its mid-epoch checkpoints
        partial_record = ac_folder.parent / "partial_epoch.json"
        if partial_record.exists() and json.loads(partial_record.read_text())["epoch_number"] <= epoch_number:
            partial_record.unlink()
        record = ac_folder.parent / "completed_epoch.json"
        temporary = record.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"epoch_number": epoch_number, "ac_checkpoint": ac_relative,
                                        "target_lora_checkpoint": target_relative}, indent=2) + "\n")
        temporary.replace(record)       # only this record declares the numbered pair complete
        return ac_path, target_path

    def _save_partial(self, ac_model: ActivationContextModel, epoch_number: int, step: int, steps: int) -> None:
        """Mid-epoch checkpoint: adapters as of `step` of `epoch_number` (saved with epoch_number - 1 so a resume restarts this epoch), optimizer.pt,
        and AC_MODELS/<name>/partial_epoch.json; earlier partials of the same epoch are removed."""
        manager = self.harness.module_manager
        target_lora = ac_model.config.target_model_lora_name
        leaf = f"epoch_{epoch_number:03d}_step_{step:05d}"
        ac_relative, target_relative = f"AC_MODELS/{ac_model.name}/{leaf}", f"LORAS/{target_lora}/{leaf}"
        ac_folder, target_folder = resolve_path(ac_relative, create=False), resolve_path(target_relative, create=False)
        for folder in (ac_folder, target_folder):
            if folder.exists():
                shutil.rmtree(folder)
        ac_model.save(str(ac_folder), epoch_number=epoch_number - 1)
        manager.save_lora(ac_model.target.model_config.model_name, target_lora, str(target_folder))
        optimizer = self.optimizers.get(ac_model.name)
        if optimizer is not None:
            torch.save({"optimizer": optimizer.state_dict(), "updates_done": self.updates_done.get(ac_model.name, 0), "epoch_number": epoch_number - 1, "step": step},
                       ac_folder / "optimizer.pt")
        record = ac_folder.parent / "partial_epoch.json"
        temporary = record.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"epoch_number": epoch_number, "step": step, "steps": steps, "ac_checkpoint": ac_relative,
                                        "target_lora_checkpoint": target_relative}, indent=2) + "\n")
        temporary.replace(record)
        for root in (ac_folder.parent, target_folder.parent):
            for previous in root.glob(f"epoch_{epoch_number:03d}_step_*"):
                if previous.name != leaf:
                    shutil.rmtree(previous)
        print(f"AC checkpoint: epoch {epoch_number} step {step}/{steps} saved to {ac_folder}", flush=True)

    def _merge_teacher_targets(self, path: Path, keys: set | None) -> int:
        """Add the saved reference entries for `keys` (None: all) that are not stored yet (same top-k required). Returns how many were added."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data["top_k"] != self.config.teacher_cache_top_k:
            return 0
        added = 0
        for key, value in data["targets"].items():
            if (keys is None or key in keys) and key not in self.teacher_cache:
                self.teacher_cache[key] = value; self.teacher_target_logp[key] = data["target_logp"][key]; added += 1
        return added

    @torch.no_grad()
    def _sample_completions(self, ac_model: ActivationContextModel, items: list[ActivationContextTrainingItem],
                            progress: ActivationContextTrainingProgress) -> list[CompletionSample]:
        if not items:
            return []
        target = ac_model.target
        manager = self.harness.module_manager
        bases = list(dict.fromkeys([target.model, ac_model.side.model]))
        modes = [(base.training, base.is_gradient_checkpointing) for base in bases]
        previous_mode = ac_model.mode
        result = []
        model_config = target.model.config.get_text_config() if hasattr(target.model.config, "get_text_config") else target.model.config
        capacity = getattr(model_config, "max_position_embeddings", None)
        try:
            ac_model.set_mode(TRAINING)
            for base in bases:
                base.eval()
                if base.is_gradient_checkpointing:
                    base.gradient_checkpointing_disable()
            for item in items:
                example = self.build_example(ac_model, item)
                texts = {}
                # The stored teacher is the reader at baseline: its completion is sampled once (before any update) and reused.
                teacher_now = self.config.loss_kind == "kl" and item.item_id not in self._teacher_texts
                for side in (("teacher", "student") if teacher_now else ("student",)):
                    sequence = example.teacher_ids if side == "teacher" else example.student_ids
                    positions = example.teacher_positions if side == "teacher" else example.student_positions
                    prefix_length = positions[0] + 1
                    ids = torch.tensor(sequence[:prefix_length], device=ac_model.device)
                    embeds = target.model.get_input_embeddings()(ids)
                    payload = None
                    if side == "student" and example.part_requests:
                        present = [(span, request) for span, request in zip(example.spans, example.part_requests) if span[1] <= prefix_length]
                        outputs = ac_model.encode_batch([request for _, request in present], with_layers=True) if present else []
                        embeds, payload = self._assemble_student(embeds, [span for span, _ in present], outputs, embeds.dtype)
                    lora = ac_model.config.target_model_lora_name
                    peft_model = manager.ensure_lora(lora)
                    budget = min(self.config.completion_max_new_tokens, capacity - prefix_length) if capacity else self.config.completion_max_new_tokens
                    generation = dict(max_new_tokens=budget, do_sample=False, use_cache=True,
                                      pad_token_id=target.tokenizer.pad_token_id if target.tokenizer.pad_token_id is not None else target.tokenizer.eos_token_id)
                    with manager.lora_context(target.model_config.model_name, lora):
                        if payload is not None:
                            output = generate_with_interventions(peft_model, inputs_embeds=embeds[None], interventions=[payload], **generation)
                        else:
                            output = peft_model.generate(inputs_embeds=embeds[None], **generation)
                    texts[side] = target.tokenizer.decode(output[0], skip_special_tokens=True)
                if "teacher" in texts:
                    self._teacher_texts[item.item_id] = texts["teacher"]
                result.append(CompletionSample(item_id=item.item_id, kind=item.kind,
                    reference=target.tokenizer.decode(item.assistant_token_ids[0], skip_special_tokens=False),
                    teacher=self._teacher_texts.get(item.item_id) if self.config.loss_kind == "kl" else None,
                    student=texts["student"], epoch_number=progress.epoch_number,
                    global_step=progress.global_step, ac_version=ac_model.version, target_lora=ac_model.config.target_model_lora_name))
        finally:
            for base, (training, checkpointed) in zip(bases, modes):
                base.train(training)
                disable_dropout(base)
                if checkpointed:
                    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            ac_model.set_mode(previous_mode)
        return result

    def eval(
        self,
        ac_model_name: str,
        items: list[ActivationContextTrainingItem],
        release: bool = True,
        reporting_name: str | None = None,
        reporter: "ActivationContextTrainingReporter | None" = None,
        progress: ActivationContextTrainingProgress | None = None,
        *,
        reference_name: ReferenceName | None = None,
        reference_provenance: str = "initial adapter; held set",
        teacher: str = "stored",
    ) -> ActivationContextEvalSummary:
        """
        Loss, agreement and target log-probs of the items without gradient (models placed as for training, moved back
        when `release`). KL items need stored teacher targets: `teacher="stored"` (default) fails when an item has none,
        so a reloaded reader is never silently made its own teacher; `teacher="current"` stores missing targets from the
        reader as it is now (right before training, or after `load_teacher_targets`, that is the intended reference).
        A supplied reporter receives this evaluation under reporting_name.
        """
        if teacher not in ("stored", "current"):
            raise ValueError("teacher must be 'stored' or 'current'")
        module_manager = self.harness.module_manager
        ac_model = module_manager.get_ac_model(ac_model_name)
        target = ac_model.target
        if target.vllm_model is not None:
            target.engine_to_device(SOURCE_DEVICE)
        variables = self._autotuning_variables(ac_model)
        kernel_scope = ExitStack()
        try:
            kernel_scope.enter_context(configure_fla_runtime(variables["fla_config"]))
            module_manager.ensure_lora(ac_model.config.target_model_lora_name)
            device = ac_model.prepare()
            if teacher == "current" and self.config.loss_kind == "kl":
                self._store_teacher_targets(ac_model, [self.build_example(ac_model, item) for item in items], device, reporter)
            summary = self._evaluate(ac_model, items, device, reporter, reporting_name or reference_name or "evaluation")
            if reporter is not None:
                reporter.report_eval(progress, reporting_name or reference_name or "evaluation", summary)
                if reference_name is not None and summary["items"]:
                    reporter.report_reference(reference_name, summary["loss"], provenance=reference_provenance, loss_kind=self.config.loss_kind)
            return summary
        finally:
            kernel_scope.close()
            if release:
                ac_model.release()
                target.model_to_device(SOURCE_DEVICE)
                if ac_model.side is not target:
                    ac_model.side.model_to_device(SOURCE_DEVICE)

    # ------------------------------------------------------------------------------------------ internals
    @torch.no_grad()
    def _evaluate(self, ac_model: ActivationContextModel, items: list[ActivationContextTrainingItem], device: torch.device,
                  reporter: "ActivationContextTrainingReporter | None" = None, reporting_name: str = "evaluation", deltas_off: bool = False) -> ActivationContextEvalSummary:
        """`deltas_off`: a deep model's items scored with their rows kept and their per-layer deltas withheld (the deltas' contribution)."""
        if reporter is not None:
            reporter.report_evaluation_progress(reporting_name, 0, len(items))
        examples = [self.build_example(ac_model, item) for item in items]
        self._check_capacity(ac_model, examples)
        previous_mode = ac_model.mode
        ac_model.set_mode(TRAINING)
        bases = list(dict.fromkeys([ac_model.target.model, ac_model.side.model]))
        training_modes = [base.training for base in bases]
        for base in bases:
            base.eval()
        per_item = []
        is_kl = self.config.loss_kind == "kl"
        try:
            def row(example, loss, agreement, positions, target_logp, hits):
                teacher_logp = self.teacher_target_logp.get(example.teacher_cache_key) if is_kl else None
                return {"item_id": example.item.item_id, "kind": example.item.kind,
                    "loss_kind": self.config.loss_kind, "loss": float(loss), "kl": float(loss) if is_kl else None,
                    "agreement": agreement, "positions": positions,
                    "token_accuracy": float(hits.float().mean()) if hits is not None and hits.numel() else 0.0,
                    "teacher_tokens": len(example.teacher_ids), "student_tokens": len(example.student_ids),
                    "gold_nll": float(-target_logp.mean()), "target_logp": [round(float(value), 4) for value in target_logp],
                    "teacher_gold_nll": float(-teacher_logp.mean()) if teacher_logp is not None else None,
                    "teacher_target_logp": [round(float(value), 4) for value in teacher_logp] if teacher_logp is not None else None}
            rows: list[dict | None] = [None] * len(examples)
            done = 0
            if self.config.micro_batch_examples > 1 and not is_kl:
                order = sorted(range(len(examples)), key=lambda index: len(examples[index].student_ids))
                for start in range(0, len(order), self.config.micro_batch_examples):
                    group = order[start:start + self.config.micro_batch_examples]
                    for index, result in zip(group, self._batch_loss(ac_model, [examples[index] for index in group], device, with_penalty=False, deltas_off=deltas_off)):
                        rows[index] = row(examples[index], result["objective"], None, result["num"], result["target_logp"], result["hits"])
                    done += len(group)
                    if reporter is not None:
                        reporter.report_evaluation_progress(reporting_name, done, len(examples))
            else:
                for index, example in enumerate(examples):
                    loss, agreement, positions, target_logp = self._example_loss(ac_model, example, device, with_penalty=False, deltas_off=deltas_off)
                    rows[index] = row(example, loss, agreement, positions, target_logp, self._last_hits)
                    done += 1
                    if reporter is not None:
                        reporter.report_evaluation_progress(reporting_name, done, len(examples))
            per_item = rows
        finally:
            for base, training in zip(bases, training_modes):
                base.train(training)
                disable_dropout(base)
            ac_model.set_mode(previous_mode)
        def summarize(rows):
            loss = sum(row["loss"] for row in rows) / len(rows) if rows else None
            teacher_rows = [row["teacher_gold_nll"] for row in rows if row["teacher_gold_nll"] is not None]
            return {"loss_kind": self.config.loss_kind, "loss": loss, "kl": loss if is_kl else None,
                    "agreement": sum(row["agreement"] for row in rows) / len(rows) if is_kl and rows else None,
                    "gold_nll": sum(row["gold_nll"] for row in rows) / len(rows) if rows else None,
                    "gold_nll_token": -sum(sum(row["target_logp"]) for row in rows) / max(1, sum(len(row["target_logp"]) for row in rows)) if rows else None,   # token-weighted: long passages count by their length
                    "token_accuracy": sum(row["token_accuracy"] for row in rows) / len(rows) if rows else None,
                    "teacher_gold_nll": sum(teacher_rows) / len(teacher_rows) if teacher_rows else None,
                    "items": len(rows)}
        by_kind = {kind: summarize([row for row in per_item if row["kind"] == kind])
                   for kind in dict.fromkeys(row["kind"] for row in per_item)}
        return {**summarize(per_item), "dropped_too_long": 0, "by_kind": by_kind, "per_item": per_item}

    def _example_loss(self, ac_model: ActivationContextModel, example: _Example, device: torch.device, with_penalty: bool = True, deltas_off: bool = False) -> tuple[torch.Tensor, float | None, int, torch.Tensor]:
        """
        (loss, agreement, positions, target log-probs) of one example: the student with the rows against the stored teacher
        targets (KL) or the target tokens (SFT). With `with_penalty` (training forwards) the sampled no-context penalty term
        lands in `_penalty_term`. The last value is the student's log-prob of every target token, detached, on `device`.
        """
        config = self.config
        target = ac_model.target
        base = target.model
        base_dtype = next(base.parameters()).dtype
        embedding = base.get_input_embeddings()
        head = base.get_output_embeddings()
        lora = ac_model.config.target_model_lora_name
        num = example.num_completion
        is_kl = config.loss_kind == "kl"
        cached = None
        if is_kl:
            cached = self.teacher_cache.get(example.teacher_cache_key)
            if cached is None:
                raise RuntimeError(f"{example.item.item_id}: no stored teacher targets; train() stores them before its first update "
                                   "and eval() needs them for its items (teacher='current' stores them from the current reader)")
            cached = tuple(tensor.to(device, non_blocking=True) for tensor in cached)
        target_ids = torch.tensor(example.target_ids, device=device, dtype=torch.long)
        # Student: rows for the parts (grad), written over the placeholders; a deep model's deltas travel as one payload.
        outputs = ac_model.encode_batch(example.part_requests, with_layers=True) if example.part_requests else []
        student_ids = torch.tensor(example.student_ids, device=device)
        with torch.no_grad():
            student_embeds = embedding(student_ids)
        student_embeds, payload = self._assemble_student(student_embeds, example.spans, outputs, base_dtype, deltas_off=deltas_off)
        checkpointed = (config.gradient_checkpointing and torch.is_grad_enabled()
                        and len(example.student_ids) >= getattr(self, "_target_gradient_checkpointing_min_tokens", config.gradient_checkpointing_min_tokens or 0))
        student_hidden = target.decoder_forward(student_embeds[None], None, lora_name=lora, gradient_checkpointing=checkpointed,
                                                interventions=[payload] if payload is not None else None)[0]
        if torch.is_grad_enabled():
            self._harvest_row_rms([payload])
        student_states = student_hidden[example.student_positions]
        if is_kl:
            loss, agreement, target_logp = self._chunked_kl(None, student_states, head, cached=cached, target_ids=target_ids)
        else:
            loss, target_logp = self._chunked_sft(student_states, head, target_ids)
            agreement = None
        self._penalty_term = None
        if with_penalty and example.penalty is not None and config.penalty_active and self._penalty_rng.random() < config.no_context_penalty_fraction:
            # The reader on the same history without its rows, held to the base's stored no-context distribution: any improvement that
            # does not need the rows (the answer prior, memorizing the question) is charged; decoding the rows is free.
            reference = example.penalty
            cached_reference = self.teacher_cache.get(reference.teacher_cache_key)
            if cached_reference is None:
                raise RuntimeError(f"{reference.item.item_id}: no stored no-context reference; train() stores it before its first update")
            cached_reference = tuple(tensor.to(device, non_blocking=True) for tensor in cached_reference)
            reference_ids = torch.tensor(reference.student_ids, device=device, dtype=torch.long)
            with torch.no_grad():
                reference_embeds = embedding(reference_ids)
            reference_checkpointed = (config.gradient_checkpointing and torch.is_grad_enabled()
                                      and len(reference.student_ids) >= getattr(self, "_target_gradient_checkpointing_min_tokens", config.gradient_checkpointing_min_tokens or 0))
            hits = self._last_hits
            reference_states = target.decoder_forward(reference_embeds[None], None, lora_name=lora, gradient_checkpointing=reference_checkpointed)[0][reference.student_positions]
            penalty, _, _ = self._chunked_kl(None, reference_states, head, cached=cached_reference)
            self._last_hits = hits
            self._penalty_term = (config.no_context_penalty_weight / config.no_context_penalty_fraction) * penalty
        return loss, agreement, num, target_logp

    def _batch_loss(self, ac_model: ActivationContextModel, batch: t.Sequence[_Example], device: torch.device, with_penalty: bool = True, deltas_off: bool = False) -> list[dict]:
        """
        SFT loss of several examples from one right-padded reader forward (and one encoder call for all their parts):
        per example {"objective", "num", "target_logp", "hits", "penalty"}, the values `_example_loss` gives one at a
        time. Under causal attention (and the causal recurrence of the linear-attention layers) a real token never sees
        a later pad, so the padded rows need no mask. The sampled no-context references go through a second padded
        forward; `penalty` is the scaled term for the sampled examples and None for the rest.
        """
        config = self.config
        assert config.loss_kind == "sft", "_batch_loss is the SFT path"
        target = ac_model.target
        base = target.model
        base_dtype = next(base.parameters()).dtype
        embedding = base.get_input_embeddings()
        head = base.get_output_embeddings()
        lora = ac_model.config.target_model_lora_name
        requests = [request for example in batch for request in example.part_requests]
        outputs = ac_model.encode_batch(requests, with_layers=True) if requests else []
        embeds, payloads, cursor_rows = [], [], 0
        for example in batch:
            student_ids = torch.tensor(example.student_ids, device=device)
            with torch.no_grad():
                student_embeds = embedding(student_ids)
            example_outputs = outputs[cursor_rows:cursor_rows + len(example.part_requests)]
            cursor_rows += len(example.part_requests)
            student_embeds, payload = self._assemble_student(student_embeds, example.spans, example_outputs, base_dtype, deltas_off=deltas_off)
            embeds.append(student_embeds)
            payloads.append(payload)
        hidden = self._padded_forward(target, embeds, lora, payloads)
        if torch.is_grad_enabled():
            self._harvest_row_rms(payloads)
        states = torch.cat([hidden[index, example.student_positions] for index, example in enumerate(batch)], dim=0)
        targets = torch.cat([torch.tensor(example.target_ids, device=device, dtype=torch.long) for example in batch])
        cross_entropy, token_logp, hits = self._chunked_sft_tokens(states, head, targets)
        results, cursor = [], 0
        for example in batch:
            num = example.num_completion
            results.append({"objective": cross_entropy[cursor:cursor + num].mean(), "num": num,
                            "target_logp": token_logp[cursor:cursor + num], "hits": hits[cursor:cursor + num], "penalty": None})
            cursor += num
        if with_penalty and config.penalty_active:
            sampled = [index for index, example in enumerate(batch) if example.penalty is not None and self._penalty_rng.random() < config.no_context_penalty_fraction]
            if sampled:
                references = [batch[index].penalty for index in sampled]
                cached = []
                for reference in references:
                    entry = self.teacher_cache.get(reference.teacher_cache_key)
                    if entry is None:
                        raise RuntimeError(f"{reference.item.item_id}: no stored no-context reference; train() stores it before its first update")
                    cached.append(tuple(tensor.to(device, non_blocking=True) for tensor in entry))
                with torch.no_grad():
                    reference_embeds = [embedding(torch.tensor(reference.student_ids, device=device, dtype=torch.long)) for reference in references]
                reference_hidden = self._padded_forward(target, reference_embeds, lora)
                for position, (index, reference) in enumerate(zip(sampled, references)):
                    penalty, _, _ = self._chunked_kl(None, reference_hidden[position, reference.student_positions], head, cached=cached[position])
                    results[index]["penalty"] = (config.no_context_penalty_weight / config.no_context_penalty_fraction) * penalty
        return results

    def _padded_forward(self, target, embeds: list[torch.Tensor], lora: str | None, payloads: list[Interventions | None] | None = None) -> torch.Tensor:
        """Last hidden states [B, L, d] of right-padded (zero embedding) rows; no mask, the pads sit after every real token.
        Right padding keeps every payload's positions physical, so the per-row payloads pass through unchanged."""
        length = max(int(item.shape[0]) for item in embeds)
        padded = torch.stack([torch.cat([item, item.new_zeros((length - item.shape[0], item.shape[1]))]) if item.shape[0] < length else item for item in embeds])
        checkpointed = (self.config.gradient_checkpointing and torch.is_grad_enabled()
                        and length * len(embeds) >= getattr(self, "_target_gradient_checkpointing_min_tokens", self.config.gradient_checkpointing_min_tokens or 0))   # the threshold counts the whole batch
        interventions = list(payloads) if payloads is not None and any(payload is not None for payload in payloads) else None
        return target.decoder_forward(padded, None, lora_name=lora, gradient_checkpointing=checkpointed, interventions=interventions)

    def _assemble_student(self, student_embeds: torch.Tensor, spans: list[tuple[int, int]], outputs: list[ACOutput], dtype: torch.dtype,
                          deltas_off: bool = False) -> tuple[torch.Tensor, Interventions | None]:
        """The rows written over the placeholder runs, and (deep models) one payload: the runs' positions in order with every
        part's deltas concatenated per layer. `deltas_off` keeps the rows and withholds the payload."""
        if not outputs:
            return student_embeds, None
        assert [int(output.input_embeds.shape[0]) for output in outputs] == [end - start for start, end in spans], "placeholder runs differ from the encoded rows"
        pieces, cursor = [], 0
        for (start, end), output in zip(spans, outputs):
            pieces.extend([student_embeds[cursor:start], output.input_embeds.to(dtype)])
            cursor = end
        pieces.append(student_embeds[cursor:])
        embeds = torch.cat(pieces, dim=0)
        if deltas_off or not outputs[0].has_interventions:
            return embeds, None
        positions = [position for start, end in spans for position in range(start, end)]
        layer_inputs = {layer: torch.cat([output.layer_inputs[layer] for output in outputs], dim=0) for layer in outputs[0].layer_inputs}
        if torch.is_grad_enabled():                                                          # training forwards only: the panels and the samples run under no_grad
            self._payload_bytes += sum(rows.numel() * 2 for rows in layer_inputs.values())  # bf16 at the reader boundary
        return embeds, {"positions": positions, "layer_inputs": layer_inputs}

    @staticmethod
    def _intervention_trace_due(update: int) -> bool:
        """Every 5 updates for the first 200 (a 50-update scale collapse is ten points, not two), every 25 afterwards."""
        return update % (5 if update < 200 else 25) == 0

    def _head_grad_norms(self, ac_model: ActivationContextModel) -> dict:
        """Per intervention layer, the norm of the head's gradient as it stands when called (before clipping), and the read-out
        block's under "attn:<layer>", on the same footing."""
        modules = ac_model.modules
        norms = {layer: self._grad_norm(list(modules.delta_heads[str(layer)].parameters())) for layer in ac_model.intervention_layers}
        norms.update({f"attn:{layer}": self._grad_norm(list(modules.passage_attention[str(layer)].parameters())) for layer in ac_model.intervention_layers})
        norms.update({"attn:row": self._grad_norm(list(modules.row_readout.parameters())), "attn:recursive": self._grad_norm(list(modules.recursive_readout.parameters()))})
        return norms

    def _intervention_trace_entry(self, ac_model: ActivationContextModel, update: int, head_norms: dict | None = None) -> dict:
        """Per layer: the head's gradient norm before clipping (`head_norms`, taken next to the encoder norm; when omitted, whatever the
        gradients hold now), the residual RMS at the row positions on the training forwards since the last trace, `delta_rms`: the RMS
        of the applied delta relative to that residual RMS (`delta_rms_abs` the absolute value), the delta / input-row cosine, and the
        read-out block's output-projection weight norm (0 at init; rising = the attention path opens) with its gradient norm."""
        modules = ac_model.modules
        head_norms = self._head_grad_norms(ac_model) if head_norms is None else head_norms
        entry = {"update": update, "head_grad_norm": {}, "row_rms": {}, "cosine": {}, "attention_out_norm": {}, "attention_grad_norm": {}, "delta_rms": {}, "delta_rms_abs": {}}
        def mean_of(values) -> float | None:
            return float(torch.stack([torch.as_tensor(value, dtype=torch.float32) for value in values]).mean()) if values else None
        for layer in ac_model.intervention_layers:
            entry["attention_out_norm"][layer] = float(modules.passage_attention[str(layer)].out.weight.detach().norm())
            entry["attention_grad_norm"][layer] = head_norms[f"attn:{layer}"]
            entry["head_grad_norm"][layer] = head_norms[layer]
            entry["row_rms"][layer] = mean_of(self._row_rms.pop(layer, []))                 # 0-d tensors, converted here (one host sync per trace point, not per forward)
            entry["cosine"][layer] = mean_of(ac_model.intervention_diagnostics["cosine"].pop(layer, []))
            applied = mean_of(ac_model.intervention_diagnostics["delta_rms"].pop(layer, []))
            entry["delta_rms_abs"][layer] = applied
            entry["delta_rms"][layer] = applied / entry["row_rms"][layer] if applied is not None and entry["row_rms"][layer] else None
        for name in ("row", "recursive"):                                                       # the read-outs in front of the row heads, on the same footing
            entry["attention_out_norm"][name] = float(getattr(modules, f"{name}_readout").out.weight.detach().norm())
            entry["attention_grad_norm"][name] = head_norms[f"attn:{name}"]
        return entry

    def _harvest_row_rms(self, payloads) -> None:
        """After a training forward: the reader hooks wrote each payload's residual RMS at its row positions per layer."""
        for payload in payloads:
            for layer, value in (payload or {}).pop("row_rms", {}).items():
                self._row_rms.setdefault(int(layer), []).append(value.detach())

    def _chunked_sft_tokens(self, states: torch.Tensor, head: torch.nn.Module, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(per-token cross-entropy [C] with grad, detached per-token log-probs [C], detached top-1 hits [C])."""
        chunk = getattr(self, "_logits_chunk_tokens", self.config.logits_chunk_tokens or 64)
        def cross_entropy(hidden, labels):
            logp = F.log_softmax(head_logits(hidden, head, self.config.fp32_head_matmul), dim=-1)
            token_logp = logp.gather(-1, labels[:, None])[:, 0]
            return -token_logp, token_logp.detach(), (logp.argmax(dim=-1) == labels).detach()
        losses, logps, hits = [], [], []
        for start in range(0, targets.shape[0], chunk):
            arguments = states[start:start + chunk], targets[start:start + chunk]
            token_loss, token_logp, chunk_hits = (checkpoint(cross_entropy, *arguments, use_reentrant=False)
                                                  if torch.is_grad_enabled() and states.requires_grad else cross_entropy(*arguments))
            losses.append(token_loss)
            logps.append(token_logp)
            hits.append(chunk_hits)
        if not losses:
            return states.new_zeros((0,), dtype=torch.float32), targets.new_zeros((0,), dtype=torch.float32), targets.new_zeros((0,), dtype=torch.bool)
        return torch.cat(losses), torch.cat(logps), torch.cat(hits)

    def _chunked_sft(self, states: torch.Tensor, head: torch.nn.Module, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(mean cross-entropy over the target tokens, detached per-token log-probs [C])."""
        chunk = getattr(self, "_logits_chunk_tokens", self.config.logits_chunk_tokens or 64)
        def cross_entropy(hidden, labels):
            logp = F.log_softmax(head_logits(hidden, head, self.config.fp32_head_matmul), dim=-1)
            token_logp = logp.gather(-1, labels[:, None])[:, 0]
            return -token_logp.sum(), token_logp.detach(), (logp.argmax(dim=-1) == labels).detach()
        total = states.new_zeros((), dtype=torch.float32)
        logps, hits = [], []
        for start in range(0, targets.shape[0], chunk):
            arguments = states[start:start + chunk], targets[start:start + chunk]
            ce_sum, token_logp, chunk_hits = (checkpoint(cross_entropy, *arguments, use_reentrant=False)
                                              if torch.is_grad_enabled() and states.requires_grad else cross_entropy(*arguments))
            total = total + ce_sum
            logps.append(token_logp)
            hits.append(chunk_hits)
        self._last_hits = torch.cat(hits) if hits else targets.new_zeros((0,), dtype=torch.bool)
        return total / max(1, targets.shape[0]), torch.cat(logps) if logps else targets.new_zeros((0,), dtype=torch.float32)

    @torch.no_grad()
    def _store_teacher_targets(self, ac_model: ActivationContextModel, examples: t.Sequence[_Example], device: torch.device,
                               reporter: "ActivationContextTrainingReporter | None" = None, reporting_name: str = "teacher targets",
                               lora: str | None = "reader") -> dict:
        """
        Store, for every example without an entry, the reader's current top-k targets and its log-prob of each target
        token: the fixed teacher of the training call. Returns {"examples", "top_k", "coverage"}; coverage is the mean
        probability mass the stored top-k carry over all stored positions.
        """
        target = ac_model.target
        base = target.model
        embedding = base.get_input_embeddings()
        head = base.get_output_embeddings()
        lora = ac_model.config.target_model_lora_name if lora == "reader" else lora   # None: the base with no adapter (the penalty's reference)
        k = self.config.teacher_cache_top_k
        started = time.monotonic()
        pending = list({example.teacher_cache_key: example for example in examples if example.teacher_cache_key not in self.teacher_cache}.values())
        if reporter is not None:
            reporter.report_evaluation_progress(reporting_name, 0, len(pending))
        groups: list[list[_Example]] = [[example] for example in pending]
        if self.config.micro_batch_examples > 1:       # several teacher views per right-padded forward (no grad; pads never precede a real token)
            order = sorted(pending, key=lambda example: len(example.teacher_ids))
            groups = [order[start:start + self.config.micro_batch_examples] for start in range(0, len(order), self.config.micro_batch_examples)]
        done = 0
        for group in groups:
            if len(group) == 1:
                ids = torch.tensor(group[0].teacher_ids, device=device, dtype=torch.long)
                states_by_example = [target.decoder_forward(embedding(ids)[None], None, lora_name=lora, gradient_checkpointing=False)[0][group[0].teacher_positions]]
            else:
                embeds = [embedding(torch.tensor(example.teacher_ids, device=device, dtype=torch.long)) for example in group]
                hidden = self._padded_forward(target, embeds, lora)
                states_by_example = [hidden[index, example.teacher_positions] for index, example in enumerate(group)]
            for example, states in zip(group, states_by_example):
                cached = self._teacher_top_k(states, head, k)
                targets = torch.tensor(example.target_ids, device=device, dtype=torch.long)
                target_logp = self._target_logp(states, head, targets)
                cached = self._with_gold(cached, targets, target_logp)          # the gold token always among the stored top-k
                self.teacher_cache[example.teacher_cache_key] = tuple(tensor.cpu() for tensor in cached)
                self.teacher_target_logp[example.teacher_cache_key] = target_logp.cpu()
            done += len(group)
            if reporter is not None:
                reporter.report_evaluation_progress(reporting_name, done, len(pending))
        keys = list(dict.fromkeys(example.teacher_cache_key for example in examples))
        covered = [1.0 - self.teacher_cache[key][2].float().exp() for key in keys if key in self.teacher_cache]
        coverage = float(torch.cat(covered).mean()) if covered and sum(c.numel() for c in covered) else None
        return {"examples": len(pending), "top_k": k, "coverage": coverage, "seconds": round(time.monotonic() - started, 3)}

    @torch.no_grad()
    def _target_logp(self, states: torch.Tensor, head: torch.nn.Module, targets: torch.Tensor) -> torch.Tensor:
        """Log-prob of each target token [C] from the hidden states at the predicting positions."""
        chunk, fp32 = getattr(self, "_logits_chunk_tokens", self.config.logits_chunk_tokens or 64), self.config.fp32_head_matmul
        out = []
        for start in range(0, states.shape[0], chunk):
            logp = F.log_softmax(head_logits(states[start:start + chunk], head, fp32), dim=-1)
            out.append(logp.gather(-1, targets[start:start + chunk, None])[:, 0].float())
        return torch.cat(out) if out else states.new_zeros((0,), dtype=torch.float32)

    def save_teacher_targets(self, path: str | Path) -> str:
        """Write the stored teacher targets (top-k and target log-probs) so a later evaluation can use the same reference."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"top_k": self.config.teacher_cache_top_k, "targets": self.teacher_cache, "target_logp": self.teacher_target_logp}, path)
        return str(path)

    def load_teacher_targets(self, path: str | Path) -> int:
        """Replace the stored teacher targets with a saved set; returns the number of entries. Fails on a different top-k."""
        data = torch.load(Path(path), map_location="cpu")
        if data["top_k"] != self.config.teacher_cache_top_k:
            raise ValueError(f"saved targets use top_k={data['top_k']}, this trainer uses {self.config.teacher_cache_top_k}")
        self.teacher_cache = dict(data["targets"])
        self.teacher_target_logp = dict(data["target_logp"])
        return len(self.teacher_cache)

    @staticmethod
    def _grad_norm(parameters) -> float:
        grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
        return float(torch.norm(torch.stack([grad.norm() for grad in grads]))) if grads else 0.0

    @staticmethod
    def _with_gold(cached: tuple[torch.Tensor, torch.Tensor, torch.Tensor], targets: torch.Tensor, target_logp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The stored top-k with the gold token of every position guaranteed present: where it is absent, it replaces the k-th (least likely) entry."""
        ids, logps, rests = (tensor.clone() for tensor in cached)
        present = (ids == targets[:, None].to(ids.dtype)).any(dim=-1)
        missing = ~present
        if missing.any():
            ids[missing, -1] = targets[missing].to(ids.dtype)
            logps[missing, -1] = target_logp[missing].to(logps.dtype)
            rests[missing] = torch.log(torch.clamp(1.0 - logps[missing].float().exp().sum(dim=-1), min=1e-12)).to(rests.dtype)
        return ids, logps, rests

    @torch.no_grad()
    def _teacher_top_k(self, teacher_states: torch.Tensor, head: torch.nn.Module, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(ids [C, k], log-probs [C, k] descending, log remainder mass [C]) of the teacher at every completion position."""
        chunk, fp32 = getattr(self, "_logits_chunk_tokens", self.config.logits_chunk_tokens or 64), self.config.fp32_head_matmul
        ids, logps, rests = [], [], []
        for start in range(0, teacher_states.shape[0], chunk):
            logp = F.log_softmax(head_logits(teacher_states[start:start + chunk], head, fp32), dim=-1)
            values, indexes = logp.topk(min(k, logp.shape[-1]), dim=-1)
            ids.append(indexes.to(torch.int32))
            logps.append(values)
            rests.append(torch.log(torch.clamp(1.0 - values.exp().sum(dim=-1), min=1e-12)))
        return torch.cat(ids), torch.cat(logps), torch.cat(rests)

    def _chunked_kl(self, teacher_states: torch.Tensor | None, student_states: torch.Tensor, head: torch.nn.Module,
                    cached: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
                    target_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, float, torch.Tensor]:
        """
        Mean KL(teacher || student), top-1 agreement over the positions, and the student's detached log-prob of each
        target token ([C], empty when no target_ids), the vocabulary logits one chunk at a time. With `cached` (the
        teacher's top-k ids, log-probs and log remainder mass, see `_teacher_top_k`) the teacher states are not needed:
        the KL is between the k+1-way coarsened distributions (the top-k tokens and "everything else"), a lower bound
        of the exact KL that is tight when the top-k carry the mass. Live teacher states are only used by the drift term.
        """
        chunk = getattr(self, "_logits_chunk_tokens", self.config.logits_chunk_tokens or 64)
        fp32 = self.config.fp32_head_matmul
        total = torch.zeros((), device=student_states.device, dtype=torch.float32)
        agree = 0.0
        num = student_states.shape[0]
        logps = []

        hits = []

        def token_logp(student_logp: torch.Tensor, targets: torch.Tensor | None) -> torch.Tensor:
            if targets is None:
                return student_logp.new_zeros((0,))
            hits.append((student_logp.argmax(dim=-1) == targets).detach())
            return student_logp.gather(-1, targets[:, None])[:, 0].detach().float()

        def chunk_kl(student_chunk: torch.Tensor, teacher_chunk: torch.Tensor, targets: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                teacher_logp = F.log_softmax(head_logits(teacher_chunk, head, fp32), dim=-1)
            student_logp = F.log_softmax(head_logits(student_chunk, head, fp32), dim=-1)
            kl_sum = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1).sum()
            agreement = (teacher_logp.argmax(dim=-1) == student_logp.argmax(dim=-1)).float().sum()
            return kl_sum, agreement.detach(), token_logp(student_logp, targets)

        def chunk_kl_cached(student_chunk: torch.Tensor, ids: torch.Tensor, teacher_logp: torch.Tensor, teacher_rest: torch.Tensor,
                            targets: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            student_logp = F.log_softmax(head_logits(student_chunk, head, fp32), dim=-1)
            student_top = student_logp.gather(-1, ids.long())
            student_rest = torch.log(torch.clamp(1.0 - student_top.exp().sum(dim=-1), min=1e-12))
            kl_sum = ((teacher_logp.exp() * (teacher_logp - student_top)).sum(dim=-1) + teacher_rest.exp() * (teacher_rest - student_rest)).sum()
            agreement = (ids[:, 0].long() == student_logp.argmax(dim=-1)).float().sum()
            return kl_sum, agreement.detach(), token_logp(student_logp, targets)

        for start in range(0, num, chunk):
            student_chunk = student_states[start:start + chunk]
            targets = target_ids[start:start + chunk] if target_ids is not None else None
            if cached is not None:
                function, arguments = chunk_kl_cached, (student_chunk, *(tensor[start:start + chunk] for tensor in cached), targets)
            else:
                function, arguments = chunk_kl, (student_chunk, teacher_states[start:start + chunk], targets)
            if torch.is_grad_enabled() and student_chunk.requires_grad:
                kl_sum, agreement, chunk_logp = checkpoint(function, *arguments, use_reentrant=False)
            else:
                kl_sum, agreement, chunk_logp = function(*arguments)
            total = total + kl_sum
            agree += float(agreement)
            logps.append(chunk_logp)
        if target_ids is not None:
            self._last_hits = torch.cat(hits[:len(logps)]) if hits else target_ids.new_zeros((0,), dtype=torch.bool)
        return total / max(1, num), agree / max(1, num), (torch.cat(logps) if logps else student_states.new_zeros((0,), dtype=torch.float32))

    def _autotuning_variables(self, ac_model: ActivationContextModel) -> dict:
        variables = get_ac_training_autotuning_variables(ac_model.side.model_config, ac_model.target.model_config,
                        self.config, ac_config=ac_model.config, device=TARGET_DEVICE)
        self._logits_chunk_tokens = variables["logits_chunk_tokens"]
        return variables
