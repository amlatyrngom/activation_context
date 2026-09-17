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
from dataclasses import asdict, dataclass, field
from contextlib import ExitStack

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..agent_training.agent_training_utils import (
    disable_dropout, head_logits, optimizer_state_to, persistent_optimizer, set_learning_rates,
)
from ..autotuners import get_ac_training_autotuning_variables, configure_fla_runtime
from ..harness.hf_utils import SOURCE_DEVICE, TARGET_DEVICE
from ..common.data_syncing import resolve_path
from .ac_model import ROLLOUT, TRAINING, ActivationContextModel
from .ac_model_utils import (EncodeRequest, direct_parts, prepare_training_history,
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
    loss_kind: t.Literal["kl", "sft"] = "kl"
    learning_rate_ac: float = 5e-4             # AC modules and the side LoRA
    learning_rate_target_lora: float = 2e-5    # the target adapter
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_updates: int = 0                    # linear warm-up of both learning rates over this many updates (counted across calls)
    updates_per_epoch: int = 4                 # maximum gradient steps per epoch when examples_per_update is unset
    examples_per_update: int | None = None     # when set, overrides updates_per_epoch: ceil(items / examples_per_update) steps
    max_example_tokens: int = 72_000           # explicit error; no dropping or truncation
    teacher_cache_top_k: int = 0               # > 0: the teacher's top-k log-probs (+ the remainder mass) per completion position are
                                               # kept per item after its first pass and stand in for the teacher afterwards (epoch 2
                                               # skips the teacher forward); training then minimizes the KL of the k+1-way coarsened
                                               # distributions (a lower bound of the exact KL, the same loss in every epoch, the
                                               # teacher's adapter as of the item's last exact pass). Eval is always exact.
    drift_term_weight: float = 0.0
    logits_chunk_tokens: int | None = None           # completion positions per lm_head chunk
    fp32_head_matmul: bool = False
    gradient_checkpointing: bool = True        # on both bases
    gradient_checkpointing_min_tokens: int | None = None
    seed: int = 0
    checkpoint_every_epoch: bool = True        # AC_MODELS/<name>/epoch_<n> (+ latest) and LORAS/<target lora>/epoch_<n>
    num_epochs: int = 2
    reporting_interval: float = 0.1
    completion_samples: int = 4
    completion_max_new_tokens: int = 64

    def __post_init__(self) -> None:
        if self.loss_kind not in ("kl", "sft") or self.max_example_tokens < 1 or self.teacher_cache_top_k < 0:
            raise ValueError("Invalid loss kind, example capacity, or teacher cache size")
        if self.loss_kind == "sft" and (self.teacher_cache_top_k != 0 or self.drift_term_weight != 0):
            raise ValueError("SFT is incompatible with teacher_cache_top_k or drift_term_weight")
        if self.num_epochs < 1 or self.updates_per_epoch < 1 or not 0 < self.reporting_interval <= 1:
            raise ValueError("Epochs/updates must be positive and reporting_interval must be in (0, 1]")
        if self.examples_per_update is not None and self.examples_per_update < 1:
            raise ValueError("examples_per_update must be positive")
        if self.completion_samples < 0 or self.completion_max_new_tokens < 1:
            raise ValueError("Invalid completion inspection budget")


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


class EvalItemSummary(t.TypedDict):
    item_id: str
    kind: str
    loss_kind: str
    loss: float | None
    kl: float | None
    agreement: float | None
    positions: int
    teacher_tokens: int
    student_tokens: int


class EvalKindSummary(t.TypedDict):
    loss_kind: str
    loss: float | None
    kl: float | None
    agreement: float | None
    items: int


class ActivationContextEvalSummary(t.TypedDict):
    items: int
    dropped_too_long: int
    loss_kind: str
    loss: float | None
    kl: float | None
    agreement: float | None
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
    drift: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    step_seconds: list[float] = field(default_factory=list)
    step_tokens: list[int] = field(default_factory=list)        # teacher + student tokens per step
    kl_by_kind: dict[str, list[float]] = field(default_factory=dict)   # per example
    agreement_by_kind: dict[str, list[float]] = field(default_factory=dict)
    completion_tokens: int = 0
    teacher_tokens: int = 0
    student_tokens: int = 0
    side_tokens: int = 0
    reporting: ActivationContextEvalSummary | None = None               # exact reporting items at the last epoch boundary
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
            "baseline_samples": self.baseline_samples, "baseline_seconds": self.baseline_seconds, "global_steps": self.global_steps, "learning_rates": self.learning_rates, "items": self.items, "examples": self.examples, "dropped_too_long": self.dropped_too_long,
            "steps": self.steps, "kl_first": round(self.kl[0], 4) if self.kl else None, "kl_last": round(self.kl[-1], 4) if self.kl else None,
            "agreement_last": round(self.agreement[-1], 4) if self.agreement else None,
            "kl_by_kind": {kind: round(sum(values) / len(values), 4) for kind, values in self.kl_by_kind.items() if values},
            "completion_tokens": self.completion_tokens, "teacher_tokens": self.teacher_tokens, "student_tokens": self.student_tokens,
            "side_tokens": self.side_tokens, "tokens_per_second": round(tokens / seconds, 1) if seconds else None,
            "reporting": self.reporting, "checkpoint": self.checkpoint_path, "target_lora_checkpoint": self.target_lora_checkpoint_path,
            "peak_memory_gb": round(self.peak_memory_bytes / 2**30, 2), "duration_s": round(self.duration_s, 1),
            "seconds_by_bucket": self.seconds_by_bucket(), "teacher_cache_hits": self.teacher_cache_hits,
            "teacher_cache_mb": round(self.teacher_cache_bytes / 2**20, 1),
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

    @property
    def num_completion(self) -> int:
        return len(self.target_ids)


class ActivationContextTrainer:
    def __init__(self, harness: "HarnessRuntime", config: ActivationContextTrainingConfig | None = None) -> None:
        self.harness = harness
        self.config = config or ActivationContextTrainingConfig()
        self.epochs_done: dict[str, int] = {}
        self.updates_done: dict[str, int] = {}                    # optimizer steps so far per AC model (the warm-up clock)
        self.optimizers: dict[str, torch.optim.AdamW] = {}        # one per AC model; moments survive train() calls
        self.teacher_cache: dict[tuple[str, str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}   # prepared-token identity -> (ids [C, k], log-probs [C, k], log remainder [C]), CPU

    # ------------------------------------------------------------------------------------------ examples
    def build_example(self, ac_model: ActivationContextModel, item: ActivationContextTrainingItem) -> _Example:
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
        if self.config.loss_kind == "kl":
            if item.in_context_messages is None:
                raise ValueError(f"{item.item_id}: KL needs an in-context teacher history")
            validate_paired_histories(item.in_context_messages, item.in_context_start, item.ac_messages, item.ac_start)
            teacher_ids, _, teacher_positions = prepare_training_history(tokenizer, item.in_context_messages,
                                item.in_context_start, item.assistant_token_ids, [], tools=item.tools, chat_template_kwargs=chat_kwargs)
        for view, ids in (("teacher", teacher_ids), ("student", student_ids)):
            if len(ids) > self.config.max_example_tokens:
                raise ValueError(f"{item.item_id}: {view} history has {len(ids)} tokens, exceeding max_example_tokens={self.config.max_example_tokens}; regenerate or raise the guard")
        digest = hashlib.sha256(json.dumps([teacher_ids, teacher_positions], separators=(",", ":")).encode()).hexdigest()
        key = (ac_model.name, digest, self.config.teacher_cache_top_k)
        return _Example(item, teacher_ids, student_ids, teacher_positions, student_positions,
                        [token for output in item.assistant_token_ids for token in output], spans, requests, key)

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
        config = self.config
        if not training_data:
            raise ValueError("No eligible AC training data")
        self.teacher_cache.clear()                 # teacher targets belong to this invocation only
        manager = self.harness.module_manager
        ac_model = manager.get_ac_model(ac_model_name)
        target = ac_model.target
        target_lora = ac_model.config.target_model_lora_name
        start_epoch = self._next_epoch_number(ac_model)
        stats = ActivationContextTrainingStats(loss_kind=config.loss_kind, start_epoch=start_epoch, items=len(training_data))
        started = time.monotonic()
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
            ac_parameters = ac_model.trainable_parameters()
            target_parameters = manager.lora_parameters(target_lora)
            assert ac_parameters and target_parameters
            if reset_optimizer:
                self.optimizers.pop(ac_model_name, None)
            optimizer = persistent_optimizer(self.optimizers, ac_model_name,
                    [{"params": ac_parameters, "lr": config.learning_rate_ac}, {"params": target_parameters, "lr": config.learning_rate_target_lora}],
                    device, betas=config.adam_betas, eps=config.adam_eps, weight_decay=config.weight_decay)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            stats.placement_seconds = time.monotonic() - placement_started
            baseline = ActivationContextTrainingProgress(0, config.num_epochs, start_epoch - 1, 0, steps,
                        self.updates_done.get(ac_model_name, 0), 0.0, time.monotonic() - started)
            phase = time.monotonic()
            if reporting_data:
                stats.reporting = self._evaluate(ac_model, list(reporting_data), device)
                if reporter is not None:
                    reporter.report_eval(baseline, "reporting", stats.reporting)
            stats.baseline_samples = self._sample_completions(ac_model, fixed_items, baseline)
            stats.baseline_seconds = time.monotonic() - phase
            if reporter is not None and stats.baseline_samples:
                reporter.report_completions(baseline, stats.baseline_samples)
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
                    if reporter is not None:
                        reporter.report_phase(progress(step - 1), "training")
                    step_started = time.monotonic()
                    before_side_tokens = ac_model.stats.side_tokens
                    mini = order[(step - 1) * per_step:step * per_step]
                    optimizer.zero_grad(set_to_none=True)
                    totals = {"loss": 0.0, "agreement": 0.0, "drift": 0.0, "tokens": 0}
                    for index in mini:
                        example = examples[index]
                        example_started = time.monotonic()
                        cached = config.loss_kind == "kl" and config.teacher_cache_top_k > 0 and example.teacher_cache_key in self.teacher_cache
                        objective, agreement, drift, _ = self._example_loss(ac_model, example, device)
                        loss = (objective + config.drift_term_weight * drift) * (example.item.weight / len(mini))
                        loss.backward()
                        loss_value = float(objective.detach())
                        stats.example_records.append({"kind": example.item.kind, "teacher_tokens": len(example.teacher_ids),
                            "student_tokens": len(example.student_ids), "seconds": round(time.monotonic() - example_started, 3), "teacher_cached": cached})
                        stats.teacher_cache_hits += int(cached)
                        stats.loss_by_kind.setdefault(example.item.kind, []).append(loss_value)
                        totals["loss"] += loss_value * example.item.weight
                        if config.loss_kind == "kl":
                            stats.kl_by_kind.setdefault(example.item.kind, []).append(loss_value)
                            stats.agreement_by_kind.setdefault(example.item.kind, []).append(agreement)
                            totals["agreement"] += agreement
                        totals["drift"] += float(drift.detach()) * example.item.weight
                        totals["tokens"] += len(example.teacher_ids) + len(example.student_ids)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(ac_parameters + target_parameters, config.max_grad_norm))
                    set_learning_rates(optimizer, self.updates_done.get(ac_model_name, 0), config.warmup_updates)
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
                        stats.drift.append(totals["drift"] / len(mini))
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
                    if step in boundaries and reporting_data:
                        phase = time.monotonic()
                        if reporter is not None:
                            reporter.report_phase(progress(step), "reporting")
                        record.reporting = self._evaluate(ac_model, list(reporting_data), device)
                        record.reporting_seconds += time.monotonic() - phase
                        stats.reporting = record.reporting
                        if reporter is not None:
                            reporter.report_eval(progress(step), "reporting", record.reporting)
                if validation_data:
                    phase = time.monotonic()
                    if reporter is not None:
                        reporter.report_phase(progress(steps), "validation")
                    record.validation = (record.reporting if list(validation_data) == list(reporting_data) and record.reporting is not None
                                         else self._evaluate(ac_model, list(validation_data), device))
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
        for source, latest in ((ac_folder, ac_model.checkpoint_folder()), (target_folder, manager.checkpoint_folder(target_lora))):
            if latest.exists():
                shutil.rmtree(latest)
            shutil.copytree(source, latest)
        record = ac_folder.parent / "completed_epoch.json"
        temporary = record.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"epoch_number": epoch_number, "ac_checkpoint": ac_relative,
                                        "target_lora_checkpoint": target_relative}, indent=2) + "\n")
        temporary.replace(record)       # only this record declares the numbered pair complete
        return ac_path, target_path

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
                for side in (("teacher", "student") if self.config.loss_kind == "kl" else ("student",)):
                    sequence = example.teacher_ids if side == "teacher" else example.student_ids
                    positions = example.teacher_positions if side == "teacher" else example.student_positions
                    prefix_length = positions[0] + 1
                    ids = torch.tensor(sequence[:prefix_length], device=ac_model.device)
                    embeds = target.model.get_input_embeddings()(ids)
                    if side == "student" and example.part_requests:
                        present = [(span, request) for span, request in zip(example.spans, example.part_requests) if span[1] <= prefix_length]
                        rows = ac_model.encode_batch([request for _, request in present]) if present else []
                        pieces, cursor = [], 0
                        for ((start, end), _), part_rows in zip(present, rows):
                            pieces.extend([embeds[cursor:start], part_rows.to(embeds.dtype)])
                            cursor = end
                        embeds = torch.cat(pieces + [embeds[cursor:]], dim=0)
                    lora = ac_model.config.target_model_lora_name
                    peft_model = manager.ensure_lora(lora)
                    budget = min(self.config.completion_max_new_tokens, capacity - prefix_length) if capacity else self.config.completion_max_new_tokens
                    with manager.lora_context(target.model_config.model_name, lora):
                        output = peft_model.generate(inputs_embeds=embeds[None], max_new_tokens=budget,
                                        do_sample=False, use_cache=True,
                                        pad_token_id=target.tokenizer.pad_token_id if target.tokenizer.pad_token_id is not None else target.tokenizer.eos_token_id)
                    texts[side] = target.tokenizer.decode(output[0], skip_special_tokens=True)
                result.append(CompletionSample(item_id=item.item_id, kind=item.kind,
                    reference=target.tokenizer.decode(item.assistant_token_ids[0], skip_special_tokens=False),
                    teacher=texts.get("teacher"), student=texts["student"], epoch_number=progress.epoch_number,
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
    ) -> ActivationContextEvalSummary:
        """
        KL and agreement of the items without gradient (models placed as for training, moved back when `release`).
        A supplied reporter receives this exact evaluation under reporting_name.
        """
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
            summary = self._evaluate(ac_model, items, device)
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
    def _evaluate(self, ac_model: ActivationContextModel, items: list[ActivationContextTrainingItem], device: torch.device) -> ActivationContextEvalSummary:
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
            for example in examples:
                loss, agreement, _, positions = self._example_loss(ac_model, example, device, with_drift=False)
                per_item.append({"item_id": example.item.item_id, "kind": example.item.kind,
                    "loss_kind": self.config.loss_kind, "loss": float(loss), "kl": float(loss) if is_kl else None,
                    "agreement": agreement, "positions": positions,
                    "teacher_tokens": len(example.teacher_ids), "student_tokens": len(example.student_ids)})
        finally:
            for base, training in zip(bases, training_modes):
                base.train(training)
                disable_dropout(base)
            ac_model.set_mode(previous_mode)
        def summarize(rows):
            loss = sum(row["loss"] for row in rows) / len(rows) if rows else None
            return {"loss_kind": self.config.loss_kind, "loss": loss, "kl": loss if is_kl else None,
                    "agreement": sum(row["agreement"] for row in rows) / len(rows) if is_kl and rows else None,
                    "items": len(rows)}
        by_kind = {kind: summarize([row for row in per_item if row["kind"] == kind])
                   for kind in dict.fromkeys(row["kind"] for row in per_item)}
        return {**summarize(per_item), "dropped_too_long": 0, "by_kind": by_kind, "per_item": per_item}

    def _example_loss(self, ac_model: ActivationContextModel, example: _Example, device: torch.device, with_drift: bool = True) -> tuple[torch.Tensor, float | None, torch.Tensor, int]:
        """(kl, agreement, drift, positions) of one example: teacher no-grad, student with the rows; drift on the in-context tokens when configured."""
        config = self.config
        target = ac_model.target
        base = target.model
        base_dtype = next(base.parameters()).dtype
        embedding = base.get_input_embeddings()
        head = base.get_output_embeddings()
        lora = ac_model.config.target_model_lora_name
        num = example.num_completion
        is_kl = config.loss_kind == "kl"
        top_k = config.teacher_cache_top_k if is_kl and torch.is_grad_enabled() else 0                  # eval is exact
        cached = self.teacher_cache.get(example.teacher_cache_key) if top_k else None
        teacher_ids = torch.tensor(example.teacher_ids, device=device, dtype=torch.long) if is_kl else None
        teacher_states = None
        if is_kl and cached is None:
            # Teacher.
            with torch.no_grad():
                teacher_hidden = target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=lora, gradient_checkpointing=False)[0]
                teacher_states = teacher_hidden[example.teacher_positions]
            if top_k:
                cached = self._teacher_top_k(teacher_states, head, top_k)
                self.teacher_cache[example.teacher_cache_key] = tuple(tensor.cpu() for tensor in cached)
                teacher_states = None
        elif cached is not None:
            cached = tuple(tensor.to(device, non_blocking=True) for tensor in cached)
        # Student: rows for the parts (grad), written over the placeholders.
        rows = ac_model.encode_batch(example.part_requests) if example.part_requests else []
        student_ids = torch.tensor(example.student_ids, device=device)
        with torch.no_grad():
            student_embeds = embedding(student_ids)
        if rows:
            assert [int(part_rows.shape[0]) for part_rows in rows] == [end - start for start, end in example.spans], "placeholder runs differ from the encoded rows"
            pieces, cursor = [], 0
            for (start, end), part_rows in zip(example.spans, rows):
                pieces.extend([student_embeds[cursor:start], part_rows.to(base_dtype)])
                cursor = end
            pieces.append(student_embeds[cursor:])
            student_embeds = torch.cat(pieces, dim=0)
        checkpointed = (config.gradient_checkpointing and torch.is_grad_enabled()
                        and len(example.student_ids) >= getattr(self, "_target_gradient_checkpointing_min_tokens", config.gradient_checkpointing_min_tokens or 0))
        student_hidden = target.decoder_forward(student_embeds[None], None, lora_name=lora, gradient_checkpointing=checkpointed)[0]
        student_states = student_hidden[example.student_positions]
        if is_kl:
            loss, agreement = self._chunked_kl(teacher_states, student_states, head, cached=cached)
        else:
            loss, agreement = self._chunked_sft(student_states, head, example.target_ids), None
        drift = torch.zeros((), device=device)
        if is_kl and with_drift and config.drift_term_weight > 0:
            # The adapter's in-context behaviour on the last in-context positions, held to the frozen base's.
            prefix = example.teacher_positions[0] + 1
            window = slice(max(0, prefix - 1024), prefix)
            with torch.no_grad():
                frozen_states = target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=None, gradient_checkpointing=False)[0][window]
            drift_checkpointed = (config.gradient_checkpointing and torch.is_grad_enabled()
                                  and len(example.teacher_ids) >= getattr(self, "_target_gradient_checkpointing_min_tokens", config.gradient_checkpointing_min_tokens or 0))
            adapted_states = target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=lora,
                                                    gradient_checkpointing=drift_checkpointed)[0][window]
            drift, _ = self._chunked_kl(frozen_states, adapted_states, head)
        return loss, agreement, drift, num

    def _chunked_sft(self, states: torch.Tensor, head: torch.nn.Module, target_ids: list[int]) -> torch.Tensor:
        chunk = getattr(self, "_logits_chunk_tokens", self.config.logits_chunk_tokens or 64)
        targets = torch.tensor(target_ids, device=states.device)
        def cross_entropy(hidden, labels):
            return F.cross_entropy(head_logits(hidden, head, self.config.fp32_head_matmul), labels, reduction="sum")
        total = states.new_zeros((), dtype=torch.float32)
        for start in range(0, len(target_ids), chunk):
            arguments = states[start:start + chunk], targets[start:start + chunk]
            total = total + (checkpoint(cross_entropy, *arguments, use_reentrant=False)
                             if torch.is_grad_enabled() and states.requires_grad else cross_entropy(*arguments))
        return total / len(target_ids)

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
                    cached: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, float]:
        """
        Mean KL(teacher || student) and top-1 agreement over the positions, the vocabulary logits one chunk at a
        time. With `cached` (the teacher's top-k ids, log-probs and log remainder mass, see `_teacher_top_k`) the
        teacher states are not needed: the KL is between the k+1-way coarsened distributions (the top-k tokens
        and "everything else"), a lower bound of the exact KL that is tight when the top-k carry the mass.
        """
        chunk = getattr(self, "_logits_chunk_tokens", self.config.logits_chunk_tokens or 64)
        fp32 = self.config.fp32_head_matmul
        total = torch.zeros((), device=student_states.device, dtype=torch.float32)
        agree = 0.0
        num = student_states.shape[0]

        def chunk_kl(student_chunk: torch.Tensor, teacher_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                teacher_logp = F.log_softmax(head_logits(teacher_chunk, head, fp32), dim=-1)
            student_logp = F.log_softmax(head_logits(student_chunk, head, fp32), dim=-1)
            kl_sum = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1).sum()
            agreement = (teacher_logp.argmax(dim=-1) == student_logp.argmax(dim=-1)).float().sum()
            return kl_sum, agreement.detach()

        def chunk_kl_cached(student_chunk: torch.Tensor, ids: torch.Tensor, teacher_logp: torch.Tensor, teacher_rest: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            student_logp = F.log_softmax(head_logits(student_chunk, head, fp32), dim=-1)
            student_top = student_logp.gather(-1, ids.long())
            student_rest = torch.log(torch.clamp(1.0 - student_top.exp().sum(dim=-1), min=1e-12))
            kl_sum = ((teacher_logp.exp() * (teacher_logp - student_top)).sum(dim=-1) + teacher_rest.exp() * (teacher_rest - student_rest)).sum()
            agreement = (ids[:, 0].long() == student_logp.argmax(dim=-1)).float().sum()
            return kl_sum, agreement.detach()

        for start in range(0, num, chunk):
            student_chunk = student_states[start:start + chunk]
            if cached is not None:
                function, arguments = chunk_kl_cached, (student_chunk, *(tensor[start:start + chunk] for tensor in cached))
            else:
                function, arguments = chunk_kl, (student_chunk, teacher_states[start:start + chunk])
            if torch.is_grad_enabled() and student_chunk.requires_grad:
                kl_sum, agreement = checkpoint(function, *arguments, use_reentrant=False)
            else:
                kl_sum, agreement = function(*arguments)
            total = total + kl_sum
            agree += float(agreement)
        return total / max(1, num), agree / max(1, num)

    def _autotuning_variables(self, ac_model: ActivationContextModel) -> dict:
        variables = get_ac_training_autotuning_variables(ac_model.side.model_config, ac_model.target.model_config,
                        self.config, ac_config=ac_model.config, device=TARGET_DEVICE)
        self._logits_chunk_tokens = variables["logits_chunk_tokens"]
        return variables
