"""
Training the AC model in isolation (self-distillation): the target model reading the full context
(the teacher, the same target adapter, no gradient) is matched by the target model reading the AC
prefix (the student) on the completion tokens, with a full-vocabulary KL(teacher || student). One
round trains the AC modules and the side LoRA (`learning_rate_ac`) and the target LoRA
(`learning_rate_target_lora`) together; the AdamW state and the warm-up clock persist across
rounds (one optimizer per AC model, parked on the host between rounds).

Teacher tokens: the in-context prefix under the target chat template with the generation prompt
open, then (for a mid-turn cut) the text the turn had already produced, then the completion text
and, when the completion is a whole turn, the end-of-turn token. Student tokens: the AC prefix
through `tokenize_with_parts` (rows written over the placeholder runs), then the same completion
tokens. The loss averages over the completion positions.

`drift_term_weight` > 0 adds KL(frozen base || target-with-adapter) on the in-context tokens so
the target adapter cannot drift from the base's in-context behaviour; off by default (the teacher
already carries the adapter).
"""
from __future__ import annotations

import random
import time
import typing as t
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..agent_training.agent_training_utils import (
    checkpointing_min_tokens, disable_dropout, head_logits, optimizer_state_to, persistent_optimizer, set_checkpointing, set_learning_rates,
)
from ..agent_training.fla_cache import configure_fla_cache
from ..harness.hf_utils import SOURCE_DEVICE
from .ac_model import ROLLOUT, TRAINING, ActivationContextModel
from .ac_model_utils import EncodeRequest, direct_parts, tokenize_with_parts

if t.TYPE_CHECKING:
    from ..harness import HarnessRuntime
    from .ac_model_reporter import ActivationContextTrainingReporter

ITEM_KINDS = ("compaction", "traj_qa", "rag_qa")


@dataclass
class ActivationContextTrainingItem:
    item_id: str
    kind: str                                  # compaction | traj_qa | rag_qa (or a reference variant's name)
    in_context_prefix: list[dict]              # the teacher's messages (plain text)
    ac_prefix: list[dict]                      # the student's messages, with activation_context parts (or plain text for references)
    completion_text: str                       # what both must predict (a whole turn or the rest of a cut turn)
    completion_complete: bool = True           # the completion ends the turn: the end-of-turn token is a target too
    teacher_partial_text: str = ""             # mid-turn cut: the text already produced, raw tokens for the teacher (the AC side holds it compressed)
    tools: list[dict] | None = None            # tool definitions for the template (both sides)
    weight: float = 1.0
    dataset_id: str = ""
    doc_ids: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)   # generator details (depth, ratio, budgets, gold position, ...)


@dataclass
class ActivationContextTrainingConfig:
    learning_rate_ac: float = 5e-4             # AC modules and the side LoRA
    learning_rate_target_lora: float = 2e-5    # the target adapter
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_updates: int = 0                    # linear warm-up of both learning rates over this many updates (counted across rounds)
    updates_per_round: int = 4                 # gradient steps per train() call (the data split into this many mini-batches)
    examples_per_update: int | None = None     # when set, overrides updates_per_round: ceil(items / examples_per_update) steps
    max_example_tokens: int = 72_000           # longer teacher sequences are dropped (counted), never truncated: depth 2 at the
                                               # default 32k compaction threshold, depth 1 up to a 50k threshold (task + completion included)
    teacher_cache_top_k: int = 0               # > 0: the teacher's top-k log-probs (+ the remainder mass) per completion position are
                                               # kept per item after its first pass and stand in for the teacher afterwards (epoch 2
                                               # skips the teacher forward); training then minimizes the KL of the k+1-way coarsened
                                               # distributions (a lower bound of the exact KL, the same loss in every epoch, the
                                               # teacher's adapter as of the item's last exact pass). Eval is always exact.
    drift_term_weight: float = 0.0
    logits_chunk_tokens: int = 2_048           # completion positions per lm_head chunk
    fp32_head_matmul: bool = False
    gradient_checkpointing: bool = True        # on both bases
    gradient_checkpointing_min_tokens: int | None = None
    seed: int = 0
    checkpoint_every_round: bool = True        # AC_MODELS/<name>/round_<n> (+ latest) and LORAS/<target lora>/round_<n>


@dataclass
class ActivationContextTrainingStats:
    round_index: int = 0
    items: int = 0
    examples: int = 0
    dropped_too_long: int = 0
    steps: int = 0
    kl: list[float] = field(default_factory=list)               # per step, weighted mean over completion positions
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
    reporting: dict = field(default_factory=dict)               # eval() summary on the reporting items after the round
    checkpoint_path: str | None = None
    target_lora_checkpoint_path: str | None = None
    peak_memory_bytes: int = 0
    duration_s: float = 0.0
    example_records: list[dict] = field(default_factory=list)   # per example: kind, teacher/student tokens, seconds, teacher_cached
    teacher_cache_hits: int = 0
    teacher_cache_bytes: int = 0

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
            "round": self.round_index, "items": self.items, "examples": self.examples, "dropped_too_long": self.dropped_too_long,
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
    spans: list[tuple[int, int]]
    part_requests: list[EncodeRequest]
    num_completion: int                        # completion tokens (+ eot) at the end of both sequences


class ActivationContextTrainer:
    def __init__(self, harness: "HarnessRuntime", config: ActivationContextTrainingConfig | None = None):
        self.harness = harness
        self.config = config or ActivationContextTrainingConfig()
        self.rounds_done: dict[str, int] = {}
        self.updates_done: dict[str, int] = {}                    # optimizer steps so far per AC model (the warm-up clock)
        self.optimizers: dict[str, torch.optim.AdamW] = {}        # one per AC model, kept across rounds (moments survive a round boundary)
        self.teacher_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}   # item id -> (ids [C, k], log-probs [C, k], log remainder [C]), CPU

    # ------------------------------------------------------------------------------------------ examples
    def build_example(self, ac_model: ActivationContextModel, item: ActivationContextTrainingItem) -> _Example:
        tokenizer = ac_model.target.tokenizer
        eot = ac_model.target.model_config.model_description.eot_token
        completion_ids = tokenizer.encode(item.completion_text, add_special_tokens=False)
        if item.completion_complete and eot:
            completion_ids = completion_ids + tokenizer.encode(eot, add_special_tokens=False)
        teacher_prefix = tokenizer.apply_chat_template(_flatten(item.in_context_prefix), tools=item.tools, add_generation_prompt=True, tokenize=True)
        teacher_prefix = list(teacher_prefix["input_ids"] if hasattr(teacher_prefix, "keys") else teacher_prefix)
        if item.teacher_partial_text:
            teacher_prefix += tokenizer.encode(item.teacher_partial_text, add_special_tokens=False)
        parts = direct_parts(item.ac_prefix)
        requests = [ac_model._child_request(part, ac_model.config.default_compression_ratio) for part in parts]
        requests = [EncodeRequest(request.messages, request.compression_ratio, False) for request in requests]   # top-level parts: target rows
        lengths = [ac_model.part_view_rows(request.messages, request.compression_ratio) for request in requests]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
        student_prefix, spans = tokenize_with_parts(tokenizer, item.ac_prefix, lengths, pad_id, tools=item.tools, add_generation_prompt=True)
        return _Example(item, teacher_prefix + completion_ids, student_prefix + completion_ids, spans, requests, len(completion_ids))

    # ------------------------------------------------------------------------------------------ training
    def train(
        self,
        ac_model_name: str,
        training_data: list[ActivationContextTrainingItem],
        reporting_data: t.Sequence[ActivationContextTrainingItem] = (),
        reporter: "ActivationContextTrainingReporter | None" = None,
    ) -> ActivationContextTrainingStats:
        """
        One round: the target engine to sleep, target + target LoRA and side + side LoRA on the GPU
        with the AC modules, one pass over the items in steps of examples_per_update (or
        updates_per_round mini-batches), one example per micro-batch, the reporting items evaluated
        without gradient, checkpoints written, models back to host RAM.
        """
        config = self.config
        assert training_data, "No training data."
        module_manager = self.harness.module_manager
        ac_model = module_manager.get_ac_model(ac_model_name)
        target = ac_model.target
        target_lora = ac_model.config.target_model_lora_name
        round_index = self.rounds_done.get(ac_model_name, 0)
        stats = ActivationContextTrainingStats(round_index=round_index, items=len(training_data))
        started = time.time()
        torch.manual_seed(config.seed + round_index)
        examples: list[_Example] = []
        for item in training_data:
            example = self.build_example(ac_model, item)
            if len(example.teacher_ids) > config.max_example_tokens or len(example.student_ids) > config.max_example_tokens:
                stats.dropped_too_long += 1
            else:
                examples.append(example)
        assert examples, f"every item was dropped ({stats.dropped_too_long} too long)"
        stats.examples = len(examples)
        stats.completion_tokens = sum(example.num_completion for example in examples)
        stats.teacher_tokens = sum(len(example.teacher_ids) for example in examples)
        stats.student_tokens = sum(len(example.student_ids) for example in examples)
        if reporter is not None:
            reporter.initialize_round(round_index, config, stats)

        # Placement: engine asleep, both bases with their adapters resident and training, dropout off.
        if target.vllm_model is not None:
            target.engine_to_device(SOURCE_DEVICE)
        module_manager.ensure_lora(target_lora)
        device = ac_model.prepare()
        self._configure_kernels(device)
        ac_model.set_mode(TRAINING)
        target_base, side_base = target.model, ac_model.side.model
        for base in (target_base, side_base):
            base.train()
            disable_dropout(base)
        min_tokens = checkpointing_min_tokens(config, device)
        side_config = _CheckpointConfig(config.gradient_checkpointing and ac_model.config.side_gradient_checkpointing, config.gradient_checkpointing_min_tokens)
        set_checkpointing(target_base, config, min_tokens, min_tokens)
        set_checkpointing(side_base, side_config, min_tokens, min_tokens)
        ac_parameters = ac_model.trainable_parameters()
        target_parameters = module_manager.lora_parameters(target_lora)
        assert ac_parameters and target_parameters
        optimizer = persistent_optimizer(self.optimizers, ac_model_name,
                                         [{"params": ac_parameters, "lr": config.learning_rate_ac}, {"params": target_parameters, "lr": config.learning_rate_target_lora}],
                                         device, betas=config.adam_betas, eps=config.adam_eps, weight_decay=config.weight_decay)
        all_parameters = ac_parameters + target_parameters
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            order = list(range(len(examples)))
            random.Random(config.seed + round_index).shuffle(order)
            if config.examples_per_update:
                per_step = int(config.examples_per_update)
            else:
                per_step = -(-len(order) // max(1, min(config.updates_per_round, len(order))))
            steps = -(-len(order) // per_step)
            for step in range(steps):
                step_started = time.time()
                mini = order[step * per_step:(step + 1) * per_step]
                optimizer.zero_grad(set_to_none=True)
                totals = {"kl": 0.0, "agreement": 0.0, "drift": 0.0, "positions": 0, "tokens": 0}
                for index in mini:
                    example = examples[index]
                    example_started = time.time()
                    set_checkpointing(target_base, config, len(example.teacher_ids), min_tokens)
                    cached = config.teacher_cache_top_k > 0 and example.item.item_id in self.teacher_cache
                    kl, agreement, drift, positions = self._example_loss(ac_model, example, device)
                    loss = (kl + config.drift_term_weight * drift) * (example.item.weight / len(mini))
                    loss.backward()
                    kl_value, agreement_value = float(kl.detach()), float(agreement)      # synchronizes: the example's work is done
                    stats.example_records.append({"kind": example.item.kind, "teacher_tokens": len(example.teacher_ids), "student_tokens": len(example.student_ids),
                                                  "seconds": round(time.time() - example_started, 3), "teacher_cached": cached})
                    stats.teacher_cache_hits += int(cached)
                    stats.kl_by_kind.setdefault(example.item.kind, []).append(kl_value)
                    stats.agreement_by_kind.setdefault(example.item.kind, []).append(agreement_value)
                    totals["kl"] += kl_value * positions
                    totals["agreement"] += agreement_value * positions
                    totals["drift"] += float(drift.detach()) * positions
                    totals["positions"] += positions
                    totals["tokens"] += len(example.teacher_ids) + len(example.student_ids)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(all_parameters, config.max_grad_norm))
                set_learning_rates(optimizer, self.updates_done.get(ac_model_name, 0), config.warmup_updates)
                optimizer.step()
                self.updates_done[ac_model_name] = self.updates_done.get(ac_model_name, 0) + 1
                divisor = max(1, totals["positions"])
                seconds = time.time() - step_started
                stats.steps += 1
                stats.kl.append(totals["kl"] / divisor)
                stats.agreement.append(totals["agreement"] / divisor)
                stats.drift.append(totals["drift"] / divisor)
                stats.grad_norm.append(grad_norm)
                stats.step_seconds.append(seconds)
                stats.step_tokens.append(totals["tokens"])
                print(f"ActivationContextTrainer - round {round_index} step {step + 1}/{steps}: kl {stats.kl[-1]:.4f} agreement {stats.agreement[-1]:.3f} "
                      f"grad {grad_norm:.3f} {totals['tokens']} tokens in {seconds:.1f}s ({totals['tokens'] / max(seconds, 1e-6):.0f} tok/s)", flush=True)
                if reporter is not None:
                    reporter.report_step(round_index, step + 1, steps, stats)
            if reporting_data:
                stats.reporting = self._evaluate(ac_model, list(reporting_data), device)
                if reporter is not None:
                    reporter.report_eval(round_index, "reporting", stats.reporting)
            stats.teacher_cache_bytes = sum(sum(tensor.numel() * tensor.element_size() for tensor in entry) for entry in self.teacher_cache.values())
            if config.checkpoint_every_round:
                stats.checkpoint_path = ac_model.save()
                stats.target_lora_checkpoint_path = module_manager.save_lora(target.model_config.model_name, target_lora)
        finally:
            optimizer_state_to(optimizer, SOURCE_DEVICE)
            for base in (target_base, side_base):
                base.eval()
                if base.is_gradient_checkpointing:
                    base.gradient_checkpointing_disable()
            ac_model.set_mode(ROLLOUT)
            if device.type == "cuda":
                stats.peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
            stats.side_tokens = ac_model.stats.side_tokens
            ac_model.release()
            target.model_to_device(SOURCE_DEVICE)
            ac_model.side.model_to_device(SOURCE_DEVICE)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.rounds_done[ac_model_name] = round_index + 1
        stats.duration_s = time.time() - started
        if reporter is not None:
            reporter.report_round(stats)
        print(f"ActivationContextTrainer - round {round_index} done: {stats.summarize()}", flush=True)
        return stats

    def eval(self, ac_model_name: str, items: list[ActivationContextTrainingItem], release: bool = True) -> dict:
        """KL and agreement of the items without gradient (models placed as for training, moved back when `release`)."""
        module_manager = self.harness.module_manager
        ac_model = module_manager.get_ac_model(ac_model_name)
        target = ac_model.target
        if target.vllm_model is not None:
            target.engine_to_device(SOURCE_DEVICE)
        module_manager.ensure_lora(ac_model.config.target_model_lora_name)
        device = ac_model.prepare()
        self._configure_kernels(device)
        try:
            return self._evaluate(ac_model, items, device)
        finally:
            if release:
                ac_model.release()
                target.model_to_device(SOURCE_DEVICE)
                ac_model.side.model_to_device(SOURCE_DEVICE)

    # ------------------------------------------------------------------------------------------ internals
    @torch.no_grad()
    def _evaluate(self, ac_model: ActivationContextModel, items: list[ActivationContextTrainingItem], device) -> dict:
        previous_mode = ac_model.mode
        ac_model.set_mode(TRAINING)                                          # no cache; the no_grad above makes it free of gradients
        target_base = ac_model.target.model
        was_training = target_base.training
        target_base.eval()
        ac_model.side.model.eval()
        per_item = []
        dropped = 0
        try:
            for item in items:
                example = self.build_example(ac_model, item)
                if len(example.teacher_ids) > self.config.max_example_tokens or len(example.student_ids) > self.config.max_example_tokens:
                    dropped += 1
                    continue
                kl, agreement, _, positions = self._example_loss(ac_model, example, device, with_drift=False)
                per_item.append({"item_id": item.item_id, "kind": item.kind, "kl": float(kl), "agreement": float(agreement), "positions": positions,
                                 "teacher_tokens": len(example.teacher_ids), "student_tokens": len(example.student_ids)})
        finally:
            if was_training:
                for base in (target_base, ac_model.side.model):
                    base.train()
                    disable_dropout(base)
            ac_model.set_mode(previous_mode)
        by_kind: dict[str, list[dict]] = {}
        for row in per_item:
            by_kind.setdefault(row["kind"], []).append(row)
        summary = {kind: {"kl": sum(r["kl"] for r in rows) / len(rows), "agreement": sum(r["agreement"] for r in rows) / len(rows), "items": len(rows)}
                   for kind, rows in by_kind.items()}
        return {"items": len(per_item), "dropped_too_long": dropped, "kl": sum(r["kl"] for r in per_item) / max(1, len(per_item)),
                "agreement": sum(r["agreement"] for r in per_item) / max(1, len(per_item)), "by_kind": summary, "per_item": per_item}

    def _example_loss(self, ac_model: ActivationContextModel, example: _Example, device, with_drift: bool = True):
        """(kl, agreement, drift, positions) of one example: teacher no-grad, student with the rows; drift on the in-context tokens when configured."""
        config = self.config
        target = ac_model.target
        base = target.model
        base_dtype = next(base.parameters()).dtype
        embedding = base.get_input_embeddings()
        head = base.get_output_embeddings()
        lora = ac_model.config.target_model_lora_name
        num = example.num_completion
        top_k = config.teacher_cache_top_k if torch.is_grad_enabled() else 0                  # eval is exact
        cached = self.teacher_cache.get(example.item.item_id) if top_k else None
        teacher_ids = torch.tensor(example.teacher_ids, device=device)
        teacher_states = None
        if cached is None:
            # Teacher.
            with torch.no_grad():
                teacher_hidden = target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=lora)[0]
                teacher_states = teacher_hidden[-num - 1:-1]
            if top_k:
                cached = self._teacher_top_k(teacher_states, head, top_k)
                self.teacher_cache[example.item.item_id] = tuple(tensor.cpu() for tensor in cached)
                teacher_states = None
        else:
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
        student_hidden = target.decoder_forward(student_embeds[None], None, lora_name=lora)[0]
        student_states = student_hidden[-num - 1:-1]
        kl, agreement = self._chunked_kl(teacher_states, student_states, head, cached=cached)
        drift = torch.zeros((), device=device)
        if with_drift and config.drift_term_weight > 0:
            # The adapter's in-context behaviour on the last in-context positions, held to the frozen base's.
            prefix = len(example.teacher_ids) - num
            window = slice(max(0, prefix - 1024), prefix)
            with torch.no_grad():
                frozen_states = target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=None)[0][window]
            adapted_states = target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=lora)[0][window]
            drift, _ = self._chunked_kl(frozen_states, adapted_states, head)
        return kl, agreement, drift, num

    @torch.no_grad()
    def _teacher_top_k(self, teacher_states: torch.Tensor, head, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(ids [C, k], log-probs [C, k] descending, log remainder mass [C]) of the teacher at every completion position."""
        chunk, fp32 = self.config.logits_chunk_tokens, self.config.fp32_head_matmul
        ids, logps, rests = [], [], []
        for start in range(0, teacher_states.shape[0], chunk):
            logp = F.log_softmax(head_logits(teacher_states[start:start + chunk], head, fp32), dim=-1)
            values, indexes = logp.topk(k, dim=-1)
            ids.append(indexes.to(torch.int32))
            logps.append(values)
            rests.append(torch.log(torch.clamp(1.0 - values.exp().sum(dim=-1), min=1e-12)))
        return torch.cat(ids), torch.cat(logps), torch.cat(rests)

    def _chunked_kl(self, teacher_states: torch.Tensor | None, student_states: torch.Tensor, head,
                    cached: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, float]:
        """
        Mean KL(teacher || student) and top-1 agreement over the positions, the vocabulary logits one chunk at a
        time. With `cached` (the teacher's top-k ids, log-probs and log remainder mass, see `_teacher_top_k`) the
        teacher states are not needed: the KL is between the k+1-way coarsened distributions (the top-k tokens
        and "everything else"), a lower bound of the exact KL that is tight when the top-k carry the mass.
        """
        chunk = self.config.logits_chunk_tokens
        fp32 = self.config.fp32_head_matmul
        total = torch.zeros((), device=student_states.device, dtype=torch.float32)
        agree = 0.0
        num = student_states.shape[0]

        def chunk_kl(student_chunk, teacher_chunk):
            with torch.no_grad():
                teacher_logp = F.log_softmax(head_logits(teacher_chunk, head, fp32), dim=-1)
            student_logp = F.log_softmax(head_logits(student_chunk, head, fp32), dim=-1)
            kl_sum = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1).sum()
            agreement = (teacher_logp.argmax(dim=-1) == student_logp.argmax(dim=-1)).float().sum()
            return kl_sum, agreement.detach()

        def chunk_kl_cached(student_chunk, ids, teacher_logp, teacher_rest):
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

    def _configure_kernels(self, device) -> None:
        if configure_fla_cache(device) is None and device.type == "cuda" and not getattr(self, "_kernels_warned", False):
            self._kernels_warned = True
            print("ActivationContextTrainer - no fla kernel configs for this GPU: the Triton kernels autotune per length bucket (see agent_training/fla_cache.py)", flush=True)


@dataclass
class _CheckpointConfig:
    gradient_checkpointing: bool
    gradient_checkpointing_min_tokens: int | None


def _flatten(messages: list[dict]) -> list[dict]:
    out = []
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
        out.append(message)
    return out
