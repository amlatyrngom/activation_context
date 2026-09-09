"""
Slice 1 requires e2e training and reporting to work well.
Slice 1 is complete when:
- An independent review has marked our default configs as standard, without pointless pitfalls that make lose percentage points.
    - This does not include things like the embedding dimensions, but the default lrs, clippings, warmup, etc. That we are not forgetting anything.
    - That's because I am not a hyperparam expert, and don't want to mess this up. So defaults should be very good and standard.
- We have a 1000-train/50-report run passing and displayed in a nice report.
    - The independent reviewer should also read this and make sure it's sensible.
    - With warmups and all.
- The loss trend is as expected.

Slice 2 is multi-vector indexing and retrieval.

Slice 3 will include dataset studying into the mix.

The trainer is a plain single-process loop: physical batch = logical batch, activation checkpointing
on the base and the AC model, a batch size from the config or from the GPU sizing heuristic, and a
worst-case probe before the first step. The loss is form A: one softmax row per (query, own
positive) with the query's other positives and same-document collisions masked, mean over a query's
positives then over queries; scores are MaxSim over the view vectors, divided by the temperature.
"""
import math
import random
import time
import typing as t

import torch
import torch.nn.functional as F

from ..dataset.dataset import LabeledRetrievalQAExample
from .retrieval_batching import (
    DatasetIndexes,
    RetrievalBatch,
    configured_batch_sizing,
    embed_in_length_groups,
    fixed_batches,
    flatten_candidates,
    make_batches,
    probe_batch_size,
    recommended_batch_size,
)
from .retrieval_model import RetrievalModel
from .retrieval_reporter import METRIC_NAMES, RetrievalReporter
from .retrieval_training_config import RetrievalTrainingConfig, RetrievalTrainingStats

if t.TYPE_CHECKING:
    from ..harness import HarnessRuntime


class RetrievalTrainer:
    def __init__(self, harness: "HarnessRuntime"):
        self.harness = harness

    # ----------------------------------------------------------------------------- scoring
    def _batch_scores(self, retrieval_model: RetrievalModel, batch: RetrievalBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Similarities [B, N] between the batch's queries and its distinct candidates, plus the
        own-positive mask [B, N] and the same-document collision mask [B, N] (a candidate from one
        of the query's positive documents that is not one of its own positives).
        """
        chunks, per_example = flatten_candidates(batch)
        queries = embed_in_length_groups(retrieval_model, [example.query for example in batch.examples], is_query=True)
        candidates = embed_in_length_groups(retrieval_model, [chunk.chunk_text for chunk in chunks], is_query=False)
        scores = retrieval_model.similarity(queries, candidates)                            # [B, N]
        num_queries, num_candidates = scores.shape
        own_positive = torch.zeros(num_queries, num_candidates, dtype=torch.bool)
        collision = torch.zeros(num_queries, num_candidates, dtype=torch.bool)
        candidate_docs = [chunk.doc_id for chunk in chunks]
        for query_index, (example, indices, num_positives) in enumerate(zip(batch.examples, per_example, batch.num_positives)):
            own_positive[query_index, indices[:num_positives]] = True
            positive_docs = set(example.positive_doc_ids or []) | {candidate_docs[index] for index in indices[:num_positives]}
            for candidate_index, doc_id in enumerate(candidate_docs):
                if doc_id in positive_docs:
                    collision[query_index, candidate_index] = True
        own_positive = own_positive.to(scores.device)
        collision = collision.to(scores.device) & ~own_positive
        return scores, own_positive, collision

    def _loss_from_scores(self, scores: torch.Tensor, own_positive: torch.Tensor, collision: torch.Tensor, temperature: float) -> torch.Tensor:
        logits = (scores / temperature).masked_fill(collision, float("-inf"))
        row_query, row_target = own_positive.nonzero(as_tuple=True)                        # one row per (query, own positive)
        other_own_positives = own_positive[row_query].clone()
        other_own_positives[torch.arange(row_query.shape[0], device=logits.device), row_target] = False
        row_logits = logits[row_query].masked_fill(other_own_positives, float("-inf"))
        row_loss = F.cross_entropy(row_logits.float(), row_target, reduction="none")
        num_queries = scores.shape[0]
        per_query_sum = torch.zeros(num_queries, device=row_loss.device).index_add_(0, row_query, row_loss)
        per_query_count = torch.zeros(num_queries, device=row_loss.device).index_add_(0, row_query, torch.ones_like(row_loss))
        return (per_query_sum / per_query_count.clamp_min(1)).mean()

    def _metrics_from_scores(self, scores: torch.Tensor, own_positive: torch.Tensor, collision: torch.Tensor) -> dict[str, float]:
        """
        In-batch ranking metrics over the unmasked candidates, averaged over queries: rank-1 (the
        top candidate is an own positive), MRR@10 (reciprocal rank of the first own positive within
        the top 10, else 0) and nDCG@10 with binary relevance over the query's own positives.
        """
        k = min(10, scores.shape[1])
        top = scores.masked_fill(collision, float("-inf")).topk(k, dim=1).indices                # [B, k]
        relevant = own_positive.gather(1, top).float()                                            # [B, k]
        discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=scores.device, dtype=torch.float))
        rank1 = relevant[:, 0]
        hit = relevant.any(dim=1)
        first = relevant.argmax(dim=1)                                                            # first True (0 when none)
        mrr = torch.where(hit, 1.0 / (first + 1).float(), torch.zeros_like(rank1))
        dcg = (relevant * discounts).sum(dim=1)
        num_ideal = own_positive.sum(dim=1).clamp(min=1, max=k)
        idcg = discounts.cumsum(dim=0)[num_ideal - 1]
        ndcg = dcg / idcg
        return {"in-batch rank-1": float(rank1.mean()), "in-batch mrr@10": float(mrr.mean()), "in-batch ndcg@10": float(ndcg.mean())}

    def contrastive_loss(self, retrieval_model: RetrievalModel, batch: RetrievalBatch, temperature: float) -> torch.Tensor:
        """
        Form A: embed the batch's queries and its distinct candidates, one softmax row per
        (query, own positive) with the query's other positives and same-document collisions masked,
        mean over a query's positives then over queries. Logits are similarity / temperature.
        """
        scores, own_positive, collision = self._batch_scores(retrieval_model, batch)
        return self._loss_from_scores(scores, own_positive, collision, temperature)

    def in_batch_metrics(self, retrieval_model: RetrievalModel, batch: RetrievalBatch) -> dict[str, float]:
        """In-batch rank-1, MRR@10 and nDCG@10 of the batch's queries (see _metrics_from_scores)."""
        scores, own_positive, collision = self._batch_scores(retrieval_model, batch)
        return self._metrics_from_scores(scores, own_positive, collision)

    @torch.no_grad()
    def _evaluate(self, retrieval_model: RetrievalModel, batches: list[RetrievalBatch], temperature: float) -> tuple[float, dict[str, float]]:
        """Example-weighted loss and in-batch metrics over the given batches, in eval mode."""
        total_loss, total_examples = 0.0, 0
        totals = {name: 0.0 for name in METRIC_NAMES}
        for batch in batches:
            scores, own_positive, collision = self._batch_scores(retrieval_model, batch)
            count = len(batch.examples)
            total_loss += float(self._loss_from_scores(scores, own_positive, collision, temperature).item()) * count
            for name, value in self._metrics_from_scores(scores, own_positive, collision).items():
                totals[name] += value * count
            total_examples += count
        divisor = max(1, total_examples)
        return total_loss / divisor, {name: value / divisor for name, value in totals.items()}

    # ----------------------------------------------------------------------------- training
    def _dataset_indexes(self, *example_lists: list[LabeledRetrievalQAExample]) -> DatasetIndexes:
        dataset_ids = {example.dataset_id for examples in example_lists for example in examples}
        indexes = {dataset_id: self.harness.dataset_manager._get_or_create_index(dataset_id) for dataset_id in dataset_ids}
        return next(iter(indexes.values())) if len(indexes) == 1 else indexes

    @staticmethod
    def _make_optimizer(config: RetrievalTrainingConfig, groups: dict[str, list[torch.nn.Parameter]]) -> torch.optim.AdamW:
        learning_rates = {"lora": config.lora_learning_rate, "ac": config.ac_learning_rate, "head": config.head_learning_rate}
        param_groups = []
        for name, params in groups.items():                                               # no decay on norms, biases, scalars
            decay = [p for p in params if p.ndim >= 2]
            no_decay = [p for p in params if p.ndim < 2]
            if decay:
                param_groups.append({"params": decay, "lr": learning_rates[name], "name": name, "weight_decay": config.weight_decay})
            if no_decay:
                param_groups.append({"params": no_decay, "lr": learning_rates[name], "name": name, "weight_decay": 0.0})
        return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    @staticmethod
    def _make_scheduler(config: RetrievalTrainingConfig, optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        warmup_steps = int(round(config.warmup_fraction * total_steps))

        def factor(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, progress)
            if config.schedule == "cosine":
                return 0.5 * (1 + math.cos(math.pi * progress))
            return 1 - progress
        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    def train(
        self,
        training_config: RetrievalTrainingConfig,
        retrieval_model: RetrievalModel,
        retrieval_reporter: RetrievalReporter,
        training_data: list[LabeledRetrievalQAExample],
        reporting_data: list[LabeledRetrievalQAExample],
        validation_data: list[LabeledRetrievalQAExample] | None = None,
    ) -> RetrievalTrainingStats:
        """
        Holds base residency for the whole run. Order: ensure_resident -> batch size (config or
        recommended) -> probe -> epochs. Raises before the first real step if the probe cannot fit.
        """
        config = training_config
        reporter = retrieval_reporter
        stats = RetrievalTrainingStats(num_examples=len(training_data), num_epochs=config.epochs)
        validation_data = validation_data or []
        assert training_data, "No training data."
        torch.manual_seed(config.seed)
        retrieval_model.ensure_resident()
        device = retrieval_model.device
        dataset_index = self._dataset_indexes(training_data, reporting_data, validation_data)
        groups = retrieval_model.trainable_parameter_groups()
        assert groups, "Nothing to train: no LoRA, AC model or head."
        all_parameters = [parameter for params in groups.values() for parameter in params]

        # Batch size: config or the sizing from the heaviest real batch, then the probe on that batch.
        retrieval_model.set_training_mode(True, config.gradient_checkpointing)
        if config.batch_size is None:
            sizing = recommended_batch_size(
                retrieval_model, training_data, dataset_index,
                config.gradient_checkpointing, config.memory_headroom_fraction, device,
            )
        else:
            sizing = configured_batch_sizing(config.batch_size, training_data, dataset_index, retrieval_model)
        batch_sizing = sizing.explanation

        def probe_step(batch: RetrievalBatch) -> None:
            try:
                loss = self.contrastive_loss(retrieval_model, batch, config.temperature)
                loss.backward()
            finally:
                for parameter in all_parameters:
                    parameter.grad = None
        batch_size, probe_attempts = probe_batch_size(probe_step, retrieval_model, sizing.probe_examples, dataset_index)
        if probe_attempts:
            batch_sizing += f"\nprobe passed at {batch_size} on attempt {probe_attempts}"
        stats.batch_size, stats.probe_attempts, stats.batch_sizing = batch_size, probe_attempts, batch_sizing
        print(f"Batch sizing:\n{batch_sizing}")

        steps_per_epoch = math.ceil(len(training_data) / batch_size)
        total_steps = steps_per_epoch * config.epochs
        reporting_interval = max(1, round(config.reporting_fraction * steps_per_epoch))
        optimizer = self._make_optimizer(config, groups)
        scheduler = self._make_scheduler(config, optimizer, total_steps)
        reporting_batches = fixed_batches(reporting_data, dataset_index, None, config.seed)
        validation_batches = fixed_batches(validation_data, dataset_index, max(1, len(reporting_data)), config.seed)
        counts = {
            "training_examples": len(training_data), "validation_examples": len(validation_data),
            "reporting_examples": len(reporting_data), "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "batch_size": batch_size,
        }
        lora_config = self.harness.module_manager.get_lora_config(retrieval_model.lora_name) if retrieval_model.lora_name else None
        reporter.initialize_run(config, retrieval_model, lora_config, counts, batch_sizing, groups)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def peak_memory() -> int | None:
            return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None

        def report_progress(step: int, progress: float) -> None:
            if not reporting_batches:
                return
            start = time.time()
            retrieval_model.set_training_mode(False)
            loss, metrics = self._evaluate(retrieval_model, reporting_batches, config.temperature)
            retrieval_model.set_training_mode(True, config.gradient_checkpointing)
            stats.total_reporting_time += time.time() - start
            stats.reporting_losses.append((progress, loss))
            stats.reporting_metrics.append((progress, metrics))
            reporter.report_reporting_point(step, progress, loss, metrics)

        run_start = time.time()
        step = 0
        last_report_step = -1
        for epoch in range(1, config.epochs + 1):
            rng = random.Random(config.seed + epoch)
            epoch_start = time.time()
            epoch_shapes: list[tuple[int, int, int, int, int]] = []
            epoch_steps = 0
            for batch in make_batches(training_data, dataset_index, batch_size, rng):
                step_start = time.time()
                real_before, padded_before = retrieval_model.real_tokens_embedded, retrieval_model.padded_tokens_embedded
                forwards_before = retrieval_model.forwards_embedded
                loss = self.contrastive_loss(retrieval_model, batch, config.temperature)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_parameters, config.max_grad_norm)
                optimizer.step()
                learning_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}  # the rates this step used
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                epoch_steps += 1
                step_time = time.time() - step_start
                loss_value = float(loss.item())
                distinct_candidates = len(flatten_candidates(batch)[0])
                shape = (
                    len(batch.examples), distinct_candidates,
                    retrieval_model.real_tokens_embedded - real_before,
                    retrieval_model.padded_tokens_embedded - padded_before,
                    retrieval_model.forwards_embedded - forwards_before,
                )
                stats.step_losses.append((step, loss_value))
                stats.step_learning_rates.append((step, learning_rates))
                stats.step_batch_shapes.append(shape)
                stats.step_times.append(step_time)
                epoch_shapes.append(shape)
                stats.total_train_time += step_time
                progress = step / steps_per_epoch
                reporter.report_training_step(
                    step, progress, loss_value, learning_rates, time.time() - run_start, shape[2], step_time, peak_memory(),
                )
                if step % reporting_interval == 0 and epoch_steps < steps_per_epoch:
                    report_progress(step, progress)
                    last_report_step = step
                    reporter.report_epoch(f"{epoch} (running)", epoch_steps, epoch_shapes, time.time() - epoch_start, peak_memory(), running=True)
                    reporter.render(force=True)
                else:
                    reporter.render()
            # Epoch end: reporting point, full validation, throughput row.
            epoch_time = time.time() - epoch_start
            if last_report_step != step:
                report_progress(step, step / steps_per_epoch)
                last_report_step = step
            if validation_batches:
                start = time.time()
                retrieval_model.set_training_mode(False)
                validation_loss, validation_metrics = self._evaluate(retrieval_model, validation_batches, config.temperature)
                retrieval_model.set_training_mode(True, config.gradient_checkpointing)
                stats.total_validation_time += time.time() - start
                stats.validation_losses.append((epoch, validation_loss))
                stats.validation_metrics.append((epoch, validation_metrics))
                reporter.report_validation(epoch, validation_loss, validation_metrics)
            reporter.report_epoch(str(epoch), epoch_steps, epoch_shapes, epoch_time, peak_memory(), running=False)
            reporter.render(force=True)

        stats.num_steps = step
        stats.peak_memory_bytes = peak_memory() or 0
        retrieval_model.set_training_mode(False)
        reporter.report_finished(step, time.time() - run_start)
        return stats
