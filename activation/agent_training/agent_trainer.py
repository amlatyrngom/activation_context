"""
Agent training: weighted trajectories in, a trained LoRA checkpoint out.

The interface is `select_agent_runs` (a selection function turns each group of rollouts into weighted
items) then `train(model_name, lora_name, ...)` for one round of gradient steps on that adapter, then
`module_manager.exchange_lora` so the serving engine sees the checkpoint; evaluation is the rollout
manager. Standard and "just works": the objective is the clipped surrogate

    L = -mean_i (1/|a_i|) sum_t min(s_t A_i, clip(s_t, 1-eps, 1+eps) A_i),   s_t = pi_theta(a_t|prefix) / pi_old(a_t|prefix)

over the sampled assistant tokens of every trajectory (prompts and tool outputs masked), with A_i the
item's weight. pi_old is the engine's stored log-prob of each sampled token; for `ignore_logprobs`
items (teacher or hinted trajectories) it is the current policy at the start of the round, so the ratio
starts at 1 and the first step is weighted supervised fine-tuning. RAFT++, group-mean advantages,
Reinforce-Rej and W-REINFORCE are all selection functions on top of this one loss.

One trajectory is one training sequence (the agent records its token segments), gradient
checkpointing is on, micro-batches are packed by token budget and accumulated into
`updates_per_round` mini-batch steps, and the GPU is shared with the serving engine through sleep
mode: the engine sleeps while the model trains; the model moves to host RAM afterwards and the next
rollout wakes the engine. The messy helpers live in agent_training_utils.
"""
from __future__ import annotations

import random
import time
import typing as t
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field

import torch

from activation.agent import AgentRunResult
from activation.harness import SOURCE_DEVICE, TARGET_DEVICE

from .agent_training_config import AgentTrainingConfig, AgentTrainingStats
from .agent_training_reporter import AgentTrainingReporter
from ..autotuners import get_agent_training_autotuning_variables, configure_fla_runtime
from .agent_training_utils import (
    Collated,
    TrainingExample,
    build_example,
    check_packing_support,
    clipped_surrogate,
    collate,
    disable_dropout,
    micro_batches,
    optimizer_state_to,
    packed_attention_masks,
    persistent_optimizer,
    packing_spacer_tokens,
    set_attention_implementation,
    set_checkpointing,
    set_learning_rates,
    sampled_logprobs,
)

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime


@dataclass
class AgentTrainingItem:
    run_results: AgentRunResult          # one agent's run (top level or a subagent's; selection functions unroll)
    model_name: str
    lora_name: str                       # the adapter this item trains
    weight: float                        # the advantage A: sign = direction, magnitude = strength; never 0
    ignore_logprobs: bool = False        # teacher / hinted data: pi_old := the current policy at round start
    group_key: str = field(default_factory=lambda: uuid.uuid4().hex[:8])   # select_agent_runs stamps one key per group


def unroll(result: AgentRunResult) -> list[AgentRunResult]:
    """The run's compacted segments, the run, and every subagent run beneath it (with theirs), depth first."""
    out = []
    for segment in list(result.compactions) + [result]:
        out.append(segment)
        for child in segment.subagent_results:
            out.extend(unroll(child))
    return out


class AgentTrainer:
    def __init__(self, harness: "HarnessRuntime", config: AgentTrainingConfig | None = None):
        self.harness = harness
        self.config = config or AgentTrainingConfig()
        self.rounds_done: dict[str, int] = {}
        self.updates_done: dict[str, int] = {}                    # optimizer steps so far per adapter (the warm-up clock)
        self.optimizers: dict[str, torch.optim.AdamW] = {}        # one per adapter, kept across rounds (moments survive a round boundary)

    # ----------------------------------------------------------------------------- selection
    def select_agent_runs(
        self,
        run_results: list[list[AgentRunResult]],
        selection_function: t.Callable[[list[AgentRunResult]], list[AgentTrainingItem]],
    ) -> dict[tuple[str, str], list[AgentTrainingItem]]:
        """Applies the function to every group, stamps its items with one group_key, buckets by (model_name, lora_name)."""
        selected: dict[tuple[str, str], list[AgentTrainingItem]] = {}
        for group in run_results:
            group_key = uuid.uuid4().hex[:8]
            for item in selection_function(list(group)):
                assert item.weight != 0.0, "selection functions must not emit zero-weight items"
                item.group_key = group_key
                selected.setdefault((item.model_name, item.lora_name), []).append(item)
        return selected

    # ----------------------------------------------------------------------------- training
    def train(
        self,
        model_name: str,
        lora_name: str,
        training_data: list[AgentTrainingItem],
        reporting_data: t.Sequence[AgentTrainingItem] = (),
        reporter: AgentTrainingReporter | None = None,
    ) -> AgentTrainingStats:
        """
        One round on `lora_name`: engine to sleep, base + adapter on the GPU, one clipped-surrogate pass
        over the items in `updates_per_round` steps (the adapter's AdamW persists across rounds), the
        objective on `reporting_data` without gradient, the adapter checkpointed, the model moved to host RAM. Does not exchange: the caller decides when
        the engine sees the new adapter.
        """
        config = self.config
        assert training_data, "No training data."
        assert all(item.model_name == model_name and item.lora_name == lora_name for item in training_data), "items of another (model, lora)"
        module_manager = self.harness.module_manager
        loaded_model = self.harness.loaded_models[model_name]
        round_index = self.rounds_done.get(lora_name, 0)
        stats = AgentTrainingStats(round_index=round_index, items=len(training_data))
        started = time.time()
        torch.manual_seed(config.seed + round_index)

        # Examples: one sequence per trajectory.
        examples: list[TrainingExample] = []
        for index, item in enumerate(training_data):
            example = build_example(item, index)
            if example is None:
                stats.dropped_empty += 1
            elif len(example.token_ids) > config.max_example_tokens:
                stats.dropped_too_long += 1
            else:
                examples.append(example)
        assert examples, f"every item was dropped ({stats.dropped_too_long} too long, {stats.dropped_empty} without sampled tokens)"
        stats.examples = len(examples)
        stats.total_tokens = sum(len(example.token_ids) for example in examples)
        stats.assistant_tokens = sum(example.num_loss_tokens for example in examples)
        stats.positive_weight_sum = sum(example.weight for example in examples if example.weight > 0)
        stats.negative_weight_sum = sum(example.weight for example in examples if example.weight < 0)
        reporting_examples = [example for example in (build_example(item, index) for index, item in enumerate(reporting_data))
                              if example is not None and len(example.token_ids) <= config.max_example_tokens]
        if reporter is not None:
            reporter.initialize_round(round_index, config, len(training_data), len(examples), stats.dropped_too_long + stats.dropped_empty, stats.total_tokens)

        # The GPU: engine asleep, base + adapter resident, checkpointing on, dropout off, base frozen.
        if loaded_model.vllm_model is not None:
            loaded_model.engine_to_device(SOURCE_DEVICE)                  # sleep: weights to host RAM, the GPU to the trainer
        variables = get_agent_training_autotuning_variables(loaded_model.model_config, config, device=TARGET_DEVICE)
        self._logits_chunk_tokens = variables["logits_chunk_tokens"]
        stats.autotuning = {**variables["provenance"], "logits_chunk_tokens": self._logits_chunk_tokens,
                            "gradient_checkpointing_min_tokens": variables["gradient_checkpointing_min_tokens"]}
        module_manager.ensure_lora(lora_name)
        base = loaded_model.model
        device = next(base.parameters()).device
        base.train()
        previous_attn = base.config._attn_implementation
        if config.attn_implementation and config.attn_implementation != previous_attn:
            set_attention_implementation(base, config.attn_implementation)
        if config.pack_micro_batches:
            check_packing_support(base)
        self._spacer_tokens = packing_spacer_tokens(base)
        min_tokens = variables["gradient_checkpointing_min_tokens"]
        set_checkpointing(base, config, min_tokens, min_tokens)
        disable_dropout(base)
        parameters = module_manager.lora_parameters(lora_name)
        assert parameters, f"LoRA {lora_name!r} has no trainable parameters"
        optimizer = persistent_optimizer(self.optimizers, lora_name, [{"params": parameters, "lr": config.learning_rate}], device,
                                         betas=config.adam_betas, eps=config.adam_eps, weight_decay=config.weight_decay)
        pad_token_id = loaded_model.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = loaded_model.tokenizer.eos_token_id or 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        kernel_scope = ExitStack()
        try:
            kernel_scope.enter_context(configure_fla_runtime(variables["fla_config"]))
            # pi_old for items without usable engine log-probs: the current policy, before any step.
            needing = [example for example in examples + reporting_examples if example.needs_old_logprobs]
            if needing:
                self._fill_old_logprobs(loaded_model, lora_name, needing, pad_token_id, device)
                stats.recomputed_old_logprobs = sum(1 for example in examples if example.needs_old_logprobs)

            # One pass, updates_per_round mini-batches, token-budgeted micro-batches with accumulation.
            order = list(range(len(examples)))
            random.Random(config.seed + round_index).shuffle(order)
            steps = max(1, min(config.updates_per_round, len(order)))
            per_step = -(-len(order) // steps)
            for step in range(steps):
                step_started = time.time()
                mini = order[step * per_step:(step + 1) * per_step]
                if not mini:
                    break
                optimizer.zero_grad(set_to_none=True)
                totals = {"loss": 0.0, "mean_ratio": 0.0, "clip_fraction": 0.0, "mean_logprob": 0.0, "tokens": 0, "loss_tokens": 0}
                micro_batch_times = []
                for group in micro_batches([examples[index] for index in mini], config.micro_batch_tokens, config.pad_free_micro_batches,
                                           config.pack_micro_batches):
                    batch = collate(examples, [mini[i] for i in group], pad_token_id, device, config.length_multiple,
                                    packed=config.pack_micro_batches, spacer_tokens=self._spacer_tokens)
                    set_checkpointing(base, config, int(batch.input_ids.numel()), min_tokens)
                    micro_batch_started = time.time()
                    loss, batch_stats, loss_tokens = self._objective(loaded_model, lora_name, batch, len(mini))
                    loss.backward()
                    micro_batch_times.append([int(batch.input_ids.numel()), round(time.time() - micro_batch_started, 2)])
                    totals["loss"] += float(loss.detach())
                    for key in ("mean_ratio", "clip_fraction", "mean_logprob"):
                        totals[key] += batch_stats[key] * loss_tokens
                    totals["tokens"] += int(batch.attention_mask.sum())
                    totals["loss_tokens"] += loss_tokens
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm))
                set_learning_rates(optimizer, self.updates_done.get(lora_name, 0), config.warmup_updates)
                optimizer.step()
                self.updates_done[lora_name] = self.updates_done.get(lora_name, 0) + 1
                divisor = max(1, totals["loss_tokens"])
                seconds = time.time() - step_started
                stats.steps += 1
                stats.loss.append(totals["loss"])
                stats.mean_ratio.append(totals["mean_ratio"] / divisor)
                stats.clip_fraction.append(totals["clip_fraction"] / divisor)
                stats.mean_logprob.append(totals["mean_logprob"] / divisor)
                stats.grad_norm.append(grad_norm)
                stats.step_seconds.append(seconds)
                slowest = sorted(micro_batch_times, key=lambda pair: -pair[1])[:3]
                stats.slow_micro_batches.append(slowest)
                print(f"AgentTrainer - round {round_index} step {step + 1}/{steps}: loss {totals['loss']:.4f} ratio {stats.mean_ratio[-1]:.3f} "
                      f"clip {stats.clip_fraction[-1]:.3f} logprob {stats.mean_logprob[-1]:.3f} grad {grad_norm:.3f} "
                      f"{totals['tokens']} tokens in {seconds:.1f}s ({len(micro_batch_times)} micro-batches, slowest "
                      f"{', '.join(f'{t}s@{n}' for n, t in slowest)})", flush=True)
                if reporter is not None:
                    reporter.report_step(round_index, step + 1, totals["loss"], stats.mean_ratio[-1], stats.clip_fraction[-1], stats.mean_logprob[-1],
                                         grad_norm, totals["tokens"], seconds)

            # The objective on the reporting items, no gradient, after the round.
            if reporting_examples:
                with torch.no_grad():
                    total = 0.0
                    for group in micro_batches(reporting_examples, config.micro_batch_tokens, config.pad_free_micro_batches, config.pack_micro_batches):
                        batch = collate(reporting_examples, group, pad_token_id, device, config.length_multiple,
                                        packed=config.pack_micro_batches, spacer_tokens=self._spacer_tokens)
                        loss, _, _ = self._objective(loaded_model, lora_name, batch, len(reporting_examples))
                        total += float(loss)
                stats.reporting_loss = total

            if config.checkpoint_every_round:
                stats.checkpoint_path = module_manager.save_lora(model_name, lora_name)
        finally:
            kernel_scope.close()
            optimizer_state_to(optimizer, SOURCE_DEVICE)
            base.eval()
            if base.is_gradient_checkpointing:
                base.gradient_checkpointing_disable()
            if base.config._attn_implementation != previous_attn:
                set_attention_implementation(base, previous_attn)
            if device.type == "cuda":
                stats.peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
            loaded_model.model_to_device(SOURCE_DEVICE)          # off the GPU; the next rollout wakes the engine
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.rounds_done[lora_name] = round_index + 1
        stats.duration_s = time.time() - started
        if reporter is not None:
            reporter.report_round(stats)
            reporter.finish()
        print(f"AgentTrainer - round {round_index} done: {stats.summarize()}", flush=True)
        return stats

    # ----------------------------------------------------------------------------- internals
    def _hidden_and_logprobs(self, loaded_model, lora_name: str, batch: Collated) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = loaded_model.model
        inputs_embeds = base.get_input_embeddings()(batch.input_ids)
        kwargs = dict(self.config.decoder_kwargs)
        attention_mask = batch.model_attention_mask
        if batch.packed:
            kwargs["cu_seq_lens_q"] = batch.cu_seq_lens                      # the gated-delta layers restart their state per example
            attention_mask = packed_attention_masks(base, inputs_embeds, batch)
        hidden = loaded_model.decoder_forward(inputs_embeds, attention_mask, batch.position_ids, lora_name, **kwargs)
        return sampled_logprobs(hidden, base.get_output_embeddings(), batch, getattr(self, "_logits_chunk_tokens", self.config.logits_chunk_tokens or 64), self.config.fp32_head_matmul)

    def _objective(self, loaded_model, lora_name: str, batch: Collated, num_examples: int) -> tuple[torch.Tensor, dict[str, float], int]:
        """The negated clipped surrogate of this micro-batch, scaled so the mini-batch's total is a mean over its examples."""
        logprobs, segments, _ = self._hidden_and_logprobs(loaded_model, lora_name, batch)
        old = batch.old_logprobs[batch.loss_mask]
        weights = batch.weights[segments]
        surrogate, batch_stats = clipped_surrogate(logprobs, old, weights, self.config.clip_epsilon)
        count = len(batch.example_indices)
        per_sample = torch.zeros(count, device=logprobs.device, dtype=torch.float32).index_add_(0, segments, surrogate)
        counts = torch.zeros(count, device=logprobs.device, dtype=torch.float32).index_add_(0, segments, torch.ones_like(surrogate))
        loss = -(per_sample / counts.clamp(min=1)).sum() / max(1, num_examples)
        return loss, batch_stats, int(logprobs.shape[0])

    @torch.no_grad()
    def _fill_old_logprobs(self, loaded_model, lora_name: str, examples: list[TrainingExample], pad_token_id: int, device) -> None:
        """pi_old := the current policy for examples without engine log-probs (teacher / hinted / unrecorded)."""
        config = self.config
        for group in micro_batches(examples, config.micro_batch_tokens, config.pad_free_micro_batches, config.pack_micro_batches):
            batch = collate(examples, group, pad_token_id, device, config.length_multiple, packed=config.pack_micro_batches,
                            spacer_tokens=self._spacer_tokens)
            logprobs, segments, _ = self._hidden_and_logprobs(loaded_model, lora_name, batch)
            positions = batch.loss_mask.nonzero(as_tuple=True)[1]
            for ordinal, example_index in enumerate(batch.example_indices):
                mask = segments == ordinal
                values = logprobs[mask].tolist()
                for position, value in zip(positions[mask].tolist(), values):
                    examples[example_index].old_logprobs[position - batch.segment_offsets[ordinal]] = value
