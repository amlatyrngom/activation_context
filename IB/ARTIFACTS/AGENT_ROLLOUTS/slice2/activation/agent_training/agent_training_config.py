"""
Configuration and statistics of one agent-training round.

`AgentTrainingConfig` carries standard, LoRA-appropriate defaults for on-policy clipped training:
PPO clip 0.2 on the token ratio, a handful of gradient steps per round over one pass of the data
(samples go stale after that), AdamW at 2e-5 with (0.9, 0.95) and no weight decay on the adapter,
gradient-norm clip 1.0, bf16 model with the ratio in fp32, activation checkpointing, token-budgeted
micro-batches with gradient accumulation, and dropout off (an on-policy ratio needs a deterministic
policy). `AgentTrainingStats` records what a round did, per step, for the report and the tests.
"""
from dataclasses import dataclass, field


@dataclass
class AgentTrainingConfig:
    # The objective.
    clip_epsilon: float = 0.2                  # PPO clip on the token ratio pi_theta / pi_old
    updates_per_round: int = 4                 # gradient steps per train() call: the data is split into this many mini-batches, one pass
    max_example_tokens: int = 16_384           # longer trajectories are dropped (counted), never truncated
    # Optimisation (LoRA values; a full fine-tune would sit two orders of magnitude lower).
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    # Memory.
    micro_batch_tokens: int = 32_768          # token budget per micro-batch when examples are packed with padding (pad_free_micro_batches=False)
    pad_free_micro_batches: bool = True       # one example per micro-batch, no padding mask: sdpa takes its causal (flash) path, nothing is wasted on pads
    length_multiple: int = 1                  # sequence length rounded up to a multiple (tail pads carry no loss). 1: rounding to 512 cost 30% on the node (profile), and no per-shape compile cost was measurable           # padded tokens per forward/backward; gradient accumulation fills the mini-batch
    logits_chunk_tokens: int = 2_048           # loss positions per lm_head chunk (the full logits of a long sequence would not fit)
    gradient_checkpointing: bool = True
    # Bookkeeping.
    seed: int = 0
    checkpoint_every_round: bool = True        # adapter saved under LORAS/<lora_name>/round_<n> (+ latest) after the round


@dataclass
class AgentTrainingStats:
    round_index: int = 0
    items: int = 0                             # AgentTrainingItems received
    examples: int = 0                          # sequences trained on
    dropped_too_long: int = 0                  # over max_example_tokens
    dropped_empty: int = 0                     # no token segments or no sampled tokens (a run recorded without sampling)
    recomputed_old_logprobs: int = 0           # ignore_logprobs items whose pi_old came from a no-grad pass
    total_tokens: int = 0                      # tokens read (prompts included)
    assistant_tokens: int = 0                  # tokens that carried loss
    steps: int = 0
    loss: list[float] = field(default_factory=list)            # per step
    mean_ratio: list[float] = field(default_factory=list)      # per step, pi_theta / pi_old over loss tokens (1.0 at step 1 when everything aligns)
    clip_fraction: list[float] = field(default_factory=list)   # per step, share of loss tokens outside 1 +- epsilon
    mean_logprob: list[float] = field(default_factory=list)    # per step, mean log pi_theta of the sampled tokens (an entropy proxy)
    grad_norm: list[float] = field(default_factory=list)       # per step, before clipping
    step_seconds: list[float] = field(default_factory=list)
    reporting_loss: float | None = None        # the objective on reporting_data after the round, no gradient
    positive_weight_sum: float = 0.0
    negative_weight_sum: float = 0.0
    duration_s: float = 0.0
    checkpoint_path: str | None = None
    peak_memory_bytes: int = 0

    def summarize(self) -> dict:
        """One flat dict for logs and tests."""
        return {
            "round_index": self.round_index, "items": self.items, "examples": self.examples,
            "dropped_too_long": self.dropped_too_long, "dropped_empty": self.dropped_empty,
            "recomputed_old_logprobs": self.recomputed_old_logprobs,
            "total_tokens": self.total_tokens, "assistant_tokens": self.assistant_tokens, "steps": self.steps,
            "first_loss": self.loss[0] if self.loss else None, "last_loss": self.loss[-1] if self.loss else None,
            "first_mean_ratio": self.mean_ratio[0] if self.mean_ratio else None,
            "first_clip_fraction": self.clip_fraction[0] if self.clip_fraction else None,
            "mean_logprob_first": self.mean_logprob[0] if self.mean_logprob else None,
            "mean_logprob_last": self.mean_logprob[-1] if self.mean_logprob else None,
            "reporting_loss": self.reporting_loss,
            "positive_weight_sum": self.positive_weight_sum, "negative_weight_sum": self.negative_weight_sum,
            "duration_s": round(self.duration_s, 1), "tokens_per_s": round(self.total_tokens / self.duration_s, 1) if self.duration_s else 0.0,
            "checkpoint_path": self.checkpoint_path, "peak_memory_gb": round(self.peak_memory_bytes / 2**30, 2),
        }
