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
    max_example_tokens: int = 52_000           # longer trajectories are dropped (counted), never truncated; matches the engine's context (50k + buffer)
    # Optimisation (LoRA values; a full fine-tune would sit two orders of magnitude lower).
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_updates: int = 0                    # linear warm-up of the learning rate over this many updates, counted across rounds (0: none)
    # Memory.
    micro_batch_tokens: int = 32_768          # how many tokens may share one micro-batch when examples are packed (pack_micro_batches) or padded (pad_free_micro_batches=False); unused pad-free (one example each). Never drops or splits: an example over the budget gets a micro-batch of its own, up to max_example_tokens
    pad_free_micro_batches: bool = True       # one example per micro-batch, no padding mask: sdpa takes its causal (flash) path, nothing is wasted on pads
    # Packed rows (measured exact with the spacers, see agent_training_utils.collate): several examples in one row under
    # micro_batch_tokens with per-example attention and recurrent state (needs the fla kernels). Off by default: on the
    # RTX 6000 Pro packed rows under flex attention run at 4,000 tok/s against 4,250 for single long rows, and only the
    # short tail gains; enable with attn_implementation="flex_attention" and decoder_kwargs={"kernel_options": {"BLOCK_M1": 16,
    # "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 16}} (flex's default backward blocks need more shared memory than sm_120 has
    # for head_dim 256); sdpa packs too but with a dense block mask (1,800 tok/s).
    pack_micro_batches: bool = False
    attn_implementation: str | None = None    # set on the base model for the round and restored after (None: leave it)
    decoder_kwargs: dict = field(default_factory=dict)  # extra keywords for every decoder pass, e.g. flex attention's kernel_options
    length_multiple: int = 1                  # sequence length rounded up to a multiple (tail pads carry no loss). 1: rounding to 512 cost 30% on the node (profile), and no per-shape compile cost was measurable
    logits_chunk_tokens: int | None = None           # loss positions per lm_head chunk (the full logits of a long sequence would not fit)
    fp32_head_matmul: bool = False             # the head's GEMM in fp32 on the SIMT kernels instead of bf16 operands with fp32 accumulation (same products, 6x slower; a reference)
    gradient_checkpointing: bool = True
    # Recompute activations only for micro-batches of at least this many tokens; shorter ones keep them (+35% on the RTX 6000 Pro,
    # 6,334 against 4,677 tok/s at 5k tokens) at about 9 GB per 1k tokens on top of 15 GB (60 GB at 5k). Auto/None
    # without a compatible whole-model profile keeps recomputation on. 0 always recomputes.
    gradient_checkpointing_min_tokens: int | None = None
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
    slow_micro_batches: list[list] = field(default_factory=list)   # per step: the three slowest micro-batches as [tokens, seconds] (first-encounter kernel compiles show here)
    reporting_loss: float | None = None        # the objective on reporting_data after the round, no gradient
    positive_weight_sum: float = 0.0
    negative_weight_sum: float = 0.0
    duration_s: float = 0.0
    checkpoint_path: str | None = None
    peak_memory_bytes: int = 0
    autotuning: dict = field(default_factory=dict)

    def summarize(self) -> dict:
        """One flat dict for logs and tests."""
        return {
            "autotuning": self.autotuning,
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
