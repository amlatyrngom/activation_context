"""
Configuration and statistics of a retrieval training run.

`RetrievalTrainingConfig` carries the reviewer-audited defaults (Qwen3-Embedding / Sentence-Transformers
recipe: LoRA 1e-4, AC 5e-4, head 1e-3, weight decay 0.01 on matrices only, AdamW (0.9, 0.999) eps 1e-8,
10 % warmup then cosine to zero, clip 1.0, temperature 0.02 on view-mean MaxSim, checkpointing on).
`RetrievalTrainingStats` records what the run did and `summarize()` flattens it so a 1000-query run
extrapolates to 50k by arithmetic.
"""
import typing as t
from dataclasses import dataclass, field


@dataclass
class RetrievalTrainingConfig:
    epochs: int = 1
    batch_size: int | None = None           # examples per step; None: recommended_batch_size
    lora_learning_rate: float = 1e-4
    ac_learning_rate: float = 5e-4
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    schedule: t.Literal["cosine", "linear"] = "cosine"
    max_grad_norm: float = 1.0
    temperature: float = 0.02
    gradient_checkpointing: bool = True     # base and AC model
    memory_headroom_fraction: float = 0.1
    reporting_fraction: float = 0.1         # of an epoch
    seed: int = 0
    baseline_model_name: str | None = None  # a harness embedding model scored on the validation batches as a trained reference
    reference_frozen_base: bool = True      # also score the frozen base alone (no LoRA, AC or head) as the floor


@dataclass
class RetrievalTrainingStats:
    num_examples: int = 0
    num_epochs: int = 0
    num_steps: int = 0
    batch_size: int = 0
    probe_attempts: int = 0
    total_train_time: float = 0.0
    total_reporting_time: float = 0.0
    step_losses: list[tuple[int, float]] = field(default_factory=list)                  # (step, loss)
    step_learning_rates: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    step_batch_shapes: list[tuple[int, int, int, int, int]] = field(default_factory=list)  # (examples, candidates, real tokens, padded tokens, forwards)
    step_times: list[float] = field(default_factory=list)
    reporting_losses: list[tuple[float, float]] = field(default_factory=list)           # (progress in epochs, loss)
    reporting_metrics: list[tuple[float, dict[str, float]]] = field(default_factory=list)  # (progress, {in-batch rank-1, mrr@10, ndcg@10})
    validation_losses: list[tuple[int, float]] = field(default_factory=list)            # (epoch, loss)
    validation_metrics: list[tuple[int, dict[str, float]]] = field(default_factory=list)  # (epoch, in-batch metrics)
    total_validation_time: float = 0.0
    reference_losses: dict[str, float] = field(default_factory=dict)                    # {reference name: validation loss}
    reference_metrics: dict[str, dict[str, float]] = field(default_factory=dict)        # {reference name: in-batch metrics}
    peak_memory_bytes: int = 0
    batch_sizing: str = ""                                                              # the printed arithmetic

    def summarize(self) -> dict:
        """One flat dict built so a 1000-query run extrapolates to 50k by arithmetic."""
        def average(values: list) -> float:
            return float(sum(values) / len(values)) if values else 0.0
        examples = sum(shape[0] for shape in self.step_batch_shapes)
        candidates = sum(shape[1] for shape in self.step_batch_shapes)
        real_tokens = sum(shape[2] for shape in self.step_batch_shapes)
        padded_tokens = sum(shape[3] for shape in self.step_batch_shapes)
        train_time = self.total_train_time
        losses = [loss for _, loss in self.step_losses]
        reporting = [loss for _, loss in self.reporting_losses]
        return {
            # Counts.
            "num_examples": self.num_examples,
            "num_epochs": self.num_epochs,
            "num_steps": self.num_steps,
            "batch_size": self.batch_size,
            "probe_attempts": self.probe_attempts,
            "total_candidates": candidates,
            "avg_candidates_per_example": candidates / examples if examples else 0.0,
            "avg_tokens_per_example": real_tokens / examples if examples else 0.0,
            # Time.
            "total_train_time": train_time,
            "train_time_per_epoch": train_time / self.num_epochs if self.num_epochs else 0.0,
            "avg_step_time": average(self.step_times),
            "total_reporting_time": self.total_reporting_time,
            "total_validation_time": self.total_validation_time,
            "reporting_time_fraction": self.total_reporting_time / train_time if train_time else 0.0,
            # Throughput (training steps only; reporting and validation excluded).
            "examples_per_s": examples / train_time if train_time else 0.0,
            "candidates_per_s": candidates / train_time if train_time else 0.0,
            "tokens_per_s": real_tokens / train_time if train_time else 0.0,
            "padded_tokens_per_s": padded_tokens / train_time if train_time else 0.0,
            "avg_padding_fraction": 1 - real_tokens / padded_tokens if padded_tokens else 0.0,
            "forwards_per_step": average([shape[4] for shape in self.step_batch_shapes]),
            "seconds_per_10k_examples": 10_000 * train_time / examples if examples else 0.0,
            # Memory.
            "peak_memory_gb": self.peak_memory_bytes / 1024 ** 3,
            "batch_sizing": self.batch_sizing,
            # Loss trend.
            "first_step_loss": losses[0] if losses else None,
            "last_step_loss": losses[-1] if losses else None,
            "min_step_loss": min(losses) if losses else None,
            "first_reporting_loss": reporting[0] if reporting else None,
            "last_reporting_loss": reporting[-1] if reporting else None,
            "min_reporting_loss": min(reporting) if reporting else None,
            "last_reporting_metrics": self.reporting_metrics[-1][1] if self.reporting_metrics else None,
            "validation_losses": list(self.validation_losses),
            "validation_metrics": list(self.validation_metrics),
            "reference_losses": dict(self.reference_losses),
            "reference_metrics": dict(self.reference_metrics),
        }
