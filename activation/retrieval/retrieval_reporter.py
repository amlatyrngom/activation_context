"""
The retrieval-training report: which widgets exist and what the trainer feeds them.

`HtmlReporter` (common/reporting.py) owns the page, the widgets and the atomic writes; this
subclass owns the layout for a retrieval training run and turns the trainer's raw numbers into
status fields, curve points and table rows, so the trainer only calls named report_* methods.
"""
import json
import typing as t
from dataclasses import asdict

from ..common.reporting import HtmlReporter, format_rate, format_seconds

if t.TYPE_CHECKING:
    from ..harness.module_manager import LoraConfig
    from .retrieval_model import RetrievalModel
    from .retrieval_training_config import RetrievalTrainingConfig

METRIC_NAMES = ("in-batch rank-1", "in-batch mrr@10", "in-batch ndcg@10")
"""In-batch ranking metrics, named so in every plot and log: the candidate pool is the batch's distinct
candidates (each query's own positives and hard negatives plus the other queries' candidates as distractors)
with same-document collisions masked; relevance is binary over the query's own positives. Nothing beyond the
batch is embedded; corpus-level retrieval metrics belong to the slice 2 index."""


class RetrievalReporter(HtmlReporter):
    def __init__(
        self,
        folder: str,
        title: str,
        description: str,
        refresh_seconds: int = 30,
        min_render_interval_seconds: float = 5.0,
    ):
        super().__init__(
            folder, title, description, eyebrow="Retrieval training report",
            refresh_seconds=refresh_seconds, min_render_interval_seconds=min_render_interval_seconds,
        )
        self.total_steps = 0
        self.epochs = 0

    # ----------------------------------------------------------------------------- layout
    def initialize_run(
        self,
        config: "RetrievalTrainingConfig",
        retrieval_model: "RetrievalModel",
        lora_config: "LoraConfig | None",
        counts: dict,
        batch_sizing: str,
        groups: dict[str, list],
    ) -> None:
        """Create every widget of the run and write the first page."""
        self.total_steps = counts["total_steps"]
        self.epochs = config.epochs
        reporting_examples = counts["reporting_examples"]
        self.initialize_line_plot(
            "reporting", "Reporting and validation loss",
            f"Reporting: the fixed {reporting_examples}-query batch every {config.reporting_fraction:.0%} of an epoch. "
            f"Validation: the full validation split at each epoch end in batches of {reporting_examples} queries, "
            "so both losses see the same number of in-batch candidates.",
            "epochs", "loss", ["reporting", "validation"],
        )
        self.initialize_line_plot(
            "metrics", "In-batch ranking metrics",
            "Rank-1 (top candidate is an own positive), MRR@10 and nDCG@10 (binary relevance over own positives), all "
            "in-batch: each query is ranked against the batch's distinct candidates (its own positives and hard negatives "
            "plus the other queries' candidates) with same-document collisions masked, not against the corpus. "
            "The reporting batch at every reporting point, the validation split at each epoch end.",
            "epochs", "metric",
            [f"reporting {metric}" for metric in METRIC_NAMES] + [f"validation {metric}" for metric in METRIC_NAMES],
        )
        self.initialize_line_plot(
            "step_loss", "Training loss per step", "Raw per-step loss (faint) with a 25-step moving average.",
            "step", "loss", ["loss"], smoothing_window=25,
        )
        self.initialize_line_plot(
            "learning_rate", "Learning rates", "Warmup then cosine, one group per module.",
            "step", "lr", list(groups.keys()), log_y=True,
        )
        self.initialize_bar_plot(
            "epoch_validation", "Validation per epoch", "Loss and the in-batch ranking metrics on the validation split (batches of the reporting size).",
            "epoch", "value", ["loss", *METRIC_NAMES],
        )
        self.initialize_table(
            "epochs", "Throughput per epoch",
            "Padding fraction is the share of padded positions in the embedded sequences after length grouping.",
            ["epoch", "steps", "examples/s", "candidates/s", "tokens/s (real)", "padded tokens/s", "padding fraction", "forwards/step", "step time", "peak memory", "time"],
        )
        self.set_text("batch_sizing", "Batch sizing", batch_sizing)
        lora = None
        if lora_config is not None:
            lora = {"name": lora_config.lora_name, "rank": lora_config.rank, "alpha": lora_config.alpha, "dropout": lora_config.dropout}
        ac = None
        if retrieval_model.ac_model is not None:
            ac_model = retrieval_model.ac_model
            ac = {"name": retrieval_model.ac_name, "d_ac_model": ac_model.d_ac_model,
                  "num_view_tokens": ac_model.num_view_tokens, "num_ac_layers": ac_model.num_ac_layers}
        self.set_text("run", "Run configuration", json.dumps({
            "training_config": asdict(config), **counts,
            "base_model": retrieval_model.loaded_model.model_config.model_id, "lora": lora, "ac": ac,
            "d_embedding_result": retrieval_model.d_embedding_result,
            "trainable_parameters": {name: sum(p.numel() for p in params) for name, params in groups.items()},
        }, indent=2))
        self.set_status(epoch=f"0 / {self.epochs}", step=f"0 / {self.total_steps:,}", batch=f"{counts['batch_size']} examples")
        self.render(force=True)

    # ----------------------------------------------------------------------------- events
    def report_training_step(
        self, step: int, progress: float, loss: float, learning_rates: dict[str, float],
        elapsed: float, real_tokens: int, step_time: float, peak_memory_bytes: int | None,
    ) -> None:
        self.add_data_point("step_loss", {"x": step, "loss": loss})
        self.add_data_point("learning_rate", {"x": step, **learning_rates})
        self.set_status(
            epoch=f"{progress:.2f} / {self.epochs}", step=f"{step:,} / {self.total_steps:,}",
            elapsed=format_seconds(elapsed), remaining_est=format_seconds(elapsed / step * (self.total_steps - step)),
            last_loss=f"{loss:.3f}", tokens_per_s_real=format_rate(real_tokens / step_time),
            peak_memory=_format_memory(peak_memory_bytes),
        )

    def report_reporting_point(self, step: int, progress: float, loss: float, metrics: dict[str, float]) -> None:
        self.add_data_point("reporting", {"x": progress, "reporting": loss})
        self.add_data_point("metrics", {"x": progress, **{f"reporting {name}": value for name, value in metrics.items()}})
        self.set_status(last_reporting_loss=f"{loss:.3f}", **{f"reporting_{name}": f"{value:.2f}" for name, value in metrics.items()})
        print(f"Step {step}/{self.total_steps} (epoch {progress:.2f}): reporting loss {loss:.4f}, {_format_metrics(metrics)}")

    def report_validation(self, epoch: int, loss: float, metrics: dict[str, float]) -> None:
        self.add_data_point("reporting", {"x": float(epoch), "validation": loss})
        self.add_data_point("metrics", {"x": float(epoch), **{f"validation {name}": value for name, value in metrics.items()}})
        self.add_data_point("epoch_validation", {"label": f"epoch {epoch}", "loss": loss, **metrics})
        print(f"Epoch {epoch}/{self.epochs}: validation loss {loss:.4f}, {_format_metrics(metrics)}")

    def report_epoch(
        self, label: str, steps: int, shapes: list[tuple[int, int, int, int, int]], epoch_time: float,
        peak_memory_bytes: int | None, running: bool,
    ) -> None:
        """One throughput row; a running row is replaced by the next call for the same epoch."""
        examples = sum(shape[0] for shape in shapes)
        candidates = sum(shape[1] for shape in shapes)
        real = sum(shape[2] for shape in shapes)
        padded = sum(shape[3] for shape in shapes)
        self.add_data_point("epochs", {
            "epoch": label, "steps": steps,
            "examples/s": f"{examples / epoch_time:.1f}" if epoch_time else "",
            "candidates/s": f"{candidates / epoch_time:.0f}" if epoch_time else "",
            "tokens/s (real)": format_rate(real / epoch_time) if epoch_time else "",
            "padded tokens/s": format_rate(padded / epoch_time) if epoch_time else "",
            "padding fraction": f"{1 - real / padded:.1%}" if padded else "",
            "forwards/step": f"{sum(shape[4] for shape in shapes) / steps:.1f}" if steps else "",
            "step time": f"{epoch_time / steps:.2f} s" if steps else "",
            "peak memory": _format_memory(peak_memory_bytes),
            "time": format_seconds(epoch_time), "running": running,
        })

    def report_finished(self, step: int, elapsed: float) -> None:
        self.set_status(epoch=f"{self.epochs} / {self.epochs}", step=f"{step:,} / {self.total_steps:,}",
                        elapsed=format_seconds(elapsed), remaining_est="done")
        self.finish()


def _format_memory(peak_memory_bytes: int | None) -> str:
    return "n/a" if peak_memory_bytes is None else f"{peak_memory_bytes / 1024 ** 3:.1f} GB"


def _format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{name} {value:.2f}" for name, value in metrics.items())
