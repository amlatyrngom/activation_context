"""AC epoch progress, exact evaluation, training diagnostics and retained greedy samples."""
from __future__ import annotations

import json
import typing as t
from dataclasses import asdict

from ..common.reporting import HtmlReporter

if t.TYPE_CHECKING:
    from .ac_model_training import (ActivationContextTrainingConfig, ActivationContextTrainingStats,
        ActivationContextTrainingProgress, ActivationContextEvalSummary, ActivationContextEpochStats, CompletionSample, ReferenceName)

REFERENCE_SERIES = ("no_context", "recent_text", "untrained_ac", "in_context")


class ActivationContextTrainingReporter(HtmlReporter):
    def __init__(self, folder: str, title: str = "AC training", description: str = "", **kwargs: t.Any) -> None:
        super().__init__(folder, title, description, eyebrow="AC training", **kwargs)
        self.references: dict[str, float] = {}
        self.global_step = 0
        self.epoch_x = 0.0
        self.epoch_origin: int | None = None
        self.initialize_line_plot("kl", "Training KL", "Weighted item mean; cached teachers use coarsened KL.",
                                  "global optimizer step", "KL", ["train"], smoothing_window=5)
        self.initialize_line_plot("reporting", "Exact held-out KL", "Current-adapter reporting/validation, with labeled initial-adapter references.",
                                  "epochs in this report", "KL", ["reporting", "validation", *REFERENCE_SERIES])
        self.initialize_line_plot("agreement", "Top-1 agreement", "Mean teacher/student agreement over items.", "global optimizer step", "fraction", ["train"])
        self.initialize_line_plot("throughput", "Token-equivalent throughput", "Original teacher + student tokens per training second; cached teacher tokens are equivalent work, not executed tokens.",
                                  "global optimizer step", "tokens/s", ["tokens/s"])
        self.initialize_line_plot("grad", "Gradient norm", "Before clipping.", "global optimizer step", "norm", ["grad_norm"])
        self.initialize_table("epochs", "Epochs", "Phase times in seconds.",
                              ["epoch", "global step", "train s", "report s", "validation s", "samples s", "checkpoint s", "total s", "checkpoint"])
        self.initialize_table("kinds", "Per kind", "Exact reporting/validation results.", ["epoch", "set", "kind", "items", "kl", "agreement"])
        self.initialize_table("evals", "Evaluations", "Exact current-adapter results; empty and dropped sets are counted.",
                              ["epoch", "global step", "name", "items", "dropped", "kl", "agreement"])
        self.initialize_table("references", "Reference measurements", "QA recent_text is oracle gold text; compaction uses the text tail.", ["name", "KL", "measured with"])

    def initialize_training(self, config: "ActivationContextTrainingConfig", stats: "ActivationContextTrainingStats") -> None:
        if self.epoch_origin is None:
            self.epoch_origin = stats.start_epoch - 1
        self.epoch_x = float(stats.start_epoch - 1 - self.epoch_origin)
        self.set_status(epoch=f"0/{config.num_epochs}", phase="baseline", examples=str(stats.examples), dropped=str(stats.dropped_too_long),
                        completion_tokens=str(stats.completion_tokens), teacher_tokens=str(stats.teacher_tokens))
        self.set_text("config", "Training config", json.dumps(asdict(config), indent=2))
        self.render()

    def report_phase(self, progress: "ActivationContextTrainingProgress", phase: str) -> None:
        self.global_step = progress.global_step
        if self.epoch_origin is None:
            self.epoch_origin = progress.epoch_number if progress.epoch == 0 else progress.epoch_number - 1
        completed = progress.epoch_number if progress.epoch == 0 else progress.epoch_number - 1 + progress.step / max(1, progress.steps)
        self.epoch_x = completed - self.epoch_origin
        self.set_status(epoch=f"{progress.epoch}/{progress.epochs}", checkpoint_epoch=str(progress.epoch_number), step=f"{progress.step}/{progress.steps}",
                        global_step=str(progress.global_step), phase=phase, epoch_elapsed=f"{progress.epoch_elapsed_s:.1f}s", elapsed=f"{progress.elapsed_s:.1f}s")
        self.render()

    def report_step(self, progress: "ActivationContextTrainingProgress", stats: "ActivationContextTrainingStats") -> None:
        self.report_phase(progress, "training")
        self.add_data_point("kl", {"x": progress.global_step, "train": stats.kl[-1]})
        self.add_data_point("agreement", {"x": progress.global_step, "train": stats.agreement[-1]})
        self.add_data_point("throughput", {"x": progress.global_step, "tokens/s": stats.step_tokens[-1] / max(stats.step_seconds[-1], 1e-6)})
        self.add_data_point("grad", {"x": progress.global_step, "grad_norm": stats.grad_norm[-1]})
        if self.references:
            self.add_data_point("reporting", {"x": self.epoch_x, **self.references})
        self.set_status(kl=f"{stats.kl[-1]:.4f}", agreement=f"{stats.agreement[-1]:.3f}", learning_rates=", ".join(f"{lr:.3g}" for lr in stats.learning_rates[-1]))
        self.render()

    def report_reference(self, name: "ReferenceName", kl: float, *, provenance: str = "initial adapter; held set") -> None:
        assert name in REFERENCE_SERIES, name
        self.references[name] = kl
        self.add_data_point("references", {"name": name, "KL": kl, "measured with": provenance})
        self.add_data_point("reporting", {"x": 0, name: kl})
        if self.epoch_x:
            self.add_data_point("reporting", {"x": self.epoch_x, name: kl})
        self.render()

    def report_eval(self, progress: "ActivationContextTrainingProgress | None", name: str, summary: "ActivationContextEvalSummary") -> None:
        if progress is not None:
            self.report_phase(progress, name)
        epoch = progress.epoch_number - (self.epoch_origin or 0) if progress else "external"
        self.add_data_point("evals", {"epoch": epoch, "global step": self.global_step, "name": name, "items": summary["items"],
                                      "dropped": summary["dropped_too_long"], "kl": summary["kl"], "agreement": summary["agreement"]})
        if name in ("reporting", "validation") and summary["items"]:
            self.add_data_point("reporting", {"x": self.epoch_x, name: summary["kl"], **self.references})
        for kind, values in summary["by_kind"].items():
            self.add_data_point("kinds", {"epoch": epoch, "set": name, "kind": kind, **values})
        self.render()

    def report_completions(self, progress: "ActivationContextTrainingProgress", rows: list["CompletionSample"]) -> None:
        text = "\n\n".join(f"[{r['item_id']}] ({r['kind']}, global step {r['global_step']}, AC version {r['ac_version']})\n"
                           f"reference: {r['reference']}\nteacher: {r['teacher']}\nstudent: {r['student']}" for r in rows)
        label = "Baseline" if progress.epoch == 0 else f"Epoch {progress.epoch_number - (self.epoch_origin or 0)}"
        key = f"completions_{'baseline' if progress.epoch == 0 else 'epoch'}_{progress.epoch_number:03d}"
        self.set_text(key, f"{label} inspection samples", text)
        self.render()

    def report_epoch(self, epoch: "ActivationContextEpochStats", stats: "ActivationContextTrainingStats") -> None:
        self.report_phase(epoch.progress, "epoch complete")
        self.add_data_point("epochs", {"epoch": epoch.progress.epoch_number - (self.epoch_origin or 0), "global step": epoch.progress.global_step,
            "train s": round(epoch.training_seconds, 2), "report s": round(epoch.reporting_seconds, 2), "validation s": round(epoch.validation_seconds, 2),
            "samples s": round(epoch.sample_seconds, 2), "checkpoint s": round(epoch.checkpoint_seconds, 2),
            "total s": round(epoch.progress.epoch_elapsed_s, 2), "checkpoint": epoch.checkpoint_path or "disabled"})
        self.set_text(f"epoch_{epoch.progress.epoch_number:03d}", "Epoch record", json.dumps(asdict(epoch), indent=2))
        self.render(force=True)

    def report_training(self, stats: "ActivationContextTrainingStats") -> None:
        self.set_status(phase="complete", peak_memory=f"{stats.peak_memory_bytes / 2**30:.2f} GB", elapsed=f"{stats.duration_s:.1f}s")
        groups = {name: [r["seconds"] for r in stats.example_records if r["teacher_cached"] == cached]
                  for name, cached in (("live", False), ("cached", True))}
        self.set_text("rates", "Live and cached teacher examples", json.dumps({name: {"examples": len(values), "seconds": sum(values),
            "examples_per_second": len(values) / sum(values) if sum(values) else None} for name, values in groups.items()}, indent=2))
        self.set_text("training_summary", "Training summary", json.dumps(stats.summarize(), indent=2))
        self.render(force=True)
