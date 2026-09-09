"""HTML report of AC training rounds: KL and agreement per step, throughput, per-kind table, reference points, sample completions."""
from __future__ import annotations

import json
import typing as t

from ..common.reporting import HtmlReporter

if t.TYPE_CHECKING:
    from .ac_model_training import ActivationContextTrainingConfig, ActivationContextTrainingStats

REFERENCE_SERIES = ("no_context", "recent_text", "untrained_ac", "in_context")


class ActivationContextTrainingReporter(HtmlReporter):
    def __init__(self, folder: str, title: str = "AC training", description: str = "", **kwargs):
        super().__init__(folder, title, description, eyebrow="AC training", **kwargs)
        self.global_step = 0
        self.initialize_line_plot("kl", "KL(teacher || student)", "Per optimizer step, mean over the completion positions; flat lines are the reference points.",
                                  "step", "kl", ["train"] + list(REFERENCE_SERIES), log_y=True, smoothing_window=5)
        self.initialize_line_plot("agreement", "Top-1 agreement", "Teacher and student argmax agree on the next token.", "step", "fraction", ["train"], smoothing_window=5)
        self.initialize_line_plot("throughput", "Throughput", "Teacher + student tokens per second of each step.", "step", "tokens/s", ["tokens/s"])
        self.initialize_line_plot("grad", "Gradient norm", "Before clipping.", "step", "norm", ["grad_norm"], log_y=True)
        self.initialize_table("rounds", "Rounds", "One row per train() call.", ["round", "examples", "steps", "kl first", "kl last", "agreement", "tok/s", "peak GB", "duration"])
        self.initialize_table("kinds", "Per kind", "KL / agreement by item kind, latest round.", ["round", "kind", "examples", "kl", "agreement"])
        self.initialize_table("evals", "Evaluations", "eval() summaries (reporting items, references).", ["round", "name", "items", "kl", "agreement", "by kind"])

    def initialize_round(self, round_index: int, config: "ActivationContextTrainingConfig", stats: "ActivationContextTrainingStats") -> None:
        self.set_status(round=str(round_index), examples=str(stats.examples), dropped=str(stats.dropped_too_long),
                        completion_tokens=str(stats.completion_tokens), teacher_tokens=str(stats.teacher_tokens))
        self.set_text("config", "Training config", json.dumps(config.__dict__, indent=2, default=str))
        self.render()

    def report_step(self, round_index: int, step: int, steps: int, stats: "ActivationContextTrainingStats") -> None:
        self.global_step += 1
        seconds = stats.step_seconds[-1]
        self.add_data_point("kl", {"x": self.global_step, "train": stats.kl[-1]})
        self.add_data_point("agreement", {"x": self.global_step, "train": stats.agreement[-1]})
        self.add_data_point("throughput", {"x": self.global_step, "tokens/s": stats.step_tokens[-1] / max(seconds, 1e-6)})
        self.add_data_point("grad", {"x": self.global_step, "grad_norm": max(stats.grad_norm[-1], 1e-8)})
        self.set_status(step=f"{step}/{steps}", kl=f"{stats.kl[-1]:.4f}", agreement=f"{stats.agreement[-1]:.3f}")
        self.render()

    def report_reference(self, name: str, kl: float) -> None:
        """A flat reference line (drawn at the current step and refreshed as steps arrive)."""
        assert name in REFERENCE_SERIES, name
        self.add_data_point("kl", {"x": max(1, self.global_step), name: kl})
        self.render()

    def report_eval(self, round_index: int, name: str, summary: dict) -> None:
        by_kind = ", ".join(f"{kind}: {values['kl']:.4f} / {values['agreement']:.3f}" for kind, values in summary.get("by_kind", {}).items())
        self.add_data_point("evals", {"round": round_index, "name": name, "items": summary.get("items", 0), "kl": f"{summary.get('kl', 0.0):.4f}",
                                      "agreement": f"{summary.get('agreement', 0.0):.3f}", "by kind": by_kind})
        self.render()

    def report_completions(self, rows: list[dict]) -> None:
        """A few teacher-vs-student greedy completions as text: [{"item_id", "kind", "reference", "teacher", "student"}]."""
        text = "\n\n".join(f"[{row['item_id']}] ({row['kind']})\n  reference: {row['reference']}\n  teacher:   {row['teacher']}\n  student:   {row['student']}" for row in rows)
        self.set_text("completions", "Completions", text)
        self.render()

    def report_round(self, stats: "ActivationContextTrainingStats") -> None:
        seconds = sum(stats.step_seconds)
        self.add_data_point("rounds", {"round": stats.round_index, "examples": stats.examples, "steps": stats.steps,
                                       "kl first": f"{stats.kl[0]:.4f}" if stats.kl else "", "kl last": f"{stats.kl[-1]:.4f}" if stats.kl else "",
                                       "agreement": f"{stats.agreement[-1]:.3f}" if stats.agreement else "",
                                       "tok/s": f"{sum(stats.step_tokens) / seconds:.0f}" if seconds else "", "peak GB": f"{stats.peak_memory_bytes / 2**30:.1f}",
                                       "duration": f"{stats.duration_s:.0f}s"})
        for kind, values in stats.kl_by_kind.items():
            agreements = stats.agreement_by_kind.get(kind, [])
            self.add_data_point("kinds", {"round": stats.round_index, "kind": kind, "examples": len(values), "kl": f"{sum(values) / len(values):.4f}",
                                          "agreement": f"{sum(agreements) / len(agreements):.3f}" if agreements else ""})
        self.render(force=True)
