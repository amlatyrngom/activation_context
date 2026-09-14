"""
The agent-training report: per-step curves (loss, ratio and clip fraction, mean log-prob, gradient
norm, throughput) across rounds, a per-round table, and a status strip. `HtmlReporter`
(common/reporting.py) owns the page; the trainer calls initialize_round / report_step / report_round.
"""
from __future__ import annotations

import typing as t

from activation.common.reporting import HtmlReporter, format_seconds

if t.TYPE_CHECKING:
    from .agent_training_config import AgentTrainingConfig, AgentTrainingStats

ROUND_COLUMNS = ["round", "items", "examples", "dropped", "tokens", "loss tokens", "steps", "first loss", "last loss",
                 "step-1 ratio", "step-1 clip", "reporting loss", "+weights", "-weights", "duration", "checkpoint"]


class AgentTrainingReporter(HtmlReporter):
    def __init__(self, report_folder: str, title: str, description: str = ""):
        super().__init__(report_folder, title, description, eyebrow="Agent training", refresh_seconds=10, min_render_interval_seconds=2.0)
        self.global_step = 0
        self.initialize_line_plot("loss", "Loss per step", "The clipped surrogate (negated), mean over the mini-batch's samples.", "step", "loss", ["loss"])
        self.initialize_line_plot(
            "ratio", "Ratio to the sampling policy",
            "Mean pi_theta / pi_old over loss tokens (1.0 at the first step of a round when trainer and engine agree) and the share of tokens outside 1 +- epsilon.",
            "step", "value", ["mean ratio", "clip fraction"],
        )
        self.initialize_line_plot("logprob", "Mean log-prob of the sampled tokens", "An entropy proxy: rising means the policy sharpens on its own samples.", "step", "log-prob", ["mean logprob"])
        self.initialize_line_plot("grad_norm", "Gradient norm", "Before clipping.", "step", "norm", ["grad norm"])
        self.initialize_line_plot("throughput", "Tokens per second", "Tokens read per second of step time (prompts included).", "step", "tokens/s", ["tokens/s"])
        self.initialize_table("rounds", "Rounds", "One row per train() call.", ROUND_COLUMNS)
        self.render(force=True)

    @property
    def report_folder(self) -> str:
        return str(self.folder)

    def initialize_round(self, round_index: int, config: "AgentTrainingConfig", items: int, examples: int, dropped: int, tokens: int) -> None:
        self.set_status(round=str(round_index), items=str(items), examples=str(examples), dropped=str(dropped), tokens=str(tokens),
                        steps_this_round=str(config.updates_per_round), learning_rate=f"{config.learning_rate:g}", clip_epsilon=f"{config.clip_epsilon:g}")
        self.render(force=True)

    def report_step(self, round_index: int, step: int, loss: float, mean_ratio: float, clip_fraction: float, mean_logprob: float,
                    grad_norm: float, tokens: int, seconds: float) -> None:
        self.global_step += 1
        x = self.global_step
        self.add_data_point("loss", {"x": x, "loss": loss})
        self.add_data_point("ratio", {"x": x, "mean ratio": mean_ratio, "clip fraction": clip_fraction})
        self.add_data_point("logprob", {"x": x, "mean logprob": mean_logprob})
        self.add_data_point("grad_norm", {"x": x, "grad norm": grad_norm})
        self.add_data_point("throughput", {"x": x, "tokens/s": tokens / seconds if seconds > 0 else 0.0})
        self.set_status(step=f"{step} (round {round_index})", last_loss=f"{loss:.4f}", last_ratio=f"{mean_ratio:.3f}")
        self.render(force=True)

    def report_round(self, stats: "AgentTrainingStats") -> None:
        self.add_data_point("rounds", {
            "round": stats.round_index, "items": stats.items, "examples": stats.examples,
            "dropped": stats.dropped_too_long + stats.dropped_empty, "tokens": stats.total_tokens, "loss tokens": stats.assistant_tokens,
            "steps": stats.steps, "first loss": f"{stats.loss[0]:.4f}" if stats.loss else "-", "last loss": f"{stats.loss[-1]:.4f}" if stats.loss else "-",
            "step-1 ratio": f"{stats.mean_ratio[0]:.3f}" if stats.mean_ratio else "-",
            "step-1 clip": f"{stats.clip_fraction[0]:.3f}" if stats.clip_fraction else "-",
            "reporting loss": "-" if stats.reporting_loss is None else f"{stats.reporting_loss:.4f}",
            "+weights": f"{stats.positive_weight_sum:.2f}", "-weights": f"{stats.negative_weight_sum:.2f}",
            "duration": format_seconds(stats.duration_s), "checkpoint": stats.checkpoint_path or "-",
        })
        self.set_status(round_duration=format_seconds(stats.duration_s))
        self.render(force=True)
