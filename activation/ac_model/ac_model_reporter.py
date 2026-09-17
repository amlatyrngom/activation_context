"""AC training progress, evaluation curves, gold-answer log-loss, training diagnostics and retained greedy samples."""
from __future__ import annotations

import json
import typing as t
from dataclasses import asdict

from ..common.reporting import HtmlReporter

if t.TYPE_CHECKING:
    from .ac_model_training import (ActivationContextTrainingConfig, ActivationContextTrainingStats,
        ActivationContextTrainingProgress, ActivationContextEvalSummary, ActivationContextEpochStats, CompletionSample, ReferenceName)

REFERENCE_SERIES = ("no_context", "recent_text", "untrained_ac", "in_context")
TEACHER_SERIES = "teacher (stored at training start)"


class ActivationContextTrainingReporter(HtmlReporter):
    """
    The page a training call writes. Widgets, in reading order: gold-answer log-loss of every evaluation set and the
    stored teacher, evaluation loss (KL to the stored teacher, or cross-entropy), training loss per update, then the
    tables (evaluations, epochs, per kind, references) and the diagnostics (agreement, gradient norm, throughput).
    `set_labels` renames evaluation sets in plots and tables, e.g. {"reporting": "held-out, AC student"}.
    """

    def __init__(self, folder: str, title: str = "AC training", description: str = "", set_labels: dict[str, str] | None = None,
                 **kwargs: t.Any) -> None:
        super().__init__(folder, title, description, eyebrow="AC training", **kwargs)
        self.loss_kind: str | None = None
        self.references: dict[str, float] = {}
        self.set_labels = dict(set_labels or {})
        self.global_step = 0
        self.epoch_x = 0.0
        self.epoch_origin: int | None = None
        self.planned_updates: int | None = None
        self._interventions_reported = 0
        self._encoder_reported = 0
        self.initialize_line_plot("gold_nll", "Gold-answer log-loss",
                                  "Negative log-likelihood per target token of the reference completion, item mean. "
                                  "Every reader is scored on the same tokens; lower is better and the teacher is not a ceiling.",
                                  "epochs in this report", "nats per target token", [])
        self.initialize_line_plot("reporting", "Evaluation loss", "Named evaluation sets and initial-reader references.",
                                  "epochs in this report", "loss", [*REFERENCE_SERIES])
        self.initialize_line_plot("loss", "Training loss per update", "Weighted item mean of the update's items.", "global update", "loss", ["train"], smoothing_window=5)
        self.initialize_table("evals", "Evaluations", "One row per evaluation of a named set.",
                              ["epoch", "global update", "set", "items", "loss kind", "loss", "agreement", "gold NLL", "token acc", "teacher gold NLL"])
        self.initialize_table("epochs", "Epochs", "Phase times in seconds; the checkpoint column names what was actually written.",
                              ["epoch", "global update", "train s", "panels s", "validation s", "samples s", "checkpoint s", "total s", "checkpoint"])
        self.initialize_table("kinds", "Per kind", "Evaluation results by item kind.", ["epoch", "set", "kind", "items", "loss kind", "loss", "agreement", "gold NLL", "token acc"])
        self.initialize_table("references", "Reference measurements", "QA recent_text is oracle gold text; compaction uses the text tail.", ["name", "loss kind", "loss", "measured with"])
        self.initialize_line_plot("throughput", "Token throughput", "Tokens per training second.", "global update", "tokens/s", ["tokens/s"])
        self.initialize_line_plot("grad", "Gradient norm", "Before clipping.", "global update", "norm", ["grad_norm"])

    # ------------------------------------------------------------------------------------------ helpers
    def label(self, name: str) -> str:
        return self.set_labels.get(name, name)

    def _ensure_series(self, widget: str, series: str) -> None:
        if series not in {entry["name"] for entry in self.widgets[widget]["series"]}:
            self.widgets[widget]["series"].append({"name": series, "x": [], "y": []})

    def _clear_status(self, *keys: str) -> None:
        for key in keys:
            self.status.pop(key.replace("_", " "), None)

    def _set_loss_kind(self, loss_kind: str) -> None:
        if self.loss_kind == loss_kind:
            return
        if loss_kind not in ("kl", "sft") or self.loss_kind is not None:
            raise ValueError("Use a separate AC reporter for each loss kind")
        self.loss_kind = loss_kind
        if loss_kind == "kl":
            self.widgets["reporting"].update(title="KL to the stored teacher", y_label="KL",
                description="Top-k coarsened KL, teacher || student, per evaluation set; the teacher is the reader as it was at training start. "
                            "Zero would mean the student matches it exactly.")
            self.widgets["loss"].update(title="Training KL per update", y_label="KL",
                                        description="Coarsened KL to the stored teacher on the update's items; smoothed over 5 updates.")
            self.initialize_line_plot("agreement", "Top-1 agreement", "Mean teacher/student agreement over items (token agreement, not answer accuracy).",
                                      "global update", "fraction", ["train"])
        else:
            self.widgets["reporting"].update(title="Evaluation cross-entropy", y_label="cross-entropy",
                                             description="Mean cross-entropy over the target tokens per evaluation set.")
            self.widgets["loss"].update(title="Training cross-entropy per update", y_label="cross-entropy",
                                        description="Weighted mean cross-entropy over retained assistant tokens; smoothed over 5 updates.")
            self.remove_widget("agreement")
            for name in ("kinds", "evals"):
                self.widgets[name]["columns"] = [column for column in self.widgets[name]["columns"] if column not in ("agreement", "teacher gold NLL")]

    # ------------------------------------------------------------------------------------------ trainer callbacks
    def initialize_training(self, config: "ActivationContextTrainingConfig", stats: "ActivationContextTrainingStats") -> None:
        self._set_loss_kind(config.loss_kind)
        if self.epoch_origin is None:
            self.epoch_origin = stats.start_epoch - 1
        self.epoch_x = float(stats.start_epoch - 1 - self.epoch_origin)
        self.set_status(phase="baseline", epoch=f"0 of {config.num_epochs}", items=str(stats.examples))
        self.set_text("config", "Training config", json.dumps(asdict(config), indent=2))
        self.render()

    def report_teacher_targets(self, targets: dict) -> None:
        coverage = targets.get("coverage")
        self.set_status(teacher_targets=f"{targets.get('examples', 0)} items · top-{targets.get('top_k')}"
                                        + (f" · {coverage:.1%} of the mass" if coverage is not None else ""))
        self.render(force=True)

    def report_phase(self, progress: "ActivationContextTrainingProgress", phase: str) -> None:
        self.global_step = progress.global_step
        if self.epoch_origin is None:
            self.epoch_origin = progress.epoch_number if progress.epoch == 0 else progress.epoch_number - 1
        completed = progress.epoch_number if progress.epoch == 0 else progress.epoch_number - 1 + progress.step / max(1, progress.steps)
        self.epoch_x = completed - self.epoch_origin
        if self.planned_updates is None and progress.epoch:
            self.planned_updates = progress.global_step - ((progress.epoch - 1) * progress.steps + progress.step) + progress.epochs * progress.steps
        done = (progress.epoch - 1) * progress.steps + progress.step if progress.epoch else 0
        total = progress.epochs * progress.steps
        remaining = (progress.elapsed_s / done * (total - done)) if done and total > done else None
        self.set_status(phase=phase, epoch=f"{progress.epoch} of {progress.epochs}", update=f"{progress.step} of {progress.steps} in this epoch",
                        global_updates=f"{progress.global_step}" + (f" of {self.planned_updates}" if self.planned_updates else ""),
                        elapsed=_duration(progress.elapsed_s),
                        remaining=("about " + _duration(remaining)) if remaining is not None else ("done" if done == total and total else "estimating"))
        if phase == "training":
            self._clear_status("evaluation_items")
        self.render()

    def report_step(self, progress: "ActivationContextTrainingProgress", stats: "ActivationContextTrainingStats") -> None:
        self._set_loss_kind(stats.loss_kind)
        self.report_phase(progress, "training")
        self.add_data_point("loss", {"x": progress.global_step, "train": stats.loss[-1]})
        if stats.loss_kind == "kl":
            self.add_data_point("agreement", {"x": progress.global_step, "train": stats.agreement[-1]})
        self.add_data_point("throughput", {"x": progress.global_step, "tokens/s": stats.step_tokens[-1] / max(stats.step_seconds[-1], 1e-6)})
        self.add_data_point("grad", {"x": progress.global_step, "grad_norm": stats.grad_norm[-1]})
        if stats.penalty:
            if "penalty" not in self.widgets:
                self.initialize_line_plot("penalty", "No-context penalty",
                    "KL of the adapted reader to the base on the same items without their rows (unweighted, sampled examples); "
                    "a rising curve means the reader improves without the rows.", "global update", "KL", ["train"])
            self.add_data_point("penalty", {"x": progress.global_step, "train": stats.penalty[-1]})
            self.set_status(no_context_penalty=f"{stats.penalty[-1]:.4f}")
        if stats.distill_lambda:
            if "distill_loss" not in self.widgets:
                self.initialize_line_plot("distill_loss", "Dense supervision: distillation term per layer (training, rows on)",
                    "Per supervised reader layer, the mean over the update's examples of the normalised context-distillation term on the training forwards: "
                    "mean over the target positions of ||A_student - A_teacher||^2 / the teacher's mean squared norm there, A = the attention block's output (after o_proj). "
                    "The teacher is the base reader with no adapters on the full-context history; the student the row-fed reader as trained. 1 = the student writes nothing "
                    "like the teacher, 0 = it writes the teacher's tensor. Read with 'Dense supervision: held-out panel' (rows on vs rows off): the rows-independent part "
                    "of the fit falls first in every arm.", "global update", "normalised term", [], smoothing_window=5)
                self.initialize_line_plot("distill_lambda", "Dense supervision: λ per update",
                    "The weight of the distillation term on the schedule clock: held for distill_hold of the planned updates, a linear decay to the floor at distill_decay_end, "
                    "then the floor.", "global update", "λ", ["λ"])
            for layer, value in stats.distill_loss[-1].items():
                self._ensure_series("distill_loss", f"layer {layer}")
                self.add_data_point("distill_loss", {"x": progress.global_step, f"layer {layer}": value})
            self.add_data_point("distill_lambda", {"x": progress.global_step, "λ": stats.distill_lambda[-1]})
            self.set_status(distill_lambda=f"{stats.distill_lambda[-1]:.3g}")
        if self.references:
            self.add_data_point("reporting", {"x": self.epoch_x, **self.references})
        if stats.encoder_trace and len(stats.encoder_trace) != self._encoder_reported:
            self._encoder_reported = len(stats.encoder_trace)
            self._report_encoder_entry(stats.encoder_trace[-1], stats.encoder_trace[-1]["update"], "global update", "")
        if stats.intervention_trace and len(stats.intervention_trace) != self._interventions_reported:
            entry = stats.intervention_trace[-1]
            self._interventions_reported = len(stats.intervention_trace)
            if "interventions" not in self.widgets:
                self.initialize_line_plot("interventions", "Delta scales relative to calibration",
                    "Deep inputs, scaled heads: each intervention layer's learned relative scale (the applied delta RMS in units of that layer's residual RMS at calibration; "
                    "K/V targets: one series per half, 'layer l:k' / 'layer l:v', each in units of its own K or V RMS); the initial value is the scale fraction. "
                    "The scales train in their own Adam group, so one update moves every layer by about the same share of its residual; a plateau well before "
                    "the decay is a learned magnitude, a straight walk at the group's rate is not. Lora heads: inert, flat by design (a guard: if it moves, the scales were not frozen); "
                    "read the applied magnitude from 'Applied delta RMS' instead.",
                    "global update", "scale / residual RMS", [])
                self.initialize_line_plot("intervention_grads", "Delta head gradient norms",
                    "Per intervention layer, before clipping. Scaled heads: proportional to the scale by construction (unit rows x scale), so this tracks the scale walk, not learning. "
                    "Lora heads: starts small (the delta is zero at init) and must climb; a monotone decay toward zero means the path is dying.",
                    "global update", "norm", [])
                self.initialize_line_plot("intervention_delta_rms", "Applied delta RMS relative to the residual at the rows",
                    "Per intervention layer, the RMS of the delta rows the reader received (mean over the encoded rows since the last point) divided by the residual RMS "
                    "measured at the row positions on the same training forwards (K/V targets: per half against the K or V RMS at the rows; kv_slots: the side-computed "
                    "slot content against the reader's own K or V; cross_attention: the block's delta against the residual at the target positions it wrote): the real "
                    "relative magnitude whatever the head style. In bf16 the addition rounds away below ~0.4 %; watch it cross ~1 % and settle in the 2-10 % band.", "global update", "delta RMS / row RMS", [])
                self.initialize_line_plot("intervention_row_rms", "Residual RMS at the row positions",
                    "Per intervention layer, measured by the reader hooks on the training forwards (reader as trained, rows present): the deltas' true denominator, "
                    "against the calibration RMS from no-row sequences.", "global update", "RMS", [])
                self.initialize_line_plot("intervention_cosine", "Delta vs input row",
                    "Per intervention layer, mean cosine between the delta rows and target_head(z) over the encoded rows since the last point: "
                    "near 1 means the delta re-injects the input row; near 0, a different direction. Residual targets only.", "global update", "cosine", [])
            for layer, value in entry["relative"].items():
                self._ensure_series("interventions", f"layer {layer}")
                self.add_data_point("interventions", {"x": entry["update"], f"layer {layer}": value})
            for layer, value in entry["head_grad_norm"].items():
                self._ensure_series("intervention_grads", f"layer {layer}")
                self.add_data_point("intervention_grads", {"x": entry["update"], f"layer {layer}": value})
            if entry.get("attention_out_norm") and "intervention_attention" not in self.widgets:      # A2 and cross-attention arms only, created once
                self.initialize_line_plot("intervention_attention", "Attention blocks: out-projection weight norm",
                    "A2 sources and the cross-attention ceiling. Per intervention layer, the norm of the zero-initialized output projection of the attention block "
                    "(A2: the row over the passage in the encoder; cross_attention: every reader position over the whole passage): "
                    "0 at init (nothing is read through the block), rising as the attention path opens.", "global update", "norm", [])
            if entry.get("slot_rms") and "intervention_slot_rms" not in self.widgets:                # kv_slots / kv_transfer arms only, created once
                self.initialize_line_plot("intervention_slot_rms", "K/V replacement RMS relative to the reader's own K/V at the rows",
                    "kv_slots and kv_transfer. Per intervention layer and half ('layer l:k' / 'layer l:v'), the RMS of the K or V written over the row positions "
                    "(keep x the reader's own + the content: a side-computed slot, or takeover x the side model's own K/V) divided by the RMS of the reader's own K or V "
                    "at the same rows on the same training forwards: exactly 1 at init for kv_slots (the slot is the reader's own value), about the side / reader K/V "
                    "RMS ratio for kv_transfer. Read with 'Applied delta RMS' (the content alone) and 'K/V takeover'.", "global update", "written RMS / own RMS", [])
                self.initialize_line_plot("intervention_takeover", "K/V takeover",
                    "kv_slots and kv_transfer. Per intervention layer and half, the learned takeover of the reader's own K or V at the rows: 0 = the reader's own value is "
                    "kept whole (kv_slots' init), 1 = the content alone: a side-computed slot, or the side model's own K/V (kv_transfer's default init; a per-request "
                    "K/V write in the engine). Negative or above 1 is allowed (a linear mix).",
                    "global update", "takeover", [])
            for widget, key in (("intervention_row_rms", "row_rms"), ("intervention_cosine", "cosine"), ("intervention_attention", "attention_out_norm"), ("intervention_delta_rms", "delta_rms"),
                                ("intervention_slot_rms", "slot_rms"), ("intervention_takeover", "takeover")):
                for layer, value in (entry.get(key) or {}).items():
                    if value is None or widget not in self.widgets:
                        continue
                    self._ensure_series(widget, f"layer {layer}")
                    self.add_data_point(widget, {"x": entry["update"], f"layer {layer}": value})
        self.set_status(loss=f"{stats.loss[-1]:.4f}", learning_rates=", ".join(f"{lr:.3g}" for lr in stats.learning_rates[-1]))
        if stats.loss_kind == "kl":
            self.set_status(agreement=f"{stats.agreement[-1]:.3f}")
        self.render()

    def report_reference(self, name: "ReferenceName", loss: float, *, provenance: str = "initial adapter; held set", loss_kind: str = "kl") -> None:
        assert name in REFERENCE_SERIES, name
        self._set_loss_kind(loss_kind)
        self.references[name] = loss
        self.add_data_point("references", {"name": name, "loss kind": loss_kind, "loss": loss, "measured with": provenance})
        self.add_data_point("reporting", {"x": 0, name: loss})
        if self.epoch_x:
            self.add_data_point("reporting", {"x": self.epoch_x, name: loss})
        self.render()

    def report_eval(self, progress: "ActivationContextTrainingProgress | None", name: str, summary: "ActivationContextEvalSummary") -> None:
        self._set_loss_kind(summary["loss_kind"])
        if progress is not None:
            self.report_phase(progress, name)
        epoch = (round(progress.epoch_number - (self.epoch_origin or 0) - (1 - progress.step / max(1, progress.steps)), 3)
                 if progress and progress.epoch else (0 if progress else "external"))
        label = self.label(name)
        gold, teacher_gold = summary.get("gold_nll"), summary.get("teacher_gold_nll")
        self.add_data_point("evals", {"epoch": epoch, "global update": self.global_step, "set": label, "items": summary["items"],
                                      "loss kind": summary["loss_kind"], "loss": _round(summary["loss"]),
                                      **({"agreement": _round(summary["agreement"]), "teacher gold NLL": _round(teacher_gold)} if self.loss_kind == "kl" else {}),
                                      "gold NLL": _round(gold), "token acc": _round(summary.get("token_accuracy"))})
        if summary["items"]:
            self._ensure_series("reporting", label)
            self.add_data_point("reporting", {"x": self.epoch_x, **self.references, label: summary["loss"]})
            accuracy = summary.get("token_accuracy")
            if accuracy is not None:
                if "token_accuracy" not in self.widgets:
                    self.initialize_line_plot("token_accuracy", "Teacher-forced token accuracy",
                        "Share of target positions where the reader's most likely token is the gold token, per evaluation set.",
                        "epochs in this report", "fraction", [])
                self._ensure_series("token_accuracy", label)
                self.add_data_point("token_accuracy", {"x": self.epoch_x, label: accuracy})
            for key, title, description in (("rare_token_loss", "Rare-token loss (held-out, top-20 % base surprisal)",
                                             "Per evaluation set, the item mean of the student's mean -log p over the target tokens whose stored base no-context surprisal is in "
                                             "the item's top 20 % (the identifiers, numbers and other unpredictable-in-context pieces the rows lose first). Surprisal arms only "
                                             "(rare_weight / pool_surprisal); read next to the unweighted evaluation loss."),
                                            ("rare_token_accuracy", "Rare-token accuracy (held-out, top-20 % base surprisal)",
                                             "Teacher-forced accuracy on the same rare tokens, per evaluation set.")):
                value = summary.get(key)
                if value is not None:
                    if key not in self.widgets:
                        self.initialize_line_plot(key, title, description, "epochs in this report", "nats per token" if key == "rare_token_loss" else "fraction", [])
                    self._ensure_series(key, label)
                    self.add_data_point(key, {"x": self.epoch_x, label: value})
            if gold is not None:
                self._ensure_series("gold_nll", label)
                point = {"x": self.epoch_x, label: gold}
                if teacher_gold is not None and name == "reporting":
                    self._ensure_series("gold_nll", TEACHER_SERIES)
                    point[TEACHER_SERIES] = teacher_gold
                self.add_data_point("gold_nll", point)
            if name == "reporting":
                self.set_status(last_panel=f"update {self.global_step}")
        for kind, values in summary["by_kind"].items():
            self.add_data_point("kinds", {"epoch": epoch, "set": label, "kind": kind, "items": values["items"],
                "loss kind": values["loss_kind"], "loss": _round(values["loss"]), "gold NLL": _round(values.get("gold_nll")), "token acc": _round(values.get("token_accuracy")),
                **({"agreement": _round(values["agreement"])} if self.loss_kind == "kl" else {})})
        self.render()

    def report_distill_panel(self, progress: "ActivationContextTrainingProgress", panel: dict) -> None:
        """Dense supervision, at every panel on a few held-out items: the distillation term per layer with the rows on and with the rows off (the reader on the
        stripped history), and the rows' attention share per layer."""
        if "distill_panel" not in self.widgets:
            self.initialize_line_plot("distill_panel", "Dense supervision: held-out panel, rows on vs rows off",
                "Per supervised reader layer on a few held-out items, the normalised distillation term of the reader as trained with the rows present ('rows on') and "
                "of the same reader on the stripped history without any rows ('rows off'): the rows-off curve is the rows-independent part of the fit (the reader "
                "LoRA matching the teacher's conditional mean) and falls first; the gap between the two curves is what the rows deliver at that layer.",
                "epochs in this report", "normalised term", [])
            self.initialize_line_plot("rows_attention_share", "Rows' attention share at the target positions",
                "Per supervised reader layer on the same held-out items, the share of the target positions' attention mass that lands on the row positions "
                "(manual attention over the reader's q / k with its RoPE, GQA grouping and scaling; mean over heads, positions and items). Rising = the reader "
                "has started reading the rows at that layer.", "epochs in this report", "share", [])
        if progress is not None:
            self.report_phase(progress, "dense supervision panel")
        for name, label in (("rows_on", "rows on"), ("rows_off", "rows off")):
            for layer, value in panel.get(name, {}).items():
                self._ensure_series("distill_panel", f"layer {layer} ({label})")
                self.add_data_point("distill_panel", {"x": self.epoch_x, f"layer {layer} ({label})": value})
        for layer, value in panel.get("rows_attention_share", {}).items():
            self._ensure_series("rows_attention_share", f"layer {layer}")
            self.add_data_point("rows_attention_share", {"x": self.epoch_x, f"layer {layer}": value})
        self.render()

    def report_encoder_panel(self, progress: "ActivationContextTrainingProgress", entry: dict) -> None:
        """Round 8 write-side diagnostics on the held-out panel's forwards: the refine adapter's relative RMS and the pass-1 -> last-pass row cosine
        (encoder_passes > 1), the surprisal pooling's bin statistics (pool_surprisal > 0); the training-forward twins of these curves are on the update axis."""
        if progress is not None:
            self.report_phase(progress, "encoder panel")
        self._report_encoder_entry(entry, self.epoch_x, "epochs in this report", " (panel)")
        self.render()

    def _report_encoder_entry(self, entry: dict, x: float, x_label: str, suffix: str) -> None:
        widgets = {
            "refine_rms": ("Iterative encoder: refine adapter RMS relative to the pass-1 summaries", "RMS of refine_adapter(unit rows of the previous pass) divided by the RMS of the "
                           "pass-1 summary inputs it is added to, mean over the encoded parts: 0 at init (an exact re-read); a curve that stays at 0 is a path that never opened.",
                           "ratio", ["refine_rms"]),
            "pass_cosine": ("Iterative encoder: pass-1 vs last-pass row cosine", "Mean cosine between the unit rows of pass 1 and of the last pass (the side model's final states at the row "
                            "positions): 1 at init; how far the later passes move the rows.", "cosine", ["pass_cosine"]),
            "pool_bins": ("Surprisal pooling: bin statistics", "pool_surprisal: the mean base no-context surprisal (nats) of the tokens in a summary bin - averaged over the rows, and the first and "
                          "last rows' - plus the widest bin and the uniform width in tokens (right axis in spirit; same plot).", "nats / tokens",
                          ["pool_row_mean", "pool_first", "pool_last", "pool_width_max", "pool_width_uniform"]),
        }
        for name, (title, description, y_label, keys) in widgets.items():
            values = {key: entry[key] for key in keys if entry.get(key) is not None}
            if not values:
                continue
            widget = name + ("_panel" if suffix else "")
            if widget not in self.widgets:
                self.initialize_line_plot(widget, title + suffix, description, x_label, y_label, [])
            for key, value in values.items():
                self._ensure_series(widget, key)
                self.add_data_point(widget, {"x": x, key: value})

    def report_evaluation_progress(self, name: str, completed: int, total: int) -> None:
        self.set_status(phase=self.label(name), evaluation_items=f"{completed} of {total}")
        self.render(force=completed == 0 or completed == total)

    def report_completions(self, progress: "ActivationContextTrainingProgress", rows: list["CompletionSample"]) -> None:
        text = "\n\n".join(f"[{r['item_id']}] ({r['kind']}, global update {r['global_step']}, AC version {r['ac_version']})\n"
                           f"reference: {r['reference']}\n" + (f"teacher: {r['teacher']}\n" if r["teacher"] is not None else "") + f"student: {r['student']}" for r in rows)
        label = "Baseline" if progress.epoch == 0 else f"Epoch {progress.epoch_number - (self.epoch_origin or 0)}"
        key = f"completions_{'baseline' if progress.epoch == 0 else 'epoch'}_{progress.epoch_number:03d}"
        self.set_text(key, f"{label} inspection samples", text)
        self.render()

    def report_epoch(self, epoch: "ActivationContextEpochStats", stats: "ActivationContextTrainingStats") -> None:
        self.report_phase(epoch.progress, "epoch complete")
        self.add_data_point("epochs", {"epoch": epoch.progress.epoch_number - (self.epoch_origin or 0), "global update": epoch.progress.global_step,
            "train s": round(epoch.training_seconds, 2), "panels s": round(epoch.reporting_seconds, 2), "validation s": round(epoch.validation_seconds, 2),
            "samples s": round(epoch.sample_seconds, 2), "checkpoint s": round(epoch.checkpoint_seconds, 2),
            "total s": round(epoch.progress.epoch_elapsed_s, 2), "checkpoint": epoch.checkpoint_path or "not written (disabled)"})
        self.set_status(last_checkpoint=(epoch.checkpoint_path or "none").rsplit("/", 1)[-1])
        self.set_text(f"epoch_{epoch.progress.epoch_number:03d}", "Epoch record",
                      json.dumps({key: value for key, value in asdict(epoch).items() if key not in ("reporting", "validation", "deltas_off", "passes_off", "samples")}, indent=2))
        self.render(force=True)

    def report_training(self, stats: "ActivationContextTrainingStats") -> None:
        self.set_status(phase="complete", peak_memory=f"{stats.peak_memory_bytes / 2**30:.2f} GB", elapsed=_duration(stats.duration_s), remaining="done")
        self._clear_status("evaluation_items")
        self.set_text("training_summary", "Training summary", json.dumps(stats.summarize(), indent=2, default=str))
        self.render(force=True)


def _round(value: float | None, digits: int = 4) -> float | str:
    return round(value, digits) if isinstance(value, (int, float)) else ""


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, minutes = divmod(seconds // 60, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds % 60:02d}s"
