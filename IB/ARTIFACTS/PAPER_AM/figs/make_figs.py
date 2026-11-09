"""
AM-compaction figures from ../results.json (LongHealth, AM's 4 contexts x 5 patients, Qwen3-4B-Instruct-2507, 1/10):
  fig_am_mvp     one bar per method on all 400 questions: baselines vs our trained nonuniform KV; dashed = right patient only
  fig_am_curves  accuracy on all 400 questions vs training pairs (updates x examples per update), one line per channel (resumed arms continue their line)
  fig_am_curves_tokens_log / fig_corpus_compression_curves_tokens_mvp  the same against training tokens per context (log axis /
                 linear axis with a // jump: the MVP shape, every arm to the 1x stopping point, 2,221 updates), Cartridges as a point
  fig_corpus_compression_curves_tokens_ext  the MVP shape with the resumed (more-data) points
Missing numbers (null) are left out; a figure is still drawn from what exists.
  PYTHONPATH=<dir with matplotlib> python3 make_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE.parent / "results.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# the palette of the existing paper figures: greys for plain baselines, blues for stronger ones, orange for ours
COLORS = {"no_context": "#bdbdbd", "am_uniform": "#9ecae1", "am_nonuniform": "#3182bd", "full_context": "#7f7f7f", "cartridges": "#9e9ac8",
          "trained_nonuniform": "#e6550d"}
LINES = {"trained_nonuniform": ("#e6550d", "o"), "trained_uniform": ("#fd8d3c", "s"),
         "input_residuals": ("#31a354", "^"), "input_only": ("#74c476", "D")}   # greens: purple is Cartridges' reference line


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {HERE / name}.pdf/.png")


def wrap(label: str) -> str:
    """Two-word baseline names on two lines; ours as 'AM Nonuniform / + KL Training / (ours)'."""
    if " + " in label or "(ours)" in label:
        return label.replace(" (ours)", "\n(ours)").replace(" + ", "\n+ ")
    return label.replace(" ", "\n", 1) if label.count(" ") == 1 else label


def mvp() -> None:
    """One bar per method: accuracy on all 400 questions (the 4 contexts are a setup detail for the text)."""
    bars, reference = DATA["bars"], DATA["reference"]
    names = [name for name in bars if bars[name]["all"] is not None]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for x, name in enumerate(names):
        value = 100 * bars[name]["all"]
        ax.bar(x, value, 0.72, color=COLORS.get(name, "#cccccc"))
        ax.text(x, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=10, zorder=3,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))   # readable where a label meets the reference line
    ax.axhline(100 * reference["all"], color="black", linestyle="dashed", linewidth=1.2,
               label=f"{reference['label']} ({100 * reference['all']:.1f})")
    ax.set_xticks(range(len(names)), [wrap(bars[name]["label"]) for name in names], fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(30, 95)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    save(fig, "fig_am_mvp")


ONE_X_UPDATES = 2221                 # the end of epoch 1 on v1 (1x data): every arm's original stopping point


def epoch_points(curve: dict, max_updates: int | None = None) -> list[dict]:
    """Every measured point: the first 2,221 updates and, where an arm was resumed on more data, its later points on the same line
    (no marker or label for the resumption; the text says it, user Oct 20). `max_updates`: only the points up to it."""
    return [p for p in curve["points"] if p.get("all") is not None and (max_updates is None or p["updates"] <= max_updates)]


def tokens_per_context(updates: int) -> float:
    """Question + answer tokens seen per context (millions) after this many updates (8 pairs each, spread over the 4 contexts)."""
    t = DATA["tokens"]
    return updates * DATA["examples_per_update"] * t["per_pair_question_answer"] / t["contexts"] / 1e6


def shared_updates(max_updates: int | None) -> set[int]:
    """The update counts measured on every arm (up to max_updates): the MVP figure draws only these, so no line carries a point the others
    lack (user, Oct 21: Input + Residuals' extra u2000 point between 2.0 and the endpoint)."""
    sets = [{p["updates"] for p in epoch_points(c, max_updates)} for c in DATA["curves"].values()]
    return set.intersection(*sets)


def draw_lines(ax, x_of, max_updates: int | None = None, only: set[int] | None = None) -> None:
    for name, curve in DATA["curves"].items():
        points = [p for p in epoch_points(curve, max_updates) if only is None or p["updates"] in only]
        if points:
            color, marker = LINES.get(name, ("#333333", "o"))
            ax.plot([x_of(p["updates"]) for p in points], [100 * p["all"] for p in points], color=color, marker=marker, linewidth=2,
                    markersize=5, label=curve["label"])


# legend names on the curve figures (user, Oct 20): references named by role, no parenthetical notes
CURVE_NAMES = {"full_context": "Long Context Reference", "am_nonuniform": "AM Non-Uniform", "cartridges": "Cartridges",
               "reference": "Gold Context Reference"}


def draw_references(ax, cartridges_line: bool, labels: bool = True) -> None:
    """Horizontal references in their bar colours; with a tokens axis Cartridges is a point instead of a line."""
    keys = [("full_context", "dashed"), ("am_nonuniform", "dotted")]
    if cartridges_line:
        keys.append(("cartridges", (0, (5, 2, 1, 2))))
    for key, style in keys:
        bar = DATA["bars"][key]
        if bar["all"] is not None:
            ax.axhline(100 * bar["all"], color=COLORS[key], linestyle=style, linewidth=1.3, label=CURVE_NAMES[key] if labels else None)
    ax.axhline(100 * DATA["reference"]["all"], color="black", linestyle="dashdot", linewidth=1.1,
               label=CURVE_NAMES["reference"] if labels else None)


def draw_cartridges_point(ax, x: float) -> None:
    """Cartridges as a plain point labelled by name (its x is AM Table 3's low end, 100M self-study tokens per context)."""
    value = 100 * DATA["bars"]["cartridges"]["all"]
    ax.plot([x], [value], marker="o", markersize=7, color=COLORS["cartridges"], markeredgecolor="#54278f", linestyle="none", zorder=4)
    ax.annotate("Cartridges", (x, value), xytext=(0, -10), textcoords="offset points", ha="center", va="top", fontsize=9, color="#54278f")


def curves() -> None:
    """Accuracy vs training pairs (linear), Cartridges as its Fig. 3 line."""
    per_update = DATA["examples_per_update"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    draw_lines(ax, lambda u: u * per_update / 1000)
    draw_references(ax, cartridges_line=True)
    ax.set_xlabel("Training pairs (thousands)")
    ax.set_ylabel("Accuracy, all 400 questions (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5, frameon=False)
    save(fig, "fig_am_curves")


def curves_tokens_log() -> None:
    """Accuracy vs training tokens per context on a log axis; the untrained starts sit at a '0' tick left of a break."""
    cartridges = DATA["tokens"]["cartridges_per_context"] / 1e6
    zero = 0.05                                    # where the closed-form starts (0 tokens) are drawn
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    draw_lines(ax, lambda u: tokens_per_context(u) if u else zero)
    draw_references(ax, cartridges_line=False)
    draw_cartridges_point(ax, cartridges)
    ax.set_xscale("log")
    ax.set_xlim(zero / 1.6, cartridges * 2.2)
    ticks = [zero, 0.1, 1, 10, 100]
    ax.set_xticks(ticks, ["0", "0.1", "1", "10", "100"])
    ax.minorticks_off()
    gap = zero * 1.35                              # // on the axis between 0 and the log decades
    kw = dict(transform=ax.get_xaxis_transform(), color="black", clip_on=False, linewidth=1)
    for dx in (0.93, 1.0):
        ax.plot([gap * dx * 0.96, gap * dx * 1.04], [-0.02, 0.02], **kw)
    ax.set_xlabel("Training tokens per context (millions, question + answer, log scale)")
    ax.set_ylabel("Accuracy, all 400 questions (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5, frameon=False)
    save(fig, "fig_am_curves_tokens_log")


def curves_tokens_break(max_updates: int | None, name: str, shared: bool = False) -> None:
    """Accuracy vs training tokens per context on a linear axis broken (//) between our range and Cartridges'; `max_updates`: every line
    stops there (the MVP figure stops all arms at the 1x stopping point, user Oct 20 ≈20:05Z); `shared`: only the update counts every arm has."""
    cartridges = DATA["tokens"]["cartridges_per_context"] / 1e6
    ours_max = tokens_per_context(max(p["updates"] for c in DATA["curves"].values() for p in epoch_points(c, max_updates)))
    fig, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(8.2, 4.2), gridspec_kw={"width_ratios": [5, 1], "wspace": 0.06})
    for ax in (left, right):
        draw_lines(ax, tokens_per_context, max_updates, shared_updates(max_updates) if shared else None)
        draw_references(ax, cartridges_line=False, labels=ax is left)
    draw_cartridges_point(right, cartridges)
    left.set_xlim(-0.08, ours_max * 1.08)
    right.set_xlim(cartridges - 12, cartridges + 12)
    right.set_xticks([cartridges], [f"{cartridges:.0f}"])
    right.spines["left"].set_visible(False)
    right.tick_params(axis="y", left=False)
    d = 0.012                                      # one slash at each cut end: together they read as // where the axis jumps
    for ax, x0, scale in ((left, 1.0, 1.0), (right, 0.0, 5.0)):
        ax.plot([x0 - d * scale / 2, x0 + d * scale / 2], [-0.025, 0.025], transform=ax.transAxes, color="black", clip_on=False, linewidth=1)
    left.set_xlabel("Training tokens per context (millions, question + answer)")
    left.xaxis.set_label_coords(0.6, -0.1)
    left.set_ylabel("Accuracy, all 400 questions (%)")
    handles, labels = left.get_legend_handles_labels()
    right.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=8.5, frameon=False)
    save(fig, name)


if __name__ == "__main__":
    mvp()
    curves()
    curves_tokens_log()
    curves_tokens_break(ONE_X_UPDATES, "fig_corpus_compression_curves_tokens_mvp", shared=True)   # the MVP shape (user, Oct 20): every arm to the 1x stop
    curves_tokens_break(None, "fig_corpus_compression_curves_tokens_ext")             # the same with the resumed (more-data) points
