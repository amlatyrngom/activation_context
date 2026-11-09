"""
Retrieval-QA figures from results.json (HotpotQA, BM25 over the HotpotQA corpus, thinking off):
  fig_rq_frontier[_eval500]  token/accuracy frontier: accuracy against the reader's input tokens (mean per question, log axis). The base 9B
                 reading the BM25 top K as text (K = 20, 100, 200, 400) vs ours (2B encoder -> 9B reader at 1/10, the same top K as K/20
                 question-first calls of 20); thin dotted links join the same K; horizontal references: gold passages as text, no passages.
                 Linear token axis (user, Oct 21 ≈18:50Z: "avoid the log scale"); the _log variants keep the log axis.
Missing numbers (null) are left out; a figure is still drawn from what exists.
  uv run --no-project --python 3.13 --with matplotlib python make_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "results.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# the palette of the other paper figures: grey for the plain baseline, orange for ours (paler: our earlier checkpoint)
TEXT, OURS, STAGE2, GOLD = ("#7f7f7f", "s"), ("#e6550d", "o"), ("#fdae6b", "o"), "black"


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {HERE / name}.pdf/.png")


def series(data: dict, key: str, tokens: str) -> list[tuple[int, float, float]]:
    """(K, tokens, accuracy) for the measured points of a series, by K."""
    return sorted((int(k), data["tokens"][tokens][k], v) for k, v in data[key].items()
                  if v is not None and data["tokens"][tokens].get(k) is not None)


def inset(ax, text, ours, stage2) -> None:
    """Linear axis: ours sits in the first ~5k tokens, so a zoom of that corner carries its labels (top-K under each point)."""
    near = [p for p in text if p[1] <= 1.25 * max(q[1] for q in ours)]
    pts = ours + stage2 + near
    x1, ys = 1.2 * max(p[1] for p in pts), [p[2] for p in pts]
    sub = ax.inset_axes([0.42, 0.16, 0.40, 0.29])
    for points, (color, marker), size, width, style in ((stage2, STAGE2, 5, 1.3, "dashed"), (near, TEXT, 6, 1.6, "solid"), (ours, OURS, 7, 2.0, "solid")):
        if points:
            sub.plot([p[1] for p in points], [p[2] for p in points], color=color, marker=marker, markersize=size, linewidth=width, linestyle=style,
                     markeredgecolor="#a63603" if color == OURS[0] else color)
    for k, x, y in ours:
        sub.annotate(f"top-{k}", (x, y), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#a63603", fontweight="bold")
    for k, x, y in near:
        sub.annotate(f"top-{k}", (x, y), xytext=(0, -6), textcoords="offset points", ha="center", va="top", fontsize=8, color="#525252")
    for k, x, y in stage2:
        sub.annotate(f"top-{k}", (x, y), xytext=(0, -6), textcoords="offset points", ha="center", va="top", fontsize=7.5, color="#e6550d")
    sub.set_xlim(0, x1)
    sub.set_ylim(min(ys) - 3, max(ys) + 3)
    sub.tick_params(labelsize=8)
    sub.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:g}k" if v else "0")
    ax.indicate_inset_zoom(sub, edgecolor="#969696", alpha=0.8)


def frontier(split: str, name: str, log: bool = False, max_k: int | None = None, zoom: bool = True, with_stage2: bool = True) -> None:
    data = DATA[split]
    text, ours, stage2 = series(data, "text", "text"), series(data, "ours", "ours"), series(data, "stage2", "ours")
    keep = lambda points: [p for p in points if max_k is None or p[0] <= max_k]
    text, ours, stage2 = keep(text), keep(ours), keep(stage2) if with_stage2 else []
    if not text and not ours:
        print(f"skip {name}: no points")
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    if log:
        ax.set_xscale("log")
    for key, style, text_label in (("gold", "dashdot", "9B reads only the gold passages as text"), ("none", "dotted", "9B with no passages")):
        if data.get(key) is not None:
            ax.axhline(data[key], color=GOLD, linestyle=style, linewidth=1.1)
            right = key == "gold" or log or not zoom                          # linear axis: the zoom inset sits over the right end of the bottom line
            ax.text(0.99 if right else 0.01, data[key] + 0.6, f"{text_label} ({data[key]:.1f})", transform=ax.get_yaxis_transform(),
                    ha="right" if right else "left", va="bottom", fontsize=9)
    by_k = {k: (x, y) for k, x, y in ours}
    for k, x, y in text:                                           # the same K: ours sits ~10x to the left
        if k in by_k:
            ax.plot([by_k[k][0], x], [by_k[k][1], y], color="#d9d9d9", linestyle="dotted", linewidth=1.2, zorder=1)
    if stage2:
        ax.plot([p[1] for p in stage2], [p[2] for p in stage2], color=STAGE2[0], marker=STAGE2[1], markersize=6, linewidth=1.5,
                linestyle="dashed", label="Ours, before KL toward gold-passage references", zorder=2)
    if text:
        ax.plot([p[1] for p in text], [p[2] for p in text], color=TEXT[0], marker=TEXT[1], markersize=7, linewidth=2,
                label="9B reads the BM25 top K as text", zorder=3)
        for k, x, y in text if log or not zoom else [p for p in text if p[1] > 1.25 * max((q[1] for q in ours), default=0)]:   # the inset labels the rest
            ax.annotate(f"top-{k}", (x, y), xytext=(0, -10), textcoords="offset points", ha="center", va="top", fontsize=9, color="#525252")
    if ours:
        ax.plot([p[1] for p in ours], [p[2] for p in ours], color=OURS[0], marker=OURS[1], markersize=8, linewidth=2.4,
                markeredgecolor="#a63603", label="Ours: 2B encoder → 9B reader, 1/10 (K/20 calls of 20)", zorder=4)
        for i, (k, x, y) in enumerate(ours if log or not zoom else []):   # linear: alternate above/below (the points sit close together)
            up = log or (len(ours) - 1 - i) % 2 == 0           # the highest-K point above, then below, above, ...
            ax.annotate(f"top-{k}", (x, y), xytext=(0, 9 if up else -11), textcoords="offset points", ha="center", va="bottom" if up else "top", fontsize=9.5,
                        color="#a63603", fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    if ours and not log and zoom:
        inset(ax, text, ours, stage2)
    xs = [p[1] for p in text + ours + stage2]
    if log:
        ticks = [t for t in (100, 300, 1000, 3000, 10000, 30000, 100000) if min(xs) / 1.6 <= t <= max(xs) * 1.6]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{t // 1000}k" if t >= 1000 else str(t) for t in ticks])
        ax.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
    else:
        step = 10000 if max(xs) > 30000 else 5000
        ticks = list(range(0, int(max(xs) * 1.05) + 1, step))
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([f"{t / 1000:g}k" if t else "0" for t in ticks])
        ax.set_xlim(0, max(xs) * 1.05)
    ys = [p[2] for p in text + ours + stage2] + [data[k] for k in ("gold", "none") if data.get(k) is not None]
    ax.set_ylim(5 * int((min(ys) - 4) // 5), 5 * int((max(ys) + 9) // 5))
    ax.set_xlabel("Reader input tokens per question (mean" + (", log scale)" if log else ")"))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"HotpotQA {data['label']}", fontsize=10.5, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: (not labels[i].startswith("Ours:"), labels[i].startswith("9B")))   # ours first, text last
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False, fontsize=9)
    save(fig, name)


def frontier_paper(split: str, name: str, max_k: int = 200, ours_min_k: int = 100, title: bool = True) -> None:
    """The paper version (user, Oct 21 ≈19:10Z; style of the accepted figures in IB/ARTIFACTS/PAPER_FIGURES, e.g. the corpus-compression curves):
    linear thousands axis, K <= max_k, AC (ours) vs Text, references Gold Reference / No Context in the legend right of the axes, no title
    except on the dev set (it says which set it is)."""
    data = DATA[split]
    keep = lambda points: [(k, x / 1000, y) for k, x, y in points if k <= max_k]
    text, ours = keep(series(data, "text", "text")), keep(series(data, "ours", "ours"))
    ours = [p for p in ours if p[0] in {q[0] for q in text}]       # only the K that Text also has (the comparison is at equal K)
    ours = [p for p in ours if p[0] >= ours_min_k]                 # user ≈20:05Z: no AC top-20 point (Text keeps its top-20)
    if not text and not ours:
        print(f"skip {name}: no points")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    by_k = {k: (x, y) for k, x, y in ours}
    for k, x, y in text:                                           # the same K: ours sits ~10x to the left
        if k in by_k:
            ax.plot([by_k[k][0], x], [by_k[k][1], y], color="#d9d9d9", linestyle="dotted", linewidth=1.1, zorder=1)
    if ours:
        ax.plot([p[1] for p in ours], [p[2] for p in ours], color=OURS[0], marker=OURS[1], linewidth=2, markersize=6, label="AC (ours)", zorder=4)
    if text:
        ax.plot([p[1] for p in text], [p[2] for p in text], color=TEXT[0], marker=TEXT[1], linewidth=2, markersize=6, label="Text", zorder=3)
    if data.get("gold") is not None:
        ax.axhline(data["gold"], color="black", linestyle="dashdot", linewidth=1.1, label="Gold Reference")
    if data.get("none") is not None:
        ax.axhline(data["none"], color="#969696", linestyle="dotted", linewidth=1.3, label="No Context")
    for i, (k, x, y) in enumerate(ours):                           # the highest-K point above, the others centred below their point
        last = i == len(ours) - 1
        ax.annotate(f"top-{k}", (x, y), xytext=(0, 7) if last else (0, -9), textcoords="offset points", ha="center",
                    va="bottom" if last else "top", fontsize=9, color="#a63603")
    for k, x, y in text:
        ax.annotate(f"top-{k}", (x, y), xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=9, color="#525252")
    xs = [p[1] for p in text + ours]
    step = 5 if max(xs) > 12 else 1
    ax.set_xticks([t for t in range(0, int(max(xs) * 1.06) + 1, step)])
    ax.set_xlim(0, max(xs) * 1.06)
    ys = [p[2] for p in text + ours] + [data[k] for k in ("gold", "none") if data.get(k) is not None]
    ax.set_ylim(5 * int((min(ys) - 4) // 5), 5 * int((max(ys) + 9) // 5))
    ax.set_xlabel("Mean Input Tokens (Thousands)")
    ax.set_ylabel("Accuracy (%)")
    if title:
        ax.set_title(f"HotpotQA {data['label']}", fontsize=10, loc="left")
    if data.get("note"):                                           # a work-in-progress flag (results.json "note"), in red
        ax.text(0.98, 0.32, data["note"], transform=ax.transAxes, ha="right", va="center", fontsize=11, color="#d62728", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5, frameon=False)
    save(fig, name)


if __name__ == "__main__":
    for split, name in (("dev200", "fig_rq_frontier"), ("eval500", "fig_rq_frontier_eval500")):
        frontier(split, name)
        frontier(split, name + "_log", log=True)
        frontier_paper(split, name + "_k200", title=split == "dev200")   # user ≈19:00Z: up to top-200, no inset, no stage-2 line; ≈19:10Z paper style
