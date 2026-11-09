"""
Mock figures for the unified DeepMath synthesis table (Oct 18; user: separate layout, majority with random tie-break, paper names
Single Solver / Majority Vote / Results-Only / Results + LoRA / Results + AC; Results + Summary queued, kept out of the figures).
Reads figs/unified_mocks/tqa_unified.json (tqa_unified_table.py --json).
  --style solved: % solved per difficulty, grouped bars with values; Any of 3 as a dashed ceiling over each group.
  --style gain:   points over Single Solver per difficulty, grouped bars with values.

    python make_fig_tqa_unified.py --style solved|gain --out ../figs/unified_mocks/fig_tqa_unified_<style>
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("single", "Single Solver", "#b9b9b9"), ("majority", "Majority Vote", "#7f7f7f"), ("results", "Results-Only", "#6baed6"),
        ("M1 no rows", "Results + LoRA", "#2171b5"), ("M1", "Results + AC (ours)", "#e6550d")]
COLUMNS = {"6-8": "6–8", "8.5": "8.5", ">=9": "≥ 9"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "figs/unified_mocks/tqa_unified.json"))
    parser.add_argument("--style", choices=("solved", "gain"), default="solved")
    parser.add_argument("--fig-width", type=float, default=7.0)
    parser.add_argument("--ylim", type=float, nargs=2, default=(65, 100), help="solved style: y range")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.data).read_text())
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    arms = ARMS[1:] if args.style == "gain" else ARMS
    figure, axis = plt.subplots(figsize=(args.fig_width, 2.9))
    x = range(len(data["columns"]))
    width = 0.8 / len(arms)
    for i, (key, label, color) in enumerate(arms):
        values = data["solved"][key]
        if args.style == "gain":
            values = [v - s for v, s in zip(values, data["solved"]["single"])]
        offset = (i - (len(arms) - 1) / 2) * width
        axis.bar([xi + offset for xi in x], values, width * 0.92, color=color, label=label)
        for xi, v in zip(x, values):
            axis.text(xi + offset, v + (0.25 if args.style == "solved" else 0.15), f"{v:.1f}", ha="center", va="bottom", fontsize=5.6,
                      zorder=3, bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    if args.style == "solved":
        for xi, v in zip(x, data["solved"]["any@3"]):
            axis.plot([xi - 0.42, xi + 0.42], [v, v], color="#222222", ls="--", lw=0.8, zorder=2, label="Any of 3 (ceiling)" if xi == 0 else None)
        axis.set_ylim(*args.ylim)
        axis.set_ylabel("Problems Solved (%)")
    else:
        axis.axhline(0, color="#222222", lw=0.6)
        axis.set_ylabel("Gain over Single Solver (Points)")
    axis.set_xticks(list(x))
    axis.set_xticklabels([f"{COLUMNS[c]}\n(n = {n})" for c, n in zip(data["columns"], data["n"])])
    axis.set_xlabel("Problem Difficulty")
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, fontsize=6.5, frameon=False, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0))
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(f"{args.out}.pdf")
    figure.savefig(f"{args.out}.png", dpi=200)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
