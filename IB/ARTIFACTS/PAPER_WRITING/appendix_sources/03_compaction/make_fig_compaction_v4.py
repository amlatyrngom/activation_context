"""
Compaction figure by difficulty (user, Oct 18: "nice figures that take these quantiles into account ... per-difficulty quality and
cost to finish. Only (c) stays as is"). Three mock styles for panels (a) quality and (b) cost to finish; panel (c), compaction time,
is drawn exactly as in make_fig_compaction.py from the frozen-100 timing numbers (figs/frozen100_h0_fig.json).

Benchmark: eval-200 (9B-written prefixes, 50k continuation budget; an unanswered run counts the whole budget), seeds --read-seeds
pooled per task, from heldout_quantiles_cache.pkl (only 200-row runs without "error" rows). Difficulty: the original 9B run's
tokens after the cut (no compared run involved), as in make_fig_compaction_v3.py.

    --style quartiles   x = disjoint difficulty quartiles (50 tasks each, easiest to hardest); lines per method
    --style cumulative  x = all / hardest 50% / 25% / 10% (the table's columns); lines per method
    --style delta       x = difficulty quartiles; each method minus full context (points solved, k tokens); grouped bars
    --style paper       user's choice (Oct 18): x = Mean (all tasks), P50, P75, P90 (the hardest 50 / 25 / 10 %), axis "Continuation
                        Difficulty"; --kind line (lines per method) or --kind bar (grouped bars; the truncation slot is an empty
                        dashed bar while it is pending)

Truncation ("recent text") has no runs on this benchmark yet: it is drawn as a placeholder with a red warning until --trunc-arms
names cached arms (e.g. "T 1/5,T 1/10,T 1/20").

    uv run --no-project --with matplotlib --with numpy python make_fig_compaction_v4.py --style quartiles --out figs/fig_v4_quartiles
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = [("full", "Full context", "#222222", None, "--"), ("none", "No context", "#888888", None, ":"),
        ("summary", "Text summary (9B)", "#d62728", "D", "-")]
AC_SHADES = {"1/5": "#08519c", "1/10": "#3182bd", "1/20": "#9ecae1"}
TRUNC_COLOR = "#7f7f7f"
WARNING = "PLACEHOLDER: truncation not yet run on this benchmark. Refresh this figure when it lands."


def load(args) -> tuple[dict, list]:
    cache = pickle.load(open(ROOT / "heldout_quantiles_cache.pkl", "rb"))
    good = {k: v["tasks"] for k, v in cache.items() if len(v["tasks"]) == 200 and not v["reasons"].get("error")}
    seeds = [int(s) for s in args.read_seeds.split(",")]
    teacher = {p["task_id"]: p["teacher"]["continuation_tokens"] for p in json.load(open(args.index))["prefixes"]}
    order = sorted(good[("full", seeds[0])], key=lambda task: teacher[task])          # easiest first
    per_arm = {}
    for arm in ["full", "none", "summary"] + [f"{args.ac} {r}" for r in AC_SHADES] + (args.trunc_arms.split(",") if args.trunc_arms else []):
        have = [s for s in seeds if (arm, s) in good]
        if not have:
            raise SystemExit(f"no usable seed for {arm}")
        per_arm[arm] = {t: (np.mean([good[(arm, s)][t][0] for s in have]), np.mean([good[(arm, s)][t][2] for s in have])) for t in order}
    return per_arm, order


def groups(style: str, order: list) -> tuple[list[str], list[list]]:
    n = len(order)
    if style == "cumulative":
        return (["All\n(200)", "Hardest 50%\n(100)", "Hardest 25%\n(50)", "Hardest 10%\n(20)"],
                [order, order[n // 2:], order[3 * n // 4:], order[9 * n // 10:]])
    if style == "paper":
        return ["Mean", "P50", "P75", "P90"], [order, order[n // 2:], order[3 * n // 4:], order[9 * n // 10:]]
    q = [order[i * n // 4:(i + 1) * n // 4] for i in range(4)]
    return ["Easiest\n25%", "2nd", "3rd", "Hardest\n25%"], q


def series(args) -> list[tuple]:
    out = [(arm, label, color, marker, ls) for arm, label, color, marker, ls in BASE]
    out += [(f"{args.ac} {r}", f"AC rows {r}", AC_SHADES[r], "o", "-") for r in AC_SHADES]
    if args.trunc_arms:
        for arm, r in zip(args.trunc_arms.split(","), AC_SHADES):
            out.append((arm, f"Recent text {r}", TRUNC_COLOR, "s", "-"))
    return out


def panel_c(axis) -> None:
    timing = json.load(open(ROOT / "figs/frozen100_h0_fig.json"))
    bars = [("Text summary\n(9B)", timing["variants"]["summary_write"]["compact_seconds"], "#d62728"),
            ("AC rows", timing["ac_encode_seconds"], "#1f77b4")]
    axis.bar(range(len(bars)), [v[0] for _, v, _ in bars], color=[c for _, _, c in bars], width=0.6,
             yerr=[[v[0] - v[1] for _, v, _ in bars], [v[2] - v[0] for _, v, _ in bars]], capsize=2)
    axis.set_xticks(range(len(bars)))
    axis.set_xticklabels([name for name, _, _ in bars], fontsize=6)
    top = max(v[2] for _, v, _ in bars)
    axis.set_ylim(0, top * 1.2)
    for position, (_, v, _) in enumerate(bars):
        axis.text(position, v[2] + top * 0.02, f"{v[0]:.1f} s", ha="center", va="bottom", fontsize=6)
    axis.set_ylabel("Compaction Duration (Seconds)")
    axis.set_title("(c) Compaction time", fontsize=8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=["quartiles", "cumulative", "delta", "paper"], required=True)
    parser.add_argument("--kind", choices=["line", "bar"], default="line", help="--style paper only")
    parser.add_argument("--ratios", default="1/5,1/10,1/20", help="AC ratios to draw (user Oct 18: 1/5,1/10 to uncrowd)")
    parser.add_argument("--no-trunc", action="store_true", help="leave truncation out entirely (no placeholder, no warning)")
    parser.add_argument("--upto", default="P90", choices=["P75", "P90"], help="--style paper: last difficulty group (P75 drops the noisy 20-task tail)")
    parser.add_argument("--fig-width", type=float, default=7.4)
    parser.add_argument("--values", action="store_true", help="bar kind: print each bar's value on top")
    parser.add_argument("--ac", default="H8A", help="cached AC model name (H8A, H7S, H8F)")
    parser.add_argument("--trunc-arms", default="", help="cached truncation arms for 1/5,1/10,1/20; empty = placeholder")
    parser.add_argument("--index", default=str(ROOT / "node_sync/pc-b_live/pc/prefixes_eval9c_200/index.json"))
    parser.add_argument("--read-seeds", default="2,3,4")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    per_arm, order = load(args)
    names, bins = groups(args.style, order)
    if args.style == "paper" and args.upto == "P75":
        names, bins = names[:3], bins[:3]
    rows = [r for r in series(args) if not r[0].startswith(args.ac + " ") or r[0].split(" ")[-1] in args.ratios.split(",")]
    show_trunc = not args.no_trunc
    x = np.arange(len(names))

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 3, figsize=(args.fig_width, 2.75), gridspec_kw={"width_ratios": [1.35, 1.35, 0.8]})
    numbers = {"style": args.style, "groups": [n.replace("\n", " ") for n in names], "n": [len(b) for b in bins], "arms": {}}
    for arm, label, color, marker, ls in rows:
        solved = [100 * np.mean([per_arm[arm][t][0] for t in b]) for b in bins]
        tokens = [np.mean([per_arm[arm][t][1] for t in b]) / 1000 for b in bins]
        numbers["arms"][label] = {"solved_pct": solved, "tokens_k": tokens}

    if args.style == "delta":
        movers = [r for r in rows if r[0] not in ("full", "none")]   # no context (+11–18k tokens, −3 to −15 points) would dwarf the rest
        width = 0.8 / (len(movers) + (0 if args.trunc_arms else 1))
        full = numbers["arms"]["Full context"]
        for i, (arm, label, color, marker, ls) in enumerate(movers):
            offset = (i - (len(movers) - 1) / 2) * width
            d_solved = np.array(numbers["arms"][label]["solved_pct"]) - np.array(full["solved_pct"])
            d_tokens = np.array(numbers["arms"][label]["tokens_k"]) - np.array(full["tokens_k"])
            axes[0].bar(x + offset, d_solved, width, color=color, label=label, hatch="//" if arm == "none" else None, edgecolor="white", lw=0.3)
            axes[1].bar(x + offset, d_tokens, width, color=color, hatch="//" if arm == "none" else None, edgecolor="white", lw=0.3)
        for axis in axes[:2]:
            axis.axhline(0, color="#222222", lw=0.8, ls="--")
        axes[0].set_ylabel("Solved minus full context (pts)")
        axes[1].set_ylabel("Tokens minus full context (k)")
        axes[0].set_title("(a) Quality vs full context, by difficulty", fontsize=8)
        axes[1].set_title("(b) Cost to finish vs full context, by difficulty", fontsize=8)
    elif args.style == "paper" and args.kind == "bar":
        slots = len(rows) + (0 if args.trunc_arms or not show_trunc else 1)
        width = 0.88 / slots
        for i, (arm, label, color, marker, ls) in enumerate(rows):
            offset = (i - (slots - 1) / 2) * width
            v = numbers["arms"][label]
            style = dict(color=color, edgecolor="white", lw=0.3, hatch={"none": "//", "full": None}.get(arm))
            axes[0].bar(x + offset, v["solved_pct"], width, label=label, **style)
            axes[1].bar(x + offset, v["tokens_k"], width, **style)
            if args.values:
                for xi, a, b in zip(x, v["solved_pct"], v["tokens_k"]):
                    axes[0].text(xi + offset, a + 0.4, f"{a:.1f}", ha="center", va="bottom", fontsize=5.2)
                    axes[1].text(xi + offset, b + 0.3, f"{b:.1f}", ha="center", va="bottom", fontsize=5.2)
        low = min(min(v["solved_pct"]) for v in numbers["arms"].values())
        high = max(max(v["solved_pct"]) for v in numbers["arms"].values())
        axes[0].set_ylim(max(0, 10 * int(low / 10) - 10), 10 * int(high / 10) + 10)
        if not args.trunc_arms and show_trunc:   # empty dashed full-height slot marked TBD where truncation will go (no fake height)
            offset = (slots - 1 - (slots - 1) / 2) * width
            for axis in axes[:2]:
                bottom, top = axis.get_ylim()
                axis.bar(x + offset, [top - bottom] * len(x), width, bottom=bottom, fill=False, edgecolor="red", ls="--", lw=0.6)
                for xi in x:
                    axis.text(xi + offset, bottom + 0.5 * (top - bottom), "TBD", color="red", rotation=90, ha="center", va="center", fontsize=5.5)
        axes[0].set_ylabel("Solved (%)")
        axes[1].set_ylabel("Continuation Tokens (Thousands)")
        axes[0].set_title("(a) Quality", fontsize=8)
        axes[1].set_title("(b) Cost to finish", fontsize=8)
    else:
        for arm, label, color, marker, ls in rows:
            v = numbers["arms"][label]
            kw = dict(color=color, ls=ls, lw=1.6 if marker == "o" else 1.2, marker=marker, ms=3.5 if marker else 0, label=label)
            axes[0].plot(x, v["solved_pct"], **kw)
            axes[1].plot(x, v["tokens_k"], **kw)
        axes[0].set_ylabel("Solved (%)")
        axes[1].set_ylabel("Continuation Tokens (Thousands)")
        if args.style == "paper":
            axes[0].set_title("(a) Quality", fontsize=8)
            axes[1].set_title("(b) Cost to finish", fontsize=8)
        else:
            what = "hardest share of tasks" if args.style == "cumulative" else "difficulty quartile"
            axes[0].set_title(f"(a) Quality by {what}", fontsize=8)
            axes[1].set_title(f"(b) Cost to finish by {what}", fontsize=8)
    if not args.trunc_arms and show_trunc:   # placeholder: an empty legend entry plus the red warning
        if args.style == "paper" and args.kind == "bar":
            axes[0].bar([0], [0], 0, fill=False, edgecolor="red", ls="--", lw=0.6, label="Recent text (truncation): pending")
        else:
            axes[0].plot([], [], color=TRUNC_COLOR, ls="-", marker="s", ms=3.5, label="Recent text (truncation): pending")
        figure.text(0.5, 0.005, WARNING, color="red", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for axis in axes[:2]:
        axis.set_xticks(x)
        axis.set_xticklabels(names, fontsize=6.5)
        axis.set_xlabel("Continuation Difficulty" if args.style == "paper" else "Difficulty (original agent's tokens still to go)",
                        fontsize=7 if args.style == "paper" else 6.5)
    panel_c(axes[2])
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, fontsize=6.5, frameon=False, loc="upper center", ncol=4 if len(labels) > 6 else len(labels),
                  handlelength=1.8, columnspacing=1.0)
    figure.tight_layout(rect=(0, 0.04 if (not args.trunc_arms and show_trunc) else 0, 1, 0.84 if len(labels) > 6 else 0.9))
    figure.savefig(f"{args.out}.pdf")
    figure.savefig(f"{args.out}.png", dpi=200)
    Path(f"{args.out}.json").write_text(json.dumps(numbers, indent=1))
    for label, v in numbers["arms"].items():
        print(f"{label:20s} " + "  ".join(f"{s:5.1f}% {t:4.1f}k" for s, t in zip(v["solved_pct"], v["tokens_k"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
