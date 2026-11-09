"""Section 4.5 figure (Oct 27), measured. Qwen3-4B-Instruct-2507 reader + LoRA; encoder = 4B copy with its own adapter.
R = FineWeb-Edu reconstruction (1,524 planned updates x 8; jobs killed at 22:00Z Oct 26, so T starts from the step-1143 pair).
S = HotpotQA task from scratch, T = task from R's step-1143 pair; 1,992 updates x 8 each. Gold documents among top-20 BM25.
(a) R held-out NLL (500 passages); (b) task held-out NLL (200 items); (c) HotpotQA dev-200 accuracy.
Data: IB/TMP/PAPER_SEC45/runs/<run>/ (S3 sec45_20261026/runs/). Identity from mocks_r2/make_r2.py.
Run: uv run --no-project --python 3.13 --with matplotlib --with numpy python make_sec45.py
"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocks_r2"))
from make_r2 import setup, clean, OURS, OURS_LIGHT, OURS_DARK, DCOL, SMALL  # noqa: E402

OUT = Path(__file__).parent
RUNS = Path(__file__).resolve().parents[3] / "TMP/PAPER_SEC45/runs"
MODES = [("in", "Input Embeddings", OURS_LIGHT, "o"), ("res", "Hidden Residuals", OURS, "s"), ("kv", "All-Layer KVs", OURS_DARK, "D")]
DASH = (0, (4, 2))
R_START = 1143


def held(run):
    rows = [json.loads(l) for l in open(RUNS / run / "sync" / f"{run}.held.jsonl")]
    return [r["update"] for r in rows], [r["held_nll"] for r in rows]


def acc(run):
    xs, ys = [], []
    for f in sorted((RUNS / run / "eval").glob("*.summary.json")):
        u = f.name.removesuffix(".summary.json")
        xs.append(1992 if u == "final" else int(u[1:])); ys.append(json.loads(f.read_text())["acc"])
    return zip(*sorted(zip(xs, ys)))


def main():
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 1.9), gridspec_kw=dict(wspace=0.38))
    table = {}
    ax = axes[0]
    for m, name, c, mk in MODES:
        x, y = held(f"R_{m}")
        ax.plot(x, y, color=c, marker=mk, ms=3)
    ax.axvline(R_START, color="#999999", lw=0.6, ls=":")
    ax.text(R_START - 40, 3.0, "T starts", ha="right", va="center", fontsize=SMALL - 1, style="italic", color="#666666")
    ax.set_yscale("log"); ax.set_title("(a) Reconstruction", loc="center")
    ax.set_xlabel("Reconstruction updates"); ax.set_ylabel("Held-out NLL")
    ax = axes[1]
    for m, name, c, mk in MODES:
        x, y = held(f"T_{m}"); ax.plot(x, y, color=c, marker=mk, ms=3, markevery=2)
        x, y = held(f"S_{m}"); ax.plot(x, y, color=c, ls=DASH, lw=1.1)
    ax.set_yscale("log"); ax.set_title("(b) Task loss (HotpotQA)", loc="center")
    ax.set_xlabel("Task updates"); ax.set_ylabel("Held-out NLL")
    ax = axes[2]
    for m, name, c, mk in MODES:
        x, y = acc(f"T_{m}"); ax.plot(x, y, color=c, marker=mk, ms=3); table[f"T_{m}"] = dict(zip(x, y))
        x, y = acc(f"S_{m}"); ax.plot(x, y, color=c, ls=DASH, lw=1.1, marker=mk, ms=2.5, mfc="white"); table[f"S_{m}"] = dict(zip(x, y))
    ax.set_ylim(0, 75); ax.set_title("(c) Task accuracy", loc="center")
    ax.set_xlabel("Task updates"); ax.set_ylabel("Accuracy (%)")
    for ax in axes:
        clean(ax)
    handles = [Line2D([], [], color=c, marker=mk, ms=3, label=n) for _, n, c, mk in MODES]
    handles += [Line2D([], [], color="#666666", label="from reconstruction"), Line2D([], [], color="#666666", ls=DASH, label="from scratch")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.10), frameon=False, fontsize=SMALL,
               handlelength=2.2, columnspacing=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_sec45.{ext}", bbox_inches="tight", pad_inches=0.02)
    for k, v in table.items():
        print(k, v)


if __name__ == "__main__":
    main()
