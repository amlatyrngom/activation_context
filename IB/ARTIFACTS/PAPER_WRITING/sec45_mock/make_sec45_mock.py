"""Section 4.5 MVP figure mock (Oct 26). ILLUSTRATIVE CURVES ONLY: shapes follow the hypotheses discussed with the
user (KVs reconstruct fastest; starting from a reconstruction checkpoint specializes faster than from scratch),
not measured data. Identity from mocks_r2/make_r2.py.
Run: uv run --no-project --python 3.13 --with matplotlib --with numpy python make_sec45_mock.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocks_r2"))
from make_r2 import setup, clean, OURS, OURS_LIGHT, OURS_DARK, REF, LS_DOT, DCOL, SMALL  # noqa: E402

OUT = Path(__file__).parent
MODES = [("Input Embeddings", OURS_LIGHT, "o"), ("Hidden Residuals", OURS, "s"), ("All-Layer KVs", OURS_DARK, "D")]


def curve(x, start, end, rate):
    return end + (start - end) * np.exp(-x / rate)


def main():
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 1.9), gridspec_kw=dict(wspace=0.38))
    x = np.linspace(0, 4000, 41)
    xt = np.linspace(0, 2000, 21)
    # (a) reconstruction: held-out loss on the reconstruction corpus
    ax = axes[0]
    for (name, c, m), (end, rate) in zip(MODES, [(1.15, 1400), (0.80, 1100), (0.30, 450)]):
        ax.plot(x, curve(x, 3.2, end, rate), color=c, marker=m, markevery=8, label=name)
    ax.set_title("(a) Reconstruction", loc="center")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Held-out loss")
    # (b) task loss, from the reconstruction checkpoint (solid) vs from scratch (dashed)
    ax = axes[1]
    for (name, c, m), (rs, re, rr), (ss, se, sr) in zip(
            MODES, [(1.6, 0.95, 250), (1.5, 0.85, 220), (1.3, 0.70, 180)],
            [(2.4, 1.30, 900), (2.4, 1.15, 800), (2.2, 0.85, 450)]):
        ax.plot(xt, curve(xt, rs, re, rr), color=c, marker=m, markevery=5)
        ax.plot(xt, curve(xt, ss, se, sr), color=c, ls=(0, (4, 2)), lw=1.1)
    ax.set_title("(b) Task loss (HotpotQA)", loc="center")
    ax.set_xlabel("Task training steps")
    ax.set_ylabel("Validation loss")
    # (c) task accuracy, same encoding, with text references
    ax = axes[2]
    for (name, c, m), (re, rr), (se, sr) in zip(MODES, [(48, 300), (52, 260), (56, 220)],
                                                [(36, 1100), (41, 950), (52, 500)]):
        ax.plot(xt, 23 + (re - 23) * (1 - np.exp(-xt / rr)), color=c, marker=m, markevery=5)
        ax.plot(xt, 23 + (se - 23) * (1 - np.exp(-xt / sr)), color=c, ls=(0, (4, 2)), lw=1.1)
    ax.axhline(23, color=REF, lw=0.8, ls=LS_DOT)
    ax.text(2000, 24, "no context", ha="right", va="bottom", fontsize=SMALL, style="italic")
    ax.set_title("(c) Task accuracy", loc="center")
    ax.set_xlabel("Task training steps")
    ax.set_ylabel("Accuracy (%)")
    for ax in axes:
        clean(ax)
        ax.text(0.5, 0.5, "MOCK", transform=ax.transAxes, ha="center", va="center", fontsize=22, color="#000000",
                alpha=0.06, weight="bold", rotation=20)
    # one shared legend: colors = write modes, line style = starting point
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker=m, label=n) for n, c, m in MODES]
    handles += [Line2D([], [], color="#666666", label="from reconstruction ckpt"),
                Line2D([], [], color="#666666", ls=(0, (4, 2)), label="from scratch")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.10), frameon=False,
               fontsize=SMALL, handlelength=2.2, columnspacing=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"sec45_mvp_mock.{ext}", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
