"""Fig. 3 mocks with the ideal-recall (golds injected) runs (Oct 27; user: show complementarity with sophisticated retrieval
via ideal recall, "without making it too busy"; maybe log scale, maybe 2-column, or a simple update).
IDEAL values are PLACEHOLDERS until the eval500 runs land (EC2 s45-ec2e/f: AC 1x20i/5x20i/10x20i, Text top-20i/100i/200i).
Run: uv run --no-project --python 3.13 --with matplotlib python make_fig3_mocks.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocks_r2"))
import make_r2 as R  # noqa: E402

OUT = Path(__file__).parent
R.OUT = OUT    # make_r2.save writes to R.OUT
RQ = R.RQ
# PLACEHOLDERS (shape only): ideal recall = both golds injected into the top k (same passage lists for Text and AC)
IDEAL = dict(text_tok=[2.40, 12.0, 24.35], text_acc=[62.0, 60.0, 58.5], text_k=[20, 100, 200],
             ac_tok=[0.27, 1.31, 2.56], ac_acc=[61.0, 60.5, 59.5], ac_k=[20, 100, 200])


def stamp(fig):
    fig.text(0.5, 0.5, "PLACEHOLDER VALUES", ha="center", va="center", fontsize=16, color="#1f77b4", alpha=0.18,
             rotation=20, weight="bold", zorder=100)


def refs(ax, label=True):
    R.ref(ax, RQ["gold"], R.LS_GOLD, "Gold Reference" if label else None)
    R.ref(ax, RQ["none"], R.LS_DOT, "No Context" if label else None)


def series(ax, d, hollow, lab_dy=-9, ac_k=None, ac_off=None, tx_off=None):
    kw_t = dict(marker="s", color=R.G_MID, ms=3.8, zorder=3)
    kw_a = dict(marker="o", color=R.OURS, ms=4.4, zorder=4, path_effects=R.HALO)
    if hollow:
        kw_t.update(ls=(0, (3, 1.5)), mfc="white"); kw_a.update(ls=(0, (3, 1.5)), mfc="white", mec=R.OURS, mew=1.0)
    else:
        kw_a.update(mec="white", mew=0.6)
    ax.plot(d["text_tok"], d["text_acc"], **kw_t)
    ax.plot(d["ac_tok"], d["ac_acc"], **kw_a)
    for i, (x, y, k) in enumerate(zip(d["text_tok"], d["text_acc"], d["text_k"])):
        ax.annotate(f"{k}", (x, y), xytext=(tx_off or {}).get(i, (0, lab_dy)), textcoords="offset points", ha="center", color="#666666", fontsize=6.5)
    for i, (x, y, k) in enumerate(zip(d["ac_tok"], d["ac_acc"], d["ac_k"])):
        ax.annotate(f"{k}", (x, y), xytext=(ac_off or {}).get(i, (0, 5)), textcoords="offset points", ha="center", color=R.OURS, fontsize=6.5)


# ---------------------------------------------------------------- A: simple update (current single-column figure + hollow ideal series)
def mock_a():
    R.setup()
    fig = plt.figure(figsize=(R.COL, 2.35))
    gs = GridSpec(1, 2, width_ratios=[2.7, 1], wspace=0.45, figure=fig)
    ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    R.clean(ax); R.clean(axr)
    ax.set_xlim(-0.5, 26.5); ax.set_ylim(15, 75)
    refs(ax)
    series(ax, RQ, hollow=False, ac_off={0: (2, -10), 1: (6, -4)})
    series(ax, IDEAL, hollow=True, tx_off={0: (10, 4)}, ac_off={0: (-6, 5), 1: (0, 6), 2: (6, 6)})
    ax.set_xlabel("Mean Input Tokens (Thousands)"); ax.set_ylabel("Accuracy (%)"); ax.set_title("(a) Accuracy vs. input")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=R.OURS, marker="o", ms=4, mec="white", label="AC"),
         Line2D([], [], color=R.G_MID, marker="s", ms=3.5, label="Text"),
         Line2D([], [], color="#555555", ls=(0, (3, 1.5)), marker="o", mfc="white", ms=3.5, label="ideal recall")]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.2), fontsize=6.5, handlelength=2.0, borderaxespad=0.2)
    xs = range(3)
    axr.bar([x - 0.17 for x in xs], RQ["recall_all"], color=R.G_LIGHT, width=0.34, zorder=2, edgecolor=R.G_MID, lw=0.5, label="BM25")
    axr.bar([x + 0.17 for x in xs], [100] * 3, color="white", width=0.34, zorder=2, edgecolor=R.G_MID, lw=0.5, hatch="////", label="ideal")
    axr.set_xticks(list(xs)); axr.set_xticklabels(["20", "100", "200"]); axr.set_ylim(0, 112)
    axr.set_xlabel("Top-k"); axr.set_ylabel("Recall@k (%)", labelpad=1); axr.set_title("(b) Recall")
    axr.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=6, handlelength=1.0, columnspacing=0.6)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.19, top=0.89)
    stamp(fig); R.save(fig, "mock_A_simple_update")


# ---------------------------------------------------------------- B: one log-x panel, single column (recall panel dropped)
def mock_b():
    R.setup()
    fig, ax = plt.subplots(figsize=(R.COL, 2.3))
    R.clean(ax); ax.set_xscale("log"); ax.set_xlim(0.18, 40); ax.set_ylim(15, 75)
    refs(ax)
    series(ax, RQ, hollow=False, ac_off={0: (0, -10), 1: (0, -10)})
    series(ax, IDEAL, hollow=True, tx_off={0: (7, 6)}, ac_off={2: (-9, 5)})   # Text ideal 20 above-right, AC ideal 200 above-left
    ax.set_xticks([0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.3k", "1k", "3k", "10k", "30k"])
    ax.minorticks_off()
    ax.text(36, 45.5, "BM25", ha="right", color="#555555", fontsize=7, style="italic")
    ax.text(36, 61.8, "ideal recall", ha="right", color="#555555", fontsize=7, style="italic")
    ax.text(0.2, 56.0, "AC (ours)", color=R.OURS, fontsize=R.LAB, weight="bold")
    ax.text(3.2, 41.5, "Text", color="#666666", fontsize=R.LAB)
    ax.set_xlabel("Mean Input Tokens (log)"); ax.set_ylabel("Accuracy (%)")
    ax.text(0.99, 0.02, "labels: top-k", transform=ax.transAxes, ha="right", fontsize=6, color="#888888")
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.19, top=0.97)
    stamp(fig); R.save(fig, "mock_B_log_single")


# ---------------------------------------------------------------- C: two-column figure*: (a) BM25, (b) ideal recall, (c) recall
def mock_c():
    R.setup()
    fig = plt.figure(figsize=(R.DCOL, 2.1))
    gs = GridSpec(1, 3, width_ratios=[2.2, 2.2, 1], wspace=0.32, figure=fig)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    for ax, d, title in ((axs[0], RQ, "(a) BM25 retrieval"), (axs[1], IDEAL, "(b) Ideal recall (golds in the top-k)")):
        R.clean(ax); ax.set_xscale("log"); ax.set_xlim(0.18, 40); ax.set_ylim(15, 75)
        refs(ax, label=ax is axs[1])
        series(ax, d, hollow=False, tx_off={0: (9, 3)} if d is IDEAL else None)
        ax.set_xticks([0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.3k", "1k", "3k", "10k", "30k"]); ax.minorticks_off()
        ax.set_xlabel("Mean Input Tokens (log)"); ax.set_title(title)
    axs[0].set_ylabel("Accuracy (%)")
    axs[1].set_yticklabels([])
    axs[0].text(0.2, 60, "AC (ours)", color=R.OURS, fontsize=R.LAB, weight="bold")
    axs[0].text(5, 42, "Text", color="#666666", fontsize=R.LAB)
    ax = axs[2]; R.clean(ax); xs = range(3)
    ax.bar([x - 0.17 for x in xs], RQ["recall_all"], color=R.G_LIGHT, width=0.34, zorder=2, edgecolor=R.G_MID, lw=0.5, label="BM25")
    ax.bar([x + 0.17 for x in xs], [100] * 3, color="white", width=0.34, zorder=2, edgecolor=R.G_MID, lw=0.5, hatch="////", label="ideal")
    for x, v in zip(xs, RQ["recall_all"]):
        ax.text(x - 0.17, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(list(xs)); ax.set_xticklabels(["20", "100", "200"]); ax.set_ylim(0, 112)
    ax.set_xlabel("Top-k"); ax.set_ylabel("Recall@k (%)", labelpad=1); ax.set_title("(c) Recall")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6, handlelength=1.0, columnspacing=0.6)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.2, top=0.88)
    stamp(fig); R.save(fig, "mock_C_two_column")


if __name__ == "__main__":
    mock_a(); mock_b(); mock_c()
