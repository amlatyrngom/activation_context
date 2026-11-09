"""Paper Fig. 3 with ideal recall (Oct 27; user: "B-style in fig 3a"). Single column, two panels:
(a) accuracy vs mean reader input tokens, log x: BM25 top-k (solid; fig_retrieval values) and ideal recall (hollow dashed: both golds
    injected into the top k, rq_common.inject_golds); Text = base Qwen3.5-9B thinking off, AC = Arm G at 1/10; eval500, mean of 3 passes.
(b) BM25 recall@k (both golds), unchanged from fig_retrieval.
Ideal values: S3 rq_runs/20261027_ideal_recall/ (text 66.33/64.07/63.80 at 2,402/11,996/24,350 tokens; AC 67.13/63.67/62.87 at the
frontier's 331/1,306/2,559 tokens).
Run: uv run --no-project --python 3.13 --with matplotlib python make_fig_retrieval_ideal.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocks_r2"))
import make_r2 as R  # noqa: E402

OUT = Path(__file__).parent
R.OUT = OUT
RQ = R.RQ
IDEAL = dict(text_tok=[2.402, 11.996, 24.350], text_acc=[66.33, 64.07, 63.80], text_k=[20, 100, 200],
             ac_tok=[0.331, 1.306, 2.559], ac_acc=[67.13, 63.67, 62.87], ac_k=[20, 100, 200])
FS = 6.0


def labels(ax, xs, ys, ks, color, offsets):
    for i, (x, y, k) in enumerate(zip(xs, ys, ks)):
        ax.annotate(f"{k}", (x, y), xytext=offsets.get(i, (0, -8.5)), textcoords="offset points", ha="center", color=color, fontsize=FS)


def main():
    R.setup()
    fig = plt.figure(figsize=(R.COL, 2.3))
    gs = GridSpec(1, 2, width_ratios=[2.7, 1], wspace=0.42, figure=fig)
    ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    R.clean(ax); R.clean(axr)
    ax.set_xscale("log"); ax.set_xlim(0.2, 40); ax.set_ylim(15, 75)
    R.ref(ax, RQ["gold"], R.LS_GOLD, "Gold Reference"); R.ref(ax, RQ["none"], R.LS_DOT, "No Context", x=0.22, ha="left")
    dash = (0, (3, 1.5))
    # BM25 (solid)
    ax.plot(RQ["text_tok"], RQ["text_acc"], marker="s", color=R.G_MID, ms=3.4, zorder=3)
    ax.plot(RQ["ac_tok"], RQ["ac_acc"], marker="o", color=R.OURS, ms=4.0, mec="white", mew=0.6, zorder=4, path_effects=R.HALO)
    # ideal recall (hollow, dashed)
    ax.plot(IDEAL["text_tok"], IDEAL["text_acc"], marker="s", color=R.G_MID, ms=3.4, ls=dash, mfc="white", zorder=3)
    ax.plot(IDEAL["ac_tok"], IDEAL["ac_acc"], marker="o", color=R.OURS, ms=4.0, ls=dash, mfc="white", mec=R.OURS, mew=1.0, zorder=4)
    labels(ax, RQ["text_tok"], RQ["text_acc"], RQ["text_k"], "#666666", {})
    labels(ax, RQ["ac_tok"], RQ["ac_acc"], RQ["ac_k"], R.OURS, {})
    labels(ax, IDEAL["text_tok"], IDEAL["text_acc"], IDEAL["text_k"], "#666666", {0: (-8, -3)})
    labels(ax, IDEAL["ac_tok"], IDEAL["ac_acc"], IDEAL["ac_k"], R.OURS, {})
    ax.text(0.23, 44.0, "AC (ours)", color=R.OURS, fontsize=R.LAB, weight="bold")
    ax.text(3.3, 40.5, "Text", color="#666666", fontsize=R.LAB)
    ax.set_xticks([0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.3k", "1k", "3k", "10k", "30k"]); ax.minorticks_off()
    ax.set_xlabel("Mean Input Tokens (log)"); ax.set_ylabel("Accuracy (%)"); ax.set_title("(a) Accuracy vs. input")
    h = [Line2D([], [], color="#555555", lw=1.2, label="BM25"),
         Line2D([], [], color="#555555", lw=1.2, ls=dash, marker="o", ms=3, mfc="white", label="Ideal recall")]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.13), fontsize=6.3, handlelength=2.4, borderaxespad=0.2)
    xs = range(3)
    axr.bar(xs, RQ["recall_all"], color=R.G_LIGHT, width=0.62, zorder=2, edgecolor=R.G_MID, lw=0.5)
    for x, v in zip(xs, RQ["recall_all"]):
        axr.text(x, v + 1.5, f"{v:.0f}", ha="center")
    axr.set_xticks(list(xs)); axr.set_xticklabels([str(k) for k in RQ["recall_k"]])
    axr.set_ylim(0, 100); axr.set_xlabel("Top-k"); axr.set_ylabel("BM25 Recall@k (%)", labelpad=1)
    axr.set_title("(b) Recall")
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.19, top=0.89)
    R.save(fig, "fig_retrieval_ideal")


if __name__ == "__main__":
    main()
