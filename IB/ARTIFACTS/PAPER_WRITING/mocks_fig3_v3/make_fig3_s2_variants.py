"""S2 variants (user, Oct 27 ~19:00Z): panel (a)'s legend was confusing (the select triangle unexplained; the "Ideal recall" key did not
read as the dashed lines). Panel (b) unchanged (make_fig3_v3.panel_ttft).
  L1  one legend, two blocks: methods (colour + marker) | retrieval (solid = BM25 top-k, dashed hollow = ideal recall)
  L2  direct labels at each line's end, no legend
  L3  legend row above panel (a): methods and retrieval keys in one line
Run: uv run --no-project --python 3.13 --with matplotlib python make_fig3_s2_variants.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import make_fig3_v3 as V

R, F, SEL = V.R, V.F, V.SEL
DASH = (0, (3, 1.5))


def base(ax, k_labels=True):
    RQ, I = R.RQ, F.IDEAL
    R.clean(ax); ax.set_xscale("log"); ax.set_xlim(0.2, 40); ax.set_ylim(15, 75)
    R.ref(ax, RQ["gold"], R.LS_GOLD, "Gold Reference"); R.ref(ax, RQ["none"], R.LS_DOT, "No Context", x=0.22, ha="left")
    ax.plot(RQ["text_tok"], RQ["text_acc"], marker="s", color=R.G_MID, ms=3.4, zorder=3)
    ax.plot(RQ["ac_tok"], RQ["ac_acc"], marker="o", color=R.OURS, ms=4.0, mec="white", mew=0.6, zorder=4, path_effects=R.HALO)
    ax.plot(I["text_tok"], I["text_acc"], marker="s", color=R.G_MID, ms=3.4, ls=DASH, mfc="white", zorder=3)
    ax.plot(I["ac_tok"], I["ac_acc"], marker="o", color=R.OURS, ms=4.0, ls=DASH, mfc="white", mec=R.OURS, mew=1.0, zorder=4)
    ax.plot(SEL["tok"], SEL["acc"], marker="^", color=R.G_DARK, ms=3.8, lw=1.0, zorder=3)
    if k_labels:
        F.labels(ax, RQ["text_tok"], RQ["text_acc"], RQ["text_k"], "#666666", {0: (-9, 2)})
        F.labels(ax, RQ["ac_tok"], RQ["ac_acc"], RQ["ac_k"], R.OURS, {})
        F.labels(ax, I["text_tok"], I["text_acc"], I["text_k"], "#666666", {0: (-8, -3)})
        F.labels(ax, I["ac_tok"], I["ac_acc"], I["ac_k"], R.OURS, {2: (12, -2.5)})
        F.labels(ax, SEL["tok"], SEL["acc"], SEL["k"], R.G_DARK, {0: (-12, -2.5), 1: (9, -8)})
    ax.set_xticks([0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.3k", "1k", "3k", "10k", "30k"]); ax.minorticks_off()
    ax.set_xlabel("Mean Input Tokens (log); labels = top-k"); ax.set_ylabel("Accuracy (%)"); ax.set_title("(a) Accuracy vs. input")


def method_handles():
    return [Line2D([], [], color=R.OURS, marker="o", ms=4, mec="white", lw=1.4, ls="-", label="AC (ours)"),
            Line2D([], [], color=R.G_MID, marker="s", ms=3.4, lw=1.4, label="Text"),
            Line2D([], [], color=R.G_DARK, marker="^", ms=3.8, lw=1.0, label="Select → text")]


def recall_handles():
    return [Line2D([], [], color="#555555", lw=1.2, marker="o", ms=3, label="BM25 top-k"),
            Line2D([], [], color="#555555", lw=1.2, ls=DASH, marker="o", ms=3, mfc="white", label="Ideal recall (golds in top-k)")]


def l1(ax):
    base(ax)
    leg1 = ax.legend(handles=method_handles(), loc="lower right", bbox_to_anchor=(1.0, 0.25), fontsize=6.2, handlelength=2.2,
                     borderaxespad=0.2, labelspacing=0.3)
    ax.add_artist(leg1)
    ax.legend(handles=recall_handles(), loc="lower right", bbox_to_anchor=(1.0, 0.11), fontsize=6.2, handlelength=2.2,
              borderaxespad=0.2, labelspacing=0.3)


def l2(ax):
    base(ax)
    kw = dict(fontsize=6.3, va="center")
    ax.text(0.62, 59.0, "AC, ideal recall", color=R.OURS, **kw)
    ax.text(2.95, 55.0, "AC, BM25", color=R.OURS, fontweight="bold", **kw)
    ax.text(36, 57.6, "Text, ideal recall", color="#666666", ha="right", **kw)
    ax.text(36, 43.8, "Text, BM25", color="#666666", ha="right", **kw)
    ax.text(1.17, 32.6, "Select → text, BM25", color=R.G_DARK, ha="center", **kw)


def l3(ax):
    base(ax)
    h = method_handles() + recall_handles()
    ax.set_title("(a) Accuracy vs. input", pad=26)
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=6.0, handlelength=2.0, columnspacing=0.9,
              handletextpad=0.4, frameon=False, borderaxespad=0.1)


def fig(name, panel, top=0.94, h=3.35):
    f = plt.figure(figsize=(R.COL, h))
    gs = GridSpec(2, 1, height_ratios=[2.3, 1.05], hspace=0.62, figure=f)
    panel(f.add_subplot(gs[0])); V.panel_ttft(f.add_subplot(gs[1]), "(b) Latency (2B encoder, 9B reader)", xmax=3000)
    f.subplots_adjust(left=0.17, right=0.985, bottom=0.1, top=top)
    R.save(f, name)


def l2t(ax, right=False):
    """L2 with the user's tweaks (Oct 27 ~19:15Z): AC ideal recall into the free area at the left; Select centred on its 100-200 segment,
    a little higher; AC BM25 above its 200 point; Text ideal recall right of its 200 point (right=True, wider x range) or under its own
    dashed line between k = 20 and 100 (right=False)."""
    base(ax)
    kw = dict(fontsize=6.3, va="center")
    ax.text(0.215, 58.6, "AC, ideal recall", color=R.OURS, **kw)
    ax.text(1.306 / 1.16, 51.7, "AC, BM25", color=R.OURS, fontweight="bold", ha="right", **kw)   # left of its k = 100 point (user, 19:25Z)
    ax.text(1.77, 35.2, "Select → text, BM25", color=R.G_DARK, ha="center", **kw)
    if right:
        ax.set_xlim(0.2, 120)
        ax.text(24.35 * 1.35, 63.8, "Text, ideal recall", color="#666666", ha="left", **kw)
        ax.text(26.0 * 1.35, 52.7, "Text, BM25", color="#666666", ha="left", **kw)
    else:
        ax.text(6.3, 61.4, "Text, ideal recall", color="#666666", ha="center", **kw)
        ax.text(36, 43.8, "Text, BM25", color="#666666", ha="right", **kw)


if __name__ == "__main__":
    R.setup()
    fig("fig3_S2_L1_two_legends", l1)
    fig("fig3_S2_L2_direct_labels", l2)
    fig("fig3_S2_L3_legend_row", l3, top=0.86, h=3.6)
    fig("fig3_S2_L2t_under_line", lambda ax: l2t(ax, right=False))
    fig("fig3_S2_L2t_right_of_200", lambda ax: l2t(ax, right=True))
