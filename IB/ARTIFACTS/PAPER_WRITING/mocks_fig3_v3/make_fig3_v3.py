"""Fig. 3 mocks v3 (user, Oct 27 ~18:45Z): add the timing (TTFT bars, accuracy style D) and the select baseline; decide single vs
double column (the paper is at the 10-page limit).
  S1  single column: (a) accuracy vs tokens + (b) recall (as now), then (c) TTFT bars underneath
  S2  single column: (a) accuracy vs tokens, (b) TTFT bars underneath; recall@k moves to the text
  D1  double column, one row: (a) accuracy vs tokens | (b) recall | (c) TTFT bars
  D2  double column, one row: (a) accuracy vs tokens | (b) TTFT bars; recall@k in the text
Data: accuracy/tokens = make_fig_retrieval_ideal.py; select → text (2B selector, at most 2 per call, ranked prompt; eval500, 3 passes)
39.13% at 1,272 tokens (k=100), 42.67% at 2,461 (k=200), S3 rq_runs/20261027_select_baseline/; timing = IB/TMP/PAPER_SYSTEMS
results.json setup B (k=200, medians): text 1,543 ms; select 553 + 530 + 141 = 1,214 ms; AC 665 + 0.8 + 144 = 807 ms; TPOT 17.7 / 16.2 / 16.2.
Run: uv run --no-project --python 3.13 --with matplotlib python make_fig3_v3.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mocks_r2")); sys.path.insert(0, str(HERE.parent / "fig3_ideal_final"))
import make_r2 as R  # noqa: E402
import make_fig_retrieval_ideal as F  # noqa: E402
R.OUT = HERE

SEL = dict(tok=[1.272, 2.461], acc=[39.13, 42.67], k=[100, 200])
BARS = [("Text", {"prefill": 1543}, 17.7, 52.7, False),
        ("Select → text", {"sprefill": 553, "sdecode": 530, "prefill": 141}, 16.2, 42.7, False),
        ("AC (ours)", {"encode": 665, "write": 0.8, "prefill": 144}, 16.2, 54.1, True)]
ST = {"encode": ("Encode (AC)", R.OURS_LIGHT, None), "write": ("AC write", R.OURS_DARK, None),
      "sprefill": ("Selector prefill", R.G_LIGHT, None), "sdecode": ("Selector decode", R.G_DARK, "////"), "prefill": ("Reader prefill", R.G_MID, None)}
ORDER = ["encode", "write", "sprefill", "sdecode", "prefill"]


def panel_acc(ax, title="(a) Accuracy vs. input"):
    """make_fig_retrieval_ideal's panel (a) plus the select baseline (grey triangles)."""
    RQ, I, dash = R.RQ, F.IDEAL, (0, (3, 1.5))
    R.clean(ax); ax.set_xscale("log"); ax.set_xlim(0.2, 40); ax.set_ylim(15, 75)
    R.ref(ax, RQ["gold"], R.LS_GOLD, "Gold Reference"); R.ref(ax, RQ["none"], R.LS_DOT, "No Context", x=0.22, ha="left")
    ax.plot(RQ["text_tok"], RQ["text_acc"], marker="s", color=R.G_MID, ms=3.4, zorder=3)
    ax.plot(RQ["ac_tok"], RQ["ac_acc"], marker="o", color=R.OURS, ms=4.0, mec="white", mew=0.6, zorder=4, path_effects=R.HALO)
    ax.plot(I["text_tok"], I["text_acc"], marker="s", color=R.G_MID, ms=3.4, ls=dash, mfc="white", zorder=3)
    ax.plot(I["ac_tok"], I["ac_acc"], marker="o", color=R.OURS, ms=4.0, ls=dash, mfc="white", mec=R.OURS, mew=1.0, zorder=4)
    ax.plot(SEL["tok"], SEL["acc"], marker="^", color=R.G_DARK, ms=3.8, lw=1.0, zorder=3)
    F.labels(ax, RQ["text_tok"], RQ["text_acc"], RQ["text_k"], "#666666", {0: (-9, 2)})
    F.labels(ax, RQ["ac_tok"], RQ["ac_acc"], RQ["ac_k"], R.OURS, {})
    F.labels(ax, I["text_tok"], I["text_acc"], I["text_k"], "#666666", {0: (-8, -3)})
    F.labels(ax, I["ac_tok"], I["ac_acc"], I["ac_k"], R.OURS, {})
    F.labels(ax, SEL["tok"], SEL["acc"], SEL["k"], R.G_DARK, {0: (-9, -6), 1: (9, -8)})
    ax.text(0.23, 44.0, "AC (ours)", color=R.OURS, fontsize=R.LAB, weight="bold")
    ax.text(3.3, 40.5, "Text", color="#666666", fontsize=R.LAB)
    ax.text(0.55, 32.0, "Select → text", color=R.G_DARK, fontsize=R.SMALL)
    ax.set_xticks([0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.3k", "1k", "3k", "10k", "30k"]); ax.minorticks_off()
    ax.set_xlabel("Mean Input Tokens (log)"); ax.set_ylabel("Accuracy (%)"); ax.set_title(title)
    h = [Line2D([], [], color="#555555", lw=1.2, label="BM25"),
         Line2D([], [], color="#555555", lw=1.2, ls=dash, marker="o", ms=3, mfc="white", label="Ideal recall")]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.13), fontsize=6.3, handlelength=2.4, borderaxespad=0.2)


def panel_recall(ax, title="(b) Recall"):
    R.clean(ax); xs = range(3)
    ax.bar(xs, R.RQ["recall_all"], color=R.G_LIGHT, width=0.62, zorder=2, edgecolor=R.G_MID, lw=0.5)
    for x, v in zip(xs, R.RQ["recall_all"]):
        ax.text(x, v + 1.5, f"{v:.0f}", ha="center")
    ax.set_xticks(list(xs)); ax.set_xticklabels([str(k) for k in R.RQ["recall_k"]])
    ax.set_ylim(0, 100); ax.set_xlabel("Top-k"); ax.set_ylabel("BM25 Recall@k (%)", labelpad=1); ax.set_title(title)


def panel_ttft(ax, title, xmax=2500, legend_ax=None, fs=6):
    R.clean(ax); ax.grid(False); ax.grid(axis="x", color="#e8e8e8", lw=0.6, zorder=0)
    for i, (lab, st, tpot, acc, ours) in enumerate(BARS):
        left = 0.0
        for key in ORDER:
            v = st.get(key, 0)
            if v > 0:
                ax.barh(i, v, left=left, height=0.62, color=ST[key][1], hatch=ST[key][2], edgecolor="white", lw=0.5, zorder=2)
                left += v
        t = ax.text(left + xmax * 0.012, i, f"{left:,.0f} ms", va="center", fontsize=fs + (0.5 if ours else 0),
                    color=R.OURS if ours else "#333333", fontweight="bold" if ours else "normal")      # AC's TTFT in bold vermilion, like its accuracy
        t = ax.annotate(f" · {tpot:.1f} ms/tok", xycoords=t, xy=(1, 0.5), va="center", ha="left", fontsize=fs, color="#333333")
        ax.annotate(f"  {acc:.1f}% acc.", xycoords=t, xy=(1, 0.5), va="center", ha="left", fontsize=fs + 0.5,
                    color=R.OURS if ours else "#777777", fontweight="bold" if ours else "normal")
    ax.set_yticks(range(len(BARS))); ax.set_yticklabels([b[0] for b in BARS]); ax.set_ylim(len(BARS) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0); ax.set_xlim(0, xmax)
    ax.set_xlabel("Time to first token (ms), k = 200, one request"); ax.set_title(title)
    handles = [Patch(facecolor=ST[k][1], hatch=ST[k][2], edgecolor="white", label=ST[k][0]) for k in ORDER if k != "write"]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.8, ncol=4, handlelength=1.3,
              columnspacing=0.8, handletextpad=0.4, frameon=False, borderaxespad=0.15)
    ax.set_title(title, pad=13)


def s1():
    fig = plt.figure(figsize=(R.COL, 3.55))
    gs = GridSpec(2, 2, width_ratios=[2.7, 1], height_ratios=[2.3, 1.05], wspace=0.42, hspace=0.62, figure=fig)
    panel_acc(fig.add_subplot(gs[0, 0])); panel_recall(fig.add_subplot(gs[0, 1]))
    panel_ttft(fig.add_subplot(gs[1, :]), "(c) Latency (2B encoder, 9B reader)", xmax=3000)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.1, top=0.94)
    R.save(fig, "fig3_S1_single_3panel")


def s2():
    fig = plt.figure(figsize=(R.COL, 3.35))
    gs = GridSpec(2, 1, height_ratios=[2.3, 1.05], hspace=0.62, figure=fig)
    panel_acc(fig.add_subplot(gs[0])); panel_ttft(fig.add_subplot(gs[1]), "(b) Latency (2B encoder, 9B reader)", xmax=3000)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.1, top=0.94)
    R.save(fig, "fig3_S2_single_2panel")


def d1():
    fig = plt.figure(figsize=(R.DCOL, 1.95))
    gs = GridSpec(1, 3, width_ratios=[2.3, 0.85, 2.6], wspace=0.45, figure=fig)
    panel_acc(fig.add_subplot(gs[0])); panel_recall(fig.add_subplot(gs[1]))
    ax = fig.add_subplot(gs[2]); panel_ttft(ax, "(c) Latency, k = 200", xmax=3000)
    ax.set_xlabel("Time to first token (ms), one request")
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.2, top=0.9)
    R.save(fig, "fig3_D1_double_3panel")


def d2():
    fig = plt.figure(figsize=(R.DCOL, 1.95))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.15], wspace=0.42, figure=fig)
    panel_acc(fig.add_subplot(gs[0])); ax = fig.add_subplot(gs[1]); panel_ttft(ax, "(b) Latency, k = 200", xmax=2900)
    ax.set_xlabel("Time to first token (ms), one request")
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.2, top=0.9)
    R.save(fig, "fig3_D2_double_2panel")


if __name__ == "__main__":
    R.setup()
    s1(); s2(); d1(); d2()
