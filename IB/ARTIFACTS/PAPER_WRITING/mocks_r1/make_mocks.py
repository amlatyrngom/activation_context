"""Style mocks for the evaluation figures (Oct 24). Every figure is drawn at its print width
(single column 3.25 in, double column 6.75 in), so font sizes here are the sizes in the PDF.
Numbers are the verified values behind the accepted figures (sources in DATA below).
Run: uv run --no-project --python 3.13 --with matplotlib python make_mocks.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
COL, DCOL = 3.25, 6.75

# ---------------------------------------------------------------- data
# Retrieval: PAPER_CORPUS_RETRIEVAL_QA/figs/results.json (eval500, mean of 3 passes);
# recall: question-only BM25 top-k over the pooled corpus, eval500, both gold paragraphs retrieved.
RQ = dict(text_tok=[2.396, 11.996, 24.351], text_acc=[43.93, 49.2, 52.67], text_k=[20, 100, 200],
          ac_tok=[1.306, 2.559], ac_acc=[52.87, 54.07], ac_k=[100, 200], gold=67.73, none=23.07,
          recall_k=[20, 100, 200], recall_all=[52.0, 65.2, 72.0], recall_mean=[74.3, 81.9, 85.7])
# Corpus compression: PAPER_AM/results.json, updates 0/500/1000/1500/2221, tokens per context = u x 1037.
CC_X = [u * 1037 / 1e6 for u in (0, 500, 1000, 1500, 2221)]
CC = {"AC KV Non-Uniform": [75.75, 79.5, 81.5, 81.25, 81.75],
      "AC KV Uniform": [55.75, 72.0, 74.5, 75.0, 75.75],
      "AC Input + Residuals": [36.25, 40.5, 47.75, 53.5, 58.0],
      "AC Input": [36.25, 44.25, 42.75, 48.0, 46.75]}
CC_REF = dict(am=75.25, long=79.5, gold=83.75, cart=69.0)
# Compaction: PAPER_COMPACTION/figs/fig_compaction_mvp.json; time from RESULTS_SO_FAR.md.
CP_GROUPS = ["Mean", "P50", "P75"]
CP = {"Full Context": ([71.67, 63.0, 62.67], [17.29, 23.83, 28.07]),
      "No Context": ([63.17, 49.33, 50.0], [32.71, 38.55, 39.51]),
      "Text Compaction": ([68.67, 57.0, 55.33], [21.17, 29.30, 34.36]),
      "AC 5×": ([71.33, 61.67, 62.0], [17.08, 24.14, 28.49]),
      "AC 10×": ([71.0, 59.67, 60.0], [18.31, 25.47, 29.07])}
CP_TIME = {"Text Compaction": (7.4, 6.9, 7.9), "AC": (1.3, 1.2, 1.5)}
# Multi-agent: paper/figures/tab_tqa_mvp.tex.
MA_GROUPS = ["6–8\n(n=272)", "8.5\n(n=226)", "9–9.5\n(n=74)"]
MA = {"Single Solver": [87.5, 75.1, 72.5], "Majority Vote": [90.1, 81.0, 76.1],
      "Results-Only": [92.4, 81.2, 82.4], "Results + Summary": [92.6, 79.6, 83.5],
      "Results + LoRA": [93.3, 81.8, 83.2], "Results + AC (ours)": [91.5, 81.8, 85.7]}
MA_CEIL = [94.1, 87.6, 90.5]

# ---------------------------------------------------------------- styles
OI = dict(verm="#D55E00", orange="#E69F00", blue="#0072B2", sky="#56B4E9", green="#009E73",
          purple="#CC79A7", yellow="#F0E442")
STYLES = {
    # A: one signature colour for ours everywhere, every other series in greys, references labelled in place.
    "A_accent": dict(
        font="STIXGeneral", base=8, panel_bg="white", grid="#e6e6e6", spines=("left", "bottom"),
        ours=["#C8102E", "#E8606F", "#8C0B20", "#F2A0A9"], ours_markers=["o", "s", "^", "D"],
        base_cols=["#4d4d4d", "#8c8c8c", "#b3b3b3", "#d0d0d0", "#6e6e6e"],
        ref_col="#222222", none_col="#9a9a9a", direct=True, band=None, callouts=False, halo=True,
        hatch_none=False, legend_top=False),
    # B: warm family = ours, cool family = baselines, black/grey references, one legend row on top.
    "B_warmcool": dict(
        font="Liberation Sans", base=7.5, panel_bg="white", grid="#ececec", spines=("left", "bottom"),
        ours=[OI["verm"], OI["orange"], "#A63A00", "#F5C26B"], ours_markers=["o", "s", "^", "D"],
        base_cols=["#08306b", "#2171b5", "#6baed6", "#9ecae1", "#4292c6"],
        ref_col="#111111", none_col="#8a8a8a", direct=False, band=None, callouts=False, halo=False,
        hatch_none=True, legend_top=True),
    # C: A's palette plus the story drawn on the figure: shaded headroom, deltas, halos, bold value for ours.
    "C_story": dict(
        font="STIXGeneral", base=8, panel_bg="white", grid="#eeeeee", spines=("left", "bottom"),
        ours=["#C8102E", "#E8606F", "#8C0B20", "#F2A0A9"], ours_markers=["o", "s", "^", "D"],
        base_cols=["#4d4d4d", "#8c8c8c", "#b3b3b3", "#d0d0d0", "#6e6e6e"],
        ref_col="#222222", none_col="#9a9a9a", direct=True, band="#f6efe6", callouts=True, halo=True,
        hatch_none=False, legend_top=False),
    # D: tinted panel (ggplot-lite) with a deep teal signature and warm-grey baselines.
    "D_tinted": dict(
        font="Liberation Sans", base=7.5, panel_bg="#f2f2ee", grid="white", spines=(),
        ours=["#00727A", "#3FA7AE", "#00474C", "#8FD0D4"], ours_markers=["o", "s", "^", "D"],
        base_cols=["#5b5148", "#8f8478", "#b9afa3", "#d8d0c6", "#746a60"],
        ref_col="#2b2b2b", none_col="#9a9a9a", direct=True, band=None, callouts=False, halo=False,
        hatch_none=False, legend_top=True),
}


def setup(s):
    plt.rcParams.update({
        "font.family": s["font"], "mathtext.fontset": "stix", "font.size": s["base"],
        "axes.labelsize": s["base"], "axes.titlesize": s["base"] + 0.5, "xtick.labelsize": s["base"] - 0.5,
        "ytick.labelsize": s["base"] - 0.5, "legend.fontsize": s["base"] - 0.5, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.4, "lines.markersize": 4, "legend.frameon": False, "savefig.dpi": 300,
        "pdf.fonttype": 42, "axes.titleweight": "bold"})


def style_ax(ax, s, ygrid=True):
    ax.set_facecolor(s["panel_bg"])
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(side in s["spines"])
    if ygrid:
        ax.grid(axis="y", color=s["grid"], lw=0.6 if s["panel_bg"] == "white" else 1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0 if not s["spines"] else 2.5)


def halo(s):
    return [pe.withStroke(linewidth=2.6, foreground="white")] if s["halo"] else None


def ref_line(ax, y, label, s, ls, col, xr=None, side="right", va="bottom", dy=0.6):
    ax.axhline(y, color=col, lw=0.8, ls=ls, zorder=1)
    if s["direct"]:
        x0, x1 = ax.get_xlim() if xr is None else xr
        x = x1 if side == "right" else x0
        ax.text(x, y + (dy if va == "bottom" else -dy), label, ha=side, va=va, fontsize=s["base"] - 1,
                color=col, style="italic")


# ---------------------------------------------------------------- retrieval
def fig_retrieval(name, s):
    setup(s)
    fig = plt.figure(figsize=(COL, 2.15))
    gs = GridSpec(1, 2, width_ratios=[2.6, 1], wspace=0.42, figure=fig)
    ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    style_ax(ax, s); style_ax(axr, s)
    ours, text = s["ours"][0], s["base_cols"][1] if not s["legend_top"] else s["base_cols"][1]
    ax.set_xlim(0, 26.5); ax.set_ylim(15, 75)
    if s["band"]:
        ax.axhspan(RQ["none"], RQ["gold"], color=s["band"], zorder=0, lw=0)
    ref_line(ax, RQ["gold"], "Gold Reference", s, (0, (5, 2, 1, 2)), s["ref_col"], va="bottom", dy=0.8)
    ref_line(ax, RQ["none"], "No Context", s, ":", s["none_col"], va="bottom", dy=0.8)
    ax.plot(RQ["text_tok"], RQ["text_acc"], marker="s", color=text, label="Text", zorder=3, ms=3.6)
    ax.plot(RQ["ac_tok"], RQ["ac_acc"], marker="o", color=ours, label="AC (ours)", zorder=4,
            path_effects=halo(s), ms=4.2, mec="white", mew=0.6)
    for x, y, k in zip(RQ["text_tok"], RQ["text_acc"], RQ["text_k"]):
        ax.annotate(f"top-{k}", (x, y), xytext=(0, -8), textcoords="offset points", ha="center",
                    fontsize=s["base"] - 1.5, color=text)
    for (x, y, k), off in zip(zip(RQ["ac_tok"], RQ["ac_acc"], RQ["ac_k"]), [(5, -8), (4, 4)]):
        ax.annotate(f"top-{k}", (x, y), xytext=off, textcoords="offset points", ha="left",
                    fontsize=s["base"] - 1.5, color=ours, weight="bold" if s["callouts"] else None)
    if s["direct"]:
        ax.text(17.0, 46.2, "Text", color=text, fontsize=s["base"] - 0.5, ha="left")
        ax.text(1.0, 60.0, "AC (ours)", color=ours, fontsize=s["base"] - 0.5, weight="bold", ha="left")
    if s["callouts"]:
        ax.annotate("", xy=(RQ["ac_tok"][1] + 0.4, 54.07), xytext=(RQ["text_tok"][2] - 0.6, 53.3),
                    arrowprops=dict(arrowstyle="->", color=ours, lw=0.8, shrinkA=0, shrinkB=0))
        ax.text(15.5, 58.0, "9.5× fewer tokens, +1.4 pts", color=ours, fontsize=s["base"] - 1.5, ha="center")
    ax.set_xlabel("Mean Input Tokens (Thousands)")
    ax.set_ylabel("Accuracy (%)")
    if not s["direct"]:
        h, l = ax.get_legend_handles_labels()
        from matplotlib.lines import Line2D
        h += [Line2D([], [], color=s["ref_col"], ls=(0, (5, 2, 1, 2)), lw=0.8),
              Line2D([], [], color=s["none_col"], ls=":", lw=0.8)]
        l += ["Gold Reference", "No Context"]
        fig.legend(h[::-1][2:][::-1] + h[2:], l[:2] + l[2:], loc="upper center", ncol=4,
                   bbox_to_anchor=(0.5, 1.02), handlelength=1.6, columnspacing=1.0)
    # recall panel
    xs = range(3)
    axr.bar(xs, RQ["recall_all"], color=s["base_cols"][2] if not s["legend_top"] else s["base_cols"][2],
            width=0.62, zorder=2)
    for x, v in zip(xs, RQ["recall_all"]):
        axr.text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=s["base"] - 1.5)
    axr.set_xticks(list(xs)); axr.set_xticklabels([str(k) for k in RQ["recall_k"]])
    axr.set_ylim(0, 100); axr.set_xlabel("Top-k"); axr.set_ylabel("Recall@k (%)", labelpad=1)
    if s["legend_top"] or s["callouts"] or True:
        ax.set_title("(a) Accuracy vs. input", loc="left", fontsize=s["base"])
        axr.set_title("(b) Recall", loc="left", fontsize=s["base"])
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.19, top=0.80 if not s["direct"] else 0.88)
    fig.savefig(OUT / f"{name}_retrieval.png"); fig.savefig(OUT / f"{name}_retrieval.pdf"); plt.close(fig)


# ---------------------------------------------------------------- corpus compression
def fig_corpus(name, s):
    setup(s)
    fig = plt.figure(figsize=(COL, 2.35))
    gs = GridSpec(1, 2, width_ratios=[5.2, 0.8], wspace=0.06, figure=fig)
    ax, axc = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharey=None)
    for a in (ax, axc):
        style_ax(a, s)
        a.set_ylim(22, 88)
    axc.spines["left"].set_visible(False); axc.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlim(-0.1, 2.45); axc.set_xlim(90, 110)
    if s["band"]:
        for a in (ax, axc):
            a.axhspan(CC_REF["long"], 88, color=s["band"], zorder=0, lw=0)
    refs = [("Gold Context Reference", CC_REF["gold"], (0, (5, 2, 1, 2)), s["ref_col"]),
            ("Long Context Reference", CC_REF["long"], "--", s["ref_col"]),
            ("AM Non-Uniform", CC_REF["am"], ":", s["base_cols"][0])]
    from matplotlib.lines import Line2D
    ref_handles = []
    for lab, y, ls, col in refs:
        for a in (ax, axc):
            a.axhline(y, color=col, lw=0.8, ls=ls, zorder=1)
        ref_handles.append((Line2D([], [], color=col, ls=ls, lw=0.8), lab))
    handles = []
    for (lab, ys), col, mk in zip(CC.items(), s["ours"], s["ours_markers"]):
        h, = ax.plot(CC_X, ys, marker=mk, color=col, zorder=3, ms=3.4, mec="white", mew=0.5,
                     path_effects=halo(s) if lab == "AC KV Non-Uniform" else None)
        handles.append((h, lab))
    if s["direct"] and not s["legend_top"]:
        short = {"AC KV Non-Uniform": "KV Non-Uniform", "AC KV Uniform": "KV Uniform",
                 "AC Input + Residuals": "Input + Residuals", "AC Input": "Input"}
        rr = {"Gold Context Reference": "Gold Context", "Long Context Reference": "Long Context",
              "AM Non-Uniform": "AM Non-Uniform"}
        hs = [h for h, _ in handles] + [h for h, _ in ref_handles]
        ls_ = [short[l] for _, l in handles] + [rr[l] for _, l in ref_handles]
        leg = ax.legend(hs, ls_, loc="lower right", ncol=2, fontsize=s["base"] - 1.8, handlelength=1.5,
                        columnspacing=0.8, labelspacing=0.2, borderaxespad=0.2)
    axc.plot([100], [CC_REF["cart"]], marker="D", color=s["base_cols"][1], ms=4, zorder=3)
    axc.annotate("Cartridges", (100, CC_REF["cart"]), xytext=(0, -9), textcoords="offset points",
                 ha="center", fontsize=s["base"] - 1.5, color=s["base_cols"][0])
    axc.set_xticks([100]); ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    if s["callouts"]:
        ax.annotate("+6.5 pts\nfrom training", xy=(2.30, 81.75), xytext=(1.05, 63.5), fontsize=s["base"] - 1.5,
                    color=s["ours"][0], ha="center",
                    arrowprops=dict(arrowstyle="->", color=s["ours"][0], lw=0.7))
    d = .012
    kw = dict(color="k", clip_on=False, lw=0.6)
    if "bottom" in s["spines"]:
        ax.plot((1 - d * 0.6, 1 + d * 0.6), (-d * 2, d * 2), transform=ax.transAxes, **kw)
        axc.plot((-d * 3.5, d * 3.5), (-d * 2, d * 2), transform=axc.transAxes, **kw)
        axc.spines["bottom"].set_visible(True)
    fig.supxlabel("Training Tokens per Context (Millions)", fontsize=s["base"], y=0.015)
    ax.set_ylabel("Accuracy (%)")
    if not s["direct"] or s["legend_top"]:
        order = ref_handles[::-1]  # AM Non-Uniform, then Long Context and Gold Context together
        hs = [h for h, _ in handles] + [h for h, _ in order]
        ls_ = [l for _, l in handles] + [l for _, l in order]
        # order requested: AM Non-Uniform, then the two references together
        fig.legend(hs, ls_, loc="upper center", ncol=2 if s["direct"] is False else 3, bbox_to_anchor=(0.5, 1.0),
                   handlelength=1.6, columnspacing=0.9, fontsize=s["base"] - 1.5, labelspacing=0.25)
        top = 0.74
    else:
        top = 0.97
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.15, top=top)
    fig.savefig(OUT / f"{name}_corpus.png"); fig.savefig(OUT / f"{name}_corpus.pdf"); plt.close(fig)


# ---------------------------------------------------------------- compaction
def fig_compaction(name, s):
    setup(s)
    fig = plt.figure(figsize=(DCOL, 1.85))
    gs = GridSpec(1, 3, width_ratios=[2.2, 2.2, 1.25], wspace=0.32, figure=fig)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    cols = {"Full Context": s["base_cols"][0], "No Context": s["base_cols"][3],
            "Text Compaction": s["base_cols"][1] if s["legend_top"] is False or s["font"] != "Liberation Sans" else s["base_cols"][1],
            "AC 5×": s["ours"][0], "AC 10×": s["ours"][1]}
    w = 0.16
    for k, (ax, ylab, ylim) in enumerate(zip(axes[:2], ["Solved (%)", "Continuation Tokens (K)"],
                                             [(40, 76), (0, 44)])):
        style_ax(ax, s)
        for i, (arm, vals) in enumerate(CP.items()):
            xs = [g + (i - 2) * w for g in range(3)]
            hatch = "////" if (arm == "No Context" and s["hatch_none"]) else None
            ax.bar(xs, vals[k], w, color=cols[arm], label=arm, zorder=2, hatch=hatch, edgecolor="white", lw=0.3)
            for x, v in zip(xs, vals[k]):
                bold = arm.startswith("AC") and s["callouts"]
                ax.text(x, v + (0.5 if k == 0 else 0.6), f"{v:.0f}" if k == 0 else f"{v:.0f}", ha="center",
                        va="bottom", fontsize=s["base"] - 2.5, weight="bold" if bold else None,
                        color=cols[arm] if arm.startswith("AC") else "#333333")
        ax.set_xticks(range(3)); ax.set_xticklabels(CP_GROUPS); ax.set_ylim(*ylim); ax.set_ylabel(ylab)
        ax.set_xlabel("Continuation Difficulty")
    axes[0].set_title("(a) Quality", loc="left"); axes[1].set_title("(b) Cost to finish", loc="left")
    ax = axes[2]; style_ax(ax, s)
    for i, (arm, (m, lo, hi)) in enumerate(CP_TIME.items()):
        col = cols["Text Compaction"] if i == 0 else s["ours"][0]
        ax.bar(i, m, 0.55, color=col, zorder=2, yerr=[[m - lo], [hi - m]], capsize=2,
               error_kw=dict(lw=0.7, capthick=0.7))
        ax.text(i, hi + 0.25, f"{m} s", ha="center", fontsize=s["base"] - 1.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Text\nCompaction", "AC"]); ax.set_ylim(0, 9.5)
    ax.set_ylabel("Seconds"); ax.set_title("(c) Compaction time", loc="left")
    if s["callouts"]:
        ax.annotate("5.7× faster", xy=(1, 1.9), xytext=(0.9, 5.2), fontsize=s["base"] - 1.5, color=s["ours"][0],
                    ha="center", arrowprops=dict(arrowstyle="->", color=s["ours"][0], lw=0.7))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=1.2)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.2, top=0.8)
    fig.savefig(OUT / f"{name}_compaction.png"); fig.savefig(OUT / f"{name}_compaction.pdf"); plt.close(fig)


# ---------------------------------------------------------------- multi-agent
def fig_multiagent(name, s):
    setup(s)
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    style_ax(ax, s)
    arms = list(MA)
    b = s["base_cols"]
    cols = [b[3], b[2], b[1] if not s["legend_top"] else b[2], b[4], b[0], s["ours"][0]]
    if s["legend_top"] and s["font"] == "Liberation Sans" and s["panel_bg"] == "white":
        cols = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#08519c", s["ours"][0]]
    hatches = [None, None, None, "....", None, None] if not s["legend_top"] else [None] * 6
    w = 0.13
    for i, arm in enumerate(arms):
        xs = [g + (i - 2.5) * w for g in range(3)]
        ax.bar(xs, MA[arm], w, color=cols[i], label=arm, zorder=2, edgecolor="white", lw=0.3, hatch=hatches[i])
        if arm.endswith("(ours)"):
            for x, v in zip(xs, MA[arm]):
                ax.text(x, v + 0.4, f"{v:.1f}", ha="center", fontsize=s["base"] - 2, color=cols[i], weight="bold")
    for g, c in enumerate(MA_CEIL):
        ax.plot([g - 0.45, g + 0.45], [c, c], color=s["ref_col"], lw=0.8, ls=(0, (3, 1.5)), zorder=3)
    if s["direct"]:
        ax.text(2.45, MA_CEIL[2] + 0.4, "Any of 3", fontsize=s["base"] - 2, style="italic", ha="right", va="bottom")
    if s["callouts"]:
        ax.annotate("+3.3 over\nResults-Only", xy=(2 + 2.5 * w, 85.7), xytext=(1.45, 93.0),
                    fontsize=s["base"] - 2, color=s["ours"][0], ha="center",
                    arrowprops=dict(arrowstyle="->", color=s["ours"][0], lw=0.6))
    ax.set_ylim(70, 96); ax.set_xticks(range(3)); ax.set_xticklabels(MA_GROUPS)
    ax.set_ylabel("Problems Solved (%)"); ax.set_xlabel("Problem Difficulty", labelpad=1)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01), handlelength=1.0,
               columnspacing=0.8, fontsize=s["base"] - 1.5, labelspacing=0.25)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.2, top=0.79)
    fig.savefig(OUT / f"{name}_multiagent.png"); fig.savefig(OUT / f"{name}_multiagent.pdf"); plt.close(fig)


def sheet(name):
    """Contact sheet: the four figures at print scale on a page-width canvas."""
    import matplotlib.image as mpimg
    imgs = {k: mpimg.imread(OUT / f"{name}_{k}.png") for k in ("retrieval", "corpus", "compaction", "multiagent")}
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.2, 6.4), dpi=150)
    fig.text(0.01, 0.985, name, fontsize=10, weight="bold", va="top")
    slots = {"retrieval": [0.01, 0.62, 0.47, 0.34], "corpus": [0.51, 0.60, 0.47, 0.37],
             "compaction": [0.01, 0.33, 0.97, 0.27], "multiagent": [0.255, 0.0, 0.47, 0.31]}
    for k, r in slots.items():
        a = fig.add_axes(r); a.imshow(imgs[k]); a.axis("off")
    fig.savefig(OUT / f"sheet_{name}.png"); plt.close(fig)


if __name__ == "__main__":
    for name, s in STYLES.items():
        fig_retrieval(name, s); fig_corpus(name, s); fig_compaction(name, s); fig_multiagent(name, s); sheet(name)
    print("wrote", sorted(p.name for p in OUT.glob("sheet_*.png")))
