"""Round-2 style mocks (Oct 24): the direction agreed with the reviewer agent.

Identity: one signature red-vermilion for ours (light/dark variants with distinct markers, KV channels solid,
input channels dashed), greys for every baseline (kept out of the L* 40-62 band so ours stays unique in
grayscale), near-black reference lines, STIX (Times) type, drawn at print width so sizes are final:
labels 8 pt, ticks/legend/annotations 7 pt, nothing under 6.5 pt.
Two variants: "plain" (no callouts) and "callout" (one callout in retrieval, one value callout in compaction).
Numbers: the verified values listed in DATA (sources in comments).
Run: uv run --no-project --python 3.13 --with matplotlib python make_r2.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path

OUT = Path(__file__).parent
COL, DCOL = 3.25, 6.75

# ---------------------------------------------------------------- data
# Retrieval: PAPER_CORPUS_RETRIEVAL_QA/figs/results.json (eval500, mean of 3 passes).
# Recall: question-only BM25 over the pooled corpus, eval500, both supporting paragraphs in the top k.
RQ = dict(text_tok=[2.396, 11.996, 24.351], text_acc=[43.93, 49.2, 52.67], text_k=[20, 100, 200],
          ac_tok=[1.306, 2.559], ac_acc=[52.87, 54.07], ac_k=[100, 200], gold=67.73, none=23.07,
          recall_k=[20, 100, 200], recall_all=[52.0, 65.2, 72.0])
# Corpus compression: PAPER_AM/results.json, updates 0/500/1000/1500/2221; tokens per context = u x 1037.
CC_X = [u * 1037 / 1e6 for u in (0, 500, 1000, 1500, 2221)]
CC = {"KV Non-Uniform": [75.75, 79.5, 81.5, 81.25, 81.75],
      "KV Uniform": [55.75, 72.0, 74.5, 75.0, 75.75],
      "Input + Residuals": [36.25, 40.5, 47.75, 53.5, 58.0],
      "Input": [36.25, 44.25, 42.75, 48.0, 46.75]}
CC_REF = dict(am=75.25, long=79.5, gold=83.75, cart=69.0)
# Compaction: PAPER_COMPACTION/figs/fig_compaction_mvp.json; time: RESULTS_SO_FAR.md (mean, 95% interval).
CP_GROUPS = ["Mean", "P50", "P75"]
CP = {"Full Context": ([71.67, 63.0, 62.67], [17.29, 23.83, 28.07]),
      "No Context": ([63.17, 49.33, 50.0], [32.71, 38.55, 39.51]),
      "Text Compaction": ([68.67, 57.0, 55.33], [21.17, 29.30, 34.36]),
      "AC 5×": ([71.33, 61.67, 62.0], [17.08, 24.14, 28.49]),
      "AC 10×": ([71.0, 59.67, 60.0], [18.31, 25.47, 29.07])}
CP_TIME = [("Text", 7.4, 6.9, 7.9), ("AC", 1.3, 1.2, 1.5)]
# Multi-agent: paper/figures/tab_tqa_mvp.tex. The top bucket holds 72 problems at 9 and 2 at 9.5.
MA_GROUPS = ["6–8 (n=272)", "8.5 (n=226)", "9–9.5 (n=74)"]
MA = {"Single Solver": [87.5, 75.1, 72.5], "Majority Vote": [90.1, 81.0, 76.1],
      "Results-Only": [92.4, 81.2, 82.4], "Results + Summary": [92.6, 79.6, 83.5],
      "Results + LoRA": [93.3, 81.8, 83.2], "Results + AC (ours)": [91.5, 81.8, 85.7]}
MA_CEIL = [94.1, 87.6, 90.5]

# ---------------------------------------------------------------- identity
OURS, OURS_LIGHT, OURS_DARK, OURS_MID = "#D7301F", "#F4A582", "#8C1A0E", "#EF6548"
G_DARK, G_MID, G_LIGHT = "#333333", "#969696", "#CCCCCC"
REF = "#1a1a1a"
LAB, SMALL = 8, 7
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]
TXT_HALO = [pe.withStroke(linewidth=1.8, foreground="white")]
LS_GOLD, LS_LONG, LS_DOT = (0, (6, 2, 1.2, 2)), (0, (4, 2)), (0, (1, 1.6))


def setup():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": SMALL,
        "axes.labelsize": LAB, "axes.titlesize": LAB, "axes.titleweight": "bold", "xtick.labelsize": SMALL,
        "ytick.labelsize": SMALL, "legend.fontsize": SMALL, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.4,
        "lines.markersize": 4, "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
        "hatch.linewidth": 0.5, "hatch.color": "#666666"})


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def ref(ax, y, ls, label=None, x=None, ha="right", va="bottom", dy=0.7):
    ax.axhline(y, color=REF, lw=0.8, ls=ls, zorder=1)
    if label:
        x0, x1 = ax.get_xlim()
        ax.text(x1 if x is None else x, y + (dy if va == "bottom" else -dy), label, ha=ha, va=va,
                fontsize=SMALL, style="italic", color=REF, path_effects=TXT_HALO)


def save(fig, name):
    # tight crop: LaTeX (mlsys2025.sty) already adds ~20 pt above each caption
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- retrieval (single column, 2 panels)
def retrieval(variant):
    setup()
    fig = plt.figure(figsize=(COL, 2.2))
    gs = GridSpec(1, 2, width_ratios=[2.7, 1], wspace=0.45, figure=fig)
    ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    clean(ax); clean(axr)
    ax.set_xlim(0, 26.5); ax.set_ylim(15, 75)
    ref(ax, RQ["gold"], LS_GOLD, "Gold Reference")
    ref(ax, RQ["none"], LS_DOT, "No Context")
    ax.plot(RQ["text_tok"], RQ["text_acc"], marker="s", color=G_MID, zorder=3, ms=3.8)
    ax.plot(RQ["ac_tok"], RQ["ac_acc"], marker="o", color=OURS, zorder=4, ms=4.4, mec="white", mew=0.6,
            path_effects=HALO)
    for x, y, k in zip(RQ["text_tok"], RQ["text_acc"], RQ["text_k"]):
        ax.annotate(f"top-{k}", (x, y), xytext=(0, -9), textcoords="offset points", ha="center", color="#666666")
    ax.annotate("top-100", (RQ["ac_tok"][0], RQ["ac_acc"][0]), xytext=(3, -9), textcoords="offset points",
                ha="left", color=OURS)
    ax.annotate("top-200", (RQ["ac_tok"][1], RQ["ac_acc"][1]), xytext=(4, 2), textcoords="offset points",
                ha="left", color=OURS)
    ax.text(0.8, 62.0, "AC (ours)", color=OURS, fontsize=LAB, weight="bold")
    ax.text(16.5, 46.0, "Text", color="#666666", fontsize=LAB)
    if variant == "callout":
        ax.annotate("", xy=(RQ["ac_tok"][1] + 0.5, 54.0), xytext=(RQ["text_tok"][2] - 0.7, 53.0),
                    arrowprops=dict(arrowstyle="->", color=OURS, lw=0.8, shrinkA=0, shrinkB=0))
        ax.text(15.5, 56.0, "9.5× fewer tokens\n+1.4 pts", color=OURS, ha="center", va="bottom",
                linespacing=1.0)
    elif variant == "quiet":
        # thin, light, dashed link between the two top-200 points; small grey note on it
        ax.annotate("", xy=(RQ["ac_tok"][1] + 0.6, 54.0), xytext=(RQ["text_tok"][2] - 0.8, 52.9),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", color=OURS_LIGHT, lw=0.6,
                                    ls=(0, (3, 2)), shrinkA=0, shrinkB=0))
        ax.text(15.2, 54.3, "9.5× fewer tokens", color="#777777", ha="center", va="bottom", fontsize=6.5,
                style="italic")
    elif variant == "links":
        # the earlier accepted convention: faint dotted links between same-k points, no arrow
        for i, k in enumerate(RQ["ac_k"]):
            j = RQ["text_k"].index(k)
            ax.plot([RQ["ac_tok"][i], RQ["text_tok"][j]], [RQ["ac_acc"][i], RQ["text_acc"][j]],
                    color="#d0d0d0", lw=0.6, ls=(0, (1, 1.5)), zorder=2)
        ax.text(15.2, 54.3, "9.5× fewer tokens", color="#777777", ha="center", va="bottom",
                fontsize=6.5, style="italic")
    ax.set_xlabel("Mean Input Tokens (Thousands)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) Accuracy vs. input")
    xs = range(3)
    axr.bar(xs, RQ["recall_all"], color=G_LIGHT, width=0.62, zorder=2, edgecolor=G_MID, lw=0.5)
    for x, v in zip(xs, RQ["recall_all"]):
        axr.text(x, v + 1.5, f"{v:.0f}", ha="center")
    axr.set_xticks(list(xs)); axr.set_xticklabels([str(k) for k in RQ["recall_k"]])
    axr.set_ylim(0, 100); axr.set_xlabel("Top-k"); axr.set_ylabel("Recall@k (%)", labelpad=1)
    axr.set_title("(b) Recall")
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.19, top=0.89)
    save(fig, f"r2_{variant}_retrieval")


# ---------------------------------------------------------------- corpus compression (single column)
def corpus(variant):
    setup()
    fig = plt.figure(figsize=(COL, 2.4))
    gs = GridSpec(1, 2, width_ratios=[5.2, 0.85], wspace=0.07, figure=fig)
    ax, axc = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for a in (ax, axc):
        clean(a); a.set_ylim(15, 88); a.set_yticks(range(20, 90, 10))
    axc.spines["left"].set_visible(False); axc.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlim(-0.1, 2.45); axc.set_xlim(90, 110)
    refs = [("Gold Context", CC_REF["gold"], LS_GOLD), ("Long Context", CC_REF["long"], LS_LONG),
            ("AM Non-Uniform", CC_REF["am"], LS_DOT)]
    for _, y, ls in refs:
        for a in (ax, axc):
            a.axhline(y, color=REF, lw=0.8, ls=ls, zorder=1)
    styles = {"KV Non-Uniform": (OURS, "o", "-"), "KV Uniform": (OURS_LIGHT, "s", "-"),
              "Input + Residuals": (OURS_DARK, "^", (0, (4, 1.5))), "Input": (OURS_MID, "D", (0, (4, 1.5)))}
    hs = []
    for lab, ys in CC.items():
        col, mk, ls = styles[lab]
        h, = ax.plot(CC_X, ys, marker=mk, color=col, ls=ls, zorder=3, ms=3.6, mec="white", mew=0.5,
                     path_effects=HALO)
        hs.append(h)
    rh = [Line2D([], [], color=REF, ls=ls, lw=0.8) for _, _, ls in refs]
    leg1 = ax.legend(hs, list(CC), loc="lower right", bbox_to_anchor=(1.0, 0.0), title="AC (ours)",
                     handlelength=2.0, labelspacing=0.2, borderaxespad=0.1, title_fontsize=SMALL,
                     alignment="left")
    leg1.get_title().set_fontweight("bold"); leg1.get_title().set_color(OURS)
    ax.add_artist(leg1)
    leg2 = ax.legend(rh, [r[0] for r in refs], loc="lower left", bbox_to_anchor=(0.0, 0.0), title="References",
                     handlelength=2.0, labelspacing=0.2, borderaxespad=0.1, title_fontsize=SMALL,
                     alignment="left", prop=dict(style="italic", size=SMALL))
    leg2.get_title().set_fontweight("bold")
    axc.plot([100], [CC_REF["cart"]], marker="D", color=G_MID, ms=4.2, zorder=3)
    axc.annotate("Cartridges", (100, CC_REF["cart"]), xytext=(0, -10), textcoords="offset points",
                 ha="center", color="#555555")
    axc.set_xticks([100]); ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    d, kw = .012, dict(color="k", clip_on=False, lw=0.6)
    ax.plot((1 - d * 0.6, 1 + d * 0.6), (-d * 2, d * 2), transform=ax.transAxes, **kw)
    axc.plot((-d * 3.5, d * 3.5), (-d * 2, d * 2), transform=axc.transAxes, **kw)
    fig.supxlabel("Training Tokens per Context (Millions)", fontsize=LAB, y=0.01)
    ax.set_ylabel("Accuracy (%)")
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.15, top=0.98)
    save(fig, f"r2_{variant}_corpus")


# ---------------------------------------------------------------- compaction (double column)
def compaction(variant):
    setup()
    fig = plt.figure(figsize=(DCOL, 1.95))
    gs = GridSpec(1, 3, width_ratios=[2.2, 2.2, 1.0], wspace=0.30, figure=fig)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    sty = {"Full Context": dict(color=G_DARK), "No Context": dict(color="white", hatch="//////", edgecolor=G_MID),
           "Text Compaction": dict(color=G_MID), "AC 5×": dict(color=OURS), "AC 10×": dict(color=OURS_LIGHT)}
    w = 0.16
    for k, (ax, ylab, ylim) in enumerate(zip(axes[:2], ["Solved (%)", "Continuation Tokens (K)"],
                                             [(40, 77), (0, 45)])):
        clean(ax)
        for i, (arm, vals) in enumerate(CP.items()):
            xs = [g + (i - 2) * w for g in range(3)]
            kw = dict(sty[arm]); kw.setdefault("edgecolor", "white")
            ax.bar(xs, vals[k], w, label=arm, zorder=2, lw=0.5, **kw)
            for x, v in zip(xs, vals[k]):
                ours = arm.startswith("AC")
                ax.text(x, v + 0.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5,
                        color=OURS_DARK if ours else "#444444", weight="bold" if ours else None)
        ax.set_xticks(range(3)); ax.set_xticklabels(CP_GROUPS); ax.set_ylim(*ylim); ax.set_ylabel(ylab)
        ax.set_xlabel("Continuation Difficulty")
    axes[0].set_title("(a) Quality"); axes[1].set_title("(b) Cost to finish")
    ax = axes[2]; clean(ax)
    for i, (lab, m, lo, hi) in enumerate(CP_TIME):
        ax.bar(i, m, 0.55, color=G_MID if i == 0 else OURS, zorder=2, yerr=[[m - lo], [hi - m]], capsize=2,
               error_kw=dict(lw=0.7, capthick=0.7))
        txt = f"{m} s"
        if i == 1 and variant == "callout":
            txt = "1.3 s\n5.7× faster"
        ax.text(i, hi + 0.45, txt, ha="center", va="bottom", linespacing=1.0,
                color=OURS_DARK if i == 1 else "#333333", weight="bold" if i == 1 else None)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Text", "AC"]); ax.set_ylim(0, 10)
    ax.set_ylabel("Seconds"); ax.set_xlabel("Compaction"); ax.set_title("(c) Compaction time")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), handlelength=1.3, columnspacing=1.4)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.2, top=0.8)
    save(fig, f"r2_{variant}_compaction")


# ---------------------------------------------------------------- multi-agent (single column)
def multiagent(variant):
    setup()
    fig, ax = plt.subplots(figsize=(COL, 2.15))
    clean(ax)
    sty = [dict(color=G_LIGHT), dict(color="white", hatch="//////", edgecolor=G_MID), dict(color=G_MID),
           dict(color="white", hatch="......", edgecolor=G_DARK), dict(color=G_DARK), dict(color=OURS)]
    w = 0.135
    for i, (arm, vals) in enumerate(MA.items()):
        xs = [g + (i - 2.5) * w for g in range(3)]
        kw = dict(sty[i]); kw.setdefault("edgecolor", "white")
        ax.bar(xs, vals, w, label=arm, zorder=2, lw=0.5, **kw)
        if arm.endswith("(ours)"):
            for x, v in zip(xs, vals):
                ax.text(x + 0.02, v + 0.35, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=OURS_DARK,
                        weight="bold", path_effects=TXT_HALO, zorder=5)
    for g, c in enumerate(MA_CEIL):
        ax.plot([g - 0.46, g + 0.46], [c, c], color=REF, lw=0.8, ls=LS_DOT, zorder=3)
    ax.text(2.46, MA_CEIL[2] + 0.35, "Any of 3", ha="right", va="bottom", style="italic", color=REF,
            path_effects=TXT_HALO)
    ax.set_ylim(70, 96); ax.set_yticks(range(70, 96, 5))
    ax.set_xticks(range(3)); ax.set_xticklabels(MA_GROUPS)
    ax.set_ylabel("Problems Solved (%)"); ax.set_xlabel("Problem Difficulty", labelpad=2)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.52, 1.0), handlelength=1.1,
               columnspacing=0.9, labelspacing=0.25)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.19, top=0.8)
    save(fig, f"r2_{variant}_multiagent")


def multiagent_r3(name, wide, decimals, rotate=False):
    """Round 3: a number on every bar; wide = double column (figure*), else single column."""
    setup()
    fig, ax = plt.subplots(figsize=(DCOL, 1.7) if wide else (COL, 2.5))
    clean(ax)
    sty = [dict(color=G_LIGHT), dict(color="white", hatch="//////", edgecolor=G_MID), dict(color=G_MID),
           dict(color="white", hatch="......", edgecolor=G_DARK), dict(color=G_DARK), dict(color=OURS)]
    w = 0.14
    for i, (arm, vals) in enumerate(MA.items()):
        xs = [g + (i - 2.5) * w for g in range(3)]
        kw = dict(sty[i]); kw.setdefault("edgecolor", "white")
        ax.bar(xs, vals, w, label=arm, zorder=2, lw=0.5, **kw)
        ours = arm.endswith("(ours)")
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.3, f"{v:.1f}" if decimals else f"{v:.0f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=90 if rotate else 0, color=OURS_DARK if ours else "#333333",
                    weight="bold" if ours else None, path_effects=TXT_HALO, zorder=5)
    for g, c in enumerate(MA_CEIL):
        ax.plot([g - 0.46, g + 0.46], [c, c], color=REF, lw=0.8, ls=LS_DOT, zorder=3)
        ax.text(g - 0.46, c + 0.3, f"Any of 3: {c:.1f}" if decimals else f"Any of 3: {c:.0f}", ha="left",
                va="bottom", style="italic", fontsize=6.5, color=REF, path_effects=TXT_HALO)
    top = 99 if rotate else 97
    ax.set_ylim(70, top); ax.set_yticks(range(70, 96, 5))
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(MA_GROUPS)
    ax.set_ylabel("Problems Solved (%)"); ax.set_xlabel("Problem Difficulty", labelpad=2)
    h, l = ax.get_legend_handles_labels()
    if wide:
        fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), handlelength=1.1, columnspacing=1.3)
        fig.subplots_adjust(left=0.065, right=0.995, bottom=0.21, top=0.86)
    else:
        fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.52, 1.0), handlelength=1.1,
                   columnspacing=0.9, labelspacing=0.25)
        fig.subplots_adjust(left=0.13, right=0.99, bottom=0.16, top=0.83)
    save(fig, f"r3_{name}_multiagent")


def sheet(variant):
    import matplotlib.image as mpimg
    imgs = {k: mpimg.imread(OUT / f"r2_{variant}_{k}.png") for k in ("retrieval", "corpus", "compaction", "multiagent")}
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.2, 6.6), dpi=150)
    fig.text(0.01, 0.99, f"Round 2 – {variant}", fontsize=10, weight="bold", va="top")
    slots = {"retrieval": [0.01, 0.63, 0.47, 0.33], "corpus": [0.51, 0.60, 0.47, 0.37],
             "compaction": [0.01, 0.32, 0.97, 0.28], "multiagent": [0.255, 0.0, 0.47, 0.31]}
    for k, r in slots.items():
        a = fig.add_axes(r); a.imshow(imgs[k]); a.axis("off")
    fig.savefig(OUT / f"sheet_r2_{variant}.png"); plt.close(fig)


if __name__ == "__main__":
    for v in ("plain", "callout"):
        retrieval(v); corpus(v); compaction(v); multiagent(v); sheet(v)
    # round 3 (Oct 25): quieter retrieval link; multi-agent with a number on every bar
    retrieval("quiet"); retrieval("links")
    multiagent_r3("wide_dec", wide=True, decimals=True)
    multiagent_r3("wide_int", wide=True, decimals=False)
    multiagent_r3("col_int", wide=False, decimals=False)
    multiagent_r3("col_dec_rot", wide=False, decimals=True, rotate=True)
    print("ok")
