"""Figure 1 round 3 (Oct 26; user: keep b1 grids, better vocabulary and side text, show more layers above L3;
(a): no "(20k)", "summary" -> "compaction" placed right after the task, no free slots, the corpus becomes a long
document whose slot is an AC fed by precomputed document KVs).
Two layouts of the AC row: v1 packs the AC row to the left (the saved space shows as empty strip),
v2 keeps each AC under the text segment it replaces. Helpers and identity come from make_fig1.py.
Run: uv run --no-project --python 3.13 --with matplotlib python make_fig1_r3.py
"""
from matplotlib.patches import Rectangle
from make_fig1 import (DCOL, OURS, OURS_LIGHT, OURS_DARK, G_DARK, G_MID, G_LIGHT, G_PALE, LAB, SMALL, TINY,
                       setup, canvas, box, arrow, save)

H = 2.55
X0 = 0.80           # left edge of the strips
B0 = 4.62           # left edge of panel (b)
ROW_Y = {"naive": 1.86, "today": 1.30, "ac": 0.70}
SH = 0.22
ACW = 0.30          # width of an AC slot (incl. gap)

# today's trajectory: (label, width, what the text channel loses, AC source)
TODAY = [("task", 0.30, None, None),
         ("compaction", 0.58, "history\nelided", ("AC encoder", "full history")),
         ("steps", 0.36, None, None),
         ("subagent\nanswers", 0.56, "process\nhidden", ("AC encoder", "subagent\ntrajectories")),
         ("steps", 0.30, None, None),
         ("top-20 docs", 0.62, "low recall", ("AC encoder", "top-200 docs\n(high recall)")),
         ("long document", 0.74, "costly to\nprefill", ("kv", "precomputed\ndocument KVs"))]
WIN = X0 + sum(s[1] for s in TODAY)   # the context window ends where today's strip ends


def strip_label(ax, y, text, color):
    ax.text(0.02, y + SH / 2, text, fontsize=SMALL, va="center", weight="bold", color=color, linespacing=1.0)


def seg(ax, x, y, w, lab, kind):
    fc, ec, color = {"text": (G_PALE, G_MID, G_DARK), "lossy": (G_MID, "none", "white"),
                     "full": (G_LIGHT, G_MID, G_DARK)}[kind]
    box(ax, x, y, w - 0.04, SH, fc, ec=ec, text=lab, color=color, size=TINY)


def source(ax, cx, kind, label):
    """An AC encoder box, or a small block of precomputed KVs, centered at cx under the AC row."""
    if kind == "kv":
        for k in range(3):
            for j in range(4):
                ax.add_patch(Rectangle((cx - 0.15 + j * 0.08, 0.32 + k * 0.065), 0.07, 0.055, fc=OURS_LIGHT,
                                       ec="white", lw=0.4))
        top = 0.51
    else:
        box(ax, cx - 0.25, 0.34, 0.50, 0.18, OURS_LIGHT, text="AC encoder", size=TINY)
        top = 0.52
    ax.text(cx, 0.29, label, ha="center", va="top", fontsize=TINY, color=G_DARK, linespacing=1.0)
    return top


def panel_a(ax, layout):
    ax.text(X0, H - 0.12, "(a) Passing harness information to the model", fontsize=LAB, weight="bold",
            va="center")
    ax.plot([WIN + 0.02] * 2, [0.10, ROW_Y["naive"] + SH + 0.10], color=G_DARK, lw=0.8, ls=(0, (4, 2)))
    ax.text(WIN + 0.02, ROW_Y["naive"] + SH + 0.13, "context window", fontsize=TINY, style="italic",
            ha="center", va="bottom", color=G_DARK)

    # full text (naive)
    y = ROW_Y["naive"]
    strip_label(ax, y, "Full text\n(naive)", G_DARK)
    x = X0
    for lab, w in [("task", 0.30), ("full history", 0.86), ("steps", 0.36), ("subagent trajectories", 1.02),
                   ("steps", 0.30)]:
        seg(ax, x, y, w, lab, "full")
        x += w
    box(ax, x, y, WIN - x + 0.02, SH, G_LIGHT, ec=G_MID, text="docs + doc.", color=G_DARK, size=TINY)
    ax.add_patch(Rectangle((WIN + 0.04, y), 0.24, SH, fc="none", ec=G_MID, lw=0.6, hatch="////",
                           ls=(0, (2, 1.5))))
    ax.text(WIN - 0.02, y - 0.04, "overflows the window", fontsize=TINY, style="italic", color=G_MID,
            va="top", ha="right")

    # text channel (today)
    y = ROW_Y["today"]
    strip_label(ax, y, "Text channel\n(today)", G_DARK)
    x = X0
    centers = []
    for lab, w, loss, _ in TODAY:
        seg(ax, x, y, w, lab, "lossy" if loss else "text")
        if loss:
            ax.text(x + (w - 0.04) / 2, y - 0.04, loss, ha="center", va="top", fontsize=TINY, style="italic",
                    color=G_MID, linespacing=1.0)
        centers.append(x + (w - 0.04) / 2)
        x += w

    # Activation Context
    y = ROW_Y["ac"]
    strip_label(ax, y, "Activation\nContext", OURS_DARK)
    x = X0
    for (lab, w, loss, src), tc in zip(TODAY, centers):
        if layout == "aligned":    # every element sits under the text segment it corresponds to
            x = tc - ((ACW if src else w) - 0.04) / 2
        if src is None:
            seg(ax, x, y, w, lab, "text")
            x += w
            continue
        box(ax, x, y, ACW - 0.04, SH, OURS, text="AC", color="white", size=TINY, weight="bold")
        slot_c = x + (ACW - 0.04) / 2
        top = source(ax, tc, *src)
        arrow(ax, tc, top, slot_c, y)
        x += ACW
    if layout == "packed":
        ax.text(x + 0.04, y + SH / 2, "room for more", fontsize=TINY, style="italic", color=G_MID,
                va="center")


# ------------------------------------------------------------------ (b) write depths, position x layer grids
def legend(ax, x, y, items):
    for fc, lab in items:
        ax.add_patch(Rectangle((x, y), 0.09, 0.08, fc=fc, ec="white", lw=0.4))
        ax.text(x + 0.12, y + 0.04, lab, fontsize=TINY, va="center", color=G_DARK)
        x += 0.16 + 0.05 * len(lab)


def panel_b(ax):
    ax.text(B0, H - 0.12, "(b) Three write depths", fontsize=LAB, weight="bold", va="center")
    legend(ax, B0, H - 0.36, [(G_PALE, "text token"), (OURS, "written by AC"), (OURS_LIGHT, "derived")])
    cw, ch, ncol, ac_cols = 0.10, 0.08, 6, (2, 3)
    rows = ["emb", "L1", "L2", "L3"]
    specs = [("Input Embeddings", "AC vectors replace the token\nembeddings; every layer above\nis derived from them"),
             ("Hidden Residuals", "AC vectors are also added to\nthe hidden states of selected\nlayers"),
             ("All-Layer KVs", "AC keys and values are written\ninto every layer's cache; no\nforward pass is needed")]
    gx = B0 + 0.22
    for g, (name, meaning) in enumerate(specs):
        gy = 1.47 - g * 0.66
        ax.text(B0, gy + 4 * ch + 0.17, name, fontsize=SMALL, weight="bold", color=OURS_DARK)
        for r, rl in enumerate(rows):
            ax.text(gx - 0.04, gy + r * ch + ch / 2, rl, fontsize=TINY, ha="right", va="center", color=G_MID)
            for c in range(ncol):
                fc, ec, ls = G_PALE, "white", "-"
                if c in ac_cols:
                    if name.startswith("Input"):
                        fc = OURS if r == 0 else OURS_LIGHT
                    elif name.startswith("Hidden"):
                        fc = OURS if r in (0, 2) else OURS_LIGHT
                    else:  # no token at the embedding layer; keys/values written at every layer
                        fc, ec, ls = ("none", G_LIGHT, (0, (1.5, 1.5))) if r == 0 else (OURS, "white", "-")
                ax.add_patch(Rectangle((gx + c * cw, gy + r * ch), cw, ch, fc=fc, ec=ec, lw=0.5, ls=ls))
        # more layers above L3
        ax.text(gx + ncol * cw / 2, gy + 4 * ch + 0.005, r"$\vdots$", ha="center", va="bottom", fontsize=SMALL,
                color=G_MID)
        ax.text(gx + ncol * cw + 0.08, gy + 2 * ch, meaning, fontsize=TINY, va="center", color=G_DARK,
                linespacing=1.05)
    ax.text(gx + ncol * cw / 2, 0.11, r"token positions $\rightarrow$", fontsize=TINY, ha="center", va="top",
            style="italic", color=G_MID)


def build(name, layout):
    fig, ax = canvas(DCOL, H)
    panel_a(ax, layout)
    panel_b(ax)
    save(fig, name)


if __name__ == "__main__":
    setup()
    build("fig1_r3_v1_packed", "packed")
    build("fig1_r3_v2_aligned", "aligned")
