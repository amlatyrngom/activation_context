"""Figure 1 round 2 (Oct 26; user: S1 + mock A). Panel (a) now has three rows: full text (naive, overflows),
today's text channel, and Activation Context (corpus as precomputed compressed KVs, no encoder).
Panel (b) is redone three ways: b1 position x layer grids, b2 one transformer block with three taps,
b3 a captioned list. Helpers and identity come from make_fig1.py.
Run: uv run --no-project --python 3.13 --with matplotlib python make_fig1_r2.py
"""
from matplotlib.patches import Rectangle
from make_fig1 import (DCOL, OURS, OURS_LIGHT, OURS_DARK, G_DARK, G_MID, G_LIGHT, G_PALE, LAB, SMALL, TINY,
                       setup, canvas, box, arrow, save)

H = 2.55            # figure height (in)
X0 = 0.80           # left edge of the trajectory strips
WIN = 4.26          # right edge of the context window (end of today's strip)
B0 = 4.62           # left edge of panel (b)
ROW_Y = {"naive": 1.86, "today": 1.30, "ac": 0.70}
SH = 0.22           # strip height


def strip_label(ax, y, text, color):
    ax.text(0.02, y + SH / 2, text, fontsize=SMALL, va="center", weight="bold", color=color, linespacing=1.0)


def seg(ax, x, y, w, lab, kind):
    if kind == "text":
        box(ax, x, y, w - 0.04, SH, G_PALE, ec=G_MID, text=lab, color=G_DARK, size=TINY)
    elif kind == "lossy":
        box(ax, x, y, w - 0.04, SH, G_MID, text=lab, color="white", size=TINY)
    elif kind == "full":
        box(ax, x, y, w - 0.04, SH, G_LIGHT, ec=G_MID, text=lab, color=G_DARK, size=TINY)


def panel_a(ax):
    ax.text(X0, H - 0.12, "(a) Passing harness information to the model", fontsize=LAB, weight="bold",
            va="center")
    # context window marker
    ax.plot([WIN + 0.02] * 2, [0.10, ROW_Y["naive"] + SH + 0.10], color=G_DARK, lw=0.8, ls=(0, (4, 2)))
    ax.text(WIN + 0.02, ROW_Y["naive"] + SH + 0.13, "context window", fontsize=TINY, style="italic",
            ha="center", va="bottom", color=G_DARK)

    # row 1: full text (naive); clipped at the window edge and running past it
    y = ROW_Y["naive"]
    strip_label(ax, y, "Full text\n(naive)", G_DARK)
    x = X0
    for lab, w in [("task", 0.30), ("steps", 0.36), ("full history (20k)", 0.90), ("steps", 0.30),
                   ("3 subagent trajectories", 1.02)]:
        seg(ax, x, y, w, lab, "full")
        x += w
    # top-200 docs + corpus: starts inside the window and runs past it (the part past the edge is hatched)
    box(ax, x, y, WIN - x + 0.02, SH, G_LIGHT, ec=G_MID, text="docs + corpus", color=G_DARK, size=TINY)
    ax.add_patch(Rectangle((WIN + 0.04, y), 0.24, SH, fc="none", ec=G_MID, lw=0.6, hatch="////",
                           ls=(0, (2, 1.5))))
    ax.text(WIN - 0.02, y - 0.04, "overflows the window", fontsize=TINY, style="italic", color=G_MID,
            va="top", ha="right")

    # row 2: today's text channel
    y = ROW_Y["today"]
    strip_label(ax, y, "Text channel\n(today)", G_DARK)
    today = [("task", 0.30, None), ("steps", 0.36, None), ("summary", 0.55, "history\nelided"),
             ("steps", 0.30, None), ("subagent\nanswers", 0.55, "process\nhidden"),
             ("top-20 docs", 0.62, "low recall"), ("corpus (60k)", 0.80, "fills the\nwindow")]
    x = X0
    for lab, w, loss in today:
        seg(ax, x, y, w, lab, "lossy" if loss else "text")
        if loss:
            ax.text(x + (w - 0.04) / 2, y - 0.04, loss, ha="center", va="top", fontsize=TINY, style="italic",
                    color=G_MID, linespacing=1.0)
        x += w

    # row 3: Activation Context
    y = ROW_Y["ac"]
    strip_label(ax, y, "Activation\nContext", OURS_DARK)
    acw = 0.26
    x = X0
    for lab, w, src in [("task", 0.30, None), ("steps", 0.36, None), ("", 0.55, "full history"),
                        ("steps", 0.30, None), ("", 0.55, "3 subagent\ntrajectories"),
                        ("", 0.62, "top-200 docs\n(high recall)"), ("corpus", 0.80, "compressed\ncorpus KVs")]:
        if not src:
            seg(ax, x, y, w, lab, "text")
            x += w
            continue
        cx = x + (w - 0.04) / 2
        slot = "KV" if lab == "corpus" else "AC"
        box(ax, x, y, acw - 0.04, SH, OURS, text=slot, color="white", size=TINY, weight="bold")
        box(ax, x + acw, y, w - acw - 0.04, SH, "none", ec=G_LIGHT, ls=(0, (2, 1.5)),
            text="free" if w - acw > 0.2 else None, color=G_MID, size=TINY, style="italic")
        if lab == "corpus":  # precomputed offline: a small KV stack, no encoder at inference time
            for k in range(3):
                for j in range(4):
                    ax.add_patch(Rectangle((cx - 0.15 + j * 0.08, 0.30 + k * 0.07), 0.07, 0.06,
                                           fc=OURS_LIGHT, ec="white", lw=0.4))
            ax.text(cx, 0.26, "precomputed\noffline", ha="center", va="top", fontsize=TINY, color=G_DARK,
                    linespacing=1.0)
            arrow(ax, cx, 0.51, x + (acw - 0.04) / 2, y)
        else:
            box(ax, cx - 0.25, 0.34, 0.50, 0.18, OURS_LIGHT, text="AC encoder", size=TINY)
            ax.text(cx, 0.30, src, ha="center", va="top", fontsize=TINY, color=G_DARK, linespacing=1.0)
            arrow(ax, cx, 0.52, x + (acw - 0.04) / 2, y)
        x += w


# ------------------------------------------------------------------ (b) variants
def legend(ax, x, y, items):
    for fc, ec, hatch, lab in items:
        ax.add_patch(Rectangle((x, y), 0.09, 0.08, fc=fc, ec=ec, lw=0.4, hatch=hatch))
        ax.text(x + 0.12, y + 0.04, lab, fontsize=TINY, va="center", color=G_DARK)
        x += 0.16 + 0.055 * len(lab)


def panel_b1(ax):
    """Position x layer grids, one per depth, stacked; each with a one-line meaning."""
    ax.text(B0, H - 0.12, "(b) Three write depths", fontsize=LAB, weight="bold", va="center")
    legend(ax, B0, H - 0.36, [(G_PALE, G_MID, None, "text"), (OURS, "white", None, "written"),
                              (OURS_LIGHT, "white", None, "computed")])
    cw, ch, ncol = 0.13, 0.085, 6
    ac_cols = (2, 3)
    rows = ["emb", "L1", "L2", "L3"]
    specs = [("Input", "replaces token\nembeddings"), ("Residual", "added to hidden\nstates at layers"),
             ("KV", "inserted into each\nlayer's cache")]
    for g, (name, meaning) in enumerate(specs):
        gy = 1.55 - g * 0.66
        gx = B0 + 0.22
        ax.text(B0, gy + 4 * ch + 0.07, name, fontsize=SMALL, weight="bold", color=OURS_DARK)
        for r, rl in enumerate(rows):
            ax.text(gx - 0.04, gy + r * ch + ch / 2, rl, fontsize=5.5 if False else TINY, ha="right",
                    va="center", color=G_MID) if g == 0 or True else None
            for c in range(ncol):
                fc, ec, ls = G_PALE, G_MID, "-"
                if c in ac_cols:
                    if name == "Input":
                        fc = OURS if r == 0 else OURS_LIGHT
                    elif name == "Residual":
                        fc = OURS if r in (0, 2) else OURS_LIGHT
                    else:  # KV: no token computation; keys/values written at every layer
                        fc, ec, ls = (("none", G_LIGHT, (0, (1.5, 1.5))) if r == 0 else (OURS, "white", "-"))
                ax.add_patch(Rectangle((gx + c * cw, gy + r * ch), cw, ch, fc=fc, ec=ec if ec != G_MID else "white",
                                       lw=0.5, ls=ls))
        ax.text(gx + ncol * cw + 0.08, gy + 2 * ch, meaning, fontsize=TINY, va="center", color=G_DARK,
                linespacing=1.05)
    ax.text(B0 + 0.22 + 3 * cw, 0.10, r"token positions $\rightarrow$", fontsize=TINY, ha="center", style="italic",
            color=G_MID)


def panel_b2(ax):
    """One transformer block with the three write points."""
    ax.text(B0, H - 0.12, "(b) Three write depths", fontsize=LAB, weight="bold", va="center")
    # bottom to top: embeddings, residual add point, attention (+ KV cache), MLP; AC source bottom right
    sx = B0 + 0.30          # residual stream x
    bx, bw = B0 + 0.50, 0.62
    box(ax, B0 + 0.05, 0.22, 0.95, 0.20, "white", ec=G_DARK, text="token embeddings", size=TINY)
    ax.plot([sx, sx], [0.42, 1.98], color=G_MID, lw=2.2, solid_capstyle="butt")
    ax.text(sx + 0.06, 1.98, "residual stream", fontsize=TINY, va="top", color=G_MID)
    ax.add_patch(Rectangle((B0 + 0.12, 0.50), 1.62, 1.32, fc="none", ec=G_LIGHT, lw=0.6, ls=(0, (3, 2))))
    ax.text(B0 + 1.70, 1.76, "× L layers", fontsize=TINY, style="italic", color=G_MID, ha="right", va="top")
    box(ax, bx, 0.86, bw, 0.24, "white", ec=G_DARK, text="attention", size=TINY)
    box(ax, bx, 1.32, bw, 0.24, "white", ec=G_DARK, text="MLP", size=TINY)
    for y0 in (0.98, 1.44):
        ax.plot([sx, bx], [y0, y0], color=G_MID, lw=0.6)
    kx = B0 + 1.40
    ax.plot([bx + bw, kx], [0.98, 0.98], color=G_MID, lw=0.6, ls=(0, (1, 1.2)))
    for j in range(4):
        ax.add_patch(Rectangle((kx + j * 0.06, 0.90), 0.05, 0.16, fc=OURS if j == 3 else G_PALE, ec=G_MID,
                               lw=0.4))
    ax.text(kx + 0.12, 1.09, "KV cache", fontsize=TINY, ha="center", va="bottom", color=G_DARK)
    ax.plot([sx], [0.64], marker="o", ms=7, mfc="white", mec=OURS, mew=0.9)
    ax.text(sx, 0.645, "+", ha="center", va="center", fontsize=SMALL, color=OURS, weight="bold")
    ex0, ex1, ey0, ey1 = B0 + 1.40, B0 + 1.86, 0.26, 0.56
    box(ax, ex0, ey0, ex1 - ex0, ey1 - ey0, OURS_LIGHT, text="AC", size=SMALL, weight="bold")
    arrow(ax, ex0, 0.32, B0 + 1.00, 0.32, ls=(0, (3, 1.5)))
    ax.text(B0 + 1.20, 0.22, "input", fontsize=TINY, color=OURS_DARK, style="italic", ha="center", va="top")
    arrow(ax, ex0, 0.50, sx + 0.08, 0.64)
    ax.text(B0 + 0.62, 0.62, "residual", fontsize=TINY, color=OURS_DARK, style="italic", va="bottom")
    arrow(ax, kx + 0.21, ey1, kx + 0.21, 0.90, color=OURS_DARK)
    ax.text(kx + 0.26, 0.70, "KV", fontsize=TINY, color=OURS_DARK, style="italic", va="center")


def panel_b3(ax):
    """Captioned list with a small glyph per depth."""
    ax.text(B0, H - 0.12, "(b) Three write depths", fontsize=LAB, weight="bold", va="center")
    items = [("Input", "AC vectors take the place of\ntoken embeddings, like image\npatches in a VLM",
              "retrieval"),
             ("Residual", "AC vectors are added to the\nhidden states at chosen layers",
              "retrieval"),
             ("KV", "AC keys and values are written\ninto every layer's cache, with\nno forward pass",
              "corpus compression")]
    for i, (name, text, used) in enumerate(items):
        y = 1.78 - i * 0.62
        # glyph: 4-layer stack with the written level highlighted
        for r in range(4):
            fc = G_PALE
            if (name == "Input" and r == 0) or (name == "Residual" and r == 2) or (name == "KV" and r > 0):
                fc = OURS
            ax.add_patch(Rectangle((B0, y - 0.30 + r * 0.085), 0.22, 0.07, fc=fc, ec="white", lw=0.4))
        ax.text(B0 + 0.30, y, name, fontsize=SMALL, weight="bold", color=OURS_DARK, va="top")
        ax.text(B0 + 0.30, y - 0.12, text, fontsize=TINY, va="top", color=G_DARK, linespacing=1.05)
        ax.text(2.0 + B0 - 0.02, y, f"e.g. {used}", fontsize=TINY, va="top", ha="right", style="italic",
                color=G_MID)


def build(name, panel_b):
    fig, ax = canvas(DCOL, H)
    panel_a(ax)
    panel_b(ax)
    save(fig, name)


if __name__ == "__main__":
    setup()
    build("fig1_r2_b1_grids", panel_b1)
    build("fig1_r2_b2_block", panel_b2)
    build("fig1_r2_b3_list", panel_b3)
