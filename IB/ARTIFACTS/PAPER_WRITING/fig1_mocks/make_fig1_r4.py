"""Figure 1 round 4 (Oct 26; user: show scale; wider gap between panels; (b) without side text and as narrow as
possible; (a) larger; text channel and AC about the same length, the naive full context ~1.5-2x longer and
crossing the context window in the middle of the subagent trajectories).
Helpers and identity come from make_fig1.py.
Run: uv run --no-project --python 3.13 --with matplotlib python make_fig1_r4.py
"""
from matplotlib.patches import Rectangle
from make_fig1 import (DCOL, OURS, OURS_LIGHT, OURS_DARK, G_DARK, G_MID, G_LIGHT, G_PALE, LAB, SMALL, TINY,
                       setup, canvas, box, arrow, save)

H = 2.70
X0 = 0.92           # left edge of the strips
B0 = 5.85           # left edge of panel (b)
ROW_Y = {"naive": 1.96, "today": 1.36, "ac": 0.74}
SH = 0.32

# (label, width, what the text channel loses, AC source kind, AC source label)
TODAY = [("task", 0.24, None, None, None),
         ("compaction", 0.54, "history\nelided", "enc", "full\nhistory"),
         ("steps", 0.28, None, None, None),
         ("subagent\nanswers", 0.54, "process\nhidden", "enc", "subagent\ntrajectories"),
         ("steps", 0.26, None, None, None),
         ("top-20\ndocs", 0.50, "low\nrecall", "enc", "top-200\ndocs\n(high recall)"),
         ("selected\npassages", 0.60, "information\nloss", "kv", "precomputed\ndoc KVs")]
WIN = X0 + sum(s[1] for s in TODAY)          # the context window ends where today's strip ends
NAIVE = [("task", 0.24), ("full history", 1.94), ("steps", 0.28), ("subagent trajectories", 1.00),
         ("steps", 0.24), ("top-200\ndocs", 0.44), ("long\ndocument", 0.50)]


def strip_label(ax, y, text, color, sub=None):
    if sub is None:
        ax.text(0.02, y + SH / 2, text, fontsize=SMALL, va="center", weight="bold", color=color, linespacing=1.0)
        return
    ax.text(0.02, y + SH / 2 + 0.01, text, fontsize=SMALL, va="bottom", weight="bold", color=color)
    ax.text(0.02, y + SH / 2 - 0.01, sub, fontsize=TINY, va="top", color=color)


def seg(ax, x, y, w, lab, kind):
    fc, ec, color = {"text": (G_PALE, G_MID, G_DARK), "lossy": (G_MID, "none", "white"),
                     "full": (G_LIGHT, G_MID, G_DARK)}[kind]
    box(ax, x, y, w - 0.04, SH, fc, ec=ec, text=lab, color=color, size=TINY)


def panel_a(ax):
    naive_end = X0 + sum(w for _, w in NAIVE) - 0.04
    ax.text((X0 + naive_end) / 2, H - 0.12, "(a) Passing harness information to the model", fontsize=LAB,
            weight="bold", va="center", ha="center")
    ax.plot([WIN - 0.02] * 2, [0.12, ROW_Y["naive"] + SH + 0.08], color=G_DARK, lw=0.8, ls=(0, (4, 2)))
    ax.text(WIN - 0.02, ROW_Y["naive"] + SH + 0.11, "context window", fontsize=TINY, style="italic",
            ha="center", va="bottom", color=G_DARK)

    # naive full context: crosses the window inside the subagent trajectories
    y = ROW_Y["naive"]
    strip_label(ax, y, "Naive full\ncontext", G_DARK)
    x = X0
    for lab, w in NAIVE:
        seg(ax, x, y, w, lab, "full")
        x += w
    end = x - 0.04
    ax.add_patch(Rectangle((WIN - 0.02, y - 0.02), end - WIN + 0.04, SH + 0.04, fc="white", alpha=0.35,
                           ec="none", zorder=2))
    ax.add_patch(Rectangle((WIN - 0.02, y - 0.02), end - WIN + 0.04, SH + 0.04, fc="none", ec=G_MID, lw=0.6,
                           ls=(0, (2, 1.5)), zorder=2))
    ax.text((WIN + end) / 2, y - 0.05, "overflows the context window", fontsize=TINY, style="italic",
            color=G_MID, ha="center", va="top")

    # text channel (today)
    y = ROW_Y["today"]
    strip_label(ax, y, "Lossy Text", G_DARK, sub="(Current Approaches)")
    x = X0
    centers = []
    for lab, w, loss, *_ in TODAY:
        seg(ax, x, y, w, lab, "lossy" if loss else "text")
        if loss:
            ax.text(x + (w - 0.04) / 2, y - 0.04, loss, ha="center", va="top", fontsize=TINY, style="italic",
                    color=G_MID, linespacing=1.0)
        centers.append((x, w))
        x += w

    # Activation Context: same budget as the text channel, each AC fed by its source
    y = ROW_Y["ac"]
    strip_label(ax, y, "Activation\nContext", OURS_DARK)
    for (lab, w, loss, kind, src), (x, _) in zip(TODAY, centers):
        if kind is None:
            seg(ax, x, y, w, lab, "text")
            continue
        box(ax, x, y, w - 0.04, SH, OURS, text="AC", color="white", size=TINY, weight="bold")
        cx = x + (w - 0.04) / 2
        if kind == "kv":
            for k in range(3):
                for j in range(4):
                    ax.add_patch(Rectangle((cx - 0.15 + j * 0.08, 0.33 + k * 0.065), 0.07, 0.055,
                                           fc=OURS_LIGHT, ec="white", lw=0.4))
            top = 0.52
        else:
            box(ax, cx - 0.22, 0.34, 0.44, 0.18, OURS_LIGHT, text="encoder", size=TINY)
            top = 0.52
        ax.text(cx, 0.29, src, ha="center", va="top", fontsize=TINY, color=G_DARK, linespacing=1.0)
        arrow(ax, cx, top, cx, y)


# ------------------------------------------------------------------ (b) write depths, grids only
DERIVED = "derived"   # drawn as a pale tint with vermilion hatching, distinct from the encoder fill


def cell(ax, x, y, w, h, fc, ec, ls):
    if fc == DERIVED:
        ax.add_patch(Rectangle((x, y), w, h, fc="#FFF3EE", ec=OURS, lw=0, hatch="//////"))
        fc = "none"
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.5, ls=ls))


def panel_b(ax):
    naive_end = X0 + sum(w for _, w in NAIVE) - 0.04   # thin divider halfway between the panels
    ax.plot([(naive_end + B0) / 2] * 2, [0.05, H - 0.02], color=G_LIGHT, lw=0.6)
    ax.text(B0 + 0.45, H - 0.12, "(b) Write Modes", fontsize=LAB, weight="bold", va="center", ha="center")
    cw, ch, ncol, ac_cols = 0.10, 0.08, 6, (2, 3)
    gx = B0 + 0.24
    rows = ["emb", "L1", "L2", "L3"]
    for g, name in enumerate(["Input Embeddings", "Hidden Residuals", "All-Layer KVs"]):
        gy = 1.78 - g * 0.60
        ax.text(B0, gy + 4 * ch + 0.15, name, fontsize=SMALL, weight="bold", color=OURS_DARK)
        for r, rl in enumerate(rows):
            ax.text(gx - 0.04, gy + r * ch + ch / 2, rl, fontsize=TINY, ha="right", va="center", color=G_MID)
            for c in range(ncol):
                fc, ec, ls = G_PALE, "white", "-"
                if c in ac_cols:
                    if g == 0:
                        fc = OURS if r == 0 else DERIVED
                    elif g == 1:
                        fc = OURS if r in (0, 2) else DERIVED
                    else:
                        fc, ec, ls = ("none", G_LIGHT, (0, (1.5, 1.5))) if r == 0 else (OURS, "white", "-")
                cell(ax, gx + c * cw, gy + r * ch, cw, ch, fc, ec, ls)
        ax.text(gx + ncol * cw / 2, gy + 4 * ch + 0.005, r"$\vdots$", ha="center", va="bottom", fontsize=SMALL,
                color=G_MID)
    # compact legend, one item per line
    for i, (fc, lab) in enumerate([(G_PALE, "Text token"), (OURS, "AC"), (DERIVED, "Derived from AC")]):
        yy = 0.30 - i * 0.11
        cell(ax, B0, yy, 0.09, 0.08, fc, "white", "-")
        ax.text(B0 + 0.13, yy + 0.04, lab, fontsize=TINY, va="center", color=G_DARK)


if __name__ == "__main__":
    setup()
    fig, ax = canvas(DCOL, H)
    panel_a(ax)
    panel_b(ax)
    save(fig, "fig1_r4")
