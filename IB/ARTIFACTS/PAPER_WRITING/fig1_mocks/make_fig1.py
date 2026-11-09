"""Figure 1 mocks (Oct 26): three candidate overview figures for the introduction, in the shared identity of
mocks_r2/make_r2.py (vermilion = ours, greys = text/baselines, STIX type, drawn at print width).
A: before/after trajectory + write depths (double column).  B: interface swimlanes (single column).
C: design space of write depth x producer (single column).
Run: uv run --no-project --python 3.13 --with matplotlib python make_fig1.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path

OUT = Path(__file__).parent
COL, DCOL = 3.25, 6.75
OURS, OURS_LIGHT, OURS_DARK = "#D7301F", "#F4A582", "#8C1A0E"
G_DARK, G_MID, G_LIGHT, G_PALE = "#333333", "#969696", "#CCCCCC", "#EDEDED"
LAB, SMALL, TINY = 8, 7, 6.5


def setup():
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": SMALL,
                         "savefig.dpi": 300, "pdf.fonttype": 42, "hatch.linewidth": 0.5})


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc, ec="none", text=None, color="black", size=SMALL, lw=0.6, hatch=None, r=0.03,
        weight="normal", style="normal", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                hatch=hatch, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
                weight=weight, style=style)


def arrow(ax, x0, y0, x1, y1, color=OURS, ls="-", lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, color=color, lw=lw,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ------------------------------------------------------------------ A: before / after + write depths
def mock_a():
    fig, ax = canvas(DCOL, 2.05)
    # (a) trajectories
    segs = [  # label, text width, lossy?, AC width, source shown to the encoder, what text loses
        ("task", 0.34, False, None, None, None),
        ("steps", 0.42, False, None, None, None),
        ("summary", 0.62, True, 0.30, "full history\n(20k)", "history\nelided"),
        ("steps", 0.34, False, None, None, None),
        ("subagent\nanswers", 0.62, True, 0.30, "3 subagent\ntrajectories", "process\nhidden"),
        ("top-20 docs", 0.70, True, 0.30, "top-200\ndocs", "180 docs\ndropped"),
        ("corpus (60k)", 0.80, True, 0.30, "corpus", "fills the\nwindow"),
    ]
    x0 = 0.80
    ax.text(0.02, 1.62, "Text channel\n(today)", fontsize=SMALL, va="center", weight="bold", color=G_DARK)
    ax.text(0.02, 0.86, "Activation\nContext", fontsize=SMALL, va="center", weight="bold", color=OURS_DARK)
    x = x0
    for lab, w, lossy, *_ in segs:
        box(ax, x, 1.50, w - 0.04, 0.24, G_MID if lossy else G_PALE, ec=G_MID, text=lab,
            color="white" if lossy else G_DARK, size=TINY)
        if lossy:
            ax.text(x + (w - 0.04) / 2, 1.45, segs[[s[0] for s in segs].index(lab)][5], ha="center", va="top",
                    fontsize=TINY, style="italic", color=G_MID, linespacing=1.0)
        x += w
    x = x0
    for lab, w, lossy, acw, src, _ in segs:
        if lossy:  # AC slot on the left; the rest of the text span is context freed for the agent
            cx = x + (w - 0.04) / 2
            box(ax, x, 0.74, acw - 0.04, 0.24, OURS, text="AC", color="white", size=TINY, weight="bold")
            box(ax, x + acw, 0.74, w - acw - 0.04, 0.24, "none", ec=G_LIGHT, ls=(0, (2, 1.5)),
                text="free" if w - acw > 0.3 else None, color=G_MID, size=TINY, style="italic")
            box(ax, cx - 0.26, 0.36, 0.52, 0.20, OURS_LIGHT, text="AC encoder", size=TINY)
            ax.text(cx, 0.30, src, ha="center", va="top", fontsize=TINY, color=G_DARK, linespacing=1.0)
            arrow(ax, cx, 0.56, x + (acw - 0.04) / 2, 0.74)
            x += w
        else:
            box(ax, x, 0.74, w - 0.04, 0.24, G_PALE, ec=G_MID, text=lab, color=G_DARK, size=TINY)
            x += w
    ax.text(x0, 1.95, "(a) The same trajectory, as text and with Activation Context", fontsize=LAB,
            weight="bold", va="center")

    # (b) write depths into the reader
    bx, bw = 5.55, 0.80
    ax.text(bx + bw / 2 - 0.25, 1.95, "(b) Three write depths", fontsize=LAB, weight="bold", va="center",
            ha="center")
    ys = [0.36, 0.62, 0.88, 1.14, 1.40]
    for i, y in enumerate(ys):
        box(ax, bx, y, bw, 0.20, "white", ec=G_DARK, text="embeddings" if i == 0 else f"layer {i}",
            size=TINY, color=G_DARK)
        if i:  # per-layer KV cache cells
            for j in range(4):
                fc = OURS if (i in (2, 3, 4) and j == 3) else G_PALE
                ax.add_patch(Rectangle((bx + bw + 0.06 + j * 0.07, y + 0.04), 0.06, 0.12, fc=fc, ec=G_MID,
                                       lw=0.4))
    ax.text(bx + bw + 0.20, ys[0] + 0.10, "KV cache", fontsize=TINY, ha="center", va="center", color=G_DARK)
    for y0, y1 in zip(ys, ys[1:]):
        ax.plot([bx + bw / 2] * 2, [y0 + 0.20, y1], color=G_DARK, lw=0.6)
    ex = 4.86
    box(ax, ex - 0.02, 0.80, 0.42, 0.26, OURS_LIGHT, text="AC\nencoder", size=TINY)
    arrow(ax, ex + 0.40, 0.86, bx, ys[0] + 0.10, ls=(0, (3, 1.5)), rad=-0.2)
    ax.text(ex + 0.42, 0.44, "input", fontsize=TINY, color=OURS_DARK, style="italic")
    arrow(ax, ex + 0.40, 0.98, bx, ys[2] + 0.10)
    ax.text(ex + 0.44, 1.01, "residual", fontsize=TINY, color=OURS_DARK, style="italic", va="bottom")
    arrow(ax, ex + 0.20, 1.06, bx + bw + 0.27, ys[4] + 0.17, rad=-0.35)
    ax.text(ex + 0.30, 1.50, "KV", fontsize=TINY, color=OURS_DARK, style="italic")
    save(fig, "fig1_A_before_after")


# ------------------------------------------------------------------ B: interface swimlanes
def mock_b():
    fig, ax = canvas(COL, 2.35)
    lanes = [("Harness", 0.02, 1.38), ("Engine", 1.46, 0.78), ("Reader", 2.30, 0.93)]
    for name, x, w in lanes:
        box(ax, x, 0.05, w, 2.05, "#FAFAFA", ec=G_LIGHT, r=0.04)
        ax.text(x + w / 2, 2.20, name, ha="center", fontsize=LAB, weight="bold")
    code = ('declare("traj",\n'
            '  depth="residual",\n'
            "  ratio=1/10)\n\n"
            "msg = [text(task),\n"
            '  ac("traj", history)]')
    ax.text(0.08, 1.98, code, fontsize=TINY, family="DejaVu Sans Mono", va="top", color=G_DARK,
            linespacing=1.25)
    ax.text(0.72, 0.18, "declares encoders,\nplaces AC in messages", ha="center", fontsize=TINY,
            style="italic", color=G_MID)
    box(ax, 1.54, 1.30, 0.62, 0.34, OURS_LIGHT, text="AC encoder", size=TINY)
    box(ax, 1.54, 0.70, 0.62, 0.34, "white", ec=G_DARK, text="splice / add /\ningest KV", size=TINY)
    arrow(ax, 1.40, 1.47, 1.54, 1.47, color=G_DARK)
    arrow(ax, 1.85, 1.30, 1.85, 1.04)
    ax.text(1.85, 0.18, "KV-cache safe:\nwrites only in\nthe ongoing turn", ha="center", fontsize=TINY,
            style="italic", color=G_MID)
    ys = [0.55, 0.85, 1.15, 1.45]
    for i, y in enumerate(ys):
        box(ax, 2.42, y, 0.62, 0.22, "white", ec=G_DARK, text="embed" if i == 0 else f"layer {i}", size=TINY)
    arrow(ax, 2.16, 0.80, 2.42, ys[0] + 0.11, ls=(0, (3, 1.5)))
    arrow(ax, 2.16, 0.87, 2.42, ys[2] + 0.11)
    arrow(ax, 2.16, 0.94, 2.42, ys[3] + 0.11, color=OURS_DARK)
    save(fig, "fig1_B_interface")


# ------------------------------------------------------------------ C: design space
def mock_c():
    fig, ax = canvas(COL, 2.25)
    rows = ["Input", "Residual", "KV"]
    cols = ["Encoder (online)", "Fitted per corpus (offline)"]
    cells = {  # (row, col): (prior work, ours)
        (0, 0): ("Multimodal (LLaVA, Qwen-VL)\nICAE, xRAG, gist tokens", "Retrieval"),
        (1, 0): ("—", "Retrieval, Compaction?,\nMulti-agent?"),
        (2, 0): ("C2C, KVComm", "(open)"),
        (0, 1): ("—", "Corpus (ablation)"),
        (1, 1): ("—", "Corpus (ablation)"),
        (2, 1): ("Prefix-tuning, Cartridges, AM", "Corpus"),
    }
    x0, y0, cw, rh = 0.62, 0.08, 1.30, 0.58
    for j, c in enumerate(cols):
        ax.text(x0 + cw * (j + 0.5), y0 + 3 * rh + 0.08, c, ha="center", fontsize=SMALL, weight="bold")
    for i, r in enumerate(rows):
        y = y0 + (2 - i) * rh
        ax.text(x0 - 0.06, y + rh / 2, r, ha="right", va="center", fontsize=SMALL, weight="bold")
        for j in range(2):
            x = x0 + j * cw
            ax.add_patch(Rectangle((x, y), cw, rh, fc="white", ec=G_LIGHT, lw=0.6))
            prior, ours = cells[(i, j)]
            ax.text(x + 0.06, y + rh - 0.06, prior, fontsize=TINY, color=G_MID, va="top")
            ax.text(x + 0.06, y + 0.06, ours, fontsize=TINY, color=OURS, va="bottom", weight="bold")
    ax.text(x0 - 0.06, y0 + 3 * rh + 0.08, "write depth", ha="right", fontsize=TINY, style="italic",
            color=G_MID)
    save(fig, "fig1_C_design_space")


if __name__ == "__main__":
    setup()
    mock_a()
    mock_b()
    mock_c()
