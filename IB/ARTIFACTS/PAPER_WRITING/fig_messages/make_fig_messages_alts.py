"""Figure 2 alternatives to JSON (Oct 26; user: dict/JSON is not paper-friendly outside appendices).
  A  chat cards: one row per message, role in a gutter, each content part a typed chip (grey text, vermilion AC)
  B  typed constructors: system(...), user(text(...), activation_context(...)) in monospace, changed parts shaded
  C  diff table: rows = messages, columns = text only | with Activation Context, cells hold the part chips
Run: uv run --no-project --with matplotlib python make_fig_messages_alts.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fig1_mocks"))
from make_fig1 import DCOL, OURS, OURS_DARK, G_DARK, G_MID, G_LIGHT, G_PALE, LAB, SMALL, TINY, setup  # noqa: E402

OUT = Path(__file__).parent
MONO = "DejaVu Sans Mono"
TINT = "#FFF3EE"

# message rows: (role, [parts]); a part is (kind, type label, content, note); kind: text | lossy | ac
TEXT_ONLY = [("system", [("text", "text", "You are an agent ...", None)]),
             ("user", [("text", "text", "Task: Which ...?", None),
                       ("lossy", "text", "Summary of steps 1–140: ...", "lossy summary")]),
             ("assistant", [("text", "tool call", "search(...)", None)]),
             ("tool", [("lossy", "text", "[1] ... [20]", "top 20 only")])]
WITH_AC = [("system", [("text", "text", "You are an agent ...", None)]),
           ("user", [("text", "text", "Task: Which ...?", None),
                     ("ac", "activation_context", "trajectory: steps 1–140", "all 140 steps")]),
           ("assistant", [("text", "tool call", "search(...)", None)]),
           ("tool", [("ac", "activation_context", "documents: top 200", "top 200")])]


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def text_w(ax, x, y, txt, **kw):
    """Draw text and return its width in inches (measured, so chips fit their text)."""
    t = ax.text(x, y, txt, **kw)
    fig = ax.figure
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    return bb.width / fig.dpi


def chip(ax, x, y, kind, label, content, h=0.17):
    """A typed content part: small type tag + content. Returns its width."""
    fc, ec, tag_c, txt_c = {"text": ("white", G_LIGHT, G_MID, G_DARK), "lossy": (G_PALE, G_MID, G_MID, G_DARK),
                            "ac": (TINT, OURS, OURS, OURS_DARK)}[kind]
    bold = "bold" if kind == "ac" else "normal"
    tw = text_w(ax, x + 0.05, y + h / 2, label, fontsize=TINY - 0.3, family=MONO, color=tag_c, va="center",
                weight=bold, zorder=3)
    sep = x + 0.05 + tw + 0.05
    cw = text_w(ax, sep + 0.05, y + h / 2, content, fontsize=SMALL, color=txt_c, va="center", weight=bold, zorder=3)
    w = sep + 0.05 + cw + 0.06 - x
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.03", fc=fc, ec=ec, lw=0.6, zorder=2))
    ax.plot([sep, sep], [y + 0.03, y + h - 0.03], color=ec, lw=0.5, zorder=3)
    return w


# ------------------------------------------------------------------ A: chat cards
def mock_a():
    W = DCOL
    rows_h = [0.24, 0.42, 0.24, 0.24]
    H = sum(rows_h) + 0.36
    fig, ax = canvas(W, H)
    pw, gap = (W - 0.15) / 2, 0.15
    for p, (title, msgs) in enumerate([("Text only", TEXT_ONLY), ("With Activation Context", WITH_AC)]):
        x0 = p * (pw + gap)
        ax.text(x0 + pw / 2, H - 0.10, title, fontsize=LAB, weight="bold", ha="center", va="center")
        y = H - 0.26
        for (role, parts), rh in zip(msgs, rows_h):
            y -= rh
            ax.add_patch(Rectangle((x0, y + 0.02), pw, rh - 0.04, fc="#F7F7F7", ec="none"))
            ax.text(x0 + 0.06, y + rh - 0.13, role, fontsize=SMALL, style="italic", color=G_MID, va="center")
            yy = y + rh - 0.215
            for kind, label, content, note in parts:
                w = chip(ax, x0 + 0.62, yy, kind, label, content)
                if note:
                    ax.text(x0 + pw - 0.05, yy + 0.085, note, fontsize=TINY, style="italic", ha="right", va="center",
                            color=OURS if kind == "ac" else G_MID)
                yy -= 0.19
    fig.savefig(OUT / "alt_A_chat_cards.png", dpi=300, bbox_inches="tight", pad_inches=0.01)


# ------------------------------------------------------------------ B: typed constructors (monospace)
B_TEXT = [("system(\"You are an agent ...\")", None),
          ("user(text(\"Task: Which ...?\"),", None),
          ("     text(\"Summary of steps 1-140: ...\"))", "lossy summary"),
          ("assistant(tool_call(search(...)))", None),
          ("tool(text(\"[1] ... [20]\"))", "top 20 only")]
B_AC = [("system(\"You are an agent ...\")", None),
        ("user(text(\"Task: Which ...?\"),", None),
        ("     activation_context(\"trajectory\", steps))", "all 140 steps"),
        ("assistant(tool_call(search(...)))", None),
        ("tool(activation_context(\"documents\", top_200))", "top 200")]


def mono_line(ax, x, y, line, fs, cw):
    """Draw a line, coloring `activation_context` vermilion bold and role names grey."""
    import re
    for m in re.finditer(r"activation_context|system|user|assistant|tool_call|tool|text|[^a-z_]+|[a-z_0-9]+", line):
        tok = m.group(0)
        color, weight = G_DARK, "normal"
        if tok == "activation_context":
            color, weight = OURS, "bold"
        elif tok in ("system", "user", "assistant", "tool"):
            color, weight = G_DARK, "bold"
        elif tok in ("text", "tool_call"):
            color = G_MID
        ax.text(x + m.start() * cw, y, tok, fontsize=fs, family=MONO, color=color, weight=weight, va="center")


def mock_b():
    W, fs = DCOL, 7.0
    cw, lh = 0.602 * fs / 72, 0.16
    H = len(B_AC) * lh + 0.40
    fig, ax = canvas(W, H)
    pw, gap = (W - 0.15) / 2, 0.15
    for p, (title, lines, tint, edge) in enumerate([("Text only", B_TEXT, G_PALE, G_MID),
                                                    ("With Activation Context", B_AC, TINT, OURS)]):
        x0 = p * (pw + gap)
        ax.text(x0 + pw / 2, H - 0.10, title, fontsize=LAB, weight="bold", ha="center", va="center")
        ax.add_patch(Rectangle((x0, H - 0.25 - len(lines) * lh - 0.03), pw, len(lines) * lh + 0.06, fc="#F7F7F7", ec=G_LIGHT, lw=0.5))
        for i, (line, note) in enumerate(lines):
            y = H - 0.33 - i * lh
            if note:
                ax.add_patch(Rectangle((x0, y - lh / 2), pw, lh, fc=tint, ec="none"))
                ax.add_patch(Rectangle((x0, y - lh / 2), 0.025, lh, fc=edge, ec="none"))
                ax.text(x0 + pw - 0.05, y, note, fontsize=SMALL - 0.3, style="italic", ha="right", va="center",
                        color=OURS if edge == OURS else G_MID)
            mono_line(ax, x0 + 0.07, y, line, fs, cw)
    fig.savefig(OUT / "alt_B_constructors.png", dpi=300, bbox_inches="tight", pad_inches=0.01)


# ------------------------------------------------------------------ C: diff table
def mock_c():
    W = DCOL
    rows_h = [0.22, 0.40, 0.22, 0.22]
    H = sum(rows_h) + 0.30
    fig, ax = canvas(W, H)
    c0, c1, c2 = 0.0, 0.75, 0.75 + (W - 0.75) / 2
    cwid = (W - 0.75) / 2
    ax.text(c1 + cwid / 2, H - 0.12, "Text only", fontsize=LAB, weight="bold", ha="center", va="center")
    ax.text(c2 + cwid / 2, H - 0.12, "With Activation Context", fontsize=LAB, weight="bold", ha="center", va="center")
    ax.plot([0, W], [H - 0.24, H - 0.24], color=G_DARK, lw=0.8)
    y = H - 0.24
    for r, rh in enumerate(rows_h):
        y -= rh
        role = TEXT_ONLY[r][0]
        ax.text(c0 + 0.04, y + rh - 0.12, role, fontsize=SMALL, style="italic", color=G_MID, va="center")
        for cx, msgs in ((c1, TEXT_ONLY), (c2, WITH_AC)):
            yy = y + rh - 0.205
            for kind, label, content, note in msgs[r][1]:
                chip(ax, cx + 0.04, yy, kind, label, content)
                if note:
                    ax.text(cx + cwid - 0.06, yy + 0.085, note, fontsize=TINY, style="italic", ha="right",
                            va="center", color=OURS if kind == "ac" else G_MID)
                yy -= 0.18
        ax.plot([0, W], [y, y], color=G_LIGHT, lw=0.5)
    ax.plot([0, W], [y, y], color=G_DARK, lw=0.8)
    fig.savefig(OUT / "alt_C_diff_table.png", dpi=300, bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    setup()
    mock_a()
    mock_b()
    mock_c()
