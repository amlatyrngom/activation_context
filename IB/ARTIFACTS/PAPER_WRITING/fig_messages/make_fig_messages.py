"""Figure 2 (Oct 26): the same agent trajectory in the completions interface, text only vs with Activation Context.
Drawn instead of LaTeX listings (user: the listings version was unreadable in print). Monospace JSON in DejaVu Sans Mono,
changed parts shaded (grey = lossy text, pale vermilion = AC), short italic notes beside them. Identity from fig1_mocks.
Run: uv run --no-project --python 3.13 --with matplotlib python make_fig_messages.py
"""
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fig1_mocks"))
from make_fig1 import DCOL, OURS, OURS_DARK, G_DARK, G_MID, G_LIGHT, G_PALE, LAB, setup  # noqa: E402

OUT = Path(__file__).parent
MONO = "DejaVu Sans Mono"
FS = 6.6                      # code size (pt)
CW = 0.602 * FS / 72          # monospace advance (in)
LH = 0.118                    # line height (in)
KEY = "#7A7A7A"
AC_WORDS = {"activation_context", "encoder", "input"}
OURS_TINT = "#FFF3EE"

TEXT_ONLY = [
    '[{"role": "system", "content": "You are an agent ..."},',
    ' {"role": "user", "content": [',
    '   {"type": "text", "text": "Task: Which ...?"},',
    '   {"type": "text",',
    '    "text": "Summary of steps 1-140: ..."}]},',
    ' {"role": "assistant", "tool_calls": [search(...)]},',
    ' {"role": "tool", "content": [',
    '   {"type": "text", "text": "[1] ... [20]"}]},',
    ' ...]',
]
WITH_AC = [
    '[{"role": "system", "content": "You are an agent ..."},',
    ' {"role": "user", "content": [',
    '   {"type": "text", "text": "Task: Which ...?"},',
    '   {"type": "activation_context",',
    '    "encoder": "trajectory", "input": steps[1:140]}]},',
    ' {"role": "assistant", "tool_calls": [search(...)]},',
    ' {"role": "tool", "content": [',
    '   {"type": "activation_context",',
    '    "encoder": "documents", "input": top_200}]},',
    ' ...]',
]
# (first line, last line, note) of the parts that differ
BANDS_TEXT = [(3, 4, "lossy summary"), (7, 7, "top 20 only")]
BANDS_AC = [(3, 4, "all 140 steps"), (7, 8, "top 200")]

TOKEN = re.compile(r'"[^"]*"|[A-Za-z_][A-Za-z_0-9]*|\S|\s+')


def draw_line(ax, x0, y, line, next_char_is_colon):
    x, prev = x0, ""
    for m in TOKEN.finditer(line):
        tok = m.group(0)
        if tok in ",:}])[(" and prev and not prev.isspace() and (prev.endswith('"') or prev[0].isalpha() or prev[0] == "_"):
            x -= 0.35 * CW            # monospace punctuation after a word or closing quote reads as a stray space; tuck it in
        if not tok.isspace():
            word = tok.strip('"')
            color, weight = G_DARK, "normal"
            if word in AC_WORDS:
                color, weight = OURS, "bold"
            elif tok.startswith('"') and next_char_is_colon(m.end()):
                color = KEY
            ax.text(x, y, tok, fontsize=FS, family=MONO, color=color, weight=weight, va="center", ha="left")
        x += len(tok) * CW
        prev = tok


def panel(ax, x0, top, width, title, lines, bands, tint, edge):
    h = len(lines) * LH + 0.10
    ax.add_patch(Rectangle((x0, top - h), width, h, fc="#F7F7F7", ec=G_LIGHT, lw=0.5))
    ax.text(x0 + width / 2, top + 0.09, title, fontsize=LAB, weight="bold", ha="center", va="center")
    y_first = top - 0.05 - LH / 2
    for a, b, note in bands:
        yt, yb = y_first - a * LH + LH / 2, y_first - b * LH - LH / 2
        ax.add_patch(Rectangle((x0, yb), width, yt - yb, fc=tint, ec="none"))
        ax.add_patch(Rectangle((x0, yb), 0.025, yt - yb, fc=edge, ec="none"))
        ax.text(x0 + width - 0.06, yt - LH / 2, note, fontsize=FS + 0.4, style="italic", color=edge if edge == OURS else G_DARK,
                ha="right", va="center")
    for i, line in enumerate(lines):
        colon = lambda end, line=line: line[end:end + 1] == ":"
        draw_line(ax, x0 + 0.07, y_first - i * LH, line, colon)
    return h


def main():
    setup()
    W, gap = DCOL, 0.12
    pw = (W - gap) / 2
    H = len(WITH_AC) * LH + 0.10 + 0.22
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    top = H - 0.20
    panel(ax, 0, top, pw, "Text only", TEXT_ONLY, BANDS_TEXT, G_PALE, G_MID)
    panel(ax, pw + gap, top, pw, "With Activation Context", WITH_AC, BANDS_AC, OURS_TINT, OURS)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_messages.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    main()
