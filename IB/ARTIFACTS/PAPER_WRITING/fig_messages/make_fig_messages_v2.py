"""Figure 2, JSON v2 (Oct 26; user: one message per line, `[` and `]` on their own lines, smaller font).
One message per line needs ~125 characters, so the two versions are stacked (full width each) instead of side by side.
The parts that differ are shaded inline (grey = lossy text part, pale vermilion = AC part).
Run: uv run --no-project --with matplotlib python make_fig_messages_v2.py
"""
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fig1_mocks"))
from make_fig1 import DCOL, OURS, G_DARK, G_MID, G_LIGHT, G_PALE, LAB, SMALL, setup  # noqa: E402

OUT = Path(__file__).parent
MONO = "DejaVu Sans Mono"
FS = 6.0                      # recomputed in main() so the longest line plus its note fits the width
CW = 0.602 * FS / 72
LH = 0.112
NOTE_W = 0.78                 # room kept at the right for the notes (in)
KEY = "#7A7A7A"
TINT = "#FFF3EE"
AC_WORDS = {"activation_context", "encoder", "input"}

SUMMARY = '{"type": "text", "text": "Summary of steps 1-140: ..."}'
TOP20 = '{"type": "text", "text": "[1] ... [20]"}'
AC_TRAJ = '{"type": "activation_context", "encoder": "trajectory", "input": steps}'
AC_DOCS = '{"type": "activation_context", "encoder": "documents", "input": top_200}'

TEXT_ONLY = [
    '[',
    '  {"role": "system", "content": "You are an agent ..."},',
    '  {"role": "user", "content": "Task: Which ...?"},',
    '  {"role": "user", "content": [' + SUMMARY + ']},',
    '  {"role": "assistant", "tool_calls": [search(...)]},',
    '  {"role": "tool", "content": [' + TOP20 + ']},',
    '  ...',
    ']',
]
WITH_AC = [
    '[',
    '  {"role": "system", "content": "You are an agent ..."},',
    '  {"role": "user", "content": "Task: Which ...?"},',
    '  {"role": "user", "content": [' + AC_TRAJ + ']},',
    '  {"role": "assistant", "tool_calls": [search(...)]},',
    '  {"role": "tool", "content": [' + AC_DOCS + ']},',
    '  ...',
    ']',
]
# (substring to shade, note)
MARK_TEXT = [(SUMMARY, "lossy summary"), (TOP20, "top 20 only")]
MARK_AC = [(AC_TRAJ, "all 140 steps"), (AC_DOCS, "top 200")]

TOKEN = re.compile(r'"[^"]*"|[A-Za-z_][A-Za-z_0-9]*|\S|\s+')


def draw_line(ax, x0, y, line):
    for m in TOKEN.finditer(line):
        tok = m.group(0)
        if tok.isspace():
            continue
        color, weight = G_DARK, "normal"
        if tok.strip('"') in AC_WORDS:
            color, weight = OURS, "bold"
        elif tok.startswith('"') and line[m.end():m.end() + 1] == ":":
            color = KEY
        ax.text(x0 + m.start() * CW, y, tok, fontsize=FS, family=MONO, color=color, weight=weight, va="center",
                ha="left", zorder=3)


def block(ax, top, title, lines, marks, tint, edge, W):
    h = len(lines) * LH + 0.08
    ax.add_patch(Rectangle((0, top - h), W, h, fc="#F7F7F7", ec=G_LIGHT, lw=0.5, zorder=1))
    ax.text(W - 0.06, top - 0.04 - LH / 2, title, fontsize=LAB, weight="bold", ha="right", va="center", zorder=3)
    x0 = 0.07
    for i, line in enumerate(lines):
        y = top - 0.04 - LH / 2 - i * LH
        for sub, note in marks:
            j = line.find(sub)
            if j < 0:
                continue
            xs, xe = x0 + j * CW + 0.004, x0 + (j + len(sub)) * CW + 0.004
            ax.add_patch(FancyBboxPatch((xs, y - LH * 0.42), xe - xs, LH * 0.84, boxstyle="round,pad=0,rounding_size=0.02",
                                        fc=tint, ec=edge, lw=0.5, zorder=2))
            ax.text(x0 + len(line.rstrip()) * CW + 0.08, y, note, fontsize=SMALL - 0.5, style="italic", va="center",
                    ha="left", color=OURS if edge == OURS else G_MID, zorder=3)
        draw_line(ax, x0, y, line)
    return h


def main():
    global FS, CW, LH
    setup()
    W = DCOL
    longest = max(len(l) for l in TEXT_ONLY + WITH_AC)
    FS = min(6.4, (W - NOTE_W - 0.1) * 72 / (0.602 * longest))
    CW, LH = 0.602 * FS / 72, FS / 72 * 1.32
    print(f"longest line {longest} chars -> code font {FS:.2f} pt")
    gap = 0.08
    H = 2 * (len(WITH_AC) * LH + 0.08) + gap
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    h1 = block(ax, H, "Text only", TEXT_ONLY, MARK_TEXT, G_PALE, G_MID, W)
    block(ax, H - h1 - gap, "With Activation Context", WITH_AC, MARK_AC, TINT, OURS, W)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_messages_v2.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    main()
