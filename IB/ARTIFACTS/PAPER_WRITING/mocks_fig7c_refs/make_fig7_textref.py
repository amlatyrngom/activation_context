"""Fig. 7 (section 4.5) mock with text reference lines in (c) (Oct 27; user: maybe put the ideal-recall comparison in both figures).
References = base Qwen3-4B-Instruct-2507 (the section 4.5 reader, untuned) reading the same dev200 tasks as TEXT, greedy, lenient parse
(a bare one-line answer counts): Text top-20 with both golds (perfect recall@20) 53.0, Gold Reference 67.5, No Context 19.0
($CLAUDE_JOB_DIR/tmp/pr/res/text4b/, kit_pr/text4b_sec45.py). Curves: the OLD section 4.5 runs (v3) until the v4 runs finish.
Run: uv run --no-project --python 3.13 --with matplotlib --with numpy python make_fig7c_refs.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sec45")); sys.path.insert(0, str(HERE.parent / "mocks_r2"))
import make_sec45 as F  # noqa: E402
import make_r2 as R  # noqa: E402

REFS = [(53.0, (0, (3, 1.2, 1, 1.2, 1, 1.2)), "Text, perfect top-20", "#555555")]

F.OUT = HERE
_savefig = plt.Figure.savefig


def savefig(fig, path, **kw):            # draw the references on panel (c) just before make_sec45 saves
    ax = fig.axes[2]
    if not getattr(ax, "_refs", False):
        ax._refs = True
        x1 = ax.get_xlim()[1]
        for y, ls, label, c in REFS:
            ax.axhline(y, color=c, lw=0.8, ls=ls, zorder=1)
            ax.text(x1, y + 0.9, label, ha="right", va="bottom", fontsize=R.SMALL - 0.5, style="italic", color=c,
                    path_effects=R.TXT_HALO)
    return _savefig(fig, str(path).replace("fig_sec45", "fig_sec45_textref"), **kw)


plt.Figure.savefig = savefig
F.main()
