"""Fig. 3, variant B (user's pick, Oct 27), all measured. eval500, mean of 3 passes. Solid = BM25 top-k (fig_retrieval values);
hollow dashed = ideal recall (both golds injected into the top k, rq_common.inject_golds): Text = base Qwen3.5-9B thinking off,
AC = Arm G at 1/10 (S3 rq_runs/20261027_ideal_recall/). Ideal tokens: text = measured prompt tokens (2,402 / 11,996 / 24,350);
AC = the frontier's tok_counts at the same k (331 / 1,306 / 2,559).
Run: uv run --no-project --python 3.13 --with matplotlib python make_fig3_b_full.py
"""
import shutil
import make_fig3_mocks as M

M.IDEAL = dict(text_tok=[2.402, 11.996, 24.350], text_acc=[66.33, 64.07, 63.80], text_k=[20, 100, 200],
               ac_tok=[0.331, 1.306, 2.559], ac_acc=[67.13, 63.67, 62.87], ac_k=[20, 100, 200])
M.stamp = lambda fig: None
R, RQ = M.R, M.RQ


def mock_b_final():
    import matplotlib.pyplot as plt
    R.setup()
    fig, ax = plt.subplots(figsize=(R.COL, 2.3))
    R.clean(ax); ax.set_xscale("log"); ax.set_xlim(0.18, 40); ax.set_ylim(15, 75)
    M.refs(ax)
    M.series(ax, RQ, hollow=False, ac_off={0: (0, -10), 1: (0, -10)})
    M.series(ax, M.IDEAL, hollow=True, tx_off={0: (-9, -3), 1: (0, -10), 2: (0, -10)}, ac_off={0: (0, -10), 1: (0, -10), 2: (0, -10)})
    ax.set_xticks([0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.3k", "1k", "3k", "10k", "30k"]); ax.minorticks_off()
    ax.text(36, 44.5, "BM25", ha="right", color="#555555", fontsize=7, style="italic")
    ax.text(36, 56.8, "ideal recall", ha="right", color="#555555", fontsize=7, style="italic")
    ax.text(0.2, 45.5, "AC (ours)", color=R.OURS, fontsize=R.LAB, weight="bold")
    ax.text(3.2, 39.0, "Text", color="#666666", fontsize=R.LAB)
    ax.set_xlabel("Mean Input Tokens (log)"); ax.set_ylabel("Accuracy (%)")
    ax.text(0.99, 0.02, "labels: top-k", transform=ax.transAxes, ha="right", fontsize=6, color="#888888")
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.19, top=0.97)
    R.save(fig, "fig3_B_full")


mock_b_final()
