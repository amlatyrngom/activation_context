"""§3.1 retrieval figure, pre-rebuttal draft (2026-11-14, user: an adapted copy of the paper's Fig. 3 = fig_retrieval_v3; frozen-reader AC
only; untuned text as the comparison; NO text + LoRA line yet).
(a) accuracy vs mean reader input tokens (log x): BM25 top-k (solid) and ideal recall (dashed hollow: both golds injected; k = 20 is the
    20I rows), for AC = frozen-reader KL stage 2 (T_res_rq9fr_kI_KL, the user's main row) and Text = untuned Qwen3.5-9B.
    Accuracies: HotpotQA dev1000, greedy, cap 1024, one scorer, read from scoreboard.json (written by ../scoreboard.py).
    x positions: the paper's mean input tokens (eval500; text 2,396 / 11,996 / 24,351, ideal 2,402 / 11,996 / 24,350; AC 331 / 1,306 / 2,559)
    -- the dev1000 means are not yet computed (same corpus and BM25, so expected within a few percent).
    Gold reference (untuned 9B, two gold passages only) and no context (question alone): horizontal lines from ../refs.json
    (test_table.py; reads by jobs/rq_ref_orcd.sh, 2026-11-15), same split and scorer; drawn only when present. The Select -> text
    baseline stays out (eval500, old sampled scorer, no new read).
(b) TTFT at k = 200 (paper's systems measurement, unchanged: IB/TMP/PAPER_SYSTEMS setup B), Text and AC only; accuracy annotations are
    the new BM25 k = 200 numbers.
Run: uv run --no-project --python 3.13 --with matplotlib python make_fig_retrieval_s31.py [test]   (default dev1000 draft)
"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
PW = HERE.parents[2] / "PAPER_WRITING"
sys.path.insert(0, str(PW / "mocks_fig3_v3"))
import make_fig3_v3 as V  # noqa: E402  (also puts mocks_r2 and fig3_ideal_final on the path)
R, F = V.R, V.F
R.OUT = HERE
DASH = (0, (3, 1.5))

SPLIT = "test" if "--split=test" in sys.argv or "test" in sys.argv[1:] else "dev"   # test: held-out test1000 (test_table.py -> test.json)
if SPLIT == "test":
    SB = json.loads((HERE.parent / "test.json").read_text())["systems"]; AC_RUN, TEXT_RUN = "ac", "text_base"
else:
    SB = json.loads((HERE.parent / "scoreboard.json").read_text())["systems"]; AC_RUN, TEXT_RUN = "T_res_rq9fr_kI_KL", "text_base"
K = [20, 100, 200]
_rp = HERE.parent / "refs.json"
REF = json.loads(_rp.read_text()).get("test1000" if SPLIT == "test" else "dev1000", {}) if _rp.exists() else {}
TOK = dict(text_bm25=[2.396, 11.996, 24.351], text_ideal=[2.402, 11.996, 24.350], ac=[0.331, 1.306, 2.559])


def accs(run, kind):
    keys = [f"bm25_k{k}" for k in K] if kind == "bm25" else ["20I", "100I", "200I"]
    return [SB[run][c]["acc"] for c in keys]


def panel_acc(ax):
    lo = min(40, 5 * int((REF["none"] - 6) // 5)) if "none" in REF else 40
    R.clean(ax); ax.set_xscale("log"); ax.set_xlim(0.2, 120); ax.set_ylim(lo, 75)
    if "gold" in REF:
        R.ref(ax, REF["gold"], R.LS_GOLD, "Gold Reference")
    if "none" in REF:
        R.ref(ax, REF["none"], R.LS_DOT, "No Context", x=0.22, ha="left")
    tb, ti, ab, ai = accs(TEXT_RUN, "bm25"), accs(TEXT_RUN, "ideal"), accs(AC_RUN, "bm25"), accs(AC_RUN, "ideal")
    ax.plot(TOK["text_bm25"], tb, marker="s", color=R.G_MID, ms=3.4, zorder=3)
    ax.plot(TOK["ac"], ab, marker="o", color=R.OURS, ms=4.0, mec="white", mew=0.6, zorder=4, path_effects=R.HALO)
    ax.plot(TOK["text_ideal"], ti, marker="s", color=R.G_MID, ms=3.4, ls=DASH, mfc="white", zorder=3)
    ax.plot(TOK["ac"], ai, marker="o", color=R.OURS, ms=4.0, ls=DASH, mfc="white", mec=R.OURS, mew=1.0, zorder=4)
    F.labels(ax, TOK["text_bm25"], tb, K, "#666666", {0: (-9, 2)})
    F.labels(ax, TOK["ac"], ab, K, R.OURS, {})
    F.labels(ax, TOK["text_ideal"], ti, K, "#666666", {0: (2, 5)} if "gold" not in REF else {0: (-11, -1)})
    F.labels(ax, TOK["ac"], ai, K, R.OURS, {})
    # as the paper's Fig. 3: one name per system + a BM25 / ideal-recall legend (the gold line leaves no room for per-line labels)
    ax.text(0.215, ab[0] - 8.0, "AC (ours)", color=R.OURS, fontsize=R.LAB, fontweight="bold", va="center")
    ax.text(TOK["text_bm25"][2] * 1.35, tb[2], "Text", color="#666666", fontsize=R.LAB, ha="left", va="center")
    h = [Line2D([], [], color="#555555", lw=1.2, label="BM25"),
         Line2D([], [], color="#555555", lw=1.2, ls=DASH, marker="o", ms=3, mfc="white", label="Ideal recall (golds in top-k)")]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.16), fontsize=6.3, handlelength=2.4, borderaxespad=0.2)
    ax.set_xticks([0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.3k", "1k", "3k", "10k", "30k"]); ax.minorticks_off()
    ax.set_xlabel("Mean Input Tokens (log); labels = top-k"); ax.set_ylabel("Accuracy (%)"); ax.set_title("(a) Accuracy vs. input")


if __name__ == "__main__":
    # panel (b): the paper's final TTFT bars (fig_retrieval_v3: one process, sky job 33, medians), Text and AC only, new k = 200 accuracies
    a200, t200 = SB[AC_RUN]["bm25_k200"]["acc"], SB[TEXT_RUN]["bm25_k200"]["acc"]
    V.BARS = [("Text", {"prefill": 1545.5}, 17.7, t200, False),
              ("AC (ours)", {"encode": 665.7 * 808.0 / 809.3, "prefill": 143.6 * 808.0 / 809.3}, 16.2, a200, True)]
    V.ST["encode"] = ("Encode (AC)", "#D9A441", None)
    V.ORDER = ["encode", "prefill"]
    R.setup()
    f = plt.figure(figsize=(R.COL, 3.2))
    gs = GridSpec(2, 1, height_ratios=[2.3, 0.75], hspace=0.78, figure=f)
    panel_acc(f.add_subplot(gs[0]))
    V.panel_ttft(f.add_subplot(gs[1]), "(b) Latency (2B encoder, 9B reader)", xmax=3000)
    f.subplots_adjust(left=0.17, right=0.985, bottom=0.11, top=0.94)
    R.save(f, "fig_retrieval_s31_test" if SPLIT == "test" else "fig_retrieval_s31_draft")
