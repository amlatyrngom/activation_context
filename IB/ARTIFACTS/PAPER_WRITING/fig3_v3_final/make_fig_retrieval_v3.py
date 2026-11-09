"""Paper Fig. 3 v3 (user approved Oct 27 ~19:30Z: mock S2 with direct labels, "Text, ideal recall" right of its k = 200 point,
"AC, BM25" left of its k = 100 point). Single column:
(a) accuracy vs mean reader input tokens (log x): BM25 top-k (solid), ideal recall (dashed hollow), and Select → text (2B selector,
    at most 2 passages per call of 20, ranked prompt; eval500, 3 passes: 39.13 % at 1,272 tokens, 42.67 % at 2,461).
(b) time to first token, k = 200, one request (IB/TMP/PAPER_SYSTEMS/results.json, setup B: Qwen3.5-2B encoder/selector, Qwen3.5-9B
    reader, RTX PRO 6000, graph engine, 100 requests, medians). Each bar is the measured end-to-end median TTFT; its segments are the
    per-stage medians rescaled to that length. Select = decode_select_max (each request at its natural selector length).
Run: uv run --no-project --python 3.13 --with matplotlib python make_fig_retrieval_v3.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mocks_fig3_v3"))
import make_fig3_v3 as V  # noqa: E402
import make_fig3_s2_variants as S  # noqa: E402
R = V.R
R.OUT = HERE


def scaled(stages, ttft):
    k = ttft / sum(stages.values())
    return {key: v * k for key, v in stages.items()}


# Oct 27 20:10Z: all three bars from ONE process, sky job 33 (main9b|...|k=200|calls=10|twins=replay_v2, 100 requests, medians):
# select = exact replay of eval500.sel200r (the 42.7% run; selector + reader prompts matched 100/100, recorded per-call lengths).
V.BARS = [("Text", scaled({"prefill": 1545.5}, 1545.5), 17.7, 52.7, False),
          ("Select → text", scaled({"sprefill": 562.2, "sdecode": 261.6, "prefill": 140.8}, 1002.2), 16.2, 42.7, False),
          ("AC (ours)", scaled({"encode": 665.7, "prefill": 143.6}, 808.0), 16.2, 54.1, True)]

# Oct 28 (user): match Fig. 7(d): the encode segment is amber (the AC write, < 1 ms, would be invisible anyway).
ENCODE_AMBER = "#D9A441"   # = sec45_v4/make_sec45_v4.py SCHEMES["m2"]["encode"]
V.ST["encode"] = ("Encode (AC)", ENCODE_AMBER, None)
V.ST["write"] = ("AC write", ENCODE_AMBER, None)

if __name__ == "__main__":
    R.setup()
    f = plt.figure(figsize=(R.COL, 3.35))
    gs = GridSpec(2, 1, height_ratios=[2.3, 1.05], hspace=0.62, figure=f)
    S.l2t(f.add_subplot(gs[0]), right=True)
    V.panel_ttft(f.add_subplot(gs[1]), "(b) Latency (2B encoder, 9B reader)", xmax=3000)
    f.subplots_adjust(left=0.17, right=0.985, bottom=0.1, top=0.94)
    R.save(f, "fig_retrieval_v3")
