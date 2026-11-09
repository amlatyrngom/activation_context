"""Systems-breakdown mocks (Oct 27; user: reviewer flagged missing systems numbers — AC compute time per mode, prefill, generation,
prefill vanishing in KV mode; clean single-request measurements). LATENCIES ARE PLACEHOLDERS (shape only, stamped).
Memory numbers in mock D are COMPUTED from Qwen3-4B-Instruct-2507's config (36 layers, 8 KV heads, head_dim 128, hidden 2560, bf16):
  reader KV cache per row = 2 x 36 x 8 x 128 x 2 B = 147,456 B; AC row storage: Input Embeddings 2560 x 2 B = 5 KiB,
  Hidden Residuals 9 layers (3, 7, ..., 35) x 2560 x 2 B = 45 KiB, All-Layer KVs = one KV-cache row = 144 KiB.
Run: uv run --no-project --python 3.13 --with matplotlib python make_systems_mocks.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocks_r2"))
import make_r2 as R  # noqa: E402

OUT = Path(__file__).parent
R.OUT = OUT
ARMS = ["Text", "AC Input Emb.", "AC Hidden Res.", "AC All-Layer KVs", "AC KVs (cached)"]
# placeholder ms at 20 and 200 passages (4B reader, 4B encoder, 1 request at a time): encode, write, prefill
TTFT = {20: {"Text": (0, 0, 180), "AC Input Emb.": (170, 0, 25), "AC Hidden Res.": (175, 3, 28), "AC All-Layer KVs": (180, 6, 4),
             "AC KVs (cached)": (0, 6, 4)},
        200: {"Text": (0, 0, 2100), "AC Input Emb.": (1500, 0, 230), "AC Hidden Res.": (1550, 25, 250), "AC All-Layer KVs": (1600, 45, 4),
              "AC KVs (cached)": (0, 45, 4)}}
TPOT = {20: {"Text": 22.0, "AC Input Emb.": 21.0, "AC Hidden Res.": 21.0, "AC All-Layer KVs": 21.0, "AC KVs (cached)": 21.0},
        200: {"Text": 27.0, "AC Input Emb.": 21.5, "AC Hidden Res.": 21.5, "AC All-Layer KVs": 21.5, "AC KVs (cached)": 21.5}}
SEG = [("Encode", R.OURS_LIGHT, None), ("AC write", R.OURS_DARK, None), ("Reader prefill", R.G_MID, None)]
COLORS = {"Text": R.G_MID, "AC Input Emb.": R.OURS_LIGHT, "AC Hidden Res.": R.OURS, "AC All-Layer KVs": R.OURS_DARK,
          "AC KVs (cached)": R.OURS_DARK}


def stamp(fig):
    fig.text(0.5, 0.5, "PLACEHOLDER LATENCIES", ha="center", va="center", fontsize=15, color="#1f77b4", alpha=0.18,
             rotation=15, weight="bold", zorder=100)


# ---------------------------------------------------------------- A: TTFT breakdown, stacked horizontal bars, two panels (k=20, k=200)
def mock_a():
    R.setup()
    fig, axes = plt.subplots(1, 2, figsize=(R.DCOL, 1.75), gridspec_kw=dict(wspace=0.08))
    for ax, k in zip(axes, (20, 200)):
        R.clean(ax); ax.grid(False); ax.grid(axis="x", color="#e8e8e8", lw=0.6)
        for i, arm in enumerate(ARMS):
            left = 0
            for (name, color, _), v in zip(SEG, TTFT[k][arm]):
                ax.barh(i, v, left=left, color=color, height=0.62, edgecolor="white", lw=0.4, zorder=2)
                left += v
            ax.text(left + (0.01 * max(sum(x) for x in TTFT[k].values())), i, f"{left:,.0f} ms  ·  {TPOT[k][arm]:.0f} ms/tok",
                    va="center", fontsize=6.5, color="#444444")
        ax.set_yticks(range(len(ARMS))); ax.invert_yaxis()
        ax.set_yticklabels(ARMS if ax is axes[0] else [])
        ax.set_xlim(0, 1.45 * max(sum(x) for x in TTFT[k].values()))
        ax.set_xlabel("Time to first token (ms)")
        ax.set_title(f"({'ab'[k == 200]}) {k} passages ({'2.4k' if k == 20 else '24k'} text tokens)")
    fig.legend(handles=[Patch(color=c, label=n) for n, c, _ in SEG], loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.12),
               frameon=False, fontsize=R.SMALL)
    stamp(fig); R.save(fig, "mock_A_ttft_breakdown")


# ---------------------------------------------------------------- B: TTFT vs context size, log-log, single column
def mock_b():
    R.setup()
    ks = [20, 50, 100, 200, 400]
    scale = lambda v20, v200, k: v20 * (k / 20) ** (__import__("math").log(v200 / v20) / __import__("math").log(10))
    fig, ax = plt.subplots(figsize=(R.COL, 2.2))
    R.clean(ax); ax.set_xscale("log"); ax.set_yscale("log")
    styles = {"Text": dict(marker="s", ls="-"), "AC Input Emb.": dict(marker="o", ls="-"), "AC Hidden Res.": dict(marker="o", ls="--"),
              "AC All-Layer KVs": dict(marker="D", ls="-"), "AC KVs (cached)": dict(marker="D", ls=":", mfc="white")}
    for arm in ARMS:
        y = [scale(sum(TTFT[20][arm]), sum(TTFT[200][arm]), k) for k in ks]
        ax.plot([k * 120 for k in ks], y, color=COLORS[arm], ms=3.2, lw=1.2, label=arm, **styles[arm])
    ax.set_xlabel("Context (text tokens, log)"); ax.set_ylabel("Time to first token (ms, log)")
    ax.legend(fontsize=6.3, loc="upper left", handlelength=2.2)
    stamp(fig); R.save(fig, "mock_B_ttft_scaling")


# ---------------------------------------------------------------- C: where time goes, 100%-stacked incl. decode of 32 tokens
def mock_c():
    R.setup()
    fig, ax = plt.subplots(figsize=(R.COL, 1.9))
    R.clean(ax); ax.grid(False)
    segs = SEG + [("Decode (32 tok)", "#d9d9d9", None)]
    for i, arm in enumerate(ARMS):
        vals = list(TTFT[200][arm]) + [32 * TPOT[200][arm]]
        tot, left = sum(vals), 0
        for (name, color, _), v in zip(segs, vals):
            ax.barh(i, 100 * v / tot, left=left, color=color, height=0.62, edgecolor="white", lw=0.4)
            left += 100 * v / tot
        ax.text(101, i, f"{tot / 1000:.1f} s", va="center", fontsize=6.5)
    ax.set_yticks(range(len(ARMS))); ax.set_yticklabels(ARMS); ax.invert_yaxis(); ax.set_xlim(0, 112)
    ax.set_xlabel("Share of request time (%), 200 passages")
    ax.legend(handles=[Patch(color=c, label=n) for n, c, _ in segs], loc="upper center", ncol=2, bbox_to_anchor=(0.45, 1.42),
              frameon=False, fontsize=6.5)
    stamp(fig); R.save(fig, "mock_C_time_share")


# ---------------------------------------------------------------- D: memory (computed): reader cache per request + stored AC bytes
def mock_d():
    R.setup()
    kv_row = 2 * 36 * 8 * 128 * 2
    rows_text, rows_ac = 24_000, 2_400                       # 200 passages, ratio 1/10
    reader = {"Text": rows_text * kv_row, "AC (any mode)": rows_ac * kv_row}
    store = {"Text (UTF-8)": 24_000 * 4, "Input Emb.": rows_ac * 2560 * 2, "Hidden Res.": rows_ac * 9 * 2560 * 2,
             "All-Layer KVs": rows_ac * kv_row}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(R.COL, 1.8), gridspec_kw=dict(width_ratios=[1, 2], wspace=0.55))
    for ax, d, title, cols in ((a1, reader, "(a) Reader KV cache", [R.G_MID, R.OURS]),
                               (a2, store, "(b) Stored context", [R.G_LIGHT, R.OURS_LIGHT, R.OURS, R.OURS_DARK])):
        R.clean(ax)
        ax.bar(range(len(d)), [v / 2**20 for v in d.values()], color=cols, width=0.62, zorder=2)
        for x, v in enumerate(d.values()):
            ax.text(x, v / 2**20 * 1.15, f"{v / 2**20:,.0f}" if v >= 2**20 else f"{v / 2**20:.2f}", ha="center", fontsize=6.3)
        ax.set_xticks(range(len(d))); ax.set_xticklabels(list(d), rotation=30, ha="right", fontsize=6.5)
        ax.set_yscale("log"); ax.set_title(title, fontsize=7.5)
    a1.set_ylabel("MiB per request (200 passages)")
    R.save(fig, "mock_D_memory_computed")


if __name__ == "__main__":
    mock_a(); mock_b(); mock_c(); mock_d()
