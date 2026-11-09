"""Systems figure mocks v2 (user, Oct 27 ~17:00Z): one decode baseline only (select -> text), the retrieval panel moved to the retrieval
section, no "Arm G", no cached arm in the retrieval panel, the trained small encoder (0.6B, Input Embeddings) instead of a size sweep,
and no broken axis (dropping decode-summarize removes the only long bars).
NUMBERS ARE PLACEHOLDERS read off IB/TMP/PAPER_SYSTEMS/fig_ttft_baselines.png at 16:50Z (stage splits approximate; the 9B panel is
preliminary). The final figure is built from results.json by the timing agent's script with this styling.
Run: uv run --no-project --python 3.13 --with matplotlib python make_mocks_v2.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mocks_r2"))
import make_r2 as R  # noqa: E402

R.OUT = Path(__file__).parent
# stage: (legend name, color, hatch)
ST = {"encode": ("Encode (AC)", R.OURS_LIGHT, None), "write": ("AC write", R.OURS_DARK, None), "load": ("Load precomputed AC", R.OURS_MID, None),
      "sprefill": ("Selector prefill", R.G_LIGHT, None), "sdecode": ("Selector decode", R.G_DARK, "////"), "prefill": ("Reader prefill", R.G_MID, None)}

# (label, {stage: ms}, tpot, accuracy or None)
# 17:20Z: the 16:50 image's 9B and 0.6B/1.7B rows were the timing agent's SYNTHETIC layout data. The 9B rows below are its n=2 k=200
# smoke (out/v3/smoke9b.jsonl, HF eager decode ~31 ms/tok on every arm; select forced to 21 tokens per call); the graph-engine rerun replaces them.
RETR = [("Text, 200 passages", {"prefill": 1242}, 31.0, 52.7),
        ("Select → text, 2B", {"sprefill": 442, "sdecode": 475, "prefill": 98}, 31.0, 41.7),
        ("AC, 2B encoder", {"encode": 522, "write": 0.6, "prefill": 117}, 31.0, 54.1)]
# groups for the ablation panel
ABL = [(None, [("Text, 20 passages", {"prefill": 80.2}, 10.7)]),
       ("0.6B encoder", [("AC Input Emb.", {"encode": 26.6, "write": 0.1, "prefill": 22.3}, 10.1),   # composed: measured 0.6B encode + 4B-row prefill
                         ("Select → text", {"sprefill": 24, "sdecode": 27, "prefill": 20}, 10.1)]),   # GUESS (not measured; job 29)
       ("4B encoder", [("AC Input Emb.", {"encode": 92.2, "write": 0.1, "prefill": 22.3}, 10.1),
                       ("AC Hidden Res.", {"encode": 94.5, "write": 0.5, "prefill": 22.0}, 10.1),
                       ("AC All-Layer KVs", {"encode": 87.7, "write": 2.1, "prefill": 23.6}, 10.1),
                       ("Select → text", {"sprefill": 78.7, "sdecode": 85.2, "prefill": 21.4}, 10.1)]),
       ("Precomputed", [("AC All-Layer KVs", {"load": 0.8, "write": 1.5, "prefill": 23.8}, 10.2)])]
ORDER = ["load", "encode", "write", "sprefill", "sdecode", "prefill"]


def bar(ax, y, stages, h):
    left = 0.0
    for key in ORDER:
        v = stages.get(key, 0)
        if v > 0:
            name, color, hatch = ST[key]
            ax.barh(y, v, left=left, height=h, color=color, hatch=hatch, edgecolor="white", lw=0.5, zorder=2)
            left += v
    return left


def axis(ax, xmax, xlabel=True):
    R.clean(ax); ax.grid(False); ax.grid(axis="x", color="#e8e8e8", lw=0.6, zorder=0)
    ax.set_xlim(0, xmax); ax.tick_params(axis="y", length=0)
    if xlabel:
        ax.set_xlabel("Time to first token (ms), one request")


def legend(fig, keys, y=1.02, ncol=3):
    fig.legend(handles=[Patch(facecolor=ST[k][1], hatch=ST[k][2], edgecolor="white", label=ST[k][0]) for k in ORDER if k in keys],
               loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, fontsize=R.SMALL, handlelength=1.6, columnspacing=1.0)


def retrieval(name, with_acc):
    fig, ax = plt.subplots(figsize=(R.COL, 1.05))
    for i, (label, st, tpot, acc) in enumerate(RETR):
        end = bar(ax, i, st, 0.62)
        note = f"{sum(st.values()):,.0f} ms · {tpot:.0f} ms/tok" + (f" · {acc:.1f}% acc." if with_acc else "")
        ax.text(end + 25, i, note, va="center", fontsize=6, color="#333333")
    ax.set_yticks(range(len(RETR))); ax.set_yticklabels([r[0] for r in RETR]); ax.set_ylim(len(RETR) - 0.5, -0.5)
    axis(ax, 2000)
    legend(fig, {"encode", "write", "sprefill", "sdecode", "prefill"})
    R.save(fig, name)


def ablation(name, sizes=("0.6B encoder", "4B encoder"), precomputed=True):
    groups = [g for g in ABL if g[0] is None or g[0] in sizes or (g[0] == "Precomputed" and precomputed)]
    n = sum(len(m) for _, m in groups) + sum(g is not None for g, _ in groups)
    fig, ax = plt.subplots(figsize=(R.COL, 0.35 + 0.15 * n))
    y, ticks, labels = 0.0, [], []
    for group, members in groups:
        if group:
            ax.text(-0.02, y, group, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=R.SMALL, fontweight="bold")
            y += 0.85
        for label, st, tpot in members:
            end = bar(ax, y, st, 0.62)
            ax.text(end + 3, y, f"{sum(st.values()):,.0f} ms · {tpot:.1f} ms/tok", va="center", fontsize=6, color="#333333")
            ticks.append(y); labels.append(label); y += 0.85
        y += 0.3
    ax.set_yticks(ticks); ax.set_yticklabels(labels); ax.set_ylim(y - 0.3, -0.6)
    axis(ax, 255)
    legend(fig, set(ORDER), y=0.93, ncol=3)
    R.save(fig, name)


if __name__ == "__main__":
    R.setup()
    retrieval("mockR1_retrieval", with_acc=False)
    retrieval("mockR2_retrieval_acc", with_acc=True)
    ablation("mockS1_ablation_grouped")
    ablation("mockS2_ablation_small_only", sizes=("0.6B encoder",), precomputed=False)
