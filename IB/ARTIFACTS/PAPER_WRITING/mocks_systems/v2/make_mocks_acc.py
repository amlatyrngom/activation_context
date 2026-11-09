"""Accuracy-in-the-figure variants (user, Oct 27 ~17:15Z: make the select-vs-AC trade-off read at a glance).
Placeholder timings as make_mocks_v2.py. Retrieval accuracies are measured (eval500, k=200); the ablation accuracies are PLACEHOLDERS
(text 53.0 = Fig. 7's line; select 43.5 strict, pending lenient rescoring; AC values = old v3 runs; 0.6B AC = not trained yet).
Run: uv run --no-project --python 3.13 --with matplotlib python make_mocks_acc.py
  A  accuracy as a right-hand column (AC's in vermilion bold)
  B  paired panels: time (left) | accuracy (right), shared rows
  C  accuracy badges at the end of each bar
  D  (user's pick, 17:10Z) C's placement with A's text style: the accuracy right after each bar's time note, plain text, AC's in bold vermilion
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_mocks_v2 as M  # noqa: E402
R = M.R

ABL_ACC = {("Text, 20 passages", None): 53.0, ("AC Input Emb.", "0.6B encoder"): 63.0, ("Select → text", "0.6B encoder"): 38.0,
           ("AC Input Emb.", "4B encoder"): 66.0, ("AC Hidden Res.", "4B encoder"): 68.5, ("AC All-Layer KVs", "4B encoder"): 69.0,
           ("Select → text", "4B encoder"): 43.5, ("AC All-Layer KVs", "Precomputed"): 69.0}


def is_ac(label):
    return label.startswith("AC")


def rows_retrieval():
    return [(lab, st, tpot, acc, None) for lab, st, tpot, acc in M.RETR]


def rows_ablation():
    out = []
    for group, members in M.ABL:
        if group == "Precomputed":
            continue
        for lab, st, tpot in members:
            out.append((lab, st, tpot, ABL_ACC[(lab, group)], group))
    return out


def layout(rows):
    """y positions with a group header slot before each new group."""
    ys, heads, y, last = [], [], 0.0, None
    for lab, st, tpot, acc, group in rows:
        if group != last and group is not None:
            if last is not None or ys:
                y += 0.3
            heads.append((y, group)); y += 0.85
        ys.append(y); y += 0.85 if group else 1.0
        last = group
    return ys, heads, y


def time_axis(ax, rows, ys, heads, xmax, note_acc=False, badge=False, fs=6, plain=False):
    for (lab, st, tpot, acc, group), y in zip(rows, ys):
        end = M.bar(ax, y, st, 0.62)
        ax.text(end + xmax * 0.012, y, f"{sum(st.values()):,.0f} ms · {tpot:.0f} ms/tok" if tpot >= 20 else f"{sum(st.values()):,.0f} ms · {tpot:.1f} ms/tok",
                va="center", fontsize=fs, color="#333333")
        if plain:
            note = ax.texts[-1]
            ax.annotate(f"  {acc:.1f}% acc.", xycoords=note, xy=(1, 0.5), va="center", ha="left", fontsize=fs + 0.5,
                        color=R.OURS if is_ac(lab) else "#777777", fontweight="bold" if is_ac(lab) else "normal")
        if badge:
            x = end + xmax * 0.012 + xmax * (0.30 if tpot >= 20 else 0.33)
            col = R.OURS if is_ac(lab) else "#777777"
            ax.text(x, y, f"{acc:.1f}%", va="center", ha="left", fontsize=fs + 0.5, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.35", fc=col, ec="none"))
    for y, g in heads:
        ax.text(-0.02, y, g, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=R.SMALL, fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(ys[-1] + 0.6, -0.6)
    M.axis(ax, xmax)


def variant_A(name, rows, xmax, h):
    fig, ax = plt.subplots(figsize=(R.COL, h))
    ys, heads, _ = layout(rows)
    time_axis(ax, rows, ys, heads, xmax)
    ax.text(1.04, -0.75, "Acc.", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=R.SMALL, fontweight="bold")
    for (lab, *_r), y in zip(rows, ys):
        acc = _r[2]
        ax.text(1.04, y, f"{acc:.1f}%", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=R.SMALL,
                color=R.OURS if is_ac(lab) else "#555555", fontweight="bold" if is_ac(lab) else "normal")
    M.legend(fig, {k for r in rows for k in r[1]}, y=0.98)
    R.save(fig, name)


def variant_B(name, rows, xmax, h, acc_lo=0, acc_hi=80):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(R.COL, h), sharey=True, gridspec_kw=dict(width_ratios=[2.6, 1], wspace=0.08))
    ys, heads, _ = layout(rows)
    time_axis(ax, rows, ys, heads, xmax)
    R.clean(bx); bx.grid(False); bx.grid(axis="x", color="#e8e8e8", lw=0.6, zorder=0)
    for (lab, st, tpot, acc, group), y in zip(rows, ys):
        bx.barh(y, acc, height=0.62, color=R.OURS if is_ac(lab) else R.G_MID, zorder=2)
        bx.text(acc + 1.5, y, f"{acc:.1f}", va="center", fontsize=6, color=R.OURS if is_ac(lab) else "#333333",
                fontweight="bold" if is_ac(lab) else "normal")
    bx.set_xlim(acc_lo, acc_hi); bx.set_xlabel("Accuracy (%)"); bx.tick_params(axis="y", length=0, labelleft=False)
    bx.spines["left"].set_visible(False)
    M.legend(fig, {k for r in rows for k in r[1]}, y=0.98)
    R.save(fig, name)


def variant_C(name, rows, xmax, h):
    fig, ax = plt.subplots(figsize=(R.COL, h))
    ys, heads, _ = layout(rows)
    time_axis(ax, rows, ys, heads, xmax, badge=True)
    M.legend(fig, {k for r in rows for k in r[1]}, y=0.98)
    R.save(fig, name)


def variant_D(name, rows, xmax, h):
    fig, ax = plt.subplots(figsize=(R.COL, h))
    ys, heads, _ = layout(rows)
    time_axis(ax, rows, ys, heads, xmax, plain=True)
    M.legend(fig, {k for r in rows for k in r[1]}, y=0.98)
    R.save(fig, name)


if __name__ == "__main__":
    R.setup()
    r, a = rows_retrieval(), rows_ablation()
    variant_A("accA_retrieval", r, 1900, 1.05); variant_A("accA_ablation", a, 240, 1.75)
    variant_B("accB_retrieval", r, 2000, 1.05); variant_B("accB_ablation", a, 265, 1.75)
    variant_C("accC_retrieval", r, 2000, 1.05); variant_C("accC_ablation", a, 255, 1.75)
    variant_D("accD_retrieval", r, 2000, 1.05); variant_D("accD_ablation", a, 265, 1.75)
