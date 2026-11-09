"""Section 4.5 figures from the v4 runs (Oct 28). Qwen3-4B-Instruct-2507 reader (+ LoRA r128 unless frozen); encoder = 4B copy with
its own LoRA (or a Qwen3 0.6B / 1.7B with the linear-skip row head). R = FineWeb-Edu reconstruction, 2,000 updates; S = HotpotQA task
from scratch, T = task from R's final pair; 4,000 updates x 8 each. Gold documents among the top-20 BM25 (ideal recall@20). Dev200.
  fig7_<v>     training recipe: (a) R held NLL, (b) task held NLL, (c) dev200 accuracy; T solid, S dashed
  abl_<v>      ablations: (a) R held NLL with the reader trained vs frozen, (b) final accuracy trained vs frozen, (c) encoder size
Variants differ only in how the text + reader-LoRA baseline (T_text_lora: same 32,000 items, no no-context penalty) is shown.
Text references (untuned 4B reading the same tasks as text, greedy, lenient parse): top-20 53.0, gold 67.5, no context 19.0.
Data: data/<run>/ (small files from S3 sec45_20261026/runs/<run>/). Style from mocks_r2/make_r2.py.
Run: uv run --no-project --python 3.13 --with matplotlib --with numpy python make_sec45_v4.py
"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mocks_r2"))
from make_r2 import setup, clean, OURS, OURS_LIGHT, OURS_DARK, OURS_MID, COL, DCOL, SMALL, TXT_HALO, G_MID  # noqa: E402

DATA, OUT = HERE / "data", HERE / "figs"
MODES = [("in", "Input Embeddings", OURS_LIGHT, "o"), ("res", "Hidden Residuals", OURS, "s"), ("kv", "All-Layer KVs", OURS_DARK, "D")]
DASH, DOT = (0, (4, 2)), (0, (1, 1.4))
TEXT, TEXT_LORA = 53.0, "#555555"
LS_TEXT = (0, (3, 1.2, 1, 1.2, 1, 1.2))
STEPS = 4000


def held(run):
    f = next(p for p in (DATA / run / "sync" / f"{run}.held.jsonl", DATA / run / f"{run}.held.jsonl") if p.exists())
    rows = [json.loads(line) for line in open(f)]
    return [r["update"] for r in rows], [r["held_nll"] for r in rows]


def acc(run):
    if run == "T_text_lora":
        d = json.loads((DATA / run / "T_text_lora.dev200.scores.json").read_text())
        return [int(k[1:]) for k in sorted(d)], [d[k]["acc_lenient"] for k in sorted(d)]
    pts = []
    for f in (DATA / run / "eval").glob("*.summary.json"):
        u = f.name.removesuffix(".summary.json")
        pts.append((STEPS if u == "final" else int(u[1:]), json.loads(f.read_text())["acc"]))
    return [list(t) for t in zip(*sorted(pts))]


def text_ref(ax, y, label=None, ls=LS_TEXT, color="#555555", below=False):
    ax.axhline(y, color=color, lw=0.8, ls=ls, zorder=1)
    if label:
            ax.text(ax.get_xlim()[1], y - 1.2 if below else y + 0.9, label, ha="right", va="top" if below else "bottom", fontsize=SMALL - 0.5, style="italic", color=color,
                path_effects=TXT_HALO)


def fig7(variant):
    """a: text + reader-LoRA as a grey curve in (b) and (c); b: as a final-value line in (c); c: not shown (untuned text only)."""
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 1.9), gridspec_kw=dict(wspace=0.38))
    ax = axes[0]
    for m, _, c, mk in MODES:
        x, y = held(f"R_{m}_v4"); ax.plot(x, y, color=c, marker=mk, ms=2.5, markevery=2)
    ax.set_yscale("log"); ax.set_title("(a) Reconstruction")
    ax.set_xlabel("Reconstruction updates"); ax.set_ylabel("Held-out NLL")
    ax = axes[1]
    for m, _, c, mk in MODES:
        x, y = held(f"T_{m}_v4"); ax.plot(x, y, color=c, marker=mk, ms=2.5, markevery=4)
        x, y = held(f"S_{m}_v4"); ax.plot(x, y, color=c, ls=DASH, lw=1.1)
    if variant == "a":
        x, y = held("T_text_lora"); ax.plot(x, y, color=TEXT_LORA, marker="x", ms=3, markevery=4, lw=1.1)
    ax.set_yscale("log"); ax.set_title("(b) Task loss (HotpotQA)")
    ax.set_xlabel("Task updates"); ax.set_ylabel("Held-out NLL")
    ax = axes[2]
    for m, _, c, mk in MODES:
        x, y = acc(f"T_{m}_v4"); ax.plot(x, y, color=c, marker=mk, ms=2.5)
        x, y = acc(f"S_{m}_v4"); ax.plot(x, y, color=c, ls=DASH, lw=1.1, marker=mk, ms=2, mfc="white")
    ax.set_xlim(-150, STEPS + 150); ax.set_ylim(0, 80)
    if variant == "a":
        x, y = acc("T_text_lora"); ax.plot(x, y, color=TEXT_LORA, marker="x", ms=3.5, lw=1.1, zorder=3)
        text_ref(ax, TEXT, "Text (untuned)")
    elif variant == "b":
        text_ref(ax, acc("T_text_lora")[1][-1], "Text + reader LoRA", ls=(0, (5, 1.5)))
        text_ref(ax, TEXT, "Text (untuned)")
    else:
        text_ref(ax, TEXT, "Text, perfect top-20")
    ax.set_title("(c) Task accuracy"); ax.set_xlabel("Task updates"); ax.set_ylabel("Accuracy (%)")
    for ax in axes:
        clean(ax)
    handles = [Line2D([], [], color=c, marker=mk, ms=3, label=n) for _, n, c, mk in MODES]
    handles += [Line2D([], [], color="#666666", label="from reconstruction"), Line2D([], [], color="#666666", ls=DASH, label="from scratch")]
    if variant == "a":
        handles.append(Line2D([], [], color=TEXT_LORA, marker="x", ms=3, label="Text + reader LoRA"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.10), frameon=False, fontsize=SMALL,
               handlelength=2.0, columnspacing=1.0)
    save(fig, f"fig7_{variant}")


def ablations(variant):
    """a: (b) bars from reconstruction only, (c) encoder-size curves; b: (b) adds the from-scratch bars, (c) final-accuracy bars."""
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 1.9), gridspec_kw=dict(wspace=0.38, width_ratios=[1, 1.15 if variant == "b" else 1, 1]))
    ax = axes[0]
    for m, _, c, mk in MODES:
        x, y = held(f"R_{m}_v4"); ax.plot(x, y, color=c, marker=mk, ms=2.5, markevery=2)
        x, y = held(f"R_{m}_v4fr"); ax.plot(x, y, color=c, ls=DOT, lw=1.3, marker=mk, ms=2.5, mfc="white", markevery=2)
    ax.set_yscale("log"); ax.set_title("(a) Reconstruction")
    ax.set_xlabel("Reconstruction updates"); ax.set_ylabel("Held-out NLL")
    ax = axes[1]
    arms = [("T", "v4", ""), ("T", "v4fr", "////")] + ([("S", "v4", ".."), ("S", "v4fr", "xxxx")] if variant == "b" else [])
    w = 0.8 / len(arms)
    for i, (m, _, c, _) in enumerate(MODES):
        for j, (kind, tag, hatch) in enumerate(arms):
            v = acc(f"{kind}_{m}_{tag}")[1][-1]
            x = i + (j - (len(arms) - 1) / 2) * w
            ax.bar(x, v, w * 0.92, color="white" if hatch else c, edgecolor=c, hatch=hatch, lw=0.8, zorder=2)
            ax.text(x, v + 1, f"{v:.0f}" if variant == "b" else f"{v:.1f}", ha="center", va="bottom",
                    fontsize=SMALL - 1.5 if variant == "b" else SMALL - 1)
    ax.set_xticks(range(3), ["Input\nEmb.", "Hidden\nRes.", "All-Layer\nKVs"])
    ax.set_ylim(0, 82); ax.set_xlim(-0.55, 2.55)
    text_ref(ax, TEXT)
    ax.set_title("(b) Final accuracy"); ax.set_ylabel("Accuracy (%)")
    ax = axes[2]
    sizes = [("T_in_e06ls", "0.6B", "#FCBBA1", "v"), ("T_in_e17ls", "1.7B", "#FB6A4A", "^"), ("T_in_v4", "4B", "#A50F15", "o")]
    if variant == "a":
        for run, label, c, mk in sizes:
            x, y = acc(run); ax.plot(x, y, color=c, marker=mk, ms=2.5, label=f"{label} encoder")
        ax.set_xlim(-150, STEPS + 150); ax.set_xlabel("Task updates")
        ax.legend(loc="lower right", fontsize=SMALL - 0.5, handlelength=1.5, borderaxespad=0.2)
    else:
        for i, (run, label, c, _) in enumerate(sizes):
            v = acc(run)[1][-1]; ax.bar(i, v, 0.6, color=c, zorder=2)
            ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=SMALL - 1)
        ax.set_xticks(range(3), [f"{s[1]}" for s in sizes]); ax.set_xlabel("Encoder size (Input Embeddings)")
        ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 82)
    text_ref(ax, TEXT)
    ax.set_title("(c) Encoder size"); ax.set_ylabel("Accuracy (%)")
    for ax in axes:
        clean(ax)
    axes[1].grid(axis="x", visible=False)
    handles = [Line2D([], [], color=c, marker=mk, ms=3, label=n) for _, n, c, mk in MODES]
    handles += [(Line2D([], [], color="#666666"), Patch(facecolor="#666666", edgecolor="#666666")),
                (Line2D([], [], color="#666666", ls=DOT, lw=1.3), Patch(facecolor="white", edgecolor="#666666", hatch="////"))]
    labels = [h.get_label() for h in handles[:3]] + ["reader trained", "reader frozen"]
    if variant == "b":
        handles += [Patch(facecolor="white", edgecolor="#666666", hatch=h, label=l) for h, l in
                    (("..", "scratch"), ("xxxx", "scratch, frozen"))]
        labels += ["scratch", "scratch, frozen"]
    handles.append(Line2D([], [], color="#555555", lw=0.8, ls=LS_TEXT)); labels.append("Text (untuned)")
    fig.legend(handles=handles, labels=labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, loc="upper center",
               ncol=len(handles), bbox_to_anchor=(0.5, 1.10), frameon=False, fontsize=SMALL, handlelength=2.4, columnspacing=1.0)
    save(fig, f"abl_{variant}")


def unified(variant):
    """One figure (user, Oct 28: "one unified figure ... largely like B"); the task-loss panel is dropped for room.
    u4: (a) R trained vs frozen, (b) accuracy T/S + text lines, (c) final accuracy trained vs frozen (bars), (d) encoder size.
    u3: (a) R trained vs frozen, (b) accuracy T/S/T-frozen + text lines, (c) encoder size."""
    four = variant == "u4"
    fig, axes = plt.subplots(1, 4 if four else 3, figsize=(DCOL, 1.85), gridspec_kw=dict(wspace=0.42 if four else 0.38))
    lora = acc("T_text_lora")[1][-1]
    ax = axes[0]
    for m, _, c, mk in MODES:
        x, y = held(f"R_{m}_v4"); ax.plot(x, y, color=c, marker=mk, ms=2.5, markevery=2)
        x, y = held(f"R_{m}_v4fr"); ax.plot(x, y, color=c, ls=DOT, lw=1.3, marker=mk, ms=2.5, mfc="white", markevery=2)
    ax.set_yscale("log"); ax.set_title("(a) Reconstruction")
    ax.set_xlabel("Reconstruction updates"); ax.set_ylabel("Held-out NLL")
    ax = axes[1]
    for m, _, c, mk in MODES:
        x, y = acc(f"T_{m}_v4"); ax.plot(x, y, color=c, marker=mk, ms=2.5)
        x, y = acc(f"S_{m}_v4"); ax.plot(x, y, color=c, ls=DASH, lw=1.1, marker=mk, ms=2, mfc="white")
        if not four:
            x, y = acc(f"T_{m}_v4fr"); ax.plot(x, y, color=c, ls=DOT, lw=1.3, marker=mk, ms=2, mfc="white")
    ax.set_xlim(-150, STEPS + 150); ax.set_ylim(0, 80)
    text_ref(ax, lora, "Text + reader LoRA", ls=(0, (5, 1.5)))
    text_ref(ax, TEXT, "Text (untuned)", below=True)
    ax.set_title("(b) Task accuracy"); ax.set_xlabel("Task updates"); ax.set_ylabel("Accuracy (%)")
    if four:
        ax = axes[2]
        for i, (m, _, c, _) in enumerate(MODES):
            for j, (tag, hatch) in enumerate((("v4", ""), ("v4fr", "////"))):
                v = acc(f"T_{m}_{tag}")[1][-1]; x = i + (j - 0.5) * 0.4
                ax.bar(x, v, 0.37, color="white" if hatch else c, edgecolor=c, hatch=hatch, lw=0.8, zorder=2)
                ax.text(x, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=SMALL - 1)
        ax.set_xticks(range(3), ["Input", "Resid.", "KVs"]); ax.set_xlim(-0.55, 2.55); ax.set_ylim(0, 80)
        ax.grid(axis="x", visible=False)
        ax.set_title("(c) Frozen reader"); ax.set_ylabel("Final accuracy (%)")
    ax = axes[-1]
    sizes = [("T_in_e06ls", "0.6B", "#FCBBA1", "v"), ("T_in_e17ls", "1.7B", "#FB6A4A", "^"), ("T_in_v4", "4B", OURS_LIGHT, "o")]
    for run, label, c, mk in sizes:
        x, y = acc(run); ax.plot(x, y, color="#A50F15" if label == "4B" else c, marker=mk, ms=2.5, label=f"{label} enc.")
    ax.set_xlim(-150, STEPS + 150); ax.set_ylim(0, 80)
    text_ref(ax, lora, ls=(0, (5, 1.5))); text_ref(ax, TEXT)
    ax.legend(loc="lower right", fontsize=SMALL - 0.5, handlelength=1.4, borderaxespad=0.1, labelspacing=0.25)
    ax.set_title(f"({'d' if four else 'c'}) Encoder size"); ax.set_xlabel("Task updates"); ax.set_ylabel("Accuracy (%)")
    for ax in axes:
        clean(ax)
    handles = [Line2D([], [], color=c, marker=mk, ms=3) for _, _, c, mk in MODES]
    labels = [n for _, n, _, _ in MODES]
    handles += [Line2D([], [], color="#666666"), Line2D([], [], color="#666666", ls=DASH)]
    labels += ["from reconstruction", "from scratch"]
    frozen = Line2D([], [], color="#666666", ls=DOT, lw=1.3)
    handles.append((frozen, Patch(facecolor="white", edgecolor="#666666", hatch="////")) if four else frozen)
    labels.append("reader frozen")
    fig.legend(handles=handles, labels=labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, loc="upper center",
               ncol=len(handles), bbox_to_anchor=(0.5, 1.10), frameon=False, fontsize=SMALL, handlelength=2.2, columnspacing=1.0)
    save(fig, f"fig7_{variant}")


LS_FULL = dict(ls="-", lw=1.4)                                   # full recipe: reconstruction first, reader trained
LS_SCRATCH = dict(ls=(0, (5, 2.2)), lw=1.0, mfc="white")         # task from scratch
LS_FROZEN = dict(ls=(0, (1, 1.3)), lw=1.6, mfc="white")          # reader frozen
SIZES = [("T_in_e06ls", "0.6B", "#F9C2AE", "v"), ("T_in_e17ls", "1.7B", "#EB6B4F", "^"), ("T_in_v4", "4B", "#A50F15", "o")]
LORA_LS, LORA_C = (0, (6, 1.5, 1.5, 1.5)), "#555555"


def refs(ax):
    ax.axhline(acc("T_text_lora")[1][-1], color=LORA_C, lw=0.8, ls=LORA_LS, zorder=1)
    ax.axhline(TEXT, color="#999999", lw=0.8, ls=LS_TEXT, zorder=1)


def acc_axis(ax, title, ylabel=True):
    ax.set_xlim(-150, STEPS + 150); ax.set_ylim(0, 80); ax.set_xticks([0, 2000, 4000])
    ax.set_title(title); ax.set_xlabel("Training steps")
    if ylabel:
        ax.set_ylabel("Accuracy (%)")


def p_recipe(ax, ylabel=True):
    for m, _, c, mk in MODES:
        x, y = acc(f"T_{m}_v4"); ax.plot(x, y, color=c, marker=mk, ms=2.5, **LS_FULL)
        x, y = acc(f"S_{m}_v4"); ax.plot(x, y, color=c, marker=mk, ms=2.3, **LS_SCRATCH)
    refs(ax); acc_axis(ax, "Reconstruction first?", ylabel)
    style_key(ax, [("reconstruction", LS_FULL), ("scratch", LS_SCRATCH)], loc="lower right", anchor=(1.0, 0.0),
              direct=[(2300, 44, "from reconstruction", "center"), (2300, 29, "from scratch", "center")])


def p_frozen(ax, ylabel=True):
    for m, _, c, mk in MODES:
        x, y = acc(f"T_{m}_v4"); ax.plot(x, y, color=c, marker=mk, ms=2.5, **LS_FULL)
        x, y = acc(f"T_{m}_v4fr"); ax.plot(x, y, color=c, marker=mk, ms=2.3, **LS_FROZEN)
    refs(ax); acc_axis(ax, "Train the reader?", ylabel)
    style_key(ax, [("trained", LS_FULL), ("frozen", LS_FROZEN)], loc="lower right", anchor=(1.0, 0.0),
              direct=[(1900, 42, "reader trained", "right"), (1900, 28, "reader frozen", "right")])


KEY = {"style": "grey"}   # grey | black | tri (three write-mode colors per handle) | direct (labels on the curves, no key)


def style_key(ax, styles, loc, anchor, direct=()):
    """Per-panel key for the line styles; the color is the write mode, from the top legend."""
    k = KEY["style"]
    if k == "direct":
        for x, y, text, ha in direct:
            ax.text(x, y, text, ha=ha, va="center", fontsize=SMALL - 0.5, style="italic", color="#333333", path_effects=TXT_HALO)
        return
    if k == "tri":
        handles = [tuple(Line2D([], [], color=c, **st) for _, _, c, _ in MODES) for _, st in styles]
    else:
        color = "#666666" if k == "grey" else "#1a1a1a"
        handles = [Line2D([], [], color=color, marker="o", ms=2.5, **st) for _, st in styles]
    ax.legend(handles, [n for n, _ in styles], loc=loc, bbox_to_anchor=anchor, fontsize=SMALL - 1,
              handlelength=2.0 if k == "tri" else 1.8, handletextpad=0.5, borderaxespad=0.1, labelspacing=0.2, frameon=False,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)})


BLUES = {"0.6B": "#9ECAE1", "1.7B": "#4292C6", "4B": "#08519C"}
GREYS = {"0.6B": "#BDBDBD", "1.7B": "#7A7A7A", "4B": OURS_LIGHT}
# m0 = as before (red ramp, encode by write mode); m1 = one color (Input Emb.) + marker shapes, encode one steel blue;
# m2 = small encoders grey, 4B in the Input Emb. color, encode amber; m3 = blue ramp by encoder size in (c) AND in (d)'s encode segments
SCHEME = {"size": "ramp", "encode": "mode", "load": OURS_MID}
SCHEMES = {"m1": {"size": "mono", "encode": "#5B7FA6", "load": "#2F4B6E"},
           "m2": {"size": "grey", "encode": "#D9A441", "load": "#8C6414"},
           "m3": {"size": "blue", "encode": "size", "load": "#333333"}}


def p_size(ax, ylabel=True):
    for run, label, c, mk in SIZES:
        c = {"ramp": c, "mono": OURS_LIGHT, "grey": GREYS[label], "blue": BLUES[label]}[SCHEME["size"]]
        x, y = acc(run); ax.plot(x, y, color=c, marker=mk, ms=3 if SCHEME["size"] == "mono" else 2.5, lw=1.4, label=label,
                                 mfc="white" if SCHEME["size"] == "mono" and label != "4B" else c)
    refs(ax); acc_axis(ax, "Encoder size", ylabel)
    ax.legend(loc="lower right", fontsize=SMALL - 1, handlelength=1.6, borderaxespad=0.1, labelspacing=0.2)


def p_bars(ax):
    """Final accuracy per write mode: from scratch | reader frozen | full recipe."""
    arms = [("S", "v4", LS_SCRATCH, "...."), ("T", "v4fr", LS_FROZEN, "////"), ("T", "v4", LS_FULL, "")]
    w = 0.27
    for i, (m, _, c, _) in enumerate(MODES):
        for j, (kind, tag, _, hatch) in enumerate(arms):
            v = acc(f"{kind}_{m}_{tag}")[1][-1]; x = i + (j - 1) * w
            ax.bar(x, v, w * 0.9, color="white" if hatch else c, edgecolor=c, hatch=hatch, lw=0.8, zorder=2)
            ax.text(x, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=SMALL - 1.5)
    refs(ax)
    ax.set_xticks(range(3), ["Input", "Resid.", "KVs"]); ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, 80); ax.grid(axis="x", visible=False)
    ax.set_title("Training recipe"); ax.set_ylabel("Final accuracy (%)")


def p_ttft(ax, compact=False):
    """Time to first token, one request, 20 passages, 4B reader (IB/TMP/PAPER_SYSTEMS/results.json, fast engine). The encode segment takes
    the write mode's color. Text is the same with or without the reader LoRA (merged weights), hence "Text (+ LoRA)"."""
    sys.path.insert(0, "/workspace/IB/TMP/PAPER_SYSTEMS")
    import make_ttft_figs as T
    res = json.loads(Path("/workspace/IB/TMP/PAPER_SYSTEMS/results.json").read_text())
    enc = lambda s: f"encoder=Qwen/Qwen3-{s}|"
    E4 = enc("4B-Instruct-2507")
    color = dict((m, c) for m, _, c, _ in MODES)
    rows = [("Input, 0.6B enc.", "in", ("main|mode=in|", enc("0.6B"), "engine=fast", "pair=untrained"), "ac"),
            ("Input, 1.7B enc.", "in", ("main|mode=in|", enc("1.7B"), "engine=fast", "pair=untrained"), "ac"),
            ("Input, 4B enc.", "in", ("main|mode=in|", E4, "engine=fast", "pair=trained"), "ac"),
            ("Resid., 4B enc.", "res", ("main|mode=res|", E4, "engine=fast", "pair=trained"), "ac"),
            ("KVs, 4B enc.", "kv", ("main|mode=kv|", E4, "engine=fast", "pair=trained"), "ac"),
            ("KVs, precomputed", "kv", ("main|mode=kv|", E4, "engine=fast", "pair=trained"), "ac_cached"),
            ("Text (+ LoRA)", None, ("main|mode=in|", E4, "engine=fast", "pair=trained"), "text")]
    used = set()
    for i, (label, mode, parts, arm) in enumerate(rows):
        stages, ttft, _ = T.pick(res, parts, arm)
        left = 0.0
        for key in T.ORDER:
            v = stages.get(key, 0.0)
            if v > 0:
                size = (label.split(", ") + ["4B"])[1].split()[0].replace("precomputed", "4B")
                k2 = "encode" if key == "write" and SCHEME["encode"] != "mode" else key     # the < 1 ms AC write joins the encode
                c = (T.ST[key][1] if k2 not in ("encode", "load") else SCHEME["load"] if k2 == "load" else
                     color[mode] if SCHEME["encode"] == "mode" else BLUES[size] if SCHEME["encode"] == "size" else SCHEME["encode"])
                ax.barh(i, v, left=left, height=0.62, color=c, edgecolor="white", lw=0.5, zorder=2); left += v; used.add(key)
        ax.text(left + 3, i, f"{ttft:.0f} ms", va="center", fontsize=SMALL - 1,
                color=OURS if mode else "#333333", fontweight="bold" if mode else "normal")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, TTFT_XMAX if compact else 160); ax.tick_params(axis="y", length=0)
    clean(ax); ax.grid(False); ax.grid(axis="x", color="#e8e8e8", lw=0.6, zorder=0)
    ax.set_title("Time to first token"); ax.set_xlabel("ms, one request")
    handles, labels = [], []
    for key in [k for k in T.ORDER if k in used and k != "write"]:   # the AC write (< 1 ms) is invisible at this scale
        if key == "encode" and SCHEME["encode"] in ("mode", "size"):
            cs = [c for _, _, c, _ in MODES] if SCHEME["encode"] == "mode" else list(BLUES.values())
            handles.append(tuple(Patch(facecolor=c, edgecolor="white") for c in cs)); labels.append("Encode")
        elif key in ("encode", "load"):
            handles.append(Patch(facecolor=SCHEME[key], edgecolor="white")); labels.append("Encode" if key == "encode" else "Load")
        else:
            handles.append(Patch(facecolor=T.ST[key][1], edgecolor="white")); labels.append(T.ST[key][0].replace("Load precomputed AC", "Load"))
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc="upper right", fontsize=SMALL - 1.5,
              handlelength=1.6, borderaxespad=0.0, labelspacing=0.15)


def legend_v(fig, styles, y=1.17, ncol=4):
    h = [Line2D([], [], color=c, marker=mk, ms=3, lw=1.4) for _, _, c, mk in MODES]
    lab = [n for _, n, _, _ in MODES]
    for name, st in styles:
        h.append(Line2D([], [], color="#666666", marker="o", ms=2.5, **st)); lab.append(name)
    h += [Line2D([], [], color=LORA_C, lw=0.8, ls=LORA_LS), Line2D([], [], color="#999999", lw=0.8, ls=LS_TEXT)]
    lab += ["Text + reader LoRA", "Text (untuned)"]
    fig.legend(h, lab, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y), frameon=False, fontsize=SMALL, handlelength=2.6,
               columnspacing=1.2, labelspacing=0.3)


TTFT_XMAX = 175                                                   # (d) x range when compact; the compact paper version uses 225
V2_H = 1.85                                                       # v2 figure height (in); the paper's compact version uses 1.6


def variants(v, suffix=""):
    """v1: 3 accuracy panels (recipe | frozen reader | encoder size). v2: v1 + time to first token. v3: final-accuracy bars (recipe and
    frozen reader) | encoder size | time to first token. v4: 2 panels for one column: bars | encoder size."""
    styles = [("full recipe", LS_FULL), ("task from scratch", LS_SCRATCH), ("reader frozen", LS_FROZEN)]
    if v == "v1":
        fig, ax = plt.subplots(1, 3, figsize=(DCOL, 1.85), gridspec_kw=dict(wspace=0.32))
        p_recipe(ax[0]); p_frozen(ax[1], False); p_size(ax[2], False)
    elif v == "v2":
        fig, ax = plt.subplots(1, 4, figsize=(DCOL, V2_H), gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1, 1, 1.2]))
        p_recipe(ax[0]); p_frozen(ax[1], False); p_size(ax[2], False); p_ttft(ax[3], compact=True)
        fig.subplots_adjust(wspace=0.3); ax[3].set_position(ax[3].get_position().translated(0.09, 0))
    elif v == "v3":
        fig, ax = plt.subplots(1, 3, figsize=(DCOL, 1.85), gridspec_kw=dict(wspace=0.35, width_ratios=[1.1, 1, 1.25]))
        p_bars(ax[0]); p_size(ax[1]); p_ttft(ax[2])
        ax[2].set_position(ax[2].get_position().translated(0.06, 0))
        styles = [("full recipe", LS_FULL)]
    else:
        fig, ax = plt.subplots(1, 2, figsize=(COL, 1.75), gridspec_kw=dict(wspace=0.45))
        p_bars(ax[0]); p_size(ax[1])
        styles = [("full recipe", LS_FULL)]
    for i, a in enumerate(ax):
        a.set_title(f"({'abcd'[i]}) {a.get_title()}")
    if v in ("v3", "v4"):
        h = [Patch(facecolor=c, edgecolor=c, label=n) for _, n, c, _ in MODES]
        h += [Patch(facecolor="white", edgecolor="#666666", hatch=hh, label=l) for hh, l in (("....", "task from scratch"), ("////", "reader frozen"))]
        h += [Line2D([], [], color=LORA_C, lw=0.8, ls=LORA_LS, label="Text + reader LoRA"),
              Line2D([], [], color="#999999", lw=0.8, ls=LS_TEXT, label="Text (untuned)")]
        fig.legend(handles=h, loc="upper center", ncol=4 if v == "v3" else 3, bbox_to_anchor=(0.5, 1.17 if v == "v3" else 1.2), frameon=False,
                   fontsize=SMALL if v == "v3" else SMALL - 1, handlelength=2.0, columnspacing=1.0, labelspacing=0.3)
    elif v == "v2":
        legend_v(fig, [], y=1 + 0.185 / V2_H, ncol=5)                          # line styles: per-panel keys in (a) and (b)
    else:
        legend_v(fig, styles)
    save(fig, f"fig7_{v}{suffix}")


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    setup()
    for v in "abc":
        fig7(v)
    for v in "ab":
        ablations(v)
    for v in ("u4", "u3"):
        unified(v)
    for v in ("v1", "v2", "v3", "v4"):
        variants(v)
    for m, sch in SCHEMES.items():
        SCHEME.update(sch); variants("v2", suffix=f"_{m}")
    SCHEME.update(SCHEMES["m2"])
    for k, style in (("k1", "black"), ("k2", "tri"), ("k3", "direct")):
        KEY["style"] = style; variants("v2", suffix=f"_m2_{k}")
    KEY["style"] = "tri"                                          # user, Oct 28: k2 is the paper's key
    variants("v2", suffix="_paper")
    V2_H, TTFT_XMAX = 1.6, 225                                                    # Oct 28: shorter paper version (layout: Fig. 7 onto page 10)
    variants("v2", suffix="_paper_compact")
