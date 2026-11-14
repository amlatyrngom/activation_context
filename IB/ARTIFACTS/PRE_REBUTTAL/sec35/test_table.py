"""§3.1 reported numbers on the HELD-OUT test1000 (2026-11-14): main row (frozen-reader KL stage 2), untuned text, text + LoRA (reference),
from the per-question outputs of jobs/rq_test_orcd.sh (fetched by scoreboard_fetch.sh into outs/orcd_rq_eval_test_<what>{,_20i}).
Same scorer and statistics as scoreboard.py (rq_common.score; paired bootstrap 95% CI, exact McNemar p), all paired on the same 1,000
test questions. Writes TEST.md and test.json (the figure reads test.json with --split test).
    python3 -I IB/ARTIFACTS/PRE_REBUTTAL/sec35/test_table.py
"""
import json, math, random, sys, tarfile, tempfile
from pathlib import Path

ROOT = Path("/home/ubuntu/activation_context")
OUTS = ROOT / "IB/TMP/PRE_REBUTTAL/scoreboard/outs"
DATA = ROOT / "IB/TMP/PRE_REBUTTAL/data"
CODE = ROOT / "IB/TMP/PRE_REBUTTAL/code/sec35_code_v6.tar.gz"
HERE = ROOT / "IB/ARTIFACTS/PRE_REBUTTAL/sec35"

_tmp = tempfile.mkdtemp()
with tarfile.open(CODE) as t:
    t.extractall(_tmp, members=[m for m in t.getmembers() if m.name.endswith("rq_kit/rq_common.py")], filter="data")
sys.path.insert(0, str(next(Path(_tmp).rglob("rq_common.py")).parent))
from rq_common import score  # noqa: E402


def rows(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


Q20 = {r["id"]: r for r in rows(DATA / "hotpot_tasks_v2/test1000.jsonl")}
QS = {"20I": Q20, "100I": {r["id"]: r for r in rows(DATA / "hotpot_tasks_kI_v2/test1000_100I.jsonl")},
      "200I": {r["id"]: r for r in rows(DATA / "hotpot_tasks_kI_v2/test1000_200I.jsonl")}}
QB = {r["id"]: r for r in rows(DATA / "hotpot_tasks_kI_v2/test1000.jsonl")}
for k in (20, 100, 200):
    QS[f"bm25_k{k}"] = QB
for s, q in QS.items():
    assert set(q) == set(Q20), f"{s}: question ids differ from 20I test1000"
    assert all(str(q[i]["answer"]) == str(Q20[i]["answer"]) for i in q), f"{s}: answers differ"

# (label, eval-dir stem, output-name prefix)
SYSTEMS = [("Text, untuned", "text_base", "text_base"),
           ("AC KL stage 2, frozen reader (MAIN)", "ac", "T_res_rq9fr_kI_KL_u02400"),
           ("Text + LoRA (reference)", "text_lora", "text_X_rq9_text_20I_u05600")]
FILES = {"20I": ("_20i", "test1000.k20"), "100I": ("", "test1000_100I.k100"), "200I": ("", "test1000_200I.k200"),
         "bm25_k20": ("", "test1000.k20"), "bm25_k100": ("", "test1000.k100"), "bm25_k200": ("", "test1000.k200")}
SETTINGS = list(FILES)
checks = []


def vec(stem, prefix, setting):
    sub, tail = FILES[setting]
    p = OUTS / f"orcd_rq_eval_test_{stem}{sub}" / f"{prefix}.{tail}.jsonl"
    s = Path(str(p)[:-len(".jsonl")] + ".summary.json")
    if not s.exists():
        return None
    q = QS[setting]
    v = {r["id"]: score(r.get("pred"), q[r["id"]])["acc"] for r in rows(p) if r["id"] in q}
    assert len(v) == 1000, f"{p}: {len(v)} rows"
    want, got = json.loads(s.read_text()).get("acc"), 100 * sum(v.values()) / len(v)
    if want is not None and abs(got - want) > 0.05:
        checks.append(f"{p.name}: rescored {got:.1f} vs summary {want}")
    return v


def paired(a, b, n_boot=2000, seed=0):
    ids = sorted(a); d = [a[i] - b[i] for i in ids]; delta = 100 * sum(d) / len(d); rnd = random.Random(seed)
    boots = sorted(100 * sum(d[rnd.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    nb, nc = sum(1 for x in d if x > 0), sum(1 for x in d if x < 0); n, m = nb + nc, min(nb, nc)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0
    return delta, lo, hi, p


V = {stem: {s: vec(stem, pre, s) for s in SETTINGS} for _, stem, pre in SYSTEMS}
acc = lambda v: 100 * sum(v.values()) / len(v)
out = ["# §3.1 on the held-out test1000 (2B encoder -> 9B reader, HotpotQA)\n",
       "Paired on the same 1,000 test questions: delta [95% bootstrap CI], exact McNemar p; **bold** = p < 0.05.\n",
       "| System | 20I | 100I | 200I | plain k=20 | plain k=100 | plain k=200 |", "|---|---|---|---|---|---|---|"]
for label, stem, _ in SYSTEMS:
    out.append(f"| {label} | " + " | ".join("–" if V[stem][s] is None else f"{acc(V[stem][s]):.1f}" for s in SETTINGS) + " |")


def cell(a, b):
    if a is None or b is None:
        return "–"
    d, lo, hi, p = paired(a, b); st = "**" if p < 0.05 else ""
    return f"{st}{d:+.1f}{st} [{lo:+.1f}, {hi:+.1f}] " + ("p<0.001" if p < 0.001 else f"p={p:.3f}")


out += ["\n## Main row − untuned text, same setting\n", "| 20I | 100I | 200I | plain k=20 | plain k=100 | plain k=200 |", "|---|---|---|---|---|---|",
        "| " + " | ".join(cell(V["ac"][s], V["text_base"][s]) for s in SETTINGS) + " |",
        "\n## Main row at k=100 / 200 − text at plain k=20\n", "| AC k | vs untuned k=20 | vs text + LoRA k=20 |", "|---|---|---|"]
for kk in ("bm25_k100", "bm25_k200"):
    out.append(f"| {kk[6:]} | {cell(V['ac'][kk], V['text_base']['bm25_k20'])} | {cell(V['ac'][kk], V['text_lora']['bm25_k20'])} |")
out.append("\n" + ("**Rescore mismatches:** " + "; ".join(checks) if checks else "All outputs re-scored to their summary accuracy (±0.05)."))
(HERE / "TEST.md").write_text("\n".join(out) + "\n")
data = {"systems": {stem: {"label": label, **{s: {"acc": round(acc(V[stem][s]), 2), "n": 1000} for s in SETTINGS if V[stem][s] is not None}}
                    for label, stem, _ in SYSTEMS}}
(HERE / "test.json").write_text(json.dumps(data, indent=1) + "\n")
print("\n".join(out))
