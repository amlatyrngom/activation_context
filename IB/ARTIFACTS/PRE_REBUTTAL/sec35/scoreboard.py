"""§3.1 scoreboard (2026-11-13, user): AC (Qwen3.5-2B encoder -> Qwen3.5-9B reader) against text, dev1000, from per-question outputs.

  A. Ideal retrieval (golds injected: 20I, 100I at k=100, 200I at k=200): AC vs UNTUNED text (and text + LoRA) at the same setting.
  B. Plain BM25 (no injection, k = 20 / 100 / 200): AC at k = 100 / 200 vs untuned text and text + LoRA at k = 20 (and at 100 / 200).

Every output is re-scored here with rq_common.score (the scorer the evals use; checked against each file's summary acc), so all
comparisons are paired on the same 1,000 questions: delta, 95% paired-bootstrap CI, exact McNemar p. Stage-2 arms show their LATEST
scored checkpoint per setting. Inputs: IB/TMP/PRE_REBUTTAL/scoreboard/outs (scoreboard_fetch.sh). Output: markdown on stdout and in
IB/ARTIFACTS/PRE_REBUTTAL/sec35/SCOREBOARD.md.
    python3 -I IB/ARTIFACTS/PRE_REBUTTAL/sec35/scoreboard.py
"""
import glob, json, math, random, re, sys, tarfile, tempfile
from pathlib import Path

ROOT = Path("/home/ubuntu/activation_context")
OUTS = ROOT / "IB/TMP/PRE_REBUTTAL/scoreboard/outs"
DATA = ROOT / "IB/TMP/PRE_REBUTTAL/data"
CODE = ROOT / "IB/TMP/PRE_REBUTTAL/code/sec35_code_v6.tar.gz"
MD = ROOT / "IB/ARTIFACTS/PRE_REBUTTAL/sec35/SCOREBOARD.md"

# the evals' own scorer, from our code bundle
_tmp = tempfile.mkdtemp()
with tarfile.open(CODE) as t:
    t.extractall(_tmp, members=[m for m in t.getmembers() if m.name.endswith("rq_kit/rq_common.py")], filter="data")
sys.path.insert(0, str(next(Path(_tmp).rglob("rq_common.py")).parent))
from rq_common import score  # noqa: E402


def rows(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


Q20 = {r["id"]: r for r in rows(DATA / "hotpot_tasks_v2/dev1000.jsonl")}
QS = {"20I": Q20, "100I": {r["id"]: r for r in rows(DATA / "hotpot_tasks_kI_v2/dev1000_100I.jsonl")},
      "200I": {r["id"]: r for r in rows(DATA / "hotpot_tasks_kI_v2/dev1000_200I.jsonl")}}
QB = {r["id"]: r for r in rows(DATA / "hotpot_tasks_kI_v2/dev1000.jsonl")}
for k in (20, 100, 200):
    QS[f"bm25_k{k}"] = QB
for s, q in QS.items():
    assert set(q) == set(Q20), f"{s}: question ids differ from 20I dev1000"
    assert all(str(q[i]["answer"]) == str(Q20[i]["answer"]) for i in q), f"{s}: answers differ"

# ---- collect outputs: (system, step, setting) -> (path, source priority)
PRI = {"orcd_rq_eval_fin": 5, "orcd_rq_eval_fin20i": 5, "orcd_rq_eval_s2": 4, "orcd_rq_eval_kI_ac": 4, "orcd_rq_eval_kI_text": 4,
       "orcd_rq_eval20i": 4, "orcd_rq_eval_e4": 4, "s3_kI_eval_ac": 3, "s3_kI_eval_text": 3}
STAGE1_FINAL = {"T_res_rq9e4_e2": 5600, "T_res_rq9L_20I": 5600, "T_res_rq9fr_20I": 5600, "T_res_rq9L_20I_KL": 5600, "T_res_rq9fr_20I_KL": 5600}
STAGE2 = {"T_res_rq9L_kI": 2400, "T_res_rq9fr_kI": 2400, "T_res_rq9L_kI_KL": 2400, "T_res_rq9fr_kI_KL": 2400}
found = {}


def put(system, step, setting, path, pri):
    key = (system, step, setting)
    if key not in found or pri > found[key][1]:
        found[key] = (path, pri)


for p in glob.glob(str(OUTS / "*/*.jsonl")):
    src, name = Path(p).parent.name, Path(p).name
    if not Path(p[:-len(".jsonl")] + ".summary.json").exists():
        continue                                        # still being written (the summary is written last)
    if src.startswith(("s3_run_", "orcd_run_")):         # in-run evals: <uNNNNN|final>.jsonl
        run = src.split("_run_", 1)[1]
        m = re.fullmatch(r"(u(\d+)|final)\.jsonl", name)
        if not m:
            continue
        step = int(m.group(2)) if m.group(2) else STAGE1_FINAL.get(run, STAGE2.get(run, 2400 if run == "X_rq9_text_t2" else 5600))
        if run in STAGE1_FINAL or run == "X_rq9_text_20I":
            setting = "20I"
        elif run in STAGE2:
            setting = "bm25_k100"                        # EVAL_K=100 (batch 8; the side scorer's batch-32 read wins when present)
        elif run == "X_rq9_text_t2":
            setting = "bm25_k20"
        else:
            continue
        system = "text_X_rq9_text_20I" if run == "X_rq9_text_20I" else ("text_X_rq9_text_t2" if run == "X_rq9_text_t2" else run)
        put(system, step, setting, p, 1)
        continue
    m = re.fullmatch(r"(.+?)(?:_u(\d+))?\.dev1000(_100I|_200I)?\.k(\d+)\.jsonl", name)
    if not m:
        continue
    system, step, inj, k = m.group(1), int(m.group(2) or 0), m.group(3), int(m.group(4))
    if inj:
        setting = inj[1:]
    elif src in ("orcd_rq_eval20i", "orcd_rq_eval_fin20i"):
        setting = "20I"
    else:
        setting = f"bm25_k{k}"
    put(system, step, setting, p, PRI.get(src, 2))

checks = []


def correct(path, setting):
    rs = rows(path)
    q = QS[setting]
    v = {r["id"]: score(r.get("pred"), q[r["id"]])["acc"] for r in rs if r["id"] in q}
    assert len(v) == 1000, f"{path}: {len(v)} scored rows"
    summ = Path(str(path).replace(".jsonl", ".summary.json"))
    if summ.exists():
        want = json.loads(summ.read_text()).get("acc")
        got = 100 * sum(v.values()) / len(v)
        if want is not None and abs(got - want) > 0.05:
            checks.append(f"{Path(path).name}: rescored {got:.1f} vs summary {want}")
    return v


cache = {}


def get(system, setting, step=None):
    """(acc vector, step) for a system/setting; step=None: the latest step present."""
    steps = sorted(s for (sy, s, se) in found if sy == system and se == setting)
    if not steps:
        return None
    s = steps[-1] if step is None else (step if step in steps else None)
    if s is None:
        return None
    key = (system, s, setting)
    if key not in cache:
        cache[key] = correct(found[key][0], setting)
    return cache[key], s


def acc(v):
    return 100 * sum(v.values()) / len(v)


def paired(a, b, n_boot=2000, seed=0):
    ids = sorted(a)
    d = [a[i] - b[i] for i in ids]
    delta = 100 * sum(d) / len(d)
    rnd = random.Random(seed)
    boots = sorted(100 * sum(d[rnd.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    nb, nc = sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)
    n, m = nb + nc, min(nb, nc)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0
    return delta, lo, hi, p


def cell_cmp(a, b):
    if a is None or b is None:
        return "–"
    d, lo, hi, p = paired(a, b)
    star = "**" if p < 0.05 else ""
    return f"{star}{d:+.1f}{star} [{lo:+.1f}, {hi:+.1f}] p={p:.3f}" if p >= 0.001 else f"{star}{d:+.1f}{star} [{lo:+.1f}, {hi:+.1f}] p<0.001"


SYSTEMS = [  # (label, system name, stage-1 fixed step or None = latest)
    ("Text, untuned", "text_base", 0),
    ("Text + LoRA (20I)", "text_X_rq9_text_20I", 5600),
    ("Text + LoRA stage 2", "text_X_rq9_text_t2", None),
    ("AC SFT 20I, LoRA reader", "T_res_rq9L_20I", 5600),
    ("AC SFT 20I, frozen reader", "T_res_rq9fr_20I", 5600),
    ("AC SFT 20I, LoRA reader, 4B encoder", "T_res_rq9e4_e2", 5600),
    ("AC KL 20I, LoRA reader", "T_res_rq9L_20I_KL", 5600),
    ("AC KL 20I, frozen reader", "T_res_rq9fr_20I_KL", 5600),
    ("AC SFT stage 2, LoRA", "T_res_rq9L_kI", None),
    ("AC SFT stage 2, frozen", "T_res_rq9fr_kI", None),
    ("AC KL stage 2, LoRA", "T_res_rq9L_kI_KL", None),
    ("AC KL stage 2, frozen", "T_res_rq9fr_kI_KL", None),
]
IDEAL = ["20I", "100I", "200I"]
BM25 = ["bm25_k20", "bm25_k100", "bm25_k200"]


def val(sysname, setting, step):
    g = get(sysname, setting, step)
    return g


out = []
w = out.append
w("# §3.1 scoreboard (2B encoder → 9B reader, HotpotQA dev1000)\n")
w("Re-scored from per-question outputs; deltas are paired on the same 1,000 questions: Δ points [95% bootstrap CI], exact McNemar p; "
  "**bold** = p < 0.05. Stage-2 rows show the latest scored checkpoint (uNNNN) per column.\n")
for title, cols, heads in (("A. Ideal retrieval (golds injected)", IDEAL, ["20I (k=20)", "100I (k=100)", "200I (k=200)"]),
                           ("B. Plain BM25 (no injection)", BM25, ["k=20", "k=100", "k=200"])):
    w(f"\n## {title}\n")
    w("| System | " + " | ".join(heads) + " |")
    w("|---|" + "---|" * len(cols))
    for label, sy, st in SYSTEMS:
        cells = []
        for c in cols:
            g = val(sy, c, st)
            cells.append("–" if g is None else (f"{acc(g[0]):.1f}" + (f" (u{g[1]})" if st is None else "")))
        if any(x != "–" for x in cells):
            w(f"| {label} | " + " | ".join(cells) + " |")

AC = [s for s in SYSTEMS if s[1].startswith("T_res")]
base, lora = ("text_base", 0), ("text_X_rq9_text_20I", 5600)

w("\n## A. Ideal retrieval: AC − text at the same setting\n")
w("| AC system | vs untuned 20I | vs untuned 100I | vs untuned 200I | vs LoRA text 20I | vs LoRA text 100I | vs LoRA text 200I |")
w("|---|---|---|---|---|---|---|")
for label, sy, st in AC:
    cells = []
    for ref in (base, lora):
        for c in IDEAL:
            a, b = val(sy, c, st), val(ref[0], c, ref[1])
            cells.append(cell_cmp(a and a[0], b and b[0]))
    if any(x != "–" for x in cells):
        w(f"| {label} | " + " | ".join(cells) + " |")

w("\n## B. Plain BM25: AC at k=100 / 200 − text at k=20 / 100 / 200\n")
w("| AC system | AC k | vs untuned k=20 | vs untuned k=100 | vs untuned k=200 | vs LoRA text k=20 | vs LoRA text k=100 | vs LoRA text k=200 |")
w("|---|---|---|---|---|---|---|---|")
for label, sy, st in AC:
    for kk in ("bm25_k100", "bm25_k200"):
        a = val(sy, kk, st)
        if a is None:
            continue
        cells = [cell_cmp(a[0], (lambda g: g and g[0])(val(ref[0], c, ref[1]))) for ref in (base, lora) for c in BM25]
        w(f"| {label}{'' if st is not None else f' (u{a[1]})'} | {kk[6:]} | " + " | ".join(cells) + " |")

if checks:
    w("\n**Rescore mismatches:** " + "; ".join(checks))
else:
    w(f"\nAll {len(cache)} outputs re-scored to their summary accuracy (±0.05).")
text = "\n".join(out) + "\n"
MD.write_text(text)
print(text)
