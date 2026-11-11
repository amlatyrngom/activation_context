"""
Section 4.5 kit, shared parts: the three write modes, the data, and the two item builders. Thin layer over existing code:
  - models and AC fields: run_qa.harness() of the 4B bench (PAPER_CORPUS_QA/kit/bench/run_qa.py, env QA4B_*), unchanged;
  - plain passage reconstruction: ActivationContextStudyGenerator.generate_reconstruction_samples(variant="reconstruct")
    (the reader sees the passage only through its compressed part and reproduces all of it; SFT);
  - the task: rq_common.build_items (PAPER_RETRIEVAL_QA/kit) with one question-first encoder call over the 20 passages and the
    answer-only target "Final answer: <answer>" (rq_common.answer_replies; SFT, thinking off).

Write modes (env of run_qa.pair_setup; a loaded pair carries its own mode in ac_modules.pt and is checked against --mode):
  in    Input Embeddings          QA4B_LAYERS=none
  res   Hidden Residuals          QA4B_LAYERS=3,7,11,15,19,23,27,31,35  (input rows plus residual deltas at every 4th reader layer)
  kv    All-Layer KVs             QA4B_KV=1 QA4B_LAYERS=none            (the encoder's keys/values at its markers enter every reader layer)
Every mode: QA4B_READER = QA4B_SIDE = Qwen/Qwen3-4B-Instruct-2507 (one loaded base with two adapters), reader LoRA rank QA4B_RANK
(128), interleaved summary layout, compression ratio 1/10, RQ_THINKING=0.
v6 (Oct 27): a smaller encoder only when asked for explicitly, SEC45_ENCODER=<Hub id> (e.g. Qwen/Qwen3-0.6B; modes in/res only:
kv needs the reader's own base). It becomes QA4B_SIDE (a separate encoder base, run_qa.harness); the AC row head keeps run_qa's
default for unequal widths (row_head_linear_skip False). Without SEC45_ENCODER any encoder/reader mismatch is still refused.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_REPO = Path(__file__).resolve().parents[5]           # pre-rebuttal kit: IB/ARTIFACTS/PRE_REBUTTAL/sec35/kit (in a bundle: env vars set)
RQ_KIT = Path(os.environ.get("SEC45_RQ_KIT", str(_REPO / "IB/TMP/PAPER_RETRIEVAL_QA/kit")))
BENCH = Path(os.environ.get("CQA_BENCH", str(_REPO / "IB/TMP/PAPER_CORPUS_QA/kit/bench")))
READER_ID = "Qwen/Qwen3-4B-Instruct-2507"
RES_LAYERS = "3,7,11,15,19,23,27,31,35"
MODES = {"in": {"QA4B_LAYERS": "none", "QA4B_KV": "0"},
         "res": {"QA4B_LAYERS": RES_LAYERS, "QA4B_KV": "0"},
         "kv": {"QA4B_LAYERS": "none", "QA4B_KV": "1"}}
RATIO = float(os.environ.get("SEC45_RATIO") or 0.1)   # rows per input token; 2026-11-11: SEC45_RATIO knob (retrieval 20I probe), unset = 0.1 as before
LAYOUT = "interleaved"
PER_CALL = 20                     # the task's 20 passages, read in ONE encoder call


def setup_environment(mode: str) -> None:
    """The process environment of a mode, before run_qa / rq_common are imported. QA4B_READER / QA4B_SIDE default to the 4B
    instruct model; a value set by the caller (the CPU smoke's tiny model) is kept. QA4B_LAYERS may be overridden for the
    'res' mode by SEC45_RES_LAYERS (the smoke's 4-layer model); the real runs never set it."""
    if mode not in MODES:
        raise SystemExit(f"--mode must be one of {sorted(MODES)}")
    os.environ["RQ_THINKING"] = "0"
    os.environ.setdefault("QA4B_READER", READER_ID)
    encoder = os.environ.get("SEC45_ENCODER")
    if encoder:
        if mode == "kv":
            raise SystemExit("SEC45_ENCODER: All-Layer KVs need the encoder to be the reader's own base")
        if os.environ.setdefault("QA4B_SIDE", encoder) != encoder:
            raise SystemExit(f"SEC45_ENCODER {encoder} but QA4B_SIDE {os.environ['QA4B_SIDE']}")
    else:
        os.environ.setdefault("QA4B_SIDE", os.environ["QA4B_READER"])
        if os.environ["QA4B_SIDE"] != os.environ["QA4B_READER"]:
            raise SystemExit("section 4.5 runs: the encoder is a copy of the reader (QA4B_SIDE must equal QA4B_READER; "
                             "a different encoder only with SEC45_ENCODER)")
    os.environ.update(MODES[mode])
    if mode == "res" and os.environ.get("SEC45_RES_LAYERS"):
        os.environ["QA4B_LAYERS"] = os.environ["SEC45_RES_LAYERS"]
    for path in (str(BENCH), str(RQ_KIT), str(HERE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ.setdefault("CQA_BENCH", str(BENCH))


def linear_skip() -> bool:
    """v7 (Oct 27): SEC45_LINEAR_SKIP=1 starts a FRESH smaller-encoder pair with the row head's linear skip (row_head_linear_skip: rows
    start as the least-squares side->reader embedding map of the encoder's final states instead of a random FFN projection). Only with
    SEC45_ENCODER (equal widths use the identity skip). Unset = v6 behaviour. A loaded or resumed pair keeps the head it was saved with."""
    on = os.environ.get("SEC45_LINEAR_SKIP") == "1"
    if on and not os.environ.get("SEC45_ENCODER"):
        raise SystemExit("SEC45_LINEAR_SKIP=1 needs SEC45_ENCODER (a smaller encoder; equal widths use the identity skip)")
    return on


def require_cuda(torch) -> None:
    """The real runs need a GPU; only the CPU smoke and the CPU job tests set SEC45_ALLOW_CPU=1."""
    if not torch.cuda.is_available() and os.environ.get("SEC45_ALLOW_CPU") != "1":
        raise SystemExit("no CUDA device (torch.cuda.is_available() is False); SEC45_ALLOW_CPU=1 is for the CPU smoke only")


def order_digest(item_ids: list[str], epu: int, first: int = 32) -> dict:
    """The first `first` items in training order, grouped by update (each update's ids sorted: a micro-batch may regroup them
    inside an update), and their sha256: equal digests = the same items in the same updates."""
    updates = -(-first // epu)
    groups = [sorted(item_ids[k * epu:(k + 1) * epu]) for k in range(updates)]
    groups = [g for g in groups if g]
    return {"first_updates": groups, "first_sha256": hashlib.sha256(json.dumps(groups).encode()).hexdigest()}


def check_pair_mode(ac_path, reader_path, mode: str) -> dict:
    """The pair's saved setup (run_qa.pair_setup) after checking that it is the requested write mode."""
    from run_qa import pair_setup
    setup = pair_setup(ac_path, reader_path)
    fields = setup["ac"]
    kv = bool(fields.get("kv_transfer"))
    layers = fields.get("intervention_layers") or []
    found = "kv" if kv else ("res" if layers or fields.get("intervention_frequency", 0) != 0 else "in")
    if found != mode:
        raise SystemExit(f"the pair {ac_path} is a '{found}' pair (kv_transfer {kv}, layers {layers}), not '{mode}'")
    encoder = os.environ.get("SEC45_ENCODER") or setup["reader"]
    if setup["side"] != encoder:
        raise SystemExit(f"the pair {ac_path} has encoder {setup['side']} and reader {setup['reader']}: expected encoder {encoder} "
                         f"(section 4.5 uses one model for both unless SEC45_ENCODER names the encoder; SEC45_ENCODER={os.environ.get('SEC45_ENCODER')})")
    return setup


def load_recon(data: Path) -> tuple[list[dict], list[dict]]:
    """(train, held) passages of build_fineweb_recon.py's material.json.gz."""
    with gzip.open(Path(data) / "material.json.gz", "rt") as f:
        material = json.load(f)
    return material["train"], material["held"]


def load_tasks(data: Path, split: str) -> list[dict]:
    """Rows of build_hotpot_tasks.py: {id, question, answer, aliases, golds, type, top20}."""
    return [json.loads(line) for line in open(Path(data) / f"{split}.jsonl")]


def recon_items(study, passages: list[dict], seed: int = 0):
    """Plain passage reconstruction at 1/10: the passage is one AC part, the target is the whole passage (SFT)."""
    rows = [{"text": p["text"], "doc_id": p["doc_id"], "source": p.get("source", "fineweb_edu")} for p in passages]
    items = study.generate_reconstruction_samples(rows, ratio=RATIO, variant="reconstruct", seed=seed)
    if len(items) != len(rows):
        raise ValueError(f"{len(rows)} passages gave {len(items)} reconstruction items")
    return items


def row_k(row: dict) -> int:
    """Passages read for one row. §3.5 rows (top20): 20. Retrieval rows (top200, pre-rebuttal 2026-11-09): SEC45_K, one k or a
    comma list (a mix: each row draws k = ks[crc32(id) % len(ks)], fixed per question, so every arm and every seed sees the same k)."""
    if "top20" in row:
        return PER_CALL
    import zlib
    ks = [int(x) for x in os.environ.get("SEC45_K", "20").split(",")]
    assert all(k % PER_CALL == 0 and 0 < k <= len(row["top200"]) for k in ks), ks
    return ks[zlib.crc32(row["id"].encode()) % len(ks)]


def row_passages(row: dict) -> list[int]:
    """The passage ids a row's reader sees: §3.5 top20 (golds injected), retrieval top200[:k] (plain BM25)."""
    return [int(x) for x in (row["top20"] if "top20" in row else row["top200"][:row_k(row)])]


def target_replies(rows: list[dict]) -> dict:
    """Retrieval target probe (2026-11-09): with SEC45_TARGETS=<gold_refs.py jsonl>, id -> the gold-view teacher's reply for rows whose
    reply stopped and is correct (the rest keep the answer-only target). Unset: {} (answer-only for every row)."""
    path = os.environ.get("SEC45_TARGETS")
    if not path:
        return {}
    ids = {r["id"] for r in rows}
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["id"] in ids and r["correct"] and r["finish"] == "stop":
            out[r["id"]] = r["text"].strip()
    return out


def task_items(study, corpus, rows: list[dict], ac_name: str):
    """HotpotQA items: question first, then the row's k passages as k/20 question-first encoder calls of 20 (§3.5: ONE call over
    the 20 with golds injected), target 'Final answer: <answer>'. Items come back in the order of `rows`."""
    import rq_common as C
    assert not C.THINK, "RQ_THINKING=0 must be set before rq_common is imported (setup_environment)"
    top = {r["id"]: row_passages(r) for r in rows}
    replies = C.answer_replies(rows) | target_replies(rows)
    ks = [len(top[r["id"]]) for r in rows]
    out = [None] * len(rows)
    for k in sorted(set(ks)):
        idx = [i for i, x in enumerate(ks) if x == k]
        group = C.build_items(study, corpus, [rows[i] for i in idx], top, ac_name, RATIO, replies=replies, calls=k // PER_CALL,
                              per_call=PER_CALL, question_first=True)
        assert len(group) == len(idx), (k, len(group), len(idx))
        for i, item in zip(idx, group):
            out[i] = item
    return out


def link_pair(ac_folder, reader_folder, destination: Path) -> bool:
    """Hard-link a saved pair into destination/{ac,reader} without optimizer.pt (never copied around). False when it exists."""
    import shutil
    destination = Path(destination)
    if destination.exists():
        return False
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    for source, name in ((Path(ac_folder), "ac"), (Path(reader_folder), "reader")):
        shutil.copytree(source, temporary / name, copy_function=os.link, ignore=shutil.ignore_patterns("optimizer.pt", "teacher_snapshots"))
    temporary.rename(destination)
    return True
