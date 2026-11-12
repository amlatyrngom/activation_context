"""Stage-2 data dir (pre-rebuttal 2026-11-12; user-approved plan: stage 2 TRAINS on golds-injected 100I / 200I rows, the run's own
eval reads PLAIN BM25 -- the MVP metric). Pure assembly, no new rows:
  train.jsonl, dev200.jsonl, hotpotqa.corpus.jsonl.gz, {dev1000,test1000}_{100I,200I}.jsonl   <- hotpot_tasks_kI_v1 (build_kI_sets.py)
  dev1000.jsonl, test1000.jsonl                                                             <- hotpot_tasks_bm25e_v1 (build_bm25_eval.py)
Same questions everywhere (the 20I dev1000 / test1000 ids). The trainer's --offset 44800 (seed 0 shuffle) = the 45,647 questions stage 1
never saw; SEC45_K=100,200 draws the same k per question as the injection (crc32(id) % 2).
  python3 -I IB/ARTIFACTS/PRE_REBUTTAL/sec35/build_kI_v2.py
--text: the TEXT arm's stage-2 dir hotpot_tasks_t2_v1 (parity: the text reader keeps its own operating point, 20I, on the same held-back
questions): train.jsonl / dev200.jsonl / corpus <- hotpot_tasks_v2 (20I, top20), dev1000 / test1000 <- hotpot_tasks_bm25e_v1 (plain BM25,
read at k=20).
"""
import json, shutil, sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[4] / "IB/TMP/PRE_REBUTTAL/data"
KI, BM, OUT = DATA / "hotpot_tasks_kI_v1", DATA / "hotpot_tasks_bm25e_v1", DATA / "hotpot_tasks_kI_v2"


def ids(p):
    return [json.loads(l)["id"] for l in open(p)]


def text():
    out, v2 = DATA / "hotpot_tasks_t2_v1", DATA / "hotpot_tasks_v2"
    if out.exists():
        raise SystemExit(f"{out} exists (never replaced)")
    tmp = out.with_name(out.name + ".tmp"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    for n in ("train.jsonl", "dev200.jsonl", "hotpotqa.corpus.jsonl.gz"):
        shutil.copy(v2 / n, tmp / n)
    for s in ("dev1000", "test1000"):
        shutil.copy(BM / f"{s}.jsonl", tmp / f"{s}.jsonl")
        assert ids(tmp / f"{s}.jsonl") == ids(v2 / f"{s}.jsonl"), s
    assert ids(tmp / "train.jsonl") == ids(KI / "train.jsonl")
    assert (v2 / "hotpotqa.corpus.jsonl.gz").read_bytes() == (BM / "hotpotqa.corpus.jsonl.gz").read_bytes()
    (tmp / "stats.json").write_text(json.dumps({"from": {"20I": "hotpot_tasks_v2", "plain": "hotpot_tasks_bm25e_v1"},
                                                "plain_bm25": ["dev1000", "test1000"], "injected_20I": ["train", "dev200"]}, indent=1))
    tmp.rename(out); print(sorted(p.name for p in out.iterdir()))


def main():
    if "--text" in sys.argv[1:]:
        return text()
    if OUT.exists():
        raise SystemExit(f"{OUT} exists (never replaced)")
    tmp = OUT.with_name(OUT.name + ".tmp"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    for n in ["train.jsonl", "dev200.jsonl", "hotpotqa.corpus.jsonl.gz"] + [f"{s}_{k}I.jsonl" for s in ("dev1000", "test1000") for k in (100, 200)]:
        shutil.copy(KI / n, tmp / n)
    for s in ("dev1000", "test1000"):
        shutil.copy(BM / f"{s}.jsonl", tmp / f"{s}.jsonl")
        assert ids(tmp / f"{s}.jsonl") == ids(tmp / f"{s}_100I.jsonl") == ids(tmp / f"{s}_200I.jsonl"), s
    assert (KI / "hotpotqa.corpus.jsonl.gz").read_bytes() == (BM / "hotpotqa.corpus.jsonl.gz").read_bytes()
    (tmp / "stats.json").write_text(json.dumps({"from": {"kI": "hotpot_tasks_kI_v1", "plain": "hotpot_tasks_bm25e_v1"},
                                                "plain_bm25": ["dev1000", "test1000"], "injected": ["train", "dev200", "*_100I", "*_200I"]}, indent=1))
    tmp.rename(OUT); print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
