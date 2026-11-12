"""100I / 200I task data for the retrieval stage 2 (pre-rebuttal, 2026-11-12): BM25 top k with both golds injected (the 20I rule at k = 100, 200).

Rows {id, question, answer, aliases, golds, type, k, top200}. `top200[:k]` is rq_common.inject_golds(BM25 top 200, k): the row's k
passages with every gold present at a seeded rank (golds the BM25 top k already holds keep their rank); the rest of the list is the
BM25 tail (unused at that k, kept so the rows load with sec45_common.row_passages / SEC45_K unchanged).
  train.jsonl          rq_tasks_v1/train (the 90,447 questions, same row order) with k = KS[crc32(id) % 2], KS = (100, 200): the row
                       draw of sec45_common.row_k with SEC45_K=100,200, so a run with that SEC45_K reads exactly the injected k
  dev200.jsonl         rq_tasks_v1/dev200 (the trainer's held panel), same per-row k rule
  {dev1000,test1000}_{100I,200I}.jsonl   the 20I eval questions (hotpot_tasks_v2 dev1000/test1000, same order), every row at that k
                       (evaluate with --k 100 / --k 200)
Checks: the dev1000/test1000 BM25 rankings come from the index of build_eval_sets.py, and inject_golds(ranking, 20) must equal
hotpot_tasks_v2's stored top20 for every row; the train/dev200 top200 before injection equal rq_tasks_v1's.

  .venv/bin/python IB/ARTIFACTS/PRE_REBUTTAL/sec35/build_kI_sets.py --out IB/TMP/PRE_REBUTTAL/data/hotpot_tasks_kI_v1
"""
import argparse, json, shutil, sys, zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RQ = ROOT / "IB/TMP/PAPER_RETRIEVAL_QA"
DATA = ROOT / "IB/TMP/PRE_REBUTTAL/data"
sys.path.insert(0, str(RQ / "kit"))
import rq_common as C  # noqa: E402

KS, TOP = (100, 200), 200


def row_k(qid: str) -> int:
    return KS[zlib.crc32(qid.encode()) % len(KS)]          # = sec45_common.row_k with SEC45_K="100,200"


def injected(ranking: list[int], q: dict, k: int) -> list[int]:
    head = C.inject_golds({q["id"]: ranking}, [q], k)[q["id"]]
    seen = set(head)
    return (head + [x for x in ranking if x not in seen])[:TOP]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists (never replaced)")
    tmp = a.out.with_name(a.out.name + ".tmp"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    shutil.copy(DATA / "hotpot_tasks_v2/hotpotqa.corpus.jsonl.gz", tmp / "hotpotqa.corpus.jsonl.gz")
    stats = {"rule": "BM25 top k (question-only, bm25s, k=1000 then 200) with both golds injected (rq_common.inject_golds at k)", "ks": KS}

    for split in ("train", "dev200"):                       # rq_tasks_v1 already holds each question's BM25 top 200
        rows = [json.loads(l) for l in open(DATA / f"rq_tasks_v1/{split}.jsonl")]
        n_k = {k: 0 for k in KS}; both = {k: 0 for k in KS}
        with open(tmp / f"{split}.jsonl", "w") as f:
            for r in rows:
                k = row_k(r["id"]); rank = [int(x) for x in r["top200"]]
                n_k[k] += 1; both[k] += set(r["golds"]) <= set(rank[:k])
                f.write(json.dumps({**{x: r[x] for x in ("id", "question", "answer", "aliases", "golds", "type")}, "k": k, "top200": injected(rank, r, k)}) + "\n")
        stats[split] = {"questions": len(rows), "rows_per_k": n_k, "both_golds_in_bm25_topk_before_injection": {k: round(both[k] / max(n_k[k], 1), 4) for k in KS}}
        print(split, stats[split], flush=True)

    corpus = C.load_corpus(DATA / "hotpot_tasks_v2")
    import bm25s
    r = bm25s.BM25()
    r.index(bm25s.tokenize([f"{x}\n{y}" for x, y in corpus], stopwords="en", show_progress=False), show_progress=False)
    for split in ("dev1000", "test1000"):
        v2 = [json.loads(l) for l in open(DATA / f"hotpot_tasks_v2/{split}.jsonl")]
        idx, _ = r.retrieve(bm25s.tokenize([q["question"] for q in v2], stopwords="en", show_progress=False), k=1000, show_progress=False, n_threads=8)
        ranks = {q["id"]: row[:TOP].astype(np.int64).tolist() for q, row in zip(v2, idx)}
        same20 = sum(C.inject_golds({q["id"]: ranks[q["id"]]}, [q], 20)[q["id"]] == [int(x) for x in q["top20"]] for q in v2)
        print(f"{split}: inject_golds(ranking, 20) equals hotpot_tasks_v2 top20 for {same20}/{len(v2)}", flush=True)
        assert same20 == len(v2), "the BM25 ranking differs from the one behind hotpot_tasks_v2"
        stats[split] = {"questions": len(v2), "check_20I_identical": same20}
        for k in KS:
            with open(tmp / f"{split}_{k}I.jsonl", "w") as f:
                for q in v2:
                    f.write(json.dumps({**{x: q[x] for x in ("id", "question", "answer", "aliases", "golds", "type")}, "k": k, "top200": injected(ranks[q["id"]], q, k)}) + "\n")
            stats[split][f"both_golds_in_bm25_top{k}_before_injection"] = round(sum(set(q["golds"]) <= set(ranks[q["id"]][:k]) for q in v2) / len(v2), 4)
        print(split, stats[split], flush=True)
    (tmp / "stats.json").write_text(json.dumps(stats, indent=1))
    tmp.rename(a.out)


if __name__ == "__main__":
    main()
