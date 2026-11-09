"""Retrieval Q/A task data for the pre-rebuttal recipe (2026-11-09): every question with its question-only BM25 top 200 over the pooled
HotpotQA corpus (bm25s, English stopwords, k=1000 then the first 200, exactly as m1/rq_data2.py; NO gold injection: the realistic
setting of Fig. 3). Rows {id, question, answer, aliases, golds, type, top200}; the reader reads top200[:k] as k/20 calls of 20.
  train    m1/hotpot_train_native (90,447; the same pool as §3.5's hotpot_tasks_v1/train)
  dev200   kit/dev200 (choices)          eval500  m1/eval500 (the paper's reported set)
Checked: dev200 equals the stored ranking m1/out/dev200_top1000_q.npy (first 200); eval500 (its stored ranking was not kept) gives both-gold recall at 20/100/200 of 52.0/65.2/72.0, the numbers in the paper.

  python build_rq_tasks.py --out IB/TMP/PRE_REBUTTAL/data/rq_tasks_v1
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RQ = ROOT / "IB/TMP/PAPER_RETRIEVAL_QA"
sys.path.insert(0, str(RQ / "kit"))
import rq_common as C  # noqa: E402

K = 200


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists (never replaced)")
    sets = {"train": RQ / "m1/hotpot_train_native.questions.jsonl", "dev200": RQ / "kit/dev200.questions.jsonl", "eval500": RQ / "m1/eval500.questions.jsonl"}
    qs = {n: [json.loads(l) for l in open(f)] for n, f in sets.items()}
    corpus = C.load_corpus(RQ / "ship")
    import bm25s
    r = bm25s.BM25()
    r.index(bm25s.tokenize([f"{x}\n{y}" for x, y in corpus], stopwords="en", show_progress=False), show_progress=False)
    top = {}
    for n, rows in qs.items():
        for s in range(0, len(rows), 5000):
            chunk = rows[s:s + 5000]
            idx, _ = r.retrieve(bm25s.tokenize([q["question"] for q in chunk], stopwords="en", show_progress=False), k=1000, show_progress=False, n_threads=8)
            top.update({q["id"]: row[:K].astype(np.int64).tolist() for q, row in zip(chunk, idx)})
        print(f"{n}: {len(rows)} ranked", flush=True)
    stored = dict(zip([q["id"] for q in qs["dev200"]], np.load(RQ / "m1/out/dev200_top1000_q.npy")[:, :K].tolist()))
    same = sum(top[i] == [int(x) for x in v] for i, v in stored.items())
    print(f"check: dev200 {same}/{len(stored)} identical to the stored ranking (eval500's top1000_q.npy was not kept)", flush=True)
    assert same == len(stored) == 200
    tmp = a.out.with_name(a.out.name + ".tmp"); tmp.mkdir(parents=True)
    os.link(RQ / "ship/hotpotqa.corpus.jsonl.gz", tmp / "hotpotqa.corpus.jsonl.gz")
    stats = {"k": K, "rule": "question-only BM25 (bm25s, en stopwords, k=1000 -> first 200), no gold injection"}
    for n, rows in qs.items():
        with open(tmp / f"{n}.jsonl", "w") as f:
            for q in rows:
                f.write(json.dumps({"id": q["id"], "question": q["question"], "answer": q["answer"], "aliases": q.get("aliases", []), "golds": q["golds"],
                                    "type": q["type"], "top200": top[q["id"]]}) + "\n")
        stats[n] = {"questions": len(rows)} | {f"both_golds_at_{k}": round(sum(set(q["golds"]) <= set(top[q["id"]][:k]) for q in rows) / len(rows), 4) for k in (20, 100, 200)}
    (tmp / "stats.json").write_text(json.dumps(stats, indent=1))
    tmp.rename(a.out)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
