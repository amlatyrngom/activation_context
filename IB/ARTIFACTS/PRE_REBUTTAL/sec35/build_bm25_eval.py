"""Plain-BM25 eval rows for the 20I questions (pre-rebuttal 2026-11-12; user MVP: "true bm25 100/200 should match/exceed the text 20,
definitely if untuned and preferably if tuned"). NO gold injection: rows {id, question, answer, aliases, golds, type, top200} with the
question-only BM25 top 200 (the index of build_eval_sets.py / build_kI_sets.py); read top200[:k] with --k 20 / 100 / 200 (sec45_common.row_k).
  dev1000.jsonl / test1000.jsonl   the hotpot_tasks_v2 questions in the same order;  dev200.jsonl  rq_tasks_v1's (held panel)
Check: inject_golds(top200, 20) equals hotpot_tasks_v2 top20 for every row (the same ranking as 20I).
  .venv/bin/python IB/ARTIFACTS/PRE_REBUTTAL/sec35/build_bm25_eval.py --out IB/TMP/PRE_REBUTTAL/data/hotpot_tasks_bm25e_v1
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]; RQ = ROOT / "IB/TMP/PAPER_RETRIEVAL_QA"; DATA = ROOT / "IB/TMP/PRE_REBUTTAL/data"
sys.path.insert(0, str(RQ / "kit"))
import rq_common as C  # noqa: E402


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists (never replaced)")
    tmp = a.out.with_name(a.out.name + ".tmp"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    shutil.copy(DATA / "hotpot_tasks_v2/hotpotqa.corpus.jsonl.gz", tmp / "hotpotqa.corpus.jsonl.gz")
    shutil.copy(DATA / "rq_tasks_v1/dev200.jsonl", tmp / "dev200.jsonl")
    corpus = C.load_corpus(DATA / "hotpot_tasks_v2")
    import bm25s
    r = bm25s.BM25(); r.index(bm25s.tokenize([f"{x}\n{y}" for x, y in corpus], stopwords="en", show_progress=False), show_progress=False)
    stats = {"rule": "question-only BM25 top 200, no gold injection"}
    for split in ("dev1000", "test1000"):
        v2 = [json.loads(l) for l in open(DATA / f"hotpot_tasks_v2/{split}.jsonl")]
        idx, _ = r.retrieve(bm25s.tokenize([q["question"] for q in v2], stopwords="en", show_progress=False), k=1000, show_progress=False, n_threads=8)
        ranks = {q["id"]: row[:200].astype(np.int64).tolist() for q, row in zip(v2, idx)}
        same = sum(C.inject_golds({q["id"]: ranks[q["id"]]}, [q], 20)[q["id"]] == [int(x) for x in q["top20"]] for q in v2)
        assert same == len(v2), (split, same)
        with open(tmp / f"{split}.jsonl", "w") as f:
            for q in v2:
                f.write(json.dumps({**{x: q[x] for x in ("id", "question", "answer", "aliases", "golds", "type")}, "top200": ranks[q["id"]]}) + "\n")
        stats[split] = {"questions": len(v2), "check_20I_identical": same} | {f"both_golds_in_top{k}": round(sum(set(q["golds"]) <= set(ranks[q["id"]][:k]) for q in v2) / len(v2), 4) for k in (20, 100, 200)}
        print(split, stats[split], flush=True)
    (tmp / "stats.json").write_text(json.dumps(stats, indent=1)); tmp.rename(a.out)


if __name__ == "__main__":
    main()
