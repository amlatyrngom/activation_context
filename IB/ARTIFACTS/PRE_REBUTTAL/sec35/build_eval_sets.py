"""§3.5 evaluation sets for the pre-rebuttal reruns (2026-11-09).

dev1000 (probes, checkpoint and recipe choices) and test1000 (reported numbers only; never used to choose anything): HotpotQA
validation questions (m1/hotpot_val_native, 7,405) minus the retrieval experiments' eval500 and dev200, shuffled with seed 20261109,
first 1,000 / next 1,000. Same construction as hotpot_tasks_v1/dev200: question-only BM25 (bm25s, English stopwords, k=1000 as in m1/rq_data2.py, repo
defaults) over the pooled HotpotQA corpus, top 20, both golds injected with rq_common.inject_golds. As a check the same index
reproduces dev200's stored top 20 (before injection) exactly.

  python build_eval_sets.py --out IB/TMP/PRE_REBUTTAL/sec35_eval_v1
"""
import argparse, json, random, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RQ = ROOT / "IB/TMP/PAPER_RETRIEVAL_QA"
TASKS = ROOT / "IB/TMP/PAPER_SEC45/data/hotpot_tasks_v1"
sys.path.insert(0, str(RQ / "kit"))
import rq_common as C  # noqa: E402

TOP, SEED = 20, 20261109


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists (never replaced)")
    val = [json.loads(l) for l in open(RQ / "m1/hotpot_val_native.questions.jsonl")]
    used = {json.loads(l)["id"] for l in open(RQ / "m1/eval500.questions.jsonl")} | {json.loads(l)["id"] for l in open(RQ / "kit/dev200.questions.jsonl")}
    pool = [q for q in val if q["id"] not in used]
    random.Random(SEED).shuffle(pool)
    sets = {"dev1000": pool[:1000], "test1000": pool[1000:2000]}
    dev200_rows = [json.loads(l) for l in open(TASKS / "dev200.jsonl")]
    corpus = C.load_corpus(TASKS)
    import bm25s
    r = bm25s.BM25()
    r.index(bm25s.tokenize([f"{x}\n{y}" for x, y in corpus], stopwords="en", show_progress=False), show_progress=False)

    def top_k(questions):
        idx, _ = r.retrieve(bm25s.tokenize([q["question"] for q in questions], stopwords="en", show_progress=False), k=1000, show_progress=False, n_threads=8)
        return {q["id"]: row[:TOP].astype(np.int64).tolist() for q, row in zip(questions, idx)}

    # check: dev200 through this index equals the stored ranking (dev200_top1000_q.npy, before injection)
    dev200_q = C.load_questions(RQ / "kit", "dev200")
    stored = dict(zip([q["id"] for q in dev200_q], np.load(RQ / "m1/out/dev200_top1000_q.npy")[:, :TOP].tolist()))
    mine = top_k(dev200_q)
    same = sum(mine[i] == [int(x) for x in stored[i]] for i in stored)
    injected = C.inject_golds({i: v for i, v in mine.items()}, dev200_q, TOP)
    same_rows = sum(injected[row["id"]] == row["top20"] for row in dev200_rows)
    print(f"dev200 check: raw top20 identical {same}/200, injected identical to hotpot_tasks_v1 {same_rows}/200", flush=True)
    assert same == 200 and same_rows == 200
    tmp = a.out.with_name(a.out.name + ".tmp"); tmp.mkdir(parents=True)
    stats = {"seed": SEED, "pool": len(pool), "rule": "question-only BM25 top 20, both golds injected (rq_common.inject_golds)"}
    for name, qs in sets.items():
        raw = top_k(qs)
        inj = C.inject_golds(raw, qs, TOP)
        with open(tmp / f"{name}.jsonl", "w") as f:
            for q in qs:
                f.write(json.dumps({"id": q["id"], "question": q["question"], "answer": q["answer"], "aliases": q.get("aliases", []),
                                    "golds": q["golds"], "type": q["type"], "top20": inj[q["id"]]}) + "\n")
        stats[name] = {"questions": len(qs), "both_golds_in_bm25_top20_before_injection": round(sum(set(q["golds"]) <= set(raw[q["id"]]) for q in qs) / len(qs), 4),
                       "types": {t: sum(q["type"] == t for q in qs) for t in ("bridge", "comparison")}}
    assert not ({q["id"] for q in sets["dev1000"]} & {q["id"] for q in sets["test1000"]})
    (tmp / "stats.json").write_text(json.dumps(stats, indent=1))
    tmp.rename(a.out)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
