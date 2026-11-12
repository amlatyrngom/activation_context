"""General multi-hop QA tuning data (pre-rebuttal 2026-11-12; user: "a completely different data set ... just like there is a reconstruction
tuning phase, have a QA tuning phase", one continuous run with one configuration, HotpotQA checked periodically). Sources (raw files from the
Hugging Face Hub under IB/TMP/PRE_REBUTTAL/raw_qa/, parsed as data only):
  2WikiMultihopQA (voidful/2WikiMultihopQA, Apache-2.0): train.json / dev.json; 10 context paragraphs per question, supporting_facts titles
  MuSiQue-Ans (dgslibisey/MuSiQue, CC BY 4.0): musique_ans_v1.0_{train,dev}.jsonl; 17-20 paragraphs, is_supporting marks the golds
Each question reads ITS OWN paragraphs (no retrieval), order shuffled per question (seeded by id) so the gold position carries no signal,
as ONE encoder call (10 or up to 20 passages: the 20I shape). Target "Final answer: <answer>"; the gold view (KL teacher) = its golds.
Output dir (the sec45_train / sec45_eval / gold_refs row format, rows {id, question, answer, aliases, golds, type, top20}):
  hotpotqa.corpus.jsonl.gz   the HotpotQA corpus (same ids, first) + the new paragraphs appended (deduplicated by (title, text))
  train.jsonl    N_MQ MuSiQue-Ans train (all answerable) + 2Wiki train sampled to a total of --total (seed 20261112)
  dev200.jsonl   100 MuSiQue dev + 100 2Wiki dev (the trainer's held panel)
  dev1000.jsonl  hotpot_tasks_v2 dev1000 (HotpotQA 20I) unchanged: the periodic HotpotQA check during the QA phase
  qa_dev1000.jsonl  500 MuSiQue dev + 500 2Wiki dev (in-domain check)
  .venv/bin/python -I IB/ARTIFACTS/PRE_REBUTTAL/sec35/build_qa_tune.py --out IB/TMP/PRE_REBUTTAL/data/qa_tune_v1 --total 100000
"""
import argparse, ast, gzip, json, random, shutil, zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "IB/TMP/PRE_REBUTTAL/raw_qa"; DATA = ROOT / "IB/TMP/PRE_REBUTTAL/data"
SEED = 20261112


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); p.add_argument("--total", type=int, default=100000)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists (never replaced)")
    tmp = a.out.with_name(a.out.name + ".tmp"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    corpus = [tuple(json.loads(l)) for l in gzip.open(DATA / "hotpot_tasks_v2/hotpotqa.corpus.jsonl.gz", "rt")]
    n_hotpot = len(corpus)
    index = {}                                              # (title, text) -> id, new paragraphs only

    def pid(title, text):
        key = (title, text)
        if key not in index:
            index[key] = len(corpus); corpus.append(key)
        return index[key]

    def order(qid, ids):
        ids = list(ids); random.Random(zlib.crc32(qid.encode())).shuffle(ids); return ids

    def wiki_rows(path, prefix):
        out, skipped = [], 0
        for e in json.load(open(path)):
            ctx = ast.literal_eval(e["context"]) if isinstance(e["context"], str) else e["context"]
            sup = ast.literal_eval(e["supporting_facts"]) if isinstance(e["supporting_facts"], str) else e["supporting_facts"]
            ids, by_title = [], {}
            for title, sents in ctx:
                i = pid(title, " ".join(s.strip() for s in sents).strip()); ids.append(i); by_title.setdefault(title.lower(), i)
            golds = sorted({by_title[t.lower()] for t, _ in sup if t.lower() in by_title})
            if not golds or len(golds) < len({t.lower() for t, _ in sup}) or len(set(ids)) != len(ids):
                skipped += 1; continue
            qid = f"{prefix}{e['_id']}"
            out.append({"id": qid, "question": e["question"], "answer": e["answer"], "aliases": [], "golds": golds, "type": f"2wiki/{e['type']}", "top20": order(qid, ids)})
        return out, skipped

    def musique_rows(path, prefix):
        out, skipped = [], 0
        for l in open(path):
            e = json.loads(l)
            if not e["answerable"]:
                skipped += 1; continue
            ids = [pid(x["title"], x["paragraph_text"].strip()) for x in e["paragraphs"]]
            golds = sorted({ids[k] for k, x in enumerate(e["paragraphs"]) if x["is_supporting"]})
            if not golds or len(set(ids)) != len(ids):
                skipped += 1; continue
            qid = f"{prefix}{e['id']}"
            out.append({"id": qid, "question": e["question"], "answer": e["answer"], "aliases": e.get("answer_aliases", []), "golds": golds,
                        "type": f"musique/{e['id'].split('__')[0]}", "top20": order(qid, ids)})
        return out, skipped

    mq_tr, s1 = musique_rows(RAW / "MuSiQue/musique_ans_v1.0_train.jsonl", "mq:")
    mq_dv, s2 = musique_rows(RAW / "MuSiQue/musique_ans_v1.0_dev.jsonl", "mq:")
    w_tr, s3 = wiki_rows(RAW / "2WikiMultihopQA/train.json", "2w:")
    w_dv, s4 = wiki_rows(RAW / "2WikiMultihopQA/dev.json", "2w:")
    rng = random.Random(SEED)
    rng.shuffle(w_tr); rng.shuffle(w_dv); rng.shuffle(mq_dv)
    n_w = a.total - len(mq_tr)
    assert 0 < n_w <= len(w_tr), (a.total, len(mq_tr), len(w_tr))
    train = mq_tr + w_tr[:n_w]; rng.shuffle(train)
    sets = {"train": train, "dev200": mq_dv[:100] + w_dv[:100], "qa_dev1000": mq_dv[100:600] + w_dv[100:600]}
    assert not ({r["id"] for r in sets["dev200"]} & {r["id"] for r in sets["qa_dev1000"]})
    for name, rows in sets.items():
        with open(tmp / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    shutil.copy(DATA / "hotpot_tasks_v2/dev1000.jsonl", tmp / "dev1000.jsonl")
    with gzip.open(tmp / "hotpotqa.corpus.jsonl.gz", "wt") as f:
        for row in corpus:
            f.write(json.dumps(list(row)) + "\n")
    stats = {"seed": SEED, "corpus": {"hotpot": n_hotpot, "new": len(corpus) - n_hotpot},
             "train": {"total": len(train), "musique": len(mq_tr), "2wiki": n_w},
             "skipped": {"musique_train": s1, "musique_dev": s2, "2wiki_train": s3, "2wiki_dev": s4},
             "available": {"2wiki_train": len(w_tr), "musique_dev": len(mq_dv), "2wiki_dev": len(w_dv)},
             "passages_per_question": {n: sorted({len(r["top20"]) for r in rows}) for n, rows in sets.items()},
             "golds_per_question": {n: sorted({len(r["golds"]) for r in rows}) for n, rows in sets.items()}}
    (tmp / "stats.json").write_text(json.dumps(stats, indent=1))
    tmp.rename(a.out)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
