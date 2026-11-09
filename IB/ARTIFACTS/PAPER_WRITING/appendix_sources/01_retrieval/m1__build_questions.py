"""M1 question sets (Oct 20): the pooled HotpotQA corpus exactly as bm25_recall.hotpotqa builds it (checked against the shipped corpus),
12,000 training questions (seed 1) and 500 evaluation questions from validation, disjoint from the probe's 500 (seed 2).
Writes train12k.questions.jsonl and eval500.questions.jsonl (id, question, answer, aliases, golds, type)."""
import gzip, json, random, sys
sys.path.insert(0, "/workspace/IB/TMP/PAPER_RETRIEVAL_QA")
from datasets import load_dataset
S = "/workspace/IB/TMP/PAPER_RETRIEVAL_QA/"
corpus, key_to_id, sets = [], {}, {"train": [], "validation": []}
for split in ("train", "validation"):
    for row in load_dataset("hotpotqa/hotpot_qa", "distractor", split=split):
        ids = []
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            key = (title, "".join(sents))
            if key not in key_to_id:
                key_to_id[key] = len(corpus); corpus.append(key)
            ids.append(key_to_id[key])
        gold_titles = set(row["supporting_facts"]["title"])
        golds = sorted({i for i, t in zip(ids, row["context"]["title"]) if t in gold_titles})
        sets[split].append(dict(id=row["id"], question=row["question"], answer=row["answer"], aliases=[], golds=golds, type=row["type"], level=row["level"]))
shipped = gzip.open(S + "ship/hotpotqa.corpus.jsonl.gz", "rt")
n = 0
for i, line in enumerate(shipped):
    assert tuple(json.loads(line)) == corpus[i], f"corpus mismatch at {i}"; n += 1
assert n == len(corpus) == 509315, (n, len(corpus))
probe = {json.loads(l)["id"] for l in open(S + "ship/hotpotqa.questions.jsonl")}
train = [q for q in sets["train"] if len(q["golds"]) == 2]
random.Random(1).shuffle(train)
val = [q for q in sets["validation"] if q["golds"] and q["id"] not in probe]
random.Random(2).shuffle(val)
for name, qs in (("train12k", train[:12000]), ("eval500", val[:500])):
    with open(S + f"m1/{name}.questions.jsonl", "w") as f:
        for q in qs: f.write(json.dumps(q) + "\n")
    print(name, len(qs), "types", {t: sum(q["type"] == t for q in qs) for t in ("bridge", "comparison")})
print("corpus verified", n, "train pool", len(train), "val pool (minus probe)", len(val))
