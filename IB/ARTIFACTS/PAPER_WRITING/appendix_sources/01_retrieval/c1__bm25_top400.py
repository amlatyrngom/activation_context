"""BM25 (bm25s, question only, as rq_text_eval.dev_bm25 / bm25_train.py) top-400 of the 509,315-paragraph pooled HotpotQA corpus for
every question of hotpot_train_native (Oct 21, Track A real-format arms: 20 x 20 lists, no injected golds). Writes
data/train_native_top400.npy (int32, question order of the file) + .ids.json; never replaces."""
import json, sys, time
from pathlib import Path
import bm25s, numpy as np
sys.path.insert(0, "/root/rq/kit_c1"); import rq_common as C
D = Path("/root/rq/data"); out = D / "train_native_top400.npy"
assert not out.exists()
t = time.time(); corpus = C.load_corpus(D)
qs = [json.loads(l) for l in open(D / "hotpot_train_native.questions.jsonl")]
r = bm25s.BM25(); r.index(bm25s.tokenize([f"{x}\n{y}" for x, y in corpus], stopwords="en", show_progress=False), show_progress=False)
print(f"indexed {len(corpus)} in {time.time()-t:.0f}s", flush=True)
idx, _ = r.retrieve(bm25s.tokenize([q["question"] for q in qs], stopwords="en", show_progress=False), k=400, show_progress=False, n_threads=44)
idx = idx.astype(np.int32); np.save(out, idx); (D / "train_native_top400.ids.json").write_text(json.dumps([q["id"] for q in qs]))
rec = {k: round(100 * float(np.mean([set(q["golds"]) <= set(row[:k].tolist()) for q, row in zip(qs, idx)])), 1) for k in (20, 100, 400)}
print(f"done {idx.shape} in {time.time()-t:.0f}s; all-gold recall {rec}", flush=True)
