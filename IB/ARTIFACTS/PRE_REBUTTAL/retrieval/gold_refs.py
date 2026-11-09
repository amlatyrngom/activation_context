"""Gold-view references for the retrieval target probe (pre-rebuttal 2026-11-09): the base reader (Qwen3.5-9B, thinking off) reads the
question's GOLD passages only (rq_common.teacher_messages: the READ prompt over the golds) and answers, GREEDY, at most --max-tokens.
One row per question {id, text, output_tokens, finish, pred, correct}; a training item uses `text` as its target when the reply
stopped and is correct (rq_common.score on the last "Final answer:" line), else the answer-only target "Final answer: <answer>"
(sec45_common.task_items with SEC45_TARGETS). Arm G distilled from this view (sampled, KL); here the same teacher, greedy, as text.

  python gold_refs.py --data rq_tasks_v1 --split train --first 32000 --seed 0 --out refs_train.jsonl   (the first N of train's seed shuffle,
                                                                                                      = sec45_train's item pool order)
  python gold_refs.py --data rq_tasks_v1 --split dev200 --out refs_dev200.jsonl
A rerun appends only the ids not yet in --out.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("RQ_THINKING", "0")
sys.path.insert(0, os.environ.get("RQ_KIT", str(Path(__file__).resolve().parent)))
import rq_common as C  # noqa: E402

MODEL_ID = os.environ.get("TEXT_MODEL_ID", "Qwen/Qwen3.5-9B")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--split", default="train"); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--first", type=int); p.add_argument("--seed", type=int, default=0); p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--chunk", type=int, default=4000)
    a = p.parse_args()
    assert not C.THINK
    rows = [json.loads(line) for line in open(a.data / f"{a.split}.jsonl")]
    if a.split == "train":
        random.Random(a.seed).shuffle(rows)                # sec45_train: pool shuffled with --seed, then the first steps x epu items
    rows = rows[:a.first] if a.first else rows
    done = {json.loads(line)["id"] for line in open(a.out)} if a.out.exists() else set()
    rows = [r for r in rows if r["id"] not in done]
    print(f"gold_refs: {len(rows)} questions to run ({len(done)} already in {a.out})", flush=True)
    if not rows:
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    corpus = C.load_corpus(a.data)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(model=MODEL_ID, max_model_len=8192, gpu_memory_utilization=0.9, limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    started, kept, total = time.monotonic(), 0, 0
    for s in range(0, len(rows), a.chunk):
        chunk = rows[s:s + a.chunk]
        prompts = [{"prompt_token_ids": tok.encode(tok.apply_chat_template(C.teacher_messages(corpus, r), tokenize=False, add_generation_prompt=True,
                                                                        enable_thinking=False), add_special_tokens=False)} for r in chunk]
        outs = llm.generate(prompts, params, use_tqdm=False)
        with open(a.out, "a") as f:
            for r, o in zip(chunk, outs):
                c = o.outputs[0]
                pred = C.final_answer(c.text, need_think=False) if c.finish_reason == "stop" else None
                correct = bool(pred is not None and C.score(pred, r)["acc"])
                kept += correct; total += 1
                f.write(json.dumps({"id": r["id"], "text": c.text, "output_tokens": len(c.token_ids), "finish": c.finish_reason, "pred": pred,
                                    "correct": correct}) + "\n")
        print(f"gold_refs: {s + len(chunk)}/{len(rows)}, correct {kept}/{total} ({time.monotonic() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
