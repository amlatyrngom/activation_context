"""Accuracy of one text + reader-LoRA adapter (or the untuned 4B with --base) under the pre-rebuttal single scorer (2026-11-09), the
same scorer as the AC arms' sec45_eval.py:
  decode   HF transformers generate, greedy (do_sample False), left-padded batches, stop at the end of turn, at most --max-new new
           tokens (256 for every arm); thinking off; no presence/repetition penalty
  prompt   rq_common.READ with the row's top20 (both golds injected) as text = the user turn the AC reader sees with its parts as text
  score    rq_common.score on rq_common.final_answer(text, need_think=False) (the last "Final answer:" line; strict, as sec45_eval)
           acc = normalized exact match or the normalized gold inside the normalized prediction; also em, f1, answered
OUT.jsonl gets one row per question (appended per batch; a rerun skips the ids already there), OUT.summary.json the scores (same
fields as sec45_eval.py's summary).

  python text_lora_eval_hf.py --data HOTPOT_TASKS_DIR --split dev1000 (--adapter ADAPTER_DIR | --base) --out OUT.jsonl [--label L]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RQ_THINKING", "0")
sys.path.insert(0, os.environ.get("RQ_KIT", str(Path(__file__).resolve().parent)))
import rq_common as C  # noqa: E402

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def summarize(rows, by_id):
    scores = [C.score(r["pred"], by_id[r["id"]]) for r in rows]
    n = len(scores)
    out = {k: round(100 * sum(s[k] for s in scores) / max(1, n), 2) for k in ("acc", "em", "f1", "answered")}
    return out | {"n": n, "length_stops": sum(r.get("length_stop", False) for r in rows),
                  "new_tokens_mean": round(sum(r.get("new_tokens", 0) for r in rows) / max(1, n), 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--split", default="dev1000"); p.add_argument("--out", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True); g.add_argument("--adapter", type=Path); g.add_argument("--base", action="store_true")
    p.add_argument("--label", default=""); p.add_argument("--limit", type=int); p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-new", type=int, default=256)
    a = p.parse_args()
    assert not C.THINK, "RQ_THINKING=0"
    started = time.monotonic()
    rows = [json.loads(line) for line in open(a.data / f"{a.split}.jsonl")]
    by_id = {r["id"]: r for r in rows}
    rows = rows[:a.limit] if a.limit else rows
    done = {json.loads(line)["id"] for line in open(a.out)} if a.out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    if todo:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
        if a.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(a.adapter)).merge_and_unload()     # merged: the same weights, plain decode speed
        model.eval()
        corpus = C.load_corpus(a.data)
        stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
        prompts = []
        for r in todo:
            user = C.READ.format(passages=C.passages_block(corpus, [int(x) for x in r["top20"]]), question=r["question"])
            prompts.append(tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
        order = sorted(range(len(todo)), key=lambda i: len(prompts[i]))                # similar lengths per batch (less padding)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        print(f"text_lora_eval_hf: {len(todo)} questions ({len(done)} already in {a.out}), greedy, batch {a.batch}, max_new {a.max_new}", flush=True)
        for k in range(0, len(order), a.batch):
            idx = order[k:k + a.batch]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, do_sample=False, max_new_tokens=a.max_new, eos_token_id=stop, pad_token_id=tok.pad_token_id,
                                     temperature=None, top_p=None, top_k=None)
            with open(a.out, "a") as f:
                for i, seq in zip(idx, out[:, enc["input_ids"].shape[1]:].tolist()):
                    cut = next((j for j, t in enumerate(seq) if t in stop), None)
                    ids = seq if cut is None else seq[:cut]
                    text = tok.decode(ids, skip_special_tokens=True)
                    f.write(json.dumps({"id": todo[i]["id"], "pred": C.final_answer(text, need_think=False), "new_tokens": len(ids),
                                        "length_stop": cut is None and len(ids) >= a.max_new, "text": text}) + "\n")
            print(f"text_lora_eval_hf: {min(k + a.batch, len(order))}/{len(order)} ({time.monotonic() - started:.0f}s)", flush=True)
    result = summarize([json.loads(line) for line in open(a.out)], by_id) | {"label": a.label, "adapter": str(a.adapter) if a.adapter else "base",
                                                                               "split": a.split, "max_new": a.max_new, "seconds": round(time.monotonic() - started)}
    summary = a.out.with_suffix(".summary.json")
    summary.with_suffix(".tmp").write_text(json.dumps(result, indent=1)); summary.with_suffix(".tmp").replace(summary)
    print(f"text_lora_eval_hf: {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
