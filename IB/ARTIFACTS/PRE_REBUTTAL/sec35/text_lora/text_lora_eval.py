"""Dev200 accuracy of the text + reader-LoRA baseline (text_lora_train.py) at each saved adapter, decoded like text4b_sec45.py (the
53.0 text line): rq_common.READ with the row's top20 as text, thinking off, ONE greedy sample, max 2,048 new tokens. Scores: strict =
rq_common.score on the last "Final answer:" line; lenient = the same, falling back to the reply's last non-empty line when it is at
most 120 characters and there is no "Final answer:" line (the rule that reproduces the untuned 4B's 53.0).

  python text_lora_eval.py --data HOTPOT_TASKS_DIR --out OUT [--base]     (OUT = text_lora_train.py's --out; --base adds update 0)
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RQ_THINKING", "0")
sys.path.insert(0, os.environ.get("RQ_KIT", str(Path(__file__).resolve().parent)))
import rq_common as C  # noqa: E402

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def lenient(text):
    pred = C.final_answer(text, need_think=False)
    if pred is None:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if lines and len(lines[-1]) <= 120:
            pred = lines[-1]
    return pred


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", action="store_true"); p.add_argument("--max-tokens", type=int, default=2048)
    a = p.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rows = [json.loads(line) for line in open(a.data / "dev200.jsonl")]
    corpus = C.load_corpus(a.data)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = []
    for r in rows:
        user = C.READ.format(passages=C.passages_block(corpus, [int(x) for x in r["top20"]]), question=r["question"])
        chat = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append({"prompt_token_ids": tok.encode(chat, add_special_tokens=False)})
    llm = LLM(model=MODEL_ID, max_model_len=16384, gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=128, max_loras=1, seed=0)
    ckpts = sorted((a.out / "ckpt").glob("step_*"))
    targets = ([(0, None)] if a.base else []) + [(int(c.name.split("_")[1]), c) for c in ckpts]
    scores_path = a.out / "T_text_lora.dev200.scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    for i, (update, folder) in enumerate(targets):
        key = f"u{update:05d}"
        if key in scores:
            continue
        lora = LoRARequest(f"text_lora_{update}", i + 1, str(folder)) if folder else None
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=a.max_tokens), lora_request=lora)
        strict, loose, lengths = [], [], []
        with open(a.out / f"T_text_lora.dev200.{key}.jsonl", "w") as f:
            for r, o in zip(rows, outs):
                text = o.outputs[0].text
                s, l = C.score(C.final_answer(text, need_think=False), r), C.score(lenient(text), r)
                strict.append(s); loose.append(l); lengths.append(len(o.outputs[0].token_ids))
                f.write(json.dumps({"id": r["id"], "text": text, "strict": s, "lenient": l}) + "\n")
        mean = lambda xs, k: round(100 * sum(x[k] for x in xs) / len(xs), 2)
        scores[key] = {"acc_lenient": mean(loose, "acc"), "acc_strict": mean(strict, "acc"), "em": mean(loose, "em"), "f1": mean(loose, "f1"),
                       "answered_strict": mean(strict, "answered"), "new_tokens_mean": round(sum(lengths) / len(lengths), 1), "n": len(rows)}
        scores_path.write_text(json.dumps(scores, indent=1))
        print(key, json.dumps(scores[key]), flush=True)


if __name__ == "__main__":
    main()
