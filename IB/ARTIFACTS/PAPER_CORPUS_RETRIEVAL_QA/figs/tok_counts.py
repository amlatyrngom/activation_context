"""Reader-side input tokens per eval arm (Oct 21, frontier figure): mean over a question set of the 9B's prompt length.
Text arms: rq_text_eval's prompt (chat template, generation prompt, thinking off). AC arms: rq_eval's student turn (question-first,
ratio 0.1) with every part as ceil(0.1 x its side tokens) rows (ActivationContextModel.num_view_rows; min/max rows not binding here).
  python tok_counts.py DATA SET OUT.json"""
import json, math, sys
from pathlib import Path
import rq_common as C
from transformers import AutoTokenizer
from activation.ac_model.ac_model_utils import tokenize_with_parts
data, set_, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
reader, side = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B"), AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
corpus, questions, top = C.load_corpus(data), C.load_questions(data, set_), C.load_top(data, set_)
native = {}
if set_ == "dev200":
    nat = {q["id"]: q for q in map(json.loads, open(data / "hotpot_val_native.questions.jsonl"))}
    native = C.native_top([nat[q["id"]] for q in questions])
def text_len(text):
    return len(reader.encode(reader.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False), add_special_tokens=False))
def ac_len(q, ranking, calls, per_call=20):
    msgs = C.student_messages(corpus, q["question"], ranking, "rag_poc", 0.1, calls, per_call, question_first=True)
    parts = [p for p in msgs[0]["content"] if p.get("type") == "activation_context"]
    rows = [math.ceil(0.1 * len(tokenize_with_parts(side, p["messages"], [], 0)[0])) for p in parts]
    return len(tokenize_with_parts(reader, msgs, rows, 0, add_generation_prompt=True, chat_template_kwargs={"enable_thinking": False})[0]), sum(rows)
res = {}
for k in (20, 100, 200, 400):
    res[f"text_top{k}"] = sum(text_len(C.READ.format(passages=C.passages_block(corpus, [int(x) for x in top[q["id"]][:k]]), question=q["question"])) for q in questions) / len(questions)
res["text_none"] = sum(text_len(C.NO_PASSAGES.format(question=q["question"])) for q in questions) / len(questions)
res["text_gold"] = sum(text_len(C.READ.format(passages=C.passages_block(corpus, q["golds"]), question=q["question"])) for q in questions) / len(questions)
if native:
    res["text_native"] = sum(text_len(C.READ.format(passages=C.passages_block(corpus, native[q["id"]]), question=q["question"])) for q in questions) / len(questions)
for calls in (1, 2, 5, 10, 20):
    lens = [ac_len(q, top[q["id"]], calls) for q in questions]
    res[f"ac_{calls}x20"] = sum(a for a, _ in lens) / len(lens); res[f"ac_{calls}x20_rows"] = sum(b for _, b in lens) / len(lens)
    print(calls, res[f"ac_{calls}x20"], flush=True)
if native:
    lens = [ac_len(q, native[q["id"]], 1, len(native[q["id"]])) for q in questions]
    res["ac_native"] = sum(a for a, _ in lens) / len(lens)
out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
