"""
Math recipe (2026-11-17): the dev evaluation inputs on CPU (this host), so the GPU eval (math_eval.py) needs no host-only code.
Per dev problem: the prompt texts around the three runs (texts[0] + run 1 + texts[1] + run 2 + texts[2] + run 3 + texts[3] = the
full-runs prompt, asserted), the runs, the results, the gold task, k, and the 9B chat-template token ids of the "full" and "results"
prompts (from build_data.py's prompts_dev.jsonl: the eval asserts that the harness builds exactly these ids for the text arms).

    .venv/bin/python -I make_eval_inputs.py [SPLIT]     -> IB/TMP/PRE_REBUTTAL/math_recipe/data/eval_inputs_<split>.jsonl (+ .sha256)
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_packets as B  # noqa: E402  (sets up the paths of the host kits)


def main() -> int:
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    problems = B.load_problems(split)
    prompts = {}
    for r in map(json.loads, open(B.DATA / f"prompts_{split}.jsonl")):
        prompts.setdefault(r["task_id"], {})[r["condition"]] = {"text": r["prompt"], "ids": r["prompt_token_ids"]}
    out = B.DATA / f"eval_inputs_{split}.jsonl"
    with open(out, "w") as f:
        for task_id in sorted(problems):
            p = problems[task_id]
            full, content = B.synth_views(p, "AC", 0.1)
            texts = [piece["text"] for piece in content if piece.get("type") == "text"]
            assert len(texts) == 4 and full == prompts[task_id]["full"]["text"]
            assert texts[0] + p["runs"][0] + texts[1] + p["runs"][1] + texts[2] + p["runs"][2] + texts[3] == full
            f.write(json.dumps({"task_id": task_id, "k": p["k"], "difficulty": p["difficulty"], "dataset_task": p["dataset_task"],
                                "solver_results": p["solver_results"], "solver_right": p["solver_right"], "texts": texts, "runs": p["runs"],
                                "prompts": {c: {"text": v["text"], "ids": v["ids"]} for c, v in prompts[task_id].items()}}, ensure_ascii=False) + "\n")
    (out.parent / (out.name + ".sha256")).write_text(hashlib.sha256(out.read_bytes()).hexdigest() + f"  {out.name}\n")
    print(f"{len(problems)} problems -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
