"""
Math recipe dev_hard (2026-11-18), step 0 on CPU (this host): build_data.py's per-problem records and prompts for the dev_hard problems
(make_dev_hard.py's draw), from their three 9B solver runs (devhard_solve_node.sh on a Sky node; full rollout rows, relayed from S3
into their own folder). Run with `python -I` (it reads downloaded rollout files).

Same rules as build_data.py, unchanged (its functions are imported): the runs rendered as text with the 40k rule, their results, which
are right (score >= 1), k; prompts "full" (the teacher test's, = the student's layout) and "results" (the paper's results-only prompt),
9B chat template, thinking on. Asserted: every drawn problem has seeds 0, 1, 2 (rows of other tasks are refused); 0 shared task ids and
0 shared normalised question texts with train / dev / dmext / dmhard (splits.json; texts from their rollouts) and the teacher test.

    .venv/bin/python -I build_dev_hard_data.py SAMPLE_JSON ROLLOUTS_JSONL      -> data/problems_devhard.jsonl, data/prompts_devhard.jsonl
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_data as B  # noqa: E402  (sets up the paths of the host kits)

B.OUT = B.M / "data"    # build_data reads OUT from argv[1] at import; ours is the sample path


def main() -> int:
    sample_path, rollouts_path = Path(sys.argv[1]), Path(sys.argv[2])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(B.HF9)
    drawn = json.loads(sample_path.read_text())["sample"]["deepmath"]["task_ids"]
    rows = {}
    with open(rollouts_path) as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            task_id = row["agent_config"]["dataset_task"]["task_id"]
            assert task_id in set(drawn), f"a row of an undrawn task {task_id}"
            rows[(task_id, int(row["seed"]))] = B.slim(row)
    assert set(rows) == {(t, s) for t in drawn for s in (0, 1, 2)}, f"{len(rows)} (task, seed) pairs for {len(drawn)} tasks x 3"
    splits = json.loads((B.OUT / "splits.json").read_text())
    protected = {"train": set(splits["train"]), "dev": set(splits["dev"]), "dmext": set(splits["test"]["dmext"]), "dmhard": set(splits["test"]["dmhard"])}
    protected["teacher_test"] = {json.loads(line)["task_id"] for line in open(B.TEACHER_PROMPTS) if line.strip()}
    texts = {}
    for split in ("train", "dev", "dmext", "dmhard"):
        for (task, _), row in B.load(split).items():
            texts.setdefault(split, set()).add(B.norm_question(row))
    texts["teacher_test"] = set()        # all 200 are in train (asserted below by id)
    assert protected["teacher_test"] <= protected["train"]
    mine = {B.norm_question(row) for row in rows.values()}
    overlap = {}
    for name, ids in protected.items():
        overlap[name] = {"task_ids": len(set(drawn) & ids), "question_texts": len(mine & texts[name])}
        assert overlap[name] == {"task_ids": 0, "question_texts": 0}, (name, overlap[name])
    problems, prompts = [], []
    for task_id in sorted(drawn):
        runs = [rows[(task_id, seed)] for seed in (0, 1, 2)]
        task = runs[0]["agent_config"]["dataset_task"]
        results = [B.solver_result(r, B.CAP) for r in runs]
        right = [B.score(task, a) >= 1.0 for a in results]
        record = {"task_id": task_id, "split": "devhard", "difficulty": task_id.split("-")[0][1:], "k": sum(right), "solver_right": right,
                  "solver_results": results, "dataset_task": task, "task_body": B.task_text("deepmath", runs[0]["agent_config"]["user_prompt"]),
                  "runs": [B.render_run(r, B.CAP) for r in runs], "source": "dev_hard_v1"}
        conditions = {"full": B.full_prompt(runs),
                      "results": B.results_prompt("deepmath", runs[0]["agent_config"]["user_prompt"], "synth_result", results, [None] * 3)}
        for condition, prompt in conditions.items():
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            prompts.append({"id": f"{condition}/{task_id}", "condition": condition, "task_id": task_id, "k": record["k"], "solver_right": right,
                            "dataset_task": task, "solver_results": results, "prompt": prompt, "chat_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "prompt_token_ids": tok.encode(text, add_special_tokens=False)})
        problems.append(record)
    for name, items in (("problems_devhard.jsonl", problems), ("prompts_devhard.jsonl", prompts)):
        with open(B.OUT / name, "w") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (B.OUT / f"{name}.sha256").write_text(hashlib.sha256((B.OUT / name).read_bytes()).hexdigest() + f"  {name}\n")
    lengths = sorted(len(p["prompt_token_ids"]) for p in prompts if p["condition"] == "full")
    bands = {"d8.5": [r for r in problems if float(r["difficulty"]) == 8.5], "ge9": [r for r in problems if float(r["difficulty"]) >= 9]}
    summary = {"problems": len(problems), "overlap": overlap, "difficulty": dict(sorted(Counter(r["difficulty"] for r in problems).items())),
               "k": dict(sorted(Counter(r["k"] for r in problems).items())),
               "full_prompt_tokens": {"median": lengths[len(lengths) // 2], "max": lengths[-1], "over_96k": sum(x > 96_000 for x in lengths)},
               "single_solver": sum(r["k"] for r in problems) / (3 * len(problems)), "any_of_3": sum(r["k"] > 0 for r in problems) / len(problems),
               "bands": {b: {"problems": len(v), "k": dict(sorted(Counter(r["k"] for r in v).items())), "single_solver": sum(r["k"] for r in v) / (3 * len(v)),
                             "any_of_3": sum(r["k"] > 0 for r in v) / len(v)} for b, v in bands.items() if v}}
    (B.OUT / "devhard_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
