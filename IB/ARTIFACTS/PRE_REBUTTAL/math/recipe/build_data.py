"""
Math synthesis clean recipe (§3.4, 2026-11-17), step 0 on CPU (this host): splits, overlap checks, per-problem records and the
teacher / baseline prompts. Run with `python -I` (it reads downloaded rollout files).

Splits (CLEAN_RECIPES_PLAN §4 / log 2026-11-17, DeepMath only):
  train  the 1,500 training-pool problems with three 9B solver runs (math/DATA_INVENTORY.md §1):
         tqa_m1_train_v2 (1,000, difficulty >= 8.5; slims tqa_m1_v2_9b_{a,b}) + tqa_train_solve_v1's 500 DeepMath problems (slim)
  dev    tqa_dev_solve_v1's 200 DeepMath problems (3 solver seeds; full rollouts downloaded from S3 into their own folder)
  test   dmext (tqa_dmext_solve_v1, 300) + dmhard (tqa_dmhard, 300): only their task ids and question texts are read here,
         for the overlap check. Nothing else of the test sets is touched.
Asserted: 0 shared task ids and 0 shared normalised question texts between any two of train / dev / dmext / dmhard; every train and
dev problem has seeds 0, 1, 2; the train sample files are role "training", dev "development".

Per problem (train and dev): the three runs rendered as text exactly as the teacher test (tqa_sft_prep_runs.render_run, 40k rule),
their results (solver_result), which are right (score_synth1call.score >= 1), k. Prompts (teacher test, math/TEACHER_TEST.md):
  full     task text minus its submit_answer sentence, INSTRUCTION_FULL, "Agent i\\nIts run:\\n<run>\\n\\nFinal answer: <result>" x 3,
           FINAL_RULE   (the teacher; also the student's layout with each <run> replaced by its AC part)
  results  run_synth1call_v2.build_prompt(synth_result): the paper's results-only prompt (dev baseline)
chat text = the 9B template, one user message, enable_thinking=True; token ids with the 9B tokenizer. For the 200 teacher-test
problems (all in train) the full prompt's chat text must hash equal to the teacher test's prompts.jsonl (asserted): same prompt.

Outputs (OUT, default IB/TMP/PRE_REBUTTAL/math_recipe/data): splits.json, problems_{train,dev}.jsonl (records above, no token ids),
prompts_train.jsonl (condition full), prompts_dev.jsonl (full + results), in teacher_gen.py's prompt format.

    .venv/bin/python -I IB/ARTIFACTS/PRE_REBUTTAL/math/recipe/build_data.py
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
T = ROOT / "IB/TMP/PAPER_TRAJ_QA"
for p in (ROOT, ROOT / "IB/TMP/PAPER_COMPACTION/kit", T / "kit", ROOT / "IB/ARTIFACTS/PRE_REBUTTAL/math/teacher_test"):
    sys.path.insert(0, str(p))
from run_synth1call_v2 import FINAL_RULE, build_prompt as results_prompt, task_text   # noqa: E402
from tqa_sft_prep_runs import render_run, solver_result                                # noqa: E402
from score_synth1call import score                                                     # noqa: E402
from build_prompts import INSTRUCTION_FULL, CAP                                        # noqa: E402  (the teacher test's own constants)

M = ROOT / "IB/TMP/PRE_REBUTTAL/math_recipe"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else M / "data"
HF9 = ROOT / "IB/TMP/PRE_REBUTTAL/inventory_math/hf_configs/Qwen_Qwen3.5-9B"
TEACHER_PROMPTS = ROOT / "IB/TMP/PRE_REBUTTAL/math_teacher_test/prompts.jsonl"
SOURCES = {
    "train": [("slim", T / "m1/v2/slims/tqa_m1_v2_9b_a/rollouts.jsonl"), ("slim", T / "m1/v2/slims/tqa_m1_v2_9b_b/rollouts.jsonl"),
              ("slim", T / "node_sync/slim_train_v1/ROLLOUTS/tqa_train_solve_v1/rollouts.jsonl")],
    "dev": [("full", M / "dl_dev/rollouts.jsonl")],
    "dmext": [("full", M / "dl_dmext/rollouts.jsonl")],
    "dmhard": [("full", T / "node_sync/dmhard/rollouts.jsonl")],
}
SAMPLES = {"train": [(T / "m1/v2/tqa_m1_train_v2.sample.json", "training"), (T / "node_sync/slim_train_v1/PAPER_TRAJ_QA/tqa_train_solve_v1/sample.json", "training")],
           "dev": [(T / "round2/samples/tqa_dev_solve_v1.sample.json", "development")],
           "dmext": [(T / "node_sync/dmext/sample.json", None)], "dmhard": [(T / "node_sync/dmhard/sample.json", None)]}


def slim(row: dict) -> dict:
    """A full rollout row in the slim shape render_run / solver_result read (slim_tqa_runs.py)."""
    config = row["agent_config"]
    return {"agent_config": {"dataset_task": config.get("dataset_task"), "metadata": config.get("metadata"), "user_prompt": config.get("user_prompt")},
            "seed": row.get("seed"), "score": row.get("score"), "finish_reason": row.get("finish_reason"), "answer": row.get("answer"),
            "prompt_messages": row.get("prompt_messages"), "prompt_tokens": len(row.get("prompt_token_ids") or []),
            "trajectory": [{"role": s.get("role"), "messages": s.get("messages"), "tokens": len(s.get("token_ids") or [])} for s in row["trajectory"]]}


def load(split: str) -> dict:
    """(task_id, seed) -> slim row, DeepMath only; a later row of the same (task, seed) replaces an earlier one (as slim_tqa_runs.py)."""
    rows = {}
    for kind, path in SOURCES[split]:
        with open(path) as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if (row["agent_config"].get("metadata") or {}).get("benchmark") != "deepmath":
                    continue
                row = slim(row) if kind == "full" else row
                rows[(row["agent_config"]["dataset_task"]["task_id"], int(row["seed"]))] = row
    return rows


def norm_question(row: dict) -> str:
    task = row["agent_config"]["dataset_task"]
    q = task.get("agent_prompt") or row["agent_config"].get("user_prompt") or ""
    return hashlib.sha1(re.sub(r"\W+", " ", q.lower()).strip().encode()).hexdigest()


def full_prompt(rows: list[dict]) -> str:
    """The teacher test's full-runs prompt (teacher_test/build_prompts.full_prompt), byte for byte."""
    blocks = [task_text("deepmath", rows[0]["agent_config"]["user_prompt"]), INSTRUCTION_FULL]
    for position, row in enumerate(rows, start=1):
        blocks.append(f"Agent {position}\nIts run:\n{render_run(row, CAP)}\n\nFinal answer: {solver_result(row, CAP)}")
    blocks.append(FINAL_RULE["deepmath"])
    return "\n\n".join(blocks)


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF9)
    rows = {split: load(split) for split in SOURCES}
    ids, sampled = {}, {}
    for split in SOURCES:
        ids[split] = {task for task, _ in rows[split]}
        listed = set()
        for path, role in SAMPLES[split]:
            doc = json.loads(path.read_text())
            assert role is None or str(doc.get("role")).startswith(role), (path, doc.get("role"))
            listed |= set(doc["sample"]["deepmath"]["task_ids"])
        sampled[split] = listed
        assert ids[split] == listed, (split, len(ids[split]), len(listed))
    assert len(ids["train"]) == 1500 and len(ids["dev"]) == 200 and len(ids["dmext"]) == 300 and len(ids["dmhard"]) == 300, {s: len(v) for s, v in ids.items()}
    questions = {split: {task: norm_question(row) for (task, _), row in rows[split].items()} for split in SOURCES}
    overlap = {}
    names = list(SOURCES)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared_ids = ids[a] & ids[b]
            shared_q = set(questions[a].values()) & set(questions[b].values())
            overlap[f"{a}|{b}"] = {"task_ids": len(shared_ids), "question_texts": len(shared_q)}
            assert not shared_ids and not shared_q, (a, b, len(shared_ids), len(shared_q))
    within = {split: len(questions[split]) - len(set(questions[split].values())) for split in SOURCES}

    OUT.mkdir(parents=True, exist_ok=True)
    teacher = {}
    if TEACHER_PROMPTS.exists():
        for line in open(TEACHER_PROMPTS):
            r = json.loads(line)
            if r["condition"] == "full":
                teacher[r["task_id"]] = r["chat_text_sha256"]
    checked = 0
    counts = {}
    m1_ids = set(json.loads(SAMPLES["train"][0][0].read_text())["sample"]["deepmath"]["task_ids"])
    for split in ("train", "dev"):
        problems, prompts = [], []
        for task_id in sorted(ids[split]):
            runs = [rows[split][(task_id, seed)] for seed in (0, 1, 2)]       # KeyError = a missing seed
            task = runs[0]["agent_config"]["dataset_task"]
            results = [solver_result(r, CAP) for r in runs]
            right = [score(task, a) >= 1.0 for a in results]
            record = {"task_id": task_id, "split": split, "difficulty": task_id.split("-")[0][1:], "k": sum(right), "solver_right": right,
                      "solver_results": results, "dataset_task": task, "task_body": task_text("deepmath", runs[0]["agent_config"]["user_prompt"]),
                      "runs": [render_run(r, CAP) for r in runs], "source": ("m1_train_v2" if task_id in m1_ids else "train_solve_v1") if split == "train" else "dev_solve_v1"}
            conditions = {"full": full_prompt(runs)}
            if split == "dev":
                conditions["results"] = results_prompt("deepmath", runs[0]["agent_config"]["user_prompt"], "synth_result", results, [None] * 3)
            for condition, prompt in conditions.items():
                text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
                sha = hashlib.sha256(text.encode()).hexdigest()
                if condition == "full" and task_id in teacher:
                    assert teacher[task_id] == sha, f"{task_id}: full prompt differs from the teacher test's"
                    checked += 1
                prompts.append({"id": f"{condition}/{task_id}", "condition": condition, "task_id": task_id, "k": record["k"], "solver_right": right,
                                "dataset_task": task, "solver_results": results, "prompt": prompt, "chat_text_sha256": sha,
                                "prompt_token_ids": tok.encode(text, add_special_tokens=False)})
            problems.append(record)
        with open(OUT / f"problems_{split}.jsonl", "w") as f:
            for r in problems:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(OUT / f"prompts_{split}.jsonl", "w") as f:
            for r in prompts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        lengths = sorted(len(p["prompt_token_ids"]) for p in prompts if p["condition"] == "full")
        counts[split] = {"problems": len(problems), "k": dict(sorted(Counter(r["k"] for r in problems).items())),
                         "difficulty": dict(sorted(Counter(r["difficulty"] for r in problems).items())),
                         "source": dict(Counter(r["source"] for r in problems)),
                         "full_prompt_tokens": {"median": lengths[len(lengths) // 2], "p90": lengths[int(0.9 * (len(lengths) - 1))], "max": lengths[-1],
                                                "over_72k": sum(x > 72_000 for x in lengths), "over_96k": sum(x > 96_000 for x in lengths)},
                         "single_solver": sum(r["k"] for r in problems) / (3 * len(problems)), "any_of_3": sum(r["k"] > 0 for r in problems) / len(problems)}
    assert checked == len(teacher) == 200, (checked, len(teacher))
    splits = {"rule": "items never repeat across splits (ROOT_CANON); DeepMath only; test sets read for ids/question texts only",
              "train": sorted(ids["train"]), "dev": sorted(ids["dev"]), "test": {"dmext": sorted(ids["dmext"]), "dmhard": sorted(ids["dmhard"])},
              "overlap": overlap, "duplicate_questions_within_split": within, "teacher_test_prompts_matched": checked, "counts": counts,
              "sources": {s: [str(p.relative_to(ROOT)) for _, p in v] for s, v in SOURCES.items()}}
    (OUT / "splits.json").write_text(json.dumps(splits, indent=1))
    for name in ("problems_train.jsonl", "problems_dev.jsonl", "prompts_train.jsonl", "prompts_dev.jsonl", "splits.json"):
        (OUT / f"{name}.sha256").write_text(hashlib.sha256((OUT / name).read_bytes()).hexdigest() + f"  {name}\n")
    print(json.dumps({k: v for k, v in splits.items() if k not in ("train", "dev", "test")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
