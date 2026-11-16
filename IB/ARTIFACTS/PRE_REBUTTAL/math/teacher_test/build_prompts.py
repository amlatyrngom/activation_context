"""
Math teacher test (2026-11-15), step 1 (CPU, this host): build the prompts for the 200-problem teacher set.

Inputs:
  IB/TMP/PRE_REBUTTAL/inventory_math/teacher_set_m1v2_hard200.json   (task ids, k, the inventory's solver results)
  IB/TMP/PAPER_TRAJ_QA/m1/v2/slims/tqa_m1_v2_9b_{a,b}/rollouts.jsonl (the three 9B runs per problem, seeds 0, 1, 2)
Conditions (math/DATA_INVENTORY.md §5/§6):
  full     task text minus the submit_answer sentence, INSTRUCTION_FULL, per agent "Agent i\\nIts run:\\n<render_run(cap 40,000)>\\n\\n
           Final answer: <solver_result>", FINAL_RULE  (the AC arm's layout with the runs as text)
  results  run_synth1call_v2.build_prompt(condition synth_result): the paper's results-only prompt, unchanged
Both are one user message, no system message, thinking on. The chat text is rendered with the 9B template (enable_thinking=True);
the 27B template with reasoning_effort="medium" renders the same string (checked here), and both are tokenized with the 9B
tokenizer (inventory §4). Output: one JSON line per (task, condition) with the prompt, its chat-text token ids and metadata.

    .venv/bin/python IB/ARTIFACTS/PRE_REBUTTAL/math/teacher_test/build_prompts.py OUT.jsonl
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
KIT = ROOT / "IB/TMP/PAPER_TRAJ_QA/kit"
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "IB/TMP/PAPER_COMPACTION/kit"))   # run_synth imports resume_program from there
from run_synth1call_v2 import FINAL_RULE, build_prompt as results_prompt, task_text   # noqa: E402
from tqa_sft_prep_runs import render_run, solver_result                                # noqa: E402

SET = ROOT / "IB/TMP/PRE_REBUTTAL/inventory_math/teacher_set_m1v2_hard200.json"
SLIMS = [ROOT / f"IB/TMP/PAPER_TRAJ_QA/m1/v2/slims/tqa_m1_v2_9b_{h}/rollouts.jsonl" for h in "ab"]
HF = ROOT / "IB/TMP/PRE_REBUTTAL/inventory_math/hf_configs"
CAP = 40_000
INSTRUCTION_FULL = ("Three independent agents attempted this task. Their runs (their reasoning, tool calls and tool results) and their final "
                    "answers are below. They may be wrong or disagree. No tools are available to you: reason about the task, their runs and "
                    "their answers, then give the single best answer.")


def full_prompt(rows: list[dict]) -> str:
    blocks = [task_text("deepmath", rows[0]["agent_config"]["user_prompt"]), INSTRUCTION_FULL]
    for position, row in enumerate(rows, start=1):
        blocks.append(f"Agent {position}\nIts run:\n{render_run(row, CAP)}\n\nFinal answer: {solver_result(row, CAP)}")
    blocks.append(FINAL_RULE["deepmath"])
    return "\n\n".join(blocks)


def main() -> int:
    from transformers import AutoTokenizer
    out = Path(sys.argv[1])
    teacher_set = json.loads(SET.read_text())
    wanted = set(teacher_set["task_ids"])
    rows: dict = {}
    for path in SLIMS:
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                task_id = row["agent_config"]["dataset_task"]["task_id"]
                if task_id in wanted:
                    rows[(task_id, int(row["seed"]))] = row
    tok9 = AutoTokenizer.from_pretrained(HF / "Qwen_Qwen3.5-9B")
    tok27 = AutoTokenizer.from_pretrained(HF / "unsloth_Qwen3.8-27B-NVFP4")
    mismatches, written = 0, 0
    with open(out, "w") as handle:
        for task_id in teacher_set["task_ids"]:
            solvers = [rows[(task_id, seed)] for seed in (0, 1, 2)]
            results = [solver_result(row, CAP) for row in solvers]
            info = teacher_set["per_task"][task_id]
            assert results == info["results"], (task_id, results, info["results"])
            task = solvers[0]["agent_config"]["dataset_task"]
            prompts = {"full": full_prompt(solvers),
                       "results": results_prompt("deepmath", solvers[0]["agent_config"]["user_prompt"], "synth_result", results, [None] * 3)}
            for condition, prompt in prompts.items():
                message = [{"role": "user", "content": prompt}]
                text9 = tok9.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                text27 = tok27.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True, reasoning_effort="medium")
                mismatches += text9 != text27
                ids = tok9.encode(text9, add_special_tokens=False)
                handle.write(json.dumps({"id": f"{condition}/{task_id}", "condition": condition, "task_id": task_id, "k": info["k"],
                                         "solver_right": info["solver_right"], "dataset_task": task, "solver_results": results,
                                         "prompt": prompt, "chat_text_sha256": hashlib.sha256(text9.encode()).hexdigest(),
                                         "prompt_token_ids": ids}, ensure_ascii=False) + "\n")
                written += 1
    print(f"build_prompts: {written} prompts for {len(teacher_set['task_ids'])} tasks -> {out}; 9B vs 27B(medium) chat text mismatches: {mismatches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
