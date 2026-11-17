"""
Token check of the corrected teacher view (2026-11-17, TEACHER_VIEW_FIX.md). CPU only. For every dev-75 prefix of a panel built
with the recorded teacher view (build_dev_panel.py, info["in_context_history_ids"]), the teacher's prompt up to the first
continuation turn (prepare_training_history with history_ids, as ActivationContextTrainer.build_example builds it) is compared
token by token with the prompt the agent eval's `full` arm sent for the same prefix (rollouts' prompt_token_ids, run_resume.py
--variants full), and with the old escaped rendering (no history_ids). Counts "</think>" and "[/Reasoning]" in each.

    PYTHONPATH=<repo> python check_teacher_view.py --panel PANEL --full-rollouts ROLLOUTS/cdv1base_s2_full/rollouts.jsonl [--out OUT.json]
"""
import argparse
import json
import pickle
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, required=True); p.add_argument("--full-rollouts", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    from transformers import AutoTokenizer
    from activation.ac_model.ac_model_utils import prepare_training_history
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    with open(a.panel / "panel_none.pkl", "rb") as f:
        items = pickle.load(f)["items"]
    served = {}
    for line in open(a.full_rollouts):
        row = json.loads(line)
        task = (row.get("agent_config") or {}).get("dataset_task")
        served[task["task_id"] if isinstance(task, dict) else row["config_key"].split("/")[1]] = row["prompt_token_ids"]
    think, escaped = tok.convert_tokens_to_ids("</think>"), tok.encode("[/Reasoning]", add_special_tokens=False)
    def count(ids, pattern):
        return sum(ids[i:i + len(pattern)] == pattern for i in range(len(ids) - len(pattern) + 1))
    rows = []
    for item in items:
        kwargs = dict(tools=item.tools, chat_template_kwargs=item.info.get("chat_template_kwargs"))
        history = item.info.get("in_context_history_ids")
        assert history is not None, f"{item.item_id}: panel built without the recorded teacher view"
        new, _, new_pos = prepare_training_history(tok, item.in_context_messages, item.in_context_start, item.assistant_token_ids, [], history_ids=history, **kwargs)
        old, _, old_pos = prepare_training_history(tok, item.in_context_messages, item.in_context_start, item.assistant_token_ids, [], **kwargs)
        new_prompt, old_prompt = new[:new_pos[0] + 1], old[:old_pos[0] + 1]
        full = served.get(item.info["task_id"])
        first_diff = None
        if full is not None and full != new_prompt:
            first_diff = next((i for i, (x, y) in enumerate(zip(full, new_prompt)) if x != y), min(len(full), len(new_prompt)))
        rows.append({"task_id": item.info["task_id"], "full_arm_tokens": len(full) if full is not None else None, "new_prompt_tokens": len(new_prompt),
                     "old_prompt_tokens": len(old_prompt), "identical_to_full_arm": full == new_prompt if full is not None else None, "first_difference": first_diff,
                     "continuation_identical": new[new_pos[0] + 1:] == old[old_pos[0] + 1:],
                     "new_think": new_prompt.count(think), "new_escaped": count(new_prompt, escaped), "old_think": old_prompt.count(think), "old_escaped": count(old_prompt, escaped),
                     "prefix_turns": len(history)})
    summary = {"prefixes": len(rows), "with_full_arm_prompt": sum(r["identical_to_full_arm"] is not None for r in rows),
               "identical_to_full_arm": sum(bool(r["identical_to_full_arm"]) for r in rows),
               "continuation_tokens_identical_old_new": sum(r["continuation_identical"] for r in rows),
               "prompt_tokens_new_vs_old_mean": [round(sum(r["new_prompt_tokens"] for r in rows) / len(rows), 1), round(sum(r["old_prompt_tokens"] for r in rows) / len(rows), 1)],
               "prefix_turns": sum(r["prefix_turns"] for r in rows),
               "think_in_prompt_new_old": [sum(r["new_think"] for r in rows), sum(r["old_think"] for r in rows)],
               "escaped_in_prompt_new_old": [sum(r["new_escaped"] for r in rows), sum(r["old_escaped"] for r in rows)],
               "differing": [r for r in rows if r["identical_to_full_arm"] is False][:5]}
    print(json.dumps(summary, indent=1))
    if a.out:
        a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
