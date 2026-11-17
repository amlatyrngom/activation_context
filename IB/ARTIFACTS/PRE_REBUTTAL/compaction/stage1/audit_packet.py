"""
Independent audit of a stage-1 KL packet (2026-11-17, retrain with the corrected teacher view). CPU, this host.
  - overlap of the packet's task ids with the 322 excluded tasks (IB/TMP/PAPER_COMPACTION/kit/exclude_h7.json: eval-200, dev-75,
    frozen-100), read independently of the builder's own exclusion
  - counts: items, distinct item ids / runs / tasks, by cut, seed, split side, r; items carrying info["in_context_history_ids"]
  - recipe arm's teacher_view
  - teacher view of a seeded sample of items, built as ActivationContextTrainer.build_example builds it (prepare_training_history with
    history_ids): occurrences of "[/Reasoning]" (must be 0) and "</think>" in the decoded teacher text, prefix vs targets

    PYTHONPATH=<repo> python audit_packet.py PACKET_DIR [--sample 40] [--out OUT.json]
"""
import argparse
import collections
import json
import pickle
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]


def main():
    p = argparse.ArgumentParser(); p.add_argument("packet", type=Path); p.add_argument("--sample", type=int, default=40); p.add_argument("--out", type=Path)
    a = p.parse_args()
    items = list(pickle.load(open(a.packet / "items.pkl", "rb"))["train"])
    excluded = set(json.load(open(REPO / "IB/TMP/PAPER_COMPACTION/kit/exclude_h7.json"))["task_ids"])
    C = collections.Counter
    tasks = {i.info["task_id"] for i in items}
    recipe = json.loads((a.packet / "recipe.json").read_text())
    arms = [v for k, v in recipe.items() if k == "arm"] or [recipe.get("arm")]
    out = {"packet": str(a.packet), "items": len(items), "distinct_item_ids": len({i.item_id for i in items}), "runs": len({i.info["run_key"] for i in items}),
           "tasks": len(tasks), "overlap_with_322": len(tasks & excluded), "excluded_list_size": len(excluded),
           "by_cut": dict(C(i.info["run_key"].rsplit("#c", 1)[-1] for i in items)), "by_seed": dict(C(str(i.info.get("seed")) for i in items)),
           "by_side": dict(C(str(i.info.get("side")) for i in items)), "by_r": dict(C(str(i.info["r"]) for i in items)),
           "with_history_ids": sum(bool(i.info.get("in_context_history_ids")) for i in items), "recipe_teacher_view": json.dumps(arms)[:300],
           "target_tokens": sum(sum(len(x) for x in i.assistant_token_ids) for i in items)}
    from transformers import AutoTokenizer
    from activation.ac_model.ac_model_utils import prepare_training_history
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    sample = random.Random(0).sample(items, min(a.sample, len(items)))
    counts = C()
    for item in sample:
        ids, _, positions = prepare_training_history(tok, item.in_context_messages, item.in_context_start, item.assistant_token_ids, [], tools=item.tools,
                                                     chat_template_kwargs=item.info.get("chat_template_kwargs"), history_ids=item.info.get("in_context_history_ids"))
        first = positions[0] + 1
        prefix, targets = tok.decode(ids[:first]), tok.decode(ids[first:])
        counts["prefix [/Reasoning]"] += prefix.count("[/Reasoning]"); counts["prefix </think>"] += prefix.count("</think>")
        counts["targets [/Reasoning]"] += targets.count("[/Reasoning]"); counts["targets </think>"] += targets.count("</think>")
    out["teacher_view_sample"] = {"items": len(sample), **counts}
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
