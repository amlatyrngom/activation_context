"""
Audit of a stage-2 packet (2026-11-18; CPU, this host), independent of the builder. A stage-2 item = a stage-1 c20 per-ratio item of the
same prefix (base: packets/s1kl_tv_r{5,10}_c20_s012_v1 of compaction_stage1, item <family>:<task>#s<g>#c20000:0) with the continuation
replaced by the selected one (material line run key <task>#s<g>#c20000#<sample>). Checks:
  - no item of an excluded task (every --exclude file: {"task_ids": [...]} or an index {"prefixes": [{"task_id"}]})
  - every item at r = R; one item per material line (ids missing / extra listed)
  - the prefix side equals the base item's: student view up to its first target (ac_messages[:ac_start], the AC part included),
    teacher view up to the continuation (in_context_messages[:in_context_start]), tools, info in_context_history_ids, the size fields
  - targets = the material line's target_token_ids; the continuation messages = the line's (up to its last assistant turn)
  - every other differing field path is counted; allowed: item_id, the continuation side and the info fields that describe it
    (run_key, solved, finish_reason, assistant_turns, target_tokens, reader_tokens: the reader length grows or shrinks with the continuation)
  - recipe.json: loss sft, encoder 5e-5, reader 0, warm-up 50
  - boundary samples (decoded): for one item per sample kind, the end of the teacher-view prompt and the start of the first target

    python audit_stage2_packet.py --packet PACKET --material MATERIAL --r 0.2 --exclude A.json B.json ... [--base BASE] [--out OUT.json]
"""
import argparse
import collections
import json
import pickle
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage1"))
from audit_ratio_packet import diff, fields  # noqa: E402

T1 = HERE.parents[4] / "IB/TMP/PRE_REBUTTAL/compaction_stage1/packets"
CONTINUATION_INFO = {"run_key", "solved", "finish_reason", "assistant_turns", "target_tokens", "reader_tokens"}


def task_ids(path: Path) -> set:
    d = json.loads(path.read_text())
    return set(d["task_ids"]) if "task_ids" in d else {e["task_id"] for e in d["prefixes"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packet", type=Path, required=True); p.add_argument("--material", type=Path, required=True)
    p.add_argument("--r", type=float, required=True); p.add_argument("--exclude", type=Path, nargs="+", required=True)
    p.add_argument("--base", type=Path); p.add_argument("--out", type=Path)
    a = p.parse_args()
    base_dir = a.base or T1 / f"s1kl_tv_{ {0.2: 'r5', 0.1: 'r10'}[a.r] }_c20_s012_v1"
    load = lambda d: list(pickle.load(open(d / "items.pkl", "rb"))["train"])
    items = load(a.packet)
    base = {i.info["run_key"]: i for i in load(base_dir)}
    lines = {}
    with a.material.open() as handle:
        for raw in handle:
            line = json.loads(raw)
            lines[line["run_key"]] = line
    blocked = set().union(*(task_ids(f) for f in a.exclude))
    paths, findings, kinds = collections.Counter(), collections.Counter(), collections.Counter()
    boundary, seen_kind = [], set()
    tokenizer = None
    for item in items:
        key = item.info["run_key"]; line = lines.get(key); prefix_key = key.rsplit("#", 1)[0]; b = base.get(prefix_key)
        kinds[key.rsplit("#", 1)[1]] += 1
        if line is None:
            findings["item without a material line"] += 1; continue
        if b is None:
            findings["no base item of the same prefix"] += 1; continue
        if item.ac_start != b.ac_start or item.ac_messages[:item.ac_start] != b.ac_messages[:b.ac_start]:
            findings["student view before the first target differs from the base"] += 1
        if item.in_context_start != b.in_context_start or item.in_context_messages[:item.in_context_start] != b.in_context_messages[:b.in_context_start]:
            findings["teacher view before the continuation differs from the base"] += 1
        if item.tools != b.tools or item.info.get("in_context_history_ids") != b.info.get("in_context_history_ids"):
            findings["tools or prefix history ids differ from the base"] += 1
        if [list(t) for t in item.assistant_token_ids] != line["target_token_ids"]:
            findings["targets differ from the material line"] += 1
        messages = line["continuation_messages"]
        last = max(i for i, m in enumerate(messages) if m.get("role") == "assistant")
        if item.ac_messages[item.ac_start:] != messages[:last + 1] or item.in_context_messages[item.in_context_start:] != messages[:last + 1]:
            findings["continuation messages differ from the material line"] += 1
        found = set()
        diff(fields(item), fields(b), "", found)
        for f in found:
            paths[f] += 1
        kind = key.rsplit("#", 1)[1]
        if kind not in seen_kind or (kind != "orig" and len([x for x in boundary if x["kind"] == kind]) < 2):
            seen_kind.add(kind)
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
            text = tokenizer.apply_chat_template(item.in_context_messages[:item.in_context_start], tools=item.tools or None, add_generation_prompt=True,
                                                 tokenize=False, **item.info["chat_template_kwargs"])
            boundary.append({"item": item.item_id, "kind": kind, "prompt_end": text[-260:], "first_target_start": tokenizer.decode(item.assistant_token_ids[0][:60]),
                             "prefix_last_role": item.in_context_messages[item.in_context_start - 1]["role"]})
    allowed = lambda f: (f == ".item_id" or f.startswith(".assistant_token_ids") or f.startswith(".in_context_messages") or f.startswith(".ac_messages")
                         or (f.startswith(".info.") and f.split(".")[2].split("[")[0].split("#")[0] in CONTINUATION_INFO))
    recipe = json.loads((a.packet / "recipe.json").read_text())
    arm = recipe.get("arm", {})
    recipe_ok = arm.get("loss") == "sft" and arm.get("lr_ac") == 5e-5 and arm.get("lr_reader") == 0.0 and arm.get("train_overrides", {}).get("warmup_updates") == 50
    ids = {i.info["run_key"] for i in items}
    out = {"packet": str(a.packet), "material": str(a.material), "base": str(base_dir), "r": a.r, "exclude": [str(f) for f in a.exclude],
           "excluded_task_ids": len(blocked), "items": len(items), "tasks": len({i.info["task_id"] for i in items}), "runs": len(ids),
           "items_by_sample": dict(kinds), "overlap_tasks_with_exclude": len({i.info["task_id"] for i in items} & blocked),
           "r_counts": dict(collections.Counter(str(i.info["r"]) for i in items)),
           "material_lines": len(lines), "lines_without_item": len(set(lines) - ids), "lines_without_item_examples": sorted(set(lines) - ids)[:10],
           "findings": dict(findings), "differing_paths": dict(paths.most_common()), "unexpected_paths": {f: n for f, n in paths.items() if not allowed(f)},
           "with_history_ids": sum(bool(i.info.get("in_context_history_ids")) for i in items),
           "target_tokens": sum(sum(len(t) for t in i.assistant_token_ids) for i in items),
           "recipe": {"loss": arm.get("loss"), "lr_ac": arm.get("lr_ac"), "lr_reader": arm.get("lr_reader"), "train_overrides": arm.get("train_overrides"), "ok": recipe_ok},
           "boundary_samples": boundary}
    out["ok"] = (out["overlap_tasks_with_exclude"] == 0 and set(out["r_counts"]) == {str(a.r)} and not findings and not out["unexpected_paths"]
                 and out["with_history_ids"] == len(items) and recipe_ok)
    print(json.dumps({k: v for k, v in out.items() if k != "boundary_samples"}, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
