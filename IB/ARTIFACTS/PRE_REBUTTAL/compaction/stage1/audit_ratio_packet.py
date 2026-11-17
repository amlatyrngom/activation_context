"""
Audit of a per-ratio stage-1 packet against its base (2026-11-17; CPU, this host). A per-ratio packet = the base packet's items (run M's
s1kl_tv_c162024_s012_v1) minus the excluded tasks, every item at ONE r. Checks, independent of the builder and the merge:
  - no item of an excluded task (EXCLUDE json {"task_ids": [...]}: the 322 + dev206 = 528), counted at task level
  - every item carries r = R (and nothing else)
  - item ids: the per-ratio packet's ids = the base's ids minus the excluded tasks' (missing / extra ids listed)
  - per common item, every field compared with the base item; the differing paths are counted by name. Allowed: the size fields
    (info r / ratio / rows_wanted / rows / clamped and whatever the example scan derives from rows, e.g. reader_tokens) and the AC part's
    compression target inside ac_messages (content[].compression_target). Anything else differing (targets, teacher view, history ids, continuation) is a finding.

    python audit_ratio_packet.py --packet PER_RATIO --base BASE --r 0.2 --exclude EXCLUDE.json [--out OUT.json]
"""
import argparse
import collections
import dataclasses
import json
import pickle
import re
from pathlib import Path

SIZE_INFO = {"r", "ratio", "rows_wanted", "rows", "clamped", "reader_tokens"}


def diff(a, b, path, out):
    """Paths (indices replaced by []) where a and b differ."""
    if type(a) is not type(b):
        out.add(path); return
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.add(f"{path}.{k}")
            else:
                diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.add(f"{path}#len"); return
        for x, y in zip(a, b):
            diff(x, y, f"{path}[]", out)
    elif a != b:
        out.add(path)


def fields(item):
    return {f.name: getattr(item, f.name) for f in dataclasses.fields(item)} if dataclasses.is_dataclass(item) else dict(vars(item))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--packet", type=Path, required=True); p.add_argument("--base", type=Path, required=True)
    p.add_argument("--r", type=float, required=True); p.add_argument("--exclude", type=Path, required=True); p.add_argument("--out", type=Path)
    a = p.parse_args()
    load = lambda d: list(pickle.load(open(d / "items.pkl", "rb"))["train"])
    items, base = load(a.packet), load(a.base)
    blocked = set(json.loads(a.exclude.read_text())["task_ids"])
    base_kept = {i.item_id: i for i in base if i.info["task_id"] not in blocked}
    new = {i.item_id: i for i in items}
    paths, value_paths = collections.Counter(), collections.Counter()
    for item_id in sorted(set(new) & set(base_kept)):
        x, y = fields(new[item_id]), fields(base_kept[item_id])
        found = set()
        diff(x, y, "", found)
        for f in found:
            paths[re.sub(r"\.(\w+)", r".\1", f)] += 1
    allowed = lambda f: (f.startswith(".info.") and f.split(".")[2].split("[")[0] in SIZE_INFO) or (f.startswith(".ac_messages") and f.rsplit(".", 1)[-1] in ("compression_target", "compression_ratio", "ac_compaction_ratio"))
    out = {"packet": str(a.packet), "base": str(a.base), "r": a.r, "exclude": str(a.exclude), "excluded_task_ids": len(blocked),
           "items": len(items), "tasks": len({i.info["task_id"] for i in items}), "runs": len({i.info["run_key"] for i in items}),
           "overlap_tasks_with_exclude": len({i.info["task_id"] for i in items} & blocked),
           "r_counts": dict(collections.Counter(str(i.info["r"]) for i in items)),
           "base_items": len(base), "base_items_after_exclusion": len(base_kept), "base_tasks_dropped": len({i.info["task_id"] for i in base} & blocked),
           "ids_missing_vs_base": len(set(base_kept) - set(new)), "ids_extra_vs_base": len(set(new) - set(base_kept)),
           "missing_examples": sorted(set(base_kept) - set(new))[:10], "extra_examples": sorted(set(new) - set(base_kept))[:10],
           "common_items": len(set(new) & set(base_kept)), "differing_paths": dict(paths.most_common()),
           "unexpected_paths": {f: n for f, n in paths.items() if not allowed(f)},
           "with_history_ids": sum(bool(i.info.get("in_context_history_ids")) for i in items),
           "target_tokens": sum(sum(len(t) for t in i.assistant_token_ids) for i in items)}
    out["ok"] = (out["overlap_tasks_with_exclude"] == 0 and set(out["r_counts"]) == {str(a.r)} and not out["unexpected_paths"]
                 and out["ids_extra_vs_base"] == 0 and out["with_history_ids"] == len(items))
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
