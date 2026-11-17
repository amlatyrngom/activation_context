"""Dev-206 (2026-11-17): every qualifying seed-0 evaluation-side run of pc_pool9_h6_eval outside the 322 (kit/exclude_h7.json),
cut with kit/cut_prefixes.py (eval-200 rule, 20k-50k). Writes dev206_tasks.json and exclude_h7_dev206.json (the 322 + these 206 =
the task-level exclusion for every training packet built from now on).
    python3 -I make_dev206.py PREFIXES_DIR OUT_DIR"""
import json, sys
B = "/home/ubuntu/activation_context/IB/TMP/PAPER_COMPACTION/"
idx = json.load(open(sys.argv[1] + "/index.json"))
excl = json.load(open(B + "kit/exclude_h7.json"))["task_ids"]
dev = sorted(p["task_id"] for p in idx["prefixes"])
assert len(dev) == 206 and not set(dev) & set(excl)
json.dump({"name": "dev206", "source": "pc_pool9_h6_eval seed 0, qualifying for the 20k cut, outside exclude_h7", "task_ids": dev},
          open(sys.argv[2] + "/dev206_tasks.json", "w"), indent=1)
json.dump({"rule": "task level, all seeds: exclude_h7 (eval-200, dev-75, frozen-100) + dev206", "task_ids": sorted(set(excl) | set(dev))},
          open(sys.argv[2] + "/exclude_h7_dev206.json", "w"), indent=1)
print(len(dev), len(set(excl) | set(dev)))
