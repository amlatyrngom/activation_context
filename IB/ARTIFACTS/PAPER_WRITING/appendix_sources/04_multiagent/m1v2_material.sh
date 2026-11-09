#!/bin/bash
# Container, synthesis fork Oct 16 (M1 v2): pull both nodes' 27B target writing (m1/v2/gen_{a,b}), keep (tqa_r2_keep.py over the M1
# triples: correct, closed, grounded), build the material (make_synth_sft_material_v2.py over the four M1 slims, the same reader
# prompt as evaluation), then add a Musique replay drawn (seeded) from arm B's material (round2/material_bu, whole triples' examples
# with their parts) so that it is 10 % of the final examples. Exclusion check against m1/v2/excluded_tasks_m1.json (1,800 eval ids:
# round 2's 1,200 + dmext + dmhard). Output m1/v2/material (new folder).
set -euo pipefail
P=/workspace/IB/TMP/PAPER_TRAJ_QA; V=$P/m1/v2; L=$V/slims; R=/root/activation_artifacts; N=$P/node_sync
export PYTHONPATH=/workspace/activation:$P/kit:$P/../PAPER_COMPACTION/kit SSH_AUTH_SOCK=
O="-o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=30"
stamp() { echo "$(date -u +%FT%TZ) m1v2_material: $*"; }
for X in a b; do
  mkdir -p $V/gen/$X
  timeout 1800 rsync -a -e "ssh $O" "pc-$X:$R/tqa/m1/v2/gen_$X/gen_*.jsonl" $V/gen/$X/
  stamp "gen $X: $(cat $V/gen/$X/gen_*.jsonl | grep -c .) triples"
done
[ ! -e $V/keep ] && [ ! -e $V/material ] || { stamp "keep or material exists"; exit 2; }
cd $P/kit
python3 tqa_r2_keep.py --triples $V/plan/triples.jsonl --gen $V/gen/a --gen $V/gen/b --out $V/keep 2>&1 | grep -v -E "INFO|Triton|Timeout during" | tail -3
python3 make_synth_sft_material_v2.py --kept $V/keep/kept.jsonl --slim $L/tqa_m1_v2_9b_a/rollouts.jsonl --slim $L/tqa_m1_v2_9b_b/rollouts.jsonl \
  --slim $L/tqa_m1_v2_27b_a/rollouts.jsonl --slim $L/tqa_m1_v2_27b_b/rollouts.jsonl \
  --tools-from $N/sft_readout_v1/ROLLOUTS/tqa_synth_ac_sft245_v1/rollouts.jsonl \
  --check-eval $N/sft_readout_v1/ROLLOUTS/tqa_synth_ac_sft245_v1/rollouts.jsonl --check-slim $N/slim_reserve_v1/ROLLOUTS/tqa_reserve_solve_v1m/rollouts.jsonl \
  --out $V/material 2>&1 | grep -v -E "INFO|Triton|Timeout during" | tail -3
cp $V/excluded_tasks_m1.json $V/material/excluded_tasks.json
python3 - $V/material $P/round2/material_bu <<'PY'
import json, random, sys
from collections import Counter
out, src = sys.argv[1:]
excluded = set(json.load(open(f"{out}/excluded_tasks.json")))
m1 = [json.loads(l) for l in open(f"{out}/examples.jsonl") if l.strip()]
assert not {r["task_id"] for r in m1} & excluded, "an evaluation task in the M1 material"
want = round(len(m1) * 0.1 / 0.9)
pool = [json.loads(l) for l in open(f"{src}/examples.jsonl") if l.strip()]
pool = [r for r in pool if r["benchmark"] == "musique" and r["task_id"] not in excluded]
by_content = {}
for r in pool:
    by_content.setdefault(r["content_id"], []).append(r)
contents = sorted(by_content); random.Random(20261017).shuffle(contents)
chosen, ids = [], set()
for c in contents:
    if len(chosen) >= want:
        break
    chosen += by_content[c]; ids.add(c)
with open(f"{out}/parts.jsonl", "a") as handle:
    for line in open(f"{src}/parts.jsonl"):
        if line.strip() and json.loads(line).get("content_id") in ids:
            handle.write(line if line.endswith("\n") else line + "\n")
m1_ids = {r["id"] for r in m1}
assert not {r["id"] for r in chosen} & m1_ids
with open(f"{out}/examples.jsonl", "a") as handle:
    for r in chosen:
        r["replay"] = "material_bu"; handle.write(json.dumps(r, ensure_ascii=False) + "\n")
meta = json.load(open(f"{out}/material.json"))
meta.update(m1_examples=len(m1), musique_replay_examples=len(chosen), musique_replay_contents=len(ids), replay_source=src,
            final_examples=len(m1) + len(chosen), excluded_ids=len(excluded))
json.dump(meta, open(f"{out}/material.json", "w"), indent=1)
print("M1", len(m1), dict(Counter(f"{r['subtype']}" for r in m1)), "replay", len(chosen), "final", len(m1) + len(chosen))
PY
du -sh $V/material; stamp "done"
