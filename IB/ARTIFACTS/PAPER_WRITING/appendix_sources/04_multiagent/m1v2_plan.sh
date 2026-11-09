#!/bin/bash
# Container, synthesis fork Oct 16 (M1 v2): after both nodes' phase 1 (m1/v2/phase1_done_{a,b}): pull the slim 9B and 27B solver
# caches and the 27B samples, plan the triples (tqa_m1_plan.py: all 3-subsets with <= 1 27B run, <= 3 per task with a 27B run,
# 2 agent orders per mixed triple, unanimous share 0.25, eval ids excluded), report what the 27B added, then push triples.jsonl to
# both nodes (new folder m1/v2/triples), md5-check, and touch triples.jsonl.ready (the m1v2_gen.sh queues start on it).
set -euo pipefail
P=/workspace/IB/TMP/PAPER_TRAJ_QA; V=$P/m1/v2; L=$V/slims; R=/root/activation_artifacts
export PYTHONPATH=/workspace/activation:$P/kit:$P/../PAPER_COMPACTION/kit SSH_AUTH_SOCK=
O="-o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=30"
stamp() { echo "$(date -u +%FT%TZ) m1v2_plan: $*"; }
mkdir -p $L
for X in a b; do
  H=pc-$X
  timeout 60 ssh -n $O $H "test -e $R/tqa/m1/v2/phase1_done_$X" || { stamp "$H phase1_done_$X missing"; exit 1; }
  for ID in tqa_m1_v2_9b_$X tqa_m1_v2_27b_$X; do
    mkdir -p $L/$ID
    timeout 1800 rsync -a -e "ssh $O" $H:$R/tqa/slim_m1/ROLLOUTS/$ID/rollouts.jsonl $L/$ID/rollouts.jsonl
    timeout 300 rsync -a -e "ssh $O" $H:$R/SYNC/PAPER_TRAJ_QA/$ID/sample.json $L/$ID/sample.json
    stamp "$ID: $(grep -c . $L/$ID/rollouts.jsonl) slim rows"
  done
done
SOLVES=(); for X in a b; do for ID in tqa_m1_v2_9b_$X tqa_m1_v2_27b_$X; do SOLVES+=(--solves $L/$ID/rollouts.jsonl:$L/$ID/sample.json); done; done
[ ! -e $V/plan ] || { stamp "$V/plan exists"; exit 2; }
cd $P/kit
python3 tqa_m1_plan.py "${SOLVES[@]}" --out $V/plan --benchmarks deepmath --orders 2 --unanimous-share 0.25 --per-task-27b 3 \
  --exclude-tasks $V/excluded_tasks_m1.json --seed 20261016 2>&1 | grep -v -E "INFO|Triton|Timeout during"
python3 - $L $V <<'PY' 2>&1 | grep -v -E "INFO|Triton|Timeout during"
import json, sys
from collections import Counter, defaultdict
from score_synth1call import score
from tqa_sft_prep_runs import solver_result
L, V = sys.argv[1:]
right = defaultdict(dict)
for X in "ab":
    for model in ("9b", "27b"):
        for line in open(f"{L}/tqa_m1_v2_{model}_{X}/rollouts.jsonl"):
            row = json.loads(line); task = row["agent_config"]["dataset_task"]
            right[task["task_id"]][int(row["seed"])] = score(task, solver_result(row, 40_000)) >= 1.0
k9 = {t: sum(v for s, v in r.items() if s < 3) for t, r in right.items()}
big = {t: [v for s, v in r.items() if s >= 3] for t, r in right.items()}
out = {"tasks": len(right), "k9_hist": dict(sorted(Counter(k9.values()).items())),
       "27b_ran_on": sum(1 for t in big if big[t]), "27b_runs": sum(len(v) for v in big.values()), "27b_runs_right": sum(sum(v) for v in big.values()),
       "k0_tasks_with_27b_right (unusable -> usable)": sum(1 for t in big if k9[t] == 0 and any(big[t])),
       "k0_tasks_27b_ran": sum(1 for t in big if k9[t] == 0 and big[t]),
       "k1_tasks_with_27b_right (more minority-right)": sum(1 for t in big if k9[t] == 1 and any(big[t])),
       "k1_tasks_27b_ran": sum(1 for t in big if k9[t] == 1 and big[t])}
plan = json.load(open(f"{V}/plan/plan.json"))
out["triples_by_type"] = plan["triples_by_type"]; out["triples_total"] = sum(plan["triples_by_type"].values())
out["triple_tasks"] = len({json.loads(l)["task_id"] for l in open(f"{V}/plan/triples.jsonl")})
json.dump(out, open(f"{V}/plan_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
PY
M=$(md5sum < $V/plan/triples.jsonl | cut -d' ' -f1); stamp "triples md5 $M, $(du -h $V/plan/triples.jsonl | cut -f1)"
for H in pc-a pc-b; do
  timeout 60 ssh -n $O $H "test ! -e $R/tqa/m1/v2/triples && mkdir -p $R/tqa/m1/v2/triples" || { stamp "$H: m1/v2/triples exists"; exit 2; }
  timeout 1800 rsync -a -e "ssh $O" $V/plan/triples.jsonl $H:$R/tqa/m1/v2/triples/triples.jsonl
  [ "$(timeout 300 ssh -n $O $H "md5sum < $R/tqa/m1/v2/triples/triples.jsonl | cut -d' ' -f1")" = "$M" ] || { stamp "$H md5 mismatch"; exit 1; }
done
for H in pc-a pc-b; do timeout 60 ssh -n $O $H "touch $R/tqa/m1/v2/triples/triples.jsonl.ready" && stamp "$H: triples.jsonl.ready"; done
stamp "done"
