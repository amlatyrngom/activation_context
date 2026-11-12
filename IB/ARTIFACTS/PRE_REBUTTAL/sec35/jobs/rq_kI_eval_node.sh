#!/bin/bash
# 100I / 200I read of the 20I finals, no training (pre-rebuttal 2026-11-12, user: "200 AC beating 20 text+lora and maybe 200 text-untuned
# would kinda meet our mvp"). One role per SkyPilot spot node, run as a sky job (scp ec2_prep.sh + this to /root first):
#   .venv/bin/python -m activation.cloud.sky exec NODE -- bash /root/rq_kI_eval_node.sh ac|text [SPLIT]
# Data hotpot_tasks_kI_v1 (sec35/build_kI_sets.py): the 20I dev1000 questions with BM25 top 100 / 200 and both golds injected.
# Scored by jobs/rq_eval_job.sh (the retrieval eval environment: sec45_eval.py greedy cap 1024 batch 32 for AC, text_lora_eval_hf.py
# batch 8 cap 1024 for text), first at k = 100 (SPLIT_100I) then k = 200 (SPLIT_200I); outputs mirrored to S3 runs/kI_eval_<role>/.
#   ac:   T_res_rq9L_20I and T_res_rq9fr_20I final pairs (u5600, 44,800 questions)
#   text: X_rq9_text_20I final adapter (u5600) and the untuned 9B ("base")
set -uo pipefail
ROLE=$1; SPLIT=${2:-dev1000}; W=/root/sec35; J=$W/jobs; S3PY=/root/s3venv/bin/python
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/sec35
bash /root/ec2_prep.sh || { echo "prep failed"; exit 2; }
X() { $S3PY $W/s3x.py "$@"; }
X getdir $S3/data/hotpot_tasks_kI_v1/ $W/data/hotpot_tasks_kI_v1/ || exit 2
R=$W/rq_eval_kI_$ROLE; mkdir -p $R
case $ROLE in
ac)
  for N in T_res_rq9L_20I T_res_rq9fr_20I; do P=$R/${N}_u05600; [ -f $P/meta.json ] || { rm -rf $P; X getdir $S3/runs/$N/sync/ckpt/final/ $P/ || exit 2; }; done;;
text)
  P=$R/text/X_rq9_text_20I_u05600; [ -f $P/adapter_config.json ] || { rm -rf $P; X getdir $S3/runs/X_rq9_text_20I/ckpt/step_05600/ $P/ || exit 2; }
  mkdir -p $R/text/base;;
*) echo "role ac|text"; exit 2;;
esac
FAIL=0
for k in 100 200; do
  for P in $R/*_u?????/ $R/text/*/; do [ -d "$P" ] && echo $k > $P/KS; done
  RQ_EVAL_DIR=$R RQ_EVAL_DATA=hotpot_tasks_kI_v1 EVAL_SPLIT=${SPLIT}_${k}I bash $J/rq_eval_job.sh || FAIL=1
  X putdir $R/out/ $S3/runs/kI_eval_$ROLE/out/; X putdir $R/logs/ $S3/runs/kI_eval_$ROLE/logs/
done
echo "kI eval $ROLE done (fail=$FAIL)"; exit $FAIL
