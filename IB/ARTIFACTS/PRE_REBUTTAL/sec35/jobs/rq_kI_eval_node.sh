#!/bin/bash
# 100I / 200I read of the 20I finals, no training (pre-rebuttal 2026-11-12, user: "200 AC beating 20 text+lora and maybe 200 text-untuned
# would kinda meet our mvp"). One role per SkyPilot spot node, run as a sky job (scp ec2_prep.sh + this to /root first):
#   .venv/bin/python -m activation.cloud.sky exec NODE -- bash /root/rq_kI_eval_node.sh ac|text [SPLIT]
# User MVP (2026-11-12, corrected): TRUE BM25 (no injection) AC at k = 100 / 200 should match or beat text at k = 20 (surely untuned,
# preferably tuned). Passes DATA:SPLIT:K, in order: plain BM25 hotpot_tasks_bm25e_v1 (sec35/build_bm25_eval.py; the 20I dev1000
# questions, BM25 top 200, no injection) at the MVP k first, then 100I / 200I hotpot_tasks_kI_v1 (sec35/build_kI_sets.py, golds injected).
# Scored by jobs/rq_eval_job.sh (sec45_eval.py greedy cap 1024 batch 32 for AC, text_lora_eval_hf.py batch 8 cap 1024 for text);
# outputs mirrored to S3 runs/kI_eval_<role>/ after every pass.
#   ac:   T_res_rq9L_20I and T_res_rq9fr_20I final pairs (u5600, 44,800 questions)
#   text: X_rq9_text_20I final adapter (u5600) and the untuned 9B ("base")
set -uo pipefail
ROLE=$1; SPLIT=${2:-dev1000}; W=/root/sec35; J=$W/jobs; S3PY=/root/s3venv/bin/python
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/sec35
bash /root/ec2_prep.sh || { echo "prep failed"; exit 2; }
X() { $S3PY $W/s3x.py "$@"; }
export SEC45_PY=/opt/activation/.venv/bin/python   # the EC2 venv (ec2_prep.sh, ec2_run.sh); rq_eval_job.sh's EC2 default path does not exist
for d in hotpot_tasks_kI_v1 hotpot_tasks_bm25e_v1; do X getdir $S3/data/$d/ $W/data/$d/ || exit 2; done
R=$W/rq_eval_kI_$ROLE; mkdir -p $R
case $ROLE in
ac)
  for N in T_res_rq9L_20I T_res_rq9fr_20I; do P=$R/${N}_u05600; [ -f $P/meta.json ] || { rm -rf $P; X getdir $S3/runs/$N/sync/ckpt/final/ $P/ || exit 2; }; done;;
text)
  P=$R/text/X_rq9_text_20I_u05600; [ -f $P/adapter_config.json ] || { rm -rf $P; X getdir $S3/runs/X_rq9_text_20I/ckpt/step_05600/ $P/ || exit 2; }
  mkdir -p $R/text/base;;
*) echo "role ac|text"; exit 2;;
esac
case $ROLE in
ac)   PASSES="bm25e:$SPLIT:100 bm25e:$SPLIT:200 bm25e:$SPLIT:20 kI:${SPLIT}_100I:100 kI:${SPLIT}_200I:200";;
text) PASSES="bm25e:$SPLIT:20 bm25e:$SPLIT:100 kI:${SPLIT}_100I:100 bm25e:$SPLIT:200 kI:${SPLIT}_200I:200";;
esac
FAIL=0
for pass in $PASSES; do IFS=: read D SP k <<< "$pass"; DD=hotpot_tasks_${D}_v1; [ $D = bm25e ] && DD=hotpot_tasks_bm25e_v1
  for P in $R/*_u?????/ $R/text/*/; do [ -d "$P" ] && echo $k > $P/KS; done
  RQ_EVAL_DIR=$R RQ_EVAL_DATA=$DD EVAL_SPLIT=$SP bash $J/rq_eval_job.sh || FAIL=1
  X putdir $R/out/ $S3/runs/kI_eval_$ROLE/out/; X putdir $R/logs/ $S3/runs/kI_eval_$ROLE/logs/
done
echo "kI eval $ROLE done (fail=$FAIL)"; exit $FAIL
