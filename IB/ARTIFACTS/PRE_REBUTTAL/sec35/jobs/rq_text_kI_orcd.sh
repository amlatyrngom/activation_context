#!/bin/bash
#SBATCH --job-name=s35_text_kI
#SBATCH --cpus-per-task=8 --mem=96G --time=24:00:00
# The text passes deferred from the rql read (2026-11-12, rql taken for stage 2): untuned 9B + X_rq9_text_20I u5600 on dev1000 plain BM25
# k=200, 100I (k=100), 200I (k=200), from hotpot_tasks_kI_v2 (plain dev1000 + the injected sets). Scored by jobs/rq_eval_job.sh (text:
# text_lora_eval_hf.py, batch 8, cap 1024). Staged: $W/rq_eval_kI_text/text/{X_rq9_text_20I_u05600,base}.
#   ORCD: cd /home/ngom/orcd/scratch/activation/sec35 && sbatch --export=NIL --partition=mit_preemptable --requeue --gres=gpu:rtx_pro_6000:1 jobs/rq_text_kI_orcd.sh
set -uo pipefail
W=/home/ngom/orcd/scratch/activation/sec35; R=$W/rq_eval_kI_text; FAIL=0
for pass in dev1000:200 dev1000_100I:100 dev1000_200I:200; do IFS=: read SP k <<< "$pass"
  for P in $R/text/*/; do echo $k > $P/KS; done
  RQ_EVAL_DIR=$R RQ_EVAL_DATA=hotpot_tasks_kI_v2 EVAL_SPLIT=$SP bash $W/jobs/rq_eval_job.sh || FAIL=1
done
echo "text kI passes done (fail=$FAIL)"; exit $FAIL
