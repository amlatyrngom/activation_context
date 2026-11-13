#!/bin/bash
#SBATCH --job-name=s35_s2eval
#SBATCH --cpus-per-task=8 --mem=96G --time=48:00:00
# Stage-2 side scorer (2026-11-13, user: stage 2 needs intermediate accuracies): stage-2 training leaves no room for an eval on its own GPU,
# so its K-update pairs are scored HERE, on a GPU of their own, as they appear: every pair staged as $W/rq_eval_s2/<RUN>_uNNNNN/{ac,reader}
# (+ KS "100 200") by the host feeder (EC2 runs, via S3) or linked here (the ORCD run), scored on PLAIN BM25 dev1000 (hotpot_tasks_kI_v2)
# at k=100 and k=200 by jobs/rq_eval_job.sh (sec45_eval.py, greedy, cap 1024, batch 32, code v5 = the stage-2 training code).
# Loops every 10 min (rq_eval_job.sh skips finished outputs); also links new ckpt/step_* of the ORCD run T_res_rq9fr_kI.
#   ORCD: cd /home/ngom/orcd/scratch/activation/sec35 && sbatch --export=NIL --partition=mit_preemptable --requeue --gres=gpu:rtx_pro_6000:1 jobs/rq_s2_eval_loop.sh
set -uo pipefail
W=/home/ngom/orcd/scratch/activation/sec35; R=$W/rq_eval_s2; mkdir -p $R
while true; do
  for Q in $W/runs/T_res_rq9fr_kI/sync/ckpt/step_[0-9][0-9][0-9][0-9][0-9]; do [ -f "$Q/meta.json" ] || continue
    U=${Q##*_}; P=$R/T_res_rq9fr_kI_u$U; [ -d $P ] && continue
    cp -al $Q $P.tmp && echo "100 200" > $P.tmp/KS && mv $P.tmp $P && echo "$(date -u +%FT%TZ) linked $P"
  done
  RQ_EVAL_DIR=$R RQ_EVAL_DATA=hotpot_tasks_kI_v2 EVAL_SPLIT=dev1000 SEC45_BUNDLE=sec35_code_v5.tar.gz bash $W/jobs/rq_eval_job.sh || true
  [ -f $R/STOP ] && { echo "STOP file: exiting"; exit 0; }
  sleep 600
done
