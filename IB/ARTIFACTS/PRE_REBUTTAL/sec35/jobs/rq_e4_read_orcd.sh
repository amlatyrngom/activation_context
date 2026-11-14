#!/bin/bash
#SBATCH --job-name=s35_e4_read
#SBATCH --cpus-per-task=8 --mem=96G --time=24:00:00
# 2026-11-14: zero-training read of one 4B-encoder pair (folder $W/rq_eval_e4x/<PAIR>/<PAIR>: one eval dir per pair, so two such jobs never share KS files; e.g. T_res_rq9e4fr_20I_KL_u05600) on 100I (k=100),
# 200I (k=200) and plain BM25 k = 100 / 200 / 20 (dev1000 of hotpot_tasks_kI_v2), as jobs/rq_e4_kI_orcd.sh did for T_res_rq9e4_e2.
# Scored by jobs/rq_eval_job.sh with RQ_ENCODER=Qwen/Qwen3.5-4B (batch 32, greedy, cap 1024, code v5); finished outputs are skipped (requeue resumes).
#   ORCD: cd /home/ngom/orcd/scratch/activation/sec35 && sbatch --export=NIL --time=24:00:00 --partition=mit_preemptable --requeue --gres=gpu:rtx_pro_6000:1 jobs/rq_e4_read_orcd.sh PAIR
set -uo pipefail
PAIR=${1:?pair folder name under rq_eval_e4}
W=/home/ngom/orcd/scratch/activation/sec35; R=$W/rq_eval_e4x/$PAIR; P=$R/$PAIR; FAIL=0
[ -f $P/meta.json ] && [ -f $P/ac/ac_modules.pt ] || { echo "incomplete pair $P"; exit 2; }
for pass in dev1000_100I:100 dev1000_200I:200 dev1000:"100 200 20"; do SP=${pass%%:*}; echo "${pass#*:}" > $P/KS
  RQ_ENCODER=Qwen/Qwen3.5-4B RQ_EVAL_DIR=$R RQ_EVAL_DATA=hotpot_tasks_kI_v2 EVAL_SPLIT=$SP SEC45_BUNDLE=sec35_code_v5.tar.gz bash $W/jobs/rq_eval_job.sh || FAIL=1
done
echo "e4 read of $PAIR done (fail=$FAIL)"; exit $FAIL
