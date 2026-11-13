#!/bin/bash
#SBATCH --job-name=s35_e4_kI
#SBATCH --cpus-per-task=8 --mem=96G --time=24:00:00
# 2026-11-13 (user: hill-climb on the ideal-retrieval gap to text + LoRA first; hypothesis check: encoder capacity): zero-training read of the
# 4B-encoder SFT 20I final T_res_rq9e4_e2 (20I dev1000 75.3 vs 2B 72.3) on 100I (k=100), 200I (k=200) and plain BM25 k = 20 / 100 / 200,
# dev1000 of hotpot_tasks_kI_v2, to set beside the 2B SFT final (61.5 / 56.1; plain 52.0 / 51.9 / 49.8). Scored by jobs/rq_eval_job.sh
# with RQ_ENCODER=Qwen/Qwen3.5-4B (batch 32, greedy, cap 1024, code v5); finished outputs are skipped, so a requeue resumes.
#   ORCD: cd /home/ngom/orcd/scratch/activation/sec35 && sbatch --export=NIL --partition=mit_preemptable --requeue --gres=gpu:rtx_pro_6000:1 jobs/rq_e4_kI_orcd.sh
set -uo pipefail
W=/home/ngom/orcd/scratch/activation/sec35; R=$W/rq_eval_e4; F=$W/runs/T_res_rq9e4_e2/sync/ckpt/final; P=$R/T_res_rq9e4_e2_u05600; mkdir -p $R; FAIL=0
[ -f $F/meta.json ] || { echo "missing $F"; exit 2; }
[ -d $P ] || { mkdir -p $P.tmp && cp -al $F/ac $F/reader $F/meta.json $P.tmp/ && mv $P.tmp $P; }
for pass in dev1000_100I:100 dev1000_200I:200 dev1000:"100 200 20"; do SP=${pass%%:*}; echo "${pass#*:}" > $P/KS
  RQ_ENCODER=Qwen/Qwen3.5-4B RQ_EVAL_DIR=$R RQ_EVAL_DATA=hotpot_tasks_kI_v2 EVAL_SPLIT=$SP SEC45_BUNDLE=sec35_code_v5.tar.gz bash $W/jobs/rq_eval_job.sh || FAIL=1
done
echo "e4 kI passes done (fail=$FAIL)"; exit $FAIL
