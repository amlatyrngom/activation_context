#!/bin/bash
#SBATCH --job-name=s35_fin
#SBATCH --cpus-per-task=8 --mem=96G --time=48:00:00
# Stage-2 ideal-retrieval reads (2026-11-13, user: compare AC with untuned text under ideal retrieval, and on plain BM25 at k=100/200 with
# text at k=20). The side scorer (rq_s2_eval_loop.sh) reads every stage-2 pair on plain BM25 k=100/200 only; this loop adds, for
#   - the stage-2 midpoints (u01200 of each kI arm, hard-linked from $W/rq_eval_s2): 100I k=100, 200I k=200, 20I;
#   - the stage-2 finals (u02400: EC2 finals sent by the host's jobs/rq_fin_feed.sh, the ORCD arm T_res_rq9fr_kI linked here when DONE)
#     and the text stage-2 final (text/X_rq9_text_t2_u02400): 100I, 200I, 20I, and plain BM25 k = 20 / 100 / 200.
# 100I / 200I / plain: hotpot_tasks_kI_v2 dev1000 in $W/rq_eval_fin; 20I: hotpot_tasks_v2 dev1000 in $W/rq_eval_fin20i (hard-linked pairs).
# Scored by jobs/rq_eval_job.sh (AC: sec45_eval.py batch 32; text: text_lora_eval_hf.py batch 8; greedy, cap 1024; code v5); finished
# outputs are skipped, so a requeue resumes. Loops every 15 min; exits on $W/rq_eval_fin/STOP.
#   ORCD: cd /home/ngom/orcd/scratch/activation/sec35 && sbatch --export=NIL --partition=mit_preemptable --requeue --gres=gpu:rtx_pro_6000:1 jobs/rq_fin_loop.sh
set -uo pipefail
W=/home/ngom/orcd/scratch/activation/sec35; S=$W/rq_eval_s2; R=$W/rq_eval_fin; R20=$W/rq_eval_fin20i; mkdir -p $R/text $R20/text
link() { [ -d $2 ] && return 0; mkdir -p $2.tmp && cp -al $1/ac $1/reader $1/meta.json $2.tmp/ && mv $2.tmp $2 && echo "$(date -u +%FT%TZ) linked $2"; }
ks() { for P in $R/*_u[0-9][0-9][0-9][0-9][0-9]/; do [ -d "$P/ac" ] || continue; case $P in *_u02400/) echo "$2" > $P/KS;; *) echo "$1" > $P/KS;; esac; done
       for P in $R/text/*/; do [ -d "$P" ] && echo "$2" > $P/KS; done; }
ev() { RQ_EVAL_DIR=$1 RQ_EVAL_DATA=$2 EVAL_SPLIT=$3 SEC45_BUNDLE=sec35_code_v5.tar.gz bash $W/jobs/rq_eval_job.sh || true; }
while true; do
  for Q in $S/T_res_rq9L_kI_u01200 $S/T_res_rq9fr_kI_u01200 $S/T_res_rq9L_kI_KL_u01200 $S/T_res_rq9fr_kI_KL_u01200; do
    [ -f $Q/meta.json ] && link $Q $R/$(basename $Q); done
  F=$W/runs/T_res_rq9fr_kI; [ -f $F/DONE ] && [ -f $F/sync/ckpt/final/meta.json ] && link $F/sync/ckpt/final $R/T_res_rq9fr_kI_u02400
  for P in $R/*_u[0-9][0-9][0-9][0-9][0-9]/; do [ -d "$P/ac" ] || continue; link ${P%/} $R20/$(basename $P); echo 20 > $R20/$(basename $P)/KS; done
  for P in $R/text/*/; do [ -d "$P" ] || continue; N=$(basename $P); [ -d $R20/text/$N ] || { cp -al ${P%/} $R20/text/$N.tmp && rm -f $R20/text/$N.tmp/KS && mv $R20/text/$N.tmp $R20/text/$N; }; echo 20 > $R20/text/$N/KS; done
  ks 100 100;  ev $R hotpot_tasks_kI_v2 dev1000_100I
  ks 200 200;  ev $R hotpot_tasks_kI_v2 dev1000_200I
  ev $R20 hotpot_tasks_v2 dev1000
  ks "" "20 100 200"; ev $R hotpot_tasks_kI_v2 dev1000
  [ -f $R/STOP ] && { echo "STOP file: exiting"; exit 0; }
  sleep 900
done
