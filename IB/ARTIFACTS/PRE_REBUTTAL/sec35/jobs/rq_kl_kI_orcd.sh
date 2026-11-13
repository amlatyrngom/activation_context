#!/bin/bash
#SBATCH --job-name=s35_kl_kI
#SBATCH --cpus-per-task=8 --mem=96G --time=24:00:00
# 2026-11-13: the golds-injected reads of the KL finals, to set beside the AC SFT finals (LoRA 61.5 / 56.1, frozen 63.7 / 59.3) and text
# (untuned 65.6 / 64.9, tuned 75.9 / 76.5): 100I at k=100 and 200I at k=200, dev1000 of hotpot_tasks_kI_v2. Pairs hard-linked (ac, reader,
# meta.json only: KS is written per pass here) from the side scorer's staging $W/rq_eval_s2. Scored by jobs/rq_eval_job.sh (sec45_eval.py,
# greedy, cap 1024, batch 32, code v5); finished outputs are skipped, so a requeue resumes.
#   ORCD: cd /home/ngom/orcd/scratch/activation/sec35 && sbatch --export=NIL --partition=mit_preemptable --requeue --gres=gpu:rtx_pro_6000:1 jobs/rq_kl_kI_orcd.sh
set -uo pipefail
W=/home/ngom/orcd/scratch/activation/sec35; S=$W/rq_eval_s2; R=$W/rq_eval_kI_ac; mkdir -p $R; FAIL=0
for N in T_res_rq9L_20I_KL_u05600 T_res_rq9fr_20I_KL_u05600 T_res_rq9L_QA_KL_u12500; do
  [ -d $R/$N ] && continue
  [ -f $S/$N/meta.json ] || { echo "missing $S/$N"; FAIL=1; continue; }
  mkdir -p $R/$N.tmp && cp -al $S/$N/ac $S/$N/reader $S/$N/meta.json $R/$N.tmp/ && mv $R/$N.tmp $R/$N
done
for pass in dev1000_100I:100 dev1000_200I:200; do IFS=: read SP k <<< "$pass"
  for P in $R/*_u[0-9][0-9][0-9][0-9][0-9]/; do echo $k > $P/KS; done
  RQ_EVAL_DIR=$R RQ_EVAL_DATA=hotpot_tasks_kI_v2 EVAL_SPLIT=$SP SEC45_BUNDLE=sec35_code_v5.tar.gz bash $W/jobs/rq_eval_job.sh || FAIL=1
done
echo "KL kI passes done (fail=$FAIL)"; exit $FAIL
