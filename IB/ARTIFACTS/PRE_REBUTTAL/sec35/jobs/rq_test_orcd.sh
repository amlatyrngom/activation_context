#!/bin/bash
#SBATCH --job-name=s35_test
#SBATCH --cpus-per-task=8 --mem=96G --time=24:00:00
# §3.1 reported numbers on the HELD-OUT test1000 (2026-11-14, user: "do the real test run"; dev1000 was used for every choice). One system per
# job, its own eval dirs ($W/rq_eval_test_<WHAT>{,_20i}), all six settings of the figure:
#   20I = hotpot_tasks_v2 test1000 (k=20, golds injected); 100I / 200I = hotpot_tasks_kI_v2 test1000_100I (k=100) / test1000_200I (k=200);
#   plain BM25 = hotpot_tasks_kI_v2 test1000 at k = 20 / 100 / 200.
# WHAT: ac        = T_res_rq9fr_kI_KL final (frozen reader, KL, stage 2: the §3.1 main row), hard-linked from $W/rq_eval_fin
#       text_base = untuned Qwen3.5-9B
#       text_lora = X_rq9_text_20I u5600 (text + LoRA reference; read now so it is ready if the user includes it)
# Scored by jobs/rq_eval_job.sh (AC: sec45_eval.py batch 32; text: text_lora_eval_hf.py batch 8; greedy, cap 1024; code v5), the dev1000 scorer;
# finished outputs are skipped, so a requeue resumes.
#   ORCD: cd /home/ngom/orcd/scratch/activation/sec35 && sbatch --export=NIL --time=24:00:00 --partition=mit_preemptable --requeue --gres=gpu:rtx_pro_6000:1 jobs/rq_test_orcd.sh WHAT
set -uo pipefail
WHAT=${1:?ac|text_base|text_lora}; export RQ_TEXT_BATCH=${2:-8}   # 2nd arg: text batch (2026-11-15: 2 for the k=200 retries)
W=/home/ngom/orcd/scratch/activation/sec35; R=$W/rq_eval_test_$WHAT; R20=${R}_20i; FAIL=0
mkdir -p $R/text $R20/text
case $WHAT in
  ac)        N=T_res_rq9fr_kI_KL_u02400; S=$W/rq_eval_fin/$N
             for D in $R $R20; do [ -d $D/$N ] || { mkdir -p $D/$N.tmp && cp -al $S/ac $S/reader $S/meta.json $D/$N.tmp/ && mv $D/$N.tmp $D/$N; }; done
             P=$R/$N; P20=$R20/$N;;
  text_base) mkdir -p $R/text/base $R20/text/base; P=$R/text/base; P20=$R20/text/base;;
  text_lora) N=X_rq9_text_20I_u05600; S=$W/rq_eval_kI_text/text/$N
             for D in $R $R20; do [ -d $D/text/$N ] || { cp -al $S $D/text/$N.tmp && rm -f $D/text/$N.tmp/KS && mv $D/text/$N.tmp $D/text/$N; }; done
             P=$R/text/$N; P20=$R20/text/$N;;
  *) echo "WHAT $WHAT"; exit 2;;
esac
ev() { RQ_EVAL_DIR=$1 RQ_EVAL_DATA=$2 EVAL_SPLIT=$3 SEC45_BUNDLE=sec35_code_v5.tar.gz bash $W/jobs/rq_eval_job.sh || FAIL=1; }
echo 100 > $P/KS;          ev $R hotpot_tasks_kI_v2 test1000_100I
echo 200 > $P/KS;          ev $R hotpot_tasks_kI_v2 test1000_200I
echo 20 > $P20/KS;         ev $R20 hotpot_tasks_v2 test1000
echo "20 100 200" > $P/KS; ev $R hotpot_tasks_kI_v2 test1000
echo "test1000 reads of $WHAT done (fail=$FAIL)"; exit $FAIL
