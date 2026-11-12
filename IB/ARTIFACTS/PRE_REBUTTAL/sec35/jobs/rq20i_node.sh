#!/bin/bash
# Retrieval 20I phase (pre-rebuttal 2026-11-11, PLAN "retrieval 20I phase"): one role per SkyPilot spot node, run as a sky job:
#   .venv/bin/python -m activation.cloud.sky exec NODE -- bash /root/rq20i_node.sh frozen|lora|text      (scp ec2_prep.sh + this to /root first)
# 20I = hotpot_tasks_v2 (the §3.5 rows: BM25 top 20, both golds injected at seeded positions), one encoder call of 20; the first 44,800
# questions of the fixed seed-0 order (5,600 x 8); dev1000 every 800 updates, greedy, cap 1024.
#   frozen: R_res_rq9fr (2B->9B Input + Residuals, LS head, 2,000 x 8 FineWeb, reader FROZEN) then T_res_rq9fr_20I from its final pair (reader frozen)
#   lora:   T_res_rq9L_20I from R_res_rq9's final pair (reader LoRA 2e-5, penalty 1.0)
#   text:   X_rq9_text_20I (9B text + LoRA r128, LR 2e-5, the same items in the same order)
# Every run folder is mirrored on S3 (ec2_run.sh), so a reclaimed node's role resumes on a new node.
set -uo pipefail
ROLE=$1; W=/root/sec35; J=$W/jobs; S3PY=/root/s3venv/bin/python
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/sec35
bash /root/ec2_prep.sh || { echo "prep failed"; exit 2; }
export SEC45_EVAL_MAX_NEW=1024 SEC45_MICRO_BATCH=4
init() {   # init R: R's final pair at $W/init/R_final (linked when R ran on this node, else from S3)
  local I=$W/init/$1_final
  [ -f $I/DONE ] && return 0
  [ -f $W/runs/$1/DONE ] && return 0                         # the job script links it itself
  rm -rf $I; mkdir -p $I && $S3PY $W/s3x.py getdir $S3/runs/$1/sync/ckpt/final/ $I/ && $S3PY $W/s3x.py get $S3/runs/$1/DONE $I/DONE || { echo "$1's final pair is not on S3"; rm -rf $I; return 2; }
}
case $ROLE in
frozen)
  SEC45_MICRO_BATCH=2 bash $J/ec2_run.sh R_res_rq9fr bash $J/generated_rq9fr20I_rtx/R_res_rq9fr.sh || { echo "R_res_rq9fr failed"; exit 1; }
  init R_res_rq9fr || exit 2
  bash $J/ec2_run.sh T_res_rq9fr_20I bash $J/generated_rq9fr20I_rtx/T_res_rq9fr_20I.sh;;
lora)
  init R_res_rq9 || exit 2
  bash $J/ec2_run.sh T_res_rq9L_20I bash $J/generated_rq9L20I_rtx/T_res_rq9L_20I.sh;;
text)
  bash $J/ec2_run.sh X_rq9_text_20I env RUN=X_rq9_text_20I TEXT_MODEL_ID=Qwen/Qwen3.5-9B TASK_DATA=hotpot_tasks_v2 EVAL_SPLIT=dev1000 \
    EVAL_MAX_NEW=1024 EVAL_BATCH=16 STEPS=5600 CKPT_EVERY=800 LR=2e-5 SEED=0 bash $J/text_lora_job.sh;;
frozen_kl)                                # probe (2026-11-12): frozen reader + KL to the gold-view teacher (= the untuned 9B reading the gold passages)
  init R_res_rq9fr || exit 2
  mkdir -p $W/refs; [ -f $W/refs/refs_train.jsonl ] || $S3PY $W/s3x.py get $S3/runs/rq9_gold_refs/refs_train.jsonl $W/refs/refs_train.jsonl || exit 2
  bash $J/ec2_run.sh T_res_rq9fr_20I_kl bash $J/generated_rq9_20Iprobe_rtx/T_res_rq9fr_20I_kl.sh;;
lora90k)                                  # diagnostic (2026-11-12): the lora arm on all 90k 20I questions (11,200 x 8, the §3.5 data amount)
  init R_res_rq9 || exit 2
  bash $J/ec2_run.sh T_res_rq9L_20I_90k bash $J/generated_rq9L20I_rtx/T_res_rq9L_20I_90k.sh;;
frozen90k)                                # diagnostic (2026-11-12): the frozen arm on all 90k 20I questions (11,200 x 8; the 4B frozen res needed this amount)
  init R_res_rq9fr || exit 2
  bash $J/ec2_run.sh T_res_rq9fr_20I_90k bash $J/generated_rq9fr20I_rtx/T_res_rq9fr_20I_90k.sh;;
*) echo "role frozen|lora|text|frozen_kl|lora90k|frozen90k"; exit 2;;
esac
