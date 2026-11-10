#!/bin/bash
# Retrieval target probe (pre-rebuttal 2026-11-09), one role per SkyPilot spot node, run as a sky job (keeps the node busy for autodown):
#   .venv/bin/python -m activation.cloud.sky exec NODE -- bash /root/rq_probe_node.sh refs|ans|gold      (scp ec2_prep.sh + this to /root first)
# refs: gold-view references of the base 9B (retrieval/gold_refs.py) for the first 32,000 of train's seed-0 shuffle + dev200 -> runs/rq9_gold_refs.
# ans / gold (1,000 updates x 8 items, ~41 s/update on an RTX PRO 6000): the 9B text + reader-LoRA arm, answer-only target vs gold-view reference target (SEC45_TARGETS; wrong or truncated refs fall
# back to answer-only), same rows / order / LR / steps; dev200 at k=100, cap 1024. Every run folder is mirrored on S3 (ec2_run.sh), so a
# reclaimed node's role restarts where it stopped on a new node.
set -uo pipefail
ROLE=$1; W=/root/sec35; J=$W/jobs; C=$W/code/sec35_code_v3; PY=/opt/activation/.venv/bin/python; S3PY=/root/s3venv/bin/python
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/sec35
bash /root/ec2_prep.sh || { echo "prep failed"; exit 2; }
PROBE="TEXT_MODEL_ID=Qwen/Qwen3.5-9B TASK_DATA=rq_tasks_v1 K=20,100,200 EVAL_SPLIT=dev200 EVAL_K=100 EVAL_MAX_NEW=1024 EVAL_BATCH=8 STEPS=1000 CKPT_EVERY=250 LR=2e-5 SEED=0"
case $ROLE in
refs)
  cat > /root/rq_refs.sh <<REFS
set -e; D=$W/runs/rq9_gold_refs; cd \$D; [ -f DONE ] && exit 0
export RQ_KIT=$C/rq_kit RQ_THINKING=0 TEXT_MODEL_ID=Qwen/Qwen3.5-9B PYTHONPATH=$C/rq_kit
$PY $C/retrieval/gold_refs.py --data $W/data/rq_tasks_v1 --split dev200 --out refs_dev200.jsonl
$PY $C/retrieval/gold_refs.py --data $W/data/rq_tasks_v1 --split train --first 32000 --seed 0 --out refs_train.jsonl
date -u +%FT%TZ > DONE
REFS
  bash $J/ec2_run.sh rq9_gold_refs bash /root/rq_refs.sh;;
ans)
  bash $J/ec2_run.sh X_rq9_text_ans env RUN=X_rq9_text_ans $PROBE bash $J/text_lora_job.sh;;
gold)
  mkdir -p $W/refs
  until $S3PY $W/s3x.py get $S3/runs/rq9_gold_refs/DONE $W/refs/DONE > /dev/null 2>&1; do echo "$(date -u +%FT%TZ) waiting for the gold refs"; sleep 300; done
  $S3PY $W/s3x.py get $S3/runs/rq9_gold_refs/refs_train.jsonl $W/refs/refs_train.jsonl || exit 2
  bash $J/ec2_run.sh X_rq9_text_gold env RUN=X_rq9_text_gold SEC45_TARGETS=$W/refs/refs_train.jsonl $PROBE bash $J/text_lora_job.sh;;
*) echo "role refs|ans|gold"; exit 2;;
esac
