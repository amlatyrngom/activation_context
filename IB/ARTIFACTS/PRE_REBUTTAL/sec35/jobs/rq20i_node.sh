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
export SEC45_EVAL_MAX_NEW=1024 SEC45_MICRO_BATCH=${SEC45_MICRO_BATCH:-4}
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
text90k)                                  # parity (2026-11-12): the text + LoRA arm on all 90k 20I questions (11,200 x 8), same as the 90k AC runs
  bash $J/ec2_run.sh X_rq9_text_20I_90k env RUN=X_rq9_text_20I_90k TEXT_MODEL_ID=Qwen/Qwen3.5-9B TASK_DATA=hotpot_tasks_v2 EVAL_SPLIT=dev1000 \
    EVAL_MAX_NEW=1024 EVAL_BATCH=16 STEPS=11200 CKPT_EVERY=800 LR=2e-5 SEED=0 bash $J/text_lora_job.sh;;
refs_all)                                 # 2026-11-12: gold-view replies (gold_refs.py, base 9B greedy over the golds) for the rest of train's seed-0
  # order: first up to 44,800 (the 20I KL arms), then all 90,447 (stage 2 KL); appended to the 32,000 of rq9_gold_refs; S3 after every chunk
  D=$W/runs/rq9_gold_refs_all; mkdir -p $D; cd $D; C=$W/code/sec35_code_v3; PY=/opt/activation/.venv/bin/python
  [ -s refs_train.jsonl ] || $S3PY $W/s3x.py get $S3/runs/rq9_gold_refs_all/refs_train.jsonl refs_train.jsonl 2>/dev/null || $S3PY $W/s3x.py get $S3/runs/rq9_gold_refs/refs_train.jsonl refs_train.jsonl || exit 2
  for N in 44800 52000 60000 68000 76000 84000 90447; do
    RQ_KIT=$C/rq_kit RQ_THINKING=0 TEXT_MODEL_ID=Qwen/Qwen3.5-9B PYTHONPATH=$C/rq_kit $PY $C/retrieval/gold_refs.py --data $W/data/rq_tasks_v1 --split train --first $N --seed 0 --out refs_train.jsonl || exit 1
    $S3PY $W/s3x.py put refs_train.jsonl $S3/runs/rq9_gold_refs_all/refs_train.jsonl && echo "refs: first $N done ($(wc -l < refs_train.jsonl) rows) -> S3"
    [ "${REFS_UPTO:-90447}" -le $N ] && break
  done;;
kl_lora|kl_frozen)                        # 2026-11-12 (user: KL to the gold-view teacher is the main configuration): 20I, 44,800, both readers
  [ $ROLE = kl_lora ] && { N=T_res_rq9L_20I_KL; I=R_res_rq9; } || { N=T_res_rq9fr_20I_KL; I=R_res_rq9fr; }
  init $I || exit 2
  mkdir -p $W/refs; $S3PY $W/s3x.py get $S3/runs/rq9_gold_refs_all/refs_train.jsonl $W/refs/refs_train.jsonl || exit 2
  [ $(wc -l < $W/refs/refs_train.jsonl) -ge 44800 ] || { echo "refs cover fewer than 44,800 questions"; exit 2; }
  bash $J/ec2_prep.sh sec35_code_v5.tar.gz > /dev/null || exit 2
  bash $J/ec2_run.sh $N bash $J/generated_rq9KL20I_rtx/$N.sh;;
qa_refs)                                  # 2026-11-12: gold-view replies for the general-QA phase (qa_tune_v1 = 2Wiki + MuSiQue, 100,000; seed-0 order)
  D=$W/runs/qa_gold_refs; mkdir -p $D; cd $D; C=$W/code/sec35_code_v3; PY=/opt/activation/.venv/bin/python
  $S3PY $W/s3x.py getdir $S3/data/qa_tune_v1/ $W/data/qa_tune_v1/ || exit 2
  [ -s refs_qa_train.jsonl ] || $S3PY $W/s3x.py get $S3/runs/qa_gold_refs/refs_qa_train.jsonl refs_qa_train.jsonl 2>/dev/null || true
  for N in 12500 25000 50000 75000 100000; do
    RQ_KIT=$C/rq_kit RQ_THINKING=0 TEXT_MODEL_ID=Qwen/Qwen3.5-9B PYTHONPATH=$C/rq_kit $PY $C/retrieval/gold_refs.py --data $W/data/qa_tune_v1 --split train --first $N --seed 0 --out refs_qa_train.jsonl || exit 1
    $S3PY $W/s3x.py put refs_qa_train.jsonl $S3/runs/qa_gold_refs/refs_qa_train.jsonl && echo "qa refs: first $N done ($(wc -l < refs_qa_train.jsonl) rows) -> S3"
  done;;
qa_kl)                                    # 2026-11-12: general-QA phase, ONE continuous run: KL to the gold-view teacher on 2Wiki + MuSiQue (100,000 = 12,500 x 8),
  # from R_res_rq9 (reader LoRA), HotpotQA 20I dev1000 every 800 updates (the eval split of qa_tune_v1 is hotpot_tasks_v2's dev1000)
  init R_res_rq9 || exit 2
  $S3PY $W/s3x.py getdir $S3/data/qa_tune_v1/ $W/data/qa_tune_v1/ || exit 2
  mkdir -p $W/refs; $S3PY $W/s3x.py get $S3/runs/qa_gold_refs/refs_qa_train.jsonl $W/refs/refs_qa_train.jsonl || exit 2
  [ $(wc -l < $W/refs/refs_qa_train.jsonl) -ge 100000 ] || { echo "qa refs incomplete"; exit 2; }
  bash $J/ec2_prep.sh sec35_code_v5.tar.gz > /dev/null || exit 2
  bash $J/ec2_run.sh T_res_rq9L_QA_KL bash $J/generated_rq9QA_rtx/T_res_rq9L_QA_KL.sh;;
kI_sft_lora|kI_sft_frozen|kI_kl_lora|kI_kl_frozen)   # 2026-11-12 STAGE 2 (user-approved): from the 20I final of the same arm, train on
  # golds-injected 100I / 200I rows (k = 100 or 200 per question, crc32(id) % 2) of the 45,647 questions stage 1 never saw (--offset 44800;
  # 5,600 x 8 = 44,800 of them); the run's own eval: PLAIN BM25 dev1000 at k = 100 (hotpot_tasks_kI_v2 = kI train + bm25e dev/test)
  case $ROLE in kI_sft_lora) N=T_res_rq9L_kI;; kI_sft_frozen) N=T_res_rq9fr_kI;; kI_kl_lora) N=T_res_rq9L_kI_KL;; kI_kl_frozen) N=T_res_rq9fr_kI_KL;; esac
  init $(grep -oP "^INIT=\K\S+" $J/generated_rq9kI_rtx/$N.sh) || exit 2      # the 20I final of the same arm (ec2_run.sh: no S3 in the job)
  $S3PY $W/s3x.py getdir $S3/data/hotpot_tasks_kI_v2/ $W/data/hotpot_tasks_kI_v2/ || exit 2
  case $ROLE in kI_kl_*)
    mkdir -p $W/refs; $S3PY $W/s3x.py get $S3/runs/rq9_gold_refs_all/refs_train.jsonl $W/refs/refs_train.jsonl || exit 2
    [ $(wc -l < $W/refs/refs_train.jsonl) -ge 90447 ] || { echo "refs cover fewer than 90,447 questions (run refs_all first)"; exit 2; };; esac
  bash $J/ec2_prep.sh sec35_code_v5.tar.gz > /dev/null || exit 2
  bash $J/ec2_run.sh $N bash $J/generated_rq9kI_rtx/$N.sh;;
kI_text)                                  # 2026-11-12 STAGE 2 parity for the text arm: X_rq9_text_20I's final adapter (u5600) trained on the SAME
  # 45,647 held-back questions (--offset 44800, 5,600 x 8) at its own operating point, 20I (hotpot_tasks_v2 rows); eval PLAIN BM25 dev1000 at k=20
  # (hotpot_tasks_t2_v1 = v2 train + bm25e dev/test); fresh warm-up + cosine as the AC stage-2 arms; code v6 (text_lora_train --offset / --init-adapter)
  $S3PY $W/s3x.py getdir $S3/data/hotpot_tasks_t2_v1/ $W/data/hotpot_tasks_t2_v1/ || exit 2
  A=$W/init/X_rq9_text_20I_u05600; [ -f $A/adapter_model.safetensors ] || { rm -rf $A; $S3PY $W/s3x.py getdir $S3/runs/X_rq9_text_20I/ckpt/step_05600/ $A/ || exit 2; }
  bash $J/ec2_prep.sh sec35_code_v6.tar.gz > /dev/null || exit 2
  bash $J/ec2_run.sh X_rq9_text_t2 env RUN=X_rq9_text_t2 TEXT_MODEL_ID=Qwen/Qwen3.5-9B TASK_DATA=hotpot_tasks_t2_v1 EVAL_SPLIT=dev1000 \
    EVAL_MAX_NEW=1024 EVAL_BATCH=16 STEPS=5600 CKPT_EVERY=800 LR=2e-5 SEED=0 OFFSET=44800 INIT_ADAPTER=$A SEC45_BUNDLE=sec35_code_v6.tar.gz bash $J/text_lora_job.sh;;
kI_probe)                                 # timing / memory: 48 updates of the stage-2 SFT LoRA arm on the LONGEST k=200 items, no save
  $S3PY $W/s3x.py getdir $S3/data/hotpot_tasks_kI_v2/ $W/data/hotpot_tasks_kI_v2/ || exit 2
  bash $J/ec2_prep.sh sec35_code_v5.tar.gz > /dev/null || exit 2
  init T_res_rq9L_20I || exit 2
  C=$W/code/sec35_code_v5; I=$W/init/T_res_rq9L_20I_final; D=$W/runs/kI_probe; mkdir -p $D
  export HF_HOME=${HF_HOME:-/root/.cache/huggingface} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONPATH=$C:$C/kit:$C/rq_kit:$C/bench CQA_BENCH=$C/bench SEC45_RQ_KIT=$C/rq_kit RQ_THINKING=0 QA4B_READER=Qwen/Qwen3.5-9B QA4B_SIDE=Qwen/Qwen3.5-2B QA4B_RANK=128
  export SEC45_ENCODER=Qwen/Qwen3.5-2B SEC45_RES_LAYERS=all SEC45_K=100,200 SEC45_LINEAR_SKIP=1 QA4B_LAYERS=3,7,11,15,19,23,27,31,35 QA4B_KV=0
  for mb in ${PROBE_MB:-4 2}; do
    ACTIVATION_SYNC_ROOT=$D/mb$mb /opt/activation/.venv/bin/python $C/kit/sec45_train.py --mode res --epu 8 --encoder-lr 5e-5 --reader-lr 2e-5 --penalty-weight 1.0 --seed 0 \
      --offset 44800 --kind task --data $W/data/hotpot_tasks_kI_v2 --pair $I/ac $I/reader --name kI_probe_mb$mb --steps 48 --longest-first 3 --warmup 5 \
      --report-every 1000 --held-panel 32 --micro-batch $mb --seconds 2400 --no-save > $D/mb$mb.log 2>&1
    echo "probe mb$mb exit $?: $(tail -n 1 $D/mb$mb.log | cut -c1-600)"
  done
  $S3PY $W/s3x.py putdir $D/ $S3/runs/kI_probe/;;
*) echo "role frozen|lora|text|frozen_kl|lora90k|frozen90k|text90k|refs_all|kl_lora|kl_frozen|qa_refs|qa_kl|kI_sft_lora|kI_sft_frozen|kI_kl_lora|kI_kl_frozen|kI_text|kI_probe"; exit 2;;
esac
