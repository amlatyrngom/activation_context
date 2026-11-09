#!/bin/bash
# rq-fresh3: one Track A real-format arm (Oct 21, user: "both ... even if just 500 updates"). Usage: bash run_train_real.sh VIEW [ITEMS]
# From tA_s2 epoch_004 (Track A stage-2 final), KL toward the VIEW references (same = the 9B read the student's 400 passages; gold = the
# 9B read the golds alone), student = the BM25 top-400 as 20 question-first calls of 20 at 1/10; 2 GPUs DDP, micro-batch 2;
# enc 5e-5 / reader 2e-5, warmup 20, cooldown over the last 100 updates; held = 64 kept held references (same view).
set -u; Q=/root/rq; V=$1; ITEMS=${2:-4000}; cd ~/sky_workdir
export RQ_THINKING=0 PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$Q/kit_c2:$Q/bench:$Q CQA_BENCH=$Q/bench
export ACTIVATION_SYNC_ROOT=$Q/sync/real_$V
echo "=== $(date -u +%FT%TZ) real_$V: $ITEMS items, micro-batch ${MB:-2}"
uv run --frozen python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 $Q/kit_c2/rq_train.py --data $Q/data \
   --pair $Q/tA_s2/AC_MODELS/epoch_004 $Q/tA_s2/LORAS/epoch_004 --layout interleaved --loss kl --refs $Q/refs/real_$V.all.jsonl \
   --questions $Q/data/hotpot_train_native.questions.jsonl --teacher-targets $Q/teacher/real_$V.pt --teacher-view $V --held-panel 64 \
   --name real_$V --ratio 0.1 --report-every 125 --ckpt-every 250 --micro-batch ${MB:-2} --stages real --items $ITEMS \
   --encoder-lr 5e-5 --reader-lr 2e-5 --schedule cooldown --cooldown-updates 100 --warmup 20
code=$?; echo "=== $(date -u +%FT%TZ) real_$V train exit $code"; exit $code
