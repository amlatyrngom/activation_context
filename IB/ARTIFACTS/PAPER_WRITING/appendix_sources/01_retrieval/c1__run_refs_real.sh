#!/bin/bash
# rq-fresh3 (Oct 21 ≈10:30Z): Track A real-format references. Plan: data/trackA_real.plan.jsonl (held 200 + real N; BM25 top-400 as 20 x 20,
# no injection). GPU 0: gold-view references for every plan question (short prompts) -> refs/real_gold.jsonl, then same-view shards
# 0,2,4,...; GPU 1: same-view shards 1,3,5,... (8 shards) -> refs/real_same.sh<I>.jsonl. A requeued shard skips ids already in its file.
set -u; Q=/root/rq; cd ~/sky_workdir
export RQ_THINKING=0 PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false PYTHONPATH=$Q/kit_c2:$Q/bench:$Q CQA_BENCH=$Q/bench VLLM_LOGGING_LEVEL=WARNING
mkdir -p $Q/refs
gen() { # GPU VIEW SHARD OUT
  echo "=== $(date -u +%FT%TZ) refs view $2 shard $3 on GPU $1 -> $4"
  CUDA_VISIBLE_DEVICES=$1 uv run --frozen python $Q/kit_c2/rq_refs.py --plan $Q/data/trackA_real.plan.jsonl --stages held real --data $Q/data \
     --questions $Q/data/hotpot_train_native.questions.jsonl --out $4 --shard $3 --view $2 --kv-cache-dtype auto --max-model-len 65536 --chunk ${CHUNK:-400}
  echo "=== $(date -u +%FT%TZ) refs view $2 shard $3 exit $?"
}
( gen 0 gold 0/1 $Q/refs/real_gold.jsonl; for i in 0 2 4 6; do gen 0 same $i/8 $Q/refs/real_same.sh$i.jsonl; done ) > $Q/out/refs_gpu0.log 2>&1 &
( for i in 1 3 5 7; do gen 1 same $i/8 $Q/refs/real_same.sh$i.jsonl; done ) > $Q/out/refs_gpu1.log 2>&1 &
wait; echo "=== $(date -u +%FT%TZ) refs done"
