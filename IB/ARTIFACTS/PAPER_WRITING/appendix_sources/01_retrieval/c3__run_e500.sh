#!/bin/bash
# rq-fresh3 (Oct 21 ≈17:45Z): after run_evalsG.sh (PID $1), eval500 (1 pass) in the paper format 20x20 BM25, question-first, 1/10, thinking off:
# Arm G (sync/real_gold final) then tA_s2 epoch_004, each split over both GPUs (--shard 0/2, 1/2) -> out/e500_20x20_{armG,tA_s2}_sh{0,1}.jsonl.
# Firms up dev200 Arm G 60.0 vs tA_s2 56.0 (paired p .185 at n = 200). rq_eval appends per batch, so a cut-off run keeps what it wrote.
set -u; Q=/root/rq; cd ~/sky_workdir
export RQ_THINKING=0 PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$Q/kit_c2:$Q/bench:$Q CQA_BENCH=$Q/bench
while kill -0 $1 2>/dev/null; do sleep 30; done
G=$Q/sync/real_gold; ep=$(basename $(ls -d $G/AC_MODELS/rag_poc/epoch_00? | tail -1))
PG="$G/AC_MODELS/rag_poc/$ep $G/LORAS/rag_reader/$ep"; PA="$Q/tA_s2/AC_MODELS/epoch_004 $Q/tA_s2/LORAS/epoch_004"
ev() { # GPU NAME SHARD PAIR...
  local gpu=$1 name=$2 shard=$3; shift 3
  echo "=== $(date -u +%FT%TZ) eval500 $name 20x20 shard $shard on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu uv run --frozen python $Q/kit_c2/rq_eval.py --data $Q/data --set eval500 --pair "$@" --ratio 0.1 --calls 20 --per-call 20 \
     --question-first --batch 32 --max-new 12288 --shard $shard --out $Q/out/e500_20x20_${name}_sh${shard%/*}.jsonl > $Q/out/e500_${name}_sh${shard%/*}.log 2>&1
  echo "=== $(date -u +%FT%TZ) eval500 $name shard $shard exit $?"
}
ev 0 armG 0/2 $PG & ev 1 armG 1/2 $PG & wait
ev 0 tA_s2 0/2 $PA & ev 1 tA_s2 1/2 $PA & wait
echo "=== $(date -u +%FT%TZ) eval500 done"
