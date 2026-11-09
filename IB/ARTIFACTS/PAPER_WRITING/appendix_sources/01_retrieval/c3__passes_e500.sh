#!/bin/bash
# rq-fresh3 (Oct 21 ≈19:45Z): Arm G eval500 passes 1-2 (the Text eval500 points are means of 3 passes; ours were 1) for the figure's K:
# GPU 0: 10x20 then 1x20; GPU 1 (after gpu1_fill.sh, PID $1): 5x20. Appended to the pass-0 files (rq_eval skips (id, pass) already written;
# rq_eval --score averages every row, i.e. over passes).
set -u; Q=/root/rq; cd ~/sky_workdir
export RQ_THINKING=0 PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$Q/kit_c2:$Q/bench:$Q CQA_BENCH=$Q/bench
G=$Q/sync/real_gold; PG="$G/AC_MODELS/rag_poc/epoch_005 $G/LORAS/rag_reader/epoch_005"
run() { # GPU CALLS
  echo "=== $(date -u +%FT%TZ) passes 1-2: armG eval500 $2x20 on GPU $1"
  CUDA_VISIBLE_DEVICES=$1 uv run --frozen python $Q/kit_c2/rq_eval.py --data $Q/data --set eval500 --pair $PG --ratio 0.1 --calls $2 --per-call 20 --question-first \
     --batch 32 --max-new 12288 --passes 2 --pass-offset 1 --shard 0/1 --out $Q/out/e500_$2x20_armG_sh0.jsonl > $Q/out/passes_e500_$2x20_armG.log 2>&1
  echo "=== $(date -u +%FT%TZ) $2x20 exit $?"
}
(run 0 10; run 0 1) &
(while kill -0 $1 2>/dev/null; do sleep 20; done; run 1 5) &
wait
echo "=== $(date -u +%FT%TZ) passes done"
