#!/bin/bash
#SBATCH --account=quanta --partition=quanta-h200 --qos=quanta-main --gpus=1 --cpus-per-task=16 --mem=200G
#SBATCH --time=06:00:00 --requeue --open-mode=append --job-name=rq_txref --output=/data/scratch-fast/ngom/rq_v1/out/txref_%j.out
# sbatch --export=NIL text_ref.sbatch SET PASSES K... (user, Oct 21 ≈14:50Z: "reference evals for 1x100 and 1x200"): the base 9B reads the BM25
# top K passages as text, thinking off, cap 12,288, bf16 KV (as the dev200/eval500 thinking-off top20/top400 baselines) -> out/text_nothink/SET.topK.jsonl
source /data/scratch-fast/ngom/rq_v1/env9.sh
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin PYTHONPATH=${PYTHONPATH/kit_s7/kit_s20}
set_=$1; passes=$2; shift 2
echo "=== $(date -u +%FT%TZ) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname): $set_ passes $passes ks $*"
$PY $F/kit_s20/rq_text_eval.py --kv-cache-dtype auto --no-thinking --data $D --set $set_ --out-dir $F/out/text_nothink --passes $passes --ks "$@" --arms top; code=$?
echo "=== $(date -u +%FT%TZ) exit $code"; exit $code
