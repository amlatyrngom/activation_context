#!/bin/bash
#SBATCH --account=quanta --partition=quanta-h200 --qos=quanta-main --gpus=1 --cpus-per-task=16 --mem=200G
#SBATCH --time=06:00:00 --requeue --open-mode=append --job-name=rq_evps --output=/data/scratch-fast/ngom/rq_v1/out/evps_%j.out
# sbatch --export=NIL ev_passes.sbatch SET NAME SYNC_ROOT CALLS SHARD (Oct 21 ≈20:15Z): passes 1-2 (--passes 2 --pass-offset 1) of a BM25 CALLSx20 eval,
# question-first, thinking off, 1/10, kit_s20 -> out/SET_ps_NAME_CALLSx20_shI.jsonl; with pass 0 they make the 3-pass mean the Text eval500 points use.
source /data/scratch-fast/ngom/rq_v1/env9.sh
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin RQ_THINKING=0 PYTHONPATH=${PYTHONPATH/kit_s7/kit_s20}
set_=$1; name=$2; root=$3; calls=$4; shard=$5
ep=$(basename $(ls -d $root/AC_MODELS/rag_poc/epoch_00? | tail -1)); P="$root/AC_MODELS/rag_poc/$ep $root/LORAS/rag_reader/$ep"
echo "=== $(date -u +%FT%TZ) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname): $set_ $name $ep ${calls}x20 passes 1-2 shard $shard"
$PY $F/kit_s20/rq_eval.py --data $D --set $set_ --pair $P --ratio 0.1 --calls $calls --per-call 20 --question-first --batch 32 --max-new 12288 \
   --passes 2 --pass-offset 1 --shard $shard --out $F/out/${set_}_ps_${name}_${calls}x20_sh${shard%/*}.jsonl
echo "=== $(date -u +%FT%TZ) ${calls}x20 passes shard $shard exit $?"
