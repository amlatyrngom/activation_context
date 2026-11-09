#!/bin/bash
#SBATCH --account=quanta --partition=quanta-h200 --qos=quanta-main --gpus=1 --cpus-per-task=16 --mem=200G
#SBATCH --time=03:00:00 --requeue --open-mode=append --job-name=rq_tAteach --output=/data/scratch-fast/ngom/rq_v1/out/tA_teach_%j.out
# sbatch --export=NIL tA_teach.sbatch NAME SHARD STAGE... (Track A, Oct 21): KL targets (top-32) of the base 9B (no reader adapter) on the
# same-context view (the reference's own N x 20 injected list as text) + its kept reference reply, and the no-context penalty references;
# refs $F/refs/NAME.jsonl (correct + stopped kept) -> $F/teacher/tA_NAME.<shard>.pt. kit_s18.
source /data/scratch-fast/ngom/rq_v1/env9.sh
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin RQ_THINKING=0 PYTHONPATH=${PYTHONPATH/kit_s7/kit_s18}
name=$1; shard=$2; shift 2; out=$F/teacher/tA_$name.$(echo $shard | tr / _).pt
echo "=== $(date -u +%FT%TZ) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname): teacher $name shard $shard stages $*"
[ -f $out ] && { echo "exists: $out"; exit 0; }
A=$F/sync/sftd_dense_r10b
$PY $F/kit_s18/rq_teacher.py --data $D --pair $A/AC_MODELS/rag_poc/epoch_002 $A/LORAS/rag_reader/epoch_002 --refs $F/refs/$name.jsonl \
   --questions $D/hotpot_train_native.questions.jsonl --stages "$@" --micro-batch 4 --shard $shard --out $out; code=$?
echo "=== $(date -u +%FT%TZ) teacher exit $code"; exit $code
