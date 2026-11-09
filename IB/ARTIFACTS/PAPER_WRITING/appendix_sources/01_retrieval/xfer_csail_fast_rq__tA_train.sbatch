#!/bin/bash
#SBATCH --account=quanta --partition=quanta-h200 --qos=quanta-main --gpus=2 --cpus-per-task=32 --mem=400G
#SBATCH --time=1-00:00:00 --requeue --open-mode=append --job-name=rq_tA --output=/data/scratch-fast/ngom/rq_v1/out/tA_%j.out
# sbatch --export=NIL tA_train.sbatch NAME REFS TARGETS AC READER [extra rq_train args] (Track A, Oct 21: "optimize Run A"): KL on the full
# same-context reference reply (rq_train --refs, kit_s18), each item reading its own N x 20 injected list as N question-first calls at 1/10;
# held = the kept held-stage references (answer-only numbers from the reply's last "Final answer:", teacher's alongside); 2x H200 DDP.
source /data/scratch-fast/ngom/rq_v1/env9.sh
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin RQ_THINKING=0 PYTHONPATH=${PYTHONPATH/kit_s7/kit_s18}
name=$1; refs=$2; targets=$3; ac=$4; reader=$5; shift 5; extra="$*"
export ACTIVATION_SYNC_ROOT=$F/sync/tA_$name
echo "=== $(date -u +%FT%TZ) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' '): tA_$name $extra"
[ "${SLURM_RESTART_COUNT:-0}" != 0 ] && [ -d $ACTIVATION_SYNC_ROOT ] && mv $ACTIVATION_SYNC_ROOT $ACTIVATION_SYNC_ROOT.r$SLURM_RESTART_COUNT
$PY -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=${SLURM_GPUS_ON_NODE:-2} $F/kit_s18/rq_train.py --data $D --pair $ac $reader --layout interleaved \
   --loss kl --refs $refs --questions $D/hotpot_train_native.questions.jsonl --teacher-targets $targets --held-panel 200 \
   --name tA_$name --ratio 0.1 --report-every 250 --ckpt-every 250 --micro-batch 2 $extra; code=$?
echo "=== $(date -u +%FT%TZ) train exit $code"; exit $code
