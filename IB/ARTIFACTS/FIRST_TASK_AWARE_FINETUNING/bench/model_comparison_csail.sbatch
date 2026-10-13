#!/bin/bash
#SBATCH --job-name=qa_16k
#SBATCH --account=csail
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=05:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_CLUSTER_ROOT=/data/scratch/ngom/activation
QA_ROOT=$QA_CLUSTER_ROOT/taskaware
QA_RUN_NAME=$1
QA_MATERIAL=$QA_ROOT/$2
QA_ITEMS=$3
QA_EPOCHS=$4
QA_ENCODER_LR=$5
QA_READER_LR=$6
shift 6
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports QA_P2_ROOT=$QA_ROOT/checkpoints/P2_archived
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$QA_CLUSTER_ROOT/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACTIVATION_SYNC_ROOT=$QA_ROOT/$QA_RUN_NAME
export QA_CAMPAIGN_DEADLINE=2026-09-22T12:00:00+00:00
mkdir -p "$ACTIVATION_SYNC_ROOT"
printf 'Active overnight experiment; retain paired saves and reports.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "$QA_CLUSTER_ROOT/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$QA_CLUSTER_ROOT/venv/bin/python" "$QA_ROOT/bench/run_qa_night_v3.py" \
    --arm joint --run-name "$QA_RUN_NAME" --material "$QA_MATERIAL" \
    --items "$QA_ITEMS" --epochs "$QA_EPOCHS" --seconds "${QA_TRAIN_SECONDS:-14400}" \
    --encoder-lr "$QA_ENCODER_LR" --reader-lr "$QA_READER_LR" "$@"
