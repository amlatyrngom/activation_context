#!/bin/bash
#SBATCH --job-name=qa_encoder4b
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=07:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
QA_RUN_NAME=$1
QA_MATERIAL=$QA_ROOT/$2
QA_ITEMS=$3
QA_EPOCHS=$4
QA_ENCODER_LR=$5
QA_READER_LR=$6
shift 6
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports QA_P4_ROOT=$HOME/orcd/scratch/activation/KEEP/P4
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACTIVATION_SYNC_ROOT=$QA_ROOT/$QA_RUN_NAME
unset QA_CAMPAIGN_DEADLINE  # User explicitly permits this 4B experiment after the campaign deadline.
mkdir -p "$ACTIVATION_SYNC_ROOT"
printf 'Active overnight experiment; retain paired saves and reports.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/run_qa_encoder4b_v2.py" \
    --arm joint --run-name "$QA_RUN_NAME" --material "$QA_MATERIAL" \
    --items "$QA_ITEMS" --epochs "$QA_EPOCHS" --seconds "${QA_TRAIN_SECONDS:-21600}" \
    --encoder-lr "$QA_ENCODER_LR" --reader-lr "$QA_READER_LR" "$@"
