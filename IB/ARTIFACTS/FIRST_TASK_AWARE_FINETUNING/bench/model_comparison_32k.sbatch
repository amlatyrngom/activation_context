#!/bin/bash
#SBATCH --job-name=qa_27b_32k
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
QA_ENCODER_LR=$2
QA_RATIO=$3
shift 3
[[ $# -eq 0 || ( $# -eq 1 && ${1:-} == --resume ) ]] || { printf 'Only an optional --resume is accepted after run, LR and ratio.\n' >&2; exit 2; }
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports QA_P2_ROOT=$HOME/orcd/scratch/activation/KEEP/P2
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACTIVATION_SYNC_ROOT=$QA_ROOT/$QA_RUN_NAME
# 16:00 EDT is a reporting target, not permission to truncate the requested full pass.
unset QA_CAMPAIGN_DEADLINE
mkdir -p "$ACTIVATION_SYNC_ROOT"
printf 'Active 32k experiment; retain paired saves and reports.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/run_qa_comparison_v2.py" \
    --arm joint --run-name "$QA_RUN_NAME" \
    --material "$QA_ROOT/model_comparison_32k_material/material_27b_32000.json" \
    --items 32000 --epochs 1 --seconds 24600 \
    --encoder-lr "$QA_ENCODER_LR" --reader-lr 0.00002 --ratio "$QA_RATIO" "$@"
