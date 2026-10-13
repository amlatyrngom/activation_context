#!/bin/bash
#SBATCH --job-name=qa_encoder4b_restore
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export QA_P4_ROOT=$HOME/orcd/scratch/activation/KEEP/P4
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACTIVATION_SYNC_ROOT=$QA_ROOT/encoder4b_restore_smoke
unset QA_CAMPAIGN_DEADLINE
mkdir -p "$ACTIVATION_SYNC_ROOT"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/run_qa_encoder4b_v1.py" \
    --arm smoke --run-name encoder4b_restore_smoke \
    --material "$QA_ROOT/data_budget_material/material_8000.json" \
    --items 8 --epochs 1 --seconds 1200 \
    --encoder-lr 5e-5 --reader-lr 2e-5 --ratio 0.0625 --micro-batch-examples 2
