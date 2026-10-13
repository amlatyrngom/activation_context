#!/bin/bash
#SBATCH --job-name=qa_piecewise_probe
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1
export QA_P2_ROOT=$HOME/orcd/scratch/activation/KEEP/P2
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/piecewise_original_probe_v1
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_ROOT/reports/piecewise_original_probe_v1"
printf 'Preserve train-only batching equivalence and timing receipts. No model checkpoints written.\n' > "$QA_ROOT/reports/piecewise_original_probe_v1/PRESERVE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/probe_piecewise_original_v1.py" \
    --material "$QA_ROOT/piecewise_training_4k_8k_material/material.json" \
    --bench "$QA_ROOT/bench/run_qa_comparison_v3.py" \
    --p2-root "$QA_P2_ROOT" \
    --output "$QA_ROOT/reports/piecewise_original_probe_v1" \
    --workspace "$ACTIVATION_SYNC_ROOT" \
    --autotune-root "$HOME/activation_artifacts/AUTOTUNE" --seconds 1080
