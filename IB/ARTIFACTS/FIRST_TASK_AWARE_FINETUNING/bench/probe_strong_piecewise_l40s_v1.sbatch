#!/bin/bash
#SBATCH --job-name=qa_piecewise_l40s_probe
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUBLAS_WORKSPACE_CONFIG
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/probe_strong_piecewise_l40s_v1.py" \
    --material "$QA_ROOT/piecewise_training_4k_8k_material/material.json" \
    --bench "$QA_ROOT/bench/run_qa_piecewise_v2.py" \
    --design "$QA_ROOT/bench/PIECEWISE_TRAINING_DESIGN_V2.json" \
    --parent-receipt "$QA_ROOT/bench/STRONG_PIECEWISE_BUDGET_PARENT_V1.json" \
    --memory-receipt "$QA_ROOT/bench/PIECEWISE_CHECKPOINT_MEMORY_FIT.json" \
    --reports "$QA_REPORT_ROOT" --runs-root "$QA_ROOT" \
    --output "$QA_REPORT_ROOT/strong_piecewise_l40s_probe_v1" \
    --workspace "$QA_ROOT/strong_piecewise_l40s_probe_v1" \
    --autotune-root "$HOME/activation_artifacts/AUTOTUNE" --seconds 2520
