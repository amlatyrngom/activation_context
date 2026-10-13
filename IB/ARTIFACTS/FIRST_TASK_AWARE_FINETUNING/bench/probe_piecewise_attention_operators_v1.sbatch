#!/bin/bash
#SBATCH --job-name=qa_piecewise_attn_ops
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
QA_WORK=$QA_ROOT/piecewise_attention_operators_v1
test ! -e "$QA_WORK/AC_MODELS"
mkdir -p "$QA_WORK"
test -e "$QA_WORK/AUTOTUNE" || ln -s "$HOME/activation_artifacts/AUTOTUNE" "$QA_WORK/AUTOTUNE"
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 ACTIVATION_SYNC_ROOT=$QA_WORK
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/probe_piecewise_attention_operators_v1.py" \
    --material "$QA_ROOT/piecewise_training_4k_8k_material/material.json" \
    --bench "$QA_ROOT/bench/run_qa_piecewise_v2.py" \
    --design "$QA_ROOT/bench/PIECEWISE_TRAINING_DESIGN_V2.json" \
    --parent-receipt "$QA_ROOT/bench/STRONG_PIECEWISE_BUDGET_PARENT_V1.json" \
    --memory-receipt "$QA_ROOT/bench/PIECEWISE_CHECKPOINT_MEMORY_FIT.json" \
    --reports "$QA_ROOT/reports" --runs-root "$QA_ROOT" \
    --output "$QA_ROOT/reports/piecewise_attention_operators_v1"
