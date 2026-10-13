#!/bin/bash
#SBATCH --job-name=qa_reader_memory
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
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
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/probe_reader_memory_v1.py" \
    --cases "$QA_ROOT/bench/reader_checkpoint_geometry_v1/selected_cases.json" \
    --census "$QA_ROOT/bench/reader_checkpoint_geometry_v1/summary.json" \
    --proof "$QA_REPORT_ROOT/reader_checkpoint_probe_v1/proof/proof.json" \
    --parent-receipt "$QA_ROOT/bench/STRONG_PIECEWISE_BUDGET_PARENT_V1.json" \
    --reports "$QA_REPORT_ROOT" --runs-root "$QA_ROOT" \
    --output "$QA_REPORT_ROOT/reader_memory_probe_v1" \
    --workspace "$QA_ROOT/reader_memory_probe_v1" \
    --autotune-root "$HOME/activation_artifacts/AUTOTUNE" --seconds 840
