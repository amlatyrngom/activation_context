#!/bin/bash
#SBATCH --job-name=qa_adam_offload_probe
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/probe_adam_moment_offload_v1.py" \
    --manifest "$QA_ROOT/bench/ADAM_MOMENT_OFFLOAD_PROBE_V1.json" \
    --design "$QA_ROOT/bench/PI42K_MERGED32K_FRESH960_DESIGN_V1.json" \
    --pair "$QA_ROOT/fresh42k_native8k_final_lr_v1/initial/pair.json" \
    --material "$QA_ROOT/fresh42k_native8k_final_lr_v1/material.json" \
    --subset "$QA_ROOT/adam_moment_offload_probe_v1_inputs/subset.json" \
    --output "$QA_REPORT_ROOT/adam_moment_offload_probe_v1" \
    --workspace "$QA_ROOT/adam_moment_offload_probe_v1" \
    --autotune-root "$HOME/activation_artifacts/AUTOTUNE" --seconds 2520
