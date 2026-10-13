#!/bin/bash
#SBATCH --job-name=qa_piecewise_ckpt
#SBATCH --account=csail
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_BASE=/data/scratch/ngom/activation
QA_ROOT=$QA_BASE/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_P2_ROOT=$QA_BASE/KEEP/P2
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$QA_BASE/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/piecewise_checkpoint_probe_csail_v1
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_ROOT/reports/piecewise_checkpoint_probe_csail_v1"
printf 'Preserve checkpoint-only proof and exact failed-batch geometry/fit. No model checkpoints written.\n' > "$QA_ROOT/reports/piecewise_checkpoint_probe_csail_v1/PRESERVE"
cd "$QA_SOURCE_ROOT"
"$QA_BASE/venv/bin/python" "$QA_ROOT/bench/probe_piecewise_checkpoint_v1.py" \
    --material "$QA_ROOT/piecewise_training_4k_8k_material/material.json" \
    --bench "$QA_ROOT/bench/run_qa_piecewise_v1.py" \
    --failed-report "$QA_ROOT/piecewise_checkpoint_failed_metadata" \
    --p2-root "$QA_P2_ROOT" \
    --output "$QA_ROOT/reports/piecewise_checkpoint_probe_csail_v1" \
    --workspace "$ACTIVATION_SYNC_ROOT" \
    --autotune-root "$QA_BASE/AUTOTUNE" --seconds 1080
