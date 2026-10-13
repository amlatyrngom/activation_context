#!/bin/bash
#SBATCH --job-name=long_adapt_32k
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
QA_RUN_NAME=long_adapt_32k
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export QA_P2_ROOT=$HOME/orcd/scratch/activation/KEEP/P2
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACTIVATION_SYNC_ROOT=$QA_ROOT/$QA_RUN_NAME
export QA_CAMPAIGN_DEADLINE=2026-09-22T12:00:00+00:00
mkdir -p "$ACTIVATION_SYNC_ROOT"
printf 'Active old-question long-context adaptation; retain paired saves and reports.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/run_qa_night_v4.py" \
    --arm joint --run-name "$QA_RUN_NAME" \
    --material "$QA_ROOT/long_adaptation_material/material.json" \
    --initial-pair "$QA_ROOT/long_adaptation_material/initial_pair.json" \
    --items 640 --epochs 1 --seconds 19800 \
    --encoder-lr 5e-5 --reader-lr 2e-5 --ratio 0.0625 --micro-batch-examples 1 \
    --evidence-tokens 32768 --max-passages 64 --resume-baseline-receipt "$QA_ROOT/bench/long32k_resume24_baseline_receipt.json" "$@"
