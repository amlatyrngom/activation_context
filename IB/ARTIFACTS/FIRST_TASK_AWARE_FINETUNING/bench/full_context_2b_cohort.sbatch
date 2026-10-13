#!/bin/bash
#SBATCH --job-name=qa_full_context_2b_100
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/full_context_2b_cohort_cache
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_ROOT/reports/full_context_2b_cohort100"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
printf 'User-requested frozen2B full-context check on original development100; preserve results.\n' > "$QA_ROOT/reports/full_context_2b_cohort100/PRESERVE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_full_context_2b_cohort.py" \
  --input "$QA_ROOT/reports/data_amount_cohort100/oracle_27b_32k.json" \
  --output "$QA_ROOT/reports/full_context_2b_cohort100" --seconds 1200
