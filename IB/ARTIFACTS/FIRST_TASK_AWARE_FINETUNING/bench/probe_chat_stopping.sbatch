#!/bin/bash
#SBATCH --job-name=qa_chat_stop
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/chat_stopping_probe_cache
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_REPORT_ROOT/chat_stopping_probe"
printf 'Preserve raw generated token IDs and decoder configuration.\n' > "$QA_REPORT_ROOT/chat_stopping_probe/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/probe_chat_stopping.py" \
    --design "$QA_ROOT/bench/MODEL_COMPARISON_DESIGN.json" \
    --bench "$QA_ROOT/bench/run_qa_confirmation_v3.py" \
    --material "$QA_ROOT/model_comparison_material/material_27b_16000.json" \
    --run-dir "$QA_REPORT_ROOT/oracle_27b_16k_r16" \
    --checkpoint-root "$QA_ROOT/oracle_27b_16k_r16" \
    --output "$QA_REPORT_ROOT/chat_stopping_probe/update01000.json" --seconds 600
