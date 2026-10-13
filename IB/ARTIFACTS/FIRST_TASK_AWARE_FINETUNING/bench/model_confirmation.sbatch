#!/bin/bash
#SBATCH --job-name=model_confirmation
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
QA_ARMS=("$@")
if [[ ${#QA_ARMS[@]} == 0 ]]; then
    QA_ARMS=(oracle_9b_16k_r16 oracle_9b_16k_r8 oracle_27b_16k_r16 oracle_27b_16k_r8 encoder4b_oracle27b_16k_r16)
fi
cd "$QA_SOURCE_ROOT"
for QA_ARM in "${QA_ARMS[@]}"; do
    export ACTIVATION_SYNC_ROOT=$QA_ROOT/model_confirmation_cache/$QA_ARM
    mkdir -p "$ACTIVATION_SYNC_ROOT"
    ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
    "$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_model_confirmation.py" \
        --design "$QA_ROOT/model_comparison_material/confirmation_design.json" \
        --bench "$QA_ROOT/bench/run_qa_confirmation_v3.py" \
        --harness-4b "$QA_ROOT/bench/run_qa_encoder4b_v1.py" \
        --material-dir "$QA_ROOT/model_comparison_material" --reports "$QA_REPORT_ROOT" \
        --output "$QA_REPORT_ROOT/model_comparison_confirmation" --arm "$QA_ARM" --seconds 1200
done
