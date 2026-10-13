#!/bin/bash
#SBATCH --job-name=data_confirmation
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
QA_ARMS=("$@")
if [[ ${#QA_ARMS[@]} == 0 ]]; then QA_ARMS=(data_2k_4ep data_4k_2ep data_8k_1ep); fi
cd "$QA_SOURCE_ROOT"
for QA_ARM in "${QA_ARMS[@]}"; do
    export ACTIVATION_SYNC_ROOT=$QA_ROOT/data_confirmation_cache/$QA_ARM
    mkdir -p "$ACTIVATION_SYNC_ROOT"
    ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
    "$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_data_confirmation.py" \
        --bench "$QA_ROOT/bench/run_qa_confirmation_v3.py" \
        --material-dir "$QA_ROOT/data_budget_material" --reports "$QA_REPORT_ROOT" \
        --output "$QA_REPORT_ROOT/data_budget_confirmation" --arm "$QA_ARM" \
        --seconds 420 --decode-samples 12
done
