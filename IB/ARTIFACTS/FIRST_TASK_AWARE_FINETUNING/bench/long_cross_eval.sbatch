#!/bin/bash
#SBATCH --job-name=long_cross_eval
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
QA_CELLS=("$@")
if [[ ${#QA_CELLS[@]} == 0 ]]; then
    QA_CELLS=(control900__2k control900__32k long_adapt_2k__2k long_adapt_2k__32k long_adapt_32k__2k long_adapt_32k__32k)
fi
cd "$QA_SOURCE_ROOT"
for QA_CELL in "${QA_CELLS[@]}"; do
    export ACTIVATION_SYNC_ROOT=$QA_ROOT/long_cross_eval_cache/$QA_CELL
    mkdir -p "$ACTIVATION_SYNC_ROOT"
    ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
    "$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_long_cross.py" \
        --bench "$QA_ROOT/bench/run_qa_long_cross_v2.py" \
        --material "$QA_ROOT/long_adaptation_material/material.json" \
        --initial-pair "$QA_ROOT/long_adaptation_material/initial_pair.json" \
        --reports "$QA_REPORT_ROOT" --output "$QA_REPORT_ROOT/long_cross_eval" \
        --cell "$QA_CELL" --seconds 1200 --decode-samples 12
done
