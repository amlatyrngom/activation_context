#!/bin/bash
#SBATCH --job-name=freeze_data_amount
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/freeze_data_amount_design.py" \
    --control "${1:?Pass the selected completed development arm}" \
    --design "$QA_ROOT/bench/MODEL_COMPARISON_DESIGN.json" \
    --selection "$QA_ROOT/bench/MODEL_COMPARISON_COHORT100_SELECTION.json" \
    --reports "$QA_ROOT/reports" --run-root "$QA_ROOT" \
    --output "$QA_ROOT/bench/DATA_AMOUNT_32K_EVALUATION.json"
