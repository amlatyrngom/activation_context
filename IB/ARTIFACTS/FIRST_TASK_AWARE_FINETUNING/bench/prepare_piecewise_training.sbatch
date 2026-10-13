#!/bin/bash
#SBATCH --job-name=prepare_piecewise
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/prepare_piecewise_training.py" \
    --source "$QA_ROOT/model_comparison_material/material_27b_16000.json" \
    --output "$QA_ROOT/piecewise_training_4k_8k_material"
