#!/bin/bash
#SBATCH --job-name=prepare_gentle42k
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1
export PYTHONPATH=$QA_SOURCE_ROOT
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/prepare_gentle42k_long_v1.py" \
  --material "$QA_ROOT/strong_long_adaptation_8k_material/material.json" \
  --continuation-design "$QA_ROOT/bench/CONTINUATION_10K_DESIGN_ORCD.json" \
  --continuation-initial-pair "$QA_ROOT/continuation_32k_initial/pair.json" \
  --reports "$QA_ROOT/reports" \
  --parent-readout "$QA_ROOT/reports/continuation_long_cohort100/continue32k_10k_final_lr__8k/result.json" \
  --parent-arm continue32k_10k_final_lr \
  --output "$QA_ROOT/gentle42k_final_lr_familiar4k_v1"
