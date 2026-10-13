#!/bin/bash
#SBATCH --job-name=prepare_fresh42k8k
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
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/prepare_fresh42k_native8k_v1.py" \
  --material "$QA_ROOT/model_comparison_32k_material/material_27b_32000.json" \
  --bank "$QA_ROOT/reports/synthesis_bank_27b" \
  --continuation-design "$QA_ROOT/bench/CONTINUATION_10K_DESIGN_ORCD.json" \
  --continuation-initial-pair "$QA_ROOT/continuation_32k_initial/pair.json" \
  --reports "$QA_ROOT/reports" \
  --parent-readout "$QA_ROOT/reports/continuation_cohort100/continue32k_10k_final_lr.json" \
  --parent-arm continue32k_10k_final_lr \
  --reservations "$QA_ROOT/model_comparison_32k_material/reserved_questions.json" "$QA_ROOT/model_comparison_material/reserved_questions.json" "$QA_ROOT/qa27b_reserved_questions.json" \
  --output "$QA_ROOT/fresh42k_native8k_final_lr_v1"
