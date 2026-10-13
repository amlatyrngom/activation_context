#!/bin/bash
#SBATCH --job-name=piecewise_cohort
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_CONDITION=${1:?Pass frozen_p2, piecewise_27b_4k_8k, or merged_27b_4k_8k}
QA_CONTEXT=${2:?Pass 8k or 32k}
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/piecewise_cohort_cache/${QA_CONDITION}__${QA_CONTEXT}
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_ROOT/reports/piecewise_cohort100"
printf 'Preserve fixed development100 evidence, NLL replay and actual decoded IDs.\n' > "$QA_ROOT/reports/piecewise_cohort100/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_piecewise_cohort_v2.py" \
    --design "$QA_ROOT/bench/PIECEWISE_TRAINING_DESIGN.json" \
    --material "$QA_ROOT/piecewise_training_4k_8k_material/material.json" \
    --bench "$QA_ROOT/bench/run_qa_piecewise_v1.py" \
    --selection "$QA_ROOT/bench/MODEL_COMPARISON_COHORT100_SELECTION.json" \
    --reports "$QA_ROOT/reports" --runs-root "$QA_ROOT" \
    --p2-root "$HOME/orcd/scratch/activation/KEEP/P2" \
    --condition "$QA_CONDITION" --context "$QA_CONTEXT" \
    --output "$QA_ROOT/reports/piecewise_cohort100" --seconds "${QA_EVAL_SECONDS:-2100}"
