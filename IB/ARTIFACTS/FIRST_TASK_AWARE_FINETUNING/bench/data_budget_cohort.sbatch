#!/bin/bash
#SBATCH --job-name=data_cohort100
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export QA_P2_ROOT=$HOME/orcd/scratch/activation/KEEP/P2
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/data_budget_cohort_cache
mkdir -p "$ACTIVATION_SYNC_ROOT"
printf 'Active fixed100 qualitative comparison of three completed data-budget schedules.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
for QA_CONDITION in frozen_p2 data_2k_4ep data_4k_2ep data_8k_1ep; do
    "$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_data_budget_cohort.py" \
        --bench "$QA_ROOT/bench/run_qa_night_v1.py" --reports "$QA_REPORT_ROOT" \
        --material-dir "$QA_ROOT/data_budget_material" \
        --selection "$QA_ROOT/bench/DATA_BUDGET_COHORT100_SELECTION.json" \
        --p2-root "$QA_P2_ROOT" --condition "$QA_CONDITION" \
        --output "$QA_REPORT_ROOT/data_budget_cohort100" --seconds 600
done
