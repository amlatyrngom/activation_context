#!/bin/bash
#SBATCH --job-name=qualitative100
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:15:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws QA_REPORT_ROOT=$QA_ROOT/reports QA_P2_ROOT=$HOME/orcd/scratch/activation/KEEP/P2
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACTIVATION_SYNC_ROOT=$QA_ROOT/qualitative_cohort
mkdir -p "$ACTIVATION_SYNC_ROOT"
printf 'Active complete100-item qualitative evaluation; retain outputs.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
for condition in frozen_p2 control_500 control_900; do
    "$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_qualitative_cohort.py" \
        --bench "$QA_ROOT/bench/run_qa_hill_v1.py" --run-dir "$QA_ROOT/reports/hill_control" \
        --material "$QA_ROOT/hillclimb_material/material.json" --p2-root "$QA_P2_ROOT" \
        --checkpoint-root "$QA_ROOT/hill_control" --condition "$condition" \
        --output-dir "$QA_ROOT/reports/qualitative_cohort100" --seconds 1800
done
