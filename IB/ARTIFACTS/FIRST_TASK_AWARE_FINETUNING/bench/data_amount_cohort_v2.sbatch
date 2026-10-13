#!/bin/bash
#SBATCH --job-name=data_amount_cohort100
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:25:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_CONDITION=${1:?Pass oracle_27b_32k or oracle_27b_16k_2ep}
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/data_amount_cohort_cache/$QA_CONDITION
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_REPORT_ROOT/data_amount_cohort100"
printf 'Active development100 evaluation; retain partial responses for exact resume.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
printf 'Source-grounded development100 readout; retain raw evidence and responses.\n' > "$QA_REPORT_ROOT/data_amount_cohort100/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_data_amount_cohort_v2.py" \
    --evaluation-design "$QA_ROOT/bench/DATA_AMOUNT_32K_EVALUATION_V2.json" \
    --design "$QA_ROOT/bench/MODEL_COMPARISON_DESIGN.json" \
    --bench "$QA_ROOT/bench/run_qa_confirmation_v3.py" \
    --harness-4b "$QA_ROOT/bench/run_qa_encoder4b_v1.py" \
    --material-dir "$QA_ROOT/model_comparison_material" --reports "$QA_REPORT_ROOT" \
    --selection "$QA_ROOT/bench/MODEL_COMPARISON_COHORT100_SELECTION.json" \
    --p2-root "$HOME/orcd/scratch/activation/KEEP/P2" \
    --p4-root "$HOME/orcd/scratch/activation/KEEP/P4" \
    --condition "$QA_CONDITION" --output "$QA_REPORT_ROOT/data_amount_cohort100" \
    --seconds "${QA_EVAL_SECONDS:-1200}"
