#!/bin/bash
#SBATCH --job-name=continue_cohort100
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_MODE=${1:?Pass final_lr or rewarm}
case "$QA_MODE" in final_lr|rewarm) ;; *) exit 2 ;; esac
QA_ROOT=$HOME/orcd/pool/activation/taskaware
QA_AUTOTUNE_ROOT=$HOME/activation_artifacts/AUTOTUNE
QA_CONDITION=continue32k_10k_$QA_MODE
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/continuation_cohort_cache/$QA_CONDITION
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_REPORT_ROOT/continuation_cohort100"
printf 'Fixed development100 continuation readout; preserve actual evidence and responses.\n' > "$QA_REPORT_ROOT/continuation_cohort100/PRESERVE"
printf 'Active continuation readout cache.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "$QA_AUTOTUNE_ROOT" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_continuation_10k_v1.py" \
    --design "$QA_ROOT/bench/CONTINUATION_10K_DESIGN_ORCD.json" \
    --bench "$QA_ROOT/bench/run_qa_continuation_10k_v1.py" \
    --material "$QA_ROOT/continuation_10k_material/material_27b_10000.json" \
    --initial-pair "$QA_ROOT/continuation_32k_initial/pair.json" \
    --reports "$QA_REPORT_ROOT" --selection "$QA_ROOT/bench/MODEL_COMPARISON_COHORT100_SELECTION.json" \
    --condition "$QA_CONDITION" --output "$QA_REPORT_ROOT/continuation_cohort100" \
    --seconds "${QA_EVAL_SECONDS:-2100}"
