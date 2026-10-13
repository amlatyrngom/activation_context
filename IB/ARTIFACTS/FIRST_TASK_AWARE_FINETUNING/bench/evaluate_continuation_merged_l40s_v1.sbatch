#!/bin/bash
#SBATCH --job-name=continue42k_merged_l40s
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
QA_CONTEXT=32k
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/continuation_merged_l40s_cache/continue32k_10k_final_lr__merged_32k_l40s
QA_OUTPUT=$QA_ROOT/reports/continuation_merged_l40s_cohort100
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_OUTPUT/continue32k_10k_final_lr__merged_32k_l40s"
printf 'Frozen42k L40S merged32k; fresh device-specific scores and actual answers.\n' > "$QA_OUTPUT/continue32k_10k_final_lr__merged_32k_l40s/PRESERVE"
printf 'Active merged development readout cache.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_continuation_merged_l40s_v1.py" \
    --design "$QA_ROOT/bench/STRONG_LONG_ADAPTATION_DESIGN_V1.json" \
    --material "$QA_ROOT/strong_long_adaptation_8k_material/material.json" \
    --bench "$QA_ROOT/bench/run_qa_strong_long_v1.py" \
    --selection "$QA_ROOT/bench/MODEL_COMPARISON_COHORT100_SELECTION.json" \
    --reports "$QA_ROOT/reports" \
    --continuation-design "$QA_ROOT/bench/CONTINUATION_10K_DESIGN_ORCD.json" \
    --continuation-initial-pair "$QA_ROOT/continuation_32k_initial/pair.json" \
    --native-readout "$QA_ROOT/reports/continuation_cohort100/continue32k_10k_final_lr.json" \
    --piecewise-readout "$QA_ROOT/reports/continuation_long_cohort100/continue32k_10k_final_lr__$QA_CONTEXT/result.json" \
    --adam-receipt "$QA_ROOT/gentle42k_final_lr_familiar4k_v1/initial/pair.json" \
    --context "$QA_CONTEXT" --output "$QA_OUTPUT" --seconds "${QA_EVAL_SECONDS:-2250}"
