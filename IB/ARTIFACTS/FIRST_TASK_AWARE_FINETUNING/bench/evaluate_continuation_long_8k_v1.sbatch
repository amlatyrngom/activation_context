#!/bin/bash
#SBATCH --job-name=continue42k_long8k
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/continuation_long_cohort_cache/continue32k_10k_final_lr__8k
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_ROOT/reports/continuation_long_cohort100"
printf 'Preserve completed42k paired-checkpoint matched8k development readout.\n' > "$QA_ROOT/reports/continuation_long_cohort100/PRESERVE"
printf 'Active matched8k development readout cache.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_continuation_long_8k_v1.py" \
    --design "$QA_ROOT/bench/STRONG_LONG_ADAPTATION_DESIGN_V1.json" \
    --material "$QA_ROOT/strong_long_adaptation_8k_material/material.json" \
    --bench "$QA_ROOT/bench/run_qa_strong_long_v1.py" \
    --selection "$QA_ROOT/bench/MODEL_COMPARISON_COHORT100_SELECTION.json" \
    --reports "$QA_ROOT/reports" \
    --continuation-design "$QA_ROOT/bench/CONTINUATION_10K_DESIGN_ORCD.json" \
    --continuation-initial-pair "$QA_ROOT/continuation_32k_initial/pair.json" \
    --native-readout "$QA_ROOT/reports/continuation_cohort100/continue32k_10k_final_lr.json" \
    --initial8k-readout "$QA_ROOT/reports/strong_long_cohort100/initial32k__8k/result.json" \
    --output "$QA_ROOT/reports/continuation_long_cohort100" --seconds "${QA_EVAL_SECONDS:-2100}"
