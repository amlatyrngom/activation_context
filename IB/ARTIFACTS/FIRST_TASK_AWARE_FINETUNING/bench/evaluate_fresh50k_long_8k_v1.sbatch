#!/bin/bash
#SBATCH --job-name=fresh50k_long8k
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
QA_DESIGN=${1:?Pass the frozen FRESH42K_NATIVE8K_DESIGN_V1.json path used by training/native100}
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/fresh50k_long_cohort_cache/continue42k_8k_final_lr_floor__8k
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_ROOT/reports/continuation_long_cohort100/continue42k_8k_final_lr_floor__8k"
printf 'Completed50k fresh-native paired checkpoint at matched8k development inputs.\n' > "$QA_ROOT/reports/continuation_long_cohort100/continue42k_8k_final_lr_floor__8k/PRESERVE"
printf 'Active matched8k development readout cache.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_fresh50k_long_8k_v1.py" \
    --design "$QA_ROOT/bench/STRONG_LONG_ADAPTATION_DESIGN_V1.json" \
    --material "$QA_ROOT/strong_long_adaptation_8k_material/material.json" \
    --bench "$QA_ROOT/bench/run_qa_strong_long_v1.py" \
    --selection "$QA_ROOT/bench/MODEL_COMPARISON_COHORT100_SELECTION.json" \
    --reports "$QA_ROOT/reports" --runs-root "$QA_ROOT" \
    --fresh-design "$QA_DESIGN" \
    --native-readout "$QA_ROOT/reports/fresh42k_native_cohort100/continue42k_8k_final_lr_floor__2k/result.json" \
    --initial8k-readout "$QA_ROOT/reports/strong_long_cohort100/initial32k__8k/result.json" \
    --output "$QA_ROOT/reports/continuation_long_cohort100" --seconds "${QA_EVAL_SECONDS:-1350}"
