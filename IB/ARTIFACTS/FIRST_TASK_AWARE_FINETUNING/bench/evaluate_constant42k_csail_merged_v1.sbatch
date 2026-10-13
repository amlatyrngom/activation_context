#!/bin/bash
#SBATCH --job-name=parent42k_merged100
#SBATCH --account=csail
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_BASE=${QA_BASE:-/data/scratch/ngom/activation}
QA_ROOT=${QA_ROOT:-$QA_BASE/taskaware}
QA_PYTHON=${QA_PYTHON:-$QA_BASE/venv/bin/python}
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=${QA_HF_HOME:-$QA_BASE/hf} HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/constant42k_csail_merged_cache
QA_OUTPUT=$QA_ROOT/reports/continuation_merged_csail_cohort100
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_OUTPUT/continue32k_10k_final_lr__merged_32k_h200"
printf 'Independent frozen42k merged32k H200 readout; actual answers and fresh baselines.\n' > "$QA_OUTPUT/continue32k_10k_final_lr__merged_32k_h200/PRESERVE"
printf 'Active independent frozen42k readout cache.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "${QA_READOUT_AUTOTUNE:-$QA_BASE/AUTOTUNE}" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$QA_PYTHON" "$QA_ROOT/bench/evaluate_constant42k_csail_merged_v1.py" \
 --manifest "$QA_ROOT/bench/CONSTANT42K_CSAIL_MERGED_MANIFEST_V1.json" \
 --stage "$QA_ROOT/constant42k_adam5250_staging_v1" \
 --bench "$QA_ROOT/bench/run_qa_strong_long_v1.py" \
 --selection "$QA_ROOT/bench/MODEL_COMPARISON_COHORT100_SELECTION.json" \
 --output "$QA_OUTPUT" --seconds "${QA_EVAL_SECONDS:-2250}" --decode-full
