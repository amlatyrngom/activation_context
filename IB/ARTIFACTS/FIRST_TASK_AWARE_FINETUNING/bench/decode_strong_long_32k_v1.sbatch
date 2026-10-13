#!/bin/bash
#SBATCH --job-name=strong_long_32k_answers
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/strong_long_32k_answers_cache
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_ROOT/reports/strong_long_32k_answers100"
printf 'Preserve final32k fixed100 actual answers and reused NLL provenance.\n' > "$QA_ROOT/reports/strong_long_32k_answers100/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/decode_strong_long_32k_v1.py" \
    --design "$QA_ROOT/bench/STRONG_LONG_ADAPTATION_DESIGN_V1.json" \
    --material "$QA_ROOT/strong_long_adaptation_8k_material/material.json" \
    --bench "$QA_ROOT/bench/run_qa_strong_long_v1.py" \
    --selection "$QA_ROOT/bench/MODEL_COMPARISON_COHORT100_SELECTION.json" \
    --reports "$QA_ROOT/reports" --runs-root "$QA_ROOT" \
    --nll-input "$QA_ROOT/reports/strong_long_cohort100/strong32k_piecewise_8k_8k__32k/result.json" \
    --output "$QA_ROOT/reports/strong_long_32k_answers100" --seconds "${QA_DECODE_SECONDS:-1500}"
