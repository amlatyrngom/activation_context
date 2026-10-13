#!/bin/bash
#SBATCH --job-name=passage_layout32k_cohort100
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:55:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$QA_SOURCE_ROOT"
# Each variant gets a fresh Python/CUDA process and its own disposable runtime cache.
for QA_VARIANT in merged_16 parts_16; do
    export ACTIVATION_SYNC_ROOT=$QA_ROOT/passage_layout32k_cohort100_cache/$QA_VARIANT
    mkdir -p "$ACTIVATION_SYNC_ROOT"
    ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
    "$HOME/activation_venv/bin/python" "$QA_ROOT/bench/screen_passage_variants_qual100.py" \
        --bench "$QA_ROOT/bench/run_qa_screen_v2_snapshot.py" \
        --material "$QA_ROOT/long_adaptation_material/material.json" \
        --initial-pair "$QA_ROOT/long_adaptation_material/initial_pair.json" \
        --output "$QA_REPORT_ROOT/passage_layout32k_cohort100" --variant "$QA_VARIANT" \
        --seconds 1500 --decode-samples 100 --evidence-tokens 32768 --max-passages 64 --micro-batch-examples 1
done
