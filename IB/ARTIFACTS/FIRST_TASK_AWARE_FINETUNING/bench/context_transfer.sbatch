#!/bin/bash
#SBATCH --job-name=context_transfer100
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_GROUP=${1:?Pass 2b or 4b}
case "$QA_GROUP" in
  2b)
    QA_ARMS=(oracle_27b_16k_r16 oracle_27b_16k_r8)
    # A full allocation must fit inside the authorized ordinary campaign window.
    if (( $(date -u +%s) + 2700 > $(date -u -d '2026-09-22 12:00:00' +%s) )); then
      printf 'DEFERRED: insufficient campaign time for the optional2B context screen.\n'
      exit 0
    fi
    ;;
  4b) QA_ARMS=(encoder4b_oracle27b_16k_r16) ;;
  *) printf 'Expected2b or4b\n' >&2; exit 2 ;;
esac
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$QA_REPORT_ROOT/model_context_transfer100"
printf 'Fixed development100 context transfer; retain input and per-item score evidence.\n' > "$QA_REPORT_ROOT/model_context_transfer100/PRESERVE"
cd "$QA_SOURCE_ROOT"
for QA_ARM in "${QA_ARMS[@]}"; do
  for QA_CONTEXT in 2k 32k; do
    export ACTIVATION_SYNC_ROOT=$QA_ROOT/context_transfer_cache/${QA_ARM}__${QA_CONTEXT}
    mkdir -p "$ACTIVATION_SYNC_ROOT"
    printf 'Active evaluation cache; retain until result verified.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
    ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
    "$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_context_transfer.py" \
      --design "$QA_ROOT/bench/MODEL_COMPARISON_DESIGN.json" \
      --bench "$QA_ROOT/bench/run_qa_confirmation_v3.py" \
      --harness-4b "$QA_ROOT/bench/run_qa_encoder4b_v1.py" \
      --material-dir "$QA_ROOT/model_comparison_material" --reports "$QA_REPORT_ROOT" \
      --selection "$QA_ROOT/bench/MODEL_COMPARISON_COHORT100_SELECTION.json" \
      --condition "$QA_ARM" --context "$QA_CONTEXT" \
      --output "$QA_REPORT_ROOT/model_context_transfer100" --seconds 1500
  done
done
