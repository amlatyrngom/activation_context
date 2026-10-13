#!/bin/bash
#SBATCH --job-name=data_amount_confirmation
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
QA_ARMS=("$@")
if [[ ${#QA_ARMS[@]} == 0 ]]; then
    QA_ARMS=(oracle_27b_32k oracle_27b_16k_2ep)
fi
cd "$QA_SOURCE_ROOT"
for QA_ARM in "${QA_ARMS[@]}"; do
    export ACTIVATION_SYNC_ROOT=$QA_ROOT/data_amount_confirmation_cache/$QA_ARM
    mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_REPORT_ROOT/data_amount_confirmation"
    printf 'Fixed final comparison; preserve partial scores and provenance.\n' > "$QA_REPORT_ROOT/data_amount_confirmation/PRESERVE"
    ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
    "$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_data_amount_confirmation_v2.py" \
        --evaluation-design "$QA_ROOT/bench/DATA_AMOUNT_32K_EVALUATION_V2.json" \
        --bench "$QA_ROOT/bench/run_qa_confirmation_v4.py" \
        --confirmation "$QA_ROOT/model_comparison_32k_material/confirmation.json" \
        --material-dir "$QA_ROOT/model_comparison_material" --reports "$QA_REPORT_ROOT" \
        --output "$QA_REPORT_ROOT/data_amount_confirmation" --arm "$QA_ARM" --seconds 1200
done
