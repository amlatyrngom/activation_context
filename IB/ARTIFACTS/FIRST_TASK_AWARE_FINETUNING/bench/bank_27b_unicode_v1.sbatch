#!/bin/bash
#SBATCH --job-name=qa_bank_27b
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_synthesis_unicode_v1 QA_REPORT_ROOT=$QA_ROOT/reports QA_BANK_SOURCES=$QA_ROOT/bank_sources_200k.json
export QA_BANK_SOURCE_EXTENSIONS=$QA_ROOT/bank_source_extensions
export QA_RESERVED_MATERIAL=$QA_ROOT/hillclimb_material/confirmation.json
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf ACTIVATION_SYNC_ROOT=$QA_ROOT/bank_cache_27b
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false TZ=UTC
QA_MODEL_SNAPSHOT=$(cat "$QA_ROOT/model_prefetch_27b/snapshot_path.txt")
test -f "$QA_MODEL_SNAPSHOT/config.json"
mkdir -p "$ACTIVATION_SYNC_ROOT"
# The campaign owner atomically adds future reserved questions to this simple file.
if [[ ! -e "$QA_ROOT/qa27b_reserved_questions.json" ]]; then
    printf '{"questions":[]}\n' > "$QA_ROOT/qa27b_reserved_questions.json"
fi
if [[ ! -e "$QA_ROOT/qa27b_target.json" ]]; then
    printf '{"target_questions":32000}\n' > "$QA_ROOT/qa27b_target.json"
fi
cd "$QA_SOURCE_ROOT"
QA_WORKER_PID=
QA_FORWARDED_SIGNAL=0
QA_CANCELLED=0
forward_stop() {
    QA_FORWARDED_SIGNAL=1
    if [[ -n "$QA_WORKER_PID" ]]; then kill -USR1 "$QA_WORKER_PID" 2>/dev/null || true; fi
}
trap forward_stop USR1
trap 'QA_CANCELLED=1; forward_stop' TERM
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/generate_bank_27b.py" \
    --model-id Qwen/Qwen3.5-27B --model-path "$QA_MODEL_SNAPSHOT" \
    --bank-name synthesis_bank_27b --target 32000 --target-file "$QA_ROOT/qa27b_target.json" --seconds 10200 --microbatch 64 \
    --reserved-material "$QA_ROOT/data_budget_material/material_2000.json" \
    --reserved-material "$QA_ROOT/data_budget_material/confirmation.json" \
    --reserved-material "$QA_ROOT/qa27b_reserved_questions.json" &
QA_WORKER_PID=$!
while true; do
    set +e
    wait "$QA_WORKER_PID"
    QA_WORKER_STATUS=$?
    set -e
    if [[ "$QA_FORWARDED_SIGNAL" == 1 && "$QA_WORKER_STATUS" -gt 128 ]]; then
        QA_FORWARDED_SIGNAL=0
        continue
    fi
    break
done
if [[ "$QA_CANCELLED" == 1 ]]; then exit 143; fi
if [[ "$QA_WORKER_STATUS" == 75 ]]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
# 76 means a source family is exhausted; stop visibly, without a futile requeue.
exit "$QA_WORKER_STATUS"
