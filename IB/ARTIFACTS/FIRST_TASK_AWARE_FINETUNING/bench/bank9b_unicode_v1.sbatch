#!/bin/bash
#SBATCH --job-name=qa_bank_v2
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_9b_synthesis_unicode_v1 QA_REPORT_ROOT=$QA_ROOT/reports QA_BANK_SOURCES=$QA_ROOT/bank_sources_200k.json
export QA_RESERVED_MATERIAL=$QA_ROOT/hillclimb_material/confirmation.json
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf ACTIVATION_SYNC_ROOT=$QA_ROOT/bank_cache_v2
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false TZ=UTC
mkdir -p "$ACTIVATION_SYNC_ROOT"
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
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/generate_bank.py" --bank-name synthesis_bank_v2 --target 200000 --seconds 10200 --microbatch 32 &
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
# 76 means a source family is exhausted: stop visibly, never loop on empty input.
exit "$QA_WORKER_STATUS"
