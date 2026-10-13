#!/bin/bash
#SBATCH --job-name=checker_prefill_27b
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
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export ACTIVATION_SYNC_ROOT=$QA_ROOT/checker_prefill_27b_v1
QA_MODEL_SNAPSHOT=$(cat "$QA_ROOT/model_prefetch_27b/snapshot_path.txt")
test -f "$QA_MODEL_SNAPSHOT/config.json"
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_REPORT_ROOT/checker_prefill_27b_v1"
printf 'Uncached checker throughput probe; diagnostic only.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
cd "$QA_SOURCE_ROOT"
for QA_PREFILL_TOKENS in 2048 8192; do
    QA_ARM_DIR=$QA_REPORT_ROOT/checker_prefill_27b_v1/p$QA_PREFILL_TOKENS
    mkdir -p "$QA_ARM_DIR"
    test ! -e "$QA_ARM_DIR/engine.log"
    "$HOME/activation_venv/bin/python" "$QA_ROOT/bench/run_checker_prefill_27b_v1.py" \
        --cases "$QA_ROOT/checker_prefill_27b_v1/cases_128.jsonl" --model-path "$QA_MODEL_SNAPSHOT" \
        --prefill-tokens "$QA_PREFILL_TOKENS" --run-name checker_prefill_27b_v1 > "$QA_ARM_DIR/engine.log" 2>&1
done
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/run_checker_prefill_27b_v1.py" \
    --cases "$QA_ROOT/checker_prefill_27b_v1/cases_128.jsonl" --run-name checker_prefill_27b_v1 --summarize
