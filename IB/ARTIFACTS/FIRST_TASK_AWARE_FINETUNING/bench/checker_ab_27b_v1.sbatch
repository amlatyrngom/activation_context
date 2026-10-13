#!/bin/bash
#SBATCH --job-name=checker_ab_27b_v1
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export ACTIVATION_SYNC_ROOT=$QA_ROOT/checker_ab_27b_v1
QA_MODEL_SNAPSHOT=$(cat "$QA_ROOT/model_prefetch_27b/snapshot_path.txt")
test -f "$QA_MODEL_SNAPSHOT/config.json"
mkdir -p "$ACTIVATION_SYNC_ROOT"
printf 'Bounded 27B checker comparison; diagnostic data only.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
cd "$QA_SOURCE_ROOT"
exec "$HOME/activation_venv/bin/python" "$QA_ROOT/bench/run_checker_ab_27b_v1.py" \
    --cases "$QA_ROOT/checker_ab_27b_v1/cases_23.jsonl" --model-path "$QA_MODEL_SNAPSHOT" --run-name checker_ab_27b_v1
