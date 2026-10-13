#!/bin/bash
#SBATCH --job-name=long_qa_parent_27b
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACTIVATION_SYNC_ROOT=$QA_ROOT/long_qa_whole_parent_27b_v1_pilot
QA_MODEL_SNAPSHOT=$(cat "$QA_ROOT/model_prefetch_27b/snapshot_path.txt")
test -f "$QA_MODEL_SNAPSHOT/config.json"
mkdir -p "$ACTIVATION_SYNC_ROOT"
printf 'Same75-parent pinned27B replay; preserve portable QA and attempts.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
exec "$HOME/activation_venv/bin/python" "$QA_ROOT/bench/generate_long_qa_whole_parent_27b_v1.py" \
    --sources "$QA_ROOT/long_qa_whole_parent_27b_v1_pilot/parents_75.jsonl" --model-path "$QA_MODEL_SNAPSHOT" \
    --run-name long_qa_whole_parent_27b_v1_pilot --batch-size 2 --seconds 2400
