#!/bin/bash
#SBATCH --job-name=long_qa_parent_v1
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports QA_P2_ROOT=$HOME/orcd/scratch/activation/KEEP/P2
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACTIVATION_SYNC_ROOT=$QA_ROOT/long_qa_whole_parent_v1_pilot
mkdir -p "$ACTIVATION_SYNC_ROOT"
printf 'Portable whole-parent QA pilot; agent owns retention.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
exec "$HOME/activation_venv/bin/python" "$QA_ROOT/bench/generate_long_qa_whole_parent_v1.py" \
    --sources "$QA_ROOT/long_qa_whole_parent_v1_pilot/parents_75.jsonl" \
    --run-name long_qa_whole_parent_v1_pilot --batch-size 2 --seconds 3300
