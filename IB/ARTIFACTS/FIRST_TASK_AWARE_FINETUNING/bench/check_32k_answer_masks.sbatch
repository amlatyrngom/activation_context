#!/bin/bash
#SBATCH --job-name=qa_answer_masks
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
set -euo pipefail
export QA_RUN_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_RUN_ROOT/train_ws_night_v1
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export ACTIVATION_SYNC_ROOT=$QA_RUN_ROOT/answer_mask_check_cache TOKENIZERS_PARALLELISM=false
"$HOME/activation_venv/bin/python" "$QA_RUN_ROOT/bench/check_32k_answer_masks.py"
