#!/bin/bash
#SBATCH --job-name=prepare_qa10k_cont
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/prepare_continuation_10k.py" \
  --root "$QA_ROOT" --parent-pin "$QA_ROOT/bench/CONTINUATION_PARENT_32K.json"
