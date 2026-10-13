#!/bin/bash
#SBATCH --job-name=qa_reference_reuse
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -euo pipefail
"$HOME/activation_venv/bin/python" "$HOME/orcd/pool/activation/taskaware/bench/reuse_base_references.py"
