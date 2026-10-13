#!/bin/bash
#SBATCH --job-name=pi960_h200
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_PARENT=$HOME/orcd/pool/activation/taskaware
export QA_ROOT=$QA_PARENT/pi960_h200_v1
export QA_SOURCE_ROOT=$QA_PARENT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
exec bash "$QA_PARENT/bench/train_pi42k_merged32k_v1.sbatch" "$@"
