#!/bin/bash
#SBATCH --job-name=pi42k_merged32k
#SBATCH --account=csail
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_BASE=/data/scratch/ngom/activation
export QA_ROOT=$QA_BASE/taskaware QA_PYTHON=$QA_BASE/venv/bin/python
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export QA_P2_ROOT=$QA_BASE/KEEP/P2 QA_AUTOTUNE_ROOT=$QA_BASE/AUTOTUNE HF_HOME=$QA_BASE/hf
# Common launcher checks the host-specific frozen design and uses identical scientific settings.
exec bash "$QA_ROOT/bench/train_pi42k_merged32k_v1.sbatch" "$@"
