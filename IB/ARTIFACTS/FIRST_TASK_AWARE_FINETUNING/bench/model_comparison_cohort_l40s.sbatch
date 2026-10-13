#!/bin/bash
#SBATCH --job-name=oracle_cohort100_l40s
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_CONDITION=${1:?Pass frozen_p2, frozen_p4, or one fixed-final arm name}
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/model_comparison_cohort_l40s_cache/$QA_CONDITION
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_REPORT_ROOT/model_comparison_cohort100_l40s"
printf 'Active development100 evaluation; retain partial responses for exact resume.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
printf 'Source-grounded development100 readout; retain raw evidence and responses.\n' > "$QA_REPORT_ROOT/model_comparison_cohort100_l40s/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
"$HOME/activation_venv/bin/python" - <<'PYINFO'
import json, platform, torch, transformers
from importlib.metadata import version, PackageNotFoundError
versions = {}
for package in ("triton", "flash-linear-attention", "causal-conv1d", "flash-attn"):
    try: versions[package] = version(package)
    except PackageNotFoundError: versions[package] = None
print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "packages": versions, "device": torch.cuda.get_device_name(0)}))
PYINFO
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_model_comparison_cohort.py" \
    --design "$QA_ROOT/bench/MODEL_COMPARISON_DESIGN.json" \
    --bench "$QA_ROOT/bench/run_qa_confirmation_v3.py" \
    --harness-4b "$QA_ROOT/bench/run_qa_encoder4b_v1.py" \
    --material-dir "$QA_ROOT/model_comparison_material" --reports "$QA_REPORT_ROOT" \
    --selection "$QA_ROOT/bench/MODEL_COMPARISON_COHORT100_SELECTION.json" \
    --p2-root "$HOME/orcd/scratch/activation/KEEP/P2" \
    --p4-root "$HOME/orcd/scratch/activation/KEEP/P4" \
    --condition "$QA_CONDITION" --output "$QA_REPORT_ROOT/model_comparison_cohort100_l40s" \
    --seconds "${QA_EVAL_SECONDS:-1200}"
