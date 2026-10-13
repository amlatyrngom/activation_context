#!/bin/bash
#SBATCH --job-name=gentle42k_long32k
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
QA_DESIGN=${1:?Pass the frozen GENTLE42K_LONG_DESIGN_V1.json used by training/own8k100}
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/gentle42k_long_cohort_cache/gentle42k_final_lr_familiar4k_8k__32k
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_ROOT/reports/gentle42k_long_cohort100/gentle42k_final_lr_familiar4k_8k__32k"
printf 'Completed gentle familiar4k adaptation: matched32k development100 readout.\n' > "$QA_ROOT/reports/gentle42k_long_cohort100/gentle42k_final_lr_familiar4k_8k__32k/PRESERVE"
printf 'Active matched32k development readout cache.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_gentle42k_long_32k_v1.py" \
    --design "$QA_ROOT/bench/STRONG_LONG_ADAPTATION_DESIGN_V1.json" \
    --material "$QA_ROOT/strong_long_adaptation_8k_material/material.json" \
    --bench "$QA_ROOT/bench/run_qa_strong_long_v1.py" \
    --selection "$QA_ROOT/bench/MODEL_COMPARISON_COHORT100_SELECTION.json" \
    --reports "$QA_ROOT/reports" --runs-root "$QA_ROOT" \
    --gentle-design "$QA_DESIGN" \
    --own8k-readout "$QA_ROOT/reports/gentle42k_long_cohort100/gentle42k_final_lr_familiar4k_8k__8k/result.json" \
    --parent32k-readout "$QA_ROOT/reports/continuation_long_cohort100/continue32k_10k_final_lr__32k/result.json" \
    --output "$QA_ROOT/reports/gentle42k_long_cohort100" --seconds "${QA_EVAL_SECONDS:-1650}"
