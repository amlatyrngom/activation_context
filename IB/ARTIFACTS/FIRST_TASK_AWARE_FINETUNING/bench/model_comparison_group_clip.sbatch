#!/bin/bash
#SBATCH --job-name=qa_group_clip
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:30:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
QA_RUN_NAME=$1
QA_MATERIAL=$QA_ROOT/$2
QA_ITEMS=$3
QA_EPOCHS=$4
QA_ENCODER_LR=$5
QA_READER_LR=$6
shift 6
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_group_clip_v1 QA_REPORT_ROOT=$QA_ROOT/reports QA_P2_ROOT=$HOME/orcd/scratch/activation/KEEP/P2
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACTIVATION_SYNC_ROOT=$QA_ROOT/$QA_RUN_NAME
unset QA_CAMPAIGN_DEADLINE
mkdir -p "$ACTIVATION_SYNC_ROOT"
printf 'Active overnight experiment; retain paired saves and reports.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
"$HOME/activation_venv/bin/python" - <<'PYCODE'
import hashlib, json, os
from pathlib import Path
source=Path(os.environ['QA_SOURCE_ROOT'])
assert hashlib.sha256((source/'activation/ac_model/ac_model_training.py').read_bytes()).hexdigest()=='8ae0a9fb4c2a514c84e9baee3f3619d5c06961800a7268f36ada85246f2af1de'
run=Path(os.environ['ACTIVATION_SYNC_ROOT']).name
report=Path(os.environ['QA_REPORT_ROOT'])/run;report.mkdir(parents=True,exist_ok=True)
variant=json.loads((source/'SOURCE_VARIANT.json').read_text());variant.update(status='training source selected',run=run)
(report/'experiment_variant.json').write_text(json.dumps(variant,indent=2)+'\n')
PYCODE
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/run_qa_comparison_v1.py" \
    --arm joint --run-name "$QA_RUN_NAME" --material "$QA_MATERIAL" \
    --items "$QA_ITEMS" --epochs "$QA_EPOCHS" --seconds "${QA_TRAIN_SECONDS:-14400}" \
    --encoder-lr "$QA_ENCODER_LR" --reader-lr "$QA_READER_LR" "$@"
