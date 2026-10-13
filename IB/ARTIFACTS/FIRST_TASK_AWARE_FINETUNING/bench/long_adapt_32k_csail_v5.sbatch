#!/bin/bash
#SBATCH --job-name=long32k_csail_resume
#SBATCH --account=csail
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=/data/scratch/ngom/activation/taskaware
QA_RUN_NAME=long_adapt_32k_csail
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export QA_P2_ROOT=/data/scratch/ngom/activation/KEEP/P2
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=/data/scratch/ngom/activation/hf
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACTIVATION_SYNC_ROOT=$QA_ROOT/$QA_RUN_NAME
export QA_CAMPAIGN_DEADLINE=2026-09-22T12:00:00+00:00
mkdir -p "$ACTIVATION_SYNC_ROOT"
printf 'Exact ORCD update24 migration; keep paired saves. Separate CSAIL writer, no ORCD sync merge.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn /data/scratch/ngom/activation/AUTOTUNE "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
/data/scratch/ngom/activation/venv/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,os,time
root=Path('/data/scratch/ngom/activation/taskaware')
manifest_path=root/'bench/long32k_csail_migration_manifest.json'
raw=manifest_path.read_bytes();manifest=json.loads(raw);sha=hashlib.sha256(raw).hexdigest()
stamp=Path(os.environ['ACTIVATION_SYNC_ROOT'])/'migration_verified.json'
if stamp.exists():
    assert json.loads(stamp.read_text())['manifest_sha256']==sha,'Migration receipt changed'
else:
    for name in manifest['required_for_first_resume']:
        path=root/name;expected=manifest['files'][name]
        assert path.stat().st_size==expected['bytes'],name
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        assert h.hexdigest()==expected['sha256'],'Payload hash mismatch: '+name
    for name,revision in manifest['base_revisions'].items():
        assert (Path(os.environ['HF_HOME'])/'hub'/('models--Qwen--Qwen3.5-'+name)/'refs/main').read_text().strip()==revision
    stamp.write_text(json.dumps({'manifest_sha256':sha,'verified_unix':time.time(),'files':manifest['required_for_first_resume']},indent=2))
print('CSAIL_MIGRATION_PAYLOAD_VERIFIED',sha,flush=True)
PY
/data/scratch/ngom/activation/venv/bin/python "$QA_ROOT/bench/run_qa_night_v5.py" \
    --arm joint --run-name "$QA_RUN_NAME" --resume \
    --resume-baseline-receipt "$QA_ROOT/bench/long32k_resume24_csail_receipt.json" \
    --material "$QA_ROOT/long_adaptation_material/material.json" \
    --initial-pair "$QA_ROOT/long_adaptation_material/initial_pair.json" \
    --items 640 --epochs 1 --seconds 13800 \
    --encoder-lr 5e-5 --reader-lr 2e-5 --ratio 0.0625 --micro-batch-examples 1 \
    --evidence-tokens 32768 --max-passages 64 "$@"
