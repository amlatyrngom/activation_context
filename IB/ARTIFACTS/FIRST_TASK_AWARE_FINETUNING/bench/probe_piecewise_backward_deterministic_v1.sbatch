#!/bin/bash
#SBATCH --job-name=qa_backward_deterministic
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_P2_ROOT=$HOME/orcd/scratch/activation/KEEP/P2
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
export ACTIVATION_SYNC_ROOT=$QA_ROOT/piecewise_backward_deterministic_v1
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_ROOT/reports/piecewise_backward_deterministic_v1"
printf 'Deterministic diagnostic only; retain ordinary failed gates. No production setting changes or fallback.\n' > "$QA_ROOT/reports/piecewise_backward_deterministic_v1/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" - <<'PY'
import hashlib,json,os,runpy,sys
from pathlib import Path
import torch
root=Path.home()/'orcd/pool/activation/taskaware'
worker=root/'bench/probe_piecewise_backward_v1.py'
assert hashlib.sha256(worker.read_bytes()).hexdigest()=='fb0fc3c64c505defc623ebc5e154306387c1fabb970fec51fc499298af7ae76c'
assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
assert not torch.cuda.is_initialized(), 'Deterministic policy must be established before CUDA initialization'
torch.use_deterministic_algorithms(True,warn_only=False)
output=root/'reports/piecewise_backward_deterministic_v1'
metadata={'scope':'One deterministic diagnostic; production ordinary-kernel fits are separate. Original tolerances unchanged.',
    'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],
    'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
    'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
    'worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'status':'running'}
meta_path=output/'deterministic_settings.json'
assert not meta_path.exists()
def publish():
    tmp=meta_path.with_suffix('.next');tmp.write_text(json.dumps(metadata,indent=2)+'\n');tmp.replace(meta_path)
publish()
sys.path.insert(0,str(worker.parent))
sys.argv=[str(worker),'--worker','proof','--material',str(root/'piecewise_training_4k_8k_material/material.json'),
    '--bench',str(root/'bench/run_qa_piecewise_v1.py'),'--failed-report',str(root/'reports/piecewise_27b_4k_8k'),
    '--p2-root',os.environ['QA_P2_ROOT'],'--output',str(output)]
try:
    runpy.run_path(str(worker),run_name='__main__')
    metadata['status']='complete'
except BaseException as error:
    metadata.update(status='failed',error=repr(error));raise
finally:
    metadata['deterministic_algorithms_at_exit']=torch.are_deterministic_algorithms_enabled()
    metadata['warn_only_at_exit']=torch.is_deterministic_algorithms_warn_only_enabled();publish()
PY
