#!/bin/bash
#SBATCH --job-name=recon_retention_420
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
export QA_ROOT=${QA_ROOT:-$HOME/orcd/pool/activation/taskaware}
QA_PYTHON=${QA_PYTHON:-$HOME/activation_venv/bin/python}
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1
export PYTHONPATH=$QA_SOURCE_ROOT
export HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUBLAS_WORKSPACE_CONFIG
cd "$QA_SOURCE_ROOT"
"$QA_PYTHON" - <<'PY'
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(os.environ['QA_ROOT'])
manifest_path=root/'bench/RECONSTRUCTION_RETENTION_V1.json'
manifest=json.loads(manifest_path.read_text())
for name,expected in manifest['helper_sha256'].items():
 with (root/'bench'/name).open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==expected,name
worker=root/'bench/evaluate_reconstruction_retention_v1.py'
with worker.open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==manifest['worker_sha256']
output=root/'reports/reconstruction_retention_v1';output.mkdir(parents=True,exist_ok=True)
(output/'PRESERVE').write_text('User-requested original reconstruction retention after QA; read-only paired evaluations.\n')
cache=root/'reconstruction_retention_v1_cache';cache.mkdir(exist_ok=True)
(cache/'PRESERVE').write_text('Active reconstruction readout cache.\n')
link=cache/'AUTOTUNE'
if not link.exists():link.symlink_to(Path.home()/'activation_artifacts/AUTOTUNE',target_is_directory=True)
for condition in manifest['conditions']:
 command=[sys.executable,str(worker),'--manifest',str(manifest_path),
  '--passages',str(root/'bench/reconstruction_retention_held420_v1.json'),
  '--output',str(output),'--condition',condition]
 print(json.dumps({'condition':condition,'read_only':True,'command':command}),flush=True)
 subprocess.run(command,env=dict(os.environ,ACTIVATION_SYNC_ROOT=str(cache)),check=True)
 result=json.loads((output/condition/'result.json').read_text())
 assert result['status']=='complete' and len(result['per_item'])==420
 print(json.dumps({'condition':condition,'summary':result['summary']}),flush=True)
sys.path.insert(0,str(root/'bench'))
from evaluate_reconstruction_retention_v1 import report
report(output,manifest,'all four checkpoints complete',condition,420)
PY
