#!/bin/bash
#SBATCH --job-name=pi960_native_final100
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
export QA_ROOT=${QA_ROOT:-$HOME/orcd/pool/activation/taskaware}
QA_PYTHON=${QA_PYTHON:-$HOME/activation_venv/bin/python}
QA_DESIGN=${1:?Pass the frozen FLOOR or2X native960 design}
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=${QA_HF_HOME:-$HOME/orcd/scratch/activation/hf} HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QA_READOUT_AUTOTUNE=${QA_READOUT_AUTOTUNE:-$HOME/activation_artifacts/AUTOTUNE}
unset CUBLAS_WORKSPACE_CONFIG
cd "$QA_SOURCE_ROOT"
"$QA_PYTHON" - "$QA_DESIGN" <<'PY'
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(os.environ['QA_ROOT']);design_path=Path(sys.argv[1]);design=json.loads(design_path.read_text())
manifest_path=root/'bench/PI42K_NATIVE960_LR_READOUT_MANIFEST_V1.json';manifest=json.loads(manifest_path.read_text())
for leaf,expected in manifest['helper_sha256'].items():
 with (root/'bench'/leaf).open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==expected,leaf
assert design['status']=='frozen' and design['run_name'] in ('pi42k_native2k_fresh960_floor_v1','pi42k_native2k_fresh960_2x_v1')
cell=design['run_name']+'__2k'
output=root/'reports/pi42k_native960_lr_cohort100';output.mkdir(parents=True,exist_ok=True)
(output/'PRESERVE').write_text('Own final native2k960 LR diagnostics: fresh100AC answers/NLL and final-reader full/no-context; no confirmation.\n')
cache=root/'pi42k_native960_lr_readout_cache'/cell;cache.mkdir(parents=True,exist_ok=True)
(cache/'PRESERVE').write_text('Active read-only native960 own-final readout cache.\n')
link=cache/'AUTOTUNE'
if not link.exists():link.symlink_to(Path(os.environ['QA_READOUT_AUTOTUNE']),target_is_directory=True)
env=dict(os.environ,ACTIVATION_SYNC_ROOT=str(cache))
command=[sys.executable,str(root/'bench/evaluate_pi42k_native960_lr_v1.py'),
 '--manifest',str(manifest_path),'--design',str(design_path),'--material',design['material_path'],
 '--subset',design['subset_path'],'--initial-pair',design['initial_pair_path'],
 '--bench',str(root/'bench/run_qa_pi42k_native960_lr_v1.py'),
 '--selection',str(root/'bench/MODEL_COMPARISON_COHORT100_SELECTION.json'),
 '--reports',str(root/'reports'),'--runs-root',str(root),'--output',str(output),
 '--checkpoint','final','--context','2k','--seconds','840']
print(json.dumps({'cell':cell,'command':command,'read_only':True}),flush=True)
subprocess.run(command,env=env,check=True)
result=json.loads((output/cell/'result.json').read_text())
if result['status']!='complete':raise SystemExit(75)
print(json.dumps({'cell':cell,'status':'complete','actual_answers':len(result['rows']),'optimizer_updates':result['optimizer_updates']}),flush=True)
PY
