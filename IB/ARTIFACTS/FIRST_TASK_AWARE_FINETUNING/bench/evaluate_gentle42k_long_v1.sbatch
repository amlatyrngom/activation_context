#!/bin/bash
#SBATCH --job-name=gentle42k_long100
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
QA_DESIGN=${1:?Pass the reviewed frozen GENTLE42K_LONG_DESIGN_V1.json path}
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" - "$QA_DESIGN" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
root=Path.home()/'orcd/pool/activation/taskaware';design_path=Path(sys.argv[1]);design=json.loads(design_path.read_text())
assert design['status']=='frozen'
for leaf,expected in design['helper_sha256'].items():
 with (root/'bench'/leaf).open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==expected,leaf
name=design['run_name'];cache=root/'gentle42k_long_cohort_cache'/name;cache.mkdir(parents=True,exist_ok=True)
(cache/'PRESERVE').write_text('Active readout of completed familiar4000 adaptation pair.\n')
link=cache/'AUTOTUNE'
if not link.exists():link.symlink_to(Path.home()/'activation_artifacts/AUTOTUNE',target_is_directory=True)
os.environ['ACTIVATION_SYNC_ROOT']=str(cache)
output=root/'reports/gentle42k_long_cohort100';output.mkdir(parents=True,exist_ok=True)
(output/'PRESERVE').write_text('Own500-update final8k400 replay plus fixed100 actual AC answers.\n')
command=[sys.executable,str(root/'bench/evaluate_gentle42k_long_v1.py'),
 '--design',str(design_path),'--material',design['material_path'],'--bench',str(root/'bench/run_qa_gentle42k_long_v1.py'),
 '--selection',str(root/'bench/MODEL_COMPARISON_COHORT100_SELECTION.json'),'--reports',str(root/'reports'),
 '--runs-root',str(root),'--initial-pair',design['initial_pair_path'],'--parent-readout',design['parent_readout_path'],
 '--output',str(output),'--condition',name,'--context','8k','--seconds','2100']
print(json.dumps({'command':command,'read_only':True}),flush=True);os.execv(sys.executable,command)
PY
