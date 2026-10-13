#!/bin/bash
#SBATCH --job-name=fresh42k_native8k
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export QA_P2_ROOT=$HOME/orcd/scratch/activation/KEEP/P2
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
QA_DESIGN=${1:?Pass the reviewed frozen FRESH42K_NATIVE8K_DESIGN_V1.json path}
shift
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" - "$QA_DESIGN" "$@" <<'PY'
import hashlib,json,os,shutil,sys
from pathlib import Path
import activation.ac_model.ac_model as ac_module
import activation.ac_model.ac_model_training as trainer_module
import activation.ac_model.ac_model_utils as utils_module
root=Path(os.environ['QA_REPORT_ROOT']).parent
if sys.argv[2:] not in ([],['--resume']):raise SystemExit('Only design path and optional --resume supported')
resume=bool(sys.argv[2:]);design_path=Path(sys.argv[1]);design=json.loads(design_path.read_text())
if design['status']!='frozen':raise SystemExit('Prepared only: parent choice/root review has not frozen this candidate')
def sha(path):
 with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
for leaf,expected in design['helper_sha256'].items():
 assert sha(root/'bench'/leaf)==expected,leaf
source={'ac_model':sha(ac_module.__file__),'trainer':sha(trainer_module.__file__),'utils':sha(utils_module.__file__)}
assert source==design['source_sha256']
material=Path(design['material_path']);initial=Path(design['initial_pair_path']);parent_readout=Path(design['parent_readout_path'])
assert sha(material)==design['material_sha256'] and sha(initial)==design['initial_pair_sha256']
assert sha(parent_readout)==design['parent_readout_sha256']
pair=json.loads(initial.read_text());optimizer=pair['optimizer_provenance']
assert optimizer['adam_state_preserved_byte_exact'] and optimizer['independent_files_not_hardlinks'] and optimizer['weight_bytes_unchanged']
assert optimizer['adam_parameter_steps']==5250 and optimizer['optimizer_parameter_states']==615
# Verify every independent derived payload including the unchanged-moment optimizer, on compute.
for relative,record in optimizer['derived_files'].items():assert sha(initial.parent/relative)==record['sha256'],relative
# pair.json lives in initial/, while derived-file keys already begin ac/ or reader/.
name=design['run_name'];run=root/name;run.mkdir(parents=True,exist_ok=True)
report=root/'reports'/name;report.mkdir(parents=True,exist_ok=True)
builder=root/'bench/run_qa_fresh42k_native8k_v1.py'
receipt={'builder_sha256':sha(builder),'source_sha256':source,'design_sha256':sha(design_path),
 'material_sha256':sha(material),'initial_pair_sha256':sha(initial),'initial_payload_sha256':design['initial_payload_sha256'],
 'parent_condition':design['parent_condition'],'parent_readout_sha256':sha(parent_readout),
 'optimizer_policy':'Actual Adam5250, reset only stage clock; own paired optimizer on explicit resume.',
 'training_exposure':'8000 FRESH questions ×1;50000 unique and50000 total presentations after completion.'}
path=report/'source_receipt.json'
if path.exists():assert json.loads(path.read_text())==receipt,'Resume source/design changed'
else:
 with path.open('x') as handle:json.dump(receipt,handle,indent=2)
(run/'PRESERVE').write_text('Active fresh8000 native continuation; inherited Adam5250; own paired resumption.\n')
link=run/'AUTOTUNE'
if not link.exists():link.symlink_to(Path.home()/'activation_artifacts/AUTOTUNE',target_is_directory=True)
os.environ['ACTIVATION_SYNC_ROOT']=str(run)
command=[sys.executable,str(builder),'--design',str(design_path),'--parent-readout',str(parent_readout),
 '--arm','joint','--run-name',name,'--material',str(material),'--initial-pair',str(initial),
 '--items','8000','--epochs','1','--seconds',str(design['training_config']['max_duration_seconds']),
 '--encoder-lr','0.000005','--reader-lr','0.000002','--ratio','0.0625','--evidence-tokens','2048',
 '--max-passages','4','--micro-batch-examples','2','--encode-batch-max-tokens','65536',
 '--gradient-checkpointing-min-tokens','1700']
if resume:command.append('--resume')
print(json.dumps({'receipt':receipt,'command':command}),flush=True);os.execv(sys.executable,command)
PY
