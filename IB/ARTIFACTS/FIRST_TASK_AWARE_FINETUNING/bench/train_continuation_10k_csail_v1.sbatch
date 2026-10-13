#!/bin/bash
#SBATCH --job-name=qa_continue_10k
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
QA_ROOT=$QA_BASE/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports QA_P2_ROOT=$QA_BASE/KEEP/P2
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$QA_BASE/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QA_CAMPAIGN_DEADLINE
cd "$QA_SOURCE_ROOT"
"$QA_BASE/venv/bin/python" - "$@" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
import activation.ac_model.ac_model as ac_module
import activation.ac_model.ac_model_training as training_module
import activation.ac_model.ac_model_utils as utils_module
root=Path('/data/scratch/ngom/activation/taskaware')
if len(sys.argv) not in (2,3) or (len(sys.argv)==3 and sys.argv[2]!='--resume'):
    raise SystemExit('Pass final_lr or rewarm, optionally --resume')
mode=sys.argv[1];name='continue32k_10k_'+mode
if mode not in ('final_lr','rewarm'):raise SystemExit('Unknown continuation arm')
def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
design=json.loads((root/'bench/CONTINUATION_10K_DESIGN.json').read_text())
assert design['status']=='frozen' and design['questions']==10000 and design['parent_questions']==32000
builder=root/'bench/run_qa_continuation_10k_v1.py'
assert sha(builder)==design['helper_sha256'][builder.name]
for leaf,expected in design['helper_sha256'].items():assert sha(root/'bench'/leaf)==expected,leaf
source={'ac_model':sha(ac_module.__file__),'trainer':sha(training_module.__file__),'utils':sha(utils_module.__file__)}
assert source==design['source_sha256']
material=root/'continuation_10k_material/material_27b_10000.json'
pair_path=root/'continuation_32k_initial/pair.json';pair=json.loads(pair_path.read_text())
assert sha(material)==design['material_sha256']==pair['provenance']['material_sha256']
assert sha(pair_path)==design['initial_pair_sha256']
relocation=json.loads((root/'continuation_32k_initial/CSAIL_RELOCATION.json').read_text())
assert relocation['status']=='complete'
assert relocation['pair_sha256']==design['initial_pair_sha256'] and relocation['material_sha256']==design['material_sha256']
assert pair['provenance']['new_stage_not_fresh_adam']
for relative,spec in pair['provenance']['optimizer']['derived_files'].items():
    path=root/'continuation_32k_initial'/relative
    assert path.stat().st_size==spec['bytes'] and sha(path)==spec['sha256'],relative
report=root/'reports'/name;report.mkdir(parents=True,exist_ok=True)
receipt={'builder_sha256':sha(builder),'source_sha256':source,'design_sha256':sha(root/'bench/CONTINUATION_10K_DESIGN.json'),
         'material_sha256':sha(material),'initial_pair_sha256':sha(pair_path),'mode':mode,
         'all_initial_payload_hashes_verified':True,'optimizer':'All inherited Adam states and clocks preserved; stage metadata clock starts at zero.'}
target=report/'source_receipt.json'
if target.exists():assert json.loads(target.read_text())==receipt,'Resume identity changed'
else:target.write_text(json.dumps(receipt,indent=2)+'\n')
run=root/name;run.mkdir(parents=True,exist_ok=True)
(run/'PRESERVE').write_text('Active32k→10k continuation; preserve paired checkpoints and optimizer.\n')
link=run/'AUTOTUNE'
if not link.exists():link.symlink_to(root.parent/'AUTOTUNE',target_is_directory=True)
os.environ['ACTIVATION_SYNC_ROOT']=str(run)
command=[sys.executable,str(builder),'--arm','joint','--run-name',name,'--continuation-schedule',mode,
         '--material',str(material),'--initial-pair',str(pair_path),'--items','10000','--epochs','1',
         '--seconds','10200','--ratio','0.0625','--evidence-tokens','2048','--max-passages','4','--micro-batch-examples','2']
if len(sys.argv)==3:command.append('--resume')
print(json.dumps({'source_receipt':receipt,'command':command}),flush=True)
os.execv(sys.executable,command)
PY
