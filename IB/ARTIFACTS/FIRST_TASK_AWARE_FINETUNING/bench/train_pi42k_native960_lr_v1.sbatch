#!/bin/bash
#SBATCH --job-name=pi42k_native960_lr
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=${QA_ROOT:-$HOME/orcd/pool/activation/taskaware}
QA_PYTHON=${QA_PYTHON:-$HOME/activation_venv/bin/python}
export QA_SOURCE_ROOT=${QA_SOURCE_ROOT:-$QA_ROOT/train_ws_night_v1} QA_REPORT_ROOT=${QA_REPORT_ROOT:-$QA_ROOT/reports}
export QA_P2_ROOT=${QA_P2_ROOT:-$HOME/orcd/scratch/activation/KEEP/P2}
export QA_AUTOTUNE_ROOT=${QA_AUTOTUNE_ROOT:-$HOME/activation_artifacts/AUTOTUNE}
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=${HF_HOME:-$HOME/orcd/scratch/activation/hf} HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
QA_DESIGN=${1:?Pass the reviewed frozen PI42K_NATIVE960_FLOOR_DESIGN_V1.json orPI42K_NATIVE960_2X_DESIGN_V1.json path}
shift
unset CUBLAS_WORKSPACE_CONFIG
cd "$QA_SOURCE_ROOT"
"$QA_PYTHON" - "$QA_DESIGN" "$@" <<'PY'
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
material=Path(design['material_path']);initial=Path(design['initial_pair_path']);subset=Path(design['subset_path'])
assert sha(material)==design['material_sha256'] and sha(initial)==design['initial_pair_sha256']
assert sha(subset)==design['subset_sha256']
pair=json.loads(initial.read_text());optimizer=pair['optimizer_provenance']
assert optimizer['adam_state_preserved_byte_exact'] and optimizer['independent_files_not_hardlinks'] and optimizer['weight_bytes_unchanged']
assert optimizer['adam_parameter_steps']==5250 and optimizer['optimizer_parameter_states']==615
# Recheck every scientific file, including the unchanged-moment optimizer, on compute.
# Host relocation may replace preservation flags and omit historical README files.
scientific=('ac/ac_modules.pt','ac/optimizer.pt','ac/side_lora/adapter_config.json',
 'ac/side_lora/adapter_model.safetensors','reader/adapter_config.json','reader/adapter_model.safetensors')
for relative in scientific:
 record=optimizer['derived_files'][relative]
 assert sha(initial.parent/relative)==record['sha256'],relative
# pair.json lives in initial/, while derived-file keys already begin ac/ or reader/.
name=design['run_name'];run=root/name
assert run.resolve()==Path(design['run_root']).resolve(),'Run root differs from host-specific frozen design'
run.mkdir(parents=True,exist_ok=True)
report=root/'reports'/name;report.mkdir(parents=True,exist_ok=True)
builder=root/'bench/run_qa_pi42k_native960_lr_v1.py'
receipt={'builder_sha256':sha(builder),'source_sha256':source,'design_sha256':sha(design_path),
 'material_sha256':sha(material),'initial_pair_sha256':sha(initial),'initial_payload_sha256':design['initial_payload_sha256'],
 'parent_condition':design['parent_condition'],'subset_sha256':sha(subset),
 'optimizer_policy':'Actual Adam5250, reset only stage clock; own paired optimizer on explicit resume.',
 'training_exposure':'960 FRESH questions ×1;42960 unique and42960 total presentations after completion.'}
path=report/'source_receipt.json'
if path.exists():assert json.loads(path.read_text())==receipt,'Resume source/design changed'
else:
 with path.open('x') as handle:json.dump(receipt,handle,indent=2)
(run/'PRESERVE').write_text('Active paired native2k fresh960 LR diagnostic; inherited Adam5250; own paired resumption.\n')
link=run/'AUTOTUNE'
if not link.exists():link.symlink_to(Path(os.environ['QA_AUTOTUNE_ROOT']),target_is_directory=True)
os.environ['ACTIVATION_SYNC_ROOT']=str(run)
command=[sys.executable,str(builder),'--design',str(design_path),'--selection',str(subset),
 '--arm','joint','--run-name',name,'--material',str(material),'--initial-pair',str(initial),
 '--items','960','--epochs','1','--seconds',str(design['training_config']['max_duration_seconds']),
 '--encoder-lr',str(design['training_config']['learning_rate_ac']),'--reader-lr',str(design['training_config']['learning_rate_target_lora']),'--ratio','0.0625','--evidence-tokens','2048',
 '--max-passages','4','--micro-batch-examples','1','--encode-batch-max-tokens','65536',
 '--gradient-checkpointing-min-tokens','0']
if resume:command.append('--resume')
print(json.dumps({'receipt':receipt,'command':command}),flush=True);os.execv(sys.executable,command)
PY
