#!/bin/bash
#SBATCH --job-name=strong32k_piecewise8k
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export QA_P2_ROOT=$HOME/orcd/scratch/activation/KEEP/P2
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" - "$@" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
import activation.ac_model.ac_model as ac_module
import activation.ac_model.ac_model_training as training_module
import activation.ac_model.ac_model_utils as utils_module
root=Path(os.environ['QA_REPORT_ROOT']).parent
if sys.argv[1:] not in ([],['--resume']):raise SystemExit('Only optional --resume is supported')
resume=bool(sys.argv[1:])
def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
design_path=root/'bench/STRONG_LONG_ADAPTATION_DESIGN_V1.json';design=json.loads(design_path.read_text())
if design['status']!='frozen':raise SystemExit('Larger run is not frozen; await batching result and root review')
builder=root/'bench/run_qa_strong_long_v1.py'
for leaf,expected in design['helper_sha256'].items():
    assert sha(root/'bench'/leaf)==expected,leaf
assert builder.name in design['helper_sha256']
source={'ac_model':sha(ac_module.__file__),'trainer':sha(training_module.__file__),'utils':sha(utils_module.__file__)}
assert source==design['source_sha256']
material=root/'strong_long_adaptation_8k_material/material.json'
initial=root/'bench/STRONG_LONG_INITIAL_PAIR_V1.json';pair=json.loads(initial.read_text())
assert sha(material)==design['material_sha256'] and sha(initial)==design['initial_pair_sha256']
parent=root/'reports/oracle_27b_32k'
assert {name:sha(parent/name) for name in design['parent_publication_sha256']}==design['parent_publication_sha256']
result=json.loads((parent/'result.json').read_text());ac=Path(pair['ac_checkpoint']);reader=Path(pair['reader_checkpoint'])
assert result['stop_reason']=='complete' and result['completed_global_updates']==4000
assert str(ac)==result['checkpoint'] and str(reader)==result['target_lora_checkpoint']
assert ac==root/'oracle_27b_32k/AC_MODELS/rag_poc/epoch_001' and reader==root/'oracle_27b_32k/LORAS/rag_reader/epoch_001'
marker=json.loads((ac.parent/'completed_epoch.json').read_text())
assert marker['epoch_number']==1 and root/'oracle_27b_32k'/marker['ac_checkpoint']==ac
assert root/'oracle_27b_32k'/marker['target_lora_checkpoint']==reader
files={'ac_modules':ac/'ac_modules.pt','encoder_config':ac/'side_lora/adapter_config.json',
       'encoder_weights':ac/'side_lora/adapter_model.safetensors',
       'reader_config':reader/'adapter_config.json','reader_weights':reader/'adapter_model.safetensors'}
assert {name:sha(path) for name,path in files.items()}==design['initial_payload_sha256']==pair['payload_sha256']
name=design['run_name'];report=root/'reports'/name;report.mkdir(parents=True,exist_ok=True)
receipt={'builder_sha256':sha(builder),'source_sha256':source,'design_sha256':sha(design_path),
         'material_sha256':sha(material),'initial_pair_sha256':sha(initial),
         'initial_payload_sha256':design['initial_payload_sha256'],
         'optimizer_policy':'Fresh Adam initially; own published paired optimizer on explicit resume.'}
path=report/'source_receipt.json'
if path.exists():assert json.loads(path.read_text())==receipt,'Resume source/design changed'
else:
    with path.open('x') as handle:json.dump(receipt,handle,indent=2)
run=root/name;run.mkdir(parents=True,exist_ok=True)
(run/'PRESERVE').write_text('Active strong32k to fresh8k long-context adaptation; retain own paired checkpoints/optimizer.\n')
link=run/'AUTOTUNE'
if not link.exists():link.symlink_to(Path.home()/'activation_artifacts/AUTOTUNE',target_is_directory=True)
os.environ['ACTIVATION_SYNC_ROOT']=str(run)
command=[sys.executable,str(builder),'--design',str(design_path),'--arm','joint','--run-name',name,
 '--material',str(material),'--initial-pair',str(initial),'--items','8000','--epochs','1',
 '--seconds',str(design['training_config']['max_duration_seconds']),
 '--encoder-lr','0.00002','--reader-lr','0.000008','--ratio','0.0625',
 '--evidence-tokens','8192','--max-passages','16','--micro-batch-examples','2',
 '--encode-batch-max-tokens',str(design['encode_batch_max_tokens']),
 '--gradient-checkpointing-min-tokens','0','--passage-parts']
if resume:command.append('--resume')
print(json.dumps({'source_receipt':receipt,'command':command}),flush=True)
os.execv(sys.executable,command)
PY
