#!/bin/bash
#SBATCH --job-name=qa_piecewise_4k
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export QA_P2_ROOT=$HOME/orcd/scratch/activation/KEEP/P2
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QA_CAMPAIGN_DEADLINE
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" - "$@" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
import activation.ac_model.ac_model as ac_module
import activation.ac_model.ac_model_training as training_module
import activation.ac_model.ac_model_utils as utils_module

root=Path.home()/'orcd/pool/activation/taskaware'
if len(sys.argv) not in (2,3) or (len(sys.argv)==3 and sys.argv[2]!='--resume'):
    raise SystemExit('Pass piecewise_27b_4k_8k or merged_27b_4k_8k, optionally --resume')
name=sys.argv[1]
if name not in ('piecewise_27b_4k_8k','merged_27b_4k_8k'):
    raise SystemExit('Unknown fixed experiment arm')
design=json.loads((root/'bench/PIECEWISE_TRAINING_DESIGN_V2.json').read_text())
assert design['version']==2 and design['status']=='frozen'
assert design['training']['gradient_checkpointing_min_tokens']==0
arm=design['arms'][name]
assert arm=={'passage_parts':name.startswith('piecewise_'),
             'encode_batch_max_tokens':65536,'micro_batch_examples':2}
builder=root/'bench/run_qa_piecewise_v2.py'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
receipt={'builder_sha256':sha(builder),'source_sha256':{
    'ac_model':sha(ac_module.__file__),'trainer':sha(training_module.__file__),
    'utils':sha(utils_module.__file__)}}
assert all(receipt[key]==design[key] for key in receipt), 'Unapproved source or runner'
assert receipt['source_sha256']['ac_model']=='aaeb3c68225d3092a024d44552b84480dc37cc8319d68c2ab1701fce70980bfd'
report=root/'reports'/name; report.mkdir(parents=True,exist_ok=True)
target=report/'source_receipt.json'
if target.exists():
    assert json.loads(target.read_text())==receipt, 'Resume source differs'
else:
    with target.open('x') as handle: json.dump(receipt,handle,indent=2)
run=root/name;run.mkdir(parents=True,exist_ok=True)
(run/'PRESERVE').write_text('Active matched piecewise/merged experiment; retain paired checkpoints and optimizer.\n')
link=run/'AUTOTUNE'
if not link.exists():link.symlink_to(Path.home()/'activation_artifacts/AUTOTUNE',target_is_directory=True)
os.environ['ACTIVATION_SYNC_ROOT']=str(run)
command=[sys.executable,str(builder),'--arm','joint','--run-name',name,
    '--material',str(root/'piecewise_training_4k_8k_material/material.json'),
    '--items','4000','--epochs','1','--seconds','21000',
    '--encoder-lr','0.00005','--reader-lr','0.00002','--ratio','0.0625',
    '--evidence-tokens','8192','--max-passages','16',
    '--micro-batch-examples','2','--encode-batch-max-tokens','65536',
    '--gradient-checkpointing-min-tokens','0']
if arm['passage_parts']:command.append('--passage-parts')
if len(sys.argv)==3:command.append('--resume')
print(json.dumps({'source_receipt':receipt,'command':command}),flush=True)
os.execv(sys.executable,command)
PY
