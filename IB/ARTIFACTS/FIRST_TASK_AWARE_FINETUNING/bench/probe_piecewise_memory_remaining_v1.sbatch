#!/bin/bash
#SBATCH --job-name=qa_merged_memory
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" - <<'PY'
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path.home()/'orcd/pool/activation/taskaware'
worker=root/'bench/probe_piecewise_checkpoint_v1.py'
assert hashlib.sha256(worker.read_bytes()).hexdigest()=='12df00c8fcac924cf7a11525006c626d8ef572b6fdefa809468124512fba2139'
output=root/'reports/piecewise_memory_remaining_v1';output.mkdir(parents=True,exist_ok=True)
summary_path=output/'summary.json'
assert not summary_path.exists()
(output/'PRESERVE').write_text('Memory/throughput measurements only; original numerical proof failed. Never treat this as equivalence approval.\n')
summary={'status':'running','scope':'Two unfinished merged cases after23495243 preemption. Prior piecewise results retained. Memory only; original numerical gate remains failed. No training adoption.','equivalence_approved':False,'conditions':{}}
def publish():
    temporary=summary_path.with_suffix('.next');temporary.write_text(json.dumps(summary,indent=2)+'\n');temporary.replace(summary_path)
publish();began=time.monotonic()
conditions=('merged_failing','merged_quantiles')
for condition in conditions:
    remaining=540-(time.monotonic()-began)
    if remaining<90:summary['conditions'][condition]={'status':'not_run_budget'};publish();continue
    workspace=root/'piecewise_memory_remaining_v1'/condition;workspace.mkdir(parents=True,exist_ok=True)
    (workspace/'AUTOTUNE').symlink_to(Path.home()/'activation_artifacts/AUTOTUNE',target_is_directory=True)
    env=dict(os.environ);env['ACTIVATION_SYNC_ROOT']=str(workspace)
    args=[sys.executable,str(worker),'--worker',condition,'--material',str(root/'piecewise_training_4k_8k_material/material.json'),
          '--bench',str(root/'bench/run_qa_piecewise_v1.py'),'--failed-report',str(root/'reports/piecewise_27b_4k_8k'),
          '--p2-root',os.environ['QA_P2_ROOT'],'--output',str(output)]
    with (output/(condition+'.log')).open('x') as log:
        try:
            result=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(30,remaining-20))
            record={'returncode':result.returncode,'status':'complete' if result.returncode==0 else 'failed'}
        except subprocess.TimeoutExpired:record={'status':'timeout'}
    receipt=output/(condition+'.json')
    if receipt.exists():record['worker_status']=json.loads(receipt.read_text())['status']
    summary['conditions'][condition]=record;publish();print(json.dumps({condition:record}),flush=True)
summary['status']='measurements_complete' if all(summary['conditions'][c].get('status')=='complete' and summary['conditions'][c].get('worker_status')=='complete' for c in conditions) else 'measurements_partial'
summary['elapsed_seconds']=time.monotonic()-began
for selection in ('failing','quantiles'):
    files=[output/(layout+'_'+selection+'.json') for layout in ('pieces','merged')]
    if all(p.exists() for p in files):
        a,b=[json.loads(p.read_text()) for p in files]
        fields=('item_id','full_sha256','target_sha256','compressed_rows')
        assert [tuple(x[k] for k in fields) for x in a['inputs']]==[tuple(x[k] for k in fields) for x in b['inputs']]
publish()
PY
