#!/bin/bash
#SBATCH --job-name=pi42k_final100
#SBATCH --account=csail
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_BASE=${QA_BASE:-/data/scratch/ngom/activation}
export QA_ROOT=${QA_ROOT:-$QA_BASE/taskaware}
QA_PYTHON=${QA_PYTHON:-$QA_BASE/venv/bin/python}
QA_DESIGN=${1:?Pass frozen PI42K_MERGED32K_FRESH960_DESIGN_V1.json}
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1 QA_REPORT_ROOT=$QA_ROOT/reports
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=${QA_HF_HOME:-$QA_BASE/hf} HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QA_READOUT_AUTOTUNE=${QA_READOUT_AUTOTUNE:-$QA_BASE/AUTOTUNE}
cd "$QA_SOURCE_ROOT"
"$QA_PYTHON" - "$QA_DESIGN" <<'PY'
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(os.environ['QA_ROOT']);design_path=Path(sys.argv[1]);design=json.loads(design_path.read_text())
manifest_path=root/'bench/PI42K_MERGED32K_READOUT_MANIFEST_V1.json';manifest=json.loads(manifest_path.read_text())
for leaf,expected in manifest['helper_sha256'].items():
 with (root/'bench'/leaf).open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==expected,leaf
assert design['status']=='frozen' and design['run_name']=='pi42k_merged32k_fresh960_v1'
output=root/'reports/pi42k_merged32k_cohort100';output.mkdir(parents=True,exist_ok=True)
(output/'PRESERVE').write_text('Own completed960-question final32k100 answers and same-host2k retention; no confirmation.\n')
for role,context,seconds in [('final','32k',1850),('parent42k','2k',300),('final','2k',300)]:
 cell=design['run_name']+('__parent42k' if role=='parent42k' else '')+'__'+context
 cache=root/'pi42k_merged32k_readout_cache'/cell;cache.mkdir(parents=True,exist_ok=True)
 (cache/'PRESERVE').write_text('Active read-only own960-stage diagnostic cache.\n')
 link=cache/'AUTOTUNE'
 if not link.exists():link.symlink_to(Path(os.environ['QA_READOUT_AUTOTUNE']),target_is_directory=True)
 env=dict(os.environ,ACTIVATION_SYNC_ROOT=str(cache))
 command=[sys.executable,str(root/'bench/evaluate_pi42k_merged32k_v1.py'),
  '--manifest',str(manifest_path),'--design',str(design_path),'--material',design['material_path'],
  '--subset',design['subset_path'],'--initial-pair',design['initial_pair_path'],
  '--bench',str(root/'bench/run_qa_pi42k_merged32k_v1.py'),
  '--selection',str(root/'bench/MODEL_COMPARISON_COHORT100_SELECTION.json'),
  '--reports',str(root/'reports'),'--runs-root',str(root),'--output',str(output),
  '--checkpoint',role,'--context',context,'--seconds',str(seconds)]
 print(json.dumps({'cell':cell,'command':command,'read_only':True}),flush=True)
 subprocess.run(command,env=env,check=True)
 result=json.loads((output/cell/'result.json').read_text())
 if result['status']!='complete':raise SystemExit(75)
# Retention comparison is published only when both measurements use identical
# same-allocation hardware/runtime, prompts/tools/gold, rows and fixed100 order.
def read_cell(name):return json.loads((output/name/'result.json').read_text())
parent=read_cell(design['run_name']+'__parent42k__2k');final=read_cell(design['run_name']+'__2k')
for key in ('shared_input_fingerprint','ac_input_fingerprint','total_rows_fingerprint','model_provenance','runtime_policy','selected_ids','input_rows'):
 assert parent[key]==final[key],key
summary={'status':'complete','comparison':'Same-host native2k after960 fresh merged32k questions once; final minus constant42k parent',
 'questions':100,'parent_unique_questions':42000,'final_unique_questions':42960,
 'runtime':final['runtime'],'input_fingerprints':{k:final[k] for k in ('shared_input_fingerprint','ac_input_fingerprint','total_rows_fingerprint')},'views':{}}
for view in ('AC','full context','no context'):
 p=parent['views'][view];f=final['views'][view]
 assert [r['item_id'] for r in p['per_item']]==[r['item_id'] for r in f['per_item']]
 summary['views'][view]={'parent_answer_nll':p['selected100_answer_nll'],'final_answer_nll':f['selected100_answer_nll'],
  'final_minus_parent':f['selected100_answer_nll']-p['selected100_answer_nll'],
  'cohorts':{bank:{'parent':p['cohorts'][bank]['answer_nll'],'final':f['cohorts'][bank]['answer_nll'],
     'final_minus_parent':f['cohorts'][bank]['answer_nll']-p['cohorts'][bank]['answer_nll']} for bank in p['cohorts']}}
for role,cell in [('parent',design['run_name']+'__parent42k__2k'),('final',design['run_name']+'__2k')]:
 with (output/cell/'result.json').open('rb') as handle:summary[role+'_readout_sha256']=hashlib.file_digest(handle,'sha256').hexdigest()
from activation.common.reporting import _write_atomic
_write_atomic(output/'retention_2k.json',json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps(summary),flush=True)
PY
