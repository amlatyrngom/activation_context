#!/bin/bash
#SBATCH --job-name=continue42k_layout_l40s
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
QA_ROOT=$HOME/orcd/pool/activation/taskaware
QA_CONTEXT=32k
export QA_SOURCE_ROOT=$QA_ROOT/train_ws_night_v1
export PYTHONPATH=$QA_SOURCE_ROOT HF_HOME=$HOME/orcd/scratch/activation/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTIVATION_SYNC_ROOT=$QA_ROOT/continuation_layout_l40s_cache
QA_OUTPUT=$QA_ROOT/reports/continuation_layout_l40s_cohort100
mkdir -p "$ACTIVATION_SYNC_ROOT" "$QA_OUTPUT"
# Capture one complete core on allocated compute. Later resumes never reread the
# live parent result, whose optional full-context answers may still be changing.
QA_L40S_CONTROL_SOURCE=${QA_L40S_CONTROL_SOURCE:-$QA_ROOT/reports/continuation_merged_l40s_cohort100/continue32k_10k_final_lr__merged_32k_l40s/result.json}
QA_L40S_CONTROL_SNAPSHOT=$QA_OUTPUT/input_merged_core.json
"$HOME/activation_venv/bin/python" - "$QA_L40S_CONTROL_SOURCE" "$QA_L40S_CONTROL_SNAPSHOT" <<'PY_CORE'
import hashlib,json,os,sys,uuid
from pathlib import Path
source,destination=map(Path,sys.argv[1:])
raw=destination.read_bytes() if destination.exists() else source.read_bytes()
value=json.loads(raw)
if (not value.get('core_complete') or not value.get('parent_payloads_rehashed_unchanged')
        or value.get('optimizer_updates')!=0 or len(value.get('rows',[]))!=100
        or value['identity'].get('condition')!='continue32k_10k_final_lr'
        or value['identity'].get('context')!='32k'
        or any(not value['views'].get(name,{}).get('complete') or not value['views'][name].get('fresh_measurement')
               for name in ('AC','full context','no context'))):
    raise ValueError('Cannot snapshot incomplete/wrong merged42k32k core')
if not destination.exists():
    temporary=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as handle:
            handle.write(raw);handle.flush();os.fsync(handle.fileno())
        try:os.link(temporary,destination)
        except FileExistsError:
            if destination.read_bytes()!=raw:raise ValueError('Concurrent different control snapshot; stop')
    finally:temporary.unlink(missing_ok=True)
print(json.dumps({'immutable_control_snapshot':str(destination),'source':str(source),
    'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'exact_source_bytes':True}),flush=True)
PY_CORE
printf 'Frozen42k same-L40S native2k and passagewise32k; required NLL first, optional AC answers.\n' > "$QA_OUTPUT/PRESERVE"
printf 'Active merged development readout cache.\n' > "$ACTIVATION_SYNC_ROOT/PRESERVE"
ln -sfn "$HOME/activation_artifacts/AUTOTUNE" "$ACTIVATION_SYNC_ROOT/AUTOTUNE"
cd "$QA_SOURCE_ROOT"
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/evaluate_continuation_layout_l40s_v1.py" \
    --design "$QA_ROOT/bench/STRONG_LONG_ADAPTATION_DESIGN_V1.json" \
    --material "$QA_ROOT/strong_long_adaptation_8k_material/material.json" \
    --bench "$QA_ROOT/bench/run_qa_strong_long_v1.py" \
    --selection "$QA_ROOT/bench/MODEL_COMPARISON_COHORT100_SELECTION.json" \
    --reports "$QA_ROOT/reports" \
    --continuation-design "$QA_ROOT/bench/CONTINUATION_10K_DESIGN_ORCD.json" \
    --continuation-initial-pair "$QA_ROOT/continuation_32k_initial/pair.json" \
    --native-readout "$QA_ROOT/reports/continuation_cohort100/continue32k_10k_final_lr.json" \
    --piecewise-readout "$QA_ROOT/reports/continuation_long_cohort100/continue32k_10k_final_lr__$QA_CONTEXT/result.json" \
    --adam-receipt "$QA_ROOT/gentle42k_final_lr_familiar4k_v1/initial/pair.json" \
    --l40s-merged-readout "$QA_L40S_CONTROL_SNAPSHOT" \
    --output "$QA_OUTPUT" --seconds "${QA_EVAL_SECONDS:-1400}" --decode
