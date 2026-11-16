#!/bin/bash
#SBATCH --job-name=cmp_px
#SBATCH --cpus-per-task=8 --mem=96G --time=02:00:00
#SBATCH --output=cmp_px_%j.out
# The dev-75 proxy of ONE saved checkpoint of a cmp_stage1_job.sh run, on its own GPU while the run keeps training (2026-11-16):
# the same scorer call as that job's proxy() (cmp_proxy_eval.py, panel data/panel_dev75, teacher targets data/teacher_dev75.pt
# read only, --references-from data/proxy_start_pair.json, deterministic kernels, once without them if those fail). It writes
# runs/<RUN>/proxy/<STEP>.json, which the run's own job then skips.
#   ORCD:  cd /home/ngom/orcd/scratch/activation/cmp1 && sbatch --export=NIL --partition=mit_preemptable --requeue \
#            --gres=gpu:rtx_pro_6000:1 jobs/cmp_ckpt_proxy_job.sh RUN STEP        (STEP: step_00100, final, ...)
set -uo pipefail
export PATH=${PATH:-/usr/local/bin:/usr/bin:/bin}
RUN=${1:?RUN}; STEP=${2:?STEP}
B=/home/ngom/orcd/scratch/activation; W=$B/cmp1; PY=$B/rq_v1/code/.venv/bin/python; HF=$B/hf; export HOME=${HOME:-/home/ngom}
C=$W/code/${CMP_BUNDLE:-cmp_stage1_code_v1}; PANEL=$W/data/panel_dev75; REF=$W/data/proxy_start_pair.json; D=$W/runs/$RUN; Q=$D/sync/ckpt/$STEP
mkdir -p $D/logs $D/proxy
exec > >(tee -a $D/logs/proxy_job_$STEP.log) 2>&1
say() { echo "$(date -u +%FT%TZ) [$RUN $STEP] $*"; }
for f in $C/MANIFEST.json $PANEL/panel_ac.pkl $REF $W/data/teacher_dev75.pt $Q/ac/ac_modules.pt $Q/reader/adapter_model.safetensors; do
  [ -f $f ] || { say "missing input $f"; exit 2; }; done
[ -f $D/proxy/$STEP.json ] && { say "already done"; exit 0; }
say "start on $(hostname) job ${SLURM_JOB_ID:-none} GPU '$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)'"
export HF_HOME=$HF PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$W/triton_cache
export PYTHONPATH=$C:$C/kit:$C/rq_kit:$C/bench CQA_BENCH=$C/bench SEC45_RQ_KIT=$C/rq_kit SEC45_KIT=$C/kit RQ_THINKING=0
export QA4B_READER=Qwen/Qwen3.5-9B QA4B_SIDE=Qwen/Qwen3.5-2B QA4B_RANK=128 SEC45_ENCODER=Qwen/Qwen3.5-2B SEC45_RES_LAYERS=all SEC45_LINEAR_SKIP=1 QA4B_KV=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
PROXY=($PY $C/compaction/cmp_proxy_eval.py --panel $PANEL --teacher-targets $W/data/teacher_dev75.pt --pair $Q/ac $Q/reader --label ${RUN}_$STEP --references-from $REF)
"${PROXY[@]}" --out $D/proxy/$STEP.json > $D/logs/proxy_${STEP}_side.log 2>&1 || {
  say "failed with deterministic kernels (tail: $(tail -n 2 $D/logs/proxy_${STEP}_side.log | tr '\n' ' ' | cut -c1-300)); again without"
  "${PROXY[@]}" --out $D/proxy/$STEP.json --no-deterministic > $D/logs/proxy_${STEP}_side_nodet.log 2>&1 || { say "FAILED"; exit 1; }; }
say "done: $($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in ('nll_full', 'nll_none', 'gap_closed', 'teacher', 'eval_seconds', 'peak_gb')} | {g: {m: round(v[m], 4) for m in ('kl_token', 'nll_token')} for g, v in d['groups'].items()}))" $D/proxy/$STEP.json)"
