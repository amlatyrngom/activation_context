#!/bin/bash
#SBATCH --job-name=cmp_probe
#SBATCH --cpus-per-task=8 --mem=96G --time=3:00:00
#SBATCH --output=/data/scratch-fast/ngom/cmp_csail/logs/probe_%j.out
# CSAIL: the "[/" vs "</think>" probe (compaction/agent_eval/probe_think.py, uploaded to $W/jobs/ and checked by csail_stage.sh) of
# one or more pairs on the corrected dev-75 panel (panel_dev75_tv), in the fresh environment. One GPU.
#   csail_stage.sh sbatch BUNDLE --export=NIL --account=quanta --partition=quanta-h200 --qos=quanta-main --gres=gpu:1 \
#     jobs/csail_probe_job.sh BUNDLE [--references] LABEL=PAIR_DIR(relative to $W, holding ac/ reader/) ...
# Outputs: $W/probe/probe_<LABEL>.json (+ .positions.jsonl), logs in $W/probe/logs/.
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
W=/data/scratch-fast/ngom/cmp_csail; C=$W/code/${1:?BUNDLE}; shift; PY=$W/venv/bin/python
REF=; [ "${1:-}" = --references ] && { REF=--references; shift; }
mkdir -p $W/probe/logs $W/probe/cache/triton $W/probe/cache/inductor
export HOME=$W/home HF_HOME=$W/hf PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TRITON_CACHE_DIR=$W/probe/cache/triton TORCHINDUCTOR_CACHE_DIR=$W/probe/cache/inductor PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=$W/home/.cache \
  PYTHONPATH=$C:$C/kit:$C/rq_kit:$C/bench CQA_BENCH=$C/bench SEC45_RQ_KIT=$C/rq_kit SEC45_KIT=$C/kit RQ_THINKING=0 \
  QA4B_READER=Qwen/Qwen3.5-9B QA4B_SIDE=Qwen/Qwen3.5-2B QA4B_RANK=128 SEC45_ENCODER=Qwen/Qwen3.5-2B SEC45_RES_LAYERS=all SEC45_LINEAR_SKIP=1 QA4B_KV=0 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
say() { echo "$(date -u +%FT%TZ) [probe] $*"; }
say "on $(hostname) job ${SLURM_JOB_ID:-none} GPU '$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)'"
FAILED=0
for spec in "$@"; do
  L=${spec%%=*}; P=$W/${spec#*=}; OUT=$W/probe/probe_$L.json
  [ -f $OUT ] && { say "$L: done already"; continue; }
  [ -f $P/ac/ac_modules.pt ] && [ -f $P/reader/adapter_model.safetensors ] || { say "$L: no pair in $P"; FAILED=1; continue; }
  say "$L: $P $REF"
  (cd $C && $PY $W/jobs/probe_think.py --pair $P/ac $P/reader --label $L --panel $W/data/panel_dev75_tv --out $OUT $REF) > $W/probe/logs/probe_$L.log 2>&1 || FAILED=1
  say "$L: exit $FAILED; $(tail -n 1 $W/probe/logs/probe_$L.log | cut -c1-300)"
  REF=
done
say "end (failed=$FAILED)"; exit $FAILED
