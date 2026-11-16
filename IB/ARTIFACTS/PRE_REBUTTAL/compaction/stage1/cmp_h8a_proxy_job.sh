#!/bin/bash
#SBATCH --job-name=cmp_h8a
#SBATCH --cpus-per-task=8 --mem=96G --time=04:00:00
#SBATCH --output=cmp_h8a_%j.out
# The paper's compaction model H8a (2B encoder -> 9B reader + trained reader LoRA; per-layer deltas at every full-attention layer)
# on the stage-1 dev-75 proxy, ONE GPU, same scorer / panel / env as cmp_stage1_job.sh. Two variants:
#   deep      rows + deltas, as H8a was trained (HF trainer)
#   rowsonly  --deltas-off: rows alone, as the paper served it (vLLM rows only)
# Reader vs encoder split (rows only):
#   readernone  H8a's adapted reader on the `none` arm (--references: no rows; its r5/r10 repeat rowsonly)
#   zeroreader  H8a's encoder with R_res_rq9fr's all-zero reader adapter = the untuned frozen 9B, as the new stage-1 runs; its
#               teacher is the untuned 9B, so it reuses a COPY of cmp1/data/teacher_dev75.pt (adapter hash checked by the scorer)
#   ORCD:  cd /home/ngom/orcd/scratch/activation/cmp1_h8a && sbatch --export=NIL --partition=mit_preemptable --requeue \
#            --gres=gpu:rtx_pro_6000:1 jobs/cmp_h8a_proxy_job.sh [LIMIT] [VARIANTS]   (LIMIT 0 = all 75; VARIANTS default "deep rowsonly")
# Reads (never writes) cmp1/: the code bundle, data/panel_dev75, data/proxy_start_pair.json (dev75/none and NLL_full of the untuned 9B).
# Writes cmp1_h8a/out/: <variant>[_limitN].json (+ .items.jsonl), teacher targets of H8a's adapted reader, logs.
set -uo pipefail
export PATH=${PATH:-/usr/local/bin:/usr/bin:/bin}
LIMIT=${1:-0}; VARIANTS=${2:-deep rowsonly}
B=/home/ngom/orcd/scratch/activation; W=$B/cmp1_h8a; PY=$B/rq_v1/code/.venv/bin/python; HF=$B/hf; export HOME=${HOME:-/home/ngom}
C=$B/cmp1/code/cmp_stage1_code_v1; PANEL=$B/cmp1/data/panel_dev75; REF=$B/cmp1/data/proxy_start_pair.json; I=$W/h8a_final
ZERO=$B/cmp1/init/R_res_rq9fr_final/reader
SUF=$([ "$LIMIT" = 0 ] && echo "" || echo "_limit$LIMIT"); O=$W/out
mkdir -p $O $W/triton_cache
exec > >(tee -a $O/job$SUF.log) 2>&1
say() { echo "$(date -u +%FT%TZ) [h8a$SUF] $*"; }
for f in $C/MANIFEST.json $PANEL/panel_ac.pkl $REF $I/ac/ac_modules.pt $I/reader/adapter_model.safetensors $W/cmp_proxy_eval.py; do
  [ -f $f ] || { say "missing input $f"; exit 2; }; done
say "start on $(hostname) job ${SLURM_JOB_ID:-none} GPU '$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)'"
export HF_HOME=$HF PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$W/triton_cache
export PYTHONPATH=$C:$C/kit:$C/rq_kit:$C/bench CQA_BENCH=$C/bench SEC45_RQ_KIT=$C/rq_kit SEC45_KIT=$C/kit RQ_THINKING=0
export QA4B_READER=Qwen/Qwen3.5-9B QA4B_SIDE=Qwen/Qwen3.5-2B QA4B_RANK=128 SEC45_ENCODER=Qwen/Qwen3.5-2B SEC45_RES_LAYERS=all SEC45_LINEAR_SKIP=1 QA4B_KV=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
FAILED=0
for V in $VARIANTS; do
  name=$V$SUF; [ -f $O/$name.json ] && { say "$name: already done"; continue; }
  reader=$I/reader; teacher=$W/teacher_dev75_h8a$SUF.pt; refs=(--references-from $REF)
  case $V in
    deep) extra=();;
    rowsonly) extra=(--deltas-off);;
    readernone) extra=(--deltas-off); refs=(--references);;
    zeroreader) extra=(--deltas-off); reader=$ZERO; teacher=$W/teacher_dev75_zero.pt
                for f in teacher_dev75.pt teacher_dev75.json; do [ -f $W/${f/dev75/dev75_zero} ] || cp $B/cmp1/data/$f $W/${f/dev75/dev75_zero}; done;;
    *) say "unknown variant $V"; FAILED=1; continue;;
  esac
  say "$name (reader $reader)"
  $PY $W/cmp_proxy_eval.py --pair $I/ac $reader --panel $PANEL --teacher-targets $teacher "${refs[@]}" \
    --limit $LIMIT --out $O/$name.json --label h8a_$name "${extra[@]}" > $O/$name.log 2>&1 || { say "$name FAILED: $(tail -n 3 $O/$name.log | tr '\n' ' ' | cut -c1-600)"; FAILED=1; continue; }
  say "$name: $($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d.get(k) for k in ('nll_full', 'reference_nll_full', 'nll_none', 'gap_closed_vs_reference', 'eval_seconds', 'peak_gb')} | {g: {m: round(v[m], 4) for m in ('kl_token', 'nll_token')} for g, v in d['groups'].items()}))" $O/$name.json)"
done
say "end (failed=$FAILED)"; exit $FAILED
