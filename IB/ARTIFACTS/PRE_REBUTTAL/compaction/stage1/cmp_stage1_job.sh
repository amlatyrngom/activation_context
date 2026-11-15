#!/bin/bash
#SBATCH --job-name=cmp_s1
#SBATCH --cpus-per-task=8 --mem=96G --time=12:00:00
#SBATCH --output=cmp_s1_%j.out
# Compaction stage 1 on ONE GPU (RTX PRO 6000, 96 GB): R_res_rq9fr (2B encoder -> 9B reader, Hidden Residuals at every full-attention
# layer, reader adapter all zeros) trained on a build_stage1.sh packet with the reader FROZEN (reader LR 0), encoder 5e-5, warm-up 50,
# cosine, batch 8, one pass, KL to the starting reader reading the full prefix; final checkpoint = the result.
#   ORCD:  cd /home/ngom/orcd/scratch/activation/cmp1 && sbatch --export=NIL --partition=mit_preemptable --requeue \
#            --gres=gpu:rtx_pro_6000:1 jobs/cmp_stage1_job.sh RUN PACKET [STEPS]
#   SMOKE=1 (sbatch --export=NIL,SMOKE=1 ...): 20 updates, a 4-item in-run panel every 10 updates, no recovery slots.
# Steps (each skipped when its output exists, so a requeued / resubmitted job continues; training resumes from its recovery slot):
#   1. proxy of the starting pair on dev-75 (cmp_proxy_eval.py --references: AC 1/5, 1/10, untuned no-context, untuned full prefix);
#      its teacher targets are kept in $W/data/teacher_dev75.pt for every later checkpoint (same frozen reader)
#   2. training (sec45_train.py --kind packet), the dev-75 AC panel (150 items) every REPORT updates inside the run
#   3. proxy of ckpt/final (and of any ckpt/step_* kept)
# Inputs staged under W by hand (no S3 on ORCD): code/<BUNDLE stem>/, data/packets/<PACKET>/items.pkl, data/panel_dev75/,
# init/R_res_rq9fr_final/{ac,reader,DONE}. Outputs: runs/<RUN>/ (sync/: ckpt, stats; proxy/: the proxy jsons; logs/).
set -uo pipefail
export PATH=${PATH:-/usr/local/bin:/usr/bin:/bin}
RUN=${1:?RUN}; PACKET=${2:?PACKET}; STEPS=${3:-0}
BUNDLE=${CMP_BUNDLE:-cmp_stage1_code_v1}; SMOKE=${SMOKE:-0}
ENC_LR=5e-5; READER_LR=0; EPU=8; WARMUP=50; SEED=0; MICRO=1
REPORT=${CMP_REPORT_EVERY:-40}; CKPT=${CMP_CKPT_EVERY:-40}
if [ -d /home/ngom/orcd/scratch/activation ]; then W=/home/ngom/orcd/scratch/activation/cmp1; PY=/home/ngom/orcd/scratch/activation/rq_v1/code/.venv/bin/python; HF=/home/ngom/orcd/scratch/activation/hf; export HOME=${HOME:-/home/ngom}
else W=$HOME/cmp1; PY=$HOME/sky_workdir/.venv/bin/python; HF=${HF_HOME:-$HOME/.cache/huggingface}; fi
W=${CMP_WORK:-$W}; PY=${CMP_PY:-$PY}
C=$W/code/$BUNDLE; P=$W/data/packets/$PACKET; PANEL=$W/data/panel_dev75; I=$W/init/R_res_rq9fr_final; D=$W/runs/$RUN
mkdir -p $D/logs $D/proxy $W/triton_cache
exec > >(tee -a $D/logs/job.log) 2>&1
say() { echo "$(date -u +%FT%TZ) [$RUN] $*"; }
for f in $C/MANIFEST.json $P/items.pkl $PANEL/panel_ac.pkl $PANEL/panel_none.pkl $I/ac/ac_modules.pt $I/reader/adapter_model.safetensors $I/DONE; do
  [ -f $f ] || { say "missing input $f"; exit 2; }; done
say "start on $(hostname) job ${SLURM_JOB_ID:-none} GPU '$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)' packet $P smoke $SMOKE"
export HF_HOME=$HF PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$W/triton_cache
export PYTHONPATH=$C:$C/kit:$C/rq_kit:$C/bench CQA_BENCH=$C/bench SEC45_RQ_KIT=$C/rq_kit SEC45_KIT=$C/kit RQ_THINKING=0
export QA4B_READER=Qwen/Qwen3.5-9B QA4B_SIDE=Qwen/Qwen3.5-2B QA4B_RANK=128 SEC45_ENCODER=Qwen/Qwen3.5-2B SEC45_RES_LAYERS=all SEC45_LINEAR_SKIP=1 QA4B_KV=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
N=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['training_examples'])" $P/prepared.json)
[ "$STEPS" = 0 ] && STEPS=$((N / EPU))                       # one pass: every whole batch of 8 (the last N mod 8 items do not train)
TRAIN_ARGS=(--held-items $PANEL/panel_ac.pkl --report-every $REPORT --ckpt-every $CKPT --keep-steps final)
[ "$SMOKE" = 1 ] && { STEPS=20; TRAIN_ARGS=(--held-items $PANEL/panel_ac.pkl --held-panel 4 --report-every 10 --ckpt-every 1000 --keep-steps final); }
say "packet: $N items -> $STEPS updates x $EPU"
PROXY=($PY $C/compaction/cmp_proxy_eval.py --panel $PANEL --teacher-targets $W/data/teacher_dev75.pt)
proxy() {   # proxy NAME AC READER [extra]: one checkpoint on dev-75; deterministic kernels, once without them if those fail
  local name=$1 ac=$2 reader=$3; shift 3
  [ -f $D/proxy/$name.json ] && { say "proxy $name: already done"; return 0; }
  say "proxy $name"
  "${PROXY[@]}" --pair $ac $reader --out $D/proxy/$name.json --label ${RUN}_$name "$@" > $D/logs/proxy_$name.log 2>&1 || {
    say "proxy $name failed with deterministic kernels (tail: $(tail -n 2 $D/logs/proxy_$name.log | tr '\n' ' ' | cut -c1-300)); again without"
    "${PROXY[@]}" --pair $ac $reader --out $D/proxy/$name.json --label ${RUN}_$name --no-deterministic "$@" > $D/logs/proxy_${name}_nodet.log 2>&1 || { say "proxy $name FAILED"; return 1; }; }
  say "proxy $name: $($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in ('nll_full', 'nll_none', 'gap_closed', 'eval_seconds', 'peak_gb')} | {g: {m: round(v[m], 4) for m in ('kl_token', 'nll_token')} for g, v in d['groups'].items()}))" $D/proxy/$name.json)"
}
FAILED=0
REF=$W/data/proxy_start_pair.json                             # the starting pair's proxy (with dev75/none), shared by every run
if [ ! -f $REF ]; then proxy u00000 $I/ac $I/reader --references && cp $D/proxy/u00000.json $REF || FAILED=1
else cp -n $REF $D/proxy/u00000.json; say "proxy u00000: from $REF"; fi
if [ ! -f $D/sync/ckpt/final/meta.json ]; then
  say "training: $STEPS updates"
  ACTIVATION_SYNC_ROOT=$D/sync $PY $C/kit/sec45_train.py --kind packet --mode res --data $P --pair $I/ac $I/reader --loss kl --name $RUN \
    --steps $STEPS --epu $EPU --encoder-lr $ENC_LR --reader-lr $READER_LR --warmup $WARMUP --schedule cosine --seed $SEED --micro-batch $MICRO \
    --resume-if-possible "${TRAIN_ARGS[@]}" > $D/logs/train_${SLURM_JOB_ID:-local}.log 2>&1 || FAILED=1
  say "training exit $FAILED: $(tail -n 1 $D/logs/train_${SLURM_JOB_ID:-local}.log | cut -c1-600)"
fi
[ -f $D/sync/ckpt/final/meta.json ] && { proxy final $D/sync/ckpt/final/ac $D/sync/ckpt/final/reader --references-from $REF || FAILED=1; }
for Q in $D/sync/ckpt/step_*/; do [ -d "$Q/ac" ] && { proxy $(basename $Q) $Q/ac $Q/reader --references-from $REF || FAILED=1; }; done
[ -f $D/sync/$RUN.stats.json ] && say "timing: $($PY -c "import json,sys; t=json.load(open(sys.argv[1]))['timing']; print(json.dumps({k: t.get(k) for k in ('updates', 'effective_seconds_per_update', 'seconds_per_update_median', 'seconds_per_update_mean', 'first_updates_seconds', 'peak_gb', 'baseline_panel_seconds', 'reporting_seconds', 'wall_seconds', 'device')}))" $D/sync/$RUN.stats.json)"
say "end (failed=$FAILED)"; exit $FAILED
