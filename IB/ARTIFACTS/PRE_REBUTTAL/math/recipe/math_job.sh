#!/bin/bash
#SBATCH --job-name=math
#SBATCH --cpus-per-task=8 --mem=96G --time=2-00:00:00
# Math synthesis clean recipe (§3.4, 2026-11-17): every GPU step of the recipe on ONE GPU (ORCD RTX PRO 6000, CSAIL H200 / GH200).
# Modes (first argument):
#   gen ARM CONDITION PROMPTS SEEDS SHARD      synth_gen.py: the teacher's syntheses (untuned 9B reading the three full runs; plain vLLM,
#                                              the teacher-test settings) of data/PROMPTS -> gen/ARM/gen_ARM.<shard>.jsonl (resumes)
#   train RUN PACKET INIT LOSS                 sec45_train.py --kind packet --mode res: reader FROZEN (MATH_READER_LR, default 0; the
#                                              hedge arm sets 2e-5), encoder 5e-5, warm-up 50, cosine, batch 8, one pass (every whole
#                                              batch of the packet), from init/INIT; LOSS kl | sft. In-run panel MATH_HELD (a pickle under
#                                              data/, default none) every MATH_REPORT updates; recovery slot every MATH_CKPT updates;
#                                              MATH_KEEP=all keeps every MATH_CKPT pair. Then the dev proxy of ckpt/final (MATH_PROXY=1).
#   proxy RUN STEP|final GROUPS                math_proxy_eval.py of runs/RUN/sync/ckpt/STEP on data/MATH_PANEL (GROUPS e.g. r10)
#   eval NAME PAIR ARMS INPUTS [SEEDS]        math_eval.py (dev evaluation, one code path for every arm: the harness's residual vLLM engine) of
#                                              the pair W/PAIR/{ac,reader} on data/INPUTS, arms e.g. results,full,ac_r10 -> eval/NAME/gen_<arm>.jsonl
#   proxyref NAME                              the references once: init/NAME on the panel with mdev/none (--references): NLL_full,
#                                              NLL_none and the stored teacher targets data/teacher_mdev.pt
# Each step skips what exists, so a requeued job continues; training resumes exactly from its recovery slot.
# Layout under W: code/<MATH_BUNDLE>/, data/, init/<pair>/{ac,reader,DONE}, runs/<RUN>/, gen/<ARM>/, logs/.
#   ORCD : W=/home/ngom/orcd/scratch/activation/math1, the rq_v1 venv, the shared HF cache.
#   CSAIL: W=/data/scratch-fast/ngom/cmp_csail/math1, the fresh venv / HF cache of cmp_csail (csail_setup_job.sh, 2026-11-17; GH200:
#          MATH_PY = the per-job aarch64 venv of math_gh200_job.sh), per-job JIT caches.
set -uo pipefail
export PATH=${PATH:-/usr/local/bin:/usr/bin:/bin}
# CSAIL: `sbatch --export=NIL,VAR=...` gets the job held ("user env retrieval failed") or cancelled by root (2026-11-17 test 3099870),
# so settings come as leading MATH_*=value arguments (submit with plain --export=NIL); ORCD accepts both forms.
while [ $# -gt 0 ] && [[ $1 =~ ^MATH_[A-Z_]+=[A-Za-z0-9_./,:-]*$ ]]; do export "$1"; shift; done
MODE=${1:?mode}; shift
BUNDLE=${MATH_BUNDLE:?MATH_BUNDLE}
if [ -d /home/ngom/orcd/scratch/activation ]; then W=/home/ngom/orcd/scratch/activation/math1; PY=/home/ngom/orcd/scratch/activation/rq_v1/code/.venv/bin/python; HF=/home/ngom/orcd/scratch/activation/hf; export HOME=${HOME:-/home/ngom}; CSAIL=0
elif [ -d /data/scratch-fast/ngom ]; then CW=/data/scratch-fast/ngom/cmp_csail; W=$CW/math1; PY=$CW/venv/bin/python; HF=$CW/hf; export HOME=$CW/home; CSAIL=1
else W=$HOME/math1; PY=$HOME/sky_workdir/.venv/bin/python; HF=${HF_HOME:-$HOME/.cache/huggingface}; CSAIL=0; fi
PY=${MATH_PY:-$PY}; C=$W/code/$BUNDLE; JOB=${SLURM_JOB_ID:-local}
mkdir -p $W/logs $W/runs $W/gen
say() { echo "$(date -u +%FT%TZ) [math $MODE ${1:-}] ${*:2}"; }
[ -f $C/MANIFEST.json ] || { echo "bundle $C missing"; exit 2; }
export HF_HOME=$HF PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p $W/cache/$JOB/triton $W/cache/$JOB/inductor $W/cache/$JOB/vllm $W/cache/$JOB/xdg
export TRITON_CACHE_DIR=$W/cache/$JOB/triton TORCHINDUCTOR_CACHE_DIR=$W/cache/$JOB/inductor VLLM_CACHE_ROOT=$W/cache/$JOB/vllm XDG_CACHE_HOME=$W/cache/$JOB/xdg
[ $CSAIL = 1 ] && export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$C:$C/kit:$C/rq_kit:$C/bench:$C/math CQA_BENCH=$C/bench SEC45_RQ_KIT=$C/rq_kit SEC45_KIT=$C/kit RQ_THINKING=0
export QA4B_READER=Qwen/Qwen3.5-9B QA4B_SIDE=Qwen/Qwen3.5-2B QA4B_RANK=128 SEC45_ENCODER=Qwen/Qwen3.5-2B SEC45_RES_LAYERS=all SEC45_LINEAR_SKIP=1 QA4B_KV=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# vLLM's FlashInfer top-k/top-p sampler JIT-compiles with nvcc, which the cluster nodes lack (ORCD 29462890: "Could not find nvcc");
# every generation of the recipe (teacher samples, every dev arm) uses vLLM's PyTorch sampler instead, on every cluster, so all arms
# share one sampler (the ORCD precedent of 2026-11-09: VLLM_USE_FLASHINFER_SAMPLER=0 + module cuda/13.0.1).
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ $CSAIL = 0 ] && [ -d /home/ngom/orcd/scratch/activation ]; then set +u; [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh; module load cuda/13.0.1 >/dev/null 2>&1 || true; set -u; fi
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
PANEL=$W/data/${MATH_PANEL:-m_dev_panel_v1}; TEACHER=$W/data/teacher_mdev.pt; REF=$W/data/proxy_ref_mdev.json
PROXY=($PY $C/math/math_proxy_eval.py --panel $PANEL --teacher-targets $TEACHER)

case $MODE in
gen)
  ARM=${1:?ARM}; COND=${2:?CONDITION}; PROMPTS=${3:?PROMPTS}; SEEDS=${4:-0,1,2}; SHARD=${5:-0/1}; O=$W/gen/$ARM; mkdir -p $O
  if [ -n "${SLURM_JOB_ID:-}" ]; then                          # several queued copies of one shard on this cluster: the first to start claims it
    CLAIM=$O/shard_${SHARD/\//of}.claim
    if ! mkdir $CLAIM 2>/dev/null; then other=$(cat $CLAIM/job 2>/dev/null)
      if [ -n "$other" ] && [ "$other" != "$SLURM_JOB_ID" ] && squeue -j $other -h -o %T 2>/dev/null | grep -qx RUNNING; then
        echo "gen/$ARM shard $SHARD is claimed by running job $other: this copy ($SLURM_JOB_ID) exits"; exit 0; fi; fi
    echo $SLURM_JOB_ID > $CLAIM/job
  fi
  exec > >(tee -a $O/job_${SHARD/\//of}.log) 2>&1
  say $ARM "start on $(hostname) job $JOB GPU '$GPU' prompts $PROMPTS condition $COND seeds $SEEDS shard $SHARD"
  (cd $W/data && sha256sum -c $PROMPTS.sha256) || { say $ARM "prompts checksum failed"; exit 2; }
  # the 27B NVFP4 note of teacher_test/node_job.sh does not apply (9B bf16); compile parallelism capped anyway
  MAX_JOBS=3 $PY $C/math/synth_gen.py --prompts $W/data/$PROMPTS --out $O --arm $ARM --condition $COND --seeds $SEEDS --shard $SHARD \
    ${MATH_TASKS:+--tasks-file $W/data/$MATH_TASKS} ${MATH_GPU_UTIL:+--gpu-memory-utilization $MATH_GPU_UTIL}
  RC=$?; say $ARM "exit $RC; $(cat $O/gen_$ARM.${SHARD/\//of}.jsonl 2>/dev/null | wc -l) rows"; exit $RC;;
train)
  RUN=${1:?RUN}; PACKET=${2:?PACKET}; INIT=${3:?INIT}; LOSS=${4:?LOSS}
  D=$W/runs/$RUN; P=$W/data/packets/$PACKET; I=$W/init/$INIT; mkdir -p $D/logs $D/proxy
  if [ -n "${SLURM_JOB_ID:-}" ]; then                          # several queued copies of one run on this cluster (H200 / GH200): the first to start claims it
    CLAIM=$W/runs/$RUN.claim
    if ! mkdir $CLAIM 2>/dev/null; then other=$(cat $CLAIM/job 2>/dev/null)
      if [ -n "$other" ] && [ "$other" != "$SLURM_JOB_ID" ] && squeue -j $other -h -o %T 2>/dev/null | grep -qx RUNNING; then
        echo "runs/$RUN is claimed by running job $other: this copy ($SLURM_JOB_ID) exits"; exit 0; fi; fi
    echo $SLURM_JOB_ID > $CLAIM/job
  fi
  exec > >(tee -a $D/logs/job.log) 2>&1
  READER_LR=${MATH_READER_LR:-0}; REPORT=${MATH_REPORT:-40}; CKPT=${MATH_CKPT:-40}; KEEP=${MATH_KEEP:-final}
  for f in $P/items.pkl $P/prepared.json $I/ac/ac_modules.pt $I/reader/adapter_model.safetensors $I/DONE; do [ -f $f ] || { say $RUN "missing $f"; exit 2; }; done
  (cd $P && sha256sum -c SHA256SUMS) || { say $RUN "packet checksum failed"; exit 2; }
  N=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['training_examples'])" $P/prepared.json); STEPS=${MATH_STEPS:-$((N / 8))}
  HELD=(); [ -n "${MATH_HELD:-}" ] && HELD=(--held-items $W/data/$MATH_HELD)
  say $RUN "start on $(hostname) job $JOB GPU '$GPU' packet $PACKET ($N items -> $STEPS updates) init $INIT loss $LOSS reader_lr $READER_LR held ${MATH_HELD:-none}"
  if [ ! -f $D/sync/ckpt/final/meta.json ]; then
    ACTIVATION_SYNC_ROOT=$D/sync $PY $C/kit/sec45_train.py --kind packet --mode res --data $P --pair $I/ac $I/reader --loss $LOSS --name $RUN \
      --steps $STEPS --epu 8 --encoder-lr 5e-5 --reader-lr $READER_LR --warmup 50 --schedule cosine --seed 0 --micro-batch 1 \
      --resume-if-possible --report-every $REPORT --ckpt-every $CKPT --keep-steps $KEEP "${HELD[@]}" > $D/logs/train_$JOB.log 2>&1
    RC=$?; say $RUN "training exit $RC: $(tail -n 1 $D/logs/train_$JOB.log | cut -c1-500)"; [ $RC = 0 ] || exit $RC
  fi
  [ -f $D/sync/$RUN.stats.json ] && say $RUN "timing: $($PY -c "import json,sys; t=json.load(open(sys.argv[1]))['timing']; print(json.dumps({k: t.get(k) for k in ('updates', 'effective_seconds_per_update', 'seconds_per_update_median', 'peak_gb', 'wall_seconds', 'device')}))" $D/sync/$RUN.stats.json)"
  touch $D/TRAINED
  if [ "${MATH_PROXY:-1}" = 1 ] && [ -f $REF ] && [ -n "${MATH_GROUPS:-}" ]; then
    for Q in $D/sync/ckpt/final $D/sync/ckpt/step_*; do [ -d $Q/ac ] || continue; name=$(basename $Q); [ -f $D/proxy/$name.json ] && continue
      say $RUN "proxy $name"; "${PROXY[@]}" --pair $Q/ac $Q/reader --groups $MATH_GROUPS --references-from $REF --out $D/proxy/$name.json --label ${RUN}_$name > $D/logs/proxy_$name.log 2>&1 \
        || "${PROXY[@]}" --pair $Q/ac $Q/reader --groups $MATH_GROUPS --references-from $REF --out $D/proxy/$name.json --label ${RUN}_$name --no-deterministic > $D/logs/proxy_${name}_nodet.log 2>&1 \
        || say $RUN "proxy $name FAILED"; done
  fi
  say $RUN "end"; exit 0;;
proxy)
  RUN=${1:?RUN}; STEP=${2:?STEP}; GRP=${3:?GROUPS}; D=$W/runs/$RUN; Q=$D/sync/ckpt/$STEP; mkdir -p $D/proxy $D/logs
  exec > >(tee -a $D/logs/proxy_job.log) 2>&1
  [ -f $REF ] || { say $RUN "no $REF (run proxyref first)"; exit 2; }
  say $RUN "proxy $STEP on $(hostname) job $JOB GPU '$GPU'"
  "${PROXY[@]}" --pair $Q/ac $Q/reader --groups $GRP --references-from $REF --out $D/proxy/$STEP.json --label ${RUN}_$STEP > $D/logs/proxy_$STEP.log 2>&1 \
    || "${PROXY[@]}" --pair $Q/ac $Q/reader --groups $GRP --references-from $REF --out $D/proxy/$STEP.json --label ${RUN}_$STEP --no-deterministic > $D/logs/proxy_${STEP}_nodet.log 2>&1
  RC=$?; say $RUN "proxy exit $RC"; exit $RC;;
proxyref)
  NAME=${1:?NAME}; I=$W/init/$NAME; mkdir -p $W/runs/proxyref
  exec > >(tee -a $W/runs/proxyref/job.log) 2>&1
  [ -f $REF ] && { say ref "exists"; exit 0; }
  say ref "on $(hostname) job $JOB GPU '$GPU': $NAME on $PANEL with mdev/none"
  "${PROXY[@]}" --pair $I/ac $I/reader --groups r10,r5 --references --out $W/runs/proxyref/$NAME.json --label ref_$NAME > $W/runs/proxyref/proxy.log 2>&1 \
    || "${PROXY[@]}" --pair $I/ac $I/reader --groups r10,r5 --references --out $W/runs/proxyref/$NAME.json --label ref_$NAME --no-deterministic > $W/runs/proxyref/proxy_nodet.log 2>&1
  RC=$?; [ $RC = 0 ] && cp $W/runs/proxyref/$NAME.json $REF; say ref "exit $RC"; exit $RC;;
eval)
  NAME=${1:?NAME}; PAIR=$W/${2:?PAIR}; ARMS=${3:?ARMS}; INPUTS=${4:?INPUTS}; SEEDS=${5:-0,1,2}; O=$W/eval/$NAME; mkdir -p $O
  exec > >(tee -a $O/job.log) 2>&1
  (cd $W/data && sha256sum -c $INPUTS.sha256) || { say $NAME "inputs checksum failed"; exit 2; }
  $PY -c "import importlib.metadata as m, sys; sys.exit(0 if any(e.name == 'ac_residuals' for e in m.entry_points(group='vllm.general_plugins')) else 1)" \
    || { export PYTHONPATH=$PYTHONPATH:$C/ep; say $NAME "vLLM plugin entry point from the bundle (ep/)"; }
  say $NAME "start on $(hostname) job $JOB GPU '$GPU' pair $PAIR arms $ARMS inputs $INPUTS seeds $SEEDS"
  MAX_JOBS=3 $PY $C/math/math_eval.py --inputs $W/data/$INPUTS --pair $PAIR/ac $PAIR/reader --arms $ARMS --seeds $SEEDS --out $O --label $NAME \
    ${MATH_EVAL_LIMIT:+--limit $MATH_EVAL_LIMIT} ${MATH_EVAL_CONC:+--concurrency $MATH_EVAL_CONC}
  RC=$?; say $NAME "exit $RC; $(for f in $O/gen_*.jsonl; do echo -n "$(basename $f) $(wc -l < $f) "; done)"; exit $RC;;
*) echo "unknown mode $MODE"; exit 2;;
esac
