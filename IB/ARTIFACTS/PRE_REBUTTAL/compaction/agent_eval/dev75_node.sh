#!/bin/bash
# Compaction agent evaluation on dev-75 (pre-rebuttal C4) on one SkyPilot spot node (1x RTX PRO 6000), run as a sky job through
# launch.sh (which stages this folder on S3: IB/ is not in the ~/sky_workdir mirror). Same command as the paper's 50k runs (kit/run9d_unit.sh, cmp_node.sh: --budget 50000 --trajectory-cap
# 90000 --max-model-len 100000 --max-duration 6000), through run_resume_res.py (Hidden Residuals pairs served by the vLLM residual
# path; text arms unchanged). Dev-75 only: the eval-200 (test) prefixes are never fetched.
#   dev75_node.sh run   TAG SEED VARIANTS AGENTS [PAIR_RUN|-] [run_resume.py args ...]
#        VARIANTS comma-separated (full,none,summary_write,summary,ac_4000,ac_2000); PAIR_RUN = a stage-1 run under
#        S3 compaction/stage1/runs/<PAIR_RUN>/sync/ckpt/final/ (needed by ac_* variants), '-' for none. Text summary arms are uncapped.
#        RUN_TIMEOUT=<seconds> (env, via launch.sh) stops the run after that long (load measurements; 0 = no limit).
#   dev75_node.sh check PAIR_RUN ROWS COUNT       first_turn_check.py (HF vs vLLM, trimmed vs full-length request) on the first COUNT prefixes
#   dev75_node.sh garble PAIR_RUN ROWS CACHE_ID COUNT   garble_check.py (HF teacher-forced vs vLLM log-probs of broken first turns)
#   dev75_node.sh probe LABEL=S3_CKPT_DIR ...      probe_think.py on the stage-1 dev-75 panel ("</think>" vs "[/" at the reader's thinking ends)
#   dev75_node.sh tvcompare TAG PAIR_S3_CKPT NAME=S3_ITEMS_PKL ...   teacher_view_compare.py: old vs recorded KL-teacher view (TEACHER_VIEW_FIX.md)
#   dev75_node.sh proxyref TAG PAIR_S3_CKPT PANEL_S3_DIR   the stage-1 proxy's teacher targets + references on a panel (S3 stage1/proxy_ref_TAG/)
# Resumable across spot reclaims: each ROLLOUTS/<TAG>_<variant> cache is restored from S3 when the node has none, mirrored to S3 every
# 10 min (snapshot cut at the last full line) and marked DONE on S3 when its variant finished.
# S3: pre_rebuttal_20261109/compaction/agent_eval_dev75/{ROLLOUTS,PAPER_COMPACTION,logs,checks}/
set -uo pipefail
MODE=$1; shift
S3C=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction
S3=$S3C/agent_eval_dev75
A=/root/activation_artifacts; P=$A/pc; E=$A/agent_eval; PY=/opt/activation/.venv/bin/python; HERE=$A/agent_eval/code   # fetched from S3 by launch.sh (IB/ is not in the sky mirror)
mkdir -p $P/kit $P/prefixes_dev9_75 $E/logs $E/stage $E/checks $E/pairs $A/SYNC/ROLLOUTS $A/SYNC/PAPER_COMPACTION $P/hf
(cd /root/sky_workdir && UV_PROJECT_ENVIRONMENT=/opt/activation/.venv /usr/local/bin/uv sync --frozen -q)   # also installs the vllm_residuals plugin entry point
[ -x /root/s3venv/bin/python ] || { /usr/local/bin/uv venv -q /root/s3venv && /usr/local/bin/uv pip install -q --python /root/s3venv/bin/python boto3; }
S3PY=/root/s3venv/bin/python
X() { $S3PY $HERE/s3x.py "$@"; }
LOG=$E/logs/node.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
X getdir $S3C/pc_inputs/kit/ $P/kit/ > /dev/null
X get $S3C/pc_inputs/split.json $P/split.json > /dev/null
X getdir $S3C/pc_inputs/prefixes_dev9_75/ $P/prefixes_dev9_75/ > /dev/null
[ "$(ls $P/prefixes_dev9_75/prefixes | wc -l)" = 75 ] || { say "dev-75 prefixes incomplete"; exit 2; }
if [ ! -d $A/SYNC/REPO_CACHE ]; then X get $S3C/repo_cache.tar.gz /root/rc.tgz > /dev/null && tar -xzf /root/rc.tgz -C $A/SYNC && rm /root/rc.tgz; fi
$PY -c "import importlib.metadata as m; assert any(e.name == 'ac_residuals' for e in m.entry_points(group='vllm.general_plugins')), 'vllm_residuals plugin not installed'" \
  || { say "vllm_residuals entry point missing"; exit 2; }
. $P/kit/node_env.sh                                     # HF_HOME, sync root, CUDA paths; cd /root/sky_workdir (it sets A to $P)
A=/root/activation_artifacts
export PYTHONPATH=$HERE:$PYTHONPATH RESUME_MONITOR_SECONDS=60
unset HF_TOKEN
for m in Qwen/Qwen3.5-9B Qwen/Qwen3.5-2B; do
  for t in 1 2 3; do timeout 1800 $PY -c "from huggingface_hub import snapshot_download as s; s('$m', token=False)" > /dev/null && break; done; done

pair() {   # pair RUN -> $E/pairs/RUN/{ac,reader}
  local run=$1 d=$E/pairs/$1
  [ -f $d/DONE ] && return 0
  X getdir $S3C/stage1/runs/$run/sync/ckpt/final/ac/ $d/ac/ > /dev/null && rm -f $d/ac/teacher_targets.pt
  X getdir $S3C/stage1/runs/$run/sync/ckpt/final/reader/ $d/reader/ > /dev/null
  X get $S3C/stage1/runs/$run/sync/ckpt/final/meta.json $d/meta.json > /dev/null
  [ -f $d/ac/ac_modules.pt ] && [ -f $d/reader/adapter_model.safetensors ] && date -u +%FT%TZ > $d/DONE && say "pair $run fetched: $(cat $d/meta.json)"
}
trim() { $S3PY -c "
import sys
data = open(sys.argv[1], 'rb').read(); data = data[:data.rfind(b'\n') + 1]
open(sys.argv[2], 'wb').write(data)" "$1" "$2"; }
rows() { [ -s "$1" ] && $S3PY -c "
import json, sys
keys = set()
for line in open(sys.argv[1]):
    try: keys.add(json.loads(line)['config_key'])
    except Exception: break
print(len(keys))" "$1" || echo 0; }
mirror() {   # mirror CACHE_ID...: rollout snapshots, run notes, logs
  for c in "$@"; do
    local f=$A/SYNC/ROLLOUTS/$c/rollouts.jsonl
    [ -s $f ] && mkdir -p $E/stage/$c && trim $f $E/stage/$c/rollouts.jsonl && X put $E/stage/$c/rollouts.jsonl $S3/ROLLOUTS/$c/rollouts.jsonl > /dev/null 2>&1
    for j in resume.json summary.json; do [ -f $A/SYNC/PAPER_COMPACTION/$c/$j ] && X put $A/SYNC/PAPER_COMPACTION/$c/$j $S3/PAPER_COMPACTION/$c/$j > /dev/null 2>&1; done
  done
  X putdir $E/logs $S3/logs/ > /dev/null 2>&1
}

if [ $MODE = check ]; then
  RUN=$1; ROWS=$2; COUNT=$3; pair $RUN
  OUT=$E/checks/first_turn_${RUN}_r${ROWS}.json
  say "check $RUN rows $ROWS count $COUNT"
  $PY $HERE/first_turn_check.py --split $P/split.json --prefixes $P/prefixes_dev9_75 --ac-checkpoint $E/pairs/$RUN/ac \
    --reader-checkpoint $E/pairs/$RUN/reader --rows $ROWS --count $COUNT --out $OUT > $E/logs/check_${RUN}_r${ROWS}.log 2>&1
  say "check exit $?"; [ -f $OUT ] && X put $OUT $S3/checks/$(basename $OUT) > /dev/null; mirror; exit 0
fi

if [ $MODE = garble ]; then   # garble PAIR_RUN ROWS CACHE_ID COUNT: garble_check.py on the first COUNT broken first turns of ROLLOUTS/CACHE_ID
  RUN=$1; ROWS=$2; CACHE=$3; COUNT=$4; pair $RUN
  F=$A/SYNC/ROLLOUTS/$CACHE/rollouts.jsonl; mkdir -p $(dirname $F)
  [ -s $F ] || X get $S3/ROLLOUTS/$CACHE/rollouts.jsonl $F > /dev/null
  OUT=$E/checks/garble_${CACHE}.json
  say "garble $RUN rows $ROWS cache $CACHE count $COUNT"
  $PY $HERE/garble_check.py --split $P/split.json --prefixes $P/prefixes_dev9_75 --ac-checkpoint $E/pairs/$RUN/ac \
    --reader-checkpoint $E/pairs/$RUN/reader --rollouts $F --rows $ROWS --count $COUNT --out $OUT > $E/logs/garble_${CACHE}.log 2>&1
  say "garble exit $?"; [ -f $OUT ] && X put $OUT $S3/checks/$(basename $OUT) > /dev/null; mirror; exit 0
fi

if [ $MODE = probe ]; then   # probe LABEL=S3_CKPT_DIR ...: probe_think.py per pair in the stage-1 code bundle's environment (as cmp_ckpt_proxy_job.sh); the first pair also scores the full / none views
  C=$E/bundle/cmp_stage1_code_v1
  if [ ! -f $C/MANIFEST.json ]; then
    X get $S3C/stage1/code/cmp_stage1_code_v1.tar.gz $E/bundle.tgz > /dev/null && X get $S3C/stage1/code/cmp_stage1_code_v1.tar.gz.sha256 $E/bundle.tgz.sha256 > /dev/null
    [ "$(sha256sum $E/bundle.tgz | cut -c1-64)" = "$(cut -c1-64 $E/bundle.tgz.sha256)" ] || { say "bundle sha256 mismatch"; exit 2; }
    mkdir -p $C && tar -xzf $E/bundle.tgz -C $C
  fi
  [ -f $E/panel/panel_ac.pkl ] || X getdir $S3C/stage1/panel_dev75/ $E/panel/ > /dev/null
  REF=--references
  for spec in "$@"; do
    L=${spec%%=*}; K=${spec#*=}; D=$E/pairs/probe_$L
    if [ ! -f $D/DONE ]; then
      X getdir $K/ac/ $D/ac/ > /dev/null; rm -f $D/ac/teacher_targets.pt; X getdir $K/reader/ $D/reader/ > /dev/null
      [ -f $D/ac/ac_modules.pt ] && [ -f $D/reader/adapter_model.safetensors ] && date -u +%FT%TZ > $D/DONE || { say "pair $L incomplete from $K"; continue; }
    fi
    OUT=$E/checks/probe_$L.json
    say "probe $L from $K $REF"
    (export PYTHONPATH=$C:$C/kit:$C/rq_kit:$C/bench CQA_BENCH=$C/bench SEC45_RQ_KIT=$C/rq_kit SEC45_KIT=$C/kit RQ_THINKING=0 \
       QA4B_READER=Qwen/Qwen3.5-9B QA4B_SIDE=Qwen/Qwen3.5-2B QA4B_RANK=128 SEC45_ENCODER=Qwen/Qwen3.5-2B SEC45_RES_LAYERS=all SEC45_LINEAR_SKIP=1 QA4B_KV=0 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
     cd $C && $PY $HERE/probe_think.py --pair $D/ac $D/reader --label $L --panel $E/panel --out $OUT $REF ${PROBE_LIMIT:+--limit $PROBE_LIMIT}) > $E/logs/probe_$L.log 2>&1
    say "probe $L exit $?"
    for f in $OUT ${OUT%.json}.positions.jsonl; do [ -f $f ] && X put $f $S3/probe/$(basename $f) > /dev/null; done
    REF=
  done
  mirror; exit 0
fi

if [ $MODE = tvcompare ]; then   # tvcompare TAG PAIR_S3_CKPT NAME=S3_ITEMS_PKL ...: teacher_view_compare.py (old vs recorded teacher view) with the repository's activation package
  TAG=$1; K=$2; shift 2
  C=$E/bundle/cmp_stage1_code_v1; [ -f $C/MANIFEST.json ] || { say "no bundle (run a probe first)"; exit 2; }
  D=$E/pairs/tv_$(basename $(dirname $(dirname $(dirname $K))))
  if [ ! -f $D/DONE ]; then X getdir $K/ac/ $D/ac/ > /dev/null; rm -f $D/ac/teacher_targets.pt; X getdir $K/reader/ $D/reader/ > /dev/null
    [ -f $D/ac/ac_modules.pt ] && date -u +%FT%TZ > $D/DONE || { say "pair incomplete from $K"; exit 2; }; fi
  ARGS=()
  for spec in "$@"; do N=${spec%%=*}; U=${spec#*=}; L=$E/tv/$N.pkl; mkdir -p $E/tv; [ -f $L ] || X get $U $L > /dev/null; ARGS+=(--packet $N=$L); done
  OUT=$E/checks/tvcompare_$TAG.json
  say "tvcompare $TAG pair $K packets $*"
  (export PYTHONPATH=/root/sky_workdir:$C/kit:$C/rq_kit:$C/bench CQA_BENCH=$C/bench SEC45_RQ_KIT=$C/rq_kit SEC45_KIT=$C/kit RQ_THINKING=0 \
       QA4B_READER=Qwen/Qwen3.5-9B QA4B_SIDE=Qwen/Qwen3.5-2B QA4B_RANK=128 SEC45_ENCODER=Qwen/Qwen3.5-2B SEC45_RES_LAYERS=all SEC45_LINEAR_SKIP=1 QA4B_KV=0 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
   cd $C && $PY $HERE/teacher_view_compare.py --pair $D/ac $D/reader --out $OUT "${ARGS[@]}" ${PROBE_LIMIT:+--limit $PROBE_LIMIT}) > $E/logs/tvcompare_$TAG.log 2>&1
  say "tvcompare $TAG exit $?"; [ -f $OUT ] && X put $OUT $S3/probe/$(basename $OUT) > /dev/null; mirror; exit 0
fi

if [ $MODE = proxyref ]; then   # proxyref TAG PAIR_S3_CKPT PANEL_S3_DIR: cmp_proxy_eval.py --references on a panel (new teacher targets + NLL_none / NLL_full), repository activation package
  TAG=$1; K=$2; PANEL_URL=$3
  C=$E/bundle/cmp_stage1_code_v1; [ -f $C/MANIFEST.json ] || { say "no bundle (run a probe first)"; exit 2; }
  D=$E/pairs/tv_$(basename $(dirname $(dirname $(dirname $K))))
  if [ ! -f $D/DONE ]; then X getdir $K/ac/ $D/ac/ > /dev/null; rm -f $D/ac/teacher_targets.pt; X getdir $K/reader/ $D/reader/ > /dev/null
    [ -f $D/ac/ac_modules.pt ] && date -u +%FT%TZ > $D/DONE || { say "pair incomplete from $K"; exit 2; }; fi
  PD=$E/panel_$TAG; [ -f $PD/panel_ac.pkl ] || X getdir $PANEL_URL/ $PD/ > /dev/null
  O=$E/proxyref_$TAG; mkdir -p $O
  say "proxyref $TAG pair $K panel $PANEL_URL"
  (export PYTHONPATH=/root/sky_workdir:$C/kit:$C/rq_kit:$C/bench CQA_BENCH=$C/bench SEC45_RQ_KIT=$C/rq_kit SEC45_KIT=$C/kit RQ_THINKING=0 \
       QA4B_READER=Qwen/Qwen3.5-9B QA4B_SIDE=Qwen/Qwen3.5-2B QA4B_RANK=128 SEC45_ENCODER=Qwen/Qwen3.5-2B SEC45_RES_LAYERS=all SEC45_LINEAR_SKIP=1 QA4B_KV=0 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
   cd $C && $PY $C/compaction/cmp_proxy_eval.py --pair $D/ac $D/reader --panel $PD --teacher-targets $O/teacher_dev75_$TAG.pt --references \
     --label start_pair_$TAG --out $O/proxy_start_pair_$TAG.json) > $E/logs/proxyref_$TAG.log 2>&1
  say "proxyref $TAG exit $?"; X putdir $O $S3C/stage1/proxy_ref_$TAG/ > /dev/null; mirror; exit 0
fi

[ $MODE = run ] || { echo "mode must be run, check, garble, probe, tvcompare or proxyref" >&2; exit 2; }
TAG=$1; SEED=$2; VARIANTS=$3; AGENTS=$4; RUN=${5:--}; shift 5
IDS=(); for v in ${VARIANTS//,/ }; do IDS+=(${TAG}_$v); done
PAIR_ARGS=()
if [ "$RUN" != - ]; then pair $RUN || exit 2; PAIR_ARGS=(--ac-checkpoint $E/pairs/$RUN/ac --reader-checkpoint $E/pairs/$RUN/reader); fi
for c in "${IDS[@]}"; do
  if X get $S3/ROLLOUTS/$c/DONE $E/stage/$c.DONE > /dev/null 2>&1; then say "$c already DONE on S3"; fi
  F=$A/SYNC/ROLLOUTS/$c/rollouts.jsonl; mkdir -p $(dirname $F)
  [ -s $F ] || { X get $S3/ROLLOUTS/$c/rollouts.jsonl $F > /dev/null 2>&1 && say "$c restored from S3: $(rows $F) rows"; }
  [ -s $F ] && trim $F $F
done
( while sleep 600; do mirror "${IDS[@]}"; done ) & LOOP=$!
export RESUME_MONITOR_FILE=$E/logs/${TAG}.monitor.jsonl
say "run $TAG seed $SEED variants $VARIANTS agents $AGENTS pair $RUN extra '$*' (cached: $(for c in "${IDS[@]}"; do echo -n "$c=$(rows $A/SYNC/ROLLOUTS/$c/rollouts.jsonl) "; done))"
timeout --signal=INT --kill-after=120 ${RUN_TIMEOUT:-0} $PY $HERE/run_resume_res.py --split $P/split.json --prefixes $P/prefixes_dev9_75 --variants $VARIANTS --uncapped-summary \
  --budget 50000 --trajectory-cap 90000 --max-model-len 100000 --max-duration 6000 --seed $SEED --tag $TAG --agents-per-gpu $AGENTS \
  "${PAIR_ARGS[@]}" "$@" >> $E/logs/${TAG}.log 2>&1
CODE=$?
pkill -P $LOOP 2>/dev/null; kill $LOOP 2>/dev/null; mirror "${IDS[@]}"   # the loop's sleep too (it held the sky job open up to 10 min)
for c in "${IDS[@]}"; do
  if [ $CODE = 0 ] && [ -f $A/SYNC/PAPER_COMPACTION/$c/summary.json ]; then
    date -u +%FT%TZ > $E/stage/$c.DONE && X put $E/stage/$c.DONE $S3/ROLLOUTS/$c/DONE > /dev/null && say "$c DONE: $(cat $A/SYNC/PAPER_COMPACTION/$c/summary.json | tr -d '\n ' | cut -c1-300)"
  fi
done
say "run $TAG exit $CODE"; mirror
exit $CODE
