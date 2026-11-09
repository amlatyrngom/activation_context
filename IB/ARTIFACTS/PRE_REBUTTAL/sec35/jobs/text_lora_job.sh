#!/bin/bash
#SBATCH --job-name=s35_text
#SBATCH --cpus-per-task=8 --mem=96G --time=96:00:00
#SBATCH --output=s35_text_%j.out
# Pre-rebuttal §3.5 text + reader-LoRA arm (text_lora/text_lora_train.py), one GPU, resumable: resubmitting the same command continues
# from OUT/resume (exact: same rows, order, LR schedule, AdamW moments). Every CKPT_EVERY updates an adapter is saved to ckpt/step_N and
# evaluated by the watcher (text_lora_eval_hf.py: the single scorer, greedy, cap 256) on EVAL_SPLIT, then kept (1 GB each).
#   CSAIL: cd /data/scratch-fast/ngom/sec35 && sbatch --export=NIL,RUN=X_text_...,STEPS=11200,CKPT_EVERY=800,LR=2e-5,SEED=0 \
#            --account=quanta --partition=quanta-h200 --qos=quanta-main --gpus=1 jobs/text_lora_job.sh
# Needs the code bundle unpacked at $W/code/<stem> (an AC job does it; or this script does) and $W/data/hotpot_tasks_v2.
set -uo pipefail
export PATH=${PATH:-/usr/local/bin:/usr/bin:/bin}
: "${RUN:?RUN}" "${STEPS:?STEPS}" "${CKPT_EVERY:?CKPT_EVERY}"; LR=${LR:-2e-5}; SEED=${SEED:-0}; EVAL_SPLIT=${EVAL_SPLIT:-dev1000}
TASK_DATA=${TASK_DATA:-hotpot_tasks_v2}; TEXT_MODEL_ID=${TEXT_MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}; K=${K:-}; EVAL_K=${EVAL_K:-}   # retrieval: rq_tasks_v1, Qwen/Qwen3.5-9B, K=20,100,200, EVAL_K=100
BUNDLE=${SEC45_BUNDLE:-sec35_code_v3.tar.gz}
if [ -d /data/scratch-fast/ngom ]; then W=/data/scratch-fast/ngom/sec35; PY=/data/scratch/ngom/activation/venv/bin/python; HF=/data/scratch-fast/ngom/sec45/hf; export HOME=$W/home
elif [ -d /home/ngom/orcd/scratch/activation ]; then W=/home/ngom/orcd/scratch/activation/sec35; PY=/home/ngom/orcd/scratch/activation/rq_v1/code/.venv/bin/python; HF=/home/ngom/orcd/scratch/activation/hf
else W=$HOME/sec35; PY=$HOME/sky_workdir/.venv/bin/python; HF=${HF_HOME:-$HOME/.cache/huggingface}; fi
PY=${SEC45_PY:-$PY}
D=$W/runs/$RUN; TD=$W/data/$TASK_DATA
mkdir -p $D/logs $D/eval
exec > >(tee -a $D/logs/job.log) 2>&1
say() { echo "$(date -u +%FT%TZ) [$RUN] $*"; }
[ -f $D/DONE ] && { say "finished: $(cat $D/DONE)"; exit 0; }
STEM=${BUNDLE%.tar.gz}; C=$W/code/$STEM
if [ ! -f $C/MANIFEST.json ]; then
  (cd $W/code && sha256sum -c $BUNDLE.sha256) || { say "bundle $W/code/$BUNDLE missing or bad"; exit 2; }
  rm -rf $C.tmp.$$; mkdir -p $C.tmp.$$ && tar -xzf $W/code/$BUNDLE -C $C.tmp.$$ && { mv -T $C.tmp.$$ $C 2>/dev/null || rm -rf $C.tmp.$$; }
fi
for f in train.jsonl $EVAL_SPLIT.jsonl hotpotqa.corpus.jsonl.gz; do [ -f $TD/$f ] || { say "missing $TD/$f"; exit 2; }; done
export HF_HOME=$HF PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$C:$C/rq_kit RQ_KIT=$C/rq_kit RQ_THINKING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TEXT_MODEL_ID
if [ -n "$K" ]; then export SEC45_K=$K; else unset SEC45_K; fi
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
say "start on $(hostname) job ${SLURM_JOB_ID:-none} GPU '$GPU' code $C; model $TEXT_MODEL_ID data $TASK_DATA k ${K:-20} steps $STEPS ckpt $CKPT_EVERY lr $LR seed $SEED eval $EVAL_SPLIT ${EVAL_K:+k $EVAL_K}"
FREE_GB=$(df -Pk $W | awk 'NR==2{printf "%d", $4/1048576}'); [ "$FREE_GB" -ge 30 ] || { say "only $FREE_GB GB free under $W"; exit 2; }
EVAL=($PY $C/text_lora/text_lora_eval_hf.py --data $TD --split $EVAL_SPLIT --max-new ${EVAL_MAX_NEW:-256} --batch ${EVAL_BATCH:-32} ${EVAL_K:+--k $EVAL_K})
eval_pending() { local Q U rc=0
  for Q in $D/ckpt/step_[0-9][0-9][0-9][0-9][0-9]; do [ -d "$Q" ] || continue; U=${Q##*_}
    [ -f $D/eval/u$U.summary.json ] && continue
    say "accuracy at update $((10#$U))"
    "${EVAL[@]}" --adapter $Q --out $D/eval/u$U.jsonl --label ${RUN}_u$U > $D/logs/eval_u$U.log 2>&1 || { say "eval u$U FAILED"; rc=1; continue; }
    say "u$U: $(head -c 160 $D/eval/u$U.summary.json | tr -d '\n ')"
  done; return $rc; }
if [ ! -f $D/eval/u00000.summary.json ]; then say "accuracy of the untuned 4B (update 0)"; "${EVAL[@]}" --base --out $D/eval/u00000.jsonl --label ${RUN}_u0 > $D/logs/eval_u00000.log 2>&1 || say "base eval FAILED"; fi
ATT=$(( $(ls $D/logs/train_a*.log 2>/dev/null | wc -l) + 1 ))
if [ ! -f $D/TRAINED ]; then
  $PY $C/text_lora/text_lora_train.py --data $TD --out $D --name $RUN --steps $STEPS --ckpt-every $CKPT_EVERY --lr $LR --seed $SEED --report-every 200 > $D/logs/train_a$ATT.log 2>&1 &
  TP=$!; while kill -0 $TP 2>/dev/null; do eval_pending || true; sleep 60; done; wait $TP; RC=$?
  say "training exit $RC: $(tail -n 1 $D/logs/train_a$ATT.log | cut -c1-300)"
fi
eval_pending; ERC=$?
if [ -f $D/TRAINED ] && [ $ERC = 0 ]; then
  $PY - "$D" "$RUN" "$GPU" <<'PY'
import json, sys, time
from pathlib import Path
d, run, gpu = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
evals = {p.name.removesuffix(".summary.json"): json.loads(p.read_text())["acc"] for p in sorted((d / "eval").glob("*.summary.json"))}
out = {"run": run, "gpu": gpu, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "valid": True, "evals": evals,
       "attempts": len(list((d / "logs").glob("train_a*.log")))}
(d / "DONE.tmp").write_text(json.dumps(out) + "\n"); (d / "DONE.tmp").rename(d / "DONE"); print(json.dumps(out))
PY
  say "end, valid"; exit 0
fi
say "end without DONE (resubmit to continue)"; exit 1
