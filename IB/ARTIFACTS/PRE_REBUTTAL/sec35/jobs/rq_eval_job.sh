#!/bin/bash
#SBATCH --job-name=s35_rq_eval
#SBATCH --cpus-per-task=8 --mem=96G --time=12:00:00
#SBATCH --output=s35_rq_eval_%j.out
# Pre-rebuttal retrieval (2B encoder -> 9B reader, res all layers, linear-skip head): score kept AC pairs on a GPU of their own, with the
# exact eval environment of jobs/generated_rq9*_rtx/T_res_rq9*.sh (sec45_eval.py, dev200, greedy, cap 1024, batch 32). Used when the
# in-run eval watcher OOMs next to training on one 96 GB card (2026-11-10). Pairs staged as $W/rq_eval/<RUN>_uNNNNN/{ac,reader};
# text+LoRA adapters as $W/rq_eval/text/<NAME>/ (PEFT folder; NAME "base" = the untuned 9B), scored by text_lora_eval_hf.py (as
# jobs/text_lora_job.sh: batch 8, cap 1024). An optional KS file in a staged folder lists the passage counts k to score (default 100);
# k = 20 is one encoder call, 100 five, 200 ten (the k-diagnostic, 2026-11-10). Outputs $W/rq_eval/out/<name>.<split>.k<k>.jsonl +
# .summary.json (a rerun skips finished ones).
#   ORCD: cd /home/ngom/orcd/scratch/activation/sec35 && sbatch --export=NIL --partition=mit_preemptable --requeue --gres=gpu:rtx_pro_6000:1 jobs/rq_eval_job.sh
set -uo pipefail
export PATH=${PATH:-/usr/local/bin:/usr/bin:/bin}
EVAL_SPLIT=${EVAL_SPLIT:-dev200}; BUNDLE=${SEC45_BUNDLE:-sec35_code_v3.tar.gz}
if [ -d /data/scratch-fast/ngom ]; then W=/data/scratch-fast/ngom/sec35; PY=/data/scratch/ngom/activation/venv/bin/python; HF=/data/scratch-fast/ngom/sec45/hf; export HOME=$W/home
elif [ -d /home/ngom/orcd/scratch/activation ]; then W=/home/ngom/orcd/scratch/activation/sec35; PY=/home/ngom/orcd/scratch/activation/rq_v1/code/.venv/bin/python; HF=/home/ngom/orcd/scratch/activation/hf; export HOME=${HOME:-/home/ngom}
else W=$HOME/sec35; PY=$HOME/sky_workdir/.venv/bin/python; HF=${HF_HOME:-$HOME/.cache/huggingface}; fi
PY=${SEC45_PY:-$PY}
R=${RQ_EVAL_DIR:-$W/rq_eval}; O=$R/out; TD=$W/data/${RQ_EVAL_DATA:-rq_tasks_v1}; mkdir -p $O $R/logs   # 20I (2026-11-11): RQ_EVAL_DATA=hotpot_tasks_v2 (rows carry top20, golds injected; k is ignored), EVAL_SPLIT=dev1000
exec > >(tee -a $R/logs/job.log) 2>&1
say() { echo "$(date -u +%FT%TZ) [rq_eval] $*"; }
STEM=${BUNDLE%.tar.gz}; C=$W/code/$STEM
[ -f $C/MANIFEST.json ] || { say "bundle $C not unpacked"; exit 2; }
# RQ_ENCODER (2026-11-13): encoder model id, default Qwen/Qwen3.5-2B (Qwen/Qwen3.5-4B for the T_res_rq9e4_e2 pair)
# ---- the retrieval job's environment (generated_rq9*_rtx: READER Qwen3.5-9B, ENCODER Qwen3.5-2B, RES_LAYERS all, LINEAR_SKIP 1) ----
export HF_HOME=$HF PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$W/triton_cache
export PYTHONPATH=$C:$C/kit:$C/rq_kit:$C/bench CQA_BENCH=$C/bench SEC45_RQ_KIT=$C/rq_kit RQ_THINKING=0 QA4B_READER=Qwen/Qwen3.5-9B QA4B_SIDE=${RQ_ENCODER:-Qwen/Qwen3.5-2B} QA4B_RANK=128
export SEC45_ENCODER=${RQ_ENCODER:-Qwen/Qwen3.5-2B} SEC45_RES_LAYERS=all SEC45_K=20,100,200 SEC45_LINEAR_SKIP=1 QA4B_LAYERS=3,7,11,15,19,23,27,31,35 QA4B_KV=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
say "start on $(hostname) job ${SLURM_JOB_ID:-none} GPU '$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)' data $TD split $EVAL_SPLIT code $C"
FAIL=0
for P in $R/*_u[0-9][0-9][0-9][0-9][0-9]/; do [ -d "$P/ac" ] || continue; N=$(basename $P)
  for k in $(cat $P/KS 2>/dev/null || echo 100); do T=$N.$EVAL_SPLIT.k$k; [ -f $O/$T.summary.json ] && continue
    say "AC $N k=$k"
    $PY $C/kit/sec45_eval.py --mode res --data $TD --split $EVAL_SPLIT --batch 32 --max-new 1024 --k $k --pair $P/ac $P/reader \
      --out $O/$T.jsonl --label ${N}_k$k > $R/logs/$T.log 2>&1 || { FAIL=1; say "FAILED $T"; continue; }
    say "$T: $(head -c 140 $O/$T.summary.json | tr -d '\n ')"
  done
done
for P in $R/text/*/; do [ -d "$P" ] || continue; N=text_$(basename $P); A=(--adapter $P); [ "$(basename $P)" = base ] && A=(--base)
  for k in $(cat $P/KS 2>/dev/null || echo 100); do T=$N.$EVAL_SPLIT.k$k; [ -f $O/$T.summary.json ] && continue
    say "text $N k=$k"
    PYTHONPATH=$C:$C/rq_kit RQ_KIT=$C/rq_kit TEXT_MODEL_ID=Qwen/Qwen3.5-9B $PY $C/text_lora/text_lora_eval_hf.py --data $TD --split $EVAL_SPLIT \
      --max-new 1024 --batch 8 --k $k "${A[@]}" --out $O/$T.jsonl --label ${N}_k$k > $R/logs/$T.log 2>&1 || { FAIL=1; say "FAILED $T"; continue; }
    say "$T: $(head -c 140 $O/$T.summary.json | tr -d '\n ')"
  done
done
say "end (fail=$FAIL)"; exit $FAIL
