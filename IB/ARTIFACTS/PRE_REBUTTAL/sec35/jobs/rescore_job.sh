#!/bin/bash
#SBATCH --job-name=s35_rescore
#SBATCH --cpus-per-task=8 --mem=96G --time=24:00:00
#SBATCH --output=s35_rescore_%j.out
# Pre-rebuttal: rescore the paper's §3.5 checkpoints under the single scorer (greedy HF decode, cap 256, strict "Final answer:" parse)
# on EVAL_SPLIT (default dev1000). Inputs staged under $W/rescore/: ac/<RUN>/{ac,reader,MODE} (MODE file: in|res|kv) and
# text/<RUN>_uNNNNN/ (PEFT adapters). Outputs: $W/rescore/out/<name>.jsonl + .summary.json (a rerun skips finished ones).
#   CSAIL: cd /data/scratch-fast/ngom/sec35 && sbatch --export=NIL --account=quanta --partition=quanta-h200 --qos=quanta-main --gpus=1 jobs/rescore_job.sh
set -uo pipefail
export PATH=${PATH:-/usr/local/bin:/usr/bin:/bin}
EVAL_SPLIT=${EVAL_SPLIT:-dev1000}; BUNDLE=${SEC45_BUNDLE:-sec35_code_v3.tar.gz}
if [ -d /data/scratch-fast/ngom ]; then W=/data/scratch-fast/ngom/sec35; PY=/data/scratch/ngom/activation/venv/bin/python; HF=/data/scratch-fast/ngom/sec45/hf; export HOME=$W/home
elif [ -d /home/ngom/orcd/scratch/activation ]; then W=/home/ngom/orcd/scratch/activation/sec35; PY=/home/ngom/orcd/scratch/activation/rq_v1/code/.venv/bin/python; HF=/home/ngom/orcd/scratch/activation/hf
else W=$HOME/sec35; PY=$HOME/sky_workdir/.venv/bin/python; HF=${HF_HOME:-$HOME/.cache/huggingface}; fi
PY=${SEC45_PY:-$PY}
R=$W/rescore; O=$R/out; TD=$W/data/hotpot_tasks_v2; mkdir -p $O $R/logs
exec > >(tee -a $R/logs/job.log) 2>&1
say() { echo "$(date -u +%FT%TZ) [rescore] $*"; }
STEM=${BUNDLE%.tar.gz}; C=$W/code/$STEM
if [ ! -f $C/MANIFEST.json ]; then
  (cd $W/code && sha256sum -c $BUNDLE.sha256) || { say "bundle missing"; exit 2; }
  rm -rf $C.tmp.$$; mkdir -p $C.tmp.$$ && tar -xzf $W/code/$BUNDLE -C $C.tmp.$$ && { mv -T $C.tmp.$$ $C 2>/dev/null || rm -rf $C.tmp.$$; }
fi
export HF_HOME=$HF PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$W/triton_cache
export PYTHONPATH=$C:$C/kit:$C/rq_kit:$C/bench CQA_BENCH=$C/bench SEC45_RQ_KIT=$C/rq_kit RQ_KIT=$C/rq_kit RQ_THINKING=0
export QA4B_READER=Qwen/Qwen3-4B-Instruct-2507 QA4B_SIDE=Qwen/Qwen3-4B-Instruct-2507 QA4B_RANK=128 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN SEC45_ENCODER
say "start on $(hostname) job ${SLURM_JOB_ID:-none} GPU '$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)' split $EVAL_SPLIT"
FAIL=0
for T in $R/text/*/; do [ -d "$T" ] || continue; N=$(basename $T); [ -f $O/$N.$EVAL_SPLIT.summary.json ] && continue
  say "text $N"; $PY $C/text_lora/text_lora_eval_hf.py --data $TD --split $EVAL_SPLIT --adapter $T --out $O/$N.$EVAL_SPLIT.jsonl --label $N > $R/logs/$N.log 2>&1 || { FAIL=1; say "FAILED $N"; }
  [ -f $O/$N.$EVAL_SPLIT.summary.json ] && say "$N: $(head -c 120 $O/$N.$EVAL_SPLIT.summary.json | tr -d '\n ')"
done
if [ ! -f $O/text_base.$EVAL_SPLIT.summary.json ]; then say "untuned 4B"
  $PY $C/text_lora/text_lora_eval_hf.py --data $TD --split $EVAL_SPLIT --base --out $O/text_base.$EVAL_SPLIT.jsonl --label text_base > $R/logs/text_base.log 2>&1 || FAIL=1; fi
for P in $R/ac/*/; do [ -d "$P" ] || continue; N=$(basename $P); [ -f $O/$N.$EVAL_SPLIT.summary.json ] && continue
  M=$(cat $P/MODE); say "AC $N ($M)"
  case $M in in) export QA4B_LAYERS=none QA4B_KV=0;; res) export QA4B_LAYERS=3,7,11,15,19,23,27,31,35 QA4B_KV=0;; kv) export QA4B_LAYERS=none QA4B_KV=1;; esac
  $PY $C/kit/sec45_eval.py --mode $M --data $TD --split $EVAL_SPLIT --batch 32 --max-new 256 --pair $P/ac $P/reader --out $O/$N.$EVAL_SPLIT.jsonl --label $N > $R/logs/$N.log 2>&1 || { FAIL=1; say "FAILED $N"; }
  [ -f $O/$N.$EVAL_SPLIT.summary.json ] && say "$N: $(head -c 120 $O/$N.$EVAL_SPLIT.summary.json | tr -d '\n ')"
done
say "end (fail=$FAIL)"; exit $FAIL
