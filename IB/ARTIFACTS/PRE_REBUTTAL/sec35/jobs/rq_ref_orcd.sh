#!/bin/bash
#SBATCH --job-name=s35_ref
#SBATCH --cpus-per-task=8 --mem=96G --time=12:00:00
# §3.1 gold-reference and no-context lines (2026-11-15, user: "go ahead"): the untuned Qwen3.5-9B, read with jobs/ref_view_eval.py
# (text_lora_eval_hf.py with the view swapped: gold = the two gold passages only, the KL teacher's prompt; none = question alone),
# greedy, cap 1024, code v5, the same scorer as every other row. test1000 (reported) and dev1000 (the dev draft), hotpot_tasks_v2 rows.
# Outputs $W/rq_eval_test_refs/out/text_base_<view>.<split>.jsonl + .summary.json (finished outputs skipped; a rerun resumes per question).
#   ORCD: cd /home/ngom/orcd/scratch/activation/sec35 && sbatch --export=NIL --time=12:00:00 --partition=mit_preemptable --requeue --gres=gpu:rtx_pro_6000:1 jobs/rq_ref_orcd.sh
set -uo pipefail
export PATH=${PATH:-/usr/local/bin:/usr/bin:/bin}
W=/home/ngom/orcd/scratch/activation/sec35; PY=/home/ngom/orcd/scratch/activation/rq_v1/code/.venv/bin/python
HF=/home/ngom/orcd/scratch/activation/hf; export HOME=${HOME:-/home/ngom}
C=$W/code/sec35_code_v5; R=$W/rq_eval_test_refs; O=$R/out; TD=$W/data/hotpot_tasks_v2; mkdir -p $O $R/logs
exec > >(tee -a $R/logs/job.log) 2>&1
say() { echo "$(date -u +%FT%TZ) [rq_ref] $*"; }
[ -f $C/MANIFEST.json ] || { say "bundle $C not unpacked"; exit 2; }
export HF_HOME=$HF PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RQ_THINKING=0
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
say "start on $(hostname) job ${SLURM_JOB_ID:-none} GPU '$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)'"
FAIL=0
for SPLIT in test1000 dev1000; do for VIEW in gold none; do T=text_base_$VIEW.$SPLIT; [ -f $O/$T.summary.json ] && continue
  say "$T"
  PYTHONPATH=$C:$C/rq_kit:$C/kit:$C/text_lora RQ_KIT=$C/rq_kit TEXT_MODEL_ID=Qwen/Qwen3.5-9B $PY $W/jobs/ref_view_eval.py $VIEW --data $TD \
    --split $SPLIT --base --max-new 1024 --batch 16 --out $O/$T.jsonl --label $T > $R/logs/$T.log 2>&1 || { FAIL=1; say "FAILED $T"; continue; }
  say "$T: $(head -c 140 $O/$T.summary.json | tr -d '\n ')"
done; done
say "end (fail=$FAIL)"; exit $FAIL
