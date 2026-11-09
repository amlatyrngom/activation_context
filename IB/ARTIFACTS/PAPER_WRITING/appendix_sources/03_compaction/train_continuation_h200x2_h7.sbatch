#!/bin/bash
#SBATCH --job-name=pc_train_h2
#SBATCH --account=quanta
#SBATCH --partition=quanta-h200
#SBATCH --qos=quanta-main
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/scratch-fast/ngom/activation/paper_compaction_v1/logs/train_%x_%j.log
# H7 (Oct 15 22:55Z): as train_continuation_h200x2_fast2.sbatch with CHANNEL_QUOTA_GIB from the submitting environment (the coordinator raised the combined scratch-fast cap to 160G; compaction gets 115).
# As train_continuation_h200x2.sbatch, with the run folder, quota root, HOME and compile caches on /data/scratch-fast (/data/scratch was full on Oct 14 night;
# RUN and PAIR are copied there first). Source tree, venv and HF cache stay on /data/scratch. Below: as train_continuation_v2.sbatch on the x86 H200 pair, data parallel (2 GPUs; launch_continuation.sbatch runs torchrun with one
# process per GPU), as paper_traj_qa_v1/tqa_sft_v1/jobs/train_sft_h200.sbatch. Snapshots at 50 % and the end.
#   export RUN=packets/<name> PAIR=$W/pairs/<start>; sbatch -J <short name> jobs/train_continuation_h200x2.sbatch
set -euo pipefail
[ "$(uname -m)" = x86_64 ] || { echo "this job file is for the x86 H200 node" >&2; exit 2; }
R=/data/scratch/ngom/activation; W=/data/scratch-fast/ngom/activation/paper_compaction_v1; S=$R/probe_p_v1/source_p; PAIR=${PAIR:?PAIR=the starting pair}
export HOME=$W/home HF_HOME=$R/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$W/triton_cache TORCHINDUCTOR_CACHE_DIR=$W/inductor_cache
mkdir -p $HOME
export CHANNEL_SOURCE=$S CHANNEL_PYTHON=$R/venv/bin/python CHANNEL_RUN=$W/${RUN:?RUN=packets/<name>}
export CHANNEL_QUOTA_ROOT=$W CHANNEL_QUOTA_GIB=${CHANNEL_QUOTA_GIB:?CHANNEL_QUOTA_GIB=the guard on the compaction folder} CHANNEL_SNAPSHOT_PERCENT=${CHANNEL_SNAPSHOT_PERCENT:-50}
: "${SLURM_GPUS_ON_NODE:=$(nvidia-smi -L | grep -c '^GPU ')}";export SLURM_GPUS_ON_NODE
ls -l $CHANNEL_RUN/recipe.json $PAIR/ac/ac_modules.pt $PAIR/ac/side_lora/adapter_model.safetensors $PAIR/reader/adapter_model.safetensors
grep -q "\"$PAIR/ac\"" $CHANNEL_RUN/recipe.json || { echo "the recipe's initial pair is not $PAIR" >&2; exit 2; }
[[ ! -e $PAIR/ac/optimizer.pt && ! -e $PAIR/reader/optimizer.pt ]] || { echo 'an optimizer.pt sits in the starting pair'; exit 2; }
mkdir -p $CHANNEL_RUN/AUTOTUNE/profiles
cp -n $R/mixed_v1/run/AUTOTUNE/profiles/*.json $CHANNEL_RUN/AUTOTUNE/profiles/ 2>/dev/null || true
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID run $RUN restart ${SLURM_RESTART_COUNT:-0} on $(hostname): $SLURM_GPUS_ON_NODE x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); scratch $(df -h --output=avail $R | tail -1)"
exec bash $R/paper_compaction_v1/jobs/launch_continuation.sbatch
