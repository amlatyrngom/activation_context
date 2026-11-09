#!/bin/bash
#SBATCH --job-name=tqa_r2_h2
#SBATCH --account=quanta
#SBATCH --partition=quanta-h200
#SBATCH --qos=quanta-main
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=18:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/scratch-fast/ngom/activation/paper_traj_qa_v1/r2/logs/train_%x_%j.log
# Round-2 synthesis SFT on the x86 H200 pair (data parallel, venv), run folder r2/<RUN> on scratch-fast; as
# synth_add_v1/jobs/train_gh200_fast2.sbatch with a rows-only packet (deltas_off_training true).
# v2 (Oct 16 04:30Z): 18 h wall time (the 1,234-update union run needs ≈12.4 h at ≈36 s/update) and snapshots every 50 % by default.
# Never more than two GH200 jobs at once over every experiment (compaction included).
#   RUN=packets/<name> sbatch jobs/train_r2_gh200.sbatch
set -euo pipefail
[ "$(uname -m)" = x86_64 ] || { echo "this job file is for the x86 H200 node" >&2; exit 2; }
R=/data/scratch/ngom/activation; G=$R/gh200_v1; W=/data/scratch-fast/ngom/activation/paper_traj_qa_v1; F=$W/r2; S=$R/probe_p_v1/source_p
export HOME=$F/home HF_HOME=$R/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TZ=UTC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True UV_CACHE_DIR=$F/uv_cache TRITON_CACHE_DIR=$F/triton_cache_x86 TORCHINDUCTOR_CACHE_DIR=$F/inductor_cache_x86
export CHANNEL_SOURCE=$S CHANNEL_PYTHON=$R/venv/bin/python CHANNEL_RUN=$F/${RUN:?RUN=packets/<name>}
export CHANNEL_QUOTA_ROOT=$W CHANNEL_QUOTA_GIB=100 CHANNEL_SNAPSHOT_PERCENT=${CHANNEL_SNAPSHOT_PERCENT:-50}
: "${SLURM_GPUS_ON_NODE:=$(nvidia-smi -L | grep -c '^GPU ')}";export SLURM_GPUS_ON_NODE
PAIR=$(sed -n 's/.*"ac_checkpoint": "\(.*\)\/ac".*/\1/p' $CHANNEL_RUN/recipe.json | head -1)
ls -l $CHANNEL_RUN/recipe.json $PAIR/ac/ac_modules.pt $PAIR/ac/side_lora/adapter_model.safetensors $PAIR/reader/adapter_model.safetensors
mkdir -p $HOME $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $UV_CACHE_DIR
grep -q '"deltas_off_training": true' $CHANNEL_RUN/recipe.json || { echo "the recipe is not rows-only" >&2; exit 2; }
[[ ! -e $PAIR/ac/optimizer.pt && ! -e $PAIR/reader/optimizer.pt ]] || { echo 'an optimizer.pt sits in the starting pair'; exit 2; }
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID run $RUN restart ${SLURM_RESTART_COUNT:-0} on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader); pair $PAIR"
exec bash $R/paper_compaction_v1/jobs/launch_continuation.sbatch
