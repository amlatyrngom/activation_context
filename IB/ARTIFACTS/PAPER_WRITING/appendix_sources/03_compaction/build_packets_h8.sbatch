#!/bin/bash
#SBATCH --job-name=pc_build_h8
#SBATCH --account=quanta
#SBATCH --partition=quanta-h200
#SBATCH --qos=quanta-main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/scratch-fast/ngom/activation/paper_compaction_v1/logs/%x_%j.log
# H8 (Oct 16): build_packets_h7.sbatch running prepare_agent_continuation_h8.py (over-guard held runs leave the panel).
# H7 packets (Oct 15): build_packets_fast.sbatch with the H7 builder (prepare_agent_continuation_h7.py: per-line size slices,
# --median-below for H7M) read from a kit folder on /data/scratch-fast (our /data/scratch quota is exhausted).
# Read from the SUBMITTING environment:
#   MATERIAL  material file(s) under paper_compaction_v1/material (space separated)
#   OUT       packet folder name under paper_compaction_v1/packets (must not exist)
#   KITF      kit folder name under /data/scratch-fast/.../paper_compaction_v1 (holds prepare_agent_continuation_h7.py)
#   PAIR      the starting pair (its ac/ and reader/)
#   BUILD     further prepare_agent_continuation_h7.py arguments
set -euo pipefail
R=/data/scratch/ngom/activation; W=/data/scratch-fast/ngom/activation/paper_compaction_v1; O=$R/paper_compaction_v1; S=$R/probe_p_v1/source_p; PAIR=${PAIR:?PAIR=the starting pair}
export HOME=$W/home TRITON_CACHE_DIR=$W/triton_cache TORCHINDUCTOR_CACHE_DIR=$W/inductor_cache HF_HOME=$R/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false TZ=UTC CUDA_VISIBLE_DEVICES=
K=$W/${KITF:?KITF=<kit folder name>}   # holds prepare_agent_continuation_h7.py, ac_sizes.py, check_continuation_packet.py, resume_program.py
export PYTHONPATH=$S:$K CONTINUATION_BENCH_DIR=$S/IB/ARTIFACTS/FIRST_TASK_AWARE_FINETUNING/bench
PY=$R/venv/bin/python
[ ! -e $W/packets/${OUT:?OUT=<packet folder name>} ] || { echo "packets/$OUT exists"; exit 2; }
mkdir -p $W/packets $HOME
cd $W/material
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID on $(hostname): build packets/$OUT from ${MATERIAL:?} pair $PAIR with ${BUILD:-}"
$PY $K/prepare_agent_continuation_h8.py --material $MATERIAL --output $W/packets/$OUT --initial-ac $PAIR/ac --initial-reader $PAIR/reader --workers ${SLURM_CPUS_PER_TASK:-8} ${BUILD:-}
for packet in $W/packets/$OUT $W/packets/$OUT/*/; do [ -e $packet/recipe.json ] && $PY $K/check_continuation_packet.py --packet $packet; done
du -sh $W/packets/$OUT
echo "$(date -u +%FT%TZ) done"
