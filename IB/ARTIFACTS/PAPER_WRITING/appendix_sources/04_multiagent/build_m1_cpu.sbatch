#!/bin/bash
#SBATCH --job-name=tqa_m1_build
#SBATCH --account=quanta
#SBATCH --partition=quanta-h200
#SBATCH --qos=quanta-main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/scratch-fast/ngom/activation/paper_traj_qa_v1/r2/logs/%x_%j.log
# M1 v2 (synthesis fork, Oct 16; copy of r2/jobs/build_r2_cpu.sbatch, the start pair read from INIT): the M1 DeepMath packet.
# CPU only (no GPU): the round-2 synthesis SFT packet (kit/prepare_synth_sft_v3.py: parts joined on content_id) from r2/<MATERIAL>
# (made locally by make_synth_sft_material_v2.py), started from pairs/tqa_own_ro_final (read in place on /data/scratch), base config
# = the own-question run's recipe, 1 epoch, 2e-5 / 2e-5, 8 held training tasks per benchmark. Written on scratch-fast.
# Read from the SUBMITTING environment: MATERIAL (folder under r2/), OUT (packet name under r2/packets), INIT (pair folder with ac/ and reader/).
set -euo pipefail
R=/data/scratch/ngom/activation; W=$R/paper_traj_qa_v1; S=$R/probe_p_v1/source_p; PAIR=${INIT:?INIT=<pair folder>}
F=/data/scratch-fast/ngom/activation/paper_traj_qa_v1/r2
export HOME=$F/home HF_HOME=$R/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false TZ=UTC
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 ACTIVATION_SYNC_ROOT=$F/cpu_sync
export CHANNEL_BENCH_DIR=$S/IB/ARTIFACTS/FIRST_TASK_AWARE_FINETUNING/bench
export PYTHONPATH=$S:$CHANNEL_BENCH_DIR
PY=$R/venv/bin/python; OUT=$F/packets/${OUT:?}
[ ! -e $OUT ] || { echo "$OUT exists"; exit 2; }
[[ ! -e $PAIR/ac/optimizer.pt && ! -e $PAIR/reader/optimizer.pt ]] || { echo 'an optimizer.pt sits in the starting pair'; exit 2; }
mkdir -p $F/packets $ACTIVATION_SYNC_ROOT $HOME
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID on $(hostname): $F/${MATERIAL:?} -> $OUT; scratch-fast free $(df -h --output=avail $F | tail -1)"
cd $F
$PY $F/kit_v2/prepare_synth_sft_v3.py --material $F/$MATERIAL --output $OUT --initial-pair $PAIR \
  --base-recipe $W/tqa_own_v1/packets/tqa_own_ro_v1/recipe.json --exclude-tasks $F/$MATERIAL/excluded_tasks.json \
  --held-tasks 8 --epochs 1 --lr 2e-5 --workers ${SLURM_CPUS_PER_TASK:-8}
du -sh $OUT
echo "$(date -u +%FT%TZ) done"
