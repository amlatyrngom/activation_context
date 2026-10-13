#!/bin/bash
# Submit with site/account/partition, --gpus=1, 2 or 4, --nodes=1, CPU RAM and wall time.
# Required exported variables: CHANNEL_SOURCE, CHANNEL_PYTHON, CHANNEL_RUN,
# CHANNEL_QUOTA_ROOT, CHANNEL_QUOTA_GIB; optional CHANNEL_TASK_PANEL (evaluation-only panel folder).
# Preparation precedes GPU allocation.
#SBATCH --nodes=1
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
: "${CHANNEL_SOURCE:?}" "${CHANNEL_PYTHON:?}" "${CHANNEL_RUN:?}" "${CHANNEL_QUOTA_ROOT:?}" "${CHANNEL_QUOTA_GIB:?}"
CHANNEL_GPUS=${SLURM_GPUS_ON_NODE:?}
[[ "$CHANNEL_GPUS" == 1 || "$CHANNEL_GPUS" == 2 || "$CHANNEL_GPUS" == 4 ]] || { echo 'Request exactly 1, 2 or 4 GPUs on one node'; exit 2; }
export PYTHONPATH="$CHANNEL_SOURCE${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TZ=UTC
# Configure cuBLAS before torchrun initializes CUDA; strict mode itself remains opt-in.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
[[ ! -e "$CHANNEL_RUN/PAUSE" ]] || exit 0
CHANNEL_STOP=0
CHANNEL_CHILD=
stop_channel(){ CHANNEL_STOP=1; touch "$CHANNEL_RUN/PAUSE_ALLOCATION"; }
# The rank workers see this shared marker through their deadline check; torchrun stays alive to save.
rm -f "$CHANNEL_RUN/PAUSE_ALLOCATION"
trap stop_channel USR1
: "${SLURM_JOB_END_TIME:?Actual allocation end required}"
CHANNEL_END=$SLURM_JOB_END_TIME
CHANNEL_RECORD=$(timeout 15s scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null) || CHANNEL_RECORD=
if [[ "$CHANNEL_RECORD" == *"JobId=$SLURM_JOB_ID "* && "$CHANNEL_RECORD" == *"JobState=RUNNING "* ]]; then
  CHANNEL_START_TEXT=$(sed -n 's/.* StartTime=\([^ ]*\).*/\1/p' <<< "$CHANNEL_RECORD")
  CHANNEL_END_TEXT=$(sed -n 's/.* EndTime=\([^ ]*\).*/\1/p' <<< "$CHANNEL_RECORD")
  CHANNEL_START_ACTUAL=$(date -u -d "$CHANNEL_START_TEXT" +%s 2>/dev/null) || CHANNEL_START_ACTUAL=
  CHANNEL_END_ACTUAL=$(date -u -d "$CHANNEL_END_TEXT" +%s 2>/dev/null) || CHANNEL_END_ACTUAL=
  if [[ "$CHANNEL_START_ACTUAL" == "${SLURM_JOB_START_TIME:-}" && "$CHANNEL_END_ACTUAL" =~ ^[0-9]+$ ]] && (( CHANNEL_END_ACTUAL > $(date +%s) )); then
    CHANNEL_END=$CHANNEL_END_ACTUAL
  fi
fi
CHANNEL_SECONDS=$(( CHANNEL_END - $(date +%s) - 900 ))
(( CHANNEL_SECONDS > 0 )) || exit 2
CHANNEL_POLICY=()
[[ ! -n "${CHANNEL_EXECUTION_POLICY:-}" ]] || CHANNEL_POLICY=(--execution-policy "$CHANNEL_EXECUTION_POLICY")
CHANNEL_PANEL=()
[[ ! -n "${CHANNEL_TASK_PANEL:-}" ]] || CHANNEL_PANEL=(--task-panel "$CHANNEL_TASK_PANEL")
CHANNEL_KERNELS=()
case "${CHANNEL_DETERMINISTIC:-}" in
  1) CHANNEL_KERNELS=(--deterministic) ;;
  0) CHANNEL_KERNELS=(--no-deterministic) ;;
  '') ;;
  *) echo 'CHANNEL_DETERMINISTIC must be 0 or 1'; exit 2 ;;
esac
"$CHANNEL_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$CHANNEL_GPUS" \
 "$CHANNEL_SOURCE/IB/ARTIFACTS/FIRST_TASK_AWARE_FINETUNING/bench/run_channel_mixture.py" train \
 --output "$CHANNEL_RUN" --quota-root "$CHANNEL_QUOTA_ROOT" --quota-gib "$CHANNEL_QUOTA_GIB" \
 --seconds "$CHANNEL_SECONDS" "${CHANNEL_POLICY[@]}" "${CHANNEL_PANEL[@]}" "${CHANNEL_KERNELS[@]}" &
CHANNEL_CHILD=$!
set +e
wait "$CHANNEL_CHILD"; CHANNEL_EXIT=$?
# Bash wait returns on USR1 while torchrun is still saving.
if kill -0 "$CHANNEL_CHILD" 2>/dev/null; then wait "$CHANNEL_CHILD"; CHANNEL_EXIT=$?; fi
set -e
if (( CHANNEL_EXIT != 0 )); then exit "$CHANNEL_EXIT"; fi
[[ ! -e "$CHANNEL_RUN/PAUSE" ]] || exit 0
CHANNEL_REASON=$("$CHANNEL_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["stop_reason"])' "$CHANNEL_RUN/run_status.json")
if [[ "$CHANNEL_REASON" == "duration limit"* ]]; then scontrol requeue "$SLURM_JOB_ID"; fi
