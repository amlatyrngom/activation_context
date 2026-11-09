#!/bin/bash
# H8a evaluations (Oct 17 ≈02:55Z): one GPU's list of eval units. Waits for a PID to exit (0: none), then before each unit for
# the GPU < 3 GB and >= 40G free on /; runs the units in order; then the optional AFTER command (H8 again), else leaves the GPU empty.
#   h8a_eval_queue.sh <gpu> <pid or 0> <unit> ... ["AFTER:<command>"]
set -uo pipefail
G=$1; P=$2; shift 2
export CUDA_VISIBLE_DEVICES=$G
cd /root/activation_artifacts/pc
UNITS=(); AFTER=""
for arg in "$@"; do if [[ $arg == AFTER:* ]]; then AFTER=${arg#AFTER:}; else UNITS+=("$arg"); fi; done
[ "$P" != 0 ] && while kill -0 $P 2>/dev/null; do sleep 30; done
guards() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)" -lt 3000 ] && \
        [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -ge 40 ]; do sleep 30; done
}
for unit in "${UNITS[@]}"; do
  guards
  echo "$(date -u +%FT%TZ) GPU $G: $unit"; bash -c "$unit" < /dev/null; echo "$(date -u +%FT%TZ) GPU $G: ended ($?)"
done
echo "$(date -u +%FT%TZ) h8a_eval_queue GPU $G: evaluations done"
if [ -n "$AFTER" ]; then guards; echo "$(date -u +%FT%TZ) AFTER: $AFTER"; bash -c "$AFTER"; fi
