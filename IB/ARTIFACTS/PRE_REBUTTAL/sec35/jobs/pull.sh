#!/bin/bash
# Pull one run folder back to this container (run here):  bash jobs/pull.sh csail|orcd RUN [--upload]
#   W/runs/RUN/  ->  $SEC45_PULL_ROOT/RUN/  (default IB/TMP/PAPER_SEC45/runs/RUN)
# Excluded: AC_MODELS/ LORAS/ (hard-link duplicates of ckpt/), teacher_snapshots, AUTOTUNE, optimizer.pt, *.tmp, *.part.
# DONE is pulled in a second pass, after everything else, so a local DONE implies a complete local copy (stage_init.sh also
# re-checks the pair hash). --upload: then copy the pulled folder to $S3/runs/RUN/ (s3x: never optimizer.pt, DONE last).
# Exit 97: the shared ssh connection is down. SEC45_DRY_RUN=1 prints the commands without any ssh.
set -euo pipefail
source "$(dirname "$0")/cluster_lib.sh"
[ $# -ge 2 ] || { sed -n 2,8p "$0"; exit 2; }
cluster $1; RUN=$2; UPLOAD=${3:-}
L=$PULL_ROOT/$RUN; mkdir -p $L
EXCL=(--exclude AC_MODELS --exclude LORAS --exclude teacher_snapshots --exclude AUTOTUNE --exclude optimizer.pt --exclude '*.tmp' --exclude '*.part')
xfer -az --no-owner --no-group "${EXCL[@]}" --exclude /DONE $H:$W/runs/$RUN/ $L/
xfer -az --no-owner --no-group --ignore-missing-args $H:$W/runs/$RUN/DONE $L/
if [ -f $L/DONE ]; then echo "$RUN: DONE $(cat $L/DONE)"; else echo "$RUN: not finished (no DONE yet); pulled what exists"; fi
[ -f $L/logs/job.log ] && tail -n 3 $L/logs/job.log
if [ "$UPLOAD" = --upload ]; then
  [ "$DRY" = 1 ] && { echo "$LOCAL_PY $KIT/kit/s3x.py putdir $L $S3/runs/$RUN/ --exclude AC_MODELS LORAS teacher_snapshots optimizer.pt AUTOTUNE"; exit 0; }
  $LOCAL_PY $KIT/kit/s3x.py putdir $L $S3/runs/$RUN/ --exclude AC_MODELS LORAS teacher_snapshots optimizer.pt AUTOTUNE
fi
