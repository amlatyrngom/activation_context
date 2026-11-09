#!/bin/bash
# Stage R's final pair for T on a cluster (run here, after jobs/pull.sh pulled R):  bash jobs/stage_init.sh csail|orcd R_<mode> [--allow-invalid]
#   $SEC45_PULL_ROOT/R_<mode>/sync/ckpt/final/ + its DONE  ->  W/init/R_<mode>_final/ (+ DONE inside), replacing any older copy
# Checked here first: DONE exists, says valid (unless --allow-invalid; the T job then needs SEC45_ALLOW_INVALID_INIT=1 too),
# and the local pair's hash equals DONE's final_sha256. The T job checks the hash again on the cluster before training.
# Not needed when R and T run on the same cluster (the T job hard-links W/runs/R_<mode>/sync/ckpt/final itself).
# Exit 97: the shared ssh connection is down. SEC45_DRY_RUN=1 prints the commands without any ssh.
set -euo pipefail
source "$(dirname "$0")/cluster_lib.sh"
[ $# -ge 2 ] || { sed -n 2,8p "$0"; exit 2; }
R=$2; ALLOW=${3:-}
case $R in R_in|R_res|R_kv) ;; *) echo "the run must be R_in, R_res or R_kv" >&2; exit 2;; esac
L=$PULL_ROOT/$R; F=$L/sync/ckpt/final
[ -f $L/DONE ] || { echo "$L/DONE is missing: R is not finished or not pulled (bash jobs/pull.sh CLUSTER $R)" >&2; exit 2; }
[ -f $F/meta.json ] || { echo "$F/meta.json is missing" >&2; exit 2; }
grep -q '"valid": true' $L/DONE || [ "$ALLOW" = --allow-invalid ] || { echo "$R is INVALID ($(cat $L/DONE)); --allow-invalid to stage it anyway" >&2; exit 2; }
$LOCAL_PY $KIT/kit/pairhash.py $F --check $L/DONE || { echo "the local pair $F does not match $L/DONE: pull again" >&2; exit 2; }
cluster $1
T=$W/init/${R}_final
remote 60 "mkdir -p $T.staging"
xfer -a --no-owner --no-group --delete --exclude optimizer.pt --exclude '*.tmp' $F/ $H:$T.staging/
xfer -a --no-owner --no-group $L/DONE $H:$T.staging/DONE
remote 60 "rm -rf $T && mv $T.staging $T && cat $T/DONE"
echo "staged $R's final pair on $H: $T (the T job verifies its hash before training)"
