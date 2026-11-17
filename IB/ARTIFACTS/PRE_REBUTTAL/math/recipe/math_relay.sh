#!/bin/bash
# Host side (2026-11-17): back up the math recipe's cluster outputs to S3, relayed through this host (the clusters have no S3 credentials).
#   nohup bash math_relay.sh CLUSTER > LOG 2>&1 &        (CLUSTER = csail | orcd)
# Every RELAY_EVERY s (default 1200): rsync W/{gen,runs,eval}/ (no JIT caches, no partial files) into
# IB/TMP/PRE_REBUTTAL/math_recipe/cluster/CLUSTER/ (--delete: the local copy mirrors the cluster, so ONE recovery slot per run), then
# `aws s3 sync` it to S3 math/recipe/cluster/CLUSTER/ (no --delete: S3 keeps every slot it saw, optimizer.pt included, so another
# machine can resume). Only through the user's shared connection (sec35/jobs/cluster_lib.sh); down: exit 97, never retry.
# Nothing read from the cluster is executed here.
set -uo pipefail
CL=${1:?csail|orcd}; EVERY=${RELAY_EVERY:-1200}
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/../../sec35/jobs/cluster_lib.sh
cluster $CL
case $CL in csail) W=/data/scratch-fast/ngom/cmp_csail/math1;; orcd) W=/home/ngom/orcd/scratch/activation/math1;; esac
L=$REPO/IB/TMP/PRE_REBUTTAL/math_recipe/cluster/$CL; mkdir -p $L
S3R=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/math/recipe/cluster/$CL
say() { echo "$(date -u +%FT%TZ) [math relay $CL] $*"; }
while true; do
  ssh -O check $H >/dev/null 2>&1 || { say "EXIT 97: shared connection to $H down (needs the user's interactive login); stopped, not retrying"; exit 97; }
  for d in gen runs eval; do
    if remote 60 "[ -d $W/$d ]"; then
      mkdir -p $L/$d
      xfer -a --delete --exclude cache/ --exclude triton_cache/ --exclude '*.tmp' --exclude '*.part' --exclude '*.claim' $H:$W/$d/ $L/$d/ \
        && aws s3 sync --only-show-errors $L/$d/ $S3R/$d/ || say "$d: pass failed (next pass in $EVERY s)"
    fi
  done
  say "synced: $(find $L -type f | wc -l) files, $(du -sh $L | cut -f1)"
  [ -f $L/STOP ] && { say "STOP file: done"; exit 0; }
  sleep $EVERY
done
