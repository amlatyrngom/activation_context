#!/bin/bash
# Host side (2026-11-17): back up a stage-1 run on CSAIL or ORCD to S3, relayed through this host (neither cluster has S3 credentials).
#   nohup bash cluster_relay.sh CLUSTER RUN > LOG 2>&1 &          (CLUSTER = csail | orcd)
# Every 20 min: rsync the cluster's runs/RUN/ (no JIT caches) into IB/TMP/PRE_REBUTTAL/compaction_stage1/cluster_runs/RUN/ (--delete: the
# local copy mirrors the cluster, i.e. ONE recovery slot, as the trainer keeps), then `aws s3 sync` it to S3 compaction/stage1/runs/RUN/
# (no --delete: S3 keeps every slot it saw, optimizer.pt included, so EC2 / another cluster can resume). Stops after the pass that
# carries proxy/final.json. Only through the user's shared connection (sec35/jobs/cluster_lib.sh); if it is down: exit 97, never retry.
# Nothing read from the cluster is executed here.
set -uo pipefail
CL=${1:?csail|orcd}; RUN=${2:?RUN}; EVERY=${RELAY_EVERY:-1200}
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/../../sec35/jobs/cluster_lib.sh
cluster $CL
case $CL in csail) W=/data/scratch-fast/ngom/cmp_csail;; orcd) W=/home/ngom/orcd/scratch/activation/cmp1;; esac
L=$REPO/IB/TMP/PRE_REBUTTAL/compaction_stage1/cluster_runs/$RUN; mkdir -p $L
S3R=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/stage1/runs/$RUN
say() { echo "$(date -u +%FT%TZ) [relay $CL $RUN] $*"; }
while true; do
  ssh -O check $H >/dev/null 2>&1 || { say "EXIT 97: shared connection to $H down (needs the user's interactive login); stopped, not retrying"; exit 97; }
  if remote 60 "[ -d $W/runs/$RUN ]"; then
    xfer -a --delete --exclude cache/ --exclude triton_cache/ --exclude '*.tmp' --exclude '*.part' $H:$W/runs/$RUN/ $L/ \
      && aws s3 sync --only-show-errors --exclude 'cache/*' $L/ $S3R/ \
      && say "synced: $(find $L -type f | wc -l) files; slot $(ls -d $L/sync/AC_MODELS/*/epoch_* 2>/dev/null | tail -1 | xargs -r basename); panels $(cat $L/sync/$RUN.held.jsonl 2>/dev/null | wc -l)" \
      || say "pass failed (next pass in $EVERY s)"
    [ -f $L/proxy/final.json ] && { say "final proxy relayed: done"; exit 0; }
  else
    say "no runs/$RUN on $CL yet"
  fi
  sleep $EVERY
done
