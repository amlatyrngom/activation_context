#!/bin/bash
# Host side (2026-11-17): first copy to start wins, for runs submitted to several clusters (per_ratio_submit.sh).
#   nohup bash per_ratio_arbiter.sh > LOG 2>&1 &
# Reads IB/TMP/PRE_REBUTTAL/compaction_stage1/per_ratio_jobs.txt (lines "RUN CLUSTER JOBID"; append a line when a copy is submitted).
# Every 10 min, per RUN without a winner: the state of each copy (squeue -j ID, by id only); the first one RUNNING is the winner: every
# other copy of that RUN is cancelled by its id (scancel ID, our own jobs only), and cluster_relay.sh CLUSTER RUN starts (S3 backup).
# Winners go to per_ratio_winners.txt. A cluster whose shared connection is down (exit 97) is not contacted again (logged; never retried).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/../../sec35/jobs/cluster_lib.sh
T=$REPO/IB/TMP/PRE_REBUTTAL/compaction_stage1; JOBS=$T/per_ratio_jobs.txt; WIN=$T/per_ratio_winners.txt; touch $WIN
declare -A DOWN
say() { echo "$(date -u +%FT%TZ) [arbiter] $*"; }
state() {   # state CLUSTER JOBID -> PENDING / RUNNING / ... / GONE / DOWN
  [ -n "${DOWN[$1]:-}" ] && { echo DOWN; return; }
  ssh -O check $1 >/dev/null 2>&1 || { DOWN[$1]=1; say "EXIT 97: $1 shared connection down: not contacted again"; echo DOWN; return; }
  ( cluster $1 >/dev/null; s=$(remote 60 "squeue -j $2 -h -o %T 2>/dev/null"); echo ${s:-GONE} )
}
cancel() { ( cluster $1 >/dev/null; remote 60 "scancel $2" ) && say "cancelled $1 job $2"; }
while true; do
  for RUN in $(awk '{print $1}' $JOBS 2>/dev/null | sort -u); do
    grep -q "^$RUN " $WIN && continue
    winner=
    while read -r r cl id; do [ "$r" = "$RUN" ] || continue
      s=$(state $cl $id); say "$RUN $cl $id $s"
      [ "$s" = RUNNING ] && [ -z "$winner" ] && winner="$cl $id"
    done < $JOBS
    [ -z "$winner" ] && continue
    echo "$RUN $winner $(date -u +%FT%TZ)" >> $WIN; say "$RUN: winner $winner"
    while read -r r cl id; do [ "$r" = "$RUN" ] && [ "$cl $id" != "$winner" ] && [ -z "${DOWN[$cl]:-}" ] && cancel $cl $id; done < $JOBS
    nohup bash $HERE/cluster_relay.sh ${winner%% *} $RUN >> $T/logs/relay_$RUN.log 2>&1 &
    say "$RUN: relay started (pid $!)"
  done
  [ "$(awk '{print $1}' $JOBS 2>/dev/null | sort -u | wc -l)" -gt 0 ] && [ "$(awk '{print $1}' $JOBS | sort -u | wc -l)" = "$(wc -l < $WIN)" ] && { say "every run has a winner: done"; exit 0; }
  sleep 600
done
