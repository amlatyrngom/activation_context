#!/bin/bash
# Host side (2026-11-17): first copy to start wins, for math-recipe jobs queued on several clusters (compaction/stage1/per_ratio_arbiter.sh).
#   nohup bash math_arbiter.sh > LOG 2>&1 &
# Reads IB/TMP/PRE_REBUTTAL/math_recipe/arbiter_jobs.txt (lines "KEY CLUSTER JOBID", KEY = the run or gen shard; append a line per copy).
# Every ARB_EVERY s (default 300), per KEY without a winner: each copy's state (squeue -j ID, by id only); the first RUNNING copy wins,
# every other copy of that KEY is cancelled by its id (scancel ID: only ids listed here, all ours). Winners -> arbiter_winners.txt
# ("KEY CLUSTER JOBID UTC"; downstream steps read a KEY's outputs from its winner's cluster). The first CSAIL winner starts
# math_relay.sh csail (S3 backup; ORCD's relay already runs). Copies on one cluster also claim the run folder / gen shard in the job
# (math_job.sh), and the two clusters' file systems and S3 prefixes (cluster/<CL>/) are separate: two copies never write one output.
# A cluster whose shared connection is down (exit 97) is not contacted again (logged; never retried).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/../../sec35/jobs/cluster_lib.sh
T=$REPO/IB/TMP/PRE_REBUTTAL/math_recipe; JOBS=$T/arbiter_jobs.txt; WIN=$T/arbiter_winners.txt; touch $JOBS $WIN $T/arbiter_cancelled.txt; EVERY=${ARB_EVERY:-300}
declare -A DOWN; RELAY_CSAIL=
say() { echo "$(date -u +%FT%TZ) [math arbiter] $*"; }
host_of() { ( cluster $1 >/dev/null; echo $H ); }
state() {   # state CLUSTER JOBID -> PENDING / RUNNING / ... / GONE / DOWN
  [ -n "${DOWN[$1]:-}" ] && { echo DOWN; return; }
  ssh -O check $(host_of $1) >/dev/null 2>&1 || { DOWN[$1]=1; say "EXIT 97: $1 shared connection down: not contacted again"; echo DOWN; return; }
  ( cluster $1 >/dev/null; s=$(remote 60 "squeue -j $2 -h -o %T 2>/dev/null"); echo ${s:-GONE} )
}
cancel() { ( cluster $1 >/dev/null; remote 60 "squeue -u ngom -h -j $2 -o %i | grep -qx $2 && scancel $2" ) && say "cancelled $1 job $2"; }
while true; do
  for KEY in $(awk '{print $1}' $JOBS | sort -u); do
    if grep -q "^$KEY " $WIN; then                      # a copy added after the decision: cancel it too (once)
      w=$(grep "^$KEY " $WIN | head -1 | awk '{print $2" "$3}')
      while read -r k cl id; do [ "$k" = "$KEY" ] && [ "$cl $id" != "$w" ] && [ -z "${DOWN[$cl]:-}" ] && ! grep -qx "$cl $id" $T/arbiter_cancelled.txt \
        && { cancel $cl $id </dev/null; echo "$cl $id" >> $T/arbiter_cancelled.txt; }; done < $JOBS
      continue; fi
    winner=
    while read -r k cl id; do [ "$k" = "$KEY" ] || continue
      s=$(state $cl $id </dev/null); say "$KEY $cl $id $s"
      [ "$s" = RUNNING ] && [ -z "$winner" ] && winner="$cl $id"
    done < $JOBS
    [ -z "$winner" ] && continue
    echo "$KEY $winner $(date -u +%FT%TZ)" >> $WIN; say "$KEY: winner $winner"
    while read -r k cl id; do [ "$k" = "$KEY" ] && [ "$cl $id" != "$winner" ] && [ -z "${DOWN[$cl]:-}" ] && cancel $cl $id </dev/null; done < $JOBS
    if [ "${winner%% *}" = csail ] && [ -z "$RELAY_CSAIL" ] && ! pgrep -f "math_relay.sh csail" >/dev/null; then
      nohup bash $HERE/math_relay.sh csail >> $T/logs/relay_csail.log 2>&1 & RELAY_CSAIL=$!; say "csail relay started (pid $RELAY_CSAIL)"; fi
  done
  n=$(awk '{print $1}' $JOBS | sort -u | wc -l)
  [ $n -gt 0 ] && [ $n = "$(awk '{print $1}' $WIN | sort -u | wc -l)" ] && [ ! -f $T/ARBITER_KEEP ] && { say "every key has a winner: done"; exit 0; }
  sleep $EVERY
done
