#!/bin/bash
# Host side (2026-11-18): keep the ORCD per-ratio runs going past mit_preemptable's 2-day wall limit.
#   nohup bash per_ratio_keeper.sh > LOG 2>&1 < /dev/null &
# Every 10 min, for each ORCD winner in compaction_stage1/per_ratio_winners.txt ("RUN orcd JOBID ..."): when its current job has left the
# queue and the run has no ckpt/final yet (ORCD runs/RUN/sync/ckpt/final/meta.json), the job's sacct state decides:
#   TIMEOUT / PREEMPTED / NODE_FAIL / BOOT_FAIL / DEADLINE  -> resubmit the same RUN + PACKET on ORCD (inputs already staged; the run resumes
#                                                             from its recovery slot), at most 4 times per run; per_ratio_current.txt holds
#                                                             the current job id per run
#   anything else (FAILED, OUT_OF_MEMORY, CANCELLED, ...)   -> logged as "NEEDS ATTENTION" and not resubmitted
# (A CSAIL copy cannot take over: the trainer resumes only with the first attempt's arguments, whose paths are ORCD's.)
# Shared connection only; exit 97 when it is down (logged, never retried).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/../../sec35/jobs/cluster_lib.sh
cluster orcd
W=/home/ngom/orcd/scratch/activation/cmp1
T=$REPO/IB/TMP/PRE_REBUTTAL/compaction_stage1; WIN=$T/per_ratio_winners.txt; CUR=$T/per_ratio_current.txt
EXPORT="NIL,CMP_INIT=R_cmp_lcrec_final,CMP_BUNDLE=cmp_csail_code_v2,CMP_PANEL=panel_dev75_tv,CMP_TEACHER=teacher_dev75_tv.pt,CMP_REF=proxy_ref_tv.json,CMP_REPORT_EVERY=80,CMP_CKPT_EVERY=40"
say() { echo "$(date -u +%FT%TZ) [keeper] $*"; }
[ -f $CUR ] || awk '$2=="orcd"{print $1, $3, 0}' $WIN > $CUR          # RUN JOBID RESUBMITS
while true; do
  ssh -O check orcd >/dev/null 2>&1 || { say "EXIT 97: orcd shared connection down; stopped, not retrying"; exit 97; }
  left=0
  while read -r RUN JOB N <&3; do
    if remote 60 "[ -f $W/runs/$RUN/sync/ckpt/final/meta.json ]" < /dev/null; then say "$RUN: ckpt/final exists (job $JOB)"; continue; fi
    left=$((left+1)); [ "$N" = 99 ] && continue                          # needs attention: not touched again
    q=$(remote 60 "squeue -j $JOB -h -o %T 2>/dev/null" < /dev/null)
    [ -n "$q" ] && continue                                             # still queued / running
    st=$(remote 60 "sacct -j $JOB -X -n -o State%20 2>/dev/null | head -1" < /dev/null | xargs)
    case "$st" in
      TIMEOUT*|PREEMPTED*|NODE_FAIL*|BOOT_FAIL*|DEADLINE*)
        if [ "$N" -ge 4 ]; then say "$RUN: job $JOB $st, 4 resubmits used: NEEDS ATTENTION"; continue; fi
        PACKET=s1kl_tv_$(sed -E 's/^S1_tv_kl_(r[0-9]+)_.*/\1/' <<<"$RUN")_c162024_s012_x528_v1
        out=$(remote 60 "cd $W && sbatch --export=$EXPORT --partition=mit_preemptable --requeue --gres=gpu:rtx_pro_6000:1 --time=2-00:00:00 \
              --cpus-per-task=8 --mem=96G --job-name=$RUN --output=$W/logs/${RUN}_%j.out jobs/cmp_stage1_job.sh $RUN $PACKET" < /dev/null)
        new=$(grep -oE "Submitted batch job [0-9]+" <<<"$out" | awk '{print $4}')
        if [ -n "$new" ]; then sed -i "s/^$RUN $JOB $N\$/$RUN $new $((N+1))/" $CUR; echo "$RUN orcd $new" >> $T/per_ratio_jobs.txt
          say "$RUN: job $JOB ended $st without ckpt/final: resubmitted as $new (resubmit $((N+1)))"
        else say "$RUN: resubmit after $st failed: $out"; fi;;
      *) say "$RUN: job $JOB ended '$st' without ckpt/final: NEEDS ATTENTION (not resubmitted)"; sed -i "s/^$RUN $JOB $N\$/$RUN $JOB 99/" $CUR;;
    esac
  done 3< $CUR
  [ $left = 0 ] && { say "every run has ckpt/final: done"; exit 0; }
  [ $left = $(awk '$3 == 99' $CUR | wc -l) ] && { say "only runs needing attention are left: stopped"; exit 1; }
  sleep 600
done
