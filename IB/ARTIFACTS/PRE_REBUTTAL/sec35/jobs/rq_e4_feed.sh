#!/bin/bash
# Host loop (2026-11-14): reads of the two 4B-encoder KL 20I finals on ORCD (jobs/rq_e4_read_orcd.sh, RQ_ENCODER 4B: the 2B-encoder final-reads
# loop cannot read them). User 2026-11-14: the FROZEN 4B run has priority over the LoRA one, so it is checked and submitted first.
#   frozen: T_res_rq9e4fr_20I_KL (EC2 rqq) — when S3 DONE: final pair fetched, sent to $W/rq_eval_e4x/<N>/<N> (atomic .tmp rename), read submitted
#   LoRA:   T_res_rq9e4_20I_KL (ORCD)     — when $W/runs/.../DONE: final hard-linked to $W/rq_eval_e4x/<N>/<N>, read submitted
# Every 15 min; exits when both reads are submitted.
#   nohup bash IB/ARTIFACTS/PRE_REBUTTAL/sec35/jobs/rq_e4_feed.sh >> IB/TMP/PRE_REBUTTAL/e4_feed.log 2>&1 &
set -uo pipefail
cd /home/ubuntu/activation_context
source IB/ARTIFACTS/PRE_REBUTTAL/sec35/jobs/cluster_lib.sh; cluster orcd
SR=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/sec35/runs; C=IB/TMP/PRE_REBUTTAL/e4_feed; mkdir -p $C/state
SB="cd $W && sbatch --export=NIL --time=24:00:00 --partition=mit_preemptable --requeue --gres=gpu:rtx_pro_6000:1 jobs/rq_e4_read_orcd.sh"
say() { echo "$(date -u +%FT%TZ) $*"; }
while true; do
  N=T_res_rq9e4fr_20I_KL_u05600; RUN=T_res_rq9e4fr_20I_KL; DST=$W/rq_eval_e4x/$N/$N
  if [ ! -f $C/state/$N.sent ] && aws s3 ls $SR/$RUN/DONE > /dev/null 2>&1; then
    say "$N: fetch"; rm -rf $C/$N
    if aws s3 sync --only-show-errors $SR/$RUN/sync/ckpt/final/ $C/$N/ && [ -f $C/$N/meta.json ] && [ -f $C/$N/ac/ac_modules.pt ]; then
      remote 60 "mkdir -p $W/rq_eval_e4x/$N" && xfer -a --no-owner --no-group $C/$N/ $H:$DST.tmp/ && remote 60 "mv $DST.tmp $DST" \
        && remote 60 "$SB $N" && touch $C/state/$N.sent && say "$N: sent and read submitted"
    else say "$N: incomplete download"; fi
    rm -rf $C/$N
  fi
  N=T_res_rq9e4_20I_KL_u05600; RUN=T_res_rq9e4_20I_KL; DST=$W/rq_eval_e4x/$N/$N
  if [ ! -f $C/state/$N.sent ]; then
    out=$(remote 120 "F=$W/runs/$RUN/sync/ckpt/final; if [ -f $W/runs/$RUN/DONE ] && [ -f \$F/meta.json ]; then mkdir -p $W/rq_eval_e4x/$N
      [ -d $DST ] || { mkdir -p $DST.tmp && cp -al \$F/ac \$F/reader \$F/meta.json $DST.tmp/ && mv $DST.tmp $DST; } && $SB $N; fi" 2>&1)
    echo "$out" | grep -q "Submitted batch job" && { touch $C/state/$N.sent; say "$N: linked and read submitted ($out)"; }
    [ -n "$out" ] && ! echo "$out" | grep -q "Submitted" && say "$N: $out"
  fi
  [ -f $C/state/T_res_rq9e4fr_20I_KL_u05600.sent ] && [ -f $C/state/T_res_rq9e4_20I_KL_u05600.sent ] && { say "both 4B reads submitted"; exit 0; }
  sleep 900
done
