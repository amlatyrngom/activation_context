#!/bin/bash
# Host feeder for the stage-2 final reads (jobs/rq_fin_loop.sh on ORCD), 2026-11-13: every 15 min, each EC2 stage-2 run whose S3 DONE exists
# has its final pair (AC: sync/ckpt/final; text: ckpt/step_02400) fetched here, sent to ORCD as $W/rq_eval_fin/<RUN>_u02400 (text:
# $W/rq_eval_fin/text/X_rq9_text_t2_u02400) via a .tmp folder renamed when complete, and deleted here. Exits when all are sent.
# Run on this host:  nohup bash IB/ARTIFACTS/PRE_REBUTTAL/sec35/jobs/rq_fin_feed.sh >> IB/TMP/PRE_REBUTTAL/fin_feed.log 2>&1 &
set -uo pipefail
cd /home/ubuntu/activation_context
source IB/ARTIFACTS/PRE_REBUTTAL/sec35/jobs/cluster_lib.sh; cluster orcd
SR=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/sec35/runs; C=IB/TMP/PRE_REBUTTAL/fin_feed; mkdir -p $C/state
RUNS="T_res_rq9L_kI T_res_rq9L_kI_KL T_res_rq9fr_kI_KL X_rq9_text_t2"
while true; do
  left=0
  for RUN in $RUNS; do
    N=${RUN}_u02400; [ -f $C/state/$N.sent ] && continue; left=1
    aws s3 ls $SR/$RUN/DONE > /dev/null 2>&1 || continue
    echo "$(date -u +%FT%TZ) $N: fetch"; rm -rf $C/$N
    case $RUN in
      X_*) SRC=$SR/$RUN/ckpt/step_02400/; DST=$W/rq_eval_fin/text/$N;;
      *)   SRC=$SR/$RUN/sync/ckpt/final/;  DST=$W/rq_eval_fin/$N;;
    esac
    aws s3 sync --only-show-errors $SRC $C/$N/ || continue
    case $RUN in X_*) [ -f $C/$N/adapter_model.safetensors ];; *) [ -f $C/$N/meta.json ];; esac || { echo "$N: incomplete download"; rm -rf $C/$N; continue; }
    xfer -a --no-owner --no-group $C/$N/ $H:$DST.tmp/ && remote 60 "mv $DST.tmp $DST" && touch $C/state/$N.sent && echo "$(date -u +%FT%TZ) $N: sent"
    rm -rf $C/$N
  done
  [ $left = 0 ] && { echo "all finals sent"; exit 0; }
  sleep 900
done
