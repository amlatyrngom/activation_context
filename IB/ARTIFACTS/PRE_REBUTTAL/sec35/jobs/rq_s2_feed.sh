#!/bin/bash
# Host feeder for the stage-2 side scorer (jobs/rq_s2_eval_loop.sh on ORCD): every 15 min, each complete ckpt/step_* pair of the EC2 stage-2
# runs on S3 (all three big files present, sizes unchanged since the previous poll) is fetched here, sent to ORCD as
# $W/rq_eval_s2/<RUN>_uNNNNN.tmp, renamed when complete, and deleted here. Run on this host:  bash jobs/rq_s2_feed.sh
set -uo pipefail
cd /home/ubuntu/activation_context
source IB/ARTIFACTS/PRE_REBUTTAL/sec35/jobs/cluster_lib.sh; cluster orcd
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/sec35; C=IB/TMP/PRE_REBUTTAL/s2_feed; mkdir -p $C/state
RUNS=${S2_RUNS:-"T_res_rq9L_kI T_res_rq9L_kI_KL T_res_rq9fr_kI_KL"}
while true; do
  for RUN in $RUNS; do
    aws s3 ls --recursive $S3/runs/$RUN/sync/ckpt/ 2>/dev/null | awk '$4 ~ /step_[0-9]+\/(ac\/ac_modules.pt|ac\/side_lora\/adapter_model.safetensors|reader\/adapter_model.safetensors|meta.json)$/ {print $3, $4}' > $C/state/$RUN.now
    for st in $(grep -o 'step_[0-9]*' $C/state/$RUN.now | sort -u); do
      N=${RUN}_u${st#step_}; [ -f $C/state/$N.sent ] && continue
      cur=$(grep "/$st/" $C/state/$RUN.now | sort); [ $(echo "$cur" | wc -l) = 4 ] || continue
      prev=$(grep "/$st/" $C/state/$RUN.prev 2>/dev/null | sort); [ "$cur" = "$prev" ] || continue     # unchanged since the last poll: upload finished
      echo "$(date -u +%FT%TZ) $N: fetch"; rm -rf $C/$N
      aws s3 sync --only-show-errors $S3/runs/$RUN/sync/ckpt/$st/ $C/$N/ || continue
      xfer -a --no-owner --no-group $C/$N/ $H:$W/rq_eval_s2/$N.tmp/ && remote 60 "echo '100 200' > $W/rq_eval_s2/$N.tmp/KS && mv $W/rq_eval_s2/$N.tmp $W/rq_eval_s2/$N" \
        && touch $C/state/$N.sent && echo "$(date -u +%FT%TZ) $N: sent"
      rm -rf $C/$N
    done
    mv $C/state/$RUN.now $C/state/$RUN.prev
  done
  sleep 900
done
