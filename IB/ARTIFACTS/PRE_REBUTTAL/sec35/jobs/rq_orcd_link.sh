#!/bin/bash
# Host loop (2026-11-13): hands the checkpoints of stage-2 runs that train ON ORCD to the ORCD scorers, without touching the running loops:
# every 20 min, each new $W/runs/<RUN>/sync/ckpt/step_NNNNN is hard-linked into $W/rq_eval_s2/<RUN>_uNNNNN (KS "100 200": the side scorer,
# plain BM25), its step_01200 also into $W/rq_eval_fin (100I / 200I / 20I: jobs/rq_fin_loop.sh), and the final pair (DONE) into
# $W/rq_eval_fin/<RUN>_u02400. Exits when every run is DONE and linked.
#   nohup bash IB/ARTIFACTS/PRE_REBUTTAL/sec35/jobs/rq_orcd_link.sh >> IB/TMP/PRE_REBUTTAL/orcd_link.log 2>&1 &
set -uo pipefail
cd /home/ubuntu/activation_context
source IB/ARTIFACTS/PRE_REBUTTAL/sec35/jobs/cluster_lib.sh; cluster orcd
RUNS=${LINK_RUNS:-"T_res_rq9L_kI_KL_lr4"}
while true; do
  out=$(remote 120 "for RUN in $RUNS; do D=$W/runs/\$RUN/sync/ckpt
    for Q in \$D/step_[0-9][0-9][0-9][0-9][0-9]; do [ -f \$Q/meta.json ] || continue; U=\${Q##*_}; P=$W/rq_eval_s2/\${RUN}_u\$U
      [ -d \$P ] || { cp -al \$Q \$P.tmp && rm -f \$P.tmp/KS && echo '100 200' > \$P.tmp/KS && mv \$P.tmp \$P && echo linked \$P; }
      if [ \$U = 01200 ] && [ ! -d $W/rq_eval_fin/\${RUN}_u01200 ]; then F=$W/rq_eval_fin/\${RUN}_u01200; mkdir -p \$F.tmp && cp -al \$Q/ac \$Q/reader \$Q/meta.json \$F.tmp/ && mv \$F.tmp \$F && echo linked \$F; fi
    done
    F=$W/rq_eval_fin/\${RUN}_u02400
    if [ -f $W/runs/\$RUN/DONE ] && [ -f \$D/final/meta.json ]; then
      [ -d \$F ] || { mkdir -p \$F.tmp && cp -al \$D/final/ac \$D/final/reader \$D/final/meta.json \$F.tmp/ && mv \$F.tmp \$F && echo linked \$F; }
      echo DONE_LINKED \$RUN; fi
  done" 2>&1)
  [ -n "$out" ] && echo "$out" | sed "s/^/$(date -u +%FT%TZ) /"
  [ $(echo "$out" | grep -c DONE_LINKED) -ge $(echo $RUNS | wc -w) ] && { echo "all runs done and linked"; exit 0; }
  sleep 1200
done
