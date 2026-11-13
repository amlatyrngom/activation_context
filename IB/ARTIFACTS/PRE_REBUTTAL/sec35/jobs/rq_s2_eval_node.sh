#!/bin/bash
# EC2 stand-in for the ORCD stage-2 side scorer (2026-11-13: the ORCD connection is down until the user logs in again). Every 10 min, each
# complete ckpt/step_* pair of the EC2 stage-2 runs on S3 (the four big files present, sizes unchanged since the previous poll) is fetched
# to $R/<RUN>_uNNNNN and scored on PLAIN BM25 dev1000 (hotpot_tasks_kI_v2) at k=100 and k=200 by jobs/rq_eval_job.sh (sec45_eval.py,
# greedy, cap 1024, batch 32, code v5 = the ORCD side scorer's settings); also the final pairs named in EXTRA (k = 100, 200, 20).
# Outputs mirrored to S3 runs/s2_eval_ec2/{out,logs}/ after every pass. Pairs already scored on ORCD are listed in SKIP.
# Stops when S3 runs/s2_eval_ec2/STOP exists. Run as a sky job (scp ec2_prep.sh + this to /root first):
#   .venv/bin/python -m activation.cloud.sky exec NODE -- bash /root/rq_s2_eval_node.sh
set -uo pipefail
W=/root/sec35; J=$W/jobs; S3PY=/root/s3venv/bin/python; R=$W/rq_eval_s2e; mkdir -p $R/state
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/sec35
RUNS="T_res_rq9L_kI T_res_rq9L_kI_KL T_res_rq9fr_kI_KL"
SKIP=" T_res_rq9L_kI_u00400 T_res_rq9L_kI_u00800 T_res_rq9L_kI_u01200 "
EXTRA="T_res_rq9L_QA20I_KL"
bash /root/ec2_prep.sh sec35_code_v5.tar.gz > /dev/null || { echo "prep failed"; exit 2; }
X() { $S3PY $W/s3x.py "$@"; }
export SEC45_PY=/opt/activation/.venv/bin/python
X getdir $S3/data/hotpot_tasks_kI_v2/ $W/data/hotpot_tasks_kI_v2/ || exit 2
for N in $EXTRA; do P=$R/${N}_u05600; [ -f $P/meta.json ] && continue
  rm -rf $P; X getdir $S3/runs/$N/sync/ckpt/final/ $P/ && echo "100 200 20" > $P/KS || rm -rf $P; done
ls_run() {   # "size key" of the four big files of every ckpt/step_* of run $1
  $S3PY - "$1" <<'PY'
import sys, re, boto3
b, p = "general-bucket-aln", f"activation-context-data/pre_rebuttal_20261109/sec35/runs/{sys.argv[1]}/sync/ckpt/"
pat = re.compile(r"step_\d+/(ac/ac_modules\.pt|ac/side_lora/adapter_model\.safetensors|reader/adapter_model\.safetensors|meta\.json)$")
for page in boto3.client("s3").get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=p):
    for o in page.get("Contents", []):
        if pat.search(o["Key"]): print(o["Size"], o["Key"])
PY
}
while true; do
  for RUN in $RUNS; do
    ls_run $RUN > $R/state/$RUN.now 2>/dev/null || continue
    for st in $(grep -o 'step_[0-9]*' $R/state/$RUN.now | sort -u); do
      N=${RUN}_u${st#step_}; case "$SKIP" in *" $N "*) continue;; esac; [ -f $R/$N/meta.json ] && continue
      cur=$(grep "/$st/" $R/state/$RUN.now | sort); [ $(echo "$cur" | wc -l) = 4 ] || continue
      prev=$(grep "/$st/" $R/state/$RUN.prev 2>/dev/null | sort); [ "$cur" = "$prev" ] || continue
      rm -rf $R/$N.tmp; X getdir $S3/runs/$RUN/sync/ckpt/$st/ $R/$N.tmp/ && echo "100 200" > $R/$N.tmp/KS && mv $R/$N.tmp $R/$N && echo "$(date -u +%FT%TZ) staged $N"
    done
    mv $R/state/$RUN.now $R/state/$RUN.prev
  done
  RQ_EVAL_DIR=$R RQ_EVAL_DATA=hotpot_tasks_kI_v2 EVAL_SPLIT=dev1000 SEC45_BUNDLE=sec35_code_v5.tar.gz bash $J/rq_eval_job.sh || true
  X putdir $R/out/ $S3/runs/s2_eval_ec2/out/; X putdir $R/logs/ $S3/runs/s2_eval_ec2/logs/
  $S3PY - <<'PY' && { echo "STOP on S3: exiting"; exit 0; }
import sys, boto3
r = boto3.client("s3").list_objects_v2(Bucket="general-bucket-aln", Prefix="activation-context-data/pre_rebuttal_20261109/sec35/runs/s2_eval_ec2/STOP")
sys.exit(0 if r.get("KeyCount", 0) else 1)
PY
  sleep 600
done
