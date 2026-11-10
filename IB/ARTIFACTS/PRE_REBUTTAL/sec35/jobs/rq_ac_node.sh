#!/bin/bash
# Retrieval AC arm (pre-rebuttal 2026-11-10) on one SkyPilot spot node, as a sky job: R_res_rq9 (2B->9B fresh pair, least-squares row
# head, FineWeb reconstruction, 2,000 updates) then T_res_rq9 (task from R's final pair: 1,000 updates x 8, k in {20,100,200}, reader
# LoRA 2e-5 + no-context penalty, encoder 5e-5; dev200 at k=100, cap 1024 like the text probe). Each run folder is mirrored on S3.
#   .venv/bin/python -m activation.cloud.sky exec NODE -- bash /root/rq_ac_node.sh [R,T | S]     (scp ec2_prep.sh + this to /root first)
set -uo pipefail
W=/root/sec35; G=$W/jobs/generated_rq9_rtx
bash /root/ec2_prep.sh || { echo "prep failed"; exit 2; }
export SEC45_EVAL_MAX_NEW=1024
for r in $(echo ${1:-R,T} | tr , ' '); do
  bash $W/jobs/ec2_run.sh ${r}_res_rq9 bash $G/${r}_res_rq9.sh || { echo "${r}_res_rq9 failed"; exit 1; }
done
