#!/bin/bash
# Retrieval AC arm (pre-rebuttal 2026-11-10) on one SkyPilot spot node, as a sky job. Default: R_res_rq9 (2B->9B fresh pair, least-squares
# row head, FineWeb reconstruction, 2,000 updates) then T_res_rq9 (task from R's final pair: 1,000 updates x 8, k in {20,100,200}, reader
# LoRA 2e-5 + no-context penalty, encoder 5e-5; dev200 at k=100, cap 1024 like the text probe). Each run folder is mirrored on S3.
#   .venv/bin/python -m activation.cloud.sky exec NODE -- [env RQ_GEN=generated_rq9fr_rtx RQ_TAG=_rq9fr] bash /root/rq_ac_node.sh [R,T | S | T]
# A T run starts from R_res_rq9's final pair: linked locally when R ran on this node, else fetched from S3 (runs/R_res_rq9/sync/ckpt/final).
set -uo pipefail
W=/root/sec35; G=$W/jobs/${RQ_GEN:-generated_rq9_rtx}; TAG=${RQ_TAG:-_rq9}; S3PY=/root/s3venv/bin/python
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/sec35
bash /root/ec2_prep.sh || { echo "prep failed"; exit 2; }
export SEC45_EVAL_MAX_NEW=1024
for r in $(echo ${1:-R,T} | tr , ' '); do
  if [ $r = T ]; then
    I=$W/init/R_res_rq9_final; export SEC45_INIT_PAIR=$I
    if [ ! -f $I/DONE ] && [ ! -f $W/runs/R_res_rq9/DONE ]; then
      rm -rf $I; mkdir -p $I && $S3PY $W/s3x.py getdir $S3/runs/R_res_rq9/sync/ckpt/final/ $I/ && $S3PY $W/s3x.py get $S3/runs/R_res_rq9/DONE $I/DONE || { echo "R_res_rq9's final pair is not on S3 yet"; exit 2; }
    fi
  fi
  bash $W/jobs/ec2_run.sh ${r}_res$TAG bash $G/${r}_res$TAG.sh || { echo "${r}_res$TAG failed"; exit 1; }
done
