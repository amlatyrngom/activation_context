#!/bin/bash
# Math teacher test: run one generation job on the SkyPilot node (from this host, repo root):
#   bash IB/ARTIFACTS/PRE_REBUTTAL/math/teacher_test/launch.sh CLUSTER OUTSUB MODEL ARMS [LIMIT]
# The node fetches jobs/ (tt_s3.py, teacher_gen.py, node_job.sh) from S3 (IB/ is not in the SkyPilot mirror) and runs node_job.sh.
# Upload the job files first:  for f in tt_s3.py teacher_gen.py node_job.sh; do aws s3 cp IB/ARTIFACTS/PRE_REBUTTAL/math/teacher_test/$f $S3/jobs/$f; done
set -euo pipefail
CLUSTER=$1; OUTSUB=$2; MODEL=$3; ARMS=$4; LIMIT=${5:-}
BOOT='set -e
[ -x /root/s3venv/bin/python ] || { /usr/local/bin/uv venv -q /root/s3venv && /usr/local/bin/uv pip install -q --python /root/s3venv/bin/python boto3; }
mkdir -p /root/mteach/jobs
/root/s3venv/bin/python -c "
import boto3
c = boto3.client(\"s3\")
for f in (\"tt_s3.py\", \"teacher_gen.py\", \"node_job.sh\"):
    c.download_file(\"general-bucket-aln\", \"activation-context-data/pre_rebuttal_20261109/math/teacher_test/jobs/\" + f, \"/root/mteach/jobs/\" + f)
"
exec bash /root/mteach/jobs/node_job.sh "$@"'
exec .venv/bin/python -m activation.cloud.sky exec "$CLUSTER" -- bash -c "$BOOT" boot "$OUTSUB" "$MODEL" "$ARMS" ${LIMIT:+"$LIMIT"}
