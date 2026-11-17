#!/bin/bash
# Host side: run cmp_stage1_ec2.sh on a SkyPilot node as a sky job (blocks while it runs: start it with nohup ... &).
#   ec2_launch.sh NODE RUN PACKET [NAME=VALUE ...]     NAME=VALUE: cmp_stage1_job.sh options (CMP_INIT, CMP_PANEL, SMOKE, ...)
# The node fetches the job scripts from S3 (IB/ is not in the ~/sky_workdir mirror): compaction/stage1/ec2_jobs/.
# Create a node first: .venv/bin/python -m activation.cloud.sky setup --name NODE --gpu rtx6000pro --spot   (30-min idle autodown)
set -euo pipefail
NODE=$1; RUN=$2; PACKET=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd $HERE/../../../../../.. && pwd)
K=activation-context-data/pre_rebuttal_20261109/compaction/stage1/ec2_jobs
aws s3 cp --quiet $HERE/cmp_stage1_ec2.sh s3://general-bucket-aln/$K/cmp_stage1_ec2.sh
aws s3 cp --quiet $HERE/../cmp_stage1_job.sh s3://general-bucket-aln/$K/cmp_stage1_job.sh
aws s3 cp --quiet $ROOT/IB/ARTIFACTS/PRE_REBUTTAL/sec35/kit/s3x.py s3://general-bucket-aln/$K/s3x.py
aws s3 cp --quiet $HERE/../../agent_eval/teacher_view_compare.py s3://general-bucket-aln/$K/teacher_view_compare.py
VARS=""; for v in "$@"; do VARS="$VARS $(printf '%q' "$v")"; done
BOOT="[ -x /root/s3venv/bin/python ] || { /usr/local/bin/uv venv -q /root/s3venv && /usr/local/bin/uv pip install -q --python /root/s3venv/bin/python boto3; }; \
mkdir -p /root/cmp1/jobs && /root/s3venv/bin/python -c \"import boto3; c = boto3.client('s3'); \
[c.download_file('general-bucket-aln', '$K/' + f, '/root/cmp1/jobs/' + f) for f in ('cmp_stage1_ec2.sh', 'cmp_stage1_job.sh', 's3x.py', 'teacher_view_compare.py')]\" \
&& env$VARS bash /root/cmp1/jobs/cmp_stage1_ec2.sh $RUN $PACKET"
cd $ROOT && exec .venv/bin/python -m activation.cloud.sky exec $NODE -- bash -c "$BOOT"
