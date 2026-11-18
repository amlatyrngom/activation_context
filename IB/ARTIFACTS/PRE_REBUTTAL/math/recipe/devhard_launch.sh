#!/bin/bash
# Host side of the dev_hard solver runs (2026-11-18): stage the node script, kit/run_tqa_solve_v2.py, s3x.py and the frozen sample on
# S3, then run devhard_solve_node.sh on a Sky node as one sky job with 1 GPU (Sky queues it behind the node's running job, so the GPU
# is never shared). IB/ is not in the ~/sky_workdir mirror, so the node fetches these files from S3 first (as compaction's launch.sh).
#   devhard_launch.sh NODE ID EXPECTED_ROWS          e.g. devhard_launch.sh dvA tqa_math_dev_hard_v1 540
set -euo pipefail
NODE=$1; ID=$2; EXPECTED=$3
cd "$(dirname "$0")"
ROOT=$(git rev-parse --show-toplevel)
K=activation-context-data/pre_rebuttal_20261109/math/dev_hard/code
SAMPLE=$ROOT/IB/TMP/PRE_REBUTTAL/math_recipe/dev_hard/$ID/sample.json
sha256sum < $SAMPLE | cut -c1-64 > /tmp/devhard_sample.sha256
aws s3 cp --quiet devhard_solve_node.sh s3://general-bucket-aln/$K/devhard_solve_node.sh
aws s3 cp --quiet $ROOT/IB/TMP/PAPER_TRAJ_QA/kit/run_tqa_solve_v2.py s3://general-bucket-aln/$K/run_tqa_solve_v2.py
aws s3 cp --quiet $ROOT/IB/ARTIFACTS/PRE_REBUTTAL/sec35/kit/s3x.py s3://general-bucket-aln/$K/s3x.py
aws s3 cp --quiet $SAMPLE s3://general-bucket-aln/$K/sample.json
aws s3 cp --quiet /tmp/devhard_sample.sha256 s3://general-bucket-aln/$K/sample.json.sha256
BOOT="[ -x /root/s3venv/bin/python ] || { /usr/local/bin/uv venv -q /root/s3venv && /usr/local/bin/uv pip install -q --python /root/s3venv/bin/python boto3; }; \
mkdir -p /root/activation_artifacts/math_devhard/code && /root/s3venv/bin/python -c \"import boto3; c = boto3.client('s3'); \
[c.download_file('general-bucket-aln', '$K/' + f, '/root/activation_artifacts/math_devhard/code/' + f) for f in ('devhard_solve_node.sh', 'run_tqa_solve_v2.py', 's3x.py', 'sample.json', 'sample.json.sha256')]\" \
&& bash /root/activation_artifacts/math_devhard/code/devhard_solve_node.sh $ID $EXPECTED"
cd $ROOT && exec .venv/bin/python -m activation.cloud.sky exec $NODE -- bash -c "$BOOT"
