#!/bin/bash
# Host side: stage this folder (+ the sec35 s3x.py) on S3 and run dev75_node.sh on a sky node as a sky job (IB/ is excluded from the
# ~/sky_workdir mirror by .skyignore, so the node fetches the scripts from S3 first).
#   [RUN_TIMEOUT=s] launch.sh NODE dev75_node.sh-arguments...        e.g. launch.sh cdv1 check S1_lcrec100_kl_both_c20 4000 2
set -euo pipefail
NODE=$1; shift
cd "$(dirname "$0")"
ROOT=$(git rev-parse --show-toplevel)
K=activation-context-data/pre_rebuttal_20261109/compaction/agent_eval_dev75/code
for f in dev75_node.sh run_resume_res.py first_turn_check.py; do aws s3 cp --quiet $f s3://general-bucket-aln/$K/$f; done
aws s3 cp --quiet $ROOT/IB/ARTIFACTS/PRE_REBUTTAL/sec35/kit/s3x.py s3://general-bucket-aln/$K/s3x.py
BOOT="[ -x /root/s3venv/bin/python ] || { /usr/local/bin/uv venv -q /root/s3venv && /usr/local/bin/uv pip install -q --python /root/s3venv/bin/python boto3; }; \
mkdir -p /root/activation_artifacts/agent_eval/code && /root/s3venv/bin/python -c \"import boto3; c = boto3.client('s3'); \
[c.download_file('general-bucket-aln', '$K/' + f, '/root/activation_artifacts/agent_eval/code/' + f) for f in ('dev75_node.sh', 'run_resume_res.py', 'first_turn_check.py', 's3x.py')]\" \
&& RUN_TIMEOUT=${RUN_TIMEOUT:-0} bash /root/activation_artifacts/agent_eval/code/dev75_node.sh $*"
cd $ROOT && exec .venv/bin/python -m activation.cloud.sky exec $NODE -- bash -c "$BOOT"
