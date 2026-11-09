#!/bin/bash
# Uncapped-summary compaction baseline (pre-rebuttal 2026-11-09) on one SkyPilot spot node, resumable across reclaims:
# inputs and the rollout cache come from S3, the cache goes back every 10 min (run_resume.py skips rollouts already cached).
# The paper's command (kit/resume_node.sh, seed-3 log) with --uncapped-summary and the summary variants only, seeds 2-4.
#   on the node:  nohup setsid bash cmp_node.sh > /root/cmp_node.out 2>&1 < /dev/null &
set -uo pipefail
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction
A=/root/activation_artifacts; PY=/opt/activation/.venv/bin/python
mkdir -p $A/pc/logs $A/SYNC/ROLLOUTS
(cd /root/sky_workdir && UV_PROJECT_ENVIRONMENT=/opt/activation/.venv /usr/local/bin/uv sync --frozen -q)
[ -x /root/s3venv/bin/python ] || { /usr/local/bin/uv venv -q /root/s3venv && /usr/local/bin/uv pip install -q --python /root/s3venv/bin/python boto3; }   # boto3 is not in the project lock
S3PY=/root/s3venv/bin/python
$S3PY -c "import boto3; boto3.client('s3').download_file('general-bucket-aln', 'activation-context-data/pre_rebuttal_20261109/compaction/s3x.py', '$A/s3x.py')"
X() { S3X_ALLOW_OPTIMIZER=1 $S3PY $A/s3x.py "$@"; }
X getdir $S3/pc_inputs/ $A/pc/
if [ ! -d $A/SYNC/REPO_CACHE ]; then X get $S3/repo_cache.tar.gz /root/rc.tgz && tar -xzf /root/rc.tgz -C $A/SYNC && rm /root/rc.tgz; fi
X getdir $S3/ROLLOUTS/ $A/SYNC/ROLLOUTS/ || true
up() { X putdir $A/SYNC/ROLLOUTS $S3/ROLLOUTS/ > /dev/null 2>&1; X putdir $A/pc/logs $S3/logs/ > /dev/null 2>&1; }
( while sleep 600; do up; done ) & LOOP=$!
for s in 2 3 4; do
  bash $A/pc/kit/resume_node.sh prefixes_eval9c_200 --variants summary_write,summary --uncapped-summary --budget 50000 \
    --trajectory-cap 90000 --max-model-len 100000 --max-duration 6000 --seed $s --tag pc_r9d50u_s$s --agents-per-gpu 40 \
    >> $A/pc/logs/pc_r9d50u_s$s.log 2>&1
  echo "$(date -u +%FT%TZ) seed $s exit $?" >> $A/pc/logs/uncapped_progress.log
  up
done
kill $LOOP; up; echo "$(date -u +%FT%TZ) all seeds done" >> $A/pc/logs/uncapped_progress.log; up
