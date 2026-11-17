#!/bin/bash
# Host side: keep one stage-1 run going on a SkyPilot spot node until it has finished, across spot reclaims (start with nohup ... &).
#   ec2_keep.sh NODE RUN PACKET [NAME=VALUE ...]      (arguments as ec2_launch.sh)
# Loop: create NODE when it does not exist (sky setup, g7e spot), run ec2_launch.sh (blocks; the node restores runs/RUN from S3 and
# resumes from the recovery slot), stop when S3 holds runs/RUN/proxy/final.json (the job's last step) or after 12 attempts.
set -uo pipefail
NODE=$1; RUN=$2; PACKET=$3
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd $HERE/../../../../../.. && pwd)
S3R=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/stage1/runs/$RUN
say() { echo "$(date -u +%FT%TZ) [keep $RUN] $*"; }
cd $ROOT
for attempt in $(seq 1 12); do
  aws s3 ls $S3R/proxy/final.json > /dev/null 2>&1 && { say "finished (final proxy on S3)"; exit 0; }
  if ! .venv/bin/python -m activation.cloud.sky status $NODE 2>&1 | grep -q "^$NODE .* UP "; then
    say "attempt $attempt: node $NODE not up: creating it"
    .venv/bin/python -m activation.cloud.sky setup --name $NODE --gpu rtx6000pro --spot || { say "setup failed"; sleep 300; continue; }
  fi
  say "attempt $attempt: launching on $NODE"
  bash $HERE/ec2_launch.sh "$@"; say "launch returned $?"
  sleep 60
done
say "gave up after 12 attempts"; exit 1
