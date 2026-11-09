#!/bin/bash
# Run one pre-rebuttal job on a SkyPilot EC2 (spot) node with its run folder mirrored on S3 (run on the node, after ec2_prep.sh):
#   bash /root/sec35/jobs/ec2_run.sh RUN <command ...>      e.g.  ... ec2_run.sh X_rq_text_ans env RUN=X_rq_text_ans STEPS=... bash /root/sec35/jobs/text_lora_job.sh
# Before: runs/RUN is restored from S3 (a reclaimed node's last upload, recovery slot included), so the job script resumes.
# During: every 15 min the folder goes to S3 INCLUDING optimizer.pt (S3X_ALLOW_OPTIMIZER=1: the slot is what a new node resumes from).
set -uo pipefail
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/sec35
W=/root/sec35; PY=/opt/activation/.venv/bin/python; RUN=$1; shift
D=$W/runs/$RUN; mkdir -p $D; S3PY=/root/s3venv/bin/python   # made by ec2_prep.sh
export S3X_ALLOW_OPTIMIZER=1 SEC45_WORK=$W SEC45_PY=$PY SEC45_HF_HOME=${HF_HOME:-/root/.cache/huggingface} SEC45_USE_S3=0
$S3PY $W/s3x.py getdir $S3/runs/$RUN/ $D/ || true
up() { $S3PY $W/s3x.py putdir $D $S3/runs/$RUN/ --exclude teacher_snapshots AUTOTUNE > /dev/null 2>&1; }
( while sleep 900; do up; done ) & LOOP=$!
"$@"; RC=$?
kill $LOOP 2>/dev/null; up; echo "$(date -u +%FT%TZ) ec2_run $RUN exit $RC (uploaded to $S3/runs/$RUN/)"
exit $RC
