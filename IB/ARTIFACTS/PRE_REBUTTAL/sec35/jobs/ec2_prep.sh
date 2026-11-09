#!/bin/bash
# One-time preparation of a SkyPilot EC2 node for the pre-rebuttal runs (run on the node):  bash ec2_prep.sh [BUNDLE]
#   python env synced to the lock (the image can be older), code bundle + task data + job scripts from S3 into W=/root/sec35,
#   reader / encoder weights into the HF cache. Then: bash /root/sec35/jobs/ec2_run.sh RUN <job command ...>
set -euo pipefail
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/sec35
W=/root/sec35; PY=/opt/activation/.venv/bin/python; BUNDLE=${1:-sec35_code_v3.tar.gz}
mkdir -p $W/code $W/data $W/jobs $W/runs
(cd /root/sky_workdir && UV_PROJECT_ENVIRONMENT=/opt/activation/.venv /usr/local/bin/uv sync --frozen -q)
X() { $PY $W/s3x.py "$@"; }
[ -f $W/s3x.py ] || $PY - <<PYEOF
import boto3; boto3.client("s3").download_file("general-bucket-aln", "activation-context-data/pre_rebuttal_20261109/sec35/jobs/s3x.py", "$W/s3x.py")
PYEOF
X getdir $S3/jobs/ $W/jobs/
for f in $BUNDLE $BUNDLE.sha256; do X get $S3/code/$f $W/code/$f; done
(cd $W/code && sha256sum -c $BUNDLE.sha256)
C=$W/code/${BUNDLE%.tar.gz}; [ -f $C/MANIFEST.json ] || { mkdir -p $C.tmp && tar -xzf $W/code/$BUNDLE -C $C.tmp && mv -T $C.tmp $C; }
for d in rq_tasks_v1 hotpot_tasks_v2 fineweb_recon_v1; do X getdir $S3/data/$d/ $W/data/$d/; done
for m in Qwen/Qwen3.5-9B Qwen/Qwen3.5-2B; do
  for t in 1 2 3; do timeout 1800 env -u HF_TOKEN $PY -c "from huggingface_hub import snapshot_download as s; s('$m', token=False)" > /dev/null && break; done
done
echo "ec2_prep: ready ($C)"
