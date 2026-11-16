#!/bin/bash
# Math teacher test: one generation job on a SkyPilot EC2 node (fetched from S3 jobs/ by the exec bootstrap, then run there):
#   bash /root/mteach/jobs/node_job.sh OUTSUB MODEL ARMS [LIMIT]
#   e.g. bash node_job.sh out Qwen/Qwen3.5-9B 9b_full:full,9b_results:results
# Before: the env is synced to the lock, and OUT is restored from S3 (a reclaimed node's last upload), so teacher_gen.py resumes.
# During: OUT goes to S3 every 5 minutes; once more at the end.
set -uo pipefail
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/math/teacher_test
W=/root/mteach; PY=/opt/activation/.venv/bin/python; S3PY=/root/s3venv/bin/python
OUTSUB=$1; MODEL=$2; ARMS=$3; LIMIT=${4:-}
OUT=$W/$OUTSUB; mkdir -p $OUT $W/data
stamp() { echo "$(date -u +%FT%TZ) node_job: $*"; }
(cd /root/sky_workdir && UV_PROJECT_ENVIRONMENT=/opt/activation/.venv /usr/local/bin/uv sync --frozen -q) || { stamp "uv sync failed"; exit 1; }
$S3PY $W/jobs/tt_s3.py get $S3/data/ $W/data/
(cd $W/data && sha256sum -c prompts.jsonl.sha256) || { stamp "prompts checksum failed"; exit 1; }
$S3PY $W/jobs/tt_s3.py get $S3/$OUTSUB/ $OUT/ || true
up() { $S3PY $W/jobs/tt_s3.py put $OUT $S3/$OUTSUB/ > /dev/null 2>&1; }
( while sleep 300; do up; done ) & LOOP=$!
stamp "model $MODEL arms $ARMS limit ${LIMIT:-none} -> $OUT"
# 2026-11-16: the 27B NVFP4 JIT-compiles CUDA kernels at start; 9+ parallel cicc processes (~5.4 GB each) filled the 62 GB host and
# Ray killed the job. MAX_JOBS / FLASHINFER_NVCC_THREADS cap the compile parallelism.
env -u HF_TOKEN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MAX_JOBS=${MAX_JOBS:-3} FLASHINFER_NVCC_THREADS=1 $PY $W/jobs/teacher_gen.py --prompts $W/data/prompts.jsonl --out $OUT \
  --model "$MODEL" --arms "$ARMS" ${LIMIT:+--limit $LIMIT}
RC=$?
kill $LOOP 2>/dev/null; up
stamp "exit $RC (uploaded to $S3/$OUTSUB/)"
exit $RC
