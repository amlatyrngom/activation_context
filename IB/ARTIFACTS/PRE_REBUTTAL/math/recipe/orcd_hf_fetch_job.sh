#!/bin/bash
#SBATCH --job-name=mhffetch
#SBATCH --cpus-per-task=4 --mem=16G --time=03:00:00
# Download HF model repos into the ORCD HF cache that the math jobs read offline (2026-11-19: the 27B NVFP4 for the dev_hard
# teacher-test arm). Anonymous (no token). Then the scratch total (quota guard: keep it under ~900 GB).
#   sbatch ... jobs/orcd_hf_fetch_job.sh REPO_ID [REPO_ID ...]
export PATH=/usr/local/bin:/usr/bin:/bin
S=/home/ngom/orcd/scratch; PY=$S/activation/rq_v1/code/.venv/bin/python
export HF_HOME=$S/activation/hf HF_HUB_DISABLE_TELEMETRY=1; unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_OFFLINE
date -u
for m in "$@"; do
  for t in 1 2 3; do timeout 7200 $PY -c "from huggingface_hub import snapshot_download as s; print(s('$m', token=False))" && break; echo "retry $t"; done
  du -sh $HF_HOME/hub/models--${m//\//--} 2>/dev/null
done
echo "== scratch total"; du -sh $S/ 2>/dev/null; date -u
