#!/bin/bash
# Node preparation for the vLLM Hidden Residuals validation (run as a sky job; this folder rsynced to /root/vllm_res first).
# Python env synced to the lock with the worktree's pyproject (the vLLM plugin entry point), sec35 code bundle (task builders),
# dev1000 tasks + corpus, the T_res_rq9fr_kI_KL final pair, reader / encoder weights.
set -euo pipefail
V=/root/vllm_res; mkdir -p $V/code $V/data $V/pair $V/out
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/sec35
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/opt/activation/.venv}   # the image env (sky jobs may not inherit it)
cd /root/sky_workdir && /usr/local/bin/uv sync --frozen -q
PY=$UV_PROJECT_ENVIRONMENT/bin/python; echo "python: $PY"
$PY - <<'EOF'
from importlib.metadata import entry_points
names = [e.value for e in entry_points(group="vllm.general_plugins")]
print("vllm.general_plugins:", names)
assert "activation.vllm_residuals:register" in names, "entry point missing (re-sync the project)"
EOF
[ -x /root/s3venv/bin/python ] || { /usr/local/bin/uv venv -q /root/s3venv && /usr/local/bin/uv pip install -q --python /root/s3venv/bin/python boto3; }
X() { /root/s3venv/bin/python $V/s3x.py "$@"; }
B=sec35_code_v5
[ -f $V/code/$B/MANIFEST.json ] || { X get $S3/code/$B.tar.gz $V/code/$B.tar.gz && mkdir -p $V/code/$B && tar -xzf $V/code/$B.tar.gz -C $V/code/$B; }
for f in dev1000.jsonl hotpotqa.corpus.jsonl.gz stats.json; do [ -f $V/data/$f ] || X get $S3/data/hotpot_tasks_kI_v2/$f $V/data/$f; done
[ -f $V/pair/reader/adapter_model.safetensors ] || X getdir $S3/runs/T_res_rq9fr_kI_KL/sync/ckpt/final/ $V/pair/
for m in Qwen/Qwen3.5-9B Qwen/Qwen3.5-2B; do
  for t in 1 2 3; do timeout 1800 env -u HF_TOKEN $PY -c "from huggingface_hub import snapshot_download as s; s('$m', token=False)" > /dev/null && break; done
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2; nproc
echo "prep: ready"
