#!/bin/bash
#SBATCH --job-name=cmp_setup
#SBATCH --cpus-per-task=16 --mem=64G --time=3:00:00
#SBATCH --output=/data/scratch-fast/ngom/cmp_csail/logs/setup_%j.out
# CSAIL environment for compaction stage 1 (2026-11-17, after the CSAIL incident; CPU only, no GPU):
#   1. a fresh Python environment $W/venv built from the code bundle's uv.lock by the freshly uploaded uv ($W/tools/uv), with a new
#      uv cache and a new uv-managed CPython (nothing that was on CSAIL before is executed or imported);
#   2. a fresh Hugging Face cache $W/hf with the two models the pair uses (Qwen/Qwen3.5-9B reader, Qwen/Qwen3.5-2B encoder);
#   3. $W/venv/BUILT and $W/hf/BUILT: markers; csail_verify.sh flags any file in venv/hf/code/jobs changed (ctime) after them.
# Submit (host: csail_stage.sh setup): sbatch --export=NIL --account=quanta --partition=quanta-h200 --qos=quanta-main
#   --cpus-per-task=16 --mem=64G jobs/csail_setup_job.sh BUNDLE
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
W=/data/scratch-fast/ngom/cmp_csail; BUNDLE=${1:?BUNDLE}; C=$W/code/$BUNDLE
export HOME=$W/home UV_CACHE_DIR=$W/uv_cache UV_PYTHON_INSTALL_DIR=$W/uv_python UV_PROJECT_ENVIRONMENT=$W/venv UV_NO_CONFIG=1 \
  UV_LINK_MODE=copy UV_PYTHON_PREFERENCE=only-managed XDG_CACHE_HOME=$W/home/.cache XDG_CONFIG_HOME=$W/home/.config TZ=UTC \
  PYTHONDONTWRITEBYTECODE=1        # bytecode only from uv sync --compile-bytecode, before the BUILT marker (csail_verify: no change after it)
mkdir -p $HOME $W/logs
say() { echo "$(date -u +%FT%TZ) [setup] $*"; }
say "on $(hostname) job ${SLURM_JOB_ID:-none}; uv $($W/tools/uv --version); bundle $C"
[ -f $C/uv.lock ] && [ -f $C/pyproject.toml ] || { say "bundle missing"; exit 2; }
if [ ! -f $W/venv/BUILT ]; then
  [ -e $W/venv ] && { say "a venv without BUILT exists: moved aside"; mv $W/venv $W/venv.partial_$(date +%s); }
  $W/tools/uv python install 3.14
  (cd $C && $W/tools/uv sync --frozen --no-dev --no-install-project --compile-bytecode --project $C)
  $W/venv/bin/python -c "import torch, transformers, peft, fla; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'peft', peft.__version__)"
  date -u +%FT%TZ > $W/venv/BUILT
fi
say "venv: $(du -sh $W/venv | cut -f1)"
if [ ! -f $W/hf/BUILT ]; then
  HF_HOME=$W/hf HF_HUB_DISABLE_TELEMETRY=1 $W/venv/bin/python -c "
from huggingface_hub import snapshot_download
for m in ('Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-2B'):
    print(m, snapshot_download(m), flush=True)"
  date -u +%FT%TZ > $W/hf/BUILT
fi
say "hf: $(du -sh $W/hf | cut -f1)"
say "done"
