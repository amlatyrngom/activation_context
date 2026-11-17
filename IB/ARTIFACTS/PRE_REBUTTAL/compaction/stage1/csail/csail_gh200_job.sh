#!/bin/bash
#SBATCH --job-name=cmp_gh200
#SBATCH --cpus-per-task=16 --mem=200G --time=4-00:00:00
# A compaction stage-1 run on a CSAIL GH200 (aarch64), 2026-11-17. CSAIL incident rule: the job's first step builds a FRESH aarch64
# Python environment from the code bundle's uv.lock, with the aarch64 uv uploaded from the host (astral-sh/uv 0.12.5 release, its
# published sha256 checked on the host; the hash below is checked again here), a new uv cache and a new uv-managed CPython 3.14, all
# under $W/gh200/RUN/ (per run: two copies never share a build). Then cmp_stage1_job.sh trains with that Python (CMP_PY), the shared HF
# cache, inputs and bundle as the H200 copies. A requeued job reuses its venv only when nothing under it changed (inode change time)
# after its BUILT marker; otherwise the venv is moved aside and built again.
# Submit through csail_stage.sh sbatch (pre-job check): --export=NIL,CMP_...=... --account=quanta --partition=quanta-gh200
#   --qos=quanta-main --gpus=1 --requeue jobs/csail_gh200_job.sh RUN PACKET
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
W=/data/scratch-fast/ngom/cmp_csail; RUN=${1:?RUN}; PACKET=${2:?PACKET}; BUNDLE=${CMP_BUNDLE:?CMP_BUNDLE}; C=$W/code/$BUNDLE
UV=$W/tools_gh200/uv; UV_SHA=92804e2f635c1791bb497437d94b15970f0d6d74811979315624cbb0f45b778d
G=$W/gh200/$RUN; V=$G/venv
say() { echo "$(date -u +%FT%TZ) [gh200 $RUN] $*"; }
[ "$(uname -m)" = aarch64 ] || { say "not aarch64 ($(uname -m))"; exit 2; }
[ "$(sha256sum $UV | cut -d' ' -f1)" = $UV_SHA ] || { say "uv sha256 mismatch"; exit 2; }
export HOME=$W/home UV_CACHE_DIR=$G/uv_cache UV_PYTHON_INSTALL_DIR=$G/uv_python UV_PROJECT_ENVIRONMENT=$V UV_NO_CONFIG=1 UV_LINK_MODE=copy \
  UV_PYTHON_PREFERENCE=only-managed XDG_CACHE_HOME=$W/home/.cache XDG_CONFIG_HOME=$W/home/.config TZ=UTC PYTHONDONTWRITEBYTECODE=1
mkdir -p $G $HOME
if [ -f $V/BUILT ] && [ -n "$(find $V $G/uv_python -cnewer $V/BUILT ! -path $V/BUILT 2>/dev/null | head -1)" ]; then
  say "venv changed after BUILT: moved aside"; mv $V $V.changed_$(date +%s); fi
if [ ! -f $V/BUILT ]; then
  [ -e $V ] && mv $V $V.partial_$(date +%s)
  say "building a fresh aarch64 venv from $C/uv.lock with uv $($UV --version)"
  $UV python install 3.14 && (cd $C && $UV sync --frozen --no-dev --no-install-project --compile-bytecode --project $C) || { say "venv build failed"; exit 3; }
  $V/bin/python -c "import torch, transformers, peft, fla; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), 'transformers', transformers.__version__, 'peft', peft.__version__)" || { say "venv check failed"; exit 3; }
  date -u +%FT%TZ > $V/BUILT
fi
say "venv $V (built $(cat $V/BUILT)); training"
CMP_PY=$V/bin/python exec bash $W/jobs/cmp_stage1_job.sh $RUN $PACKET
