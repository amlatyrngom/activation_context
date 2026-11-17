#!/bin/bash
#SBATCH --job-name=math_gh200
#SBATCH --cpus-per-task=16 --mem=200G --time=4-00:00:00
# A math-recipe step on a CSAIL GH200 (aarch64), 2026-11-17 (compaction/stage1/csail/csail_gh200_job.sh's pattern). CSAIL incident
# rule: the job first builds a FRESH aarch64 Python environment from the code bundle's uv.lock with the aarch64 uv uploaded from the
# host (sha256 checked here), a new uv cache and a new uv-managed CPython 3.14, under W/gh200/<TAG>/ (one per run). Then math_job.sh
# runs with that Python (MATH_PY). A requeued job reuses its venv only when nothing under it changed after its BUILT marker.
#   sbatch --export=NIL ... --partition=quanta-gh200 jobs/math_gh200_job.sh [MATH_*=value ...] TAG MODE ARGS...   (through math_stage.sh csail sbatch: pre-job check)
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
# CSAIL: `sbatch --export=NIL,VAR=...` gets the job held ("user env retrieval failed") or cancelled by root (2026-11-17 test 3099870),
# so settings come as leading MATH_*=value arguments (submit with plain --export=NIL); ORCD accepts both forms.
while [ $# -gt 0 ] && [[ $1 =~ ^MATH_[A-Z_]+=[A-Za-z0-9_./,:-]*$ ]]; do export "$1"; shift; done
CW=/data/scratch-fast/ngom/cmp_csail; W=$CW/math1; TAG=${1:?TAG}; shift; BUNDLE=${MATH_BUNDLE:?MATH_BUNDLE}; C=$W/code/$BUNDLE
UV=$CW/tools_gh200/uv; UV_SHA=92804e2f635c1791bb497437d94b15970f0d6d74811979315624cbb0f45b778d
G=$W/gh200/$TAG; V=$G/venv
say() { echo "$(date -u +%FT%TZ) [math gh200 $TAG] $*"; }
[ "$(uname -m)" = aarch64 ] || { say "not aarch64 ($(uname -m))"; exit 2; }
[ "$(sha256sum $UV | cut -d' ' -f1)" = $UV_SHA ] || { say "uv sha256 mismatch"; exit 2; }
export HOME=$CW/home UV_CACHE_DIR=$G/uv_cache UV_PYTHON_INSTALL_DIR=$G/uv_python UV_PROJECT_ENVIRONMENT=$V UV_NO_CONFIG=1 UV_LINK_MODE=copy \
  UV_PYTHON_PREFERENCE=only-managed XDG_CACHE_HOME=$CW/home/.cache XDG_CONFIG_HOME=$CW/home/.config TZ=UTC PYTHONDONTWRITEBYTECODE=1
mkdir -p $G $HOME
if [ -f $V/BUILT ] && [ -n "$(find $V $G/uv_python -cnewer $V/BUILT ! -path $V/BUILT 2>/dev/null | head -1)" ]; then
  say "venv changed after BUILT: moved aside"; mv $V $V.changed_$(date +%s); fi
if [ ! -f $V/BUILT ]; then
  [ -e $V ] && mv $V $V.partial_$(date +%s)
  say "building a fresh aarch64 venv from $C/uv.lock with uv $($UV --version)"
  $UV python install 3.14 && (cd $C && $UV sync --frozen --no-dev --no-install-project --compile-bytecode --project $C) || { say "venv build failed"; exit 3; }
  $V/bin/python -c "import torch, transformers, peft, fla; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))" || { say "venv check failed"; exit 3; }
  date -u +%FT%TZ > $V/BUILT
fi
say "venv $V (built $(cat $V/BUILT)); math_job.sh $*"
MATH_PY=$V/bin/python exec bash $W/jobs/math_job.sh "$@"
