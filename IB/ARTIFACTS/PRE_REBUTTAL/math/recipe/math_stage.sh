#!/bin/bash
# Host-side staging and submission for the math recipe (2026-11-17), CSAIL or ORCD, ONLY through the user's shared connections
# (sec35/jobs/cluster_lib.sh: ControlMaster=no, ProxyCommand=false, BatchMode; exit 97 when a connection is down: stop, never retry).
# Every file a job reads or executes is uploaded from this host and checked by sha256 after the copy (CSAIL incident rule; ORCD the same).
#
#   math_stage.sh CL code BUNDLE           upload IB/TMP/PRE_REBUTTAL/math_recipe/code/BUNDLE.tar.gz, unpack into W/code/BUNDLE (never replaced),
#                                          check every file against MANIFEST.json
#   math_stage.sh CL put LOCAL REMOTE_REL  a file or a folder (every file's sha256 compared; ORCD: no find on the login node)
#   math_stage.sh CL jobs                  math_job.sh (+ math_gh200_job.sh on CSAIL) to W/jobs/
#   math_stage.sh CL sbatch BUNDLE ARGS..  CSAIL: the pre-job check first (no shell startup files; job scripts and bundle = this host's;
#                                          venv / HF cache unchanged since BUILT), then `cd W && sbatch ARGS`; ORCD: the job script hash, then sbatch
#   math_stage.sh CL squeue                our jobs
# CL = csail (W=/data/scratch-fast/ngom/cmp_csail/math1) | orcd (W=/home/ngom/orcd/scratch/activation/math1).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/../../sec35/jobs/cluster_lib.sh
CL=${1:?csail|orcd}; ACTION=${2:?action}; shift 2
cluster $CL
case $CL in csail) CW=/data/scratch-fast/ngom/cmp_csail; W=$CW/math1;; orcd) W=/home/ngom/orcd/scratch/activation/math1;; esac
CODE_LOCAL=$REPO/IB/TMP/PRE_REBUTTAL/math_recipe/code
JOBS=($HERE/math_job.sh); [ $CL = csail ] && JOBS+=($HERE/math_gh200_job.sh)
check_remote() { local want got; want=$(sha256sum < "$1" | cut -d' ' -f1); got=$(remote 600 "sha256sum < $2 | cut -d' ' -f1")
  [ "$want" = "$got" ] || { echo "MISMATCH $2: remote $got, local $want" >&2; return 1; }; echo "ok $2"; }
put() {
  local L=$1 R=$W/$2
  if [ -d "$L" ]; then
    remote 60 "mkdir -p $R"; xfer -a "$L/" $H:$R/
    local names; names=$(cd "$L" && ls -1 | tr '\n' ' ')
    for n in $names; do [ -d "$L/$n" ] && { put "$L/$n" "$2/$n"; continue; }
      [ "$(cd "$L" && sha256sum "$n")" = "$(remote 1800 "cd $R && sha256sum $n")" ] || { echo "MISMATCH $R/$n" >&2; exit 1; }; done
    echo "ok $R ($(wc -w <<<"$names") entries)"
  else remote 60 "mkdir -p $(dirname $R)"; xfer -a "$L" $H:$R; check_remote "$L" $R; fi; }
bundle_check() {   # every file of W/code/B = MANIFEST.json (CSAIL: find is allowed; ORCD: sha256sum of the manifest's names)
  local B=$1 manifest want got; manifest=$(mktemp); tar xzf $CODE_LOCAL/$B.tar.gz -O MANIFEST.json > $manifest
  want=$($LOCAL_PY -c "import json,sys; m=json.load(open(sys.argv[1]))['files']; print('\n'.join(f'{h}  ./{n}' for n, h in sorted(m.items())))" $manifest)
  if [ $CL = csail ]; then got=$(remote 600 "cd $W/code/$B && find . -type f ! -name MANIFEST.json -print0 | sort -z | xargs -0 sha256sum")
  else local names; names=$(awk '{print $2}' <<<"$want" | tr '\n' ' '); got=$(remote 600 "cd $W/code/$B && sha256sum $names"); fi
  [ "$(sort <<<"$want")" = "$(sort <<<"$got")" ] && echo "ok bundle $B: $(wc -l <<<"$got") files = MANIFEST" || { echo "BUNDLE MISMATCH $B" >&2; diff <(sort <<<"$want") <(sort <<<"$got") | head >&2; rm -f $manifest; return 1; }
  check_remote $manifest $W/code/$B/MANIFEST.json; rm -f $manifest; }
prejob() {
  local B=$1 bad=0 out
  for f in "${JOBS[@]}"; do check_remote $f $W/jobs/$(basename $f) || bad=1; done
  bundle_check $B || bad=1
  if [ $CL = csail ]; then
    out=$(remote 120 "ls -a ~ $CW/home 2>/dev/null | grep -xE '\.(bashrc|bash_profile|profile|bash_login|bash_logout|pam_environment|kshrc|zshrc|zshenv|login|cshrc|tcshrc)' ; ls -d ~/.config/environment.d $CW/home/.config/environment.d 2>/dev/null; true")
    [ -z "$out" ] && echo "ok no shell startup files" || { echo "FOUND startup files: $out" >&2; bad=1; }
    out=$(remote 300 "for d in venv:venv/BUILT uv_python:venv/BUILT hf:hf/BUILT tools:venv/BUILT; do dir=\${d%%:*}; m=$CW/\${d#*:}; [ -f \$m ] || { echo \"no marker \$m\"; continue; }; find $CW/\$dir -cnewer \$m ! -path $CW/venv/BUILT ! -path $CW/hf/BUILT 2>/dev/null | head -5; done")
    [ -z "$out" ] && echo "ok venv/uv_python/hf/tools unchanged since BUILT" || { echo "CHANGED after BUILT: $out" >&2; bad=1; }
  fi
  return $bad; }
case $ACTION in
  code)   B=${1:?BUNDLE}; T=$CODE_LOCAL/$B.tar.gz
          remote 30 "mkdir -p $W/code $W/jobs $W/data $W/init $W/runs $W/gen $W/logs; [ ! -e $W/code/$B ]" || { echo "$W/code/$B exists (never replaced)" >&2; exit 2; }
          remote 30 "mkdir -p $W/code"; xfer -a $T $H:$W/code/; check_remote $T $W/code/$B.tar.gz
          remote 300 "mkdir $W/code/$B.tmp && tar xzf $W/code/$B.tar.gz -C $W/code/$B.tmp && mv $W/code/$B.tmp $W/code/$B"; bundle_check $B;;
  put)    put "${1:?LOCAL}" "${2:?REMOTE_REL}";;
  jobs)   remote 30 "mkdir -p $W/jobs $W/logs"; for f in "${JOBS[@]}"; do xfer -a $f $H:$W/jobs/; check_remote $f $W/jobs/$(basename $f); done;;
  sbatch) B=${1:?BUNDLE}; shift; prejob $B || { echo "pre-job check failed: not submitting" >&2; exit 1; }
          remote 60 "cd $W && sbatch $(printf '%q ' "$@")";;
  squeue) remote 60 "squeue -u ngom -O JobID:12,Name:30,State:10,Partition:16,NodeList:18,TimeUsed:12,Reason:24";;
  *) sed -n 2,14p $0; exit 2;;
esac
