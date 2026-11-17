#!/bin/bash
# Host-side CSAIL helper for compaction stage 1 (2026-11-17). Everything goes over the user's shared connection (sec35/jobs/cluster_lib.sh:
# ControlMaster=no, ProxyCommand=false, BatchMode; exit 97 when it is down: stop, never retry). CSAIL incident rule: every file a job
# executes or imports was uploaded from this host and is checked by sha256 on arrival and again before each submit.
#
#   csail_stage.sh tools                 upload uv (this host's binary) to $W/tools/uv; sha256 checked
#   csail_stage.sh code BUNDLE           upload IB/TMP/.../code/BUNDLE.tar.gz, check its sha256, unpack into $W/code/BUNDLE (never
#                                        replaced) and check every file against MANIFEST.json (and that no other file is there)
#   csail_stage.sh jobs                  upload this folder's *_job.sh and stage1/cmp_stage1_job.sh to $W/jobs/ (sha256 checked)
#   csail_stage.sh put LOCAL REMOTE_REL  upload a file or a folder to $W/REMOTE_REL (sha256 of every file checked after the copy)
#   csail_stage.sh verify BUNDLE         the pre-job check (below); exit 1 on any finding
#   csail_stage.sh sbatch BUNDLE ARGS..  verify, then `cd $W && sbatch --export=NIL,... ARGS` (pass the account/partition flags in ARGS)
# Pre-job check: (a) the AFS home and $W/home hold no shell startup files (.bashrc, .bash_profile, .profile, .bash_login, .bash_logout,
# .pam_environment, .config/environment.d) and no BASH_ENV-style file; (b) the job scripts on CSAIL hash to this host's copies; (c) the
# bundle (the jobs' PYTHONPATH dirs) holds exactly MANIFEST.json's files with their hashes, no __pycache__ / .pth / sitecustomize;
# (d) nothing under venv/, uv_python/, hf/, tools/ changed (inode change time, which userspace cannot set back) after its BUILT marker.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/../../../sec35/jobs/cluster_lib.sh
cluster csail
W=/data/scratch-fast/ngom/cmp_csail
CODE_LOCAL=$REPO/IB/TMP/PRE_REBUTTAL/compaction_stage1/code
JOBS_LOCAL=($HERE/csail_setup_job.sh $HERE/csail_smoke_job.sh $HERE/csail_probe_job.sh $HERE/../cmp_stage1_job.sh $HERE/../../agent_eval/probe_think.py)
check_remote() {   # check_remote LOCAL_FILE REMOTE_PATH: sha256 equal or fail
  local want got; want=$(sha256sum "$1" | cut -d' ' -f1); got=$(remote 120 "sha256sum $2 | cut -d' ' -f1")
  [ "$want" = "$got" ] || { echo "MISMATCH $2: remote $got, local $want" >&2; return 1; }; echo "ok $2 ($want)"; }
verify() {
  local B=${1:?BUNDLE} bad=0 out
  out=$(remote 120 "ls -a ~ $W/home 2>/dev/null | grep -xE '\.(bashrc|bash_profile|profile|bash_login|bash_logout|pam_environment|kshrc|zshrc|zshenv|login|cshrc|tcshrc)' ; ls -d ~/.config/environment.d $W/home/.config/environment.d 2>/dev/null; true")
  [ -z "$out" ] && echo "ok no shell startup files in ~ or $W/home" || { echo "FOUND startup files: $out" >&2; bad=1; }
  for f in "${JOBS_LOCAL[@]}"; do [ -f $f ] && { check_remote $f $W/jobs/$(basename $f) || bad=1; }; done
  local manifest=$(mktemp); tar xzf $CODE_LOCAL/$B.tar.gz -O MANIFEST.json > $manifest
  local want; want=$($LOCAL_PY -c "import json,sys; m=json.load(open(sys.argv[1]))['files']; print('\n'.join(f'{h}  ./{n}' for n, h in sorted(m.items())))" $manifest)
  local got; got=$(remote 600 "cd $W/code/$B && find . -type f ! -name MANIFEST.json -print0 | sort -z | xargs -0 sha256sum")
  [ "$(sort <<<"$want")" = "$(sort <<<"$got")" ] && echo "ok bundle $B: $(wc -l <<<"$got") files = MANIFEST" || { echo "BUNDLE MISMATCH:"; diff <(sort <<<"$want") <(sort <<<"$got") | head -20; bad=1; } >&2
  check_remote $manifest $W/code/$B/MANIFEST.json || bad=1; rm -f $manifest
  out=$(remote 300 "for d in venv:venv/BUILT uv_python:venv/BUILT hf:hf/BUILT tools:venv/BUILT; do dir=\${d%%:*}; m=$W/\${d#*:}; [ -f \$m ] || { echo \"no marker \$m\"; continue; }; find $W/\$dir -cnewer \$m ! -path $W/venv/BUILT ! -path $W/hf/BUILT 2>/dev/null | head -5; done")
  [ -z "$out" ] && echo "ok venv/uv_python/hf/tools unchanged since BUILT" || { echo "CHANGED after BUILT: $out" >&2; bad=1; }
  return $bad
}
case ${1:-} in
  tools) remote 60 "mkdir -p $W/tools $W/code $W/jobs $W/data $W/init $W/runs $W/home $W/logs"
         xfer -a --chmod=F755 $(command -v uv) csail:$W/tools/uv; check_remote $(command -v uv) $W/tools/uv;;
  code)  B=${2:?BUNDLE}; T=$CODE_LOCAL/$B.tar.gz
         remote 30 "[ ! -e $W/code/$B ]" || { echo "$W/code/$B exists (never replaced)" >&2; exit 2; }
         xfer -a $T csail:$W/code/; check_remote $T $W/code/$B.tar.gz
         remote 120 "mkdir $W/code/$B && tar xzf $W/code/$B.tar.gz -C $W/code/$B"; verify $B || true;;
  jobs)  for f in "${JOBS_LOCAL[@]}"; do [ -f $f ] && { xfer -a $f csail:$W/jobs/; check_remote $f $W/jobs/$(basename $f); }; done;;
  put)   L=${2:?LOCAL}; R=$W/${3:?REMOTE_REL}
         if [ -d "$L" ]; then remote 30 "mkdir -p $R"; xfer -a "$L/" csail:$R/
           want=$(cd "$L" && find . -type f -print0 | sort -z | xargs -0 sha256sum); got=$(remote 1800 "cd $R && find . -type f -print0 | sort -z | xargs -0 sha256sum")
           [ "$want" = "$got" ] && echo "ok $R: $(wc -l <<<"$got") files" || { echo "MISMATCH under $R" >&2; diff <(echo "$want") <(echo "$got") | head; exit 1; }
         else remote 30 "mkdir -p $(dirname $R)"; xfer -a "$L" csail:$R; check_remote "$L" $R; fi;;
  verify) verify ${2:?BUNDLE};;
  sbatch) B=${2:?BUNDLE}; shift 2; verify $B || { echo "pre-job check failed: not submitting" >&2; exit 1; }
          remote 60 "cd $W && sbatch $(printf '%q ' "$@")";;
  *) sed -n 2,20p $0; exit 2;;
esac
