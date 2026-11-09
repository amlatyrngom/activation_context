# Shared by stage.sh, stage_init.sh, pull.sh (run in THIS container; sourced, not executed).
# Cluster access ONLY through the live shared ssh connection (ControlMaster socket opened by the user): ControlMaster=no +
# ProxyCommand=false + BatchMode=yes (+ ProxyJump=none for CSAIL) make a fresh authentication impossible; if the master is down
# the scripts exit 97 before any network contact. ORCD login node: commands run under timeout + nice + ionice, and never
# du / quota / find. SEC45_DRY_RUN=1 prints every remote command instead of running it (no ssh at all).
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)          # IB/ARTIFACTS/PRE_REBUTTAL/sec35
REPO=$(cd $KIT/../../../.. && pwd)
PULL_ROOT=${SEC45_PULL_ROOT:-$REPO/IB/TMP/PRE_REBUTTAL/sec35_runs}
DATA_ROOT=${SEC45_DATA_ROOT:-$REPO/IB/TMP/PRE_REBUTTAL/data}
LOCAL_PY=${SEC45_LOCAL_PY:-$REPO/.venv/bin/python}
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/sec35
MODEL=Qwen/Qwen3-4B-Instruct-2507
DRY=${SEC45_DRY_RUN:-0}

cluster() {   # sets H W HF PY SSHO RSH for $1
  case $1 in
    csail) H=csail; W=/data/scratch-fast/ngom/sec35; HF=/data/scratch-fast/ngom/sec45/hf; PY=/data/scratch/ngom/activation/venv/bin/python
           SSHO=(-o ControlMaster=no -o ProxyJump=none -o ProxyCommand=false -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30);;
    orcd)  H=orcd; W=/home/ngom/orcd/scratch/activation/sec35; HF=/home/ngom/orcd/scratch/activation/hf; PY=/home/ngom/orcd/scratch/activation/rq_v1/code/.venv/bin/python
           SSHO=(-o ControlMaster=no -o ProxyCommand=false -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30);;
    *) echo "CLUSTER must be csail or orcd" >&2; exit 2;;
  esac
  RSH="ssh ${SSHO[*]}"
  if [ "$DRY" = 1 ]; then echo "# dry run: ssh -O check $H (skipped)"; return; fi
  ssh -O check $H >/dev/null 2>&1 || { echo "$H: shared connection down; needs one interactive login by the user (ssh $H echo ok)" >&2; exit 97; }
}

remote() {    # remote SECONDS 'command'   (ORCD: timeout + idle priority; refuses du/quota/find)
  local limit=$1; shift; local cmd="$*"
  if [ $H = orcd ] && grep -Eq '(^|[;&|( ]|\$\()(du|quota|find|locate)( |$)' <<<"$cmd"; then echo "refused on the ORCD login node: $cmd" >&2; exit 98; fi
  local wrapped="timeout -k 5 $limit bash -c $(printf '%q' "$cmd")"
  [ $H = orcd ] && wrapped="timeout -k 5 $limit nice -n 19 ionice -c 3 bash -c $(printf '%q' "$cmd")"
  if [ "$DRY" = 1 ]; then echo "ssh ${SSHO[*]} $H $wrapped"; return 0; fi
  ssh "${SSHO[@]}" $H "$wrapped"
}

xfer() {      # rsync with the guarded transport
  if [ "$DRY" = 1 ]; then echo "rsync -e \"$RSH\" $*"; return 0; fi
  rsync -e "$RSH" "$@"
}
