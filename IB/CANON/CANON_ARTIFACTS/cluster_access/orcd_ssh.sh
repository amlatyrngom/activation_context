#!/bin/bash
# ssh to ORCD ONLY through the live shared connection (ControlMaster socket). Never starts a fresh authentication unattended:
# ORCD needs Duo on every new connection, and a burst of failed keyboard-interactive attempts got this host's address blocked
# (2026-09-17 22:20-22:48 UTC). Exit 97 = shared connection down -> the user runs `ssh orcd echo ok` once.
#
# Login-node protection (09-26): status greps over the 57 GB teacher rollout caches wedged login009 and then login010 for
# ngom (every new session hung after auth until the reads drained). So every call runs remotely under a hard timeout at
# idle CPU/IO priority, and the known heavy patterns are refused here. Data scans go through srun/sbatch on compute nodes.
#   ORCD_TIMEOUT=<s> (default 60, max 900) for calls that only wait on srun.  Exit 98 = refused as heavy.
# Newer OpenSSH folds ProxyJump/ProxyCommand into %C, so pin the live master's socket path explicitly (2026-11-09).
sock=$(ssh -G orcd 2>/dev/null | awk '$1=="controlpath"{print $2}')
ssh -O check orcd >/dev/null 2>&1 || { echo "orcd: shared connection down; needs one interactive login (ssh orcd echo ok)" >&2; exit 97; }
cmd="$*"
limit=${ORCD_TIMEOUT:-60}
(( limit > 900 )) && limit=900
if [[ "$cmd" != *srun* && "$cmd" != *sbatch* ]]; then
  if grep -Eq '(^|[;&|( ]|\$\()(du|quota|find|locate)( |$)' <<<"$cmd" || grep -Eq 'rollouts\.jsonl' <<<"$cmd"; then
    echo "orcd_ssh: refused on the login node (du/quota/find or a rollout-cache read); run it via srun on a compute node" >&2
    exit 98
  fi
fi
exec ssh -o ControlMaster=no -o ControlPath="$sock" -o ProxyCommand=false -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 orcd \
  "timeout -k 5 $limit nice -n 19 ionice -c 3 bash -c $(printf '%q' "$cmd")"
