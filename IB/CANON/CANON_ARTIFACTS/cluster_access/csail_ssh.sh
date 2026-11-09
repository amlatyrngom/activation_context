#!/bin/bash
# Guarded CSAIL call: only through the live slurm-login master (opened by the user: ssh csail echo ok = jump-host Duo + CSAIL password).
# If the master is gone, exit 97 without any network contact: ProxyJump=none + ProxyCommand=false make a fresh connection impossible
# (slurm-login refuses direct connections from the internet, and no jump-host Duo flow is ever started by a script).
# Newer OpenSSH folds ProxyJump/ProxyCommand into %C, so pin the live master's socket path explicitly (2026-11-09).
sock=$(ssh -G csail 2>/dev/null | awk '$1=="controlpath"{print $2}')
ssh -O check csail >/dev/null 2>&1 || { echo "csail: shared connection down; needs one interactive login (ssh csail echo ok)" >&2; exit 97; }
exec ssh -o ControlMaster=no -o ControlPath="$sock" -o ProxyJump=none -o ProxyCommand=false -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 csail "$@"
