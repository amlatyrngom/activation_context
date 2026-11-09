# Guarded cluster access (CSAIL, ORCD)

`csail_ssh.sh` and `orcd_ssh.sh` run one remote command over the shared SSH connection that the user
opened by hand (`ssh csail echo ok`, `ssh orcd echo ok`; Duo each time, valid up to 7 days). They
never start a login: if the shared connection is down they exit 97 without network contact, and the
user re-opens it. `orcd_ssh.sh` also runs every call under a timeout at idle priority and refuses
heavy login-node commands (`du`, `quota`, `find`, rollout-cache reads; exit 98) unless they go
through `srun`/`sbatch`.

Usage, from the repo root:

```bash
IB/CANON/CANON_ARTIFACTS/cluster_access/csail_ssh.sh 'squeue -u ngom'
ORCD_TIMEOUT=300 IB/CANON/CANON_ARTIFACTS/cluster_access/orcd_ssh.sh 'srun ... '
```

Prerequisites: the `csail`, `jump.csail.mit.edu`, `orcd-hop` and `orcd` blocks in `~/.ssh/config`
(copy kept in `IB/ARTIFACTS/HANDOFF_EC2/ssh_config_20261109`) with the key `~/.ssh/id_ed25519_orcd`.

**Socket names (2026-11-09).** OpenSSH 10 folds `ProxyJump`/`ProxyCommand` overrides into `%C`, so
a guarded call that adds `-o ProxyCommand=false` would look for a different socket than the one the
user's login created. The host config therefore uses `ControlPath ~/.ssh/cm/%r@%h:%p`, which the
overrides do not change, and both scripts also pin the socket path from `ssh -G`. Old kit scripts
that call `ssh`/`rsync` with these overrides work for the same reason.

Validated 2026-11-09 on the EC2 host: both scripts reached `slurm-login-0` and `login010`.
The old paths `IB/TMP/AGENT_AC_PRETRAINING/ac_recon/{csail,orcd}_ssh.sh` are symlinks to these files.
