---
name: orcd-access
description: What the user must do by hand for the MIT ORCD Engaging cluster (account, keys, Duo-gated ControlMaster, giving the agent container access) and the rules the agent follows when using that access. Cluster operation itself is planned in the slice 4a plan (`orcd.py`).
---

# ORCD access: user-side setup and agent rules

Engaging (MIT ORCD) is a Slurm cluster reached over SSH with Kerberos identity and Duo. Keys do not bypass Duo: sshd chains Duo after a successful public key and offers no GSSAPI, so every fresh connection needs one Duo push (probed 2026-07-16; do not re-investigate). The only practical way for an agent to operate is an SSH ControlMaster channel that the user opens once and that later commands multiplex over. Everything in "User steps" is interactive and is the user's to do; the agent never starts a Duo flow.

## User steps (one-time)

1. **Account.** Log into the OnDemand portal https://orcd-ood.mit.edu with MIT Kerberos once; the account is provisioned on first login and job submission activates within about an hour.
2. **SSH key.** `ssh-keygen -t ed25519` if needed, then `ssh-copy-id USERNAME@orcd-login.mit.edu` (Kerberos password plus Duo for that one login).
3. **Tell the agent the Kerberos USERNAME.** It appears in every remote path and in the SSH config below.
4. **Storage and conda redirection on the cluster** (once, on a login node): create `~/orcd/pool/activation` and `~/orcd/scratch/activation`; put `export HF_HOME=$HOME/orcd/pool/hf` in `~/.bashrc`; write `~/.condarc` pointing `envs_dirs` and `pkgs_dirs` at `~/orcd/scratch/.conda/{envs,pkgs}` with `auto_activate_base: false`. Never keep model weights in home (200 GB, backed up, and the whole system gets laggy); pool (1 TB) keeps weights and datasets, scratch (1 TB) is purged after six months without a login. Quotas: `cat ~/orcd/.quota`.

## User steps (recurring): keep the master channel warm

The ControlMaster is a background `ssh` process holding one authenticated TCP connection. `ControlPersist 168h` is a ceiling, not a guarantee: a sleep, a network change or a container restart kills it. Expected cadence is roughly weekly plus after real interruptions. Opening it is one command plus one Duo push:

```bash
ssh orcd true        # answers the Duo prompt; leaves the master running in the background
ssh -O check orcd    # "Master running" confirms the channel
```

Failed authentications in a row temporarily lock the account (it recovers on its own), so do not retry a Duo prompt blindly.

## Giving the agent container access

The agent runs in a devcontainer on the AWS host. Its SSH client can only use a master whose Unix socket is reachable inside the container. Two ways, in order of preference.

**A. Open the master inside the container (simplest).** The agent installs the `Host orcd` block below in the container's `~/.ssh/config`. The user opens a VS Code terminal attached to the container (any shell inside the container, not the `!` prefix of the agent prompt, which has no interactive stdin for Duo) and runs `ssh orcd true`, answering the Duo push. The container has its own key, `~/.ssh/id_ed25519_orcd` (generated 2026-09-09, no passphrase, `IdentitiesOnly yes` in the host block); its public half must be appended to `~/.ssh/authorized_keys` on Engaging (`ssh-copy-id -i` from a machine that already has access, or the OnDemand file editor at orcd-ood.mit.edu under Files > Home Directory with dotfiles shown). Never remove existing lines from `authorized_keys`. The docs (orcd-docs.mit.edu, SSH key setup) say keys skip neither the Kerberos password nor Duo, so the opening command may ask for both; regenerate the key after a container rebuild. The master lives for as long as the container does; after a container rebuild the user repeats the command.

**B. Mount a host master into the container.** The user keeps the master on the AWS host with `ControlPath ~/.ssh/cm/%C` there, and the devcontainer mounts the host's `~/.ssh/cm` directory (not the socket file, which may not exist at container start) to `/home/node/.ssh/cm`. The socket's owner must be writable by the container user (uid 1000): same uid on the host, or `chmod 770` on the directory plus a shared group. This survives container rebuilds but depends on uid alignment; use it only if A proves annoying.

Container `~/.ssh/config` block (installed in this container with `User ngom`, the Kerberos id without the `@mit.edu` realm):

```
Host orcd
  HostName orcd-login.mit.edu
  User USERNAME
  ControlMaster auto
  ControlPath ~/.ssh/cm/%C
  ControlPersist 168h
  ServerAliveInterval 60
  ServerAliveCountMax 10
  TCPKeepAlive yes
```

`%C` keeps the socket path short (Unix sockets are limited to 108 characters), which matters inside `/home/node/.ssh/cm`.

## Agent rules

- Before any remote command run `ssh -O check orcd`. It talks only to the local socket. If it fails, stop and ask the user to reopen the channel; do not run `ssh orcd` without a live master.
- Run remote commands as `ssh -o BatchMode=yes -o ControlMaster=no orcd '<cmd>'`. BatchMode refuses every interactive prompt and `ControlMaster=no` never spawns a new master, so a cold channel fails fast instead of pushing a Duo notification to the user's phone.
- File transfer rides the same channel: `rsync -az -e "ssh -o BatchMode=yes -o ControlMaster=no" ...`.
- Login nodes run nothing heavier than `rsync`, `sbatch`, `squeue`, `scancel`, `sacct`, `cat`. Compute goes through Slurm. Cancel jobs by id (`scancel JOBID`), never by user or `--all`.
- Weights, caches, virtual environments and job output live under `~/orcd/pool/activation` (kept) or `~/orcd/scratch/activation` (active I/O), never in home.
- Base tier: `mit_normal_gpu` gives at most 2 GPUs (H200 141 GB or L40S) for 6 hours per job; `mit_preemptable` gives 4 GPUs for 48 hours but may be killed. Always set `-c` and `--mem`; H200s can queue for hours, L40S schedules fast for small checks. Compute nodes have internet.
- Support: orcd-help-engaging@mit.edu with job ids and full errors. Maintenance: third Tuesday monthly, Mondays 7am briefly.

## Probed facts (2026-09-09, jobs 22418007 and 22418125 on L40S compute nodes)

- No sudo for this account (`sudo -n` asks for a password on login and compute nodes; the uid comes from the central directory). No podman, no docker, no subuid or subgid ranges, cgroup v1 under Slurm. Rootless podman is therefore not an option without the admins.
- Apptainer works as the user: `module load apptainer/1.5.2` (batch scripts must first `source /etc/profile.d/modules.sh`; `/usr/bin/apptainer` exists only on the login node). Pulling `docker://python:3.12-slim` to a SIF in pool took 4 s; `--fakeroot` works through a root-mapped namespace; instances with `--contain --writable-tmpfs` keep files across exec calls; `--nv` passes the GPU. Cache dirs: `APPTAINER_CACHEDIR` in scratch, `APPTAINER_TMPDIR` on the node's `/tmp`.
- Compute nodes: Rocky 8.10, local xfs `/tmp` and `/scratch` with about 690 GB free, `/dev/fuse` and `newuidmap` present, internet access. `mit_normal_gpu` has H200 nodes (8 per node, 2 per user), some H100, and many L40S (46 GB). L40S jobs started within seconds.
- Keep `#SBATCH` directives contiguous at the top of a script; a command line before them ends directive parsing and the job runs with defaults.
