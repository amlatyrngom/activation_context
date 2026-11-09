---
name: csail-slurm
description: How the agent container reaches the MIT CSAIL Slurm cluster (jump host, CSAIL Duo, two persistent masters, guarded calls), what the `csail` account gives (shared H200 tier, scratch), the rules carried over from ORCD, and the open questions. STATUS — IN PROGRESS (first access 2026-09-18); update as the cluster is used.
---

# CSAIL Slurm: access, resources and agent rules (in progress)

Docs: https://tig.csail.mit.edu/shared-computing/slurm/ (partition-and-qos, storage, quick-start), SSH: https://tig.csail.mit.edu/network-wireless/ssh/, Duo: https://duo.csail.mit.edu/. Support: help@csail.mit.edu.

## Facts established 2026-09-18

- **Identity.** CSAIL account `ngom` (separate from MIT Kerberos: its own password, its own Duo instance). Home is AFS (`/afs/csail.mit.edu/u/n/ngom`), 2 TB, **not visible to jobs** and **not usable for SSH keys**: sshd cannot read `~/.ssh/authorized_keys` from AFS before the user has a token, so a public key installed there is ignored ("Permission denied (publickey,gssapi-with-mic,password)"). Authentication to `slurm-login` is the CSAIL password (or GSSAPI with a CSAIL Kerberos ticket, untested from the container).
- **Path.** `slurm-login.csail.mit.edu` refuses direct connections from the internet ("Connection refused"). Everything goes through `jump.csail.mit.edu`, which is open to the internet, asks CSAIL Duo, and runs no commands (proxy only). Internal hosts trust connections proxied from the jump host (single factor there).
- **Account and QoS.** `sacctmgr show assoc user=$USER`: account `csail`, QoS `lab-free, normal, shared-if-available, tig-debug, tig-main`, no MaxJobs / MaxSubmit / MaxTRESPU on any of them. The `quanta` account (partitions `quanta-h200`, `quanta-gh200`, QoS `quanta-main`) is granted by the group's PI; requested 09-18.
- **Shared GPU tier.** Partition `csail-shared-h200`, QoS `shared-if-available`, account `csail`: 15 nodes (13 `aia-h200-*`, `khattab-h200-1`, `quanta-h200-1`), 8 H200 141 GB each (one with 7), 256 CPUs, 2 TB RAM, 119 GPUs; MaxTime 24 h; `PreemptMode=REQUEUE`, `PreemptType=preempt/qos`, `PreemptExemptTime 0`, `KillWait 10 s` (owners aia / quanta / khattab preempt at any time; a preempted job is requeued). Also `csail-shared-l40s` (one 8-L40S node) and `tig-cpu` (7 CPU nodes, 4 days). A single-GPU `srun` was allocated within seconds on 09-18 while ~370 single-GPU jobs of other users were pending: fairshare favours a small newcomer.
- **Nodes.** Driver 580.178.04, Ubuntu 22.04, system Python 3.10, no `uv`, no `module`; compute nodes reach the internet (Hugging Face 200). Login node: 16 CPUs, 62 GB.
- **Storage.** `/data/scratch` (50 TB NFS, 5.4 TB free on 09-18, 1 TiB per-user quota, quota not visible with `quota`), `/data/scratch-fast` (18 TB, 99 % full: do not rely on it), `/data/scratch-oc40`. Our root: **`/data/scratch/ngom`** (created 09-18). Jobs get a temporary local home on the node (fast, deleted at reboot). Output must go to NFS.

## Container SSH setup (installed 09-18; repeat in a new container)

`~/.ssh/config` (key file only for the jump-host block's `IdentitiesOnly`; the key is not what authenticates):

```
Host jump.csail.mit.edu
  User ngom
  IdentityFile ~/.ssh/id_ed25519_orcd
  IdentitiesOnly yes
  ControlMaster auto
  ControlPath ~/.ssh/cm/%C
  ControlPersist 168h
  ServerAliveInterval 60
  ServerAliveCountMax 10
  TCPKeepAlive yes
  ForwardAgent no
Host csail slurm-login.csail.mit.edu
  HostName slurm-login.csail.mit.edu
  User ngom
  IdentityFile ~/.ssh/id_ed25519_orcd
  IdentitiesOnly yes
  ProxyJump jump.csail.mit.edu
  ControlMaster auto
  ControlPath ~/.ssh/cm/%C
  ControlPersist 168h
  ServerAliveInterval 60
  ServerAliveCountMax 10
```

**User step (each time the channel is gone):** in a real terminal inside the container, `ssh csail echo ok` → host-key yes (first time), CSAIL password on the jump host, CSAIL Duo push, CSAIL password on slurm-login, "ok". That leaves two masters: `ssh -O check jump.csail.mit.edu` and `ssh -O check csail` both say "Master running". Only the second matters for the agent's calls. First-ever setup also needs Duo enrollment at https://duo.csail.mit.edu/ (an un-enrolled account gets "Please enroll at …" links from ssh and is then denied).

## Agent rules (carried over from ORCD, adapted)

- **Every remote call goes through `IB/CANON/CANON_ARTIFACTS/cluster_access/csail_ssh.sh`**: `ssh -O check csail` first (local socket only), then `ssh -o ControlMaster=no -o ProxyJump=none -o ProxyCommand=false -o BatchMode=yes csail …`. With the master gone the call fails locally in milliseconds with no network contact (verified 09-18): no script ever starts a Duo flow or a password prompt on the jump host. Exit 97 = channel down; loops print one "channel down" line and stay quiet.
- **Nothing heavy on the login node** (ORCD lesson 09-18: a hung `du`/`quota` got the whole session killed): `squeue`, `sacct`, `sbatch`, `scancel`, `sinfo`, small file reads, short python over small JSON. Sizes and disk checks run inside jobs.
- One call per monitor cycle, cycles ≥ 50 min for unattended loops.
- Cancel jobs by id only; kill by PID with bracketed patterns; never `pkill -f` a pattern that matches the calling shell.
- `.env` and tokens never leave the container; `HF_TOKEN` is not needed for the public Qwen weights.
- Old run folders are moved aside, never deleted; adapters worth keeping are copied to a KEEP folder on scratch when a job ends.

## Environment and shipping (done 09-18; repeat when the code changes)

- Root `R=/data/scratch/ngom/activation`: `bin/` (uv), `uv_python/`, `uv_cache/`, `venv/` (Python 3.14, `uv sync --frozen`), `hf/` (HF_HOME with Qwen3.5-2B/9B), `ws/` (tagged checkout: `activation/`, `pyproject.toml`, `uv.lock`), `ac_recon_paper/` (bench), `AC_RECON/preparation/` (materials, calibrations), `SYNC_RUNS/<arm>`, `KEEP/<arm>`, `jobs/` (sbatch files + `.out`), `home/` (HOME for jobs: AFS is invisible).
- Bootstrap was one CPU/GPU job (`jobs/setup.sbatch`): install uv under scratch, `uv python install 3.14`, `UV_PROJECT_ENVIRONMENT=$R/venv uv sync --frozen`, `snapshot_download` of the weights. ~25 min. Ran on the H200 partition because `tig-cpu` had an 80-job backlog.
- Shipping: `git archive <tag> activation IB/ARTIFACTS/AGENT_AC_PRETRAINING/ac_recon pyproject.toml uv.lock | csail_ssh.sh "tar -x -C $R/ws_new; mv …"` (previous `ws` / `ac_recon_paper` kept as `*_old`), then the manifest check (`training_source.json` hashes against `ws` + `ac_recon_paper`, run with the login node's python3). Large files (materials, adapters) go with `scp -o ControlMaster=no -o ProxyJump=none -o ProxyCommand=false -o BatchMode=yes … csail:…` and are checked with `md5sum`.

## Job shape (`csail_run.sh LABEL [HOURS] [GPUS]`, first used 09-18 18:03 for arm E4 = job 2019908)

`sbatch --account=csail --partition=csail-shared-h200 --qos=shared-if-available --gpus=1 --cpus-per-task=8 --mem=64G --time=24:00:00 --requeue --open-mode=append`; environment, weights (`HF_HOME`), checkout and run folders under `/data/scratch/ngom/activation/{venv,hf,ws,SYNC_RUNS,KEEP}`; mid-epoch checkpoints every 300 updates and resume from `partial_epoch.json` (a preemption gives 10 s, so the checkpoint on disk is the only state that survives).

## Allocation policy (agreed 2026-09-18, revisit)

- Shared tier only, single-GPU jobs, **at most 4 in flight** (running + pending) by default; more only with an explicit go-ahead. No multi-node jobs, no `--exclusive`, no 8-GPU requests on the shared tier without a reason.
- `--time` no longer than the run needs plus margin; long arms checkpoint often so a 24 h request is never a hostage.
- Memory and CPU requests sized to the arm (`--mem=64G --cpus-per-task=8` for our single-GPU arms), not the node.
- Watch `sshare -u $USER` occasionally; if our fairshare drops far below others', slow down rather than queue more.

## Queue behaviour seen so far

- 15:00 UTC: a 1-GPU `srun` on the shared tier started within seconds (20 running / 370 pending). 17:35: a 1-GPU `srun` waited 15 min without a node; 18:00: 481 pending / 20 running, one user holding ~400 pending 20-hour single-GPU requests at priority 28 vs our 24 (`sprio -u $USER`: fairshare 15, jobsize 10, age 0, QoS 0; age grows). Position in the pending list: `squeue -p csail-shared-h200 -t PD -o "%Q %u %r" | sort -rn`.
- `sacct` times are local (ET), `squeue %M` elapsed; `sinfo -p csail-shared-h200 -o "%D %t"` shows the tier at 11 mix/alloc, 3 down/drain on 09-18.
- A pending `srun` whose client was killed (timeout) can linger PENDING: cancel it by id.
- **`sbatch --export=ALL,VAR=value` is cancelled by root** (`CANCELLED by 0`, 1–2 s after start, no log file) on quanta-h200 (09-27, reproduced with a 2-line script; the same script with plain `sbatch` or `--export=ALL` completes). Pass variables through the submitting environment (`export VAR=value; sbatch job.sbatch`). 16 CPU/300G, 32 CPU/512G and 12 h CPU-only shapes all start normally.
- quanta-h200-1 is one node with **7 H200** shared by the quanta group: at 09-27 15:20Z another user held 6 of them (plus CPU-only jobs), so a 2-GPU job waits for that group's work to end.

## Open questions / to do

- GSSAPI from the container (`kinit ngom@CSAIL.MIT.EDU`, ticket cache) would let the slurm-login leg reopen without a password while the jump master lives; untested.
- Effective queue wait for 1-GPU jobs over a day (first data point: E4 2019908 submitted 18:03 UTC); how often shared jobs are preempted; whether the CSAIL run's checkpoints need the same pruning as ORCD's.
- quanta membership (GH200 nodes); the `quanta-gh200` partition is not visible to non-members.
