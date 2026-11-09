# Host consolidation plan (2026-11-09)

Goal: work from the EC2 host (`i-0918b8f73e9a0fe03`, t3.2xlarge, 8 CPU, 30 GB RAM, no GPU,
repo at `/home/ubuntu/activation_context`) as the one normal workstation, with one git branch,
working credentials, free disk, and a short `IB/STATE.md`. Then start the pre-rebuttal work
(M5, planned separately).

## Facts found while planning

| Area | What is true now |
|---|---|
| This machine | It **is** `i-0918b8f73e9a0fe03`, the instance earlier agents marked "never touch". It hosted the VS Code container. It is the only EC2 instance in any region; no loose EBS volumes. |
| Git | `main` = `af52a4c` (your Sep 21 snapshot, 12 commits). `origin/paper-mvp-20261013` = `258fce6` (Oct 13, 17 agent commits). The two histories share only "Init Repo". `origin/ac-recon-night-20260917` is already an ancestor of the paper branch. Outside `IB/`, the host checkout has no local changes. |
| Paper code | `258fce6` + the container patch (`container_uncommitted_vs_258fce6.patch`, 10 files, +651/−74) = what the container ran for the paper. The container also had the paper repo staged as a submodule `paper` → github.com/amlatyrngom/activation_context_paper. |
| Old main vs paper | Your Sep 21 snapshot differs from the closest paper commit (`5352f5b`, Sep 17) in 24 paths under `activation/`: the `ac_model/` files, `agentic_program_probe.py`, several `tests/test_basic_*` files, and `tests/data/` fixtures (tiny Qwen3, legacy AC modules). The paper branch also drops the `IB/REPOS/vllm` submodule. |
| Credentials | AWS credentials: host and container identical. `.env`: host has the same `GITHUB_TOKEN` value plus `HF_TOKEN`, so the host's file is already the superset. `gh`: logged in. Git identity: set on the host (the container's credential helper is VS Code-only). SSH: the host has **no** `~/.ssh`, so the CSAIL/ORCD key and config are missing. SkyPilot: host `~/.sky` exists; no clusters anywhere. No `aws` CLI on PATH (`uvx --from awscli aws` works); `.venv/bin/sky` exists. No TeX. The container's Claude `settings.json` only holds container-specific auto-mode notes; not worth carrying. |
| Disk | 402 of 484 GB used. `IB/TMP` 249 GB (SYNC 106, PAPER_COMPACTION 47, PAPER_TRAJ_QA 46, PAPER_SEC45 42, AGENT_AC_PRETRAINING 6.8, rest small). Old VS Code container ≈102 GB (writable layer). Host HF cache 29 GB, uv cache 8.7 GB. |
| Container-only data | `~/.claude/jobs/6f1c1373/tmp` 28 GB (the paper session's `$CLAUDE_JOB_DIR/tmp`, referenced by STATE, e.g. `tmp/pr/res/`), `/tmp/claude-1000/-workspace` 9.9 GB (task outputs), transcripts 954 MB, `/workspace/TMP` 70 MB. The rest (`.venv`, HF/uv caches, VS Code server) is re-creatable. |
| S3 | `s3://general-bucket-aln/activation-context-data/` holds ≈1.97 TB in 250,826 objects (checkpoints, run archives, two container archives from Oct 15 and Oct 24). |
| ORCD "reservation 23493337" | Not a reservation. It is a Slurm **job** named `data_amount_confirmation`, submitted on ORCD by an agent on Sep 22 (FIRST_TASK_AWARE_FINETUNING), then put on user hold, and used in a dependency fence. Nothing refers to it after September. |
| STATE.md | 406 KB / 1,231 lines (Tressoir target ≈8 KB). Header summary dates from Oct 14; the rest is an Oct 14–27 log. Canon: ROOT 35 KB, ACTIVATION_CONTEXT 38 KB, AGENT_TRAINING 9 KB, plus a stray `ROOT_CANON.md.orig`. |

## Milestones

Order: M2 (credentials, quick) → M1 (git) → M3 (disk) → M4 (state and canon). M5 is the next plan.

### M1 — One `main` branch = the paper code

1. Tag the old main: `archive/main-20260921` → `af52a4c`; push the tag.
2. Build the new commit in a temporary worktree from `origin/paper-mvp-20261013`:
   `git apply --check` then `git apply` the container patch; add `.gitmodules` with
   `paper` (activation_context_paper at `16a0260`, the submitted version) and keep `IB/REPOS/vllm`
   (pinned v0.28.0, already on disk).
3. Bring in the current `IB/` from the host (it is newer than the IB committed on the paper branch):
   everything except `IB/TMP` (git-ignored), `IB/ARTIFACTS/HANDOFF_EC2/carry`, and artifact files
   over 5 MB (11 files, ≈260 MB, all in `FIRST_TASK_AWARE_FINETUNING/`: `*.evidence.json`,
   `*_COHORT100.tressoir.html`) — those go to S3 with a manifest, and a `.gitignore` rule keeps
   them out. Remove `IB/CANON/ROOT_CANON.md.orig`.
4. `git merge -s ours archive/main-20260921` so the old history stays reachable and the push to
   `origin/main` is a fast-forward (no force push).
5. Point the host checkout at it: `git reset --mixed <new>` (index only; IB already matches), then
   check out the non-IB paths from the new commit; `git status` must be clean.
6. Validate: `python -m compileall -q activation`, plus the CPU unit tests that need no GPU
   (`pytest activation/tests -m "not gpu"` or the fast subset that exists).
7. Push `main`; delete the remote branches `paper-mvp-20261013` and `ac-recon-night-20260917`
   (both are then ancestors of `main`).

The 24 old-main-only differences are **not** ported (see decision D1); they stay in the tag.

### M2 — Credentials and tools on the host

- SSH: install `carry/ssh/{config,id_ed25519_orcd,.pub,known_hosts}` into `~/.ssh` (none exists, so
  nothing to merge). First CSAIL/ORCD connection needs Duo: you run it.
- Move the guarded cluster wrappers `IB/TMP/AGENT_AC_PRETRAINING/ac_recon/{csail,orcd}_ssh.sh` out of
  `IB/TMP` into `IB/CANON/CANON_ARTIFACTS/cluster_access/` and drop their `/workspace` assumptions.
- AWS: already identical. Install the CLI: `uv tool install awscli`.
- `.env`, `gh`, git identity: nothing to do.
- SkyPilot: keep the host's `~/.sky`; check with `.venv/bin/sky check aws`.
- TeX: install TinyTeX in `~/.TinyTeX`; clone the paper as the `paper` submodule; repoint
  `IB/ARTIFACTS/PAPER_WRITING/build_pdf.sh` from `/workspace/paper` to the repo's `paper/`; build
  once and report page counts.
- Archive `carry/claude/paper_session_6f1c1373.jsonl.gz` (130 MB) to S3, then delete `carry/`.

### M3 — Free disk (target: well under 100 GB used)

1. **Old VS Code container** (≈102 GB): upload container-only data to
   `s3://general-bucket-aln/activation-context-data/container_final_20261109/` — the job folder
   `~/.claude/jobs/6f1c1373/` (28 GB), `/tmp/claude-1000/-workspace/` (9.9 GB), the transcripts
   (954 MB), `/workspace/TMP`, `/workspace/kl_rows.json`. Check object counts and sizes, then
   `docker rm -f` the container and `docker builder prune` (3.7 GB). Images (5 GB) stay.
2. **`IB/TMP`** (249 GB): `aws s3 sync` all of it to `.../host_ib_tmp_20261109/`, verify sizes per
   folder, then delete local data files, keeping small text files (scripts, `.md`, `.json`, logs
   under 1 MB) so kits and notes stay searchable. Directories with a `PRESERVE` flag are uploaded like
   everything else. Pre-paper folders follow decision D2.
3. Record what moved where in `IB/ARTIFACTS/HANDOFF_EC2/S3_INDEX.md` (prefix, contents, size, date).
4. Leave the existing 1.97 TB on S3 alone for now; pruning it is a later, separate pass.

### M4 — STATE consolidation, canon and memory

1. Archive the current `IB/STATE.md` verbatim as `IB/ARTIFACTS/HANDOFF_EC2/STATE_archive_20261109.md`
   (committed, so nothing is lost).
2. Read the whole archive and sort its content:
   - **Paper evidence map → new `IB/CANON/PAPER_CANON.md`:** for each paper figure/table, the script,
     data path (local or S3) and final numbers; per workload the eval set, seeds, caps, decoding
     settings and training hyperparameters actually used (this is the input for the "less arbitrary
     training" work).
   - **Durable lessons → existing canon files** (training, cluster use, storage).
   - **Chronology, dead runs, node juggling →** stays only in the archive.
3. Rewrite the guardrails in `ROOT_CANON.md` and STATE "Rules":
   - keep: never start Duo or any interactive auth on CSAIL/ORCD (you do it);
   - replace "never touch `i-0918b8f73e9a0fe03`" with "this host is the workstation: never stop or
     terminate it; other EC2 resources are ours to manage";
   - drop: ORCD 23493337 (cancel the held job by id the next time ORCD access is up, if it still
     exists), "never copy `optimizer.pt`" (a storage judgement per run), and "commit only when asked"
     (see D3); keep "cancel Slurm jobs by id" as a plain practice;
   - replace every `/workspace` path with the repo root; drop container-only notes.
4. Your working preferences (handoff section 4: before/after edits on your text, pull Overleaf first,
   dictation, HTML diffs, page fit after builds, they/them) go into `ROOT_CANON.md` so every agent
   sees them; Claude memory gets a short pointer file.
5. New `IB/STATE.md` (≈8 KB): direction (pre-rebuttal), where things live (host, S3 index, paper
   repo), open items (handoff section 3, trimmed to what matters), rules.

### M5 — Pre-rebuttal work (next plan, not detailed here)

Text + reader-LoRA baseline on the main workloads; a standard, documented training and
hyperparameter process with reruns where settings were missing; fairer baselines (e.g. retrieval
decoding settings); rerun of `S_kv_v4fr`. Minor caption/wording issues are ignored. Compute: this
host has no GPU, so GPUs come from SkyPilot/EC2, CSAIL or ORCD.

## Progress (2026-11-09)

- **M2 done.** SSH config/key/known hosts installed (`~/.ssh/config` now uses `ControlPath ~/.ssh/cm/%r@%h:%p`, because
  OpenSSH 10 folds the guard's `ProxyJump`/`ProxyCommand` overrides into `%C`); wrappers moved to
  `IB/CANON/CANON_ARTIFACTS/cluster_access/` and validated on CSAIL and ORCD. AWS CLI via `uv tool install awscli`.
  `.env`, AWS credentials, `gh`, git identity, SkyPilot state needed nothing. TinyTeX installed (plus `courier`,
  `helvetic`, `times`, `psnfss`, which the build's auto-installer misses); the paper builds: 15 pages, body ends
  on page 10, appendix 3 pages, no errors or undefined references. The build script needed no repointing.
  CSAIL held job 2741502 cancelled (user); ORCD had no jobs (23493337 is gone).
- **M1 done.** `main` = `5b48032` (paper branch + container patch + live IB + `paper` submodule at `16a0260`, with
  old main merged in by `-s ours`), pushed as a fast-forward; tag `archive/main-20260921`; remote branches
  `paper-mvp-20261013` and `ac-recon-night-20260917` deleted. Drift from the plan: `IB/REPOS/vllm` stays
  git-ignored as on the paper branch (not a submodule). `compileall` passes; CPU tests: 4 passed, 21 GPU-gated
  skipped, `test_trajectory_dataset_loading` fails: one streamed Open-SWE trajectory loses message text when chunked
  (line 229 content check). The old main (`archive/main-20260921`) fails identically, so it predates the consolidation;
  left as a known issue in STATE.
- **M4 done.** STATE archived + digest, new STATE, `PAPER_CANON.md`, ROOT_CANON guardrails/preferences/lessons,
  Claude memory pointer. Commit `c351ae6`.
- **M3 in progress.** Container data copied (41 GB) to `IB/TMP/CONTAINER_FINAL_20261109` and syncing to S3, then
  `IB/TMP`; verify per folder with `verify_prune.py`, then prune and remove the container.

## Decisions (accepted by the user, 2026-11-09)

- **D1 — Old main's code:** the paper code wins entirely; the old main stays only as tag
  `archive/main-20260921`; nothing is ported.
- **D2 — `IB/TMP` data, pre-paper included:** upload everything to S3, then delete locally (small
  scripts and notes stay).
- **D3 — Commits:** agents commit to `main` and push at milestone boundaries without asking.
- **D4 — Old VS Code container:** remove it after its unique data is on S3.
