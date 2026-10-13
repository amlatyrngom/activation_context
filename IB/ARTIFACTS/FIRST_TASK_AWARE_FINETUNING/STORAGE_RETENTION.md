# Experiment storage retention

The responsible experiment agent runs this check during monitoring and before
large runs, on the local machine, ORCD, CSAIL and any active Sky EC2 nodes.
It is an agent-operated inventory/review/delete workflow, not a background
process that deletes files without checking the experiment state.

- A plain `PRESERVE` file protects its directory and descendants. Ancestor
  flags count; a flag on either AC or reader side protects their pair.
- Keep at least the newest approximately 250 GB per storage allocation,
  including the group crossing the boundary. The first implementation is more
  conservative: it retains the newest 250 GiB of unflagged checkpoint groups
  **in addition to** all flagged data and all data outside its explicit scope.
- Mark selected important checkpoints, active outputs, frozen materials and
  reports before pruning. Never prune code, reports, canon, credentials,
  shared model caches or environments as part of this procedure.
- When an arm finishes, retire its agent-written broad active flag and obsolete
  historical-best flags after protecting the actual selected pairs. Removing a
  marker does not delete data or bypass the newest250GB rule. Preserve user or
  ambiguous flags. Check ancestors: a broad campaign flag still protects every
  descendant until it too is narrowed.
- Only important checkpoint selections go to S3. Check the archive manifest
  and checksums before depending on S3 as their persistence layer. Do not upload
  all transient checkpoints. P2/P2s5 already have verified S3 archives in
  `CHECKPOINT_ARCHIVE.json`; no additional archival is needed for this cleanup.
- MIT inventory and deletion run in CPU Slurm jobs. Guarded SSH reads only
  small manifests/logs and scheduler state; do not run `du`, recursive scans or
  quota queries on login nodes. Sky/local checks run on the relevant machine.

`bench/prune_checkpoints.py` inventories explicitly named `SYNC_RUNS`-style
roots. Each run's entire `AC_MODELS` and `LORAS` trees form one conservative
group. Other run files remain. Inspect the written manifest, confirm candidates
are inactive, then use `--execute --manifest ... --ledger ...`. Execution
recomputes the recent-data boundary, rejects changed groups and flags, checks
again before each deletion, and records both the start and completion of each
group. No symlinked root or member is selected; internal symlinks are not
traversed. Do not run it concurrently with writers to candidate groups.

Before long runs, budget peak checkpoint storage, including optimizer state,
AC/reader weights, `latest` copies and retained snapshots. Check **user quota**
as well as filesystem free space. ORCD scratch and pool have separate 1 TB
allocations. A filesystem can have free space while the user's quota is full.
An incomplete or zero-byte optimizer file is not a resumable checkpoint.

## First pass — 2026-09-21

Evidence lives in `IB/TMP/FIRST_TASK_AWARE_FINETUNING/storage_retention/` and
the small ORCD logs under `~/orcd/pool/activation/taskaware/retention_*`.

- Local: filesystem196 GB used/288 GB free; all experiment data is below the
  retention floor. No deletion. P2/P2s5 and current task-aware report/material
  roots flagged. The scoped checkpoint inventory retains all eight groups.
- CSAIL: entire activation allocation59 GiB, with12 GiB KEEP and15 GiB
  SYNC_RUNS; no deletion. KEEP and task-aware roots flagged by CPU job2104052.
- Sky: fresh `sky ls` reports no clusters, managed jobs or services, so no
  remote Sky disks require cleanup. Future Sky nodes use the same procedure
  against retained artifact roots outside the source mirror.
- ORCD: scratch quota1022.4/1024 GB; SYNC_RUNS545 GiB plus KEEP170 GiB.
  Initial dry run protects252.58 GiB of recent checkpoint groups and proposes
  292.11 GiB for removal. Scratch is too full even to create a zero-byte flag.
  The first scoped deletion frees an obsolete K16r1 checkpoint pair, then
  creates flags and recomputes the inventory before the remaining cleanup.
  KEEP, both task-aware roots and selected historical W16L/A32L/A32KVTAL/
  A32KVTALc/NKVTAL checkpoints remain protected.

Completed17:07 UTC: CPU jobs23401984 and23402061 removed52 inactive
checkpoint groups,269,638,948,458 file bytes (251.12 GiB; allocated-block
estimate251.123 GiB). All104 begin/completion ledger records are present.
SYNC_RUNS now occupies315,228,385,280 bytes; KEEP182,118,383,616 bytes and
scratch task-aware11,481,251,840 bytes remain. The newest252.58 GiB plus the
five explicitly flagged historical selections are retained. Reports, run-root
metadata, source and base caches were outside every deletion target.

The ORCD quota report still showed its cached12:56 EDT reading immediately
after cleanup; do not mistake that stale value for a failed deletion. The
deletion ledger and post-cleanup `du` are direct evidence. Creating the flags
after the first deletion also succeeded, confirming quota headroom returned.

Fresh AWS inventory covered17 enabled regions without errors: only this
workspace's EC2 host `i-0918b8f73e9a0fe03` remains. It is preserved. No EC2
experiment disks or orphan GPU nodes need pruning.

Validation: fixture checks passed for paired deletion, a reader-side flag,
ancestor flags, the crossing-boundary group, stale-manifest rejection and
symlink exclusion. Python compilation passed. Production execution rechecked
all candidate fingerprints and protection decisions before deleting anything.

Next responsible-agent action: revisit headroom at the next monitoring check
and before another long run or checkpoint-heavy sweep. Inventory active pools
separately; narrow completed-run flags to selected checkpoints. Verify S3 before
relying on it as the only persistent copy of an important checkpoint. Repeat
the same check on any newly launched Sky node and on local/CSAIL storage if it
grows beyond the retention floor.

## Completed-arm marker upkeep — 2026-09-21

At19:19 UTC, CPU job23409411 verified that control23403857 and reader-low23403858
completed1000 updates, and rechecked their `selected_checkpoint.json` records.
It removed15 markers with exact matching agent-written text: two completed-run
root flags, five superseded control selections and eight superseded reader-low
selections. It removed **no checkpoint or other data**.

The following pairs retain their own `PRESERVE` markers:

- `hill_control/selected/update_00900`: current best development answer
  NLL0.5871496525.
- `hill_control/selected/update_00500`: exact checkpoint for the user's
  qualitative report.
- `hill_reader_low/selected/update_01000`: best development answer
  NLL0.6358946043.

Receipt: `TMP/FIRST_TASK_AWARE_FINETUNING/storage_retention/retention_completed_flags_20260921.json`.
The initial15-marker change deliberately left the campaign ancestor flag in
place, so that change alone did **not** make old weights eligible for pruning.
At19:21 UTC, CPU job23409520 then removed that empty agent-created ancestor
flag, after checking explicit flags on the active bank, encoder-low arm and
the three retained pairs, and adding one reports/frozen-material-root flag.
There are now no campaign-or-higher ancestor flags implicitly pinning the old
selections. The completed-arm upkeep therefore removed16 obsolete markers,
added one narrower marker, and deleted zero data bytes.
Receipt: `TMP/FIRST_TASK_AWARE_FINETUNING/storage_retention/retention_narrow_campaign_20260921.json`.

Current headroom supplied by the monitoring agent: ORCD cached pool98.7GB and
scratch772.9GB of1024GB each; local201GB used/284GB free. No deletion is needed.
The newest250GB floor still applies independently of flags. Local monitoring
reminders and the S3-only-important-selections rule remain in force; this pass
does not upload transient checkpoints or alter source, data, reports or
environments.

## Maintenance and source capacity — 2026-09-22 06:00 UTC

No pruning: cached ORCD quota at05:36UTC is pool330.0/1024GB and
scratch824.6/1024GB, leaving694.0GB and199.4GB. This machine has264.1GiB
free (219.8GiB used). No source, cache, report or checkpoint payload was
deleted or rehashed; reclaimed bytes: **0**. No EC2/CSAIL experiment writes
were active in the campaign state, so no additional remote storage inventory
was needed there. Existing PRESERVE/ancestor and newest≈250GB rules remain.

CPU job23458765 completed in1s. It read only the committed synthesis manifest
and explicit protection paths. Reports/banks, frozen comparison material,
prior selected pairs and initialized runs are protected. CPU job23458853
completed in1s and added exactly two agent-written active/queued markers:

- `oracle_9b_16k_r16/PRESERVE`
- `encoder4b_oracle27b_16k_r16/PRESERVE`

These markers protect each complete encoder/reader/resume-state directory.
After completion, narrow them only after preserving selected pairs; do not
remove user/ambiguous flags or bypass the recent-data floor. No training or
evaluation queue entry was changed. S3 remains for important selected pairs
only, not all transient checkpoints.

The05:59 synthesis snapshot has123k balanced questions published and123,530
accepted. All nine current source pools cover the remaining77k at observed
and recent yields. Code is the tightest:37,354 windows remain for22,983
needed accepted questions; minimum future yield0.615/window versus0.921
observed and0.923 recently. Keep its existing14,339-window extension.
Recheck source capacity at150k published, on code entering its extension, or
if yield materially falls. No new source collection or GPU run is needed now.

Detailed capacity: [SYNTHESIS_CAPACITY_20260922.md](SYNTHESIS_CAPACITY_20260922.md)
and [machine snapshot](SYNTHESIS_CAPACITY_20260922.json). Receipts under
`TMP/FIRST_TASK_AWARE_FINETUNING/storage_retention/`:
`headroom_20260922.json`, `retention_capacity_20260922.json`,
`registered_source_counts_20260922.json`, and
`retention_pending_flags_20260922.json`.

## Near-150k capacity and retention recheck — 2026-09-22 08:22 UTC

CPU job23465658 completed0:0 in2s using the existing metadata-only script.
The committed snapshot has147,000 balanced questions published,147,805
accepted including surplus, and53,000 remaining to the200k target. All nine
families cover that remainder at both lifetime and recent yields. Code is
still tightest:29,386 windows for15,677 required questions; minimum future
yield0.5335/window versus0.920 lifetime and0.919 recent. Keep the existing
14,339-window code extension. Recheck at180k, code's extension boundary,
or a material yield decline toward0.6/window. No source expansion now.

No pruning: cached quota08:06:56UTC is pool412.6/1024GB and
scratch824.6/1024GB; local availability261.1GiB. The13 explicit
protection paths all exist and have their own or an ancestor PRESERVE.
AC/reader pairs, the newest≈250GB floor, and report/source/environment
exclusions remain unchanged. Deleted bytes **0**; markers changed **0**.
No payload tree scans/hashes, source/code edits, new generation, GPU jobs,
training-queue changes or S3 archive writes. Only important selected pairs
belong in S3.

Updated [capacity report](SYNTHESIS_CAPACITY_20260922.md) and
[machine receipt](SYNTHESIS_CAPACITY_20260922.json). Point-in-time receipts:
`TMP/FIRST_TASK_AWARE_FINETUNING/storage_retention/retention_capacity_20260922_0822.json`
and `headroom_20260922_0822.json`; earlier06:00 receipts are retained.

## Trajectory campaign check — 2026-09-24

CPU inventory23617133 inspected the explicit checkpoint roots. It retained257.13GiB of the newest unflagged groups plus35flagged groups. Fourteen inactive September17–18 groups under ORCD scratch SYNC_RUNS are eligible (91.09GiB); no candidate is an active trajectory/QA run or the paired initializer. Execution23617978 is queued using the exact inventory and per-group fingerprint/flag rechecks. Reports, source, data banks and selected directories remain outside this deletion. Receipt: TMP/FIRST_TASK_AWARE_FINETUNING/storage_retention/orcd_manifest_20260924.json. Do not report reclaimed space until the execution receipt arrives.

Local inventory:8groups,2.50GiB retained, no candidates. Cached ORCD quota before cleanup:pool828.7/1024GB, scratch824.6/1024GB (03:09UTC). No EC2 nodes are in use. S3 remains for important checkpoint pairs only.

Execution23617978 completed all14groups:91.09GiB of logical checkpoint data removed, with a completed deletion ledger for each group. Actual quota headroom awaits the quota service refresh; this logical total is not a measured filesystem-space delta. Ledger: TMP/FIRST_TASK_AWARE_FINETUNING/storage_retention/orcd_deletions_20260924.jsonl.

At06:41UTC, refreshed quota was pool942.4/1024GB and scratch733.6/1024GB. CPU23631385 protected the active trajectory QA/compaction outputs. CPU23631699 then removed17 exact agent-written active-run flags on completed RAG experiments after verifying their selected checkpoint pairs already had narrower protection. No ambiguous or user-authored flag was removed.

The subsequent pool inventory retained255.64GiB of recent unflagged checkpoints and proposed only the9.009GiB historical `fresh_joint` checkpoint trees. Execution23631749 rejected its fresh eligibility/fingerprint recheck before deletion. **No pool data was deleted; re-inventory before any future attempt.** The rejection cause has not been established. Current headroom is sufficient for the immediate continuation; revisit before further large caches or many paired saves. The V11 preparation retained a6GB old-cache backup until its first successful GPU resume.


September24,10:29UTC: job23639745 completed the reviewed same-node pool prune. Removed only completed fresh_joint/joint AC_MODELS+LORAS (18.0185GiB logical); retained reports/banks/source, all protected pairs, and256.497GiB newest unflagged groups. Both begin/deleted ledger entries confirmed. Earlier cross-node attempt had stopped before deletion. Receipt TMP/storage_retention/retention_pool_prune_20260924_1000_receipt.json.
