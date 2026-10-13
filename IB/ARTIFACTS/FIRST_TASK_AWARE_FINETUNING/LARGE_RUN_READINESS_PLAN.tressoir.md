# Large-run readiness

Approved in chat September25: wind down non-synthesis jobs at recoverable boundaries, keep reusable synthesis progressing, and implement/test a large training launch on the data available tonight or tomorrow. Choose hardware from measured workload speed, allocation delay and restart reliability. No large run has been launched. September26: the user explicitly requires approval before starting large training.

## Executive Summary

The training pipeline can resume the remaining100M+ reconstruction warm-up, including the accepted recursive allocation, then use a frozen manifest of available QA, summary and plain-agent data. Incomplete banks are valid inputs; counts and effective task weights are explicit. Synthesis can stop at a committed shard and later resume independently. Keep existing parts/models/trainer and put execution complexity in existing utils/autotuners.

Task-aware ratios uniformly sample[1/20,1/10]; reconstruction[1/30,1/10]. QA includes direct contextual, recursive parent-context and uncontextual summary-prompt views. Working QA split40/40/20 and task loss shares25/25/25/15/10 remain recipe proposals; record actual selections and exposure before launching. Accepted reconstruction allocation is approximately80M flat +30M recursive targets within110M, with recursive50% mixed old rows/new text,25% transport,25% balanced two-stage. Already-trained targets count.


User addition September25: train multi-target reading from two or three separate AC inputs, covering reconstruction, both QA families and summary generation. Initial recipe allocates about10% of examples within these families to groups, reusing disjoint existing source examples and simple numbered requests with `---` response separators. Preserve each input's compression ratio/recursive layout; never truncate evidence to fit. Keep ordinary agent SFT unchanged. Track unique source questions/histories separately from grouped training examples, record actual achieved fraction and show separate multi-input held/learning/frozen-control metrics. A narrow held diagnostic should exercise correct answer-to-input association and gradients through every sibling input. No new synthesis is required. Multi-target examples are for familiarity with several inputs, not long-context stress: prefer shorter intact sources and cap the combined full-context-plus-target reader view at24,576 tokens. Longer examples stay in single-input training; independent capacity probes may remain longer.

## Requested Decisions

Implementation, hardware probes/tuning and at least two independent post-implementation reviewers are authorized. Hardware selection remains open until measured. Use ORCD/CSAIL and existing bounded EC2 window; Quanta account is now confirmed. The20-hour synthesis horizon ends September26 around12:10UTC. Do not silently extend EC2 beyond September25 18:20UTC.

## Milestones

### M1 — Wind-down and durable, sliceable input (Implementing)

#### Planning Overview

Pause active training through its native save path; verify pair/Adam/RNG/counts. Retire held untrained backups. Keep all synthesis with disjoint owners, committed text shards, interruption/requeue and snapshot freezing. Extend trajectoryQA to at least100k TRAIN questions; held rows are extra. Extend compaction/teacher supply as needed and move producers off EC2 before expiry. Preserve important paired checkpoints; keep complete data on MIT as local disk is constrained.

#### Planned Changes

Private launch/manifest/transfer controls under the existing task TMP directory, durable source preparation/launch entrypoints in this task's bench directory. Frozen mixture manifest records file checksums, split/root IDs, accepted quality selection, family weights, conditioning/layout/ratio and exact exposure. Avoid copying growing full banks per run. Snapshot from committed files only; payload before manifest, separate resumable writer state from portable collection.

### M2 — Correct and efficient distributed AC training (Implementing)

#### Planning Overview

One existing encoder/reader replica per GPU. Preserve global logical batch8, per-example weighted objectives, penalty sampling, Adam clocks and schedule. Shard bounded reference preparation; balance work by actual recursive encoder/reader/target lengths. One writer for panels/checkpoints. Coordinate failures/deadlines before updates, preserve unused gradients for plain-agent batches, verify native resume and validate disk headroom/retention. Avoid silent recipe changes to obtain speedups.

#### Planned Changes

`activation/ac_model/ac_model_utils.py`: distributed lifecycle, partitioning, gradient SUM, metrics, checkpoint/RNG coordination and preservation helpers. `ac_model_training.py`: opt-in distributed config and narrow calls into utilities. `activation/autotuners/{core,worker,__init__}.py`: workload-aware measured1/2/4 GPU comparison and topology/architecture-aware caching. QA/recursive reconstruction changes stay in existing study builders/utilities. Preserve default single-GPU behavior.

### M3 — RTX, H200 and GH200 validation (Implementing)

#### Planning Overview

Measure identical real mixed updates with populated Adam, forward/backward, clipping, synchronization and optimizer work. Targets>1.8× on2 GPUs and>3.5× on4 are acceptance targets, not guaranteed. Record end-to-end serial preparation/panel/save costs separately. Include long/recursive/plain/mixed cases and smallest batches/unused parameters. Report allocation delay and preemption exposure alongside speed; raw GPU FLOPs are not the ranking.

#### Planned Changes

Bounded compute-node probes under TMP/large_run_readiness_v1. Quanta H200: single node supports4 GPUs, NVLink confirmed, current x86 environment imports. Quanta GH200: one96GiB GPU per ARM node, needs separate environment;4 GPUs would be multi-node and require an explicit tested distributed transport. RTX multi-GPU PCIe topology must be measured. GPU-memory labels and CPU/shared-memory sizes must remain distinct. Check user quota as well as filesystem free space inside compute jobs.

### M4 — Two independent readiness reviews and handoff (Implementing)

#### Planning Overview

After implementation, launch at least two independent reviewers: one for numerical/training correctness and performance; one for data, orchestration, disk quotas, checkpoint integrity/recovery and avoidable launch failures. Give actual diffs, measured results and launch packet. Resolve findings, rerun affected checks and obtain re-review. Keep user quota in mind; use substantial bounded reviews rather than repeated small ones.

#### Planned Changes

Publish reviewed launch command/config, frozen source manifest, current hardware ranking and evidence, tested pause/resume path, data snapshot procedure, remaining limitations and report locations in a handoff. Keep main live HTML stable and update JSON. Do not describe pending benchmarks or reviews as completed.

## Current evidence — September26 11:58UTC

Two independent implementation reviews, affected re-reviews and actual two-H200 launcher pause/resume passed. Current source hashes still match the reviewed20-file handoff. Strict numerical checks now pass on both two and four H200s. Cost-window128 measured1.817× on two and3.128× on four; the four-GPU3.5× target remains unmet. Original-order fourH200 measured2.665×. RTX stress passed on two at1.705×; its four-GPU attempt stopped on the unchanged native-repeat gradient check before measuring four GPUs. No thresholds were relaxed. Causal replay isolated later cross-world trajectory drift to tiny summation-roundoff sensitivity, with correct equal-state objectives/gradients and identical replica states.

Data now includes100k RAG questions,100,564 training trajectory questions,65,824 training summaries and approximately64k teacher trajectories before SFT filtering. Reconstruction remainder preparation is complete; the final full mixed-bank snapshot/materialization is still to be committed. Raw-source extension failed after publishing303,434 raw trajectories; its shutdown repair did not run during the harness interruption. Existing training data remains available. Detached synthesis and copy jobs continued, but active agent monitoring was interrupted. All EC2 spots remain terminated; only two MIT teacher synthesis GPUs are active at the live check. Large training remains on explicit approval hold. See [the implementation handoff](LARGE_RUN_IMPLEMENTATION_HANDOFF.tressoir.md).
