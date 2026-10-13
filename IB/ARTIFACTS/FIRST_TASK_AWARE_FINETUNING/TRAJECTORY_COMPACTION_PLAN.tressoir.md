# Trajectory QA and compaction — 36-hour campaign

Approved September 24 at 01:22 UTC. End September 25 at 13:22 UTC. Implement the necessary features and run substantial task-aware experiments. Preserve completed RAG evidence and continue its 27B synthesis.

## Executive Summary

Two separate training branches start from the strong 42k-question, 2B → 9B RAG pair. Trajectory QA compresses an original-task history window selected **before** the future question exists. Fixed 27B synthetic answers supervise the reader. The official compaction target now compresses a recorded prefix and predicts a fixed 27B oracle handoff with SFT. The existing full-history continuation-KL run remains a control; summary recovery and actual continuation are evaluated separately.

Reuse rank-128 LoRAs, input rows and deep residuals at full-attention layers, no pooling, depth 1, initially 1/16 compression. Use ORCD and CSAIL, including RTX PRO 6000 and H200. The earlier exceptionalEC2RTX was terminated14:12UTC. A new boundedperformanceRTXspotallocation isauthorized15:24UTC foratmost3h; seePERFORMANCE_PLAN.md. Main runs proceed alongside bounded quality and throughput experiments. Publish useful insights around +8, +12, +16, +24 and +36 hours, including ongoing evidence when runs are incomplete.

## Requested Decisions

All campaign directions are approved. The user expects to be away for 8–12 hours; no further approval is needed for routine implementation, optimization or experiments within this scope. Use reviewer agents at critical boundaries, without repeated broad reviews.

Trajectory QA: **8k unique questions × 1 epoch now**, explicitly approved September24 at13:13 while the scalable100k-question bank grows;16k remains a later expansion. Compaction: **oracle-summary SFT is the official target**. Begin with512 distinct histories as a diagnostic and expand toward5k+ as the reusable summary bank grows. The existing50M-token continuation run is a control, not the new summary-training budget. Prepare up to 100k eligible cuts while reporting distinct roots separately. A 2k-example compaction result is an early diagnostic, not the training ceiling. Prefer fresh histories over repeated epochs.

## Milestones

### M1 — Material and fixed targets (Complete)

#### Planning Overview

Select chronological 8k, 16k and 32k history windows before synthesis; retain the original task, system prompt, tools and complete turns. Hold out source-qualified roots before making windows or questions. Begin with roughly 50% coding, 30% research and 20% math/tool-use histories, subject to availability. Sources: Open-SWE, S1-DeepResearch and Nemotron math tool-use trajectories. Preserve the loaders' existing exclusions; no new broad overlap audit.

Generate up to four distinct questions per useful long window with the 27B oracle, reasoning disabled. One batched grounding request checks that window's questions. Save portable JSONL releases and actual accepted source counts. Do not use benchmark QA labels or reference patches. The initial single-history task has naturally irrelevant turns; no extra AC-only distractors.

Compaction keeps the longest complete continuation up to 8,192 rendered tokens, within the existing 72k total limit. Supervise every retained assistant output. Report actual history, suffix and assistant lengths; requested bins do not imply every document fills them. Existing synthetic public-turn normalization is retained where necessary, and does not establish live environment replay.

#### Planned Changes

`activation/ac_model/ac_model_study.py`: construct an ordinary trajectory QA item from a preselected window and fixed question/answer. Share source framing between AC and full-context views. Keep the future question out of every encoded part. Preserve the existing public generator interface.

`activation/ac_model/ac_model_training.py`: add optional bounded target preparation with a separate immutable LoRA on the existing reader base. Persist teacher identity with partial and complete checkpoints; preserve optimizer, schedule and penalty RNG on resume. Keep existing eager callers compatible. No new utility file or model architecture.

**Validation so far:** seven real source fixtures; 19 QA windows, 17 continuation cuts, 41,321 assistant targets. Serial, parallel and cross-worker resume preparation match. Question isolation, source framing and exact target positions pass real-tokenizer checks. Six tiny-model fixed-teacher/cache/resume tests pass. The initial225-question synthesis check passed; all first56history/native token counts match and oracle prompts fit below36k. The producer now shares evidence prefill between generation and checking. Research material exhausted at5,500histories; report the actual accepted mix as coding/math material continues.

### M2 — Early GPU proof and live reports (Complete)

#### Planning Overview

Run a long-history backward pass after Adam moments exist. Save after actual updates, reload in a separate process, and verify teacher weights/outputs, optimizer moments, penalty RNG, both trainable sides and plain-base no-context targets. Source is frozen per launch. Do not overwrite active producer code.

#### Planned Changes

Private campaign scripts reuse the current trainer and HTML reporter. Freeze material selection, initialization, tokenizer and recipe. On resume, preserve discarded-attempt evidence separately and continue curves only from published weights.

Use fixed held-out and learning panels. Full-context and no-context references remain the initial reader; compaction adds equal-budget recent text and first-versus-later continuation NLL. Plot compaction KL separately, labeled as the top-32-plus-remainder approximation. Use consistent token weighting for compaction NLL, teacher NLL and KL.

**Completion evidence:** real2B→9B L40S gate passed4 Adam updates across2 processes, with fixed teacher/output and restored optimizer/RNG. The full runner completed8 examples with controls, decodes, reports and paired saves. The combined32,716-history+8,185-suffix case also passed on L40S with private Adam-moment parking. Larger GPUs are being profiled with resident moments and larger readout chunks for throughput. Per-target KL separates the first128 continuation tokens from later targets.

Report unique questions × epochs for QA; compaction examples × epochs, distinct roots and supervised tokens for compaction. Panels every 10%, with useful early probes. Keep HTML stable and update JSON; show explicit preparation, queued, running and pending-measurement states.

### M3 — Sustained training and optimization (Implementing)

Begin main runs as usable material arrives and the GPU gate passes. Freeze each run's planned budget before its schedule begins. Use one coherent training call/schedule and exact resumes: encoder/heads 5e-5, reader 2e-5, eight examples per update, 50-update warmup, cosine decay to 10%, joint clip 1.0, no-context base KL with weight 1 on 25% of examples.

Measure preparation, model computation, evaluation and checkpoint costs. Optimize tokenization reuse, CPU preparation, length-aware batching, teacher preparation and readout chunks while preserving the same data, targets and input lengths. KL microbatching is currently unsupported; add it only if profiling justifies it and matched gradients validate it. Main runs continue while bounded probes investigate improvements.

**Implementation progress:** the full 50M-token run started on an H100 at approximately 05:21 UTC. Its frozen selection contains 18,408 distinct training roots/cuts and 100 held-out cuts. CPU preparation took 11m49s. Compact integer storage then reduced measured cache memory from 39.6 to 13.8 GiB, with unchanged token values and targets. A real four-update runner passed a process restart using this cache. Independently repeated GPU runs differ slightly beginning at the first backward pass, so bitwise GPU equivalence is not claimed. The frozen 2,000-question QA diagnostic is now training on L40S. Its first attempt exposed a reader-checkpointing memory gap; enabling the existing all-length checkpoint option passed two full batches and a 32,735-token history after optimizer initialization, at 37.75 GiB peak. The question/history selection and schedule are unchanged. Live progress now flushes complete phase and training counters together. The RTX PRO 6000 combined gate and 32-example profile passed; H100 also passed. These private preparation changes do not alter the product handoff.

At07:07 UTC the main run checkpointed after912 cuts and2.61M supervised tokens. Its continuation resumed on H100 at08:05. QA automatically resumed after1,000 questions. A reviewed private CPU validation now checks all18,508 prepared examples once (236.8s on eight workers), then reuses that proof only for identical prepared objects and sufficient runtime model limits. Token vectors and item aliases were unchanged by the explicit cache migration. This removes repeated tokenization from GPU startup; actual GPU startup savings will be measured on the next allocation.

### M4 — Boundary insights and handoff (Implementing)

Publish progress and limitations at the agreed boundaries. Review approximately 100 answers or continuations from promising pairs, including failures and reference concerns. Offline continuation is not live task success. Attempt a small live tool-use check after offline learning works, using actual tool execution rather than incompatible recorded future returns.

Consolidate accepted changes into one current source-sync handoff against `/source`, with exact diffs and full changed files. Mark important paired checkpoints with `PRESERVE`; copy only important checkpoints to S3. A resumable compaction export must also include its immutable teacher snapshot directory.

The eight-hour findings are published in `TRAJECTORY_INSIGHTS.md` and the live overview. The initial100QA qualitative report is complete; compaction shows early offline improvement, while the completed2kQA diagnostic shows a modest held-out gain. Keep the strongRAG initializer: the original reconstruction pair performs poorly on the sameQA panel as loaded. The user subsequently authorized an8k×1epoch run immediately while the16k pool grows. This is an explicit budget choice, not a silent reduction.

A reviewed private HF live-tool probe is queued on six fixed fictional release packets. It retains the normal Agent loop and every deep intervention, using actual tool execution after a forced empty-summary cut. OrdinaryAgent/vLLM does not yet carry deep payloads; the probe does not establish that production integration. A shared-encoding optimization failed its first gradient gate and remains excluded from training; a bounded comparison now holds encoderbatch shapes constant.

The current source handoff contains13 files against untouched `/source`, including small compaction-turn, resumed-ETA and overlong-cache-filename fixes. GPU sources remain frozen per launch.

**Priority update, September24 11:39UTC:** the user prioritizes continuing and increasing training volume over the9am findings. Keep meaningful measurements, but do not interrupt training or spend substantial time polishing that scheduled write-up. Improve effective throughput where correctness can be preserved.

### M5 — Oracle handoff SFT and shared runtime prompt (Implementing)

#### Planning Overview

Approved September24 around13:10UTC. The user will be away for about3h and prioritizes reusable synthesis for compaction, trajectoryQA and RAGQA. Keep the prompt simple for small models and use the user's five headings. The shared124-word prompt now lives in `activation/common/ac_parts.py`; the runtime tool and compaction demand use it. Resumption instructions work with text, AC or both.

The27B oracle sees only the original framed history up to a fixed cut, never its continuation. It writes a Markdown handoff. The student's encoder sees that same precut history without the oracle output or its compact call. The summary is the SFT target. Use the selected1848-history compaction pair for the first diagnostic, freshAdam, the existing golden learning rates and plain-base no-context penalty. Keep the strong42k initializer as an optional matched-data comparator.

Publish immutable reusable summary shards and a small atomic manifest, with model/prompt/source provenance. Generate the original held100 first, then mixed coding/research/math histories. Start with512 training summaries when ready; continue generation toward all18508 eligible cuts. A failed formatting check stays pending rather than stopping progress on other histories. Quick checks and a few real examples suffice; no judge pass. Prompt changes do not invalidate previously accepted summaries.

Report summary learning/held NLL separately from original full-suffix continuation NLL, first128/later behavior and live tool-use checks. Freeze full-history/no-history/oracle-summary controls anew for this study; changed resumption wording means no bitwise replay claim against old curves. Panels remain10%, HTML stable and JSON live. Preserve the original held histories and disclose actual counts.

#### Planned Changes

`activation/common/ac_parts.py`, `activation/agent/agent_tools.py`, `activation/agent/agent.py`

```diff
+ COMPACTION_SUMMARY_PROMPT: short shared five-section handoff request
+ CompactionTool.description uses that request; summary remains a plain string
+ Agent.COMPACTION_DEMAND points to the format while requiring a lone call
- Resumption always claims an activation context is present
+ Resumption refers to supplied compacted state and unfinished actions
```

These are focused hunks; unchanged runtime segmentation and tool plumbing are omitted. New synthesis/training experiments reuse existing trainer/items in the existing bench area; no new utility module or architecture. Validate source/target separation with real-tokenizer CPU checks, then a real GPU training/save/reload gate. Keep frozen sources for running jobs.

**Progress (September24,15:04UTC):** The shared124-word prompt is wired into runtime and the15-file source handoff is validated. Real2B→9B summary training/save/resume passed, including optimizer/RNG/control/order and32k input; fixtures were disposable. The8k trajectoryQA run trained416 unique questions and published a paired checkpoint before requeue. The continuation control remains paused at3,888 histories. TrajectoryQA synthesis continues onRTX; the existing58,033-question RAG bank has a guarded ORCD handoff queued. The revised prompt plus Qwen3.8-27B native medium passed a quick four-case fidelity check, including contradictory evidence and original constraint scope. Bulk summary generation is running onL40S; an RTX handoff is queued and will stopL40S only after its compatibility probe succeeds. Earlier failed summary pilots remain preserved separately and unused for training. First summary training needs100 held +512 training summaries; its large files will live onORCD scratch, with reports in the usual synced folder.

## Monitoring and recovery

Use native clock waits during the active Codex turn and review progress at phase boundaries and approximately every 30 minutes. Slurm requeue and passive file transport are independent; they do not imply a persistent Codex reasoning wakeup after the turn ends. Use guarded existing SSH masters only. Routine polls: ORCD at least 10 minutes apart, CSAIL at least 50 minutes; action-specific checks remain permitted.

Periodically inventory and prune old unflagged transient artifacts, preserving the newest roughly 250 GB and explicit preservation flags. Preserve reusable question banks and their referenced histories. Do not blindly delete models, caches or frozen source.

## Report location

`/workspace/IB/TMP/SYNC/TRAJECTORY_COMPACTION/report.tressoir.html`

Subdirectories: `trajectory_qa/`, `compaction/`, `preparation/`, `trajectory_synthesis/` and named bounded controls. Existing RAG synthesis remains in `TASK_AWARE_QA/synthesis_bank_27b/`.
