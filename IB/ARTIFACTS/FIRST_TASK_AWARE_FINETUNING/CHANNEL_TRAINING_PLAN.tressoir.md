# Channel warm-up, reusable data and mixed training

Approved September 24, 2026. This plan supersedes the earlier campaign's remaining optimization priorities. The goal is to settle training design, produce substantial checkpoints, and move to harness/model co-design while reusable data production continues.

## Executive Summary

Use the existing 2B encoder / 9B reader architecture and proven learning rates. Run reconstruction as an explicit **pre-mixing channel warm-up of at least 100M supervised reconstruction tokens**. Task-specific diagnostic branches may continue from existing strong compatible pairs while this warm-up runs. The main mixed branch starts from a compatible warmed pair; checkpoint weights are not averaged across branches.

| Phase / bank | Working target | Completion or promotion rule |
|---|---:|---|
| Reconstruction warm-up | 100M+ reconstruction target tokens | Pre-mixing phase, extend while useful; retain the held-out reconstruction gate and report actual learning |
| Trajectory QA evidence | 16k unique questions × 1 pass; matched 4k alternate-mode comparison | Meaningful held-out learning and useful outputs; counts alone cannot establish scaling |
| Compaction evidence | 8k distinct history-summary pairs × 1 pass | Summary recovery and actual continuation reported separately |
| Question-blind RAG evidence | 8k unique questions × 1 pass | Reuse existing question bank; third priority behind trajectory QA/compaction |
| RAG bank / first large mix | 100k unique questions | Keep background synthesis progressing |
| Trajectory QA bank / first large mix | 100k unique training questions; held questions additional | Include direct contextual, recursive contextual and uncontextual presentations |
| Compaction bank / first large mix | Bank grows toward 100k; 32k is an initial mixed-training budget, not a generation ceiling | Use spare GPUs to keep growing past 32k |
| Ordinary agent teacher bank | Toward 100k trajectories, interruptible and sliceable | Initial draw 5k; first mixed SFT target 20M assistant tokens; retain useful partial bank |

The 16-hour targets are planning targets needing approximately four productive strong GPU slots and overlapping synthesis/training. No guarantee of quality improvement follows from sample count. Establish the design by roughly36h when evidence supports it; use the remaining72h window for training, evaluation and bank growth. Do not wait for every asymptotic bank target before starting useful runs.

## Requested Decisions

All directions above are approved in chat. The user may be absent for4–16h or longer. Proceed within compute authorization; no routine reconfirmation. Preserve both question-aware and question-blind QA. No unrelated hyperparameter sweeps. The user subsequently authorized multi-GPU trainer/autotune work, targeting measured speedups above1.8× on2 GPUs and3.5× on4 GPUs for identical training work; these are validation targets, not achieved results. Review agents only at consequential, error-prone boundaries.

## Milestones

### M1 — Small interfaces and correctness (Implemented)

#### Planning Overview

Add `encoder_sees_question` to RAG and trajectory QA item generation, preserving current defaults (RAG true, trajectory false). The reader always receives the question. The original boolean implementation exposes the question at every encoding depth when enabled. New requested behavior: uncontextual inputs carry a short general summarization instruction, not an empty prompt; recursive contextual inputs carry the question only inside the compressed parent context. Never inject the answer. These refinements are pending implementation/validation; previous completed runs retain their original framing. Cached synthesized questions remain reusable. Prepared training identities bind the mode.

Ordinary agent SFT uses the AC trainer with no AC parts. Preserve teacher thinking, assistant text and tool calls as supervised targets; user/tool returns are conditioning only. Such examples update the reader LoRA, not encoder/heads. Existing harvesting compresses plain prefixes and requires identical tokenizers/templates, so add explicit text-only harvesting and student retokenization when needed, without replacing the trainer.

#### Planned Changes

`activation/ac_model/ac_model_study.py`: QA boolean plumbing and text-only rollout harvesting. Existing utilities own rendering/tokenization logic where needed. Keep defaults compatible. Private CPU contracts and one real mixed GPU backward/save-resume check establish absence/presence of question conditioning and parameter gradients.

### M2 — Scalable reusable synthesis (Implementing)

#### Planning Overview

Use independent slices with immutable source selections, stable IDs and isolated writer directories. A new GPU claims an unowned slice; completing or pausing another worker is unnecessary after the one-time legacy-bank transition. Never run two writers on one bank or resume a producer from an aggregate manifest. Requeue unfinished slices and append completed immutable shards to a read-only aggregate.

27B compaction uses the current short official prompt, native medium reasoning and existing quick validation. Preserve original held100. New slices contain only training roots and retain pending/rejected eligible entries for bounded retries. The initial18,408 selection is not a supply ceiling; expand material/selection without changing fixed held roots.

Single-agent teacher data reuses the existing campaign with27B medium. Force BASE-only weights, disable the subagent tool and remove delegation instructions. Retain thinking spans. Use a supported sandbox on each host; do not assume Podman exists on MIT.

#### Planned Changes

Private launch/partition/export helpers in the existing campaign working directory; no new model abstraction. Canonical teacher entrypoint is reused through a narrow configuration wrapper. Synthesis prompt/code revisions are provenance, never a reason to invalidate accepted examples.

Publish self-contained **text-only JSONL shards** (optional gzip) containing actual histories/passages, questions/answers or summaries, tools/assistant thinking where relevant, stable source/root IDs, split and recipe provenance. Include a README, schema/version, checksums, counts, source attribution and completion status. No pickle, tensor paths, absolute cache paths or original source checkout is required to consume the release. Publication-ready data collection does not itself authorize public release. Preserve banks on persistent local and MIT storage; S3 policy remains important paired checkpoints only.

### M3 — Warm-up and substantial task runs (Implementing)

#### Planning Overview

Reconstruction100M+ is a labeled pre-mixing warm-up phase, not a silent change to the historical50M Golden description. Sample the agreed1/10–1/30 range; count target tokens across ratios for one model pair. Continue an architecture-compatible reconstruction pair; preserve task checkpoints separately.

Launch substantial task runs as soon as enough verified data exists. Growing-bank consumption freezes each run's slice and planned schedule; checkpointing and later continuation preserve declared exposure, optimizer/schedule state when intended, and frozen controls. Never silently change a running selection or label duplicated QA views as unique questions.

#### Planned Changes

Reuse frozen-source training launchers, ordinary SFT and existing resume machinery. Keep input/history lengths unless capacity forces a disclosed adjustment. Report unique questions/histories × epochs, supervised tokens and wall time, with10% quality panels and stable HTML/JSON-only refreshes. Compaction summary quality, continuation quality and live agent behavior remain distinct measurements.

### M4 — Mixed training and design freeze (Planning)

#### Planning Overview

Mix RAG QA, trajectory QA, oracle compaction, reconstruction rehearsal and ordinary agent SFT through the AC trainer. Both question-visible and uncontextual modes are required. Working proposal after the recursive-context discussion: for each100k-question QA family,40k direct contextual,40k recursive contextual and20k uncontextual presentations in one pass. Fractions are proposed, not an already validated optimum; repeated views never count as additional unique questions. Bank targets and training exposure are separate; repeated questions remain one unique question. Use available balanced subsets before the full banks complete. Retain reconstruction in the mixed run as rehearsal; do not require replaying100M tokens before every mixed checkpoint.

#### Planned Changes

Freeze one explicit mixture manifest, task weights, train/held IDs, initialization and schedule before the mixed launch. Choose the first concrete mixture from measured target lengths and retention checks, without a grid search. Preserve strong paired checkpoints and continuation state. By roughly36h, record either the supported canonical recipe or the specific unresolved failure; do not keep adding optional experiments.

## September25 refinement — actual agent recursion and larger mix

The user increased the first trajectory-QA mix from64k to100k unique training questions (held rows do not count). RAG remains100k. Retain32k summary histories,20M plain-agent assistant targets including thinking/tool calls, and10M reconstruction rehearsal targets as the initial working budgets. Candidate explicit loss shares are25% RAG /25% trajectoryQA /25% summary /15% agentSFT /10% reconstruction; these are proposed and must be frozen with actual exposure counts before launch.

`Agent.nested_in_context` places a `parent_context` part before new search/tool evidence. Train that actual layout: parent trajectory containing the question → recursive encoder rows; those rows plus fresh evidence → outer encoder; output rows plus the reader's ordinary question → answer. Do not duplicate the question as outer-encoder text. Existing study depth2 recompresses the evidence and repeats question text; it is not this training case. Reuse existing parts, recursive readout/adapter and trainer. Build valid prefixes/evidence from recorded histories, stop before target answers, and retain complete matched textual evidence for frozen full-context controls. A minimal question-bearing parent can bootstrap RAG; report it separately from real parent histories. Do not splice unrelated trajectories merely to manufacture nesting.

Use simple general summarization instructions for uncontextual QA, never the held-out question. Keep mode/layout metadata and separate held panels. Check gradients reach the recursive readout/adapter and both encoder passes, then make a small matched learning comparison before expanding recursion. Count each recursive level's actual encoder work and row budget; do not multiply ratios as if all fresh evidence were compressed twice. Full flat summary SFT remains the official compaction target. Dedicated continuation training is retired unless a cheap diagnostic can answer a specific unresolved question without blocking main work; continuation evaluation stays.

Multi-GPU implementation belongs primarily in existing `ac_model_utils.py` and `autotuners/`, with narrow trainer plumbing. Preserve the global example batch, per-item weighted loss, penalty draws, populated Adam state and schedule. Coordinate rank-local failures/deadlines, shard reference work, and use one report/checkpoint writer. Validate1/2/4 GPU correctness and timed actual updates, including communication and imbalance; report whole-run scaling separately. Do not claim the requested scaling before measuring it.

## Twenty-hour synthesis window and accepted recursive reconstruction

Accepted in chat September25: prioritize20 more hours of compaction and ordinary-agent synthesis, through approximately 2026-09-26 12:09 UTC. Maintain disjoint slices and auto-requeue; extend prepared inputs as needed. Existing EC2 expiry remains18:20UTC today; sustain the later window on ORCD/CSAIL rather than assuming an EC2 extension. The working forecast of30–40k additional examples per task requires roughly two productive strong producers per task and adequate eligible source supply, not merely queued jobs.

Accepted reconstruction allocation, not yet switched into the running job: about80M flat +30M recursive target tokens within the existing110M budget. Tokens already trained count toward the flat allocation. Recursive share:15M mixed old-rows/new-text,7.5M whole-passage transport,7.5M whole-passage two-stage compression. Retain the overall final-rows/original-source-token range1/10–1/30, no content pooling, and unchanged literal gold SFT.

At an overall1/16 ratio: transport uses approximately(inner1/16,outer1); two-stage uses approximately(1/4,1/4). The main mixed case encodes an earlier25–75% span, initially at1/4, then places those rows beside the remaining fresh text in the outer encoder. Reconstruct the entire original passage from only the final AC rows. For a4096-token passage split equally,2048 old tokens→512 child rows; combine with2048 fresh tokens→2560 outer inputs; output256 rows, so outer ratio1/10. This remains overall1/16. Vary the old-span length; a minority of mixed cases can use an inner1/16 once transport learns, without adding a ratio grid search.

Compute the actual outer ratio from the desired final row count and tokenized outer input including child rows/chat wrappers; record rounding and min/max-row caps. Do not label a capped intermediate as1/4. For overall targetR, all-child transport is(R,approximately1), balanced is(sqrt(R),sqrt(R)); mixed text obeys outer≈R/((1-f)+f*inner), wheref is the recursively encoded source share. Avoid1/16→1/16 for exhaustive reconstruction at this stage: it changes the all-child target to1/256. Existing segmented reconstruction inputs already represent these layouts, so no new architecture or synthesis is required.

Use flat/transport/two-stage/mixed held panels, with old-span and fresh-span NLL for mixed examples. Ensure gradients traverse both encoder calls/recursive heads, and retain greedy checks. Introduce this as a declared continuation phase from a saved pair with explicit remaining source/layout identity; do not silently alter the active run's prepared inputs or recount previous targets. The user approved this reconstruction allocation/layout and the20-hour synthesis horizon. Implementation remains pending; the active flat run is unchanged.

## Accepted ratio sampling

Reuse `ActivationContextStudyGenerator._sample_ratio`, uniformly sampling the rows/input-token ratio itself. Task-aware RAG QA, trajectory QA and summary SFT use the interval[1/20,1/10]; reconstruction keeps[1/30,1/10]. No discrete weighted mixture or special mass at1/16. Freeze the sampled ratios in each prepared selection so resume and comparison are reproducible; ratio changes need no new synthesized answers. Keep the main held progress panels fixed at1/16, with matched endpoint checks. For recursive reconstruction, sample the overall final-rows/original-source-token budget and derive the inner/outer ratios for the accepted layout, accounting for actual templated lengths and row limits. Do not independently sample every nested level from this range. Uniform task-aware sampling averages0.075 rows per source token,20% more than fixed1/16 before rounding/caps; wall-time overhead remains to be measured.

## Resource and waiting policy

Reserve training capacity whenever ready data exists. With2 strong GPUs: one training, one critical compaction producer. With3: two producers while supply is short, then two trainers. With4: two trainers/two producers initially, then three trainers/one producer when bank headroom permits. Below2–3h of unused training data, boost critical synthesis; above4–6h, favor training. More than one otherwise free GPU enables additional compaction/teacher slices toward100k. Background trajectory/RAG synthesis continues on weaker available GPUs when off the critical path.

ORCD/CSAIL preferred. Existing authorization: two steady EC2 spots with limited bursts up to four until queues normalize. Existing node expiry stays September25 18:20UTC unless authorization is explicitly extended; new bursts get bounded expiry too. Protected workspace instance i-0918b8f73e9a0fe03 is never touched. Respect independent SSH masters, guarded access and login-node workload rules. Leave held/reserved user workloads alone.

Use30-minute active waits with early events for job termination/requeue, stalled progress, consequential queue delays, low disk and completed jobs needing successors. No model reasoning inside the wait loop. User input can interrupt the active wait. After a finished/disconnected session the watcher alone cannot reactivate the model; jobs and scheduler requeues remain independent. On resumption, inspect actual jobs before launching anything.

## Restart and preservation

Read `IB/STATE.md`, this plan and `CHANNEL_TRAINING_CAMPAIGN.json`; inspect job IDs/ownership and committed manifests. Resume existing work before creating successors. Each producer has a unique slice, bank and lock; a read-only aggregate is never a writer checkpoint. A training resume needs the AC/reader pair, optimizer/RNG/schedule, prepared identity and frozen references. Keep PRESERVE-marked outputs and the newest approximately250GB per storage allocation; prune only eligible inactive older data. Preserve portable text releases independently of transient caches.

Use `/workspace/IB/TMP/SYNC/TRAJECTORY_COMPACTION/report.tressoir.html` for the live overview. Report stages honestly as queued/preparing/warming/training/exporting; stale snapshots carry timestamps. Important checkpoints go to the existing `general-bucket-aln/activation-context-data/checkpoints` S3 location after paired verification. This plan and its manifest must also be copied with remote campaign controls so an agent-container loss does not erase the objective.


Execution state is maintained in `CHANNEL_TRAINING_CAMPAIGN.json` and `IB/STATE.md`. These record actual job owners, prepared token counts, completed quality panels, independent synthesis slices and restart instructions. Live counts are operational progress, not quality gates.
