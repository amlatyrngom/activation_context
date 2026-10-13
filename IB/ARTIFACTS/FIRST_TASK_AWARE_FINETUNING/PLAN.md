# First task-aware AC fine-tuning

Run a compact, fully synthesized RAG QA study from P2, with useful live measurements and simple controls. The [frozen Golden recipe](FROZEN_RECIPE.md) remains the architecture/training contract. Checkpoint archival and the previous source handoff are complete; implementation and experiments are green-lit.

## Executive Summary

**Qwen3.5-2B encoder → 9B reader; P2 initialization; 1/16 compression; depth 1; minimum 2 rows; no content pooling.** Keep the learned heads, rank-128 adapters, full-attention residual interventions, Golden learning rates and base-reader no-context penalty. Train with literal gold-answer SFT using synthesized answers. Each arm gets one ordinary `train()` call and a fresh QA optimizer/schedule.

Use raw code, documents, tables and tool outputs. Generate new questions with a few source-selection modes and prompt presets, then add BM25 distractors. Existing benchmark questions/answers never supply training supervision or generation prompts, even when their upstream split is called “train.” Shared raw corpora are allowed. There is no native-QA training branch in this work.

| What the run shows | Behavior |
|---|---|
| Learning | Per-update SFT loss, fixed training-panel answer NLL and separate no-context KL. |
| Held QA | Current-reader AC / full context / no context on the same fixed questions and targets; full context includes the identical distractors. |
| Progress | Baseline and every **10% of planned question presentations**, including 50% and final; primary progress is unique questions × completed passes. Optimizer counts remain technical provenance. |
| Inspection | Fixed decoded answers at baseline, midpoint and final, with retained earlier samples. |
| Comparisons | P2 joint training; P2 with reader frozen; a fresh joint baseline if capacity permits. |
| Resources | Current overnight campaign: at most four autonomous MIT GPUs including synthesis; reserve at least two user slots; no EC2. |

P2 achieved held reconstruction NLL **0.106** after about 48M tokens. Its historical encoder LR was 1e-4, and it needs the existing readout migration when loaded today. P2s5 matches LR 5e-5 but scored 0.390 on less data. Both are archived in `s3://general-bucket-aln/activation-context-data/checkpoints/reconstruction-20260921/`. [Checkpoint details](CHECKPOINTS.md). A quick real restore/update/save check remains necessary; reproducing the original reconstruction study is not a prerequisite.

Reconstruction's NLL <0.2 gate is not a QA target. Full context provides a reference for answer quality, not a guaranteed ceiling. A pilot result applies to the selected 2B→9B pair.

## Afternoon extension — approved Sep22

The user replaced the 08:00 EDT target with **about16:00 EDT /20:00 UTC**, with meaningful results at **09:00,12:00 and14:00 EDT** (13/16/18UTC). Continue active native monitoring, at most four autonomous MIT GPUs, reserved user capacity and noEC2.

1. **Generation:** stop9B synthesis at its published164k questions, preserve its portable data, and resume27B synthesis toward200k using the existing batched/autorequeue producer. Readiness for training depends on eligible family counts, not reaching the whole generation target.
2. **Resolve LR:** compare AC peakLR2.5e-5 with the completed5e-5 controls at both1/16 and1/8, each on the same16k27B questions×1epoch. Keep readerLR2e-5, initialization, order, penalty, batching and schedule unchanged. Use matched development panels, learning NLL and preclip AC/reader norms; ignore known mixed RMS/cosine diagnostics. These auxiliary trials are outside the originalfive-arm confirmation. Prefer full runs under the extended window.
3. **Start32k promptly:** after an LR/ratio choice supported by those comparisons and enough eligible27B data, launch32,000uniquequestions×1epoch with the2Bencoder/9Breader from originalP2. Retain the original16ktraining examples as a subset, the fixed400development questions,32learning questions and frozen retrieval pool. Exclude all existing reserved questions and entire synthesis requests. Reserve fresh confirmation questions before the added-training selection. Reuse the existing input builder's indexed-ID intersection so new required sources need not belong to the distractor index.
4. **Reporting/runtime:** one training call,10%panels, actual input/rowbudgets, paired saves and source-grounded final answers. Measured16k runtimes suggest32k takes about5.5–6h at1/16 or6.5–7.5h at1/8; input preparation, evaluation and queue availability affect finish time. Choose the tested ratio on quality/stability evidence, with1/16 the current practical baseline. Do not claim exact convergence or an untestedLR optimum.

5. **Fresh data versus another epoch:** once the requested 32k run is allocated, use an available MIT slot for **16,000 original 27B questions × 2 epochs**, with the same selected LR/ratio, original P2 initialization and fresh optimizer. Both runs see 32,000 question presentations, use one train call, the same 50-update warmup/cosine schedule, fixed 400 development questions, 32 learning questions and distractor index. Report actual tokens and time as well: equal questions do not guarantee equal compute. Compare fixed final pairs on the fresh reserved confirmation after both finish. This replaces the older optional 8k × 2 control and has priority over a speculative group-clipping follow-up. It must not delay the requested 32k launch.

The32k data extension uses published immutable batches, fixed family quotas and a small containment/reservation/input check. No nativeQA, reconstruction-overlap audit, new public abstraction or newutils file. The originalfive-arm final confirmation remains unchanged and does not guide these tuning choices.

Current status (Sep22 16:00 UTC): both lower-LR trials are complete and favor the original AC5e-5/reader2e-5 recipe. The32k×1 run is active;16k×2 has resumed its exact7,200-question optimizer checkpoint after ORCD preemption, with200unsavedquestions replayed. Initial baselines matched exactly. Separate clipping is active at1/8.27B synthesis is paused cleanly at53k published questions;9B remains paused. The newly authorized trained-piecewise comparison below is using the original batched encoder after rejecting an optional mask optimization; original-path timing is queued in the freed slot. Original five-arm confirmation and all six context-transfer screens are complete.

## Piecewise encoding trained from reconstruction — approved Sep22 15:20 UTC

The user requests independent piece-by-piece encoding **trained in that layout**, with batching optimized, and explicitly selects the **original reconstruction P2 pair**, fresh QA optimizer/schedule. This supersedes the earlier decision to leave separate-passage training unlaunched. The53k published27B questions are sufficient for now: synthesis stopped cleanly0:0 at15:21UTC, with accepted surplus/cursors preserved;200k is deferred until more data is needed.

| Setting | Matched experiment |
|---|---|
| Training | **4,000 unique27B questions ×1epoch**, exact source-family proportions; original32 learning questions retained. |
| Evidence | **8k-token soft budget, up to16 passages**, whole required sources retained. One query-conditioned AC part per passage versus merged control. Actual piece lengths/counts are reported; no new chunker. |
| Fairness | SameP2 pair, questions/order/retrieval, literal targets, reader,1/16 total row budget, LR5e-5/2e-5,8questions/update, original joint clipping, no-context penalty and full-run schedule. |
| Evaluation | Same400 development questions,10%panels and32learning questions; fixed100 decoded final comparisons with frozenP2 full/no-context on identical8k evidence. Add32k evidence transfer readout after completion; label it transfer rather than training-length performance. |
| Batching | Existing encoder already flattens/batches independent parts. The optional no-mask/checkpoint-threshold patch failed the preset real-GPU gradient-equivalence screen and is **not adopted**. Both arms use original `train_ws_night_v1`, unchanged masks/checkpointing and existing65,536 token batch budget. A short original-path fit/timing check precedes launch. |
| Resources | Freed synthesis slot first; merged control takes clipping's slot after its final readout. Keep four activeMIT GPUs maximum, noEC2. GPU feasibility/timing precedes training; do not interrupt the32k/data-reuse experiments. |

4k×1 is a first controlled architecture comparison at4× the prior evidence budget, not a data ceiling. Use measured throughput for the launch reservation; complete full passes even if the new request extends beyond the prior afternoon target. Existing frozen code/checkpoints/material remain immutable. Record the new variant separately; no new public abstraction or utils file. Do not use reserved confirmation to tune this experiment. No additional approval is required; initialization was resolved in chat.

## Current approved next wave — 16k questions and 1/8 compression

The completed 2k×4 /4k×2 /8k×1 study favors fresh supervision. The next priority is **16,000 unique questions ×1 epoch**, starting from the same P2 pair. On the user's Sep21 request, add **1/8 compression** alongside 1/16: twice the nominal rows per encoder token. This is a QA capacity experiment beyond the frozen reconstruction recipe's 1/10–1/30 range; it does not change that recipe.

| Synthesized supervision | Control | Added capacity arm |
|---|---|---|
| 9B bank | 16,000×1 at 1/16 | Same 16,000×1 at 1/8 |
| 27B bank | 16,000×1 at 1/16 | Same 16,000×1 at 1/8, as capacity permits |

Within each row, keep exact training questions/order, held questions, retrieval/distractors, initialization, optimizer, learning rates, penalty and planned schedule identical. Both supervision sources share separately plotted 200-question 9B and 200-question 27B development cohorts, plus separate reserved confirmation cohorts. The supervision-source comparison measures the recorded generation pipelines, not model size alone. Evidence remains the existing soft 2k budget/up to3 distractors; do not conflate more AC rows with longer source input.

Use the existing private runner's `--ratio 0.125`; no architecture or product changes. Report actual encoder tokens, AC rows, answer NLL, learning NLL, no-context drift, memory and GPU time, with panels every10% and decoded comparisons for promising pairs. Equal question presentations are the primary comparison; compute cost may differ. If a run is interrupted, retain paired recovery state and report actual completed passes.

The material is now frozen from the available 9B bank and 18k balanced 27B release: each training arm gets 16,000 questions, with shared 200+200 development and separate 200+200 confirmation cohorts. The first 27B-supervised 1/16 arm is resubmitted on ORCD (job 23447593, following a14-second startup preemption of23446872): job 23444931 failed while publishing its initial baseline, before any optimizer update. A reporting-only fix keeps nested cohort metadata in JSON and cohort curves while excluding it from the flat table; real-reporter regression checks pass for both encoder runners. The completed reference cache is retained. All five arms are now scheduled: the9B/r8 arm follows completion of the27B bank,9B/r16 follows27B/r16, and4B follows27B/r8. These dependency chains reuse GPU slots within the four-autonomous-MIT-GPU cap, including synthesis; retain at least two slots for user requests and use no EC2. The1/8 arms take priority over the optional8k×2 control and new small probes. The existing32k adaptation and its required short/long cross-evaluation continue. The original Sep22 12:00UTC target is superseded by the afternoon extension above.

The optional8,000×2 control uses the exact runner-ordered half of the new9B16k pool, common held/retrieval inputs and a shared32-question learning panel. A real-bank CPU check passed for that learning-panel containment. It is lower priority than the1/8 request. The frozen32k passage-layout screen completed: merged NLL0.8758 versus separate0.8019 at identical100 questions/evidence/targets and205,886 rows, with2.91% extra encoder tokens. Only48/100 improved, the paired interval crosseszero, and12 decoded answers contain regressions. The expanded 100-question comparison has completed all 400 answers. On the original 88-item reference set, merged encoding yields 51 fully supported answers and separate encoding 47 (frozen full context 85, no context 12). That frozen screen retained merged encoding; the newly authorized trained-piecewise comparison above now supersedes the earlier no-training decision. The [complete 100-question report](PASSAGE_LAYOUT32K_COHORT100.tressoir.html) discloses three additional ambiguities introduced by longer evidence and the corresponding denominator sensitivity. [Result and limitations](PASSAGE_LAYOUT_32K.md).


### Fixed-final confirmation

Evaluate the four 2B arms and the 4B arm only after all five planned final pairs have completed 16,000 questions ×1 epoch. Use the separately reserved 200-question 9B and 200-question 27B cohorts; compare arms within each cohort. Each arm retains its own compression ratio and final paired reader, with identical full/no-context evidence and targets. Confirmation will not select checkpoints or guide subsequent tuning. The private evaluator reuses the existing confirmation input builder; focused CPU checks and an independent correctness review passed.

### Observed synthesis formatting defect

Before freezing the new16k comparison, exclude question/answer strings with ASCII controls other than newline/tab. A concrete LaTeX/JSON escaping bug affected342/67,000 published9B rows and78/12,00027B rows in a quick scan, chiefly math. Apply the same fixed rule before training/development/confirmation selection; retain family quotas and record excluded counts. Keep portable originals, source text and prior frozen results intact. Do not heuristically rewrite gold. Focused checks preserve normal multiline/tabbed code while rejecting backspace/form-feed/carriage-return corruption. This is a specific observed input-format defect, not a new overlap-audit effort.

## Larger encoder follow-up — authorized Sep22

If a compatible trained checkpoint is accessible, compare a **4B encoder →9B reader** with the current2Bencoder setup on a large fixed QA pool. The selected candidate is P4: roughly48M reconstruction tokens, rank128/no pooling/full-attention residual architecture, held reconstructionNLL.323 versusP2's.106. The complete pair and existing readout migration passed a real GPU restore, training, evaluation and paired-save check. A second check completed two updates on eight of the longest available 27B source windows (5,387–6,114 tokens), including backward with Adam state allocated, at microbatch 2 and 61.06 GiB reported peak (job 23442317). This establishes fit for those measured inputs, not QA quality or arbitrary longer inputs. Different starting checkpoint quality must remain explicit; this is a comparison of trained pipelines, not an isolated encoder-size claim.

Prefer the same16,000-question27B supervision pool at1/16 and the matchedP2 control, holding QA data/evidence/optimizer/hyperparameters fixed. Record baseline and10%panels, learning/held quality, actual rows, memory and speed. The user specifically permits this4Bexperiment to be scheduled **after Sep22 08:00EDT on ORCD**. This was the initial exception; the later afternoon extension above now applies to the whole campaign. Keep the samefour-autonomous-GPU limit, reserved user slots, and noEC2. The large run is queued as ORCD job 23446884, following the 2B/27B-supervision/1/8 arm in the same GPU slot. The common 27B material is frozen. Initialize from originalP4, never from smoke-test weights; use a7h Slurm bound and6h trainer ceiling. [Feasibility and evidence](4B_ENCODER_FEASIBILITY.md).

## Requested Decisions

The user has selected unconditional synthesis, simple study options, selective multiple-question generation, quick validation, 10% reporting, the P2 joint/frozen-reader comparison and a fresh baseline if feasible. The earlier one-instance EC2 permission is superseded by the current MIT-only campaign: no new EC2, at most four active autonomous GPUs, with user capacity reserved.

The user green-lit this amended plan on 2026-09-21. The mix below is the working recipe; use the available raw sources, record actual proportions and freeze material before training. Do not delay the first pilot to acquire a particular named dataset.

The reconstruction generator will default to seeded uniform row ratios in [1/30, 1/10]. Implement/test this without launching reconstruction training. The later 50M-token campaign's accounting remains outside this task.

## Milestones

<details class="card" data-tressoir-markdown>
  <summary><span class="card-title">M0 — Checkpoints and current-code handoff</span><span class="card-oneliner">Completed; preserve the existing artifacts.</span><span class="card-badge">Completed</span></summary>

#### Completion report

P2/P2s5 paired checkpoints were archived with verified SHA-256 checksums, about 4.911 GB total. The archive excludes public base weights, optimizer moments and duplicate latest folders. [Manifest](CHECKPOINT_ARCHIVE.json).

The [current-code handoff](HANDOFF.md) contains 30 changes against its captured source baseline, including the required existing serialization caller migration. The source application is recorded as commit `af52a4c`; its checks included AST/patch checks and 43 focused private CPU checks, not a new GPU regression. Preserve these files as the pre-task-aware snapshot.

#### Planned Changes

`current_code/source.patch · completed source snapshot`

[Full captured diffs](CURRENT_CODE.tressoir.md). The next implementation gets a separate incremental handoff.

</details>

<details class="card" data-tressoir-markdown>
  <summary><span class="card-title">M1 — Programmable synthesis and faithful evidence</span><span class="card-oneliner">Small options on existing objects; one question per chunk by default.</span><span class="card-badge">Implementing</span></summary>

#### Planning Overview

Extend the existing study generator and manager. Continue returning `DatasetQAExample`; keep existing batching, progress and JSONL caching. Add `dataset_study_prompts.py` for a limited prompt/template set. No callback framework, study-program class or new utils file.

| `source_mode` | Source supplied to generation |
|---|---|
| `chunk` | One indexed chunk. Fast default. |
| `window` | Adjacent chunks from one document in source order. |
| `document` | One complete document or file. Count over-budget documents as ineligible; never silently truncate. |
| `related_chunks` | An anchor plus a few BM25-related chunks. Reject a bundle if it does not support a useful question. |

`prompt_kind` selects `qa`, `code`, `numeric` or `constraints`; existing text/trajectory behavior remains the `qa` preset with the existing modality. `study_context` adds domain instructions. `require_multiple_sources` requests questions needing multiple supplied units; a quick grounding check rejects obvious single-source cases rather than conducting a full counterfactual study.

**Multiple questions are selective.** Keep `max_questions_per_source=1` for ordinary single-chunk QA. Set it to 2–3 for long-document QA or substantial windows/bundles with distinct facts or reasoning angles. This is a maximum, not a quota: accept fewer rather than padded paraphrases. Return each QA as a separate training example; record unique sources as well as question counts. Do not introduce a separate 1-vs-3 benchmark or prefix-cache tuning project.

Persist the actual text/units shown to generation, including any source expansion, plus claimed evidence references. Existing positive document/chunk fields carry support labels; a small optional `study_source` field on the existing QA object carries the full selected bundle. The AC builder must consume that bundle instead of just `positive_chunk_ids[0]`. This also fixes the current expanded-chunk/evidence mismatch. Source metadata and supporting-evidence labels do not appear in reader inputs.

Add BM25 distractors after question generation, excluding all support documents and respecting existing exclusions. Retain the whole selected bundle; a distractor budget cannot discard required evidence. AC and full-context views use identical evidence, order, question and answer frame. No context removes only evidence. Track realized source tokens and rows. Use short sources around 2–4k tokens and a modest longer-document slice; avoid filling the entire dataset with long examples.

Generation sees raw document/code/table text only, not `source_datum` or benchmark QA/solutions. Reserve a fixed subset of generated examples for reporting before training; keep questions from the same generation call together. This is ordinary material splitting, not an audit of source-corpus overlap. Use no reconstruction-overlap scan, benchmark-question matching service or semantic decontamination pipeline.

Validation is one quick batched answer/support check, basic schema/nonempty checks and cheap duplicate removal within generated material. No recursive judge/rewrite loop, exhaustive code execution, synthetic-overlap test suite or proof that every question is irreducibly multi-hop. Inspect a small mixed sample and proceed unless a concrete systematic error appears.

#### Planned Changes

Behavioral sketches omit unchanged code.

`activation/dataset/dataset_manager.py and dataset_study.py · synthesize_study_examples_qa(), generate_examples_qa()`

```diff
+ # Trailing keyword-only arguments; old callers keep chunk/one-question behavior:
+ source_mode: Literal["chunk", "window", "document", "related_chunks"] = "chunk",
+ max_source_tokens: int | None = None,
+ max_source_chunks: int = 3,  # window/related_chunks only
+ prompt_kind: Literal["qa", "code", "numeric", "constraints"] = "qa",
+ require_multiple_sources: bool = False,
+ max_questions_per_source: int = 1,
+ # Existing study_context, modality, seed, caching_id and progress remain.
+ # Select sources -> batch-generate QA lists -> quick check -> flatten accepted QA.
```

`activation/dataset/dataset_study_prompts.py · prompt builders (new)`

```diff
+ # Move the current study/label prompts here; add the four small presets.
+ # Output 0..max_questions_per_source QA entries with evidence references.
+ # Let the model return fewer distinct questions; no forced quota.
```

`activation/dataset/dataset.py and dataset_caching.py · existing example/cache`

```diff
+ # DatasetQAExample.study_source: dict | None = None
+ # Preserve selected units and exact source text; stable child IDs for multiple QA.
+ # A recipe fingerprint includes source content/options, prompt/context, model and seed.
+ # Cache one source result with its QA list; resume it without duplicating children.
```

`activation/ac_model/ac_model_study.py · _study_examples(), generate_rag_qa_samples(), transform_for_eval_reference()`

```diff
- gold = index.chunks[example.positive_chunk_ids[0]]
+ # Use the full recorded study source, falling back to old single-chunk records.
+ # Forward study selection/prompt arguments through the current synthesis path.
+ # Build matched AC/full/no-context views; full_context is a named reference kind.
```

No native-support loaders, native/synthetic mode selector or new CodeQA/FinQA integrations.

</details>

<details class="card" data-tressoir-markdown>
  <summary><span class="card-title">M2 — Single training call and clear reporting</span><span class="card-oneliner">Every 10%, with honest loss and reference labels.</span><span class="card-badge">Implementing</span></summary>

#### Planning Overview

Keep Golden SFT and `KL(base no-context || current reader no-context)`, top-32 plus remainder retaining gold, weight 1, sampled on 25% and scaled by four. Only the reader adapter receives the penalty gradient. The reference is the fixed original base with adapters disabled; it is not a moving teacher. Rising KL means departure, not improvement. Reader-frozen runs disable this branch.

Use fresh QA reference caches under the new run/material identity. Add a small model/tokenizer/reference fingerprint to the existing cache header and reject mismatches. No general provenance framework or migration of historical reference caches.

Retain full-target SFT; report answer-only NLL separately using the existing renderer's target boundaries. This prevents answer metrics being dominated by submit-answer wrappers. Use simple normalization/exact match, token F1 and numeric checks where applicable; no LLM judge leaderboard or confidence-interval campaign.

Set `reporting_interval=0.1`, the trainer's existing default. Baseline and each 10% boundary score the same **100 held synthesized questions** under AC/full/no-context and **32 fixed training questions** for learning NLL. Freeze panels before the first update. If timing requires smaller panels, choose their size before training and display it; retain the 10% cadence. Decode a small fixed set (about 12 questions) at baseline/50%/final. Boundary percentages mean completed optimizer updates, not elapsed-time fractions.

Add one optional callback at the existing reporting boundaries for the bench's matched controls and decoded samples. It runs without gradients in the same model residency and restores training state. Avoid duplicate AC evaluation and duplicate final callbacks. The final panel at update 100% is emitted once. All curves identify their target subset and current reader state; initial-reader measurements are separate, clearly static references.

An optional training duration limit stops after a completed update, saves a paired partial checkpoint and returns paths/stop reason. Reference preparation and baseline count toward the run budget. If no update fits, report “no updates”; do not advance a completed-epoch marker. Reserve time for final save/export and keep the scheduler's independent hard limit.

The bench supplies a deterministic interleave of domains/sources. `shuffle_examples=True` remains the trainer default; this bench sets it false. One ordinary epoch and one train call per arm, with fresh QA update/epoch counts. Checkpoint provenance is a separate field.

#### Planned Changes

`activation/ac_model/ac_model_training.py · config, train(), paired save`

```diff
+ # Optional reporting_callback(progress, summary); initial/10%/final once each.
+ # Optional max_duration_seconds and truthful partial stop reason/paired paths.
+ # shuffle_examples=True by default; False preserves caller order.
+ # Small reference-identity header; reuse existing save/cache helpers.
```

`activation/ac_model/ac_model_reporter.py · progress, penalty, completion samples`

```diff
- "a rising curve means the reader improves without the rows."
+ "KL(base no-context || current reader no-context): departure from prior behavior."
+ # Label train-call elapsed/ETA separately from the bench's overall clock/deadline.
+ # Clear inactive preparation/evaluation counters at phase transitions.
+ # Key decoded sample snapshots by update, preserving baseline/midpoint/final.
+ # A budget stop is partial/stopped, not unconditionally complete.
```

`activation/ac_model/ac_model_utils.py · existing helpers, if needed`

```diff
+ # Small answer-mask/reference helpers only; no new utils file.
```

Reuse `activation/common/reporting.py`'s atomic JSON updates and persistent widgets. Its existing HTML snapshots may refresh periodically for the editor; preserve that compatibility rather than introduce a reload loop or an unrelated UI rewrite.

</details>

<details class="card" data-tressoir-markdown>
  <summary><span class="card-title">M3 — Reconstruction range default</span><span class="card-oneliner">Small API change; no reconstruction training run.</span><span class="card-badge">Implementing</span></summary>

#### Planning Overview

Use local seeded uniform sampling in [1/30, 1/10], once per materialized item. Preserve explicit scalar calls and nested child ratios. Precedence: passage outer ratio, call scalar, sampled default. Reuse the existing sampler and record realized ratios, including row clamps. A fixed `ratio=1/16` remains fixed.

#### Planned Changes

`activation/ac_model/ac_model_study.py · generate_reconstruction_samples()`

```diff
- ratio: float = 1.0 / 16.0,
+ ratio: float | None = None,
+ # New trailing keyword-only argument, preserving existing positional calls:
+ ratios_range: tuple[float, float] = (1.0 / 10.0, 1.0 / 30.0),
+ # Resolve precedence once per item with a local seeded RNG.
```

One quick private check covers seed stability, scalar compatibility and nested precedence. This change does not delay the first QA results and does not launch the 50M-token sweep.

</details>

<details class="card" data-tressoir-markdown>
  <summary><span class="card-title">M4 — Quick checks and the bounded experiment</span><span class="card-oneliner">Run the primary arm promptly; reuse its material for comparisons.</span><span class="card-badge">Implementing</span></summary>

#### Planning Overview

Run focused private checks on source retention, 1-vs-many QA serialization, matched views, the penalty's reference/gradient ownership, report boundaries and partial saves. Reuse existing trainer/intervention tests. One short real GPU smoke restores P2, takes a few updates and reloads a paired save; discard its updates before the real arm. No new mainline test files, hour-long certification pass, all-reader GPU matrix or reconstruction benchmark reproduction.

Use the first ordinary synthesis batch for throughput/quality inspection and keep its accepted material. Do not require an additional 500-candidate benchmark or a separate multiple-question ablation. Time a few representative training/evaluation batches, then freeze an item count that fits the budget (2–4k was provisional, not a minimum). Make each domain visible early through balanced ordering. Separate preparation time from training/panel time in the report while also showing overall elapsed time.

| Arm | Initialization | Encoder/head LR | Reader LR | Penalty |
|---|---|---:|---:|---|
| Joint, primary | P2 | 5e-5 | 2e-5 | Golden base no-context KL |
| Reader frozen | P2 | 5e-5 | 0 | Off |
| Fresh joint, if capacity permits | Same pretrained bases; fresh AC and adapters | 5e-5 | 2e-5 | Golden base no-context KL |

Comparisons reuse identical material, panels and planned update counts. Each uses one GPU, a fresh optimizer and up to two hours of active experiment time. The campaign ceiling is seven GPU-hours including shared synthesis/smoke; starting an optional arm requires room in that allowance. Compare completed matched-update panels if a run stops early. The fresh arm tests reconstruction initialization's benefit at equal QA budget, not equal lifetime compute.

After green light, prefer MIT capacity that can start promptly. The user also permits **one EC2 spot instance with one RTX PRO 6000 for at most three hours**, including setup/synthesis/training. Use the existing Sky `rtx6000pro` and `--spot` path if available. Track actual cost, enforce a wall-time stop and teardown; idle autostop alone is insufficient. Do not automatically extend its lifetime, upgrade to on-demand or launch extra EC2 instances. Spot hours count inside the campaign ceiling. Run the primary joint arm first; do not wait for comparator capacity. Comparators can use available MIT capacity or remain explicitly pending.

Monitor meaningful progress using the native interruptible wait at roughly 30-minute intervals and phase boundaries. Independent job/node time limits and periodic paired saves remain in force if the session fails. Stop on invalid supervision, non-finite loss, corrupt saves or exhausted budget. Do not retry unattended authentication.

#### Planned Changes

`IB/ARTIFACTS/FIRST_TASK_AWARE_FINETUNING/bench/ · run_qa.py and small manifests`

```diff
+ # Existing study/harness/trainer, saved synthesized material, fresh QA schedule.
+ # Balanced order, one train() call, fixed matched panels, explicit budget/stop.
+ # Small overview linking the existing per-phase/per-arm reports and sync status.
```

Publish the result, actual data proportions/resources, checkpoints and a separate implementation handoff. Review tricky implementation details directly first; use a review agent only if a subtle correctness risk warrants it.

</details>

## Working synthesis mix

These are target **question shares**; report actual question/source counts and source/answer tokens. Prefer already available raw material. Adapt source proportions before freezing the run if a corpus is short, and record the choice. No native question ingestion is needed.

| Share | Raw material / prompt | Source selection and QA count |
|---:|---|---|
| 30% | Repository code and ordinary tool/test output; `code` | Mostly chunks/windows; one QA normally, up to 3 for substantial context with distinct behaviors. |
| 25% | Mathematical/quantitative reference text and tables; `numeric` | Chunks or windows; one QA normally. Exclude evaluation problem/solution dumps. |
| 20% | Non-wiki web/search passages; `qa` | Mostly one QA per chunk. |
| 20% | Long/technical documents and explicit rules; `qa` / `constraints` | Whole documents/windows, some related bundles; up to 3 where context warrants it. |
| 5% | Wiki text; `qa` | Small diagnostic slice, normally one QA per chunk. |

Use Qwen3.5-9B FP8 with vLLM, thinking disabled and short structured responses. `study_context` asks for useful tasks in each domain without injecting actual evaluation questions. Keep generation batched; no special prefix-cache scheduler is required. Multiple QA entries share prefill, but still need decoding/verification and separate training examples.

For 1–2h pilots, use a bounded unique pool and small fixed reporting panels. For 6h, expand unique sources before repeating. For 12–24h, expand available sources and use ordinary epochs when needed, showing repeat counts. The previously discussed central synthesis estimates (roughly 30min/1h/2h/3.5h for 2–4k/6–12k/12–24k/24–48k accepted questions on one H200) remain unmeasured planning estimates, not runtime promises or RTX benchmarks. Use actual first-batch throughput to size work; account for cold preparation in the resource budget.

## Local reporting and reporting check

Reports will be under **`/workspace/IB/TMP/SYNC/TASK_AWARE_QA/`** on this machine:

```text
TASK_AWARE_QA/
  report.tressoir.html          # Overview: phase, elapsed/deadline, sync, links
  report_data.json
  preparation/
    report.tressoir.html        # Sources, synthesis, quick checks, accepted QA
    report_data.json
  joint/
    report.tressoir.html        # Primary training and matched controls
    report_data.json
  reader_frozen/                # Same pair of files when this arm starts
  fresh_joint/                  # Same pair of files when this arm starts
```

The first page appears during preparation. Overview links indicate pending arms instead of opening nonexistent reports. Sync approximately every minute and show the last successful sync; page polling cannot make remote data newer than its sync.

- **One coherent clock:** overview elapsed/deadline includes preparation; per-arm train-call timing and ETA are labeled by scope. Budget remaining is distinct from estimated time to finish updates. Do not show “done” while final evaluation/save still runs.
- **Conventional progress:** unique questions × completed epochs/passes (including fractional passes), percentage, processed question presentations and tokens. Comparisons across different pool sizes use question presentations including repeats; long-context runs state the evidence budget and passage/distractor limits. Keep optimizer counts in technical provenance. No extra data-pass counter or inherited P2 epoch pretending to be QA progress.
- **Panels at 0/10/20/…/100%:** fixed held AC/full/no-context answer NLL, training-panel NLL and correctly labeled drift. Current-reader controls are reevaluated; a starting-reader reference never masquerades as a live control. No empty “evaluation KL” chart for an SFT run.
- **Midpoint remains inspectable:** retain decoded baseline/midpoint/final snapshots keyed by update and a paired midpoint checkpoint. Final rendering cannot overwrite midpoint samples. Chart x-values and labels use the same run-local update clock.
- **Stable live UI:** atomic `report_data.json` writes, existing incremental widget updates and preserved plot state. No new full-page reload. Clear old evaluation/preparation counters when the phase changes; label partial/stopped runs honestly.

Direct source inspection confirmed the trainer already defaults to 0.1, the reporter records nonempty baseline NLL, and the shared page polls JSON without a reload loop and preserves Plotly state. The current reporter needs the targeted fixes listed in M2: sample keys, stale counters/timing scopes, partial-stop wording and the incorrect KL description. These are planned fixes, not claims that unimplemented reporting has already been exercised. A brief baseline→training→evaluation→midpoint→stop dry run will check the emitted data; one live preview check during smoke covers the visible update path if the editor session is available.

## Implementation progress — 2026-09-21

The source-selection options, exact evidence retention/cache, matched RAG views, reporting callback, duration-stop handling and reconstruction range default are implemented. Existing CPU checks passed (20); focused source/cache/view/ratio/callback/resume checks passed (3), plus the reporting sequence dry run. A focused review of the error-prone lifecycle changes passed after fixing stop/resume and callback residency cases. Real GPU smoke also passed: one eight-example update, reader-only penalty gradients and an exactly matching NLL after paired reload. Full experiment measurements are in progress.

Synthesis is complete: **1,474 training / 100 held questions**. Training shares are code521, numeric347, documents280, web236 and wiki90; the held panel is30/25/20/20/5 across code/numeric/web/documents/wiki. The primary and comparators share the frozen material.

The first synthesis process encountered a missing CUDA compiler in FlashInfer sampling; the retry uses vLLM's supported alternate sampler with the same 9B dynamic-FP8 model. Preparation uses already cached raw repository code, financial/report text, web passages, documents and wiki. It excludes cached math solutions and task-conditioned traces. The quantitative share therefore uses grounded calculations on financial documents. This pilot's available material is mostly individual source windows, so generation stays at one question per source; the multiple-question option is implemented for later substantial documents/bundles.

A further reporting fix makes generated timestamps explicitly UTC. This avoids freshness comparisons between this machine and cluster timezones treating a new page as older. JSON polling and existing HTML refresh behavior remain intact. No connected document editor session was available for a visual preview; the emitted-data sequence was checked directly.

## Scope removed in this revision

Removed native QA training/adapters/new loaders; reconstruction-corpus overlap audits; benchmark-question similarity/decontamination scans; optional document-holdout studies; exhaustive synthesis verification/rewrites/execution; separate synthesis/multi-question benchmark campaigns; full reconstruction-gate reproduction and large validation matrices. Keep the small checks that prevent wrong evidence, wrong targets, wrong gradients, lost checkpoints or misleading reports.

The current source/plan check is direct, with no review agent. The earlier review pass applies to the earlier plan; this revised plan is checked against the latest user instructions and current source. Implementation, GPU smoke and runtime reporting validation are now in progress.

The user authorizes clear mid-training bugfixes/hillclimbing on ORCD/CSAIL, with fixed reporting semantics and separate run identities for scientific changes. No additional EC2 nodes. Timing targets yield to these cases.

## Agent execution notes

- Execution is green-lit. Do not read/edit TASK.md. Preserve the frozen Golden text and completed checkpoint/source handoff. Latest user says full synthesis; do not resurrect the superseded native split/mix proposal.
- Existing synthesis call chain: DatasetManager.synthesize_study_examples_qa -> DatasetStudyGenerator.generate_examples_qa -> DatasetQAExample; ACStudy forwards the small study kwargs and consumes stored study_source. No generic options/callback registry. New dataset_study_prompts.py is specifically authorized; helpers otherwise go in existing utils.
- Keep source unit order/content and child QA IDs stable across cache restart. Cache a source bundle plus QA list; create a recipe fingerprint from selected content and relevant generation settings. Do not reinterpret a legacy one-chunk cache as a multi-source result. max_questions_per_source defaults to 1; enable 2–3 only for worthwhile long-context presets. num_samples retains its existing source-attempt budget meaning; report source attempts and accepted QA separately, and let the bench freeze its actual accepted training count before launch.
- study_source stores source units/text and mode; positive IDs remain support labels. Avoid the old generator expansion / first-positive-only truncation. Prompts see explicit raw text, not raw benchmark records or trajectory task/answer metadata. Shared corpora are allowed. Material train/report partitions use generated IDs/source-request groups; no cross-benchmark similarity or reconstruction-overlap pipeline.
- One quick batched check for supported answers is enough. No recursive repair loop, code-execution framework or leave-one-source-out certification. Reject invalid/empty entries and obvious duplicates; a small printed inspection covers semantic quality.
- Reporting callback consumes existing ActivationContextTrainingProgress/EvalSummary, returns None, uses no_grad, preserves placement/modes/trainability and never steps optimizer. Reuse the already computed held AC summary; calculate only missing controls. Deduplicate baseline, scheduled and final boundaries. Do not add extra train() calls. Keep callback failure visible with cleanup.
- Trainer config additions: shuffle_examples=True default; optional max_duration_seconds. Progress counts actual updates. Deadline checks at phase/batch boundaries and completed updates; final save/export time reserved. Partial saves return a consistent AC/reader pair without incrementing completed_epoch. Fresh P2 transfer resets QA schedule/report origin; genuine retries retain the QA run identity and skip the consumed prefix.
- Current reporter source audit: reporting_interval default0.1; reporting_boundaries deduplicates rounded endpoints. report_eval includes baseline when summary items exist. report_completions currently keys only by epoch, so add update identity. report_training currently says complete unconditionally; honor stop reason. report_phase ETA currently includes setup in its extrapolation; label/use train-call scope, keep overall budget with the bench, and avoid showing done before final saves. Clear inactive counters on transitions. Fix KL direction and interpretation text.
- Current HtmlReporter.render atomically writes report_data.json, refreshing embedded HTML on structure changes/finish and periodically (default60s). JS polls about2s, keeps plot uirevision and does not reload on fetch failure. Preserve extension compatibility; do not claim HTML is never rewritten. Remote sync cadence is separately about60s. A quick data-sequence dry run and one available live preview are enough.
- Use simple renderer-based answer positions for this known QA format. Keep full-target SFT objective. Fixed held controls use current adapter state and the same distractors/target IDs; no moving teacher, oracle answer in input or stale initial baseline relabeled current.
- Retain only focused private proofs: source bundle/QA cache round trip; matched views/answer mask; base KL direction and reader-only gradient; baseline/10%/midpoint/final counters and sample preservation; budget-stop paired reload; reconstruction scalar/range precedence. Reuse prior trainer tests. One representative P2 GPU update/save/reload smoke, reset to untouched P2 afterward. Do not run all reader sizes or rerun a historical study to certify a small addition.
- One EC2 RTXPRO6000 spot node may be used for <=3h after green light. Existing sky setup supports --gpu rtx6000pro --gpu-count 1 --spot. Verify available offering and image when launching; use bounded provisioning, a wall-time stop/teardown and existing sync. Idle autostop is not a hard lifetime limit. No extra instance, on-demand switch or extension without new authorization. MIT use follows guarded existing authentication. The original pilot budget is superseded by the user’s longer hill-climbing and200k-generation authorization; keep at least two MIT GPU slots available for supervised requests.
- Source tags/config/material and accepted QA are saved for reproducibility; do not expand this into a lineage framework. New implementation gets a concise actual-diff handoff separate from current_code. Reviewer agents are reserved for subtle implementation risks, not routine plan iterations.


The user authorizes clear mid-training bugfixes/hillclimbing on ORCD/CSAIL, with fixed reporting semantics and separate run identities for scientific changes. No additional EC2 nodes. Timing targets yield to these cases.


## Authorized extension — reusable synthesis bank and retention

The user authorized a separate MIT synthesis GPU while pilot runs continue. Publish immutable mixed batches (about1,000 source attempts each) through an atomic manifest, keeping cached source generations for restart. A later run freezes its selected IDs and input batch hashes at launch; an active generator never mutates its training data. `bench/select_bank.py` selects proportional, deterministically interleaved 30% code /25% quantitative /25% web /20% technical-document subsets. The expansion omits wiki rather than repeatedly using the small pilot wiki pool. Actual counts and source/answer tokens are reported. No native question or answer fields enter generation.

Use the existing9B dynamic-FP8 vLLM setup. The bounded first producer stops at30,000 accepted questions, three GPU-hours, or exhaustion of a source category. Raw-only local preparation adds repository file windows and full Loong raw document windows; ordinary sources produce one QA, substantial quantitative/document windows may produce2–3. One grounding pass and exact generated-question deduplication suffice. Source material is reused freely; the pilot's reserved held question texts are excluded from accepted bank supervision. The pilot's frozen training/evaluation material stays unchanged.

Reports live in `TMP/SYNC/TASK_AWARE_QA/synthesis_bank/`. Completed batches are usable before generation finishes. The synthesis job is additional authorized capacity; record its actual time separately from the original pilot budget.

Retention is now an operational duty: periodically inventory and prune old unflagged transient experiment outputs on this machine, ORCD/CSAIL, and any Sky EC2 nodes. `PRESERVE` protects important/active data; newest~250GB stays protected independently. Keep checkpoint pairs together. Archive only important checkpoints in S3, not every checkpoint. See the durable root-canon retention rule and the cleanup ledger.

The first primary attempt failed at update93 because ORCD scratch exhausted its1TB user quota; optimizer.pt was0bytes. Preserve its40% panels and restart from the original initialization in ORCD pool, with a distinct attempt record. New training/checkpoint/report and bank writes use pool storage. The current-reader full/no-context curves are explicit in chart titles; literal SFT answers and base-model KL targets stay fixed.


## Authorized extension — larger experiments and200k questions (2026-09-21)

User authorizes autonomous hill-climbing, larger generation as needed, and runs beyond two hours. Use at most four autonomous MIT GPUs concurrently, leaving at least two MIT slots for supervised requests; the previously authorized single bounded EC2 spot remains optional. No EC2 node is currently running.

First larger wave: freeze4,000 distinct training QAs and200 new confirmation QAs by source-request group; use the existing100 reporting QAs as development, now explicitly used for tuning. Each arm starts from untouched P2, with two epochs/1,000 updates in one train call,50-update warmup,10% whole-run panels and a135-minute training ceiling plus setup. Epochs shuffle deterministically. Uniformly select32 training-panel questions across the full training pool. Save development-best paired weights and use the untouched confirmation set only to compare selected control/winner. The first larger pool reuses pilot+bank sources and does not yet include new law/office material.

Matched arms: control encoder5e-5/reader2e-5; reader5e-6 with encoder unchanged; encoder2e-5 with reader unchanged. Preserve SFT, original-base no-context KL, architecture, ratio1/16 and evidence/distractor rules. Pilot joint NLL best1.061 then1.233final; frozen-reader final1.068. This motivates step-size tests, not claims of a diagnosed optimizer fault. Next candidates: one query-conditioned AC part per passage;1/10 compression as an explicitly higher-row-budget test. Changes receive separate run names; no silent mid-arm recipe changes.

Generation target200,000 accepted QAs. Accepted-output shares: code30%, math20%(arXiv12+textbooks8), financial5%, web/search25%(MS MARCO20+BRIGHT5), technical documents8%, legal6%, office6%. Raw additions include regulations/judicial opinions, LibreTexts mathematical/business exposition, repository manuals/PEPs, LibreOffice help and Office application automation. Office internal balance is explicit, not proportional to the largest corpus. No native benchmark supervision.

New producer commits small accepted JSONL chunks and source cursors atomically, keeps surplus accepted questions, and publishes balanced1,000-question batches; ten batches give a healthy10k slice. Consumers read only manifest-published immutable files and freeze their selection. Files are self-contained plain question/answer/evidence records with provenance; prompt/code changes do not invalidate their reuse. Three-hour Slurm segments requeue after a graceful microbatch stop; source exhaustion stops for agent expansion rather than looping. Test requeue with a tinyCPU job and test transactional balance/restart with small fixtures. Keep the existing legacy bank intact and usable. Source capacity is measured; expand any limiting family during monitoring.

Reports stay under TMP/SYNC/TASK_AWARE_QA; new arms use hill_control, hill_reader_low, hill_encoder_low, and generation uses synthesis_bank_v2. Same HTML/JSON reporting mechanism and UTC timestamps; phase/progress counters identify their scope. Native agent monitoring continues at30minutes and phase boundaries, including quota/retention checks.

The prepared passage-parts candidate keeps exactly the merged control’s total compressed rows for each question. It allocates rows proportionally across passage text with existing minimums, and records repeated encoder-input tokens separately. The focused CPU check verifies unchanged default inputs, identical evidence/targets, equal total rows and actual trainer placeholder spans. Its frozen-checkpoint transfer screen is complete; a separately trained passage-layout arm remains unlaunched.

CSAIL queued the lower-encoder arm for over20minutes. It was moved to ORCD; its first allocation23405695 was preempted before training and retry23405883 is running. The pending CSAIL job2104500 was cancelled. Current allocation is three ORCD training GPUs plus one ORCD synthesis GPU, leaving CSAIL slots available for supervised requests.

## Canonical run shape — data diversity versus repetition

The user wants a practical recommendation for how much synthesized QA data is enough and how many passes to use. After the learning-rate comparison, prioritize a matched data-budget study: 2,000×4, 4,000×2, and 8,000×1, each 8,000 example presentations / 1,000 updates. Freeze one architecture, ratio and learning-rate recipe before these runs. Use the expanded nine-family bank, including law/office, for all three; the current older-mix 4,000×2 run informs the hypothesis but is not the matched comparator.

Select nested training subsets with the same source shares and comparable source/answer-length distributions. Freeze a shared retrieval corpus and identical distractor construction so changing training size does not change the evidence for shared questions. Use the same 32-question learning panel drawn from the smallest subset. Reserve a balanced development panel reflecting the new mix and a separate untouched confirmation panel, grouped by synthesis source request. Existing confirmation questions remain reserved for the first wave. This is a quick deterministic split/summary check, not an overlap-audit project.

Keep the same 1,000-update cosine schedule and 50-update warmup. Report the main panels at whole-run 0/10/…/100%, regardless of epoch count; retain standard epoch diagnostics as labeled diagnostics if necessary. Record unique questions/sources, source and answer tokens processed, and actual GPU time alongside epochs and updates. Compare development answer NLL, common learning-panel NLL, matched full-context gap, and improvement per GPU-hour. Development chooses checkpoints; confirmation is used only after recipe selection.

If more unique data wins, test 16,000×1 next; if reuse wins, test where further passes stop helping. The outcome is a recommended data size, epoch count and stopping rule with an observed improvement curve, not a universal sample-count claim. More diverse data before many repetitions is a provisional preference only. The separate-passage candidate remains prepared; data-budget comparisons now take priority as GPUs free. Maintain at most four autonomous MIT GPUs including generation.

## Overnight campaign — authorized through 2026-09-22 08:00 EDT

The user removed the small-run constraint and authorized autonomous experimentation until **2026-09-22 08:00 EDT (12:00 UTC)**. Prioritize useful hill-climbing until a specific user request arrives. Strongly minimize EC2 spending: use ORCD/CSAIL, keep at least two MIT GPU slots available, and do not launch an EC2 experiment node for this campaign. The existing four-GPU autonomous allocation includes synthesis. Preserve meaningful partial results and use scheduler limits plus explicit paired resume.

| Insight checkpoint | UTC | Intended evidence |
|---|---|---|
| Sep21 16:00 EDT | Sep21 20:00 | Learning-rate comparison and second-pass benefit; balanced-bank progress; next matched comparisons running with early panels. |
| Sep21 19:00 EDT | Sep21 23:00 | First data-size/repetition comparison; initial recommendation for run shape; structural/capacity tests and larger-data promotion chosen from evidence. |
| Sep22 05:00 EDT | Sep22 09:00 | Larger-run learning curves, diminishing-return evidence, strongest recipe and failure cases; confirmation where selection is settled. |
| Sep22 08:00 EDT | Sep22 12:00 | Final recommendation, paired selected checkpoints, implementation handoff and clear remaining uncertainties; finish/pause autonomous jobs unless user extends. |

These are reporting commitments, not reasons to fake complete comparisons. State what completed, what is still running and what changed. Keep reports under the existing root. Use native agent monitoring every30minutes and at phase boundaries; once/minute file sync remains transport. A lost agent session is not a persistent wake guarantee; scheduler limits bound resource use independently.

The 2k/4k/8k matched study is an early measurement, not a size ceiling. Promote promising settings to16k/32k or larger data and longer runs while improvement and throughput justify them. In parallel, test credible larger changes such as separate passage encoding and compression-capacity changes, each with a clear hypothesis and an unchanged control. Do not alter the evaluation set/objective to manufacture a win. Keep gold SFT and fixed-base preservation as the control, source mixtures representative, and development/confirmation roles distinct. Record negative and interrupted experiments as well as successes.

## Longer side-channel inputs — 8k,16k,32k

The user explicitly requests side-channel inputs up to32k tokens and permits warm-starting other promising checkpoints. Run staged8k/16k/32k experiments from a frozen selected QA encoder/reader pair, with a same-initialization control. Keep nominal1/16 compression initially (~512/1024/2048 activation rows); report actual encoder-input tokens, row counts, source tokens, full-context quality, peak memory and GPU time. A larger evidence budget and passage-count limit must both increase; raising only the old2k budget while retaining four passages would not test32k reliably.

Distinguish two studies: (1) the same required evidence among more matched distractors, and (2) questions needing multiple support sections within a naturally long document or coherent dossier. The first isolates robustness to added context. The second tests distributed evidence; do not claim arbitrary distractor padding establishes it. Questions can be synthesized from2–3 selected support sections totaling<=6k tokens, then trained/evaluated with the longer source including every support section. This uses existing source-selection programmability and avoids a needless32k generator reconfiguration. Use multiQA where substantial source bundles justify it.

Private runner options expose evidence-token budget, maximum passages, ratio, microbatch size and explicit paired initialization. Preserve old defaults and keep each scientific change under a new run name. Whole-source gold evidence remains intact; actual lengths are reported rather than claiming a hard cap. Start the real32k mechanics check with microbatch1. The architecture permits the length, but specialized FP32 passage attention may dominate memory; check actual forward/backward memory and throughput before scheduling long training. Any memory fix must preserve the computation/compression contract and receive a focused correctness check. No pooling. Results join the overnight insight checkpoints and may justify longer runs; they do not block the data-size/repetition comparison.


## Response quality and paper oracle — 2026-09-21

The user specifies a **27B oracle for the final paper**. Current fast proof-of-concept runs retain their recorded 9B setup. Cleaner oracle-generated/checked references should improve the usefulness of NLL, but model size alone does not establish reference correctness. Do not transfer numerical reconstruction quality bands to QA: gold-token likelihood is sensitive to wording and gold-prefix conditioning. Pair it with source-grounded correctness of actual generated answers.

The completed `NLL_EXAMPLES.tressoir.html` shows twelve matched full/AC/no-context outputs at control update500. It separates valid paraphrases, model errors, two definite question/reference defects, and a missing-dependency ambiguity. All original data and scores remain unchanged; this inspection does not estimate the overall defect rate. The exact-batch no-context recheck reproduced all100 original scores. Future synthesis keeps one quick grounding pass and explicitly checks question premises, numerical direction and named code branches; no new audit campaign.

GPU mechanics now establish that the minimal singleton-batch passage-attention change preserves outputs/gradients at8k/32k and reduces isolated32k peak memory4.79→1.36GiB. A real32k eight-example update plus matched evaluation and pairedsave completed in44.9s/update with41.07GiB training peak. Auxiliary post-training penalty/reload checks are tracked separately; a probe setup error must not relabel the completed training as failed.

The200k reusable bank retains its earlier run-as-long-as-needed authorization beyond the overnight experiment deadline. The Sep22 12:00UTC boundary applies to the autonomous training campaign and its handoff.

## 27B synthesis and qualitative validation — accepted Sep21

Treat AC as a reasonably effective latent communication channel for planning: held-out improvements over no context and concrete source-dependent answers support information transfer. This remains a working assumption about the tested setting, not established reliability across all tasks or lengths.

Generate a separate **32,000-question Qwen3.5-27B bank**, dynamic FP8 and reasoning disabled in both synthesis and checking. Keep the nine-family proportions and balanced 1,000-question releases. Preserve plain JSONL, provenance, interruption recovery and requeue. CPU prefetch completed at pinned revision `fc05daec18b0a78c049392ed2e771dde82bdf654`; GPU producer23412107 started at21:06UTC after the2k training allocation completed. No extra EC2 node or increase to the four-GPU autonomous cap.

Start with **16k × 1 epoch** training from P2, keeping the 2B encoder → 9B learner, original learning rates, ratio1/16, gold SFT and fixed-base preservation penalty. Compare legacy 9B-generated and new 27B-generated supervision on the same two frozen development cohorts: 200 questions from each bank. Report the cohorts separately, with their matched full/no-context views. Hold separate confirmation examples by synthesis request. Compare arms within each cohort; a change in gold wording/difficulty can change raw NLL across cohorts. Initial learning-panel samples come from each arm's own training set.

User requests prompt fixes for oracle pathologies. Explicitly reject false question premises, arithmetic direction errors, borrowing an adjacent code branch, and invented missing dependency definitions. Require self-contained questions and answers; temporary source numbers belong only in citation metadata. Keep one quick grounding check. Prefer prompt improvements with reasoning off; test low reasoning only if these problems persist. The active 9B bank remains an immutable historical recipe. The new comparison therefore measures an improved 27B supervision pipeline, not a pure model-size effect; any later size-only claim requires a 9B companion with the same revised recipe.

For promising runs, a subagent will review **all 100 questions of the current development cohort**, with fixed full-context and no-context outputs from the P2 reader at training start, and AC outputs from paired control checkpoints after 4,000 questions ×1pass and ×1.8passes initially. Produce a standalone HTML report containing a concise summary and every example, including source evidence, gold, actual decoded responses and answer NLL. Keep these fixed baselines distinct from live current-reader training controls. Report valid paraphrases, unsupported answers, reference defects and ambiguity separately. Descriptive counts apply to this cohort; do not turn the development review into an untouched test-set claim.

NLL remains the continuous training metric. Add normalized exact match for genuinely exact answers and execution/formal checks where an existing task supports them. These checks do not substitute for semantic assessment of arbitrary prose, and token accuracy is still teacher-forced. The qualitative review establishes whether likelihood improvements correspond to useful answers.

The initial long-QA pilot exposes two limitations: arbitrary distant support sections often do not form a natural combined question, and temporary source labels leaked into accepted text. Preserve that attempt as a diagnostic, and do not use it unfiltered for training. Prepare a small separate whole-parent synthesis pilot so the oracle can choose related support sections; retain exact cited spans and the full long context. Keep this change separate from ordinary single-document bank generation.

The27B initial32k target is flexibly extendable through `qa27b_target.json`, read between microbatches; no structural cap. The producer now uses request batches of64 on its existing GPU, with the same model/prompts and a recorded recipe change. The first100-item qualitative report is complete; source-aware review found that similar NLL still conceals a substantial semantic gap.


## Queued long-context adaptation and monitoring

Two arms start from the same selected control900 encoder/reader pair. Both train on the same640 previously used questions for one epoch (80updates), retain the old100 development questions and share4100 retrieval records. One uses the2k evidence budget/four-passage limit, the other32k/64passages. Learning rates,50-update warmup,1/16 ratio and microbatch1 match. This tests adaptation to added context on familiar training questions, not the benefit of fresh supervision. Both have10% panels and paired recovery saves every8updates. Actual measured32k encoder inputs are30,549–33,272tokens;2k remains a soft budget that retains whole required sources.

The2k arm23414747 follows the8k data-study restart;32k arm23414771 follows2k. Estimated32k runtime is roughly two hours including panels, with a three-hour scheduler bound and controlled paired resume. Afterward, evaluate BOTH final pairs on the same2k AND32k held inputs, with matched full/no-context views. This cross-evaluation checks long-context improvement and short-context retention. `LONG_ADAPTATION.md` records exact material and measurements.

The whole-parent75-source synthesis pilot23413645 follows the bounded27B checker comparison, which follows the100-item qualitative evaluation. It uses9B dynamicFP8, no reasoning, batch2 and40,960-token capacity; all rendered prompts fit without truncation. The oracle chooses2–3 related support sections from each parent; one grounding check uses those exact spans. Acceptable QA and rejection-stage receipts publish after each pair of parents. The original low-yield pilot stays unchanged.

The20:15UTC native check found all three data-study arms and the9B producer advancing,24k published bank questions, and adequate storage headroom. No pruning or EC2 launch was needed. Next full native check20:45UTC, with earlier intervention at phase boundaries. The4pm checkpoint report is `INSIGHTS_1600_EDT.md`.

Before either new synthesis job started, the generator/checker presence penalty was set explicitly to0. The inherited1.5 discourages output-token reuse (including repeated JSON boolean decisions); removing it avoids that decoding bias without adding reasoning. This is a justified configuration choice, not a measured diagnosis of the earlier reference defects. Original9B production and all accepted data remain unchanged.


## Frozen layout/capacity screen and bounded oracle check — Sep21 evening

Queued passage_screen_900 job23419877 scores the same frozen control900 pair on100 old development questions with identical2k evidence and gold targets. Four fresh inference processes compare merged1/16, separate query-conditioned passage parts with exactly the same per-question row total, merged1/10 and merged1/30. Actual total rows are10,314 /10,314 /16,480 /5,525; separate parts add4.06% encoder input tokens through repeated framing. Score all100, common full/no-context controls once, and up to12 fixed greedy samples. No optimizer updates. This screens immediate transfer; poor results do not rule out training for a variant.

The required long-context cross-evaluation is queued as23420683 after successful completion of both adaptation arms. It rescores initial900 and both final pairs at both2k/32k evidence, with microbatch1 and matched full/no-context views. Completed cells remain immutable and resumable; missing/incomplete final pairs fail visibly. This yields six cells, including fresh initial baselines rather than relying on an unverified historical input fingerprint.

A fixed20-pair initial27B review found two concrete grounding failures (algebra and transferring a claim between differently scoped groups), plus separately labeled caveats. It does not estimate a population error rate. The23-case checker comparison is complete: retain the current boolean checker, which made19/20 correct scored decisions versus15/20 for the brief-check alternative (including two truncated outputs). These enriched diagnostic cases do not estimate population accuracy. It compares the current boolean-only checker with a brief independent calculation/evidence field before the verdict, on the same27BFP8 engine with thinking off. Known failures are diagnostic cases; three ambiguous cases remain unscored. Measure decisions and added decoding time before changing any future generation recipe. Existing bank records remain intact. This adds no GPU beyond the four-slot limit.


## Data amount versus repetition — confirmation

The fixed-budget study and its separate reserved200-question confirmation are complete. All arms saw8,000 question presentations. Development AC NLL:2,000×4=1.008,4,000×2=.835,8,000×1=.783; confirmation:1.078,.895,.844 respectively. Fresh data leads on both cohorts. The8k×1 advantage over4k×2 remains modest and uncertain on200 questions (paired-question95% interval for the difference:−.107 to+.020); both clearly improve over2k×4. This supports expanding unique data before repeating, without claiming a universal optimal pool size. No checkpoint switching occurred on confirmation; that cohort is now used. The private retrieval-exclusion correction preserved exact source evidence/gold/retrieval and completed GPU scoring. The later27B comparison reserves its own confirmation cohorts. See DATA_BUDGET_CONFIRMATION.md.

## Data amounts in qualitative reports

Human-facing labels use unique questions × completed passes, including fractions. The earlier100-item report concerns the old4,000-question run only; it does not contain the newer8,000×1run. Preserve that report and add DATA_BUDGET_COHORT100 on a deterministic family-balanced100-question subset of the shared200-question data-budget development cohort. Compare frozen P2 full/no-context baselines and the already-fixed final2,000×4 /4,000×2 /8,000×1pairs. Select before inspecting outputs; retain all original200-item NLL batching, then decode only the fixed100. Distinguish the subset's NLL from the whole200-question metric. Review every source and actual answer, leaving reference concerns visible. This is descriptive development analysis, not an independent test-set accuracy claim.

The live report projection changes display units only. Per-run plots use passes; equal-budget comparisons use question presentations, including repeats. Preserve raw measurements and checkpoint provenance. Routine live updates in the local projection use JSON polling, with HTML regeneration only when structure changes. The overview now uses the same policy; its periodic fallback was still enabled and was disabled in the private transport helper. Actual-constructor and live72.9-second file checks passed (JSON advances, HTML unchanged). The shared product reporter retains its documented periodic snapshots; actual editor morph behavior remains an extension concern.

## Additional data-reuse control — prepared, deferred Sep22 04:33 UTC

The next useful matched control is **8,000 unique 27B-generated questions × 2 epochs versus16,000 × 1**, both P2/1/16 and16,000 presentations. The exact existing runner-selected first half is source-balanced and contains all32 fixed learning questions. Hold the shared400 development questions, retrieval pool, original P2 pair, batch8/microbatch2,50-update warmup and2,000-update cosine schedule fixed. Use the existing single-train-call, whole-run10% panels. This auxiliary development control stays outside the unchanged five-arm reserved confirmation design.

The control was queued as23453389 while9B synthesis paused safely after a microbatch commit. It required an RTX/H200: prior equivalent training peaks of~62GiB exceed the generator's L40S. Scheduler dry-run estimates did not support useful completion within this campaign, so the still-pending control was canceled **before allocation or training**. The9B producer resumes as23453392 with its original200k target and auto-requeue;108k published questions remain cached. Existing main comparisons,4B and evaluation chains retain priority and the four-GPU cap. No additional product code or synthesis was required. The checked design is retained in DATA_REUSE_8K_2EP.json for a later available slot; the two optional overview comparison panels appear only if a future run publishes data.
