# Trajectory Q&A: encoding prerequisite and scope

Plan questions about recorded agent histories using the existing activation-context (AC) model and trainer. Resolve long-context encoding before finalizing the main trajectory training recipe.

## Executive Summary

**Accepted direction, September 22:** this parallel session plans trajectory Q&A. Long-context encoding is a prerequisite to freezing its recipe and launching the main study. The user prefers a small implementation, principally different training-item building strategies. This session does not own the running RAG/piecewise campaign.

Intended flow: recorded history → grounded question, literal answer and supporting spans → selected encoding layout → existing `ActivationContextTrainingItem` → existing SFT trainer and paired encoder/reader checkpoint → matched AC/full/no-context evaluation.

The likely work is source/question selection, evidence-preserving windows or pieces, length/row allocation, distractors and matched evaluation views. No model, loss, optimizer, runtime or serving change is currently identified. Mostly item construction and experiment configuration is a defensible expectation; a configuration-only promise is not.

Recommendation: consume the existing matched merged/piecewise study, then confirm its candidate on a small trajectory panel with the future question excluded from encoding. Select a supported operating range, without requiring a universal optimum. Keep dependent milestones provisional.

## Requested Decisions

- **Accepted:** encoding first; favor existing machinery and small item-builder changes.
- **Evidence-dependent and open:** whole-window versus chronological pieces; any need for recursive recompression; source/part/reader limits, row budget and compatible starting pair.
- **Later recipe choices:** source mixture, question mix, training count and familiar-history versus unseen-history evaluation.

No implementation or run approval is requested in this planning pass. A bounded trajectory encoding pilot may resolve the prerequisite; the main trajectory study depends on its outcome. Do not ask the user to select an empirical winner before evidence exists.

## Milestones

### M0 — Resolve long-context encoding (Planning; prerequisite open)

#### Planning Overview

Reuse current RAG evidence without duplicating or changing its jobs. Status belongs to `EVALUATION_QUEUE.json` and `IB/STATE.md`; the matched trained comparison is specified in `PIECEWISE_TRAINING_DESIGN_V2.json`. Frozen-weight NLL and memory proofs alone do not establish a quality winner.

| Candidate | Existing support | Consequences |
|---|---|---|
| One part per retained trajectory window | Direct trajectory builder support at depth 1. Distractors already have separate parts. | Least builder work; each long window remains one long encoder call. |
| Several chronological pieces per trajectory | Existing multi-part model/schema; RAG bench has equal-total-row allocation. | Add turn-aware partitioning and row allocation. Repeated frames cost tokens; independently encoded pieces leave cross-piece reasoning to the reader. |
| Pieces recursively recompressed | Nested parts and child-first encoding exist. | Build a bounded chunk tree and budgets at every level; extra compression can lose information. Keep as fallback if simpler layouts cannot meet the intended range. |

The existing depth-2 option wraps the entire window, so it does **not** bound the first encoder pass. It is not already arbitrary-length chunked encoding.

RAG includes the question inside parts. Trajectory encoding must retain the original task, system frame, tools and recorded history while excluding the later study question from every part, including recursive parents and distractors. A RAG winner is therefore only a trajectory candidate.

Gate evidence:

1. Completed matched RAG training readouts: identical sources, question order, targets, starting pair, training budget and output rows. Assess generated answers as well as answer NLL and resources.
2. A frozen trajectory panel with identical source turns, questions, answers, distractors, reader and output-row totals across layouts. Suggested total-source buckets: 8k/16k/32k, subject to corpus availability and actual capacity guards. Include early/middle/late evidence and a separate group requiring related turns. Distractor padding alone does not demonstrate distributed reasoning.
3. A cheap frozen-weight screen, then bounded matched trajectory adaptation if needed. Compare trained pilots from the same initial pair with matched question presentations/order/schedule; do not treat differently trained checkpoints as a layout-only comparison or reject a layout solely on unadapted transfer.
4. A fit check after optimizer state exists, including the heaviest planned training/evaluation shapes. Measure actual side tokens including frames, largest encoder-call length, total reader rows, full-text reader length, GPU memory and time. RAG fits do not certify trajectory shapes.

Freeze pilot counts, panel and acceptance criteria before inspecting outputs. Choose on source-supported generated-answer quality and resource feasibility, using answer-token NLL as supporting evidence. If quality is indistinguishable, prefer the simpler feasible whole-window layout and disclose uncertainty. If none meets the intended lengths and quality, keep the prerequisite open and bound the next encoding experiment; do not silently shorten the promised task.

#### Planned Changes

After selection, add `TRAJECTORY_ENCODING_DECISION.md` here: chosen layout and rationale; compatible checkpoint pair; supported total-source/per-part/reader limits; row allocation and depth; original-task-only conditioning; quality by length/position; time/memory; frozen inputs/results; remaining limitations. That selected and accepted contract unblocks the main recipe. A memory pass alone does not close M0.

This is an approach-level plan. Product interfaces and exact patch hunks depend on the selection and are intentionally deferred.

### M1 — Freeze material and construct items (TBD; depends on M0)

Reuse existing Open-SWE, S1 DeepResearch and Nemotron loaders and study synthesis; inventory availability before selecting proportions. Ask about observed outputs, decisions, errors or state changes. Start with local facts within long histories, adding naturally related multi-turn questions. An agent's assertion is not automatically externally true. Use one grounding check and bounded source-aware inspection with rejection reasons; never substitute benchmark answers or reference patches for recorded evidence.

Cache questions, literal answers and exact support spans independently of layout. Pin source content, generation prompt/model/settings and recipe identities. Preserve system/task/tools, original order, structured calls/results and all required evidence. Never split indivisible calls or silently truncate required evidence. Every encoded part excludes the future question.

Audit combined input length as well as individual windows, frames and rows. Per-part minimum rows can make an exact total impossible: coarsen the partition or report unsupported geometry, rather than invisibly granting extra rows. Keep question identity stable across layouts while assigning built items/cache entries distinct recipe identities.

Build full context from the same evidence, distractors and order as AC, preserving answer targets and masks. Reuse the no-context transform. Label any evidence-only oracle separately. For unseen-history claims, partition whole source-qualified trajectory/root groups before synthesis and distractor selection, keeping all questions/variants together. Existing shared `split_items` holds out QA questions on familiar contexts; keep it unchanged and use a bench-level origin partition, or explicitly claim only familiar-history generalization.

### M2 — Small matched trajectory pilot (TBD; depends on M0/M1)

Reuse literal-answer SFT, paired saves and the existing no-context preservation penalty. Proposed model family: 2B encoder → 9B reader with Golden QA settings; M0 determines the exact compatible pair. Fresh P2 and QA continuation are separate recipes. Start the new phase with an explicit fresh QA optimizer/schedule.

Freeze training count and final-checkpoint selection rule before launch. Report unique questions × completed passes, question presentations, actual tokens, fixed learning-panel loss and held answer NLL. Compare AC/full/no-context on identical questions, targets and masks. Distinguish frozen initial-reader baselines from current-reader controls. Review generated answers against sources; retain reference defects, invalid answers and truncations. Separate development and untouched confirmation. Agree the recipe and a slot within existing campaign resource limits before launch.

### M3 — Scale the selected recipe (TBD; depends on pilot results)

Choose data amount and source/question proportions from pilot evidence. Reuse the selected item pipeline and existing trainer. Longer or recursive/streaming histories are explicit extensions. Archive the complete pair and material recipe. Trajectory-QA results alone do not establish live-agent compaction or task-completion gains.

## Code scope grounded in the current tree

| Location | Existing behavior | Likely delta |
|---|---|---|
| `activation/ac_model/ac_model_study.py` | Source-span windows, original frame/task/tools, distractors, ratios/depth, items and eval transforms. | Main candidate file: matched full-context distractors, combined length accounting, selected partition/row strategy. |
| `activation/dataset/dataset_study.py` and `dataset_study_prompts.py` | Cached synthesis, trajectory prompts, source citations and multi-source options. | Prefer recipe reuse; expose source options through the trajectory entry point only if needed. |
| Existing campaign bench | Frozen material, equal-row allocation, answer masks and reports. | Small trajectory adapter/configuration; established reusable builder behavior belongs in the existing product file. |
| `ac_model.py`, `ac_model_training.py`, runtime/serving | Multiple/nested parts, SFT, optimizer, checkpointing and capacity guards. | No change identified. Explain any scope expansion before implementation. |

Two concrete gaps were verified: trajectory `in_context_messages` currently contains only the gold trajectory, and `trajectory_tokens_range` applies independently to gold and distractor windows. Also, the trajectory entry point invokes synthesis with defaults and centers its window on the first positive chunk. Multi-source synthesis cannot simply be enabled without retaining every supporting span. These are item-building issues, not reasons for a new trainer.

## Evidence and validation

Inspected `generate_trajectory_qa_samples`, `_chunk_window`, `_source_window_messages`, `transform_for_eval_reference`, `split_items`, synthesis/prompt construction and the current piecewise bench's allocator. Evidence: `PASSAGE_LAYOUT_32K.md`, `PASSAGE_LAYOUT32K_COHORT100.tressoir.html`, `PIECEWISE_V2_PREFLIGHT_REVIEW.md`, `PIECEWISE_TRAINING_DESIGN_V2.json`. The frozen-weight 32k screen is mixed; source-aware answers and the trained comparison must inform selection. Numerical/memory proofs establish feasibility only for their tested shapes.

Validation passed: the Markdown artifact checker reports no issues, all six projection links resolve, and direct source/projection review confirms the prerequisite, scope, dependencies and caveats agree. No product code, jobs, campaign ledger or existing main plan is changed. Live custom-editor rendering remains unverified in this session.
