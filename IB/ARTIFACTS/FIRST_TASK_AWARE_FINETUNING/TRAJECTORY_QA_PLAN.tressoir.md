# Trajectory Q&A: settle encoding first

Plan questions about recorded agent histories using the existing activation-context model and trainer. Long-context encoding is the prerequisite; most subsequent work should be training-item construction.

## Executive Summary

**Resolve long-context encoding before freezing the trajectory training recipe or launching its main study.** Reuse the current matched merged/piecewise experiment, then confirm its candidate on a small trajectory panel. Memory proofs establish feasibility, not a quality winner.

The flow is **recorded history → grounded question and answer → selected encoding layout → existing training item → existing SFT trainer → matched AC/full/no-context answers**. Keep raw supervision reusable across layouts. This parallel session plans the work; the running campaign continues under its existing owner.

| Likely work | Expected scope |
|---|---|
| Select sources, question types, lengths and distractors | Mostly recipes and experiment configuration. |
| Preserve evidence, build pieces if selected, allocate rows | Small changes in the existing trajectory item builder. |
| Match full-context evidence and account for combined length | Concrete builder changes already identified. |
| Model architecture, trainer, loss, runtime or serving | No change currently identified. Revisit only if existing layouts cannot meet the prerequisite. |

**Mostly item building is realistic; configuration alone is not assured.** Currently, the trajectory full-context reference omits distractors, and the length budget applies separately to each trajectory. Both need attention before a controlled long-context study.

## Requested Decisions

**Accepted from your request:** encoding comes first; focus implementation on existing machinery and item-building strategies.

The encoding choice is pending evidence. There is no empirical winner to approve yet and no implementation approval requested in this pass. Source mix, training count, initialization and evaluation scope remain provisional. [Agent-facing plan](TRAJECTORY_QA_PLAN.md).

## Milestones

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M0 — Resolve long-context encoding</span>
    <span class="card-oneliner">Required before the main recipe; a small encoding pilot may resolve it.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

| Approach | Tradeoff |
|---|---|
| One part per retained trajectory window | Already supported and simplest. Each long window remains a long encoder call. Distractors already have separate parts. |
| Several chronological pieces per trajectory | Existing multi-part encoding, with turn-aware partitioning and row allocation. Repeated frames cost tokens; the reader combines independently encoded pieces. |
| Pieces recursively recompressed | Nested encoding exists, but needs a bounded chunk tree and budgets at every level. Extra compression can lose information. Keep as fallback. |

The current depth-2 wrapper encloses the entire window: **it does not shorten the first encoder pass**.

RAG encodes the question with each passage. Trajectory parts must retain the **original task, system frame, tools and history, with the future Q&A question only in the reader prompt**, including for recursive parents and distractors. A RAG winner needs trajectory confirmation.

Use the completed [matched RAG study](PIECEWISE_TRAINING_DESIGN_V2.json) to narrow candidates; status stays in [the campaign ledger](EVALUATION_QUEUE.json). Freeze a small trajectory panel with identical sources, questions, targets, distractors, reader and output-row totals across layouts. Proposed total-source buckets are 8k/16k/32k, subject to corpus lengths and capacity guards. Include early/middle/late evidence and separate questions needing related turns; distractor padding alone does not show reasoning across turns.

Start with a frozen-weight screen. If adaptation is needed, compare bounded trained pilots from the same pair with matched examples, question presentations/order/schedule. Unadapted transfer alone cannot rule out a layout. Freeze pilot counts and acceptance criteria before inspecting outputs.

Select using source-supported **generated answers**, answer-token NLL and feasibility. Measure side tokens including frames, largest encoder call, total reader rows, full-text length, memory and time. Check the heaviest shapes after optimizer state exists. Existing RAG fits do not certify trajectory shapes.

If quality is indistinguishable, prefer the simpler feasible whole-window option and state uncertainty. If no option meets the intended length/quality requirements, keep the prerequisite open and bound the next encoding experiment. A supported operating range is sufficient; a universal optimum is unnecessary.

#### Planned Changes

After selection, add `TRAJECTORY_ENCODING_DECISION.md` here: layout/rationale, compatible pair, source/part/reader limits, rows/depth, conditioning, quality by length/position, resource measurements, frozen inputs/results and limitations. The selected and accepted contract unblocks the main recipe; a memory pass alone does not.

This is an approach-level plan. Interfaces and product patch hunks depend on that selection and are intentionally deferred.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M1 — Freeze material and construct items</span>
    <span class="card-oneliner">After M0: retain the same sources and supervision across layouts.</span>
    <span class="card-badge">TBD</span>
  </summary>

#### Planning Overview

Reuse Open-SWE, S1 DeepResearch and Nemotron loaders and study synthesis; inventory availability before choosing proportions. Ask about observed outputs, decisions, errors and state changes. Start with local facts in long histories and add naturally related multi-turn questions. Recorded assertions are not automatically externally true. Use one grounding check plus bounded source-aware inspection, retaining rejection reasons; never inject benchmark answers or reference patches.

Cache questions, literal answers and exact support spans independently of layout, with source/prompt/model/settings provenance. Preserve frames, tasks, tools, turn order, structured calls/results and every supporting span. No silent truncation or splitting indivisible calls. Every encoded part excludes the future question.

The focused builder delta is matched full-context distractors, combined length accounting and the selected partition/row policy. Per-part minimum rows may make an exact total impossible: coarsen the partition or report unsupported geometry instead of silently adding rows. Preserve target rendering and answer masks. Keep common question IDs but distinct built-item/cache recipe identities.

The trajectory entry point currently uses default synthesis and centers on the first positive chunk. Multi-source questions require retaining **all** support spans; enabling a synthesis flag alone is insufficient. Full context and AC share evidence, distractors and order. Reuse the no-context transform and label any evidence-only oracle separately.

For unseen-history claims, reserve whole source-qualified trajectory/root groups before synthesis and distractor selection, keeping variants together. The existing shared QA splitter holds out questions on familiar histories; use it only for that labeled claim, or partition origins in the bench without changing the shared splitter.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M2 — Small matched trajectory pilot</span>
    <span class="card-oneliner">After M0/M1: establish answer quality before scaling.</span>
    <span class="card-badge">TBD</span>
  </summary>

#### Planning Overview

Reuse literal-answer SFT, the existing preservation penalty and paired saves. Proposed family: 2B encoder → 9B reader with Golden QA settings; M0 determines the compatible pair. Fresh P2 and QA continuation are distinct recipes. Start the new phase with an explicit fresh QA optimizer/schedule.

Freeze training count and final-checkpoint rule before launch. Report unique questions × completed passes, presentations, tokens, fixed learning-panel loss and held answer NLL. Compare AC/full/no-context with identical questions, targets and masks; distinguish initial frozen baselines from current-reader controls. Review generated answers against sources, retaining reference defects, invalid outputs and truncations. Separate development from untouched confirmation. Agree the recipe and a slot within existing campaign limits before launch.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M3 — Scale the selected recipe</span>
    <span class="card-oneliner">After pilot results: select data amount and source/question mixture.</span>
    <span class="card-badge">TBD</span>
  </summary>

#### Planning Overview

Scale with the selected item pipeline and existing trainer. Longer or recursive/streaming histories are explicit extensions. Archive the complete pair and recipe. Trajectory-QA quality alone does not establish live-agent compaction or task-completion gains.

</details>

## Why the implementation should stay small

The main candidate file is `activation/ac_model/ac_model_study.py`: it already owns source-span windows, original-task framing, distractors, ratios, nested parts and training items. `dataset_study.py` and `dataset_study_prompts.py` supply cached synthesis and citations; expose options only when needed. A small experiment adapter can reuse material freezing, equal-row allocation, answer masks and reports.

No change to `ac_model.py`, `ac_model_training.py` or runtime/serving is identified. Explain any expansion before implementation.

## Evidence and validation

Inspected the current builder, synthesis, reference transforms, splitter and piecewise bench. The [32k frozen-weight screen](PASSAGE_LAYOUT_32K.md) is mixed; [source-aware answers](PASSAGE_LAYOUT32K_COHORT100.tressoir.html) and trained results inform selection. [Piecewise preflight](PIECEWISE_V2_PREFLIGHT_REVIEW.md) establishes numerical/memory feasibility for tested shapes, without choosing a quality winner.

Planning validation passed: Markdown artifact checker, all six local links and direct source/projection consistency review. No product code or jobs changed. Live custom-editor rendering is unverified in this session.
