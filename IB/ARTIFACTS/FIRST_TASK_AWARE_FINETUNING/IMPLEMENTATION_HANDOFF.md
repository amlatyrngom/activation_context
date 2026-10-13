# Task-aware QA implementation handoff

**Superseded for source synchronization:** use [CONSOLIDATION_HANDOFF.md](CONSOLIDATION_HANDOFF.md) and its current `/source` patch. This document and `implementation/` preserve the earlier implementation record; do not stack that patch with the consolidation package.

Status: CPU checks, focused review and real GPU smoke pass; pilot experiments complete; larger hill-climbing runs and synthesis expansion active. This patch starts from the source state at this task’s green light and preserves previous uncommitted reconstruction changes.

- `dataset_study_prompts.py` contains existing prompts plus QA/code/numeric/constraints presets. Existing study functions accept chunk/window/document/related-chunk selection, source budgets, optional multi-source questions and a maximum question count (default one). Whole-source requests, exact evidence and distinct child IDs survive caching.
- `DatasetQAExample.study_source` carries selected units and their order/text. RAG retains all selected evidence, excludes their documents from distractor retrieval, and uses matched distractors in full context. No-context controls remove only evidence. No native training-data ingestion was added.
- The trainer accepts an optional reporting callback, `shuffle_examples` (default true) and `max_duration_seconds`. The callback restores adapter state, model residency, modes and trainability. Duration stops check batch/update boundaries, reserve save time, preserve a paired partial checkpoint and never relabel a resumed checkpoint as step zero.
- Epoch checkpoint publication now precedes cleanup of previous recovery files. A failed publication leaves the previous paired weights and optimizer available. This later recovery fix is included in the patch; already-running learning-rate arms retain their original source snapshot.
- Passage-attention readouts now supply a singleton batch dimension to PyTorch attention, enabling the FP32 memory-efficient CUDA kernel. Weights, formula and dtype are unchanged. A trained QA readout passed output and all input/parameter gradient comparisons at 8k and 32k; the 32k isolated peak fell from 4.79 to 1.36 GiB. Whole-model 32k validation follows separately.
- Base no-context reference caches have a small identity header. Penalty reporting states its direction and interpretation correctly. Completion snapshots retain their update identity; stopped runs and update ETA use honest labels.
- Reconstruction generation defaults to a seeded uniform ratio in [1/30,1/10]; explicit passage/scalar ratios retain precedence. No reconstruction training was launched.
- Shared report timestamps use explicit UTC, avoiding cross-machine freshness mismatches. Existing JSON polling/HTML refresh behavior remains.

## Applying

The exact patch is [implementation/changes.patch](implementation/changes.patch). Before/after hashes and full changed files are alongside it. Check each destination against `manifest.json`: skip already-applied paths; inspect divergent paths rather than overwrite them. Use `git apply --check` before applying to a matching source tree. Do not apply the older cumulative `current_code/` patch again.

## Validation

20 existing intervention/nested-material CPU checks passed. Three new private checks cover source modes/cache/multiple questions, matched views/ratio precedence, and callback/budget/resume safety; all pass. The reporting dry run preserves four sample snapshots, clears old counters and labels partial stops. Qwen3.5 answer-token masks were checked with numeric, prose and Unicode answers. The scoped implementation review passes. No connected editor session was available for a visual preview.

Live results: `/workspace/IB/TMP/SYNC/TASK_AWARE_QA/`. GPU smoke on an RTX PRO 6000: one eight-example update, positive encoder/reader gradient norms, penalty gradients isolated to the reader, paired save/reload NLL identical at 0.6855256557 on its four-item smoke panel. These are mechanism checks, not held-pilot results. Primary and reader-frozen runs use the untouched P2 pair and 1,474/100 frozen synthesized train/held questions.

## Pilot outcome and active extension

The completed 1,474-question pilot finished at held answer NLL 1.2334 for P2 joint training, 1.0679 with the reader frozen, and 1.4829 from fresh adapters. P2 joint reached 1.0606 earlier in training; the larger study retains development-best checkpoints. These are QA answer-token NLLs, distinct from the reconstruction quality gate.

Private bench changes now freeze 4,000 training examples and 200 unused confirmation examples, run two epochs with global 10% reporting, compare lower encoder/reader learning rates, and publish a portable 200k synthesis bank in balanced batches with scheduler requeue. The next runner supports explicit resume with paired optimizer state and unchanged order/schedule, plus a separate-passage experiment using exactly the control's total activation-row budget. Resume parameter equality and history/identity guards pass private CPU fixtures. These runner changes live under `bench/` and add no product abstractions. The initial scratch-quota failures and completed pool replays remain separate reports.


Sep22 update: the implementation patch also rejects non-UTF8 generated question/answer text before caching. Valid Unicode and literal code escapes are preserved; four focused CPU checks and an independent review passed. The patch was reapplied to the frozen baseline and all staged hashes matched. See [recovery details](SYNTHESIS_UNICODE_RECOVERY.md).

Sep22 diagnostic update: the patch now covers ten product files. One additional line in `ac_model.py` excludes no-grad evaluation forwards from training delta/cosine buffers, matching the reader-residual scope. A real tiny-model check reproduced the old mismatch and confirmed bitwise-identical training loss, all parameter gradients and evaluation target log-probabilities after the correction. Focused review passed; no GPU check was needed for detached telemetry. Current experimental snapshots remain unchanged. See [measured stability evidence](STABILITY_1_8.md).

Sep22 chat-stopping update (CPU checks/review and real9B argument A/B passed): The latest local decoder correction preserves inherited EOS IDs and also stops at the existing model-description chat-end token. It adds no configuration surface. Focused review and real tiny-model CPU checks pass for both ordinary and deep-intervention decoding: exact shared prefix, inherited int/list/None handling, absent-EOT fallback, returned IDs and configuration/method restoration. The real9B argument A/B also passed: the old decoder generated160 tokens, including113 after chat end; the correction stopped at47 with the exact same prefix and unchanged shared configuration. All seven forthcoming fixed100 readouts now use explicit chat-end stopping and retain actual returned IDs. Running training snapshots remain unchanged. See [decoder evidence](EVALUATION_CHAT_STOPPING.md). No teacher-forced NLL path changed.
