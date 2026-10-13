# Strong-checkpoint long-context adaptation — preflight review

Status: implementation review passed after corrections; **frozen and queued23514939**. GPU budget alternatives failed the numerical gate; original batching retained. Final request9h/64GiB host with8.5h training ceiling. Root reviewed the original-to-new runner diff, actual frozen trainer evaluation ordering, reporting API, material selection and paired-final evaluator.

The new run keeps the established item builder, row allocation, masks and resume helpers. Its initial checkpoint is the preselected complete32k pair; it starts fresh Adam explicitly, and its own recovery preserves the original1000-update schedule, optimizer and question order. Material is a balanced8k subset of the already frozen fresh10k, retaining the same learning32, original400 development questions and retrieval corpus. No reserved confirmation is read or used.

Three concrete reporting findings were corrected:

- The new table no longer receives the internal `update` key that its declared columns reject. The machine-readable panel file retains that clock.
- Resume trims the new question-count table as well as the plots. In particular, an unsaved final8,000-question panel is removed when recovering7,200 saved questions.
- Descriptions now distinguish the initial-reader frozen controls from moving-reader measurements at0/50/100%. Reference identity also includes tool definitions, which form part of the model input.

Focused CPU checks use the actual HtmlReporter callback and its key validation, covering0/10/50/100%, sparse moving-control points, frozen-cache reuse and negative identities, and final-panel recovery. Full config equality and final-pair negative cases pass. Root re-reviewed the corrected table, resume, cache and description code. Input-builder and unchanged resume proofs are reused; no unnecessary GPU rerun of those interfaces was added.

The initial/final evaluator uses the same fixed100 IDs at2k/8k/32k. Both conditions use passage-parts layout and must agree on AC inputs, full/no-context inputs, gold and actual row budgets. The8k path replays all400 before retaining100 and checks any available original initial/final panel within the existing tolerance. Final inference requires the run's own complete8,000-question/1000-update paired checkpoint.2k/32k start with NLL-only transfer;8k includes actual generated answers.

Launch preparation completed: reviewed failed budget proof23511648 and retained original batching; froze design/config/helper hashes; staged and remotely verified all small files; CPU callback/config checks rerun after time-ceiling change; submitted/recheckedheld/released training and readout chains. First real training progress remains pending. A two-case memory check does not prove every batch in the new8k pool fits. The local report projection now recognizes the new arm, preserves its question-count axes and sparse moving controls, and derives last-complete-panel progress from the actual question count. Focused checks confirm ordinary progress updates change JSON without rewriting HTML (STRONG_LONG_REPORT_PROJECTION_CHECK.json).

Evidence: `STRONG_LONG_ADAPTATION_PREPARATION_V1.json`; `TMP/FIRST_TASK_AWARE_FINETUNING/STRONG_LONG_CALLBACK_CPU_CHECK.json`; `STRONG_LONG_EVALUATOR_CPU_CHECK.json`. This review covers private experiment code; product source remains unchanged.
