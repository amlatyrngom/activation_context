# Code consolidation for trajectory QA and compaction

**Historical cleanup package:** the [current trajectory handoff](TRAJECTORY_IMPLEMENTATION_HANDOFF.tressoir.md) supersedes this package for the same `/source` baseline.

The cleanup removes the obsolete single-question synthesis path while retaining the reusable model, training and reporting work. The source-sync package below combines that cleanup with the remaining task-aware fixes against the current `/source` tree. It supersedes the earlier implementation patch; apply only this package to that baseline.

## What changed in this cleanup

- Synthesis has one prompt builder and one `questions` array schema. It no longer constructs a legacy response format and removes it by splitting a string. This also fixes loss of study-context text containing `# Response Format`, and removes obsolete `has_valuable_question` instructions from the current prompt.
- Removed the unused single-question row converter and unused prompt re-exports from `dataset_study.py`.
- Updated the one remaining product caller, the small-generator throughput benchmark, to the same prompt/schema and usable-pair counting. An empty questions array or null answer does not count as generated QA.

These are three files and **49 fewer lines** than the checkpoint. There is no new utility module or configuration option. New locally generated requests receive a new recipe hash because one prompt instruction is corrected. Existing portable banks remain readable and unchanged; the running 27B producer uses its existing frozen source.

## What remains useful

The other eight packaged files carry previously implemented fixes, not fresh refactors: source selection and evidence provenance, matched RAG full-context distractors, reconstruction ratio sampling, reporting callbacks with state restoration, time-bounded paired recovery, no-context cache identity, chat-end stopping, training-only diagnostics, memory-efficient passage attention and UTC report timestamps. The exact diffs distinguish these from changes already present in `/source`.

Question-blind trajectory encoding, complete-turn compaction, nested/multiple parts, paired checkpoint loading, row tensor sidecars and fixed training targets are preserved. The optional pooling and recursive-key features belong to the separately accepted reconstruction/nesting stream and are already in `/source`; cleanup does not delete them. Golden and proposed trajectory runs keep pooling disabled. Experiment-specific clipping, moment-offload and layout controllers stayed in private bench files and were not promoted to product APIs.

Historical reports, frozen launch sources, checkpoints and synthesis data remain in place. The unrelated existing `agentic_program_probe.py` and all tests are excluded from this source-sync patch. No remote job or active generator source was changed by this cleanup.

## Apply the current package

[Review findings](CONSOLIDATION_REVIEW.md) · [Exact patch](consolidation/changes.patch) · [Before/after hashes and staged files](consolidation/manifest.json) · [Full per-file review](CONSOLIDATION_HANDOFF.tressoir.md)

The package contains **11 files** under `consolidation/activation/...`, compared directly with `/source` at packaging time. Check each destination's SHA-256 against `before_sha256` in the manifest; a null baseline means the file is new. Already-matching `after_sha256` files need no replacement. Inspect divergent files rather than overwriting them.

From the destination repository, using the actual package location:

`consolidation/changes.patch`

```bash
git apply --check /path/to/consolidation/changes.patch
git apply /path/to/consolidation/changes.patch
```

Do not apply `implementation/changes.patch` or `current_code/source.patch` afterward. Their documents remain historical records. This agent has not changed `/source`.

The complete pre-cleanup workspace is preserved on branch `checkpoint/pre-trajectory-cleanup-20260923`, commit `600137587987f13f0055c06494b788638cd0745b`. The current branch remains `main`; changes are left in the working tree for review. The backup includes historical callers of the retired prompt API for reproducing earlier experiments.

## Validation and limits

**21 CPU checks pass:** 15 focused cases cover all four prompt presets in text/trajectory modes, exact schema/instruction retention, complete study context, unchanged labeling and benchmark parsing; six existing integration checks cover source selection, multi-question caches, Unicode, matched references, ratio precedence and training callback/deadline/resume behavior. A first exact-string check caught one unintended blank-line change; that was corrected before the passing rerun.

All 11 packaged Python files parse. The patch applies to an isolated copy of the current source, reproduces every staged byte and passes reverse application. Final direct review: **pass**. The Markdown checker passes; the extension's rendered editor was not available for visual verification.

No fresh GPU run was needed for this prompt/caller cleanup. The earlier real P2 restore, training/reload, attention-gradient and chat-stopping GPU evidence remains applicable to unchanged paths; this is not a new GPU regression claim.

The proposed trajectory study still needs its own item-builder work: all supporting turns must survive window selection, full context must include the same distractors, and total input budgets and whole-trajectory splits must be explicit. Existing generic training plots still expose update diagnostics; a future trajectory report must lead with unique questions × epochs. None of those future-study requirements is claimed implemented by this consolidation.
