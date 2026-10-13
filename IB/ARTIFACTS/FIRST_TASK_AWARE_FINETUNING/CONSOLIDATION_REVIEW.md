# Consolidation review

**Verdict: pass.** The completed cleanup is deliberately small: the RAG campaign kept its experimental controllers outside the product code. The justified deletion is the superseded single-question synthesis machinery. Removing the retained trainer/model features would discard useful trajectory/compaction behavior or separately accepted reconstruction work.

## Scope and baseline

Reviewed the product changes in `activation/ac_model`, `activation/dataset` and shared reporting, then searched all product callers before deleting APIs. The preserved baseline is `checkpoint/pre-trajectory-cleanup-20260923` at `600137587987f13f0055c06494b788638cd0745b`. The final source handoff is separately based on the actual current `/source`, not that older Git HEAD. Mainline tests, remote jobs, the frozen 27B generator, synthesis banks and scientific results were not modified.

## Findings and implementation decisions

| Finding | Classification | Resolution |
|---|---|---|
| `make_source_study_prompt` calls the legacy prompt and cuts at the first `# Response Format`. It retains the obsolete junk-response instructions and can discard user study context after that heading. | genuine | Build the current prompt/schema directly in the existing prompt file. Preserve every other ordinary instruction and the complete supplied context. |
| `_example_from_row` and legacy prompt/schema re-exports have no current dataset-generator caller. | genuine | Remove them; keep the current request-to-many-QA conversion and source metadata. |
| `cheap_synthetic_bench` is the sole remaining product caller of the old schema. Deleting the exports alone would break it. | genuine | Migrate that caller to the shared array schema and reject unusable answers in its throughput count. Historical code stays in the backup. |
| Delete optional pooling/recursive-key code because Golden runs use no pooling. | bad-review | Those additions belong to the independently accepted September 18 reconstruction/nesting work and already exist in the user's source. Preserve them, along with original-checkpoint migration and trained readouts; future canonical runs select no pooling. |
| Remove reporting callbacks, time limits, cache identity, paired recovery, profiling and diagnostic traces as experiment-only machinery. | reviewer-overkill | These are reusable correctness/operating boundaries for preemptible training and long contexts. No redundant replacement was introduced. Keep the no-grad diagnostic guard and memory-efficient attention fix. |
| Delete the many private benchmark versions and archived reports to simplify the tree. | reviewer-overkill | Frozen experiment source and evidence are useful provenance and do not expand the product API. Preserve them; the backup supplies an additional restoration point. |
| Older handoffs are based on an earlier source state and could invite duplicate application. | genuine | Publish one current 11-file patch, complete per-file diff cards and hashes; mark the older handoff surfaces superseded. |
| The trajectory builder does not yet implement the newly proposed full study contract. | genuine, deferred to that study | Record all-support preservation, matched distractors, combined budgets and whole-trajectory partitioning as future item-builder work. Do not disguise a study implementation as cleanup. |

## Validation

Fifteen new private CPU checks pass. The exact prompt comparisons cover four presets and both text/trajectory modes; the only normal prompt-content change is the obsolete junk-response instruction. Context containing the former delimiter survives intact. The label prompt is byte-for-byte unchanged. The existing benchmark uses the same prompt and schema, and counts successful, empty, null-answer and unsupported-evidence fixtures correctly.

Six existing task-aware integration checks also pass: source modes, request caching, multiple questions, Unicode rejection/preservation, matched references, reconstruction ratios and callback/deadline/resume behavior. No tests were added to the mainline handoff. An initial exact comparison failed on an unintended blank line; the implementation now preserves that line and all checks pass.

The source package parses all 11 files, applies forward to a private copy of `/source`, produces byte-identical staged files and passes reverse application. Its rendered Markdown source passes the artifact checker. The extension editor itself was not available; no rendered light/dark inspection is claimed. No new GPU run was performed because the changed behavior is prompt assembly and a caller parser. Existing model/optimizer/serving boundaries were left intact.

## Final direct re-review

All in-tree references to the retired prompt/converter symbols are gone. Current callers keep the same public programmable-synthesis signature, with a default text modality added for the migrated single-source caller. The corrected prompt changes new request-cache identities but does not invalidate the independently readable published bank. The running generator is pinned to its original source. Three cleanup files have 49 fewer lines; the larger handoff includes eight unchanged-by-cleanup correctness files still awaiting source synchronization. No mass formatting, new abstraction, mainline test change, source-tree overwrite or archival deletion was introduced.

See [the handoff](CONSOLIDATION_HANDOFF.md) for exact application instructions and the remaining trajectory-study boundary.
