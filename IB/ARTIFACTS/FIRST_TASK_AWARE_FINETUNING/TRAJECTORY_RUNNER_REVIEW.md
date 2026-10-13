# Trajectory runner preflight review

September 24, 2026. Reviewed the initial `bench/run_trajectory_tasks.py`, its material builder and existing resume helper. Root is editing the runner concurrently; findings below describe the inspected initial version, not a claim that each remains unfixed. No runner or trainer edits were made during this review.

**Verdict: incorrect-or-missing until the launch blockers below are resolved.** The central task semantics are otherwise appropriate: question-independent frozen windows, future QA question only in the reader, separate SFT/KL recipes, complete compaction suffixes, root-disjoint development material, one train call and one frozen compaction teacher.

## Launch blockers

- **Reference name:** `transform_for_eval_reference` accepts `full_context`, not `in_context`. The initial runner would fail during its first baseline. Root's concurrent version now uses `full_context`.
- **Fresh phase epoch numbering:** the loaded strong checkpoint carries a historical `ac.epoch_number`. Reset fresh task runs to epoch zero / first publication one. Otherwise the trainer publishes historical epoch numbers, while `run_qa.resume_checkpoint` requires a number within this new run's `1..args.epochs` and rejects the checkpoint on requeue.
- **Resume identity:** add compression ratio to the run identity. It changes rows and gradients but is absent from the initial identity. Freeze raw selected material content and tokenizer provenance, not only shard/ordinal/row IDs and choice dictionaries. A digest recorded during the existing row parse is sufficient; no separate corpus audit is needed.
- **Rollback of reporting history:** archive panels/readouts newer than the last published checkpoint before resuming. The initial runner retains them and may append lower progress, making its curve include a discarded training branch. Preserve earlier completed panels and fixed initial references; regenerate plot series from retained records. Root is already correcting early report overwrites.
- **Interrupted final publication:** before forcing the original resumed epoch number, rename uncommitted numbered epoch directories at/after `resume['first_epoch']` on both AC and reader sides. Otherwise a crash during final save leaves an orphan folder; the restored partial checkpoint trains correctly but `_save_epoch` fails forever with `FileExistsError`. The old QA runner already implements this bounded archive step.
- **Enough frozen material:** require the requested example/token budget before writing the first selection. The initial code freezes any nonempty selection. Compaction's token target is also capped by `--items`; a 50M-token run needs a sufficiently high explicit item cap, not the QA-oriented default of 16k.

## Small semantic/reporting fixes

- Compaction `gold_nll_token` is token-weighted, while `teacher_gold_nll` and `kl` are currently example means. Use matching token-weighted teacher NLL / KL for graphs labeled nats per target token, or label their weighting explicitly. Frozen full-context NLL and the stored teacher's token-weighted gold NLL should agree at baseline within ordinary numerical tolerance.
- `global_step * batch` overcounts a partial final batch in every completed epoch. Derive presentations from completed full epochs plus the current epoch's capped example count, including the resume offset. Distinct roots, unique examples, presentations and supervised tokens are different quantities.
- Keep phase-specific counters out of training status after preparation/evaluation. The top-level report already clears status at steps/panels; block progress should make clear that it is local preparation, not a new epoch.
- Enforce the actual recent-text token budget after decode/re-encode, or label the small tokenizer-boundary discrepancy. Parent already identified this check.
- Add actual history/suffix length distributions to the recipe and smoke proof. Randomly choosing one available cut per trajectory does not guarantee that eight smoke examples include a true 32k-history/8k-suffix case.
- The current runner requests zero generated completion samples. Numerical early panels are useful, but the campaign still needs a bounded decoded-answer/continuation report before interpreting lower loss as better free-running behavior.

## Source filtering and interpretation

The preparation script invokes Nemotron's row normalizer directly, bypassing the existing loader's explicit AIME2024/2025 problem exclusion used by prior studies. Retain that small existing protected-problem check. Do not expand this into new reconstruction/source-overlap audits. Source-qualified roots otherwise keep repeated task trajectories together across train/held splits.

The long continuation is teacher-forced: later tokens have already received correct suffix outputs and tool observations. Low whole-suffix KL can therefore obscure information lost at the cut. Use existing per-target log probabilities and assistant-span lengths for immediate-versus-later NLL, and a small first-turn KL readout if useful. A live continuation/task check remains a later separate quality test.

## Minimal real GPU proof

Use the actual 2B→9B strong pair, the production bounded mode, and a frozen small selection that deliberately includes one 32k-history example with a substantial complete suffix. Keep the ratio, deep heads and reader penalty identical to the intended run.

1. **Baseline oracle:** save an independently computed full-context teacher top-k/gold-logp sample before training. Record the frozen adapter checksum and selected assistant target IDs; verify teacher/student positions select those same IDs and exclude tool tokens.
2. **Update and memory check:** perform at least two real optimizer updates, so the second includes initialized Adam moments. Record encoder, reader and head gradient norms and parameter changes. Frozen teacher and base checksums/parameters must stay unchanged. Record GPU peak and process RSS; an initialized-state 32k example must fit.
3. **Resume equivalence:** for a short three-update sequence, run uninterrupted and separately interrupt after update two, load the paired checkpoint plus teacher manifest/optimizer, then run update three. Compare the final loss, learning rate, selected parameter deltas and penalty sampler state within justified GPU tolerance. CPU tests already provide exact small-model equivalence; this GPU check catches dtype, placement and real-model issues.
4. **Frozen teacher check:** after both updates and reload, regenerate the same sampled teacher targets through the saved adapter and compare with step one. Also demonstrate that the updated student-reader full-context sample can differ; otherwise equal teacher values alone are a weak test.
5. **Report check:** verify that the resumed curve contains only retained checkpoints, references remain numerically constant, and progression names examples/epochs/tokens rather than internal target blocks.

This is sufficient to gate the first large run. Do not add KL batching to this gate; if later profiling justifies it, prove unequal-length padded outputs and gradients separately.

## Synthesis producer follow-up

Reviewed the initial `bench/generate_trajectory_questions.py` at root request. Question blindness is correct: one seeded preselected window is chosen before any questions are generated, and the published record keeps that window identity. All generated questions about a window use the same encoder material. Public records contain actual questions/answers, evidence indices, row/source digests and recipe IDs; earlier outputs remain reusable after recipe changes.

Publication is correctly ordered: an exclusive producer lock, write/fsync/rename of the question shard, then atomic manifest/cursor advancement. A crash before manifest publication replays the bounded last batch; consumers read committed shards only. Empty accepted batches still advance the cursor. The source mixture is explicitly a pre-validation target, with accepted counts exposed; inspect initial yields before adding quota machinery.

Corrections sent to root:

- Group all questions for one history into **one** grounding-check request returning supported question indices or booleans. The initial producer sends one full-history check per question, producing up to five full prefills per history and contradicting its report's “one grounding check.” Prefix caching does not guarantee eliminating repeated prefill across concurrently scheduled requests.
- Bound the checker prompt length as well as the generation prompt. Appended question/answer text can make a checker request larger than the initial synthesis request; never silently truncate the history.
- Reject malformed Unicode surrogates in generated question/answer strings before publication, using the prior synthesis safeguard. Otherwise a valid JSON escape can become an unencodable Python string and fail UTF-8 JSONL output. Normal Unicode remains valid.
- Record a hash of the actual rendered system prompt, instructions and response schema in recipe provenance. Changes to the shared prompt function should become visible without invalidating old portable questions.

These are bounded corrections, not grounds for an additional validation or overlap-audit pipeline. No edits to the producer were made by this reviewer.

## Root resolution before main training

The launch blockers above are resolved in the current runner: correct reference name, independent task epoch numbering, selected payload/tokenizer/ratio/request identity, archived discarded panels and orphan epoch directories, and an explicit full requested material budget. Compaction charts use token-weighted KL and NLL. Progress counts account for partial batches and report actual supervised tokens. First/later-turn NLL, input-length distributions and fixed small decoded panels are implemented. The existing AIME source exclusion is restored in material V3. Generator checks are grouped by history, both prompts are length-checked, malformed Unicode is rejected, and rendered prompt/schema provenance is recorded.

The tiny-model fixed-teacher tests pass (6); real-tokenizer campaign construction passes on17 compaction items and7 QA items. The real GPU gate remains pending. Existing recent-text rendering uses the row budget before decoding; exact boundary retokenization is being checked separately. Approximately100-item qualitative review remains a later campaign deliverable.

### Final bounded review resolutions

Root fixed the last publication/restart findings: the shared selected teacher is copied to a temporary directory then atomically renamed and identity-checked; turn slices are part of the first panel write; qualitative decoding inherits the training deadline; rollback/final status use exact presentations; discarded qualitative files are archived and retained selected weights are explicitly annotated. SIGUSR1/TERM request a normal paired stop through the trainer deadline path. Recent-text content is now trimmed if decode/re-encode would exceed its row-count budget. Main training remains gated on actual GPU proof.

### Small startup optimization

The L40S runner smoke reuses the GPU gate's validated AUTOTUNE directory via a filesystem link, avoiding another cold calibration. The tuner still verifies hardware, library/source versions, geometry, checksum and correctness coverage. No kernel profile is fabricated or accepted under a different identity. Future campaign launchers should share this validated cache per cluster.

## Subsequent verification — September24 02:36UTC

The allocation/chunk64 L40S gate passed four real Adam updates in two processes. Teacher weights/outputs and plain-base penalty targets stayed unchanged; optimizer moments/RNG restored; both trainable sides received finite nonzero gradients. Actual source lengths7933/30763/16156/8091 and continuations2014/4499/8176/6600. Peak allocation46,510,864,896 bytes. Combined32k-history+8k-suffix capacity remains unproven. Full runner smoke is separate and still active.

The local trainer now refuses eager KL optimizer resume without a frozen teacher snapshot. Per-target evaluation KL allows first128-versus-rest readout; six bounded-teacher tests and a focused per-target length/mean consistency check pass. SourceV4:50e21b45a557fc9633a470702860eb7b688f4361.
