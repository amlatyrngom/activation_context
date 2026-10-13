# 7pm EDT checkpoint — September 21

Fresh questions are the strongest improvement found so far. At the same 8,000 question presentations, larger unique pools led on both development and a separate reserved confirmation cohort. This supports expanding the data before repeating small pools; it does not yet establish the best size for a much longer run.

| Unique questions × epochs | Development AC NLL | Reserved confirmation AC NLL | Learning NLL |
|---|---:|---:|---:|
| 2,000 × 4 | 1.008 | 1.078 | .131 |
| 4,000 × 2 | .835 | .895 | .437 |
| 8,000 × 1 | .783 | .844 | .469 |

All three used the same architecture, learning rates, balanced source mix and retrieval corpus. Encoder-token presentations differed by less than 0.7%. The 8k-over-4k confirmation advantage is modest: a descriptive paired-question 95% bootstrap interval includes zero. Both have a clearer advantage over 2k×4. Confirmation evaluated the already-fixed final checkpoints; that cohort is now used and will not guide further tuning. The 8k×1 final AC/reader pair is preserved locally, on ORCD and in a checksum-verified S3 archive.

## Answer quality needs decoded examples

The completed [100-question review](QUALITATIVE_COHORT100.tressoir.html) covers the earlier 4,000-question run. On its 88 supported-reference questions, fully supported answers were 85 for the frozen full-context reader, 11 without context, 53 after 4k×1 pass and 59 after 4k×1.8 passes. These are descriptive agent judgments, not independent test accuracy. Six defective and six ambiguous references remain visible.

This supports useful information transfer but exposes a semantic gap that mean NLL understates. Gold-prefix conditioning can hide a wrong generated fact, and the NLL mask does not penalize an invented suffix after the reference ends. The [new matched report](DATA_BUDGET_COHORT100.tressoir.html) completed at 23:34 UTC: on 95 questions without identified reference/context problems, supported answers were 49 for2k×4,52 for4k×2 and57 for8k×1, versus87 frozen full-context and24 frozen no-context. All100 examples and500 responses remain visible. More unique questions helped, but the semantic gap is much larger than the NLL comparison suggests:8k×1 has lower NLL than the frozen full reader while answering substantially fewer questions correctly. These are transparent agent annotations of development data, not independently adjudicated accuracy.

## Longer inputs and alternative rows

A frozen-checkpoint screen, using the earlier 4k×1.8-pass pair and the same 100 short-context questions, found:

| Row construction | AC NLL |
|---|---:|
| Merged evidence, 1/16 | .589 |
| Separate question-conditioned passages, equal total rows | .604 |
| Merged evidence, 1/10 | .588 |
| Merged evidence, 1/30 | .618 |

Simply adding rows did little. This is transfer without additional training; it does not settle which layout would train best.

The matched adaptation study reuses **640 previously seen questions × 1 additional pass** from the earlier 4k×1.8-pass checkpoint. At roughly 2k evidence tokens and up to three distractors, held AC NLL worsened from .592 to .661 while learning NLL improved from .488 to .336. Repeating this small familiar subset with a restarted schedule did not help held data.

At roughly 32k evidence tokens and up to 63 distractors, initial AC NLL was .876, versus .377 for the current full-context reader. After 0.2 additional passes, AC improved to .855. The node was then preempted; paired recovery is queued. This is preliminary. A planned cross-evaluation will test the initial and both final models at both context lengths, measuring long-input gains and short-input retention.

## What runs next

- The reusable banks hold approximately **53k balanced 9B questions and 6k balanced 27B questions** as of 22:58 UTC. Each 1k release contains the agreed nine-family mix, including law and office material. The 9B target remains 200k; both banks support resumable production and portable selections.
- Once the 27B bank has enough headroom, freeze **16k questions × 1 epoch per supervision source**. Both arms start from P2 and use the same learner recipe. Shared, separate 9B- and 27B-authored evaluation cohorts distinguish training-source alignment. Revised generation prompts make this a supervision-pipeline comparison, not a pure oracle-size comparison.
- The whole-parent synthesis pilot completed: 69 accepted questions from 75 documents across 8k/16k/32k, with zero temporary-label rejections. Legal/code/math yielded most examples; manuals were sparse and office yielded none. A quick source-aware sample review precedes scaling. A bounded checker-throughput probe follows the qualitative continuation; it changes no production setting until measured.
- The explanation-before-verdict checker did worse on the small diagnostic set. Production retains the original boolean checker. Stronger oracle size alone does not guarantee clean references.

The campaign uses at most four autonomous MIT GPUs, reserves user capacity, and has no EC2 experiment nodes. Reporting now uses unique questions × passes, with context and distractor limits. Routine live updates publish JSON; raw incoming reports stage privately before conversion. Next substantive checkpoint: **5am EDT /09:00 UTC**, with active monitoring and intermediate results before then.

[Live overview](../../TMP/SYNC/TASK_AWARE_QA/report.tressoir.html) · [Confirmation details](DATA_BUDGET_CONFIRMATION.md) · [Experiment ledger](EXPERIMENTS.json)
