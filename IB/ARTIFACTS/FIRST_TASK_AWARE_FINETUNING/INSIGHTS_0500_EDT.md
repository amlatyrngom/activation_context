# 5am EDT checkpoint — September 22

AC is carrying useful information, but low gold-answer NLL still overstates generated-answer quality. The 27B-supervised 1/16 system is the strongest completed system on mean development NLL so far. The 4B encoder had an early advantage; that advantage is now mixed across cohorts, so its final result matters.

## Completed 16,000-question runs

All rows are **16,000 unique questions × 1 epoch**, with a 2B encoder and 9B reader. Each evaluation cohort contains the same 200 questions across systems. NLL is nats per gold answer token; lower is better.

| Supervision | Compression | 9B-authored evaluation | 27B-authored evaluation | Learning NLL, fixed 32 questions |
|---|---|---:|---:|---:|
| 27B | 1/16 | .643 | .564 | .424 |
| 27B | 1/8 | .667 | .625 | .522 |
| 9B | 1/16 | .638 | .678 | .457 |

Learning panels use the same 32 questions within each supervision bank, but different questions across banks. The two supervision pipelines differ in data and generation settings as well as model size. These are practical system comparisons, not isolated oracle-size effects. The original five-arm reserved confirmation is queued and has not been evaluated.

The **9B-supervised 1/8 run is still running**. After 14,400 of its 16,000 questions, it scores **.604 / .662**, with learning NLL .427. Extra rows are helping this run later in training; the completed 27B comparison shows that the benefit is not automatic.

## What the 100-question response review shows

For the two completed 27B-supervised runs, every response has been checked against the supplied evidence. On the same 87 questions with supported references:

| View | Fully supported answers |
|---|---:|
| Frozen initial full context | 78 / 87 |
| Frozen initial no context | 15 / 87 |
| AC, 1/16 | 66 / 87 |
| AC, 1/8 | 51 / 87 |

All 100 questions remain visible and included in NLL. The other 13 have ambiguous, incorrect or insufficient references. These are transparent agent annotations, not independently adjudicated test accuracy. The 1/8 system gained two fully correct answers and lost seventeen relative to 1/16. Exact numbers, table associations and code conditions remain important failure modes.

Both AC systems nevertheless have lower NLL than the frozen full-context reader on these 100 questions. Teacher-forced likelihood and generated factual correctness are giving different answers. [Inspect all sources and responses](MODEL_COMPARISON_COHORT100.tressoir.html). Update at 05:08 EDT: the 9B-supervised 1/16 response column is also fully reviewed: **52/87** supported answers, versus 66/87 with 27B supervision. It gains five and loses nineteen fully correct answers. The 9B/1/8 and 4B columns remain pending.

## The larger encoder

The 4B/P4 system uses the same 27B training questions, 1/16 ratio and learner recipe as the 2B/P2 counterpart. Its early advantage appeared on both cohorts after 3,200 and 4,800 questions. At **9,600 questions**, its NLL is **.728 / .667**, versus **.768 / .625** for the matched 2B run: better on one cohort, worse on the other. Its learning NLL is .613. The initial reconstruction checkpoints and paired reader adapters differ, so this does not isolate encoder capacity or establish a need for full fine-tuning.

The 4B run is expected to finish around **06:20 EDT**, subject to preemption. Its final response review and the unchanged five-arm confirmation follow. Its ORCD work may continue past 08:00 EDT under the user's explicit extension.

## What follows

- Finish the current five-arm comparison before choosing a successor. The strongest completed 27B/1/16 final pair is already preserved in checksum-verified S3 storage.
- Complete the short/32k cross-evaluation of the earlier adaptation experiment. Then test the newer final pairs on the same fixed 100 questions with roughly 2k versus 32k evidence tokens. This measures added-distractor robustness; it does not establish multi-document reasoning. Actual row counts and the 4,096-row cap will be reported.
- Keep the next data-scale hypothesis at **32k–64k unique questions × 1 epoch**, with 10% panels. The earlier equal-compute study favored fresh questions over repeatedly cycling 2k; these newer curves still do not prove convergence at 16k. That recommendation is a planning hypothesis, not a completed experiment.
- The reusable 9B bank has **152k published questions toward 200k**; the 27B bank has **32k**. The near-150k source check passed for all nine families, including law and office. No extra downloads or pruning are needed; recheck at 180k or the code-extension boundary.

Four active autonomous MIT GPUs maximum; no EC2 experiment nodes. Native monitoring continues through the main 08:00 EDT handoff, with phase-boundary checks. [Live overview](../../TMP/SYNC/TASK_AWARE_QA/report.tressoir.html) · [4B run](../../TMP/SYNC/TASK_AWARE_QA/encoder4b_oracle27b_16k_r16/report.tressoir.html) · [Current code handoff](IMPLEMENTATION.tressoir.md).
