# What the 1/8 instability currently tells us

The 1/8 arm has a real increase in gradient volatility and a temporary regression in compressed-context answer NLL. The available telemetry does **not** identify an encoder defect or establish that more rows are intrinsically unstable. I found one concrete diagnostic-buffer error, affecting the delta/residual charts rather than training or evaluated NLL. Leave both frozen runs unchanged and inspect the already-recorded AC/reader gradient split when their final results arrive.

This is a local, exploratory readout captured on **2026-09-22 at 04:09:56–57 UTC**, with the comparison fixed to updates **1–600**. Both arms use the same 16,000 27B-generated questions, P2 initialization, evidence, logical batches of eight, and schedule; the compression ratio differs. This investigation made no GPU, remote, queue, product, or running-source changes. The parent subsequently applied the reviewed diagnostic correction to the workspace only, as detailed below. Exact report snapshots, source hashes, and numbers are in [STABILITY_1_8.json](STABILITY_1_8.json).

| Updates | 1/8 median preclip gradient | 1/16 median | 1/8 maximum / count >100 | 1/16 maximum / count >100 |
|---|---:|---:|---:|---:|
| 1–200 | 5.75 | 4.72 | 72.82 / 0 | 323.34 / 1 |
| 201–400 | 5.14 | 3.67 | 95.27 / 0 | 33.70 / 0 |
| 401–600 | 9.26 | 4.31 | 644.40 / 9 | 234.10 / 1 |

The gradients are finite and jointly clipped to norm 1.0 before AdamW. Clipping bounds the gradient supplied to the optimizer; it does not make the direction of a highly variable gradient stable.

| Fixed-question answer NLL, update 400 → 600 | 1/8 | 1/16 |
|---|---:|---:|
| All 400 development questions, AC | 0.9052 → 0.9720 | 0.9969 → 0.8940 |
| Same questions, current-reader full context | 0.6095 → 0.6071 | 0.6521 → 0.6530 |
| Fixed 32-question learning sample, AC | 0.8190 → 0.9325 | 0.9064 → 0.7276 |

The 1/8 regression is specific to the compressed path in these panels; its full-context control stays steady. Nevertheless, median **whole-assistant SFT loss on the changing training stream** falls to almost the same value in both arms over updates 401–600: 0.3444 versus 0.3438. That is a different metric and sample from fixed-question answer NLL. This is not evidence of an unbounded loss explosion. Median sampled, unweighted no-context KL is also similar (0.00866 versus 0.00804); a small scalar KL does not prove its gradient contribution is small.

Actual capacity is as intended: identical 32,938,594 training encoder-input tokens produce 4,124,307 rows at 1/8 versus 2,066,141 at 1/16. The common 400 held inputs produce 101,590 versus 50,898 rows. There is no accidental evidence-budget or passage-layout increase. Doubling rows changes the computation and optimization problem even with the same learning rate.

The sampled module traces do not show a persistent runaway delta head or attention output projection. At trace 575, both arms' eight delta heads together account for only a small fraction of the combined squared gradient norm; the exact aligned samples are in the JSON. Attention output-projection weight norms remain near their inherited P2 values: layer 3 starts at 4.8469 and reaches 4.8736 at 1/8, 4.8763 at 1/16. The new row readout grows from zero to 0.9423/0.9558; the recursive readout remains zero in this depth-one experiment. These are **weight norms**, not attention gradients. Sparse traces miss most individual spikes, so they cannot exonerate those modules at unsampled updates. The chart's generic “zero at init” and “2–10%” descriptions are not quality criteria for these warm-started heads.

There are two concrete interpretation issues in the diagnostics:

- **Step alignment:** intervention trace `update=k` is captured before optimizer step `k+1`. Its head gradients belong with combined gradient `k+1`; the projection weights are still those after `k` completed updates. Comparing those gradients at the same plotted x coordinate is off by one.
- **Mismatched diagnostic populations:** encoder cosine/delta-RMS collection checks only `mode == TRAINING`. Evaluations and decoded samples deliberately use that mode under `torch.no_grad()`, and returning to the unchanged mode does not clear buffers. Reader residual-RMS collection already checks gradient mode. Consequently, the next ratio can divide encoder values partly from evaluation by reader values from training. The 256-entry encoder cap can also discard older training values. Avoid interpreting traces immediately after panels as measurements over matched training examples.

The reviewed one-line correction is in `activation/ac_model/ac_model.py`, line 551 of the frozen source:

```python
if self.mode == TRAINING and torch.is_grad_enabled():
```

Apply the guard to the existing **diagnostic collection block**, before its inner `torch.no_grad()`; do not gate computation of the actual deltas. Adding the check inside that inner block would incorrectly suppress training telemetry too. The block is outside the checkpointed mixer, pooling, and side-decoder calls, so their backward recomputation does not duplicate these entries. Training uses non-reentrant checkpointing and has gradient mode enabled here. This correction leaves outputs, losses, and gradients unchanged; it prevents evaluation values entering the training buffers.

A focused CPU reproduction executes the actual frozen diagnostic block and helper methods with a tiny checkpointed differentiable decoder. The old code retains 256 evaluation entries; the guard retains zero, then exactly one entry after a training forward/backward despite two decoder executions. Outputs and input/parameter gradients match exactly. Receipt: [diagnostic_guard_cpu_check.json](../../TMP/FIRST_TASK_AWARE_FINETUNING/stability_1_8/diagnostic_guard_cpu_check.json).

**Focused review verdict: pass.** The parent also reproduced the defect through the real tiny-model harness/trainer, without mocking the encoder: after two training examples, evaluating three examples increased encoder counts from 2 to 5 while reader residual counts stayed at 2. With the workspace correction, both remain at 2. Training loss, every parameter gradient, and all evaluation target log-probabilities are bitwise identical to the preserved pre-fix tensors. I reviewed the exact one-line diff, fixture, checkpoint boundaries, and both logs. This is sufficient for the telemetry correction; no GPU check is needed. Evidence: [real fixture](../../TMP/FIRST_TASK_AWARE_FINETUNING/check_training_diagnostic_scope.py), [old log](../../TMP/FIRST_TASK_AWARE_FINETUNING/diagnostic_scope_old.log), [fixed log](../../TMP/FIRST_TASK_AWARE_FINETUNING/diagnostic_scope_fixed.log). Workspace file SHA-256 is `cdf705cfc3dcf42740378a46d8a9a66950be016a8c3a9bc73df8ff8c005aedc3`; all active/pending experimental source snapshots remain unchanged. The checks establish CPU numerical equivalence, not a new large-model GPU benchmark.

The inspected frozen update path computes per-example token means, averages the eight-example logical batch, zeros gradients per update, clips AC and reader jointly, then performs one optimizer step and invalidates encoded rows. I found no obvious ratio-dependent normalization or stale-row error in this path. Physical microbatch ordering is sorted by student length, so changing row counts can also change pairing and which examples receive RNG-selected no-context penalties; equal logical batches are not a guarantee of identical physical batching.

The next useful diagnostic requires **no new GPU job**. The trainer records `grad_norm_ac` and `grad_norm_reader` at every update and stores both in final `result.json`; its full `intervention_trace` also includes attention gradient norms omitted from the live chart. Inspect those arrays at the existing spike updates, with the alignment above. Only then consider a narrowly targeted follow-up. The present findings support preserving the fixed comparison and correcting telemetry separately, not a speculative mid-run learning-rate or architecture change.


## Completed one-pass gradient split — 2026-09-22 06:07 UTC

The final records localize the extra volatility to the **AC parameter group**, which includes encoder LoRA and the compression/intervention modules. They do not isolate the encoder itself. Reader-LoRA gradient magnitudes remain similar between ratios. Both runs completed all 16,000 questions once, with finite recorded gradients throughout.

| Whole-pass preclip norm | 1/16 | 1/8 |
|---|---:|---:|
| AC median / 95th percentile / maximum | 6.09 / 15.81 / 323.02 | 11.53 / 170.61 / 6,601.86 |
| Reader median / 95th percentile | 1.24 / 1.94 | 1.27 / 1.97 |
| Logical batches with joint norm >100 | 2 of 2,000 | 174 of 2,000 |
| Minimum AC share of squared norm among those batches | 99.80% | 99.85% |

The increase starts after roughly **3,200 questions (20% of the pass)**. Across questions 3,201–4,800, AC median norm is 9.11 at 1/8 versus 4.07 at 1/16, with nine versus one >100 batches. During the last 1,600 questions it is 20.57 versus 6.86, with 23 versus zero. Reader medians in those same windows are 1.35 versus 1.31, then 1.14 versus 1.09. This is persistent AC-path gradient volatility, not a single startup outlier or a reader-gradient explosion.

These norms are measured **before the single joint clip at 1.0**, as verified in the frozen trainer. AC spikes therefore also scale down the reader gradient entering AdamW: its median postclip norm is approximately 0.109 at 1/8 versus 0.198 at 1/16. This is gradient scaling, not a claim about proportional AdamW parameter changes or an effective learning rate. The no-context penalty has no direct AC gradient path: its reference embeddings are prepared without gradients and its forward contains no AC payload. That does not exclude indirect effects through earlier reader updates.

The sparse per-module trace catches only two of the 174 large 1/8 events. At 12,008 questions, the AC norm is 134.53 while the measured delta heads and passage/row readouts jointly account for only 0.0031% of its squared norm; at 13,608 questions the share is 0.0167%. Thus those particular spikes arise predominantly elsewhere within the AC group. There is no basis to choose between encoder LoRA, mixer, input/target heads, or other unmeasured parameters from these records. Parameter-gradient traces are unaffected by the diagnostic-buffer defect; no delta-RMS, residual-RMS, or cosine ratios enter this supplement.

The 1/8 arm nevertheless recovers after its early regression: all-400 AC NLL goes from 0.972 after 4,800 questions to 0.646 after 16,000. The 1/16 final is 0.603; each arm's own full-context reader scores 0.627 and 0.644 respectively. The fixed 32-question learning sample ends at 0.522 versus 0.424. This supports optimization difficulty as a plausible contributor to the weaker 1/8 AC result; it does not prove causality, an architecture bug, or an intrinsic disadvantage of extra rows. Actual decoded/source-grounded comparison remains the next priority.

If one subsequent intervention is warranted after that comparison, the smallest is a matched 1/8 repeat with **AC-group peak LR 2.5e-5 instead of 5e-5**, leaving reader LR, data order, clipping, schedule, and initialization unchanged. Treat it as a hypothesis test, not a repair established by these statistics. No follow-up was launched and neither fixed run was changed.

The JSON's `completed_run_gradient_supplement` preserves result hashes, per-10%-pass summaries, extreme batches, aligned module traces, and final panels. Reproduction uses [analyze_final_gradient_split.py](../../TMP/FIRST_TASK_AWARE_FINETUNING/analyze_final_gradient_split.py); local frozen result copies sit beside its evidence. Earlier observations above are retained as the contemporaneous partial-run record.


## Completed 9B-supervised comparison — 2026-09-22 09:32 UTC

The ratio effect reverses in the other supervision bank. With the same 16,000 9B-generated questions, initialization and schedule within this pair, **1/16 is the volatile arm and 1/8 has better final NLL**.

| 9B-supervised, whole-pass preclip norm | 1/16 | 1/8 |
|---|---:|---:|
| AC median / 95th percentile / maximum | 12.55 / 312.63 / 4,171.85 | 8.32 / 23.12 / 166.38 |
| Reader median / 95th percentile | 1.28 / 2.03 | 1.27 / 2.05 |
| Logical batches with joint norm >100 | 289 of 2,000 | 2 of 2,000 |
| Final answer NLL, 9B-authored / 27B-authored development questions | .638 / .678 | .580 / .644 |
| Fixed learning-question answer NLL | .457 | .435 |

Within each supervision bank, the more volatile arm has worse final development NLL. Extra rows are therefore not intrinsically responsible for the instability seen in the 27B-supervised run. This strengthens optimization sensitivity as a follow-up hypothesis; it does not prove gradient spikes cause the quality gap, isolate an encoder module, or establish that a different learning rate will fix it. Different data, answers and physical microbatch pairings can interact with the training trajectory; only one seed was run. The 9B/1/8 generated-answer review is still pending.

All 2,000 AC and reader gradient records are finite and aligned to the completed logical batches. [STABILITY_9B_RATIOS.json](STABILITY_9B_RATIOS.json) records raw result hashes, full-pass statistics and 10% windows. This uses only the already-recorded parameter gradient norms, excluding the contaminated RMS/cosine diagnostics described above. The original 27B evidence remains intact.
