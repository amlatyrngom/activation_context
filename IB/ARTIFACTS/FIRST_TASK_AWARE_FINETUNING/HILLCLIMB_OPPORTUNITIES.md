# Task-aware QA hillclimbing opportunities

Read-only investigation, 2026-09-21. No jobs launched and no training/evaluation material changed by this investigation. Evidence comes from the local pilot reports, `bench/run_qa.py`, the trainer, adapter manager and decoder forwarding code.

## What the pilot establishes

| Arm / update | Held AC answer NLL | Matched full-context NLL | Fixed training-panel NLL |
|---|---:|---:|---:|
| P2 joint / 0 | 2.8202 | 0.5447 | 2.6388 |
| P2 joint / 74 | 1.0606 | 0.4480 | 1.2500 |
| P2 joint / 185, complete | 1.2334 | 0.4482 | 1.2935 |
| Fresh joint / 185, complete | 1.4829 | 0.4336 | 1.5615 |
| P2 reader frozen / 93 | 1.0681 | 0.5447 | 1.0836 |
| P2 reader frozen / 111, latest inspected | 1.1202 | 0.5447 | 1.2647 |

P2 transfers better than a fresh AC at equal QA budget. Joint training adapts faster initially, but both joint and frozen-reader curves regress later. This is not explained solely by overfitting held questions: the training panel also worsens. However, that panel is the first 32 training examples, so it specifically measures retention of early examples. For the new larger experiment, choose the fixed training panel across the full frozen training selection.

The 12 decoded joint samples demonstrate real errors in values and identifiers, alongside correct paraphrases that exact match marks wrong. Final normalized exact match is 0/12 while token F1 is 0.405. Examples include an incorrect capacity, a date off by ten years and generic cache statistics replacing the supplied statistics. Preserve these samples; do not optimize wording to inflate exact match. Decode the same small full-context panel at the final comparison so semantic quality can be compared directly.

The full-context curve uses the current reader and is a strong empirical comparison, not a mathematical ceiling. A task-aware compressor could eventually help by suppressing distracting information; no observed result establishes that yet.

## Ranked comparisons

1. **Slow reader adaptation: encoder/heads 5e-5, reader 5e-6.** Compare with the existing Golden-rate joint configuration on the same larger frozen pool. Keep initialization, ratio, literal SFT, fixed-base no-context KL, schedule and update count identical. Frozen-reader performance around midpoint makes a smaller nonzero reader rate a useful middle point. This is a bench configuration change only; the reader rate already has a supported setting.

2. **Slow encoder adaptation: encoder/heads 2e-5, reader 2e-5.** Compare directly with the same Golden-rate control. Some regression persists with the reader frozen, and the joint encoder gradient norm is frequently clipped (median 4.38, maximum 23.77 before clipping). That supports testing encoder step size rather than treating reader drift as the only cause. Gradient magnitude alone does not prove the rate is too large. This is also a bench configuration change only. Combine both lower rates only after the individual comparisons justify it.

3. **Compress each passage separately, at the same nominal 1/16 ratio.** The current bench concatenates up to four shuffled passages into one query-conditioned AC part. Build one existing AC part per passage, with the same question inside each part. Preserve exact evidence, order and distractors in the full-context control. This may preserve short factual details and let reader attention select between passages instead of requiring one encoder representation to handle the entire distractor bundle. It needs a small private material-builder option, no new model abstraction. Record total realized rows because repeated questions, wrappers, rounding and minimum rows add overhead. A result using more rows must disclose that cost. First run a small encode/backward smoke and verify that expanding all parts reproduces the same passage sequence.

4. **A 1/10 row-budget diagnostic.** Change the bench's explicit ratio everywhere from 1/16 to 1/10 while keeping the model and training recipe fixed. This tests whether the remaining gap is strongly capacity-limited and stays inside the frozen Golden ratio range. It is an accuracy/cost tradeoff, not a victory at the original compression ratio. Prefer this after the learning-rate comparisons if simple rate changes do not close the gap.

## What not to add first

- Query conditioning is already present inside the compressed part. Adding another query-conditioning abstraction would not test a missing capability.
- Do not reset trained P2 heads or abandon reconstruction initialization based on its initial QA NLL. Initial P2 samples tend to reconstruct instead of answer; QA training changes that behavior, and the final P2 result beats the fresh arm.
- Do not remove no-context preservation merely because answer NLL remains above full context. Its mean reported KL is about 0.0115 for joint, versus substantially larger SFT loss; there is no evidence it numerically dominates. Gradient conflict remains possible. If later isolated, first compare penalty sampling fraction 1.0 against 0.25 with the same lambda, which tests variance without changing the reference or expected un-clipped objective.
- Defer answer-only loss. The current token objective already spends almost no loss on correctly learned wrappers at trained panels, while P2 needs an initial format transition. Explicit answer weighting would require consistent handling in both batched and single-example SFT paths and a clear decision about the no-context penalty's token scope. It is more invasive than the rate/layout tests. Do not silently repurpose whole-item weights: they scale the penalty too.
- No moving teacher. If fixed full-context distillation is tried later, cache the teacher at train entry before updates, as the current KL path requires. A combined SFT/distillation objective would be a separate deliberate change. The current SFT target is literal synthesized answer text.

## Focused correctness inspection

No obvious optimizer or callback-state fault was found in the inspected code. Each update clears gradients, computes the jointly clipped norm and uses the recorded schedule. The callback restores adapter routing, trainability, training/checkpointing modes and residency. Decoder checkpoint recomputation restores its adapter scope. The no-context reference is stored with adapters disabled before updates. Checkpoint serialization clears row caches but does not replace live optimizer parameters. This is source inspection, not proof against every numerical/runtime bug.

One useful private GPU diagnostic, if instability persists, is to take a fixed small training batch at a saved pair, record parameters/flags and its loss/gradient norms, run the normal evaluation callback without an optimizer step, and repeat the same batch. The callback should preserve the loss and parameter identities within the normal numerical tolerance. There is no reason to repeat a broad lifecycle certification now.

## Selection without changing the target

Use the existing 100-question pilot panel explicitly as development data once it guides experiments. Freeze a fresh 200-question confirmation selection, grouped by synthesis source request, before inspecting its answers or scores. Exclude those generated questions from every training selection. Select a recipe/checkpoint using development answer NLL, with decoded samples and no-context drift as accompanying evidence. Evaluate only the selected recipe and its matched control on confirmation once; do not adapt further to that result and still call it confirmation.

Freeze larger training IDs and completed bank batch hashes at launch. New bank batches may feed later runs while generation continues, but do not mutate an active run's data. Keep one train call and one coherent update clock per arm. If a recipe changes mid-run, create a new named continuation with its starting pair, update and schedule recorded; retain the original curve. Larger data and ordinary repeated epochs are appropriate before introducing architectural complexity.

Validation should stay small: configuration/material identity checks for rate arms; one paired-state check if a callback fault is suspected; one tiny multi-part gradient and evidence-order check for separate-passage compression. No benchmark-overlap audit, exhaustive synthesis judge or full reconstruction rerun is needed.
