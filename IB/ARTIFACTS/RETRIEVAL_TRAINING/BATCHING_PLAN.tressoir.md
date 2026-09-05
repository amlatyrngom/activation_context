# Batching-level training throughput

Retrieval training does about 7 examples per second on one RTX PRO 6000, so a 100k-example epoch takes close to four hours. This plan proposes two optimizations that live in `activation/retrieval/retrieval_batching.py` and only change how work is grouped and sized, never what the loss sees. The target is a 100k-example epoch in under an hour. Candidate-pool size per example is a generation-time decision and is out of scope. Version 3 after two rounds of independent review (round 1: 18 findings, round 2: 6; every genuine finding folded in, the rest recorded below).

## Executive Summary

### Goal

Same loss, same data, same model: a 100k-example epoch (about 8 candidates per example, MS-MARCO-like lengths) in under one hour on one RTX PRO 6000, read as step time, real tokens per second and forwards per step on the existing report.

### Where the time goes today

Measured on run 3 of slice 1 (950 examples, batch 32, checkpointing on) and on real 32-query MS-MARCO batches:

| quantity | today | what it means |
| --- | --- | --- |
| step time | 4.0 s for 32 examples (7.3 examples/s) | 100k examples = 3.8 h |
| padded tokens per step | 38.6k (26 % padding) | ≈160 TFLOP per step with checkpoint recompute |
| achieved compute | ≈40 TFLOP/s, about 16 % of the card's bf16 peak | the GPU is mostly idle |
| forwards per step | 15 (11 candidate groups of 8–70 sequences, 4 query groups of 8) | each is 240–8k tokens: launch-overhead sizes for a 28-layer model, and the backward walks all 15 graphs |
| peak memory | 8.5 GB of 96 GB | batch sizing is far more conservative than the data requires |

The 15 forwards come from `embed_in_length_groups`, which cuts the length-sorted texts whenever the next one is 15 % longer than the group's first. That policy trades launches for padding; on this GPU the trade is backwards.

### The bet, stated plainly

The one-hour target at batch 32 is 1.15 s per step. At M1's padding (≈41k padded tokens per step) and without checkpoint recompute that is ≈128 TFLOP per step, so the card must sustain ≈110 TFLOP/s, about 45 % of peak, up from 16 % today. Big forwards on a 28-layer, 1024-wide model normally reach that, but it is the load-bearing assumption of this plan, so M1 starts with a five-minute measurement of it before anything else is built on it.

### Approach

| milestone | what changes | where | expected effect |
| --- | --- | --- | --- |
| M1 token-budget forwards | groups are cut by a padded-token budget (8k tokens), so a 32-query step is 1 query forward plus ≈5 candidate forwards instead of 15; one per-forward GPU sync removed; forwards per step counted | `retrieval_batching.py`; two lines in `embed_batch`; the forwards count threaded through the step shape, the stats summary and one report column; the helper check script | step time ÷ 2–3 if the compute rate rises as expected |
| M2 data-measured sizing and real-batch probe | batch size from the heaviest batch the data can actually produce; the probe runs that real batch; the sizing text prints the batch for both checkpointing settings | `retrieval_batching.py`, three trainer lines | with checkpointing off (a per-run flag, unchanged default): 1.33× fewer FLOPs at batch ≈51 |

Estimates for a 100k-example epoch, all conditional on the compute-rate measurement in M1:

| configuration | examples/s | 100k-example epoch |
| --- | --- | --- |
| today | 7.3 | 3.8 h |
| M1 (checkpointing on, batch 32) | ≈18–24 | 1.2–1.5 h |
| M1 + M2 with checkpointing off (batch ≈51) | ≈25–32 | 50–65 min |

### Dropped after measurement: length-bucketed epochs

Shuffling within length buckets was measured on real batches before it was proposed. With the best key (the example's longest candidate) and 4–8 buckets it removes 2–5 percentage points of padding at the 8k budget (12 % → 10 %), because every example brings ~8 passages spanning the whole length range, so bucketing cannot narrow the spread inside a batch. Not worth a batch-composition bias. Numbers in `IB/TMP/BATCHING/estimates_check.md`.

### Out of scope, and why

| lever | reason |
| --- | --- |
| fewer candidates per example | generation-time decision, per your instruction |
| larger batches and automatic checkpointing | raising the cap or letting the sizing turn checkpointing off changes the optimizer schedule and the in-batch pool (batch 512 at 100k examples is 196 steps per epoch): a training decision, taken per run with the existing flags, not a batching change |
| merging query and candidate forwards | possible from the batching file, but it changes how over-limit queries are truncated and saves one launch per step, which is noise after M1 |
| GradCache / two-pass embedding | buys in-batch negatives for memory, not throughput |
| reporting cadence | 10 % of run 3 only because the epoch was 30 steps; negligible at 3,000 steps per epoch |
| torch.compile, attention kernels, multi-GPU | harness levers with wider blast radius; revisit once the step is compute-bound |

## Requested Decisions

<article class="decision" data-tressoir-decision data-decision-state="unresolved"
  aria-labelledby="batching-scope">
  <header class="decision-header">
    <div>
      <h3 class="decision-title" id="batching-scope">Approve M1 and M2?</h3>
      <p class="decision-context">M1 is the batching file plus two lines in the model's embed step (a removed GPU sync and a forwards counter), the forwards count threaded into the step shape, the stats summary and one report column, and the helper check script. M2 is the batching file plus three trainer lines and one canon line. Checkpointing stays a plain flag with its current default; the batch cap stays at 128. The M1 compute-rate pre-check runs first and is reported before M2 is built.</p>
    </div>
    <span class="decision-state" data-decision-indicator role="status" aria-live="polite">Unresolved</span>
  </header>
  <fieldset class="decision-options">
    <legend class="visually-hidden">Decision answers</legend>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="batching.v2.approve">
      <span><strong>Approve M1 + M2 (recommended)</strong><small>Pre-check, M1 with its A/B, then M2 with its sizing run, on a Sky node.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="batching.v2.m1_only">
      <span><strong>M1 only for now</strong><small>Decide M2 after the pre-check and the A/B numbers.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="batching.v2.revise">
      <span><strong>Revise first</strong><small>Name the card and the change below.</small></span>
    </label>
  </fieldset>
  <div class="field decision-feedback">
    <label for="batching-v2-response">Free Response</label>
    <textarea id="batching-v2-response" rows="2" data-tressoir-input="batching.v2.feedback"
      data-tressoir-autogrow="2:6" placeholder="Add anything the choices miss…"></textarea>
  </div>
</article>

## Milestones

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M1 — Token-budget forwards</span>
    <span class="card-oneliner">Cut the length-sorted texts by a padded-token budget: about 6 forwards per step instead of 15, same loss, proven by a CPU check.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

`embed_in_length_groups` keeps its signature and its contract (same graph, same loss, original order restored). Only the cut rule changes: texts are still sorted by length, but a group closes when adding the next text would push `count × longest-in-group` past the budget, with the length estimated exactly as the model will see it: characters after the input limit, plus the query instruction for queries, divided by 4, plus EOS and the view rows. Sorting still minimizes padding inside a group; the budget bounds the number of launches. Queries and candidates are packed separately as today.

Budget: 8,192 padded tokens on a GPU, 512 on CPU. On real 32-query batches the estimate gives 5 candidate forwards at ≈12 % padding for 8k, and 3 forwards at ≈21 % for 16k; the A/B runs both. Memory does not constrain the budget: checkpointing recomputes one decoder layer at a time, so a group costs one layer's internals (≈37 KB per token, ≈300 MB at 8k), an allowance the sizing formula already carries; without checkpointing the split has no memory effect at all. The byte-pooling attention in the AC model sees about as many rows as the group has padded tokens (≈8k); its 65,535-row limit exists only on the dropout path, which is off there.

Two lines outside the batching file, both in `RetrievalModel.embed_batch`: the real-token count is taken from the CPU-side token rows instead of `attention_mask.sum().item()`, which forced a GPU sync per forward, and a `forwards_embedded` counter is added next to the token counters so the report can show forwards per step. The CPU-side count includes the view positions, exactly as the mask sum did (the mask covers the tokens and the V view rows), so the padding fraction stays comparable with run 3; all A/B arms share the same model file and differ only in the batching file.

Validation, in order:

1. CPU helper check: `token_budget_groups` cases (budgets giving 1, 2 and many groups; one text over the budget; order restore) and the invariant itself: in eval mode (no dropout) `embed_in_length_groups` at several budgets, including one-per-group and all-in-one, is `allclose` to a single `embed_batch`. This is the "same loss" proof; the Sky A/B only measures speed.
2. Existing end-to-end test (CPU, batch 2, 20 steps).
3. Compute-rate pre-check on a Sky node: 20 steps at the run-3 shape with the 8k budget; TFLOP/s = 8 × 521M × padded tokens ÷ step time. At or above ≈110 TFLOP/s the one-hour target is reachable with M2 (checkpointing off); between 90 and 110 it lands at 60–75 minutes; below 90 it is out of reach at these shapes and the plan stops for a decision.
4. Speed A/B, one epoch of the run-3 shape at batch 32, checkpointing on: baseline (a copy of today's batching file kept in `IB/TMP/BATCHING/`; `/source` has no batching file yet) versus 8k versus 16k. Compared: step time, real and padded tokens/s, forwards per step, peak memory. Pass: at least 2× real tokens/s over the baseline.

#### Planned Changes

`activation/retrieval/retrieval_batching.py · estimated_tokens(), token_budget_groups()`

```diff-python
@@ retrieval_batching.py — grouping @@
-def length_groups(lengths: list[int], tolerance: float, min_group: int) -> list[list[int]]:
-    ⋯ tolerance / min_group cut rule ⋯
+FORWARD_TOKEN_BUDGET = 8192          # padded tokens per forward on a GPU: launches versus padding, not memory
+CPU_FORWARD_TOKEN_BUDGET = 512
+
+
+def estimated_tokens(text_length: int, num_views: int) -> int:
+    """Padded-length estimate shared by grouping and sizing: characters / 4, plus EOS and the view rows."""
+    return text_length // CHARS_PER_TOKEN + 1 + num_views
+
+
+def token_budget_groups(lengths: list[int], budget_tokens: int, num_views: int) -> list[list[int]]:
+    """
+    Indices sorted by length and cut into consecutive runs whose padded size (count x longest
+    member) stays within budget_tokens. A single text over the budget forms its own run.
+    """
+    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
+    groups: list[list[int]] = []
+    for index in order:
+        longest = estimated_tokens(lengths[index], num_views)                    # ascending: the newest is the longest
+        if groups and (len(groups[-1]) + 1) * longest <= budget_tokens:
+            groups[-1].append(index)
+        else:
+            groups.append([index])
+    return groups
```

`activation/retrieval/retrieval_batching.py · embed_in_length_groups()`

```diff-python
 def embed_in_length_groups(
     retrieval_model: "RetrievalModel",
     texts: list[str],
     is_query: bool,
-    tolerance: float = 0.15,
-    min_group: int = 8,
+    budget_tokens: int | None = None,
 ) -> torch.Tensor:
-    groups = length_groups([len(text) for text in texts], tolerance, min_group)
+    """embed_batch over length-sorted groups cut by a padded-token budget, concatenated back in the original order."""
+    if budget_tokens is None:
+        budget_tokens = FORWARD_TOKEN_BUDGET if retrieval_model.device.type == "cuda" else CPU_FORWARD_TOKEN_BUDGET
+    prefix = len(retrieval_model.query_instruction) if is_query else 0            # what embed_batch will actually see
+    lengths = [min(len(text), retrieval_model.input_limit_chars) + prefix for text in texts]
+    num_views = retrieval_model.num_vectors if retrieval_model.ac_model is not None else 0
+    groups = token_budget_groups(lengths, budget_tokens, num_views)
     embedded = [retrieval_model.embed_batch([texts[index] for index in group], is_query) for group in groups]
⋯ unchanged: order restore ⋯
```

`activation/retrieval/retrieval_model.py · embed_batch()` (the two lines outside the batching file)

```diff-python
-        self.real_tokens_embedded += int(attention_mask.sum().item())             # GPU sync on every forward
+        self.real_tokens_embedded += sum(len(row) for row in token_rows) + batch_size * num_views   # same count, CPU side
         self.padded_tokens_embedded += batch_size * length
+        self.forwards_embedded += 1
```

`activation/retrieval/retrieval_trainer.py`, `retrieval_training_config.py` — the step shape gains the forwards count and `summarize()` reports `forwards_per_step`; `IB/TMP/RETRIEVAL_SLICE1/private_helpers_check.py` swaps its `length_groups` cases for the `token_budget_groups` and invariant cases.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M2 — Data-measured sizing and real-batch probe</span>
    <span class="card-oneliner">Size the batch from the heaviest batch the data can actually produce, and probe with that batch instead of a synthetic one.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

Today `recommended_batch_size` multiplies the longest text by the largest candidate count for every example, and the probe builds a synthetic batch of that shape. On MS-MARCO that charges ≈2,600 tokens per example where the heaviest real examples carry ≈1,550 and the average ≈960. The replacement measures every training example once (query tokens plus the sum of its candidates' tokens, with the M1 estimate), sorts the totals descending, and takes the largest batch whose cumulative token total fits the usable memory. That is the adversarial real batch: no shuffle of the epoch can produce a heavier one, and candidate dedup can only make a real batch lighter. The probe then runs exactly that batch (the heaviest examples with their real candidates) through the real step, so it exercises the M1 grouping, the AC model and dedup as they will run, and prints the probe batch's distinct-candidate share (warning below 0.9).

Checkpointing stays `gradient_checkpointing: bool = True` in the config. The sizing computes the batch for both settings and prints both, so a run that wants the no-checkpoint speed sets the flag and gets the matching batch. On the 96 GB GPU with MS-MARCO shapes the numbers are ≈51 without checkpointing (≈1.1 MB per token: 0.98 MB for the 28 base layers, ≈0.08 MB for the AC model's 7 layers at one window per token, ≈0.01 MB for its byte stage) and the cap of 128 with it (≈0.1 MB per token). The cap is unchanged: a larger batch is a training decision (fewer optimizer steps, a larger in-batch pool), and the trainer's per-query collision loop would also need vectorizing above a few hundred queries. A configured `batch_size` still wins and is still probed, with the heaviest `batch_size` examples.

Trainer touch: the sizing call, the probe call and the sizing string (three lines); `observed_shape` and the unused `harness` arguments go. Canon: the method-A sentence in `IB/CANON/ROOT_CANON.md` changes from "worst-case OOM probe" to "sizing from the heaviest real batch, probed with that batch".

Validation: helper check of the sizing arithmetic on a fake device description (descending cumulative sums, both checkpointing settings, the cap, the CPU path returning `CPU_BATCH_SIZE` with its heaviest examples so the probe's CPU skip still returns a usable size) and of the bound itself (random batches of B drawn from the synthetic distribution never exceed the top-B total); the end-to-end test (CPU, fixed batch 2, no sizing); a Sky run at the run-3 shape without `--batch-size`, reading the sizing text, the probe attempt count and the peak memory. Pass: the probe passes on the first attempt and peak memory is within 30 % of what the sizing arithmetic predicts for the chosen batch (with checkpointing on the batch is the cap, so the card is far from full; the criterion tests the arithmetic, not the fill).

#### Planned Changes

`activation/retrieval/retrieval_batching.py · example_token_counts(), largest_fitting_batch(), BatchSizing, recommended_batch_size()`

```diff-python
@@ retrieval_batching.py — sizing @@
+@dataclass
+class BatchSizing:
+    batch_size: int
+    explanation: str
+    probe_examples: list[LabeledRetrievalQAExample]     # the batch_size heaviest examples, on CPU too (the probe only skips the step there)
+
+
+def example_token_counts(training_data, dataset_index, retrieval_model) -> list[int]:
+    """Per example: estimated padded tokens of its query (with instruction) plus all its candidates, each cut to the input limit."""
+    ⋯ one pass with estimated_tokens(); string lengths only ⋯
+
+
+def largest_fitting_batch(counts_desc: list[int], usable_bytes: int, per_token_bytes: int, cap: int) -> int:
+    """Largest B with per_token_bytes x sum(top-B counts) <= usable_bytes, at most cap (0 when one example does not fit)."""
+
+
 def recommended_batch_size(
-    harness, retrieval_model, training_data, dataset_index, gradient_checkpointing, headroom_fraction, device, cap=128,
-) -> tuple[int, str]:
+    retrieval_model, training_data, dataset_index, gradient_checkpointing: bool, headroom_fraction, device, cap=128,
+) -> BatchSizing:
-    longest, max_candidates = observed_shape(training_data, dataset_index, retrieval_model)
+    counts = example_token_counts(training_data, dataset_index, retrieval_model)
+    order = sorted(range(len(counts)), key=lambda index: -counts[index])
+    counts_desc = [counts[index] for index in order]
⋯ unchanged: weights, trainable bytes, usable; per-token bytes for both settings as today ⋯
-    per_sequence = per_token * tokens + ac_bytes
-    sequences = 1 + max_candidates
-    per_example = per_sequence * sequences
-    recommended = int(usable // per_example) if usable > 0 else 0
+    batch_on = largest_fitting_batch(counts_desc, usable, per_token_on, cap)
+    batch_off = largest_fitting_batch(counts_desc, usable, per_token_off, cap)
+    batch_size = batch_on if gradient_checkpointing else batch_off
+    if batch_size < 1:
+        raise RuntimeError(f"GPU {properties.name} cannot fit the heaviest example: ...")
+    explanation = "\n".join([..., f"heaviest {batch_size} examples: {sum(counts_desc[:batch_size]):,} est. tokens",
+                             f"batch with checkpointing {batch_on}, without {batch_off}; using {'with' if gradient_checkpointing else 'without'}"])
+    return BatchSizing(batch_size, explanation, [training_data[index] for index in order[:batch_size]])
```

`activation/retrieval/retrieval_batching.py · probe_batch_size()`

```diff-python
-def probe_batch_size(step_fn, harness, retrieval_model, batch_size, longest_text, max_candidates, attempts=3):
-    """One synthetic forward/backward at batch_size with every sequence at the observed longest; halve on OOM."""
+def probe_batch_size(step_fn, retrieval_model, probe_examples, dataset_index, attempts=3):
+    """One real forward/backward on the heaviest examples (real queries and candidates); halve the batch on OOM."""
+    batch_size = len(probe_examples)
     for attempt in range(1, attempts + 1):
-        ⋯ synthetic examples from longest_text × max_candidates ⋯
+        batch = _batch_from(probe_examples[:batch_size], dataset_index)
+        ⋯ print distinct / total candidates of the probe batch; warn below 0.9 ⋯
⋯ unchanged: try / OutOfMemoryError / gc + empty_cache / halve ⋯
```

`activation/retrieval/retrieval_trainer.py · train()` (three lines)

```diff-python
         if config.batch_size is None:
-            batch_size, batch_sizing = recommended_batch_size(self.harness, retrieval_model, ...)
+            sizing = recommended_batch_size(retrieval_model, training_data, dataset_index, config.gradient_checkpointing, ...)
         else:
-            batch_size, batch_sizing = config.batch_size, f"batch_size = {config.batch_size} (from the config)"
-        longest_text, max_candidates = observed_shape(training_data, dataset_index, retrieval_model)
+            sizing = configured_batch_sizing(config.batch_size, training_data, dataset_index, retrieval_model)   # same shape, heaviest batch_size examples
⋯
-        batch_size, probe_attempts = probe_batch_size(probe_step, self.harness, retrieval_model, batch_size, longest_text, max_candidates)
+        batch_size, probe_attempts = probe_batch_size(probe_step, retrieval_model, sizing.probe_examples, dataset_index)
```

`IB/CANON/ROOT_CANON.md` — amend the method-A convention line as described above.

</details>

## Review record

Independent review of version 1 (`IB/TMP/BATCHING/review_1.md`): verdict incorrect-or-missing, 18 findings.

| finding | class | what changed |
| --- | --- | --- |
| M3 gain contradicted by the author's own measurement | genuine | M3 dropped; measurement re-done with candidate-length keys and recorded above |
| checkpoint transient 28× too large and double-counted | genuine | budget no longer tiered by memory; transient removed from M2; per-layer arithmetic stated |
| grouping estimate ignored the input limit and the query instruction | genuine | lengths estimated as `embed_batch` sees them |
| batch without checkpointing is ≈51–55, not 36 | genuine | numbers corrected everywhere |
| byte-pooling rows ≈ padded tokens, not hundreds of thousands | genuine | sentence corrected; the 65,535 limit is dropout-only |
| "compute floor well under one second" wrong; the plan is an MFU bet | genuine | bet stated in the summary; pre-check added as M1 step 3 |
| 15 → 3 forwards impossible at 16k | genuine | 8k default (≈6 forwards), A/B sweeps 8k and 16k |
| cap 512 with automatic checkpointing is a training-dynamics change | genuine | checkpointing stays a bool, cap stays 128, both batches printed |
| O(B·N) collision loop at cap 512 | reviewer-overkill at cap 128 (≈30 ms) | noted as the condition for raising the cap |
| empty list raises today | genuine | claim removed; path is unreachable |
| `private_helpers_check.py` tests `length_groups` | genuine | check script updated in the same change |
| `/source` has no batching file for the baseline | genuine | baseline copied from the workspace file |
| canon line needs amending | genuine | canon note in M2 |
| probe fallback with automatic checkpointing off | moot | automatic rule dropped |
| M3 code details | moot | M3 dropped |
| A/B cannot prove the invariant (dropout stream); no forwards counter | genuine | eval-mode allclose check; `forwards_embedded` counter |
| merge queries into candidate grouping; per-forward `.item()` sync | sync: genuine (one line in M1); merge: reviewer-overkill (one launch per step; changes over-limit query truncation) | out-of-scope reason corrected |
| small slips (three calls not four, test batch 2, bench flag, unused `harness`, CPU path, rename) | cheap-nit | all applied |

Round 2 (`IB/TMP/BATCHING/review_2.md`): verdict incorrect-or-missing on version 2, six findings, all applied in version 3: the CPU-side token count must include the view positions to match the mask sum (blocking; otherwise the padding fraction changes by the instrument, not the work); the CPU path with `batch_size=None` would have produced a batch of 0 (blocking; the sizing now carries the heaviest examples on CPU too); the M2 memory criterion was unreachable at cap 128 (now tests the arithmetic); the compute gate is stated at M1's padded tokens (≈110 TFLOP/s, 45 %); the M1 footprint is now the same in the approach table, the decision card and the source; the AC per-token memory is stated explicitly. The reviewer agreed with both reviewer-overkill classifications from round 1.
