# Batching-level training throughput — agent source (version 3)

Projection: `BATCHING_PLAN.tressoir.md` (same folder). This file holds the evidence, the arithmetic behind the estimates, the implementation notes and the review record. Keep the two in agreement.

## Request (user, 2026-09-05)

"You and a review should create plan for high-impact, self-contained training optimizations. A good optimization lives mostly in the batching file, and determines just how things are selected/schedule without wider impacts. Artificially limiting the pool size should not be one of them: that decision is to be made at generation time, not here."

Context: "what's the best lever to increase training throughput on cheap synthetic. Like, handling 100k-example epochs in less than an hour."

## Evidence

### Run 3 of slice 1 (`IB/TMP/RETRIEVAL_SLICE1/gpu_run3/training_stats.json`)

- 950 examples, batch 32, 2 epochs, 60 steps, checkpointing on, RTX PRO 6000 (96 GB).
- avg candidates per example 8.27; avg tokens per example 906; distinct candidates per step ≈ 262.
- avg step time 3.98 s; real tokens/s 7,202; padded tokens/s 9,700 → 38.6k padded tokens per step; padding 25.7 %.
- peak memory 8.48 GB. 1,258 s per 10k examples → 100k ≈ 3.5–3.8 h.
- reporting time 10.3 % (3 evaluations of 50 queries in a 30-step epoch; negligible at 3,000 steps per epoch).

### Forwards per step (measured; `length_groups(tolerance=0.15, min_group=8)` on real 32-query MS-MARCO train batches, seed 0)

```
batch 0: 271 candidates, char range 108-740, 11 candidate groups (sizes [8, 8, 11, 15, 32, 17, 27, 34, 71, 37, 11]), 4 query groups -> 15 forwards per step
batch 1: 271 candidates, char range 120-735, 11 candidate groups (sizes [8, 13, 18, 37, 28, 20, 27, 48, 55, 16, 1]), 4 query groups -> 15 forwards per step
batch 2: 243 candidates, char range 107-840, 11 candidate groups (sizes [8, 9, 18, 26, 33, 11, 25, 40, 45, 21, 7]), 4 query groups -> 15 forwards per step
```

### Compute per step and the MFU bet (reviewer finding 6, adopted)

- Non-embedding parameters 28 × 15.7M = 440M; LoRA rank 128 on seven projections ≈ 81M → N ≈ 521M (the embedding table does no matmul).
- Forward ≈ 2·N·T = 2 × 0.521e9 × 38,635 ≈ 40 TFLOP; forward + backward + checkpoint recompute ≈ 4× → ≈ 160 TFLOP per step. Achieved: 160 / 3.98 s ≈ 40 TFLOP/s ≈ 16 % of a ≈ 250 TFLOP/s bf16 peak.
- Target 100k examples per hour at batch 32 = 3,125 steps per hour = 1.15 s per step. At M1's padding (≈ 41k padded tokens per step) and without recompute: 6 × 521M × 41k ≈ 128 TFLOP per step → ≈ 110 TFLOP/s ≈ 45 % MFU.
- So the plan bets that 8k-token forwards lift MFU from 16 % to ≥ 45 %. M1's pre-check measures it: 20 steps at the run-3 shape with the 8k budget; TFLOP/s = 8 × 521M × padded tokens ÷ step time (checkpointing on). Gate: ≥ 110 TFLOP/s → the hour is reachable with M2; 90–110 → 60–75 min; below 90 → not reachable at these shapes, stop for a decision.
- M1 also adds FLOPs: padding rises (estimate 6 % → 12 % at 8k on the char-length model; the real run showed 26 % where the estimate said 6 %, so expect ≈ 30 %). The launch saving must outweigh ≈ 5–10 % more tokens, which is the same bet.

### Memory arithmetic (base Qwen3-0.6B, d 1024, d_ff 3072, 28 layers, bf16; reviewer findings 2 and 4 adopted)

- `_per_token_layer_internals(1024, 3072)` = (6 × 1024 + 4 × 3072) × 2 = 36,864 B per token per layer.
- checkpointing on: per token = 28 × 1024 × 2 + 36,864 ≈ 92 KB (layer inputs kept, plus one recomputed layer). Batch 32 × 38.6k tokens ≈ 3.6 GB, consistent with the 8.5 GB peak.
- checkpointing off: per token = 28 × 36,864 ≈ 0.98 MB for the base; AC model: 7 layers × `_per_token_layer_internals(256, 1024)` = 7 × 11,264 ≈ 79 KB per window position with one window per token (stride 4 bytes ≈ 1 token), plus the un-checkpointed byte stage ≈ 6 × 256 × 2 B × 4 bytes ≈ 12 KB per token → ≈ 1.07–1.12 MB per token in total (the sizing keeps the existing AC arithmetic, expressed per estimated token). With checkpointing the AC term is 7 × 256 × 2 + 11,264 ≈ 15 KB per token.
- Hugging Face checkpointing is per decoder layer, so an M1 group costs one layer's internals during its recompute: 36,864 B × 8,192 ≈ 300 MB, and that allowance is already the `+ internals` term charged for every token. The forward split has no memory effect without checkpointing. The budget is therefore a launch-versus-padding knob, not a memory knob (version 1 had this wrong by ≈ 28×).
- usable ≈ 82.5 GiB after 10 % headroom, 1 GB context, 1.2 GB weights, 0.3 GB embedding copy, 1.4 GB trainable states.
- Real MS-MARCO token totals (`IB/TMP/BATCHING/estimates_check.md`, 20k sampled rows): average 963 per example, heaviest 1,642, mean of the heaviest 36 = 1,557, of the heaviest 128 = 1,509. The current heuristic charges 9 × 291 = 2,619 per example (1.6× the real heaviest, 2.7× the average).
- Batch without checkpointing = 82.5 GiB / 1.12 MB / ≈ 1,557 ≈ 51 (55 without the AC term). With checkpointing ≈ 620 → cap 128.

### Padding and forwards under the budget (estimate on char lengths; `estimates_check.md`)

| grouping | padding | candidate forwards per 32-query step |
| --- | --- | --- |
| old (15 % tolerance, min 8) | 6.2 % | 11.0 |
| token budget 8k | 12.4 % | 5.0 |
| token budget 16k | 20.9 % | 3.0 |
| best length bucketing (max candidate length, 8 buckets) + 8k | 9.9 % | 4.6 |

Length bucketing was version 1's M3; the measured gain (2–5 points) does not justify a batch-composition bias, so it is dropped.

### Throughput estimates (conditional on the pre-check)

- M1 at 8k, checkpointing on, batch 32: 15 forwards → 6; step 4.0 s → 1.3–1.8 s if MFU reaches 40–50 % → 18–24 examples/s → 1.2–1.5 h per 100k.
- M1 + M2 with checkpointing off at batch ≈ 51: 1.33× fewer FLOPs → 25–32 examples/s → 50–65 min per 100k.

## Scope rules

- M1: `retrieval_batching.py` plus two lines in `RetrievalModel.embed_batch` (CPU-side real-token count `sum(len(row) for row in token_rows) + batch_size * num_views`, identical to the mask sum, which covers tokens and view rows, instead of `attention_mask.sum().item()`, which syncs the GPU per forward; a `forwards_embedded` counter), the step shape in the trainer/stats gaining the forwards count, one report column, and the private helper check script. All A/B arms share the same model file.
- M2: `retrieval_batching.py` plus three trainer lines (sizing call, probe call, sizing string) and the canon line. Config unchanged: `gradient_checkpointing: bool = True`, cap 128.
- No change to the loss, the metrics, the reporter's widgets, the data selection or the candidate pools. Raising the cap or automating the checkpointing choice are training decisions, not made here (reviewer finding 8).

## Milestone notes

### M1 — token-budget forwards

- `FORWARD_TOKEN_BUDGET = 8192` (GPU), `CPU_FORWARD_TOKEN_BUDGET = 512`.
- `estimated_tokens(text_length, num_views) = text_length // CHARS_PER_TOKEN + 1 + num_views`, shared by grouping and sizing.
- `embed_in_length_groups` estimates each length as `embed_batch` will see it: `min(len(text), retrieval_model.input_limit_chars) + (len(retrieval_model.query_instruction) if is_query else 0)` (reviewer finding 3: the CPU test truncates to 128 chars, so raw lengths would have split nearly every text into its own forward).
- `token_budget_groups(lengths, budget_tokens, num_views)`: ascending sort; a group closes when `(len(group) + 1) × estimated_tokens(newest)` exceeds the budget; an oversize single text is its own group (the next text is at least as long, so it can never join).
- `length_groups` removed; `IB/TMP/RETRIEVAL_SLICE1/private_helpers_check.py` replaces its `length_groups` cases (lines 6–19) with `token_budget_groups` cases and the invariant check.
- Invariant check (reviewer finding 16): in eval mode, `embed_in_length_groups(model, texts, is_query, budget_tokens=b)` is `allclose` to `embed_batch(texts, is_query)` for b giving one-per-group, a few groups and all-in-one. Proves "same loss" on CPU in seconds; the Sky A/B measures speed only (the AC dropout stream differs between arms, so loss curves cannot be compared sharply).
- Counter: `RetrievalModel.forwards_embedded`; the trainer's step shape gains a fifth element; `RetrievalTrainingStats.summarize()` reports `forwards_per_step`; the throughput table gets a "forwards/step" column (reporter: one column string).
- Baseline for the A/B: copy the workspace `retrieval_batching.py` to `IB/TMP/BATCHING/baseline_retrieval_batching.py` before editing (`/source` has no batching file: the slice-1 cards are pending apply).
- Sky node `ac-batching`: pre-check (20 steps, 8k), then three one-epoch arms at the run-3 shape (`--epochs 1 --batch-size 32`): baseline, 8k, 16k; reports under `~/activation_artifacts/batching_ab/{baseline,b8k,b16k}` watched into `IB/TMP/BATCHING/`. Compared: step time, real and padded tokens/s, forwards per step, peak memory. Pass: ≥ 2× real tokens/s over the baseline.

### M2 — sizing and probe

- `example_token_counts(training_data, dataset_index, retrieval_model)`: per example `estimated_tokens(min(len(query), limit) + len(instruction)) + Σ estimated_tokens(min(len(chunk), limit))`. String lengths only.
- `largest_fitting_batch(counts_desc, usable_bytes, per_token_bytes, cap)`: cumulative sum; largest B with `per_token_bytes × cumsum[B] ≤ usable_bytes`, capped.
- `BatchSizing(batch_size, explanation, probe_examples)`; `probe_examples` are the `batch_size` heaviest examples on every device; on CPU the probe skips the step and returns `len(probe_examples)` (round-2 finding: an empty list would have returned batch 0).
- `recommended_batch_size(retrieval_model, training_data, dataset_index, gradient_checkpointing: bool, headroom_fraction, device, cap=128) -> BatchSizing`: both per-token figures computed as today (base and AC terms), both batches printed, the configured setting chosen; `harness` argument dropped; CPU path returns `BatchSizing(CPU_BATCH_SIZE, "...", heaviest CPU_BATCH_SIZE examples)`.
- `configured_batch_sizing(batch_size, training_data, dataset_index, retrieval_model) -> BatchSizing`: the configured size with the heaviest `batch_size` examples for the probe.
- `probe_batch_size(step_fn, retrieval_model, probe_examples, dataset_index, attempts=3)`: `_batch_from(probe_examples[:batch_size], dataset_index)`; prints distinct/total candidates and warns below 0.9 (dedup can only lighten the batch; the arithmetic bound sums per-example tokens without dedup, so it dominates); halve on `torch.OutOfMemoryError` with the existing gc + empty_cache.
- `observed_shape` removed.
- Trainer: `sizing = recommended_batch_size(...) if config.batch_size is None else configured_batch_sizing(...)`; `batch_size, probe_attempts = probe_batch_size(probe_step, retrieval_model, sizing.probe_examples, dataset_index)`; `batch_sizing = sizing.explanation`. The three `set_training_mode(True, config.gradient_checkpointing)` calls are unchanged.
- Canon (`IB/CANON/ROOT_CANON.md`, method-A convention line): "a data-measured sizing heuristic capped at 128 and a worst-case OOM probe" → "sizing from the heaviest real batch the data can produce (per-example token totals, descending cumulative sum), capped at 128, probed with that real batch".
- Validation: helper check with a fake description (per-token numbers above, usable 82.5 GiB, a synthetic count distribution) asserting batch_off ≈ 51 and batch_on = 128, the cap and CPU paths, and the bound: 1,000 random batches of B from the synthetic distribution all satisfy `sum(batch) ≤ sum(top-B)`; the e2e test; a Sky run at the run-3 shape without `--batch-size` (the bench's default is None) reading the sizing text, probe attempts (expect 1) and peak memory (expect within 30 % of the sizing's prediction for the chosen batch; with checkpointing on the batch is the cap and the card sits well under half full, so the criterion tests the arithmetic).

## Out of scope (with reasons)

- Candidate pool per example: generation-time (user).
- Larger cap / automatic checkpointing: training decisions (schedule, in-batch pool); the trainer's O(B·N) collision loop must be vectorized before a cap above a few hundred (≈ 30 ms at 128, ≈ 0.4 s at 512).
- Merged query/candidate forward: doable from the batching file by prepending the instruction and calling `embed_batch(is_query=False)`, but that changes truncation for over-limit queries and saves one launch per step; deferred.
- GradCache: memory for negatives, not throughput.
- Reporting cadence: negligible at scale.
- `torch.compile`, attention kernels, tokenizer prefetch, multi-GPU: harness-level; after M1 the tokenizer (≈ 30k tokens per step, tens of milliseconds) is a few percent of a 1 s step.

## Review record

Round 2 (`IB/TMP/BATCHING/review_2.md`): verdict incorrect-or-missing on version 2, six findings, all genuine and applied in version 3 (view positions in the CPU-side count; CPU sizing path; memory criterion; compute gate at M1's tokens; consistent M1 footprint; explicit AC per-token figure). The reviewer agreed with both reviewer-overkill classifications.

Round 1 (`IB/TMP/BATCHING/review_1.md`, independent reviewer): verdict incorrect-or-missing, 18 findings. Reclassification and disposition are in the projection's Review record table; summary: 14 genuine (all applied), 2 moot (M3 dropped, automatic checkpointing dropped), 1 reviewer-overkill at the unchanged cap (collision loop; recorded as the condition for raising the cap), 1 split (per-forward sync applied; query merging deferred with the corrected reason).

## Completion record (version 4, implemented)

Approved in chat ("Go for it") with two instructions: validation by micro-benchmarks in private code, not long runs; vary candidates per example (about 2 versus about 10).

### What landed

- `activation/retrieval/retrieval_batching.py`: `FORWARD_TOKEN_BUDGET = 8192`, `CPU_FORWARD_TOKEN_BUDGET = 512`, `estimated_tokens`, `token_budget_groups`, `embedded_text_length`, `embed_in_length_groups(…, budget_tokens=None)`, `example_token_counts`, `largest_fitting_batch`, `_per_token_layer_internals(d_model, d_ff, with_lora)`, `BatchSizing`, `configured_batch_sizing`, `recommended_batch_size(retrieval_model, …) -> BatchSizing`, `probe_batch_size(step_fn, retrieval_model, probe_examples, dataset_index)`. Removed: `length_groups`, `observed_shape`, the `harness` arguments, the unused import.
- `activation/retrieval/retrieval_model.py`: CPU-side real-token count including the view rows; `forwards_embedded`.
- `activation/retrieval/retrieval_trainer.py`: sizing and probe calls; the step shape's fifth element. `retrieval_training_config.py`: shape type, `forwards_per_step` in `summarize()`. `retrieval_reporter.py`: "forwards/step" column.
- `activation/retrieval/retrieval_ac.py` (unplanned, correctness): per-text window validity in `WindowedBytePooling.forward`.
- `IB/CANON/ROOT_CANON.md`: method-A line amended. `IB/TMP/RETRIEVAL_SLICE1/private_helpers_check.py`: new grouping and sizing cases; `make_round_doc.py`: card texts.

### Private validation code (`IB/TMP/BATCHING/`)

`baseline_retrieval_batching.py` (the file before M1), `microbench.py` (timed arms, sizing rows, live report → `microbench/`), `invariant_cpu.py` (fp32 CPU exactness of embeddings and loss across budgets), `padding_diag.py` / `padding_diag2.py` (which stage depends on padding), logs `sky_*.log`.

### Measurements (node `ac-batching`, RTX PRO 6000; 8 timed steps after 2 warm-up)

matmul parameters 521M (base layers + LoRA r128). Full table in `microbench/microbench_results.json`; the summary table is in the projection. Key rows: ~8 candidates, checkpointing on, batch 32: baseline 4.12 s / 15.1 forwards / 39 TFLOP/s → budget 8k 2.23 s / 6 forwards / 78 TFLOP/s. ~8 candidates, checkpointing off, batch 16: 1.54 s → 0.87 s (17.0k real tokens/s, 80 TFLOP/s). 2 candidates, batch 32: checkpointing on 2.68 s → 0.85 s; off 1.45 s → 0.53 s (14.7k real tokens/s). Budgets: 4k and 16k lose to 8k or tie in every arm (16k pads 35–44 %).

Sizing (990 examples): per token 0.19 MB with checkpointing, 3.04 MB without. ~8 candidates: with 128 (cap; probe 10.5 s, peak 24.4 GB), without 18 (probe peak 79.4 GB, predicted 79.4 GB). 2 candidates: with 128 (peak 8.9 GB), without 70 (peak 92.2 GB, predicted 81.2 GB: short sequences carry more per-sequence overhead; the probe passed inside the 10 % headroom). Every probe passed on attempt 1.

Invariance (fp32, CPU, node): before the AC fix, grouped embeddings differed by up to 0.07 (budget 1 versus all-in-one); the diagnosis put the base decoder at 4.5e-7 and the AC rows at 0.36, and the stage trace showed the pooled window count differing (7 alone versus 8 batched). After the fix: AC rows 1.3e-6 to 2.4e-6, full embedding 7.2e-7, grouped embeddings at budgets 1 / 64 / 256 / 512 / all within 3.6e-6 of one forward; contrastive loss at budgets 1 / 256 / all = 9.649283 / 9.649287 / 9.649276 (`INVARIANT HOLDS`).

End-to-end test (node): `1 passed in 32.78s` (before the AC fix), `1 passed in 23.06s` (after); 2 forwards per step; probe passed at 2.

### Conclusions

- Default budget 8k. The step is no longer launch-bound in the sense of forward count, but the compute rate tops out near 30 % of peak; the LoRA's per-layer small kernels are the next cost and are a harness-level lever.
- 100k-example epoch: ~1.5 h at ~8 candidates per example (no checkpointing, batch 16–18) or ~1.9 h with checkpointing at batch 32; ~27 min at 2 candidates without checkpointing, ~44 min with. The hour target holds for the 2-candidate shape.
- The AC padding bug affected every slice-1 run; the next training run is the first with batch-independent AC rows.
