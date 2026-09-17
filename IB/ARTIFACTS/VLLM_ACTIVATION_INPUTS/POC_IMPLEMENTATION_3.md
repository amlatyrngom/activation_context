# Deep interventions, phase F — A2 passage attention (`intervention_source = "passage_attention"`)

Implemented 2026-09-16 (~17:05–17:15 UTC; review-3 items closed ~17:30 UTC) in the same worktree, on top of phase E and the training-review fold-in (`POC_IMPLEMENTATION_2.md`). HF path only. **Nothing applied to `/workspace`**; no node or ORCD access; `remote*.sh` / `orcd_run*.sh` untouched.

- Patch: `IB/ARTIFACTS/VLLM_ACTIVATION_INPUTS/poc_changes_3.patch` — worktree vs tag `ac-recon-20260916T155942Z` for `activation/` and the bench (8 files, +1,047 / −124; **includes phase E and the fold-in**), plus the `arms.json` hunk against `/workspace`'s current file (+228 lines: every phase-E arm and the phase-F arms). `git apply --check --binary` passes against `/workspace` (dry run only). Apply this one instead of `poc_changes_2.patch`, not after it.
- Switch: `intervention_source: "passage_attention"` (third value of the same switch, composes with `intervention_target: "kv"`), `intervention_attention_depth: "final" | "matched"` (default `"final"`).

## What A2 does

One `PassageAttention` block per intervened layer sits in front of that layer's `DeltaHead` (residual or K/V): the memory row's state is the query, the passage tokens' states are keys and values, and the block returns `rows + out(attention)` with `out` **zero-initialised** (weight and bias). At initialisation the head therefore reads exactly what it read before — the summary readout at depth "final", the layer-matched readout at depth "matched" — and the attention path opens only as `out` leaves zero (the silent-phase mitigation; the scale calibration is unchanged and the heads' draws are those of the summary model, since the blocks draw from a separate forked seed after the heads).

- **Depth "final"**: query = the row's `z` (the side model's final normed state at the row position); keys/values = **the side model's final normed states at the passage (content) positions** of the same side sequence — what the encoder already holds after its side forward, not the post-mixer states.
- **Depth "matched"**: query and keys/values are the input states of side layer s(l) (s(l) = l unless `intervention_source_layers` maps otherwise), captured by phase E's per-layer pre-hooks now spanning the whole valid sequence of each physical row (a view of the layer input, no copy; hooks removed right after the forward; nothing else materialised). The captured slice is split into passage positions and row positions inside the head loop and dropped after the head runs.
- **Block**: shared `LayerNorm(d)` on both inputs, `query/key/value: Linear(d, r)`, `out: Linear(r, d)`, `r = 640` on Qwen3.5-4B (the heads' bottleneck), **4 attention heads of 160 dims** (1 head when r is small, as in the tests), `F.scaled_dot_product_attention`, unpadded per-row inputs (no masks). Parameters per layer at d = 2560, r = 640: **6,563,200** (q/k/v 4,917,120 + out 1,640,960 + norm 5,120); with the delta head (3,280,001) ≈ 9.84 M per layer, ≈ 49 M for five layers. Payload unchanged (5,120 B or 4,096 B per row per layer).
- Composition with the fold-in: `intervention_heads_frozen` freezes the blocks too (only the relative scales train); `intervention_detach_source` detaches both the query rows and the passage states; the bench's shared-parameter hash skips `passage_attention.*` like `delta_heads.*`, so `initial_hash` still pairs with NDC / the warm pair; the deep hash covers both.
- Readouts: trace fields `attention_out_norm` (norm of `out.weight`, 0 at init, rising as the path opens) and `attention_grad_norm` per layer; widget `intervention_attention`; `intervention_summary()` adds `attention_depth` and `attention_parameters`; `load()` refuses a checkpoint whose `(source, source_layers, target, attention_depth)` differ; the input-only → deep migration also builds the blocks fresh.

## Per-file changes (phase F only)

- `activation/ac_model/ac_model_utils.py`: `PassageAttention`.
- `activation/ac_model/ac_model.py`: config `intervention_attention_depth` + validation, `layer_matched_source` property (source-layer mapping shared by `side_layers` and `passage_attention/matched`); modules build `passage_attention` after the heads under a separate forked seed; `trainable_parameters()` freezes the blocks with the heads; `intervention_summary()` fields; `_encode_prepared_batch` captures the whole valid sequence for "matched", takes the final passage states for "final", detaches both under the control, runs the block in front of the head; `load()` shape check and migration keys.
- `activation/ac_model/ac_model_training.py`: trace fields `attention_out_norm`, `attention_grad_norm`. `activation/ac_model/ac_model_reporter.py`: `intervention_attention` widget.
- Bench `run_recon.py`: arm key `intervention_attention_depth` through `harness()`; `parameter_hash` skips the blocks in the shared hash; the run summary describes the source. `arms.json`: `NDIA` (NDI + `passage_attention`, depth final), `NDIAL` (+ `matched`), `NKVA` (NDIA + kv), `NKVAL` (NDIAL + kv), warm-started `NDIAW`, `NDIALW`.
- `activation/tests/test_interventions.py`: `build(..., depth)`, helpers `attention_parameters`, `open_attention`; new `test_passage_attention` (config validation; head count rule; at init "final" is bit-identical to the summary model — state dict draws, zero `out`, encoded rows and deltas — and "matched" is bit-identical to the layer-matched model; once `out` is opened, the "matched" output equals the block applied to independently captured side-layer states split into passage/row positions, and the "final" output equals the block applied to the final normed states; frozen heads exclude the blocks; a training epoch traces the block's out-norm and gradient); the gradient test gains `passage_final` (residual) and `passage_matched_kv` cases — `out` opened, then direct vs checkpointed gradients matched and q/k/v/out weights all receive gradient, with reader and side checkpointing toggled together; the checkpoint test gains a `passage_attention/matched` case (block parameters saved and restored, depth saved, a different depth refused); bench pairing covers the six new arms.

## Test results (CPU, `/workspace/.venv`, `HF_HUB_OFFLINE=1 PYTHONPATH=<worktree>`)

| suite | result |
|---|---|
| `activation/tests/test_interventions.py` | **27 passed** in 129 s (22 → +2 new tests, +3 parametrised cases; 1 test-side `UserWarning`) |
| bench `test_batched.py` + `test_recon.py` (copied next to the worktree's current bench) | **16 passed** (13 + 3) in 27 s, 1 pre-existing warning |
| `activation/tests/test_basic_agent_ac*.py` | 6 skipped (GPU-gated) |

## Review 3 closure (`POC_REVIEW_3.md`, REVISE → fixed)

| # | item | done |
|---|---|---|
| 1 | HIGH: `intervention_attention` widget initialised inside the per-layer loop → assertion at the first deep trace under the bench reporter | created once, guarded (`not in self.widgets`), only when the entry carries `attention_out_norm`; data loops skip widgets that do not exist; new `test_reporter_takes_deep_trace_entries` drives `report_step` with four two-layer entries (two without and two with attention fields) plus a K/V-keyed entry |
| 2 | MEDIUM: a deep `optimizer.pt` from the phase-C layout cannot resume under the three-group layout | the resume compares the saved and current group layouts (parameter counts per group); on a mismatch the moments are discarded with a logged warning, the optimizer starts fresh and only the update clock resumes (`resumed_from` says "moments discarded"); tested with an input-only `optimizer.pt` resumed under a migrated deep model |
| 3 | LOW: `attention_grad_norm` read after clipping | `_head_grad_norms` now also returns `attn:<layer>` before clipping; the trace entry reads those |
| 4 | LOW: fp32 passage copy per layer at depth "final" | hoisted above the layer loop (one copy per example) |
| 5 | LOW: per-forward host syncs; undrained cosine diagnostic | `row_rms` and the cosine are kept as 0-d tensors and converted once per trace point; the cosine is recorded in training mode only, capped at 256 values per layer, and cleared by `set_mode` |
| 6 | LOW: `NKV` note said 1 % | now "5 % … (the default scale fraction)" |
| 7 | LOW: `intervention_attention_depth` silently ignored for other sources | a non-default depth with a non-attention source raises (tested) |

## Deviations / decisions

1. **"final" keys/values are the side model's final normed states of the passage positions** (not the post-mixer states): the coordinator allowed either; this is the tensor the encoder already holds and keeps the query (`z`) and the keys in the same space.
2. **4 heads of 160 at r = 640** (single head when r < 64 or not divisible by 4); a shared LayerNorm on both inputs; no masks (each row's passage is unpadded).
3. The block's parameters are excluded from the shared hash and frozen by `intervention_heads_frozen`, i.e. treated as part of "the heads".
4. Attention-mass and gradient-into-rows readouts remain deferred (extra forward).

## Launch

As before, with the arm from `NDIA`, `NDIAL`, `NKVA`, `NKVAL` or the warm `NDIAW`, `NDIALW`. Read `held − deltas_off` against NDI/NDIL (they start identical), the `intervention_attention` widget (does `out` open, and when), `interventions` (relative scales), `intervention_cosine`, `interventions.attention_parameters` in `result.json`, and `initial_hash` equal to the paired arm's.
