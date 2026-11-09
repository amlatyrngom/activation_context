# Free-weights arm for Corpus Q/A — agent-facing plan (Oct 19)

Source of truth for `FREE_WEIGHTS_PLAN.tressoir.md`. Status: **Planning** (awaiting the user's decisions; no code written yet).

## Request (user, verbatim, Oct 19 ≈15:20Z)

> Ok we need to try something a bit more radical. How about this:
> - One argument in the paper is that the exact AC model is pluggable, and is compatible with previous techniques.
> - I want you to write code that recognizes the 10 specific inputs, and routes them to something that returns "free weights" for each. Free weight means the values themselves are being trained, like cartriges. I hope this drastically reduces time per update.
>
> Now hopefully, there is a parent class that lives in the main source, possibly in ac_model.py, but the code that owns this logic lives elsewhere.
> Or better, yet, have ActivationContextModel have something like self.special_input_handler that, when it returns something that is not None, is treated as the right value. This is pluggable from outside the main code.
>
> Make a .tressoir.md plan for me regarding this.

Standing constraints that apply: ratio stays 1/15 (user, Oct 19); LoRA rank 128 (canon); reader rate 2e-5; x-axis = Q/A pairs trained; never edit running scripts or a run's recipe.json; new folders only.

## Facts from the code (read Oct 19)

- `activation/ac_model/ac_model.py`
  - `ActivationContextModel.encode_batch(requests, with_layers)` (line ≈461) is the single entry for every encode. That covers rollout and eval, training (`ac_model_training.py` lines ≈1315, 1361/1367 `_encode_parts_cached`, 1654, 1722/1725 `share_part_encodings`), and the corpus eval (`kit/bench/eval_snapshot.py` via the harness).
  - It checks the rollout row cache per request (`row_cache_key(messages, ratio, recursive, version, tools)`), then `_encode_requests` → `_encode_prepared_batch` → `_encode_kv_batch` for `kv_transfer`.
  - `_encode_kv_batch` (≈567) returns per part `ACOutput(rows, kv_values={layer: (keys, values)}, kv_positions, kv_length, kv_overwrite)`:
    - `rows` are zeros, or the marker in overwrite mode;
    - keys/values are `[V, kv_heads, head_dim]`, not rotated;
    - `kv_positions` comes from `interleaved_positions(len(ids), V, device, bin_weights)`;
    - `kv_length = len(ids) + V`.
  - `trainable_parameters()` (≈441) = AC module params + side LoRA params. The trainer puts them in the `learning_rate_ac` group (`ac_model_training.py` ≈691–696). Resume checks the saved param-group layout (≈705–711).
  - `save()` / `load()` (≈753 / ≈778) write and read `ac_modules.pt` + `side_lora/` in the checkpoint folder.
  - `invalidate_encoded_rows()` (≈744) bumps `version` and clears the row cache.
- Profile (CORPUS_PROBE.md, "Speed"): one KV update on a GH200 took 113 s:
  - encoder forward 30 s;
  - encoder backward 78 s;
  - reader and the rest ≈5 s.
  - So the encoder is ≈95 % of an update. K15 (2×H200, shared encodings) ran ≈33 s/update; H3 (all ten, 2×RTX unshared) runs 167 s/update.
- Corpus parts: each patient record is its own activation_context part. `share_part_encodings` found identical records across examples, so a record's part content is the same wherever it appears. 10 distinct parts expected; verify on the packet.
- Sizes at 1/15 (Llama-3.1-8B: 32 layers, 8 KV heads, head dim 128):
  - ≈7,200 positions over the 108,005-token corpus (per record ≈630–810);
  - key+value per position = 32 × 2 × 8 × 128 = 65,536 numbers → ≈472M free numbers;
  - ≈1.9 GB fp32, ≈7.5 GB with gradients and Adam moments. Fits next to the reader on an H200 or a 96 GB RTX.
- Cartridges (arXiv 2506.06266, repo): the trainable cache is initialized from the model's own key/values of corpus text, trained at lr 2e-2, with the model frozen.

## Design

### Main source (generic, small): a pluggable hook

`activation/ac_model/ac_model_utils.py`: a `SpecialInputHandler` protocol:
- `__call__(request: EncodeRequest) -> ACOutput | None` — None means "not mine: use the encoder";
- `parameters() -> list[nn.Parameter]` — trainable values (may be empty);
- `state_dict()` / `load_state_dict()` for checkpoints.

`activation/ac_model/ac_model.py`:
- `ActivationContextModel.special_input_handler: SpecialInputHandler | None = None` (set from outside after construction).
- `encode_batch`: before the cache lookup, ask the handler. A non-None answer is the result for that request. It is never row-cached, because its values change every update. The remaining requests go through the encoder unchanged, and the results are re-merged in request order.
- `trainable_parameters()`: unchanged, so the encoder group is unchanged. New `special_input_parameters()` returns the handler's parameters (empty without a handler).
- `save()`: when a handler exists, also writes `special_inputs.pt` (its state_dict plus a small descriptor: handler class name and input keys) next to `ac_modules.pt`.
- `load()`: if `special_inputs.pt` exists and a handler is attached, loads it. If the file exists but no handler is attached, raise. Silently evaluating without the trained values would be wrong.
- With no handler, the behaviour is byte-identical (tested).

`activation/ac_model/ac_model_training.py`:
- New config field `learning_rate_special_inputs: float = 0.0`.
- When the AC model has special-input parameters, add a third optimizer group with that rate.
- The resume layout check accepts the 2-group layout of old checkpoints unchanged and the 3-group layout of new ones.
- After each optimizer step the trainer already calls `invalidate_encoded_rows()` on encoder updates. Make sure it also does so when only the special-input group moved, so the `_encode_parts_cached` cache (keyed by version) is never stale.

### Outside the main source: the free key/value table (kit)

`IB/TMP/PAPER_CORPUS_QA/kit_free/free_kv.py · FreeKVTable(nn.Module)`:
- **Recognition:** key = `row_cache_key(messages, ratio, recursive, version=0, tools)` of each distinct record part in the packet.
  - Exactly the 10 records → index 0..9. The builder checks that there are exactly 10 distinct parts, and maps each to its patient by the record text inside it.
  - `strict=True`: an unrecognized part raises instead of falling back to the encoder, so this arm can never silently use the encoder.
- **Values:** per record r, per reader layer l: `keys[r][l]`, `values[r][l]`, each an `nn.Parameter` `[V_r, 8, 128]` fp32.
- **Placement:** `kv_positions` / `kv_length` are computed once, exactly as `_encode_kv_batch` computes them (`interleaved_positions` over the record's side ids, `num_view_rows` at 1/15). So the reader sees the free entries at the same positions as the K arm's encoder entries.
- **Rows:** zeros, or the marker in overwrite mode, exactly as the encoder emits them.
- **Initialization (decision D1; recommended: the untrained encoder's output):**
  - Run the AC model once with the handler detached, under no_grad.
  - Copy each record's `kv_values` into the parameters. Before training, the free arm then equals the K arm at update 0 (tested): the base model's own keys/values at the marker positions, the same idea as Cartridges' "init from corpus KV".
  - Alternatives: init from K15c's trained encoder (warm start), or random.
- Returns `ACOutput(...)` with the parameters themselves (no copy), so gradients flow into them.

`kit_free/run_free.sh` + sbatch files: like `kit_hill/run_hill.sh`, plus:
- `CQA_FREE=kv`;
- `CQA_LR_FREE=<rate>`;
- `CQA_FREE_INIT=fresh|k15c`;
- `CQA_READER_FROZEN=0|1`.

The runner builds the packet as usual, constructs the AC model, attaches `FreeKVTable` built from the packet's distinct parts, and trains.

The eval loader attaches the same handler before `load()`. `special_inputs.pt` carries the descriptor, so `eval_snapshot.py` builds the handler from it; this is a kit-side change only.

### Runs

Same items as H2 (`synth_full_unikeys`, sha 4826752f9b9abcca), same order, 1/15, 8 pairs per update, reader 2e-5, x-axis pairs. Encoder view: all ten records in every example (the user's choice). With free values this costs nothing extra on the encoder side.

- **F0 speed check:** 20 updates; measure s/update and peak memory.
  - Projection, not measured: reader-only ≈5–10 s/update on one H200, i.e. 3–6× faster than K15's 33 s on two H200s and ≈20–30× faster than H3 on the RTX node.
- **F1 rate probe (decision D3):** free-value rates 2e-2 (Cartridges), 2e-3 and 2e-4, about 200 updates each (≈1,360 pairs). Pick by balanced accuracy on the all-ten view, with the no-context view as a leak check.
- **F2 long run:** the best rate, to the 6,160-pair cap and then further passes if still rising (cheap now). Snapshots every 100 updates; evals all/right/none + balanced accuracy + letters.
- **Optional F3 (decision D2):** reader frozen, as in pure Cartridges, at the F1 rate.

GPUs: one H200 per run. ORCD mit_preemptable now, under the per-user cap of 4 and its maintenance at 04:00Z Oct 20, or CSAIL after 06:00Z Oct 20. Because the reader-only memory is small, the EC2 RTX node also fits, but H3 occupies it until ≈04:15Z.

### Paper framing

"The AC interface is pluggable: a Cartridges-style trained cache for a fixed corpus plugs in as a special-input handler, with the reader, positions, data and evaluation unchanged."
- The free arm is corpus-specific: it cannot encode a new document. The encoder arms can.
- This is the comparison the section can make: same reader, same data, same x-axis.

## Validation (CPU, tiny Llama fixtures as in kit/tests/test_llama_harness.py)

1. No handler: `encode_batch` outputs identical to before (bitwise), and the training step is unchanged.
2. Handler initialized from the encoder: reader logits identical to the encoder path for the same inputs (positions, masks, rotation).
3. One training step: gradients reach the free tensors; the side LoRA and AC modules get none; the version bumps; the cached encode is refreshed.
4. Save → load round trip of `special_inputs.pt`; `load()` refuses a free checkpoint without a handler.
5. Strict mode raises on an unrecognized part; non-strict falls through to the encoder.
6. Resume: a 3-group optimizer state restores; old 2-group checkpoints still resume.

## Milestones

- M1 hook in main source + tests 1, 4 (load refusal), 6.
- M2 FreeKVTable in kit_free + tests 2, 3, 4, 5.
- M3 runner/eval wiring + F0 on one H200.
- M4 F1 probe → F2 long run (+ F3 if chosen), with results in research/CORPUS_FREE_WEIGHTS.md.

## Accepted decisions (user's interactions, read Oct 19 ≈15:45Z; verbatim copy in IB/TMP/free_weights_interactions_20261019.json)

- **D1 init:** chose "fresh", with the note: "Though the untrained encoder probably does not have a notion of "marker". Maybe the last [-T:] tokens or something."
  - Default: the base model reads the record with no markers; each free position takes the key/value of the real token at the end of its bin (the token just before where the marker would sit). Real-token keys/values, evenly spread, at the same positions.
  - Extra probe: the record's last V tokens' keys/values, the user's [-T:] suggestion.
- **D2:** the reader LoRA trains at 2e-5 ("Yes.").
- **D3:** rate probe 2e-2 / 2e-3 / 2e-4.
- **D4:** all channels. Verbatim: "All. I'd like the ablation to train kv, input, and input+residuals. All should be fast, but try to hillclimb kv first, then reuse its hyperparams." The handler therefore also returns free rows `[V, d_target]`, and free rows + per-layer deltas, for the R and D channels.
- **Free-form feedback (verbatim):** "I have a problem with the loop shown (I don't know if it's pseudocode, or what you actually intend to implement). But make the input handler take in the batch, and is somehow free to return selected indices or something like that. So tensor + mask, or something easier. Not that it matters in our specific, since there is not benefit from batching. At least give it the option of being batched though. Since are compute-limited for a while, and this tends to be fast, make it a high priority; I think we can have good insights by today. Also, if we don't have enough questions compared to Cartridges, feel free to synthesize more and better questions bw."
  - **Interface:** `handler(requests: list[EncodeRequest]) -> list[ACOutput | None]`, one answer per request, None = encoder. `encode_batch` calls it once per batch.
  - **Priority:** high.
  - **Data:** more synthesized questions are allowed if needed.
- **Status:** waiting for the user's green light ("Wait a bit for green light.").
