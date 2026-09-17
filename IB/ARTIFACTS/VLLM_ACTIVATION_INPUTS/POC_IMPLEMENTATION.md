# Deep AC inputs — POC implementation (phase C)

Implemented 2026-09-16 (~14:45–15:05 UTC; review-2 items closed ~15:15 UTC) in a git worktree from tag `ac-recon-20260916T132825Z` (the working tree's `activation/` + bench), following `PLAN_POC_RECONCILED.md` with review 1 folded in (§7 there). **Nothing was applied to `/workspace`**; node 1, ORCD, `remote*.sh` and `orcd_run.sh` were not touched.

- Patch: `IB/ARTIFACTS/VLLM_ACTIVATION_INPUTS/poc_changes.patch` — `git diff --binary` of the worktree against the tag, new files included (18 files, +1,242 / −79). `git apply --check --binary` passes against `/workspace` (dry run only).
- Helper (outside the patch, already in place): `IB/TMP/AGENT_AC_PRETRAINING/ac_recon/orcd_run_deep.sh`.

## What changed, per file

**`activation/ac_model/ac_model_utils.py`**
- `ACOutput(input_embeds, layer_inputs)` dataclass (`to`, `detach_cpu`, `nbytes`); `DeltaHead(d_in, d_out, r)` = `Linear → SiLU → Linear`, rows RMS-normalised × a scalar `scale` (starts at 0, set by calibration); `resolve_intervention_layers(num_layers, frequency, add_last, prefer_attention, full_attention_layers)` implementing the plan's rule (Qwen3.5-4B → `[7, 11, 19, 23, 31]`); frequency 0 returns `[]` before any validation (review 2, item 3).
- `RowCache` stores `ACOutput` (bf16 CPU, every tensor counted); `put` still accepts a bare tensor.

**`activation/ac_model/ac_model.py`**
- Config: `intervention_frequency: int = 0`, `add_last_layer_intervention = True`, `prefer_attention_interventions = True` (validated ≥ 0). Constants `INTERVENTION_SCALE_FRACTION = 0.01`, `INTERVENTION_INIT_SEED_MASK = 0xDEE9`.
- `ActivationContextModules(config, d_side, d_target, intervention_layers)`: `delta_heads` (`nn.ModuleDict` keyed by layer, `r = max(1, min(d_side, d_target)//4)`), buffers `intervention_calibration`, `interventions_calibrated` — **all registered only when deep**, heads initialised under `fork_rng` + `manual_seed(initial_seed() ^ 0xDEE9)`. `is_deep` property.
- `ActivationContextModel`: resolves the layers from the target's `layer_descriptions`; `is_deep`, `interventions_calibrated`, `intervention_summary()`, `calibrate_interventions(sequences, item_ids)` (reader with adapter, no rows, `no_grad`, one sequence per forward, RMS per layer via `capture_layer_input_rms`; scale = 1 % of RMS; item ids kept as `calibration_items`).
- `encode_batch(requests, *, with_layers=False)`: existing callers still get row tensors; `with_layers=True` returns `ACOutput`s; a deep uncalibrated model raises. `_encode_prepared_batch` emits the per-layer deltas (empty map for recursive rows and input-only models).
- `save` adds an `interventions` entry (summary) for deep models only; `load` refuses a layer-list mismatch (legacy checkpoints = input-only) and restores `calibration_items`.

**`activation/harness/hf_utils.py`**
- `Interventions` TypedDict; `BoundInterventions` (payloads + cache offset, travels in the kwargs); `text_decoder(model)` / `decoder_layers(model)` (uses `language_model`/`text_model` below a multimodal wrapper, never a vision tower); `install_intervention_hooks(model)` (idempotent: a text-model pre-hook binds the offset from `past_key_values.get_seq_length()` **before any layer runs**; a pre-hook with kwargs on every decoder layer rebuilds kwargs without the entry and adds the deltas out of place at `positions − offset`); `capture_layer_input_rms(model, layers)`; `generate_with_interventions(peft_model, *, inputs_embeds, interventions, **kw)` (greedy, single sequence; the payload bound by wrapping the **base HF model's** `prepare_inputs_for_generation` with `functools.wraps`, restored in `finally`).

**`activation/harness/loaded_model.py`**: `decoder_forward(..., interventions=None)` — checks one payload per physical row, installs the hooks, passes the list as a kwarg. `None` = the previous call exactly.

**`activation/ac_model/ac_model_training.py`**
- Config: `deltas_off_panel: bool = False`; `learning_rate_gates_scope` accepts `"delta_scales"`.
- `_assemble_student(student_embeds, spans, outputs, dtype, deltas_off=False)` builds the rows-over-placeholders embeds and one payload per example (positions = the span runs, per-layer concatenation of the parts' deltas); used by `_example_loss`, `_batch_loss` (via `_padded_forward(..., payloads)`) and `_sample_completions` (which calls `generate_with_interventions` when a payload exists). No-context references, the live-penalty base pass and the stored-reference pass carry no payload.
- `train()`: calibrates a fresh deep model before the optimizer exists (first `CALIBRATION_EXAMPLES = 8` training examples, `_calibration_sequence` = the stripped history through a reference `build_example`); the `deltas_off` panel at the baseline and every boundary; `intervention_trace` every 25 updates (scale, scale / calibration RMS, head grad norm per layer — the norms taken by `_head_grad_norms` next to `norm_ac`, before clipping, review 2 item 2); `payload_bytes` (counted only under autograd, i.e. training forwards; the panels and the samples run under `no_grad`, review 2 item 5); stats/epoch fields and `summarize()` export them.
- `_evaluate(..., deltas_off=False)`, `_example_loss(..., deltas_off=False)`, `_batch_loss(..., deltas_off=False)`.

**`activation/ac_model/ac_model_reporter.py`**: widgets `interventions` (scale / calibration RMS per layer) and `intervention_grads` (head gradient norms), fed from the trace in `report_step`; the epoch record text excludes `deltas_off` like the other panels.

**Bench `IB/ARTIFACTS/AGENT_AC_PRETRAINING/ac_recon/run_recon.py`**: `DEFAULT_ARM += interventions=0, last_layer_intervention=True, prefer_attention=True, deltas_off=True`; `harness(...)` passes them to the model config; `config()` passes `deltas_off_panel`; `parameter_hash(ac, h, deep=False)` hashes by name and skips `delta_heads.*` (zero-mode value unchanged), `deep_hash` for the heads; reporter label for `deltas_off` and per-kind series "(deltas off)"; the run summary names the layers; `result.json` gains `deep_hash`, `interventions`, `payload_bytes`, `deltas_off`, `deltas_off_novel`, `deltas_off_wiki`.

**`arms.json`**: `NDC` (= ND + `interventions 0`), `NDI` (ND + 4 interior + last, snap to attention), `NDIg` (NDI + `gate_lr 0.003`, `gates_scope delta_scales`); existing entries byte-identical.

**`activation/tests/test_interventions.py`** (new, 12 tests) + `activation/tests/data/` (tiny 2-layer Qwen3 reader; `legacy_keys.json`, `legacy_ac_modules.pt`, `legacy_outputs.pt` written by the **tag's** code): layer resolution and config validation; text-decoder locator (multimodal skipped if not constructible); zero mode = the previous model (keys, strict legacy load, bit-identical rows and hidden states, no hooks, deep↔input-only load refused); pairing (shared draws and adapters identical, next RNG draw identical, heads a function of the seed); deep encode + calibration (uncalibrated raises, delta RMS = scale, determinism, calibration sequence = stripped history); direct vs checkpointed gradients for every head, both scales, the input head and the reader LoRA on the hybrid model, and two pending payloads — parametrised over `prefer` so one case lands a delta at a linear-attention (GDN) block input, layers `[3, 5]` (review 2 item 4); padded batch (batched == per-example with deltas, row isolation, tokens before the first position untouched, payload-count check, `deltas_off` keeps rows and drops the payload); generation (cached == manual full-prefix greedy recompute, spy hook: applied once at prefill and never at decode, logits move, raw `generate(interventions=)` raises, beams rejected, and the trainer's own `_sample_completions` observed through the same spy: one payload bound at prefill over the present spans' positions, nothing added at decode, not counted in `payload_bytes` — review 2 item 1; the mutation `payload = None` there now fails this test); checkpoint round trip and `optimizer.pt` resume with the heads; `deltas_off` panel and the `delta_scales` gate group (input-only model produces nothing); bench arm pairing.

**Helper `IB/TMP/AGENT_AC_PRETRAINING/ac_recon/orcd_run_deep.sh`**: copy of `orcd_run.sh` with `WS=${AC_DEEP_WS:-$HOME/activation_ws_deep}` and `BENCH=${AC_DEEP_BENCH:-$HOME/activation_artifacts/ac_recon_deep}` (the four path edits: `cd`, `PYTHONPATH`, `AC_RECON_SOURCE_ROOT`, the `launch.py` path); defaults `mit_preemptable rtx_pro_6000 2h`; `AC_RECON_ROOT` and the AUTOTUNE symlink unchanged.

## Test results (CPU, `/workspace/.venv`, torch 2.13, transformers 5.15.1, `HF_HUB_OFFLINE=1 PYTHONPATH=<worktree>`)

| suite | result |
|---|---|
| `activation/tests/test_interventions.py` (new) | **12 passed** in 41 s |
| `IB/TMP/.../ac_recon/test_batched.py` + `test_recon.py` (copied next to the worktree, unchanged) | **16 passed** (13 + 3) in 24 s, 1 pre-existing warning |
| `activation/tests/test_basic_agent_ac.py`, `test_basic_agent_ac_training.py` | 6 skipped (`@pytest.mark.gpu`, no `--gpu`; they construct default configs, which are input-only) |

Probes that shaped the implementation (not tests): HF's `_validate_model_kwargs` rejects `generate(interventions=…)`; PEFT's `base_model_prepare_inputs_for_generation` is not called by HF's loop for a LoRA model (the override lands on the `LoraModel`, whose `generate` resolves to the HF model's), so the binding wraps the HF model's `prepare_inputs_for_generation` (with `functools.wraps`, since HF validates against that signature); with it the kwarg reaches the text model at prefill and every decode step.

## Review 2 closure (`POC_REVIEW_2.md`, APPROVE)

| # | item | done |
|---|---|---|
| 1 | deep path through `_sample_completions` untested | generation test reinstalls its spy and calls `_sample_completions` on the same item; asserts one bound payload at prefill with the present spans' positions, none added at decode; verified the `payload = None` mutation fails it |
| 2 | head gradient norms recorded after clipping | `_head_grad_norms` computed next to `norm_ac` (before any clip) and passed into the trace entry; widget label unchanged and now true |
| 3 | frequency-0 return after the layer-count check | early return moved first |
| 4 | no delta at a linear-attention input | gradient test parametrised (`prefer` True/False → `[2, 5]` / `[3, 5]`), same direct-vs-checkpointed and two-pending-payload assertions on both |
| 5 | `payload_bytes` counts evaluation forwards | counted only when `torch.is_grad_enabled()`; the test asserts sampling leaves it at 0 |

## Deviations from the reconciled plan / review

1. **Generation binding on the HF model, not on PEFT's saved attribute** (review finding 1's exact suggestion does not reach HF's loop here; see the probe above). Same semantics: per-call, restored in `finally`, never through `generate`'s kwargs.
2. **Cache offset bound once per forward** by a text-model pre-hook rather than read per layer: after the first attention layer has run its prefill, `get_seq_length()` already reports the prompt length, so a per-layer read would skip the deltas at every later layer during prefill.
3. **Calibration sequences come from a reference `build_example` of the stripped history**, not `example.teacher_ids` (empty under SFT).
4. `encode_batch` keeps returning tensors by default (`with_layers=True` for `ACOutput`), as recommended in the plan's open question 3.
5. The logits-difference assertion replaces "the greedy continuation differs without deltas" in the generation test: on a random tiny model the argmax happened not to change even at 50× the calibrated scale.

## How the deep arms are launched (after the reviewer's second pass and the apply to `/workspace`)

1. `git apply --binary IB/ARTIFACTS/VLLM_ACTIVATION_INPUTS/poc_changes.patch` in `/workspace`; rerun the three suites above there.
2. `python IB/TMP/AGENT_AC_PRETRAINING/ac_recon/snapshot.py` → new tag + `training_source.json` (`launch.py` verifies the hashes).
3. Push to a **separate** ORCD checkout: rsync `activation/` (from the new tag) to `~/activation_ws_deep/activation/` and the bench folder (`run_recon.py`, `launch.py`, `arms.json`, `training_source.json`) to `~/activation_artifacts/ac_recon_deep/` — never `--delete`, never toward `~/activation_ws`.
4. Free 1–2 preemptable GPUs: `scontrol hold` the *queued* N-arm jobs (not running ones), then `bash orcd_run_deep.sh NDC` and `bash orcd_run_deep.sh NDI` (defaults: `mit_preemptable rtx_pro_6000 2`); `scontrol release` the N-arms once both run. `NDIg` when a slot frees.
5. Read: `result.json` `held_novel` vs NDC (ND: 0.712), `deltas_off_novel` (held − deltas_off = the deltas' contribution), `no_context`, `interventions.scales` vs `calibration_rms`, the `interventions` / `intervention_grads` widgets, `initial_hash` equal across NDC/NDI, `deep_hash` recorded.

## Not done (deferred per the plan)

M3 agent records, M4 vLLM fork, packed `[1,S,D]` collation, FP8/prefix-cache gates, legacy → deep checkpoint migration, the shuffled-bundle control, the product default flip to `intervention_frequency = 4`. GPU validation of the deep path has not run (no GPU in this environment); the first GPU run is the POC itself.
