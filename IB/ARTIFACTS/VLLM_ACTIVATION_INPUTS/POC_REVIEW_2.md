# POC implementation review 2 — worktree `deep-wt` (phase C)

**Verdict: APPROVE** — ready to apply to `/workspace` and launch NDC/NDI. One test gap (finding 1) should be closed in the same apply; nothing found that would break the run, mislead its reading, or touch existing arms.

Reviewed 2026-09-16 against the reconciled plan (§7 disposition) and review 1. Read the worktree files themselves (not only `poc_changes.patch`), re-ran the tests here (CPU, `/workspace/.venv`, `HF_HUB_OFFLINE=1`): `activation/tests/test_interventions.py` **11 passed** (40 s); `test_batched.py` + `test_recon.py` copied next to the worktree with `PYTHONPATH=<worktree>` **16 passed** (27 s). No product code, bench, ORCD or `remote*`/`orcd_run.sh` touched in either tree; `IB/TASK.md` not read. Paths below are relative to the worktree.

## Findings (by severity)

### 1. MEDIUM — the sampled-completion deep path is implemented but not asserted by any test

- **What.** `_sample_completions` builds the payload and routes to `generate_with_interventions` (`activation/ac_model/ac_model_training.py:867-878`), which reads correctly. But no test observes it: `test_generation_matches_full_recompute` calls `generate_with_interventions` directly, and the only test that runs `_sample_completions` (`test_checkpoint_roundtrip_and_resume`, via `completion_samples=1`) asserts nothing about the payload. The mutation `payload = None` before the `if payload is not None` branch passes all 27 tests, i.e. the fidelity samples could silently be generated without deltas.
- **Fix.** In `test_generation_matches_full_recompute`, after the oracle, install the same spy and call `trainer._sample_completions(ac, [items[0]], progress)`; assert one bound payload at prefill with `positions == [p for start, end in present spans for p in range(start, end)]` and none at decode. Cheap; no product change.

### 2. LOW — head gradient norms are recorded after clipping but labelled "before clipping"

- **Evidence.** `_intervention_trace_entry` is called at `ac_model_training.py:649-650`, after `clip_grad_norm_` at `:636-642`; the reporter widget says "Per intervention layer, before clipping" (`ac_model_reporter.py:145`) and the docstring "before the step". With ND's joint clip (`clip_per_group=False`, `max_grad_norm`) the plotted head norms are the scaled ones whenever the joint norm exceeds the bound — exactly the updates one wants to inspect.
- **Fix.** Compute the per-head norms where `norm_ac`/`norm_reader` are computed (before clipping) and pass them into the entry, or relabel both strings "after clipping".

### 3. LOW — `resolve_intervention_layers` validates `num_layers` before the frequency-0 early return

- **Evidence.** `activation/ac_model/ac_model_utils.py:529-533` raises on `num_layers < 1` even when `frequency == 0`; `ActivationContextModel.__init__` calls it for every model. No current breakage: `layer_descriptions` always has `num_hidden_layers` entries (fallback to full attention at `activation/harness/hf_utils.py:314-316`).
- **Fix.** Return `[]` for `frequency == 0` first, so zero mode is unconditional whatever the target description.

### 4. LOW — the hybrid tests never add a delta at a linear-attention input

- **Evidence.** `tiny_hybrid` has full attention at layers 2 and 5; every deep test uses `frequency=1, prefer=True` → layers `[2, 5]` (`test_interventions.py:194,257`). GDN layers only see the kwargs removal (`hf_utils.py:143`), not an added delta under checkpoint recompute. The POC's resolved layers are all full-attention blocks, so the run is covered; the general mechanism is not.
- **Fix.** Add a `prefer=False` case to `test_gradients_direct_vs_checkpointed` (frequency 1 → interior 3, a GDN block) with the same direct-vs-checkpointed comparison.

### 5. LOW — `payload_bytes` counts evaluation forwards too

- **Evidence.** `_assemble_student` accumulates on every call (`ac_model_training.py:361`), including the panels' `_evaluate` path; the field is documented as "over the call's training forwards" (`:98`).
- **Fix.** Count only when `torch.is_grad_enabled()`, or relabel "over every reader forward of the call".

### 6. INFO — the six stated deviations are correct as implemented

- (a) Binding on the HF model, not PEFT: `LoadedModel.model` is the object `get_peft_model` wraps in place (`activation/harness/module_manager.py:178-180`), so `peft_model.get_base_model()` is the same instance; the hook handle attribute and the `prepare_inputs_for_generation` wrapper are installed once and restored in `finally` (`hf_utils.py:207-232`); `functools.wraps` keeps HF's signature validation happy; the test asserts the raw kwarg still raises.
- (b) Offset bound once per forward by the text-model pre-hook (`hf_utils.py:124-129`): correct and necessary — after the first attention layer's cache update, `get_seq_length()` already includes the prefill tokens; reading it per layer would drop every later layer's prefill delta. At the text-model entry it is the count of previously cached tokens: 0 at prefill, the prompt length at the first decode step (spy assertions at `test_interventions.py:333-334`). `decoder_forward` passes `use_cache=False` and no cache, so training offsets are 0.
- (c) Calibration on the stripped history through a reference `build_example` (`ac_model_training.py:364-370`): `teacher_ids` is empty under SFT, so review 1's suggestion was wrong; this is the plan's "stripped histories" with the reader adapter active, `no_grad`, B = 1, item ids stored.
- (d) Logits-difference assertion: acceptable, the oracle comparison of ids is the real gate.
- (e) `encode_batch(with_layers=…)` additive: as recommended. (f) `deltas_off=True` in `DEFAULT_ARM` only changes `configuration.deltas_off_panel` in every arm's `result.json`; the panel runs only when `ac_model.is_deep` (`ac_model_training.py:549`).

## Verified without findings

- **Zero mode.** No heads, buffers or keys unless deep (`ac_model.py:165-176`); state-dict keys equal the tag's and the tag's `modules` checkpoint loads strictly; rows and hidden states bit-identical to outputs written by the tag's code; hooks installed only when a payload is passed (`loaded_model.py:509-513`), and installed hooks early-return without one (`hf_utils.py:124-126,139-141`). `parameter_hash` in zero mode hashes the same tensors in the same order as before (`named_parameters` order = `parameters` order; `run_recon.py:172-177`). Every existing arm resolves to `interventions=0` through `DEFAULT_ARM` → `harness()` → `intervention_frequency=0` (`run_recon.py:33-34,97`); an old checkpoint (no `interventions` entry) loads into a zero-mode model (`ac_model.py:581-583`, tested).
- **Forward coverage.** Payload on: `_example_loss` (`:1039-1044`), `_batch_loss` → `_padded_forward` (`:1118-1124,1181-1183`, right padding keeps positions physical, offsets applied once), `_sample_completions` (`:867-878`), the reporting panels via `_evaluate`. None on: stored-reference pass (`_store_teacher_targets`, unchanged), penalty reference and live base passes (`:1080-1093`, `:1152-1160`), `deltas_off` (`_assemble_student(deltas_off=True)` returns `None`, `:356-357`), no-context items (no parts). Positions = the span runs in order, deltas concatenated per layer (`:359-360`).
- **Gradients.** Delta added out of place on a clone (`hf_utils.py:158-160`); the bound payload travels in the layer kwargs, so `GradientCheckpointingLayer`'s partial recomputes with the same tensors; `test_gradients_direct_vs_checkpointed` matches direct and checkpointed gradients for every head, both scales and the reader LoRA on the hybrid model, and two pending payloads sum to their separate gradients.
- **Generation.** Prefill applies each delta row once; decode steps (offset ≥ prompt, S = 1) add nothing; ids equal a manual full-prefix greedy recompute; beams/sampling rejected; wrapper restored in `finally`.
- **Checkpoints / resume.** `save` writes `interventions` (layers, calibration RMS, items, scales) for deep models only; `load` refuses a layer-list mismatch in both directions and restores `calibration_items`; heads and buffers ride `modules.state_dict()`; `optimizer.pt` carries the head moments and a fresh process resumes with `updates_done` and the arm's rates; an uncalibrated deep model refuses `encode_batch` (`ac_model.py:369-370`); a checkpoint-loaded model is calibrated and is not recalibrated (`ac_model_training.py:481`).
- **Pairing.** Heads built under `fork_rng` + `manual_seed(initial_seed() ^ 0xDEE9)` after every shared draw (`ac_model.py:170-173`); the side and reader adapters are injected later and draw identically (tested, including the next global draw); `initial_hash` skips `delta_heads.*`, `deep_hash` recorded. Calibration runs no RNG (no dropout, `no_grad`) and only bumps `version` (cache identity, not training).
- **Bench / ORCD.** `NDC` = ND + `interventions 0`; `NDI` = NDC + `4 / last / prefer`; `NDIg` = NDI + `gate_lr 3e-3, gates_scope delta_scales` (asserted by `test_bench_arms_are_paired`; the `delta_scales` scope puts exactly the head scales in group 2, `ac_model_training.py:517-519`, tested). `orcd_run_deep.sh` differs from `orcd_run.sh` in the four path edits plus the preemptable defaults; `AC_RECON_ROOT`, the AUTOTUNE link and `SYNC_RUNS/<LABEL>` unchanged; nothing writes toward `~/activation_ws`. `snapshot.py` hashes `activation/**/*.py` and the bench `*.py`, so the deep checkout needs its own `training_source.json` from the new tag (documented). Reporter: `interventions` (scale / calibration RMS per layer) and `intervention_grads` widgets; `result.json` gains `deep_hash`, `interventions`, `payload_bytes`, `deltas_off(_novel/_wiki)`.
- **Test quality (spot mutations by reasoning).** Dropping the payload in `_batch_loss` breaks the batched-vs-single comparison at 50× scale; dropping the offset binding trips the spy's `offset >= prefix_length`; a hook that does not fire on recompute breaks the direct-vs-checkpointed match; removing the `fork_rng` block breaks the next-draw and adapter equality; the legacy fixtures pin zero mode to the tag's outputs. The gap is finding 1.

## Verdict

**APPROVE.** Findings: (1) sampled-completion deep path untested — add the spy assertion; (2) head gradient norms recorded after clipping, relabel or move; (3) frequency-0 early return before the layer-count check; (4) add a GDN-input delta case to the gradient test; (5) `payload_bytes` counts eval forwards; (6) deviations (a)–(f) accepted.
