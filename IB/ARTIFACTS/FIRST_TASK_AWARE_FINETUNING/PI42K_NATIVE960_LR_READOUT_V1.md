# Native960 LR final100 readout

Prepared for root review; no GPU submission. The new private evaluator is a focused adaptation of `evaluate_pi42k_merged32k_v1.py`, limited to the floor/2× native2k L40S diagnostics. Each arm requires its own completed960×1,120-stage epoch001 AC/reader pair, published marker, exact training config/material IDs and all615 populated Adam clocks5370. Partial, foreign, earlier and wrong-LR finals fail.

Each readout freshly scores AC/full-context/no-context on the exact original100, using native2k/four passages, the unchanged retrieval/gold/answer masks and microbatch1. All three views replay their own saved final panel's per-answer NLL and every gold-token log-probability with the existing1e-3 gate. The evaluator explicitly reuses the training result's actual logits chunk size so removing resident optimizer memory cannot change that policy. No RTX metrics are reused.

The AC view then generates all100 actual greedy responses, thinking disabled,160-token cap, using the existing validated inherited-EOS/chat-end union helper. Complete returned IDs, raw text, EOS/cap metadata and parser results are retained. Completion requires100 response/token-ID records. The evaluator performs no optimizer updates and rehashes the final pair afterward.

`training_snapshot.json` preserves training config/autotuning, initial AC baseline and initial full/no-context controls, with publication hashes. These are explicitly labeled **frozen42k controls**; fresh output views belong to the **own42960 final reader**. Initial-control identity/order/target lengths are checked with the existing validator. No initial baseline is relabeled as a final-reader answer view.

Files: `bench/evaluate_pi42k_native960_lr_v1.py`, `.sbatch`, and independent `PI42K_NATIVE960_LR_READOUT_MANIFEST_V1.json`. Training runners/designs remain unchanged. The launcher uses the complete activation HF cache explicitly, normal L40S/64GiB/15min, one arm per job,840s application bound. Planning estimate5–10min per arm; actual final readout duration is unmeasured. Partial results remain resumable only under the same identity/runtime.

Separate local synced folders will be:
- `IB/TMP/SYNC/TASK_AWARE_QA/pi42k_native960_lr_cohort100/pi42k_native2k_fresh960_floor_v1__2k/`
- `IB/TMP/SYNC/TASK_AWARE_QA/pi42k_native960_lr_cohort100/pi42k_native2k_fresh960_2x_v1__2k/`

Each contains `result.json`, `status.json` and `training_snapshot.json`. No new HTML/report construction is part of this preparation.

Validation:48 focused CPU checks passed against both real frozen designs/subset and small hypothetical final-publication/optimizer fixtures, including stale-clock, wrong-LR/pair/order/marker, changed baseline and per-token replay failures. Existing final-optimizer, initial-provenance and replay functions remain AST-identical. Python compilation and launcher syntax pass. No final checkpoint, GPU result or semantic quality is claimed before the corresponding run finishes.

Checks/diff: `IB/TMP/FIRST_TASK_AWARE_FINETUNING/pi42k_native960_readout_preparation/`. Run `.venv/bin/python .../check.py` to repeat the focused checks.
