# Short paired native-context LR diagnostic

Root reviewed, froze and initially submitted the two resident-optimizer arms as jobs23557303/23557305; see [submission receipt](PI_NATIVE960_LR_SUBMISSION_V1.json). Two separate L40S runs start from the same constant42k weights and actual Adam5250 and train the same frozen960 fresh questions once:

| Arm | AC / reader LR | Evidence | Training |
|---|---|---|---|
| `pi42k_native2k_fresh960_floor_v1` |5e-6 /2e-6, constant|native2k, up to4 passages|960×1|
| `pi42k_native2k_fresh960_2x_v1` |1e-5 /4e-6, constant|native2k, up to4 passages|960×1|

Both retain ratio1/16, one encoder sequence, logical8/micro1, checkpoint threshold0, encoder packing65536, unchanged frozen bases/rank128 adapters, penalty, clipping and precision. The optimizer stays on GPU. Only the two learning rates differ scientifically between these arms. The original960 IDs/order, learning32, held100 (50/50 generator cohorts), retrieval, gold and source files are reused unchanged; no new material or selection.

This is a short local LR response diagnostic. It is not a convergence or schedule test, and is not an isolated comparison with the previous RTX8k/micro2 floor run. Both short arms must run on L40S. Fresh same-device initial full/no-context controls are measured separately before each arm's training; no external RTX scores are substituted.

The existing report flow is retained: held100 and fixed learning32 every96 questions (10%), current-reader controls only at start/final, frozen initial controls throughout, no inline decoding. Each own complete final must have120 stage updates and all615 populated Adam clocks5370, with both AC and reader checkpoints. Intermediate paired saves every192 questions retain the existing resume behavior. A partial/time-limited run is not relabeled as a complete final.

Planning estimate:60–95min per arm, with parallel scheduling needed for near-term paired evidence. The resident diagnostic measured29.55/26.98s per8-question update including exact-gradient/parameter hashing and32.60GiB peak; those two batches do not establish full960 throughput or worst-case fit. Each allocation is2h/64GiB host, with6600s trainer bound including panels. Full960 and all reporting items remain unprobed. **OOM is an explicit failure: no automatic retry, offload, truncation, batch-size, checkpointing or precision fallback.** Initial and10% panels provide useful evidence if final completion falls after the PI boundary.

New bench files: `run_qa_pi42k_native960_lr_v1.py`, `prepare_pi42k_native960_lr_v1.py`, `train_pi42k_native960_lr_v1.sbatch`, and `PI42K_NATIVE960_FLOOR_DESIGN_V1.json` / `PI42K_NATIVE960_2X_DESIGN_V1.json`. Designs were frozen by root before submission; the launcher requires frozen status.

The runner changes24 lines/additions and14 removals from the frozen960 runner; launcher10/9. All helper functions, material builder, selection, scoring, reporting and resume implementations remain unchanged.72 quick CPU checks passed: unchanged helper ASTs, runtime constructors versus designs, exact frozen metadata/subset, wrong arm-LR rejection, own final-clock guard, device/settings and launcher syntax. No GPU tests performed for these new960 arms.

Root staging can reuse the already-staged original ORCD material/pair and `taskaware/adam_moment_offload_probe_v1_inputs/subset.json`. New reports will appear under `IB/TMP/SYNC/TASK_AWARE_QA/pi42k_native2k_fresh960_floor_v1/` and `.../pi42k_native2k_fresh960_2x_v1/` after syncing, with `report.tressoir.html` in each.

Local checks: `python IB/TMP/FIRST_TASK_AWARE_FINETUNING/pi42k_native960_lr_preparation/check.py`. Focused diffs and checks receipt are beside it. No product files, frozen designs, old runners, queued jobs or existing reports were changed.

Operational handoff update: root reports that2× job23557305 failed before model initialization/training because the login environment exported an incomplete HF cache (`/home/ngom/orcd/scratch/hf`). Floor23557303 was held pending. Root is preparing unchanged-code/design resubmission with explicit `HF_HOME=/home/ngom/orcd/scratch/activation/hf`; consult the submission receipt for replacement job IDs. This is an environment correction, with no optimizer-offload adoption or scientific-setting change.
