# Two matched native continuation learning-rate candidates

Ready for root review/freeze and submission. No jobs or remote calls performed.

| Run | AC / reader learning rates | Questions |
|---|---|---|
| `continue42k_8k_constant_2x` | constant 1e-5 / 4e-6 | same fresh8,000 ×1 |
| `continue42k_8k_cosine_2x_floor` | cosine 1e-5 / 4e-6 →5e-6 / 2e-6; zero warmup | same fresh8,000 ×1 |

Both reuse the completed floor baseline's material and independent initialization paths byte-for-byte. No selection, material build or optimizer copying is required. The actual42k Adam5250 state restores before candidate rates are applied. Native2k merged context, ratio1/16, checkpoint threshold1700, logical8/micro2, packing65536, clipping/penalty, shuffle seed/order, held400, learning32, retrieval and reporting remain unchanged. Both finish at50k unique questions and Adam6250.

Prepared designs are [constant](bench/FRESH42K_NATIVE8K_LR_CONSTANT2X_DESIGN_V1.json) and [cosine](bench/FRESH42K_NATIVE8K_LR_COSINE2X_DESIGN_V1.json). They deliberately remain `prepared_awaiting_root_review`; root can freeze them after review. Train with `bench/train_fresh42k_native8k_lr_v1.sbatch <frozen-design>`, then use `bench/evaluate_fresh42k_native8k_lr_v1.sbatch <same-design>` after successful completion. The latter accepts only the candidate's own complete epoch001 pair and Adam6250. No partial/best panel substitution. Readouts go to `reports/fresh42k_native_lr_cohort100/<run>__2k/result.json`.

Baseline training/reporting took92m14 and its final100 readout8min; each launcher reserves2h plus20min. Queueing/preemption is additional. Existing cosine applies indices0…999: its last applied AC rate is5.000012336995355e-6, with the nominal5e-6 endpoint at1000. The scheduler is unchanged and separately hash-pinned.

Validation:69 focused CPU checks pass, including exact scientific-config deltas, frozen input/pair hashes, unchanged runner helpers/report callbacks, scheduler endpoints and rejection of wrong optimizer clocks, incomplete finals, foreign reader pairs and wrong LR configuration. Python, shell and embedded Python syntax pass. No GPU run was performed. Existing frozen runner/helpers and product source are untouched.

[File/hash receipt](FRESH42K_NATIVE8K_LR_PREPARATION_V1.json) · [minimal delta](/workspace/IB/TMP/FIRST_TASK_AWARE_FINETUNING/native8k_lr_preparation/delta.patch) · [checks](/workspace/IB/TMP/FIRST_TASK_AWARE_FINETUNING/native8k_lr_preparation/checks.json)
