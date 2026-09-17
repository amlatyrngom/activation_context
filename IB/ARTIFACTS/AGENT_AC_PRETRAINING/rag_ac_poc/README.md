# RAG activation-context proof of concept

Authorized experiment: Qwen3.5-4B shared frozen base with distinct side/reader LoRAs; 4B-generated NQ passage QA; depth1, ratio1/16, minimum2 rows, no distractors. Accepted material is shared across all four runs and split by grouped source passages before QA synthesis.

| Run | AC/side learning rate | Reader learning rate |
| --- | --- | --- |
| A | 1e-4 | 2e-5 |
| B | 5e-4 | 2e-5 |
| C | 1e-4 | 0 |
| D | 5e-4 | 0 |

Local live entry: `/workspace/IB/TMP/SYNC/RAG_AC_POC/report.tressoir.html`. `preparation/` and `A/` through `D/` contain the detailed reports. `watch_reports.py` refreshes the overview from synced files; it does not monitor or control GPU jobs. The agent checks substantive progress with native waits every10minutes.

`run_poc.py prepare` creates the shared QA material. `calibrate` saves the common initialization before a disposable timing probe and determines one common size/pass budget. `run --run A` (etc.) verifies the initialization, trains, records exact diagnostics and saves the final pair. Remote launches use `launch.py` plus `training_source.json` to verify source hashes. Set `RAG_POC_SOURCE_ROOT=/root/sky_workdir`, `RAG_POC_ROOT=/root/activation_artifacts/SYNC/RAG_AC_POC`, and `PYTHONPATH=/root/sky_workdir`; run artifacts/checkpoints stay outside the disposable code mirror. Calibration retries charge prior calibration time.

Each run budgets up to120minutes including shared preparation/calibration, its own placement/training/reporting and final evaluation. Calibration selected 2,048 training items, 100 held-out items and three dataset passes: 6,144 presentations and 768 updates planned per run. The 512-item train calls are blocks; the report's internal epoch numbers count these calls, while `data pass` counts actual dataset passes. Training uses8 examples/update, exact full-vocabulary KL, independent side and reader learning rates, unchanged existing global gradient clipping, no top-k teacher cache and no drift penalty. LR0 keeps reader weights fixed but its gradients still enter the existing clip norm.

The verified training snapshot is `e35867eccd5213c00e83964a4f74a539ccb96ea1`, branch/tag `rag-ac-poc-20260915T171733Z`. Both nodes use `/opt/activation/.venv/bin/python`. On a fresh node, synchronize the frozen project dependencies before invoking that interpreter directly: from `/root/sky_workdir`, run `UV_PROJECT_ENVIRONMENT=/opt/activation/.venv uv sync --frozen`. The fallback image alone lacked the current project additions. The resulting installed package lists were checked to match on both nodes, and their initial exact diagnostic means matched on both panels.

Reports distinguish current-reader teacher KL from original-base teacher KL. Both teacher and student use the same authoritative answer continuation. Answer-only KL selects tokens overlapping the native answer parameter payload. Full-text and no-context controls expose reader drift and context dependence. Online loss, a fixed100-item training panel,100held items and the final full-training evaluation distinguish attained fit from generalization; they do not establish a mathematical quality ceiling or global convergence.

The generic status panel retains values from earlier phases. `phase`, `global step`, `run elapsed minutes` and `budget remaining minutes` are the live whole-run indicators. `evaluation items` can remain at100/100 during training; it describes the most recent evaluation. `examples`, `epoch`, `step`, token totals, `epoch elapsed`, `elapsed`, and peak memory belong to a 512-item training call. `checkpoint epoch` counts those calls; checkpoints are actually saved at the end of the run. `completed items` counts presentations through the last completed block; repeated passes count again. The separate `remaining minutes` is a block-boundary snapshot. The displayed loss and agreement describe the latest eight-item update; agreement is token-level teacher/student argmax agreement, not QA accuracy.

Disposable logs, tests, reviews and source snapshots are indexed under `IB/TMP/AGENT_AC_PRETRAINING/rag_ac_poc/`. The prior co-design implementation and unrelated user changes are preserved; this experiment's production delta is limited to progress/reporting in four existing files.
