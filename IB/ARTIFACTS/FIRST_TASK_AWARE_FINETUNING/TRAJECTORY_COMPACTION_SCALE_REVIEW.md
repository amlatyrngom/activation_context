# Compaction scale review — 2026-09-24

**Verdict: pass** for the revised private CPU preparation/cache boundary. The full GPU runner must still pass its prepare → train → stop → reload smoke before the 50M supervised-token launch. This review changed no product code and made no remote calls.

The target is approximately 20,000 cuts with 8k/16k/32k histories and complete suffixes up to 8,192 tokens. The small L40S execution proof establishes feasibility, not campaign throughput.

| Local measurement | Observed | Linear estimate for 20,000 cuts |
|---|---:|---:|
| Item construction + product example tokenization, 12 real cuts | 3.76 s | 105 min |
| Prepared Python graph before serialization | 20.3 MiB | 33.1 GiB |
| Streaming pickle on disk | 3.16 MiB | 5.14 GiB |
| Reloaded Python graph | 25.1 MiB | 40.9 GiB |
| Pickle load | 0.073 s | 121 s |
| Remaining capacity validation | 0.729 s | 20 min |

These are scaling estimates from a small, bucket-stratified coding/research sample, not a representative full-run benchmark. Pickle reconstructs some previously shared integer objects, explaining the larger loaded graph. Keep the planned 120 GB CPU preparation and 128 GB GPU-host memory allocations; avoid simultaneously holding fresh and reloaded full graphs.

Three high-value actions:

1. **Prepare once on CPU; reuse on every GPU restart — implemented privately.** `bench/run_trajectory_tasks.py` now streams one pickle graph containing selected items and their actual product `_Example` objects, publishes it atomically, and checks selection/configuration, both tokenizers, encoder configuration, and source identities. It preserves item aliases, returns cached examples for original items, and builds transformed controls normally. Eight bounded CPU workers avoid the projected serial startup; only the two-worker real-fixture path has been tested locally. This is trusted, owned per-run cache data, not a portable imported dataset. Host paths remain part of its identity.
2. **Measure total throughput on the larger GPU — launch gate.** The prior eight-example L40S runner recorded 82 s of training plus about 57 s inside the epoch outside reported panels, validation, and checkpointing, principally reference preparation. Therefore the displayed roughly 10 s/example omits substantial work. Including that residual gives roughly 17.4 s/example before panels/startup; extrapolating that tiny smoke would exceed 90 hours, not establish a reliable ETA. Measure at least 32 varied examples including reference preparation. The parent will test existing logits chunk size 256 and resident optimizer moments on the larger GPU, retaining the separately proven L40S chunk-64/offload fallback.
3. **Defer the remaining capacity-pass optimization.** It still tokenizes every side input on resume and could cost approximately 20 minutes. Six-hour segments reduce repetition. Caching the required lengths is a possible later private optimization; skipping validation without equivalent capacity checks is not justified.

One correctness defect was found and resolved: JSON converts `adam_betas` from tuple to list, so direct identity comparison rejected the second invocation. The parent normalized both identity dictionaries through JSON before comparison. The actual-main reload test now passes.

Validation: private `IB/TMP/FIRST_TASK_AWARE_FINETUNING/consolidation/test_compaction_prepared_cache.py`, **1 passed in 27.15 s**. Six real cuts (four training, two held) match across serial and two-worker preparation for every example field. Actual main prepare/reload preserves item identity, refuses to rebuild on cache hit, accepts a changed checkpoint path, and rebuilds transformed controls. Real local tokenizer and metadata-only AC fixtures were used; no model weights or CUDA initialization were required.

Python 3.14 warns that the parent is multithreaded when forking. The tested CPU-only path completed without hanging, but this does not prove every host/thread combination safe. Keep `CUDA_VISIBLE_DEVICES=''` before imports, tokenizer parallelism disabled, and Torch threads set to one; GPU-visible preparation must not fork. The pending real GPU smoke and first CPU preparation remain the appropriate integration checks.

## Real GPU cache/restart follow-up, September 24 04:30 UTC

Job23618567 completed the actual runner: two updates, exit75, a fresh process restoring the paired optimizer checkpoint, then two further updates. LR clocks and penalty draws match. This verifies execution and restored state, not bitwise equivalence to an independent run. Reviewer compaction_teacher_review found all V7/V8 product source, initial AC panels, frozen controls, token totals and initial forward diagnostics identical. Small gradient differences begin in the first backward pass, before restart (encoder norm1.0837835 versus1.0853164). Numerical backward variation is the leading explanation; no concrete cache/resume defect was found. A shared-prefix uninterrupted-versus-resumed replay is the focused follow-up.

Full50M CPU preparation23618804 completed in11m49s with50.1GiB peak RSS and a4.4GiB cache:18,408trainingcuts plus100held. Small GPU runner peak hostRSS was85.7GiB, so the main GPU launcher requests192GiB to leave room for the loaded full cache. Private profile wrappers were corrected before execution: they must not precreate the gate-owned training directory.

## Compact prepared vectors, September24 05:12UTC

Private sourceV9 (`c936960fe8f5e011651a3ab9c7f86fb5685eb567`) stores teacher/student token IDs, their target positions and target IDs as signed32-bit integer arrays, including penalty examples. It converts after the original cache key is computed. Public training items, assistant targets, spans, requests and identities remain unchanged. The reviewer audited all consumers and checked exact tensor/indexing values and gradients plus pickle/deepcopy. The actual six-cut serial/fork/cache-reload test passes, including unchanged key/token values. Product files remainV4.

CPUjob23624402 explicitly converted the trustedV8 cache and checked a fresh process load: full-cache peakRSS39.6→13.8GiB; observed load62.1→33.7s (a single check, not a controlled disk benchmark). File size increases4.697→5.922GB because compact raw integer arrays trade variable-length pickle coding for much smaller live objects. All18,508item/example aliases survive; old caches are retained. Main nowrequests128GiB. Real GPU stop/reload check23624499 COMPLETED in8m49s: four updates across two processes, restored optimizer, final reports and paired save; host peak63.75GiB. Main50M H100 job23624896 started after this gate. The shorter two-hour allocation obtained a slot; scheduler placement remains variable.

## Startup reporting follow-up

The private runner forwards phase changes to the overview with a throttled render. Several rapid startup transitions can leave an earlier phase in the JSON during the long all-item capacity check. The workspace private runner now forces a render only when the phase changes and exposes the unchanged capacity validation in 128-example chunks; syntax checks pass. Deploy at the next natural source revision with explicit prepared-cache identity migration. Active frozen sources remain unchanged. Do not interrupt active training for this reporting change. Current campaign overview explicitly labels startup/baseline measurements as pending.

The reporting fix also forces a complete overview write after phase, training-step and evaluation callbacks, avoiding a phase-only write suppressing the later progress fields. Private check `consolidation/check_campaign_report_phases.py` passes with the real HTML/training reporters: current phase, examples/epochs, current loss, cleared inactive evaluation count, and unchanged HTML bytes. The check loads no model.
