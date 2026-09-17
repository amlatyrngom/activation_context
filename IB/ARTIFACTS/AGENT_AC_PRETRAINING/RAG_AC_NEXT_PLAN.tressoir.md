# Reliable AC training and live reports

Plan the next implementation around one training call, teacher targets computed before the first update, and a live report that updates in place. The interrupted POC remains evidence of the old behavior; it will not be resumed as the corrected experiment.

## Executive Summary

The POC hands its whole training set to one `train()` call. The trainer already owns epochs, reporting-set evaluation at fractional-epoch boundaries, validation, samples and checkpoints at epoch ends; the POC's own block loop, phase counters and in-loop diagnostics go away. No hooks or time budget are added: the calibration step keeps choosing a dataset size that fits the run.

Before its first update, the trainer computes the teacher's top-k log-probs for every training, reporting and validation item with the reader as it is at that moment, using the existing `teacher_cache_top_k` cache. Every loss and every evaluation in the call reads those stored targets; no live teacher forward runs after that pass. That is the snapshot: stored numbers, not a copied adapter. It also removes the per-pass teacher forward the old runs paid on every epoch.

| Work | Where | Result |
| --- | --- | --- |
| One training call, trainer-owned boundaries | `rag_ac_poc/run_poc.py` | Epoch and update counters that mean what they say |
| Upfront teacher targets, evaluation from the same targets | `activation/ac_model/ac_model_training.py` | A fixed objective for the whole call |
| Honest series names and status fields | `activation/ac_model/ac_model_reporter.py` | Curves labeled by what they compare |
| In-place widget and plot updates, JSON as the live path | `activation/common/reporting.py` | Reading position and controls survive refreshes |

The extension is being fixed separately; nothing here waits on it. The plain-browser fixture is the acceptance check for our renderer. No new utility module, training framework, report framework or adapter copy is planned.

## Requested Decisions

Accepted from chat: one training call with no callback hooks; teacher log-probs computed up front instead of a branched adapter; only work on this side of the extension boundary.

**Accepted (chat, 2026-09-15): one top-k for training and evaluation, k = 256 with a remainder bucket.** The stored targets hold the top-256 token ids and log-probs plus the remaining mass per completion position, and the teacher's log-prob of the gold token. Coverage is reported on the page; the smoke run measured 99.98 % of the mass inside the top-256. Exact full-vocabulary KL is still computed once, after training, in the final controls table.

## Milestones

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title">M1 — One training call</span><span class="card-oneliner">The POC delegates the whole run to the trainer it already has.</span><span class="card-badge">Completed</span></summary>

#### Completion report

**What landed.** `run_poc.py · run()` builds one `ActivationContextTrainer` and makes one `train()` call with the training items, the held-out items as `reporting_data` and the same items without their passage as `validation_data` (`without_context()`), `num_epochs` = planned passes, panels every quarter epoch, checkpoints per epoch. After it returns: `training_stats.json`, the exact `diagnostics(controls=True)` table (a KL-config trainer for the SFT arm), one `trainer.eval` over all training items, the frozen-reader assertion for arm B, `result.json`. The block loop, `block_*.json`, in-loop diagnostics, the `diagnostic_time` plot, the private `_save_epoch`/`_sample_completions` calls and the `calibrate` command are gone. `budget()` derives `calibration_next.json` from the smoke run's measured phase rates; `RUNS` carries the four arms and the smoke shape.

**Drifts, challenges, unplanned steps.** The reviewer found `config()` passing `num_epochs` twice (crashed the first smoke launch) and the final `eval` raising in the reporter; both fixed before the second smoke launch. `budget()` now clamps to the prepared material and counts unique teacher views instead of `items + 2·held` (the no-context items share their held item's teacher). `prepare()` takes `RAG_POC_TRAIN_GOAL`; the material was re-prepared at 4096 training items (held chunks unchanged, 25 of 100 held questions re-generated) so the arm size is a real choice. The smoke run's rates put the two-hour arm at 2816 × 2, below the 3072 fallback; the budget rule was followed rather than the target size.

**Focused actual diffs.**

`IB/ARTIFACTS/AGENT_AC_PRETRAINING/rag_ac_poc/run_poc.py · run()` (excerpt; the reporter factory `RunReporter` and `budget()` omitted)

```diff
-        for epoch in range(settings['planned_passes']):
-            ... blocks, budget check, trainer.train(AC, block), phase(...), block_*.json, periodic diagnostics
+        trainer=ActivationContextTrainer(h,config(label,num_epochs=settings['planned_passes'],reporting_interval=settings['reporting_interval'],
+                                                  completion_samples=settings['completion_samples'],checkpoint_every_epoch=True))
+        stats=trainer.train(AC,train,reporting_data=held,validation_data=no_context,reporter=r)
+        write_json(folder/'training_stats.json',stats.summarize())
+        final=diagnostics(exact_trainer,ac,held,r,'final exact controls',controls=True)
+        full=trainer.eval(AC,train,release=False,reporter=r,reporting_name='final_all_training')
```

**Validation.** CPU: `python -m py_compile run_poc.py`; `budget()` and the answer parsing exercised on fabricated stats. GPU: smoke run S (256 × 2, 100 held) completed in 21.8 min on one L40S with 64 updates, every phase visible on the page; `final_controls.json`, `full_training.json`, `result.json` written. Arms A–D launched from `calibration_next.json` (2816 items, estimated 7143 s).


#### Planning Overview

`run()` builds the trainer once and calls `train()` once with all training items, the held-out set as `reporting_data` and `validation_data`, `num_epochs` set to the planned passes and `checkpoint_every_epoch` on. The trainer's `reporting_interval` replaces the POC's ten-minute timer: the held-out panel is evaluated at each fraction of an epoch. Preparation and placement happen once. Optimizer state and the global update clock are untouched by any reporting boundary because there is only one call.

The block loop, the per-block `phase()` status writes, the `block_*.json` dumps, the in-loop `diagnostics()` calls, the custom `diagnostic_time` plot and the private `_save_epoch` call are removed. Calibration still picks the size that fits the two-hour window; there is no budget check inside the trainer. The existing `diagnostics(controls=True)` runs once after `train()` returns as an exact, clearly secondary comparison against the original base and the no-context controls.

#### Planned Changes

`IB/ARTIFACTS/AGENT_AC_PRETRAINING/rag_ac_poc/run_poc.py · run()`

```diff
@@ run_poc.py — run() @@
-        for epoch in range(settings['planned_passes']):
-            order=list(train);random.Random(SEED+epoch).shuffle(order)
-            for lo in range(0,len(order),settings['block_size']):
-                block=order[lo:lo+settings['block_size']]
-                ... budget check, trainer.train(AC,block,reporter=r), phase(...), write_json(block_...)
-                if (time.monotonic()-last_eval >= 600 and ...): evaluate('periodic')
-        checkpoint=trainer._save_epoch(ac,trainer._next_epoch_number(ac))
+        trainer=ActivationContextTrainer(h,config(label,num_epochs=settings['planned_passes'],
+                                                  reporting_interval=REPORTING_FRACTION,checkpoint_every_epoch=True))
+        stats=trainer.train(AC,train,reporting_data=held,validation_data=held,reporter=r)
+        final=diagnostics(trainer,ac,held,r,'final exact controls',controls=True)   # secondary, once
```

`IB/ARTIFACTS/AGENT_AC_PRETRAINING/rag_ac_poc/run_poc.py · config(), RunReporter` — `teacher_cache_top_k` set to the chosen k; `RunReporter` keeps only the run label and the final summary. The `diagnostic_time` plot and block counters are deleted rather than hidden.

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title">M2 — Teacher targets before the first update</span><span class="card-oneliner">Store the training-start reader's log-probs once; train and evaluate against them.</span><span class="card-badge">Completed</span></summary>

#### Completion report

**What landed.** `ActivationContextTrainer.train()` clears `teacher_cache`, `teacher_target_logp` and the teacher texts at entry and, for KL, calls `_store_teacher_targets()` over training, reporting and validation examples right after the capacity check and before the optimizer exists: one no-grad teacher forward per unique teacher view, top-256 ids and log-probs plus the remainder mass, and the teacher's log-prob of every gold target token. `_example_loss` raises if a KL example has no stored targets; there is no live teacher path. `_chunked_kl`/`_chunked_sft` also return detached per-token target log-probs, from which `_evaluate` reports `gold_nll`/`teacher_gold_nll` per item, per kind and in the summary (whole call; `answer_positions` restricts to the answer in the bench). Completions sample the teacher once at baseline and reuse its text. `save_teacher_targets`/`load_teacher_targets` round-trip the store; `_save_epoch` writes `teacher_targets.pt` beside the AC checkpoint. `eval(teacher="stored"|"current")` chooses between the stored targets and filling missing ones from the current reader. `teacher_cache_top_k` defaults to 256 and must be positive for KL; SFT ignores it and only rejects the drift term.

**Drifts, challenges, unplanned steps.** The plan had the drift term reuse the stored targets; it keeps its own frozen-adapter forward (unchanged semantics, no storage growth). `validation_data` distinct from `reporting_data` now also gets a baseline evaluation (`stats.validation_baseline`). The co-design private tests (`codesign_implementation/test_ac_training.py`) needed adapting to the new return shapes and to storing targets in their fixture; the SFT config validation had to stop rejecting the new default top-k because the study generator builds SFT configs with it.

**Focused actual diffs.**

`activation/ac_model/ac_model_training.py · train()`

```diff
             self._check_capacity(ac_model, [*examples, *reporting_examples, *validation_examples])
+            if config.loss_kind == "kl":
+                # The teacher of this call is the reader as it is now; nothing after this line recomputes it.
+                stats.teacher_targets = self._store_teacher_targets(ac_model, [*examples, *reporting_examples, *validation_examples], device, reporter)
+                if reporter is not None:
+                    reporter.report_teacher_targets(stats.teacher_targets)
```

`activation/ac_model/ac_model_training.py · _example_loss()`

```diff
-        top_k = config.teacher_cache_top_k if is_kl and torch.is_grad_enabled() else 0                  # eval is exact
-        cached = self.teacher_cache.get(example.teacher_cache_key) if top_k else None
-        if is_kl and cached is None:
-            with torch.no_grad(): ... teacher forward, cache top-k ...
+        if is_kl:
+            cached = self.teacher_cache.get(example.teacher_cache_key)
+            if cached is None:
+                raise RuntimeError(f"{example.item.item_id}: no stored teacher targets; train() stores them before its first update "
+                                   "and eval() needs them for its items (teacher='current' stores them from the current reader)")
```

New methods `_store_teacher_targets`, `_target_logp`, `save_teacher_targets`, `load_teacher_targets` (about 80 lines) are omitted here; the full delta is `git diff rag-next-baseline-20260915T191150Z rag-next-20260915T194308Z -- activation/ac_model/ac_model_training.py`.

**Validation.** Private tests `IB/TMP/AGENT_AC_PRETRAINING/rag_ac_next/test_fixed_targets.py`: 5 passed (targets stored exactly once per unique view and bit-identical across updates; reporter labels and status; save/load parity; second call takes new targets and unseen items need `teacher='current'`; gold log-probs match a manual forward and SFT stores nothing; config validation; external evaluation without progress). Co-design AC tests: 19 passed; remaining co-design suite 47 passed with 4 GPU-only fixture errors (no CUDA locally) and one pre-existing trajectory-chunking failure outside this delta. The independent reviewer re-implemented the baseline `_chunked_kl`/`_chunked_sft` and found the losses and gradients bit-identical. GPU: the smoke run stored 356 unique views (256 train + 100 held; the 100 no-context items share the held teachers) in 43 s at 99.98 % coverage.


#### Planning Overview

Today `_example_loss` runs the teacher forward with the reader's own adapter name, so the teacher moves with every update, and `_evaluate` forces the exact live path. The cache exists but fills lazily during the first pass, after earlier items have already updated the reader.

The change: for KL runs, `train()` adds a "teacher targets" phase after building examples and placing the model. It runs the teacher forward under `no_grad` once per training, reporting and validation example with the reader as it is at that moment, and stores the top-k targets in the existing `teacher_cache`. After that phase the live teacher path is never taken in the call: `_example_loss` requires a cache entry when grad is enabled, and `_evaluate` reads the same entries instead of forcing exact mode. The mean top-k coverage is recorded once so the coarsening is visible in the report. `train()` already clears the cache at entry, so a later explicit call takes new targets from its then-current reader.

Teacher completion samples come from the baseline sample pass, before any update, and are retained in the samples table; later epochs sample only the student. The targets are saved beside the epoch checkpoint so a post-run evaluation can reload the same reference; loading the trained checkpoint never silently makes the final reader the teacher. SFT builds no targets. The drift term keeps its own, separately named reference (the original base with no adapter) and is unaffected.

#### Planned Changes

`activation/ac_model/ac_model_training.py · train()`

```diff
@@ ac_model_training.py — train() @@
             self._check_capacity(ac_model, [*examples, *reporting_examples, *validation_examples])
+            if config.loss_kind == "kl":
+                # The teacher for this call is the reader as it is now; nothing after this line recomputes it.
+                if reporter is not None:
+                    reporter.set_status(phase="teacher targets"); reporter.render(force=True)
+                stats.teacher_coverage = self._store_teacher_targets(ac_model, [*examples, *reporting_examples, *validation_examples], device)
```

`activation/ac_model/ac_model_training.py · _example_loss()`

```diff
@@ ac_model_training.py — _example_loss() @@
-        top_k = config.teacher_cache_top_k if is_kl and torch.is_grad_enabled() else 0                  # eval is exact
-        cached = self.teacher_cache.get(example.teacher_cache_key) if top_k else None
-        if is_kl and cached is None:
-            with torch.no_grad():
-                teacher_hidden = target.decoder_forward(..., lora_name=lora, ...)
-                ...
+        cached = self.teacher_cache.get(example.teacher_cache_key) if is_kl else None
+        if is_kl and cached is None:
+            raise RuntimeError(f"{example.item.item_id}: no stored teacher targets; train() computes them before the first update")
```

`activation/ac_model/ac_model_training.py · _store_teacher_targets()` (new method on the trainer, no new module) — one no-grad teacher forward per example with the current reader, `_teacher_top_k`, CPU storage under the existing cache key, mean coverage returned. `_evaluate` and `_sample_completions` lose their live-teacher branches. `checkpoint_every_epoch` writes `teacher_targets.pt` next to the epoch checkpoint; a `load_teacher_targets(path)` method restores it for post-run evaluation.

`activation/ac_model/ac_model_training.py · ActivationContextTrainingConfig` — `teacher_cache_top_k` must be positive for KL (default 256); its comment describes the training-start reader, not "the adapter as of the item's last exact pass".

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title">M3 — Names and status that match the run</span><span class="card-oneliner">The trainer's reporter is the report; its labels say what is compared.</span><span class="card-badge">Completed</span></summary>

#### Completion report

**What landed.** `ActivationContextTrainingReporter` takes `set_labels` so the page says "AC student, held-out" and "no context, current reader, held-out" instead of `reporting`/`validation`; the teacher series is `TEACHER_SERIES` = "teacher (stored at training start)". Widgets are created in reading order (gold NLL, reporting KL, loss, evals, epochs, kinds, references, throughput, gradient, agreement for KL). `report_teacher_targets` shows the stored count, k and coverage; `report_phase` writes `epoch` as "1 of 2", `global updates` as "n of planned", elapsed and remaining, and clears the stale evaluation-items counter while training; `report_eval` adds the gold-NLL series with the teacher line on the reporting set and records `last panel`; `report_epoch` records `last checkpoint`.

**Drifts, challenges, unplanned steps.** The reviewer found `report_eval(progress=None)` (an external `trainer.eval` with a reporter) raising inside `round("external")`; fixed and covered by a CPU test. The bench's `RunReporter` seeds the run summary first so `set_text` later replaces it in place instead of appending it at the bottom. The trainer's default reference series (`no_context`, `recent_text`, `untrained_ac`, `in_context`) still appear empty in the KL plot's legend; cosmetic, left as is.

**Focused actual diffs.**

`activation/ac_model/ac_model_reporter.py · report_phase()`

```diff
-        self.set_status(phase=name, epoch=progress.epoch_number, step=f"{progress.step} / {progress.steps}", ...)
+        self.set_status(phase=name, epoch=f"{epoch} of {planned_epochs}", update=f"{progress.step} of {progress.steps} in this epoch",
+                        global_updates=f"{self.global_step} of {planned_updates}", elapsed=_duration(elapsed), remaining=_duration(remaining))
+        self._clear_status("evaluation_items")
```

**Validation.** Covered by the M2 private tests (series names, status keys, widget order) and the smoke page: status strip `epoch 2 of 2 · global updates 64 of 64 · teacher targets 356 items · top-256 · 100.0 % of the mass`; gold-NLL plot with the three labeled readers and the teacher line.


#### Planning Overview

The main status area shows phase, epoch out of total, update within the epoch, global updates, one elapsed pair and the last completed held-out evaluation step. Item counters appear only during preparation, teacher targets and evaluation. Checkpoint identifiers and phase times stay in the epochs table. Nothing left over from the POC's block loop remains.

Series are named by what they compare. The first two plots score all three readers against the gold answer: "gold-answer log-loss on held-out items" and "exact match on held-out items", each with the AC student, the frozen teacher and the no-context control. The third is "KL to the training-start teacher" on held-out and a fixed training panel; then training loss per update. Token agreement is labeled agreement, never accuracy. The final exact controls from M1 land in one clearly secondary table. Unused plots are removed; a pending table states when it will be filled. The layout is the one in the mock dashboard under `IB/TMP/AGENT_AC_PRETRAINING/rag_ac_next/`.

#### Planned Changes

`activation/ac_model/ac_model_reporter.py · __init__(), initialize_training(), report_phase(), report_eval()` — label and status-key changes only; no new widgets. Omitted here: the exact strings, chosen at implementation.

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title">M4 — Live page that updates in place</span><span class="card-oneliner">Fix the renderer we own; JSON is the live path.</span><span class="card-badge">Completed</span></summary>

#### Completion report

**What landed.** `activation/common/reporting.py`: the page boots once (`window.__reportPageBooted`), keeps one poll loop, fetches `report_data.json` and updates widgets in place through a section map; plots use `Plotly.react` with a constant `uirevision` so zoom and scroll survive; a fetch failure shows a visible line with bounded backoff; `location.reload()` is gone. `HtmlReporter.render()` writes the HTML on the first render, when the widget set changes, on finish and otherwise at most every `page_rewrite_seconds` (default 60).

**Drifts, challenges, unplanned steps.** The plan asked for HTML writes at init, structural change and final only; a 60-second rewrite floor was kept so a page opened from a fresh copy is never more than a minute stale. This is a deliberate deviation, noted in the docstring.

**Focused actual diffs.** `activation/common/reporting.py`: 196 lines added, 74 removed, in the page script and `render()`; the delta is best read whole (`git diff rag-next-baseline-20260915T191150Z rag-next-20260915T194308Z -- activation/common/reporting.py`).

**Validation.** Headless Chromium fixture `IB/TMP/AGENT_AC_PRETRAINING/rag_ac_next/renderer/`: 28 of 28 checks (single boot, no reload, in-place updates, plot state preserved, failure line and recovery). The smoke and arm pages were pulled every 15–30 s during the runs and rendered live.


#### Planning Overview

The page purges every plot and rebuilds the widget subtree on each JSON update and on `tressoir:render`, and the HTML file is rewritten every ten seconds. Both are ours to fix. Widgets are kept by their stable names and updated in place; plots use `Plotly.react` so zoom and trace visibility survive; one polling loop is started once and the render handler is idempotent. The HTML shell is written at initialization, when a new widget is added, and at the final render; `report_data.json` carries everything else. A failed fetch shows a visible status line and keeps polling; the page never reloads itself on a data change.

The extension's own fixes are out of scope. When they arrive, the same fixture is rerun in the viewer as a separate check; it is not a gate for this work.

#### Planned Changes

`activation/common/reporting.py · render() (JavaScript), poll(), boot()`

```diff
@@ reporting.py — page script @@
-      if (window.Plotly) plots.forEach(function (p) { try { Plotly.purge(p.div); } catch (_) {} });
-      widgets.innerHTML = ""; build every widget again
+      for each widget in report.widgets: node = byName[widget.name] || create(widget); update(node, widget)
+      for each plot: Plotly.react(div, traces, layout)   // keeps the user's zoom and legend state
+      remove nodes whose names are gone
```

`activation/common/reporting.py · HtmlReporter.render()`

```diff
@@ reporting.py — HtmlReporter.render() @@
-        page written every page_rewrite_seconds
+        page written at initialization, on a new widget, and when final=True; JSON otherwise
```

`IB/ARTIFACTS/AGENT_AC_PRETRAINING/rag_ac_poc/run_poc.py · plain_report()` — drop `page_rewrite_seconds`.

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title">M5 — Small proofs before another long run</span><span class="card-oneliner">Deterministic CPU tests and a plain-browser fixture.</span><span class="card-badge">Completed</span></summary>

#### Completion report

**What landed.** `IB/TMP/AGENT_AC_PRETRAINING/rag_ac_next/test_fixed_targets.py` (5 CPU tests on the tiny fixture), the renderer browser fixture under `rag_ac_next/renderer/`, adapted co-design private tests, and the two mainline GPU tests updated to the new contract: `test_basic_agent_ac_training.py` evaluates before training with `teacher="current"` and passes the no-context references as `validation_data`; `test_basic_ac_self_distillation.py` does the same for its "before" panel and loads `teacher_targets.pt` into the fresh trainer before the reload comparison.

**Drifts, challenges, unplanned steps.** The self-distillation test was missed in the first pass and caught by the reviewer. The mainline GPU tests were not run in this block (the node's four GPUs were taken by the smoke run and the arms); they are the first thing to run on the next GPU session.

**Focused actual diffs.**

`activation/tests/test_basic_ac_self_distillation.py`

```diff
-        before = trainer.eval(ac_name, items, reporter=training_reporter, reporting_name="before")
+        before = trainer.eval(ac_name, items, reporter=training_reporter, reporting_name="before", teacher="current")
...
-        reloaded = ActivationContextTrainer(harness, config).eval(ac_name, items)
+        reloaded_trainer = ActivationContextTrainer(harness, config)
+        if loss_kind == "kl":   # a fresh trainer scores against the saved training-start targets, not against the trained reader
+            assert reloaded_trainer.load_teacher_targets(Path(stats.checkpoint_path) / "teacher_targets.pt") == len(items)
+        reloaded = reloaded_trainer.eval(ac_name, items)
```

**Validation.** `uv run --frozen python -m pytest test_fixed_targets.py -q` → 5 passed; co-design `test_ac_training.py` → 19 passed; renderer fixture 28/28; independent review `IB/TMP/AGENT_AC_PRETRAINING/rag_ac_next/IMPLEMENTATION_REVIEW.md` (four defects, all fixed before the arms; numerics verified bit-identical).


#### Planning Overview

Private tests under `IB/TMP/AGENT_AC_PRETRAINING/` on the tiny fixture with a nonzero reader adapter and several real optimizer updates: targets exist for every training, reporting and validation example before the first update; no teacher forward runs afterwards (count adapter-named decoder calls on teacher token sequences); the stored tensors are byte-identical after training while reader and encoder parameters change; evaluation reads the stored targets; a second `train()` call rebuilds them from the updated reader; saving and reloading the targets gives the same evaluation; SFT stores nothing.

One single-call multi-epoch test with the AC reporter checks one placement, continuous update clocks, held-out evaluations at the configured fractions, status keys cleared between phases and populated named series. The two existing mainline AC training tests are updated only where the positive top-k default changes their configuration.

A plain-browser fixture loads a long report, scrolls down, opens a disclosure, focuses an input and zooms a plot, then applies JSON updates, a structural update, repeated render events and a fetch failure. Scroll, focus, disclosure, zoom and a single poll loop must survive all of them.

Only after these pass and the implementation is reviewed is a fresh, bounded RAG POC proposed. Retained QA material is reused if its split and digest checks pass. A and B's moving-teacher weights are not resumed; C and D are not launched automatically.

#### Planned Changes

`IB/TMP/AGENT_AC_PRETRAINING/rag_ac_next/test_fixed_targets.py`, `test_single_call.py`, `browser_fixture/` — private. Omitted here: fixture construction, which follows the existing tiny-model tests.

</details>

## Fair metric, run sizes and the cross-run ablation

**Gold-answer log-loss is the headline metric.** Every item's target is the NQ gold answer inside a `submit_answer` call (`build_item` in `run_poc.py`), so the same tokens can be scored under the frozen teacher with the full passage, the AC student with the compressed passage, and the no-context control. None is favored by construction; the teacher can be wrong and the student can beat it. Reported per answer token on the held-out panel at every evaluation, alongside greedy exact match against the gold answer. KL to the teacher stays as the training objective and the "distance to teacher" curve. The teacher's gold log-loss is a constant per item, computed during the targets pass; the student's comes from the evaluation forward the KL already needs. The [mock dashboard](../../TMP/AGENT_AC_PRETRAINING/rag_ac_next/mock_report/report.tressoir.html) shows the agreed page: summary, three-reader gold log-loss and exact-match plots, KL plot, training loss, panels, epochs, completions, final exact controls, gradient norm and throughput, in that order.

| Run | Items × epochs | Purpose | Wall time from recorded rates |
| --- | --- | --- | --- |
| Smoke | 256 × 2, 100 held, panel every quarter epoch | 64 updates: warmup ends early, every report phase fires twice; compare the page to the mock | 15–20 min |
| Ablation arm | 4096 × 2, 100 held per panel, all 300 held for the final controls | 1,024 updates on twice the unique passages of the old plan | about 2 h at the old 0.45 s/item; calibration confirms, 3072 is the fallback |

**Cross-run ablation, one wave, one arm per GPU, identical data, seed, initial adapter hashes and k = 256.**

| Arm | Encoder LR | Reader LR | Objective | Question it answers |
| --- | --- | --- | --- | --- |
| A | 1e-4 | 2e-5 | KL to frozen teacher | The main result |
| B | 1e-4 | 0 (reader frozen) | KL to frozen teacher | Does the reader need to adapt at all, or can the compressor carry it alone? |
| C | 5e-4 | 2e-5 | KL to frozen teacher | Is the encoder learning rate in the right decade? |
| D | 1e-4 | 2e-5 | SFT on the gold answer | Does distilling the teacher beat direct supervision on the metric we actually report? |

If only two GPUs are available, A and B run first and C and D in a second wave. A and B are the pair the POC cannot do without.

**Batch block, 3–4 h, no human checkpoints.** Nodes launch and warm while M1–M5 are implemented and CPU-tested (about 75 min). The smoke run follows on one GPU (about 20 min) and its page is compared to the mock. Then all arms launch at once (about 2 h), the final controls and the per-arm summary land in each report, and a short cross-arm table is written to this folder. If implementation overruns, the arms drop to 3072 items rather than losing an arm.

**Execution record (2026-09-15 batch block).** Baseline tag `rag-next-baseline-20260915T191150Z`; reviewed training snapshot `rag-next-20260915T194308Z` (commit `f0d16431`), verified on the node by `launch.py` against `training_source.json`. One AWS node `rag-next-4` (4 × L40S; the catalog offers 1, 4 or 8 per instance). Smoke run S: 21.8 min, rates train 0.55 s/item, held panel 0.36 s/item, sampling 2.7 s/item, targets 0.12 s/view. Budget rule → 2816 × 2 (estimated 7143 s of 7200). The shared material was re-prepared at 4096 training items (611 s) so the size was not capped by the old 2048. `sky exec` reserves every GPU of the node, so the arms were started over ssh, one process per GPU, with a sleeping sky job holding off the 30-minute autostop; their folders were pulled every 30 s. Helpers: `IB/TMP/AGENT_AC_PRETRAINING/rag_ac_next/{remote.sh,snapshot.py,summarize_arms.py}`.

**Result (all four arms completed, node torn down).** Full table and reading: [ABLATION_RESULTS.md](./rag_ac_poc/ABLATION_RESULTS.md); run pages under `IB/TMP/SYNC/RAG_AC_POC/{S,A,B,C,D}/report.tressoir.html`.

| Arm | Held gold NLL: AC student / no context / teacher | Exact KL to teacher: student / no context | EM student / teacher | Reader drift (exact KL) | Minutes |
| --- | --- | --- | --- | --- | --- |
| A 1e-4 / 2e-5 KL | 0.584 / 0.586 / 0.132 | 0.466 / 0.466 | 0.06 / 0.51 | 0.019 | 108 |
| B reader frozen | 0.581 / 0.620 / 0.132 | 0.473 / 0.507 | 0.06 / 0.51 | 0 | 108 |
| C 5e-4 / 2e-5 KL | 0.586 / 0.589 / 0.132 | 0.468 / 0.470 | 0.05 / 0.51 | 0.017 | 108 |
| D SFT on gold | 0.538 / 0.538 / 0.132 | 0.525 / 0.525 | 0.07 / — | 0.098 | 92 |

With a trainable reader the AC student ends where the no-context reader ends (per-item mean difference 0.006 nats): the reader learns the answer prior in the first quarter epoch and the compressed rows add nothing on unseen passages, while the training set is fitted (A: KL 0.213 on training items vs 0.466 held). Only the frozen-reader arm shows a passage signal from the rows (0.04 nats, 0.03 KL over its control). Learning rate and objective do not change this. The recipe as run does not produce a compressor the reader uses; see the results document for what it does and does not establish.

## Stopped experiment and retained evidence

A and B were cancelled at your request; all training and monitoring processes exited. The last reported update was 665 for A and 576 for B; the last complete held-out panels were at 576 and 384. No final evaluation or trained checkpoint pair was produced. Initialization, QA material, logs and completed panels are retained.

At A's last completed panel, current-teacher held KL was 0.120 while original-teacher held KL was 22.590. A falling moving-teacher objective cannot establish preservation of the original behavior. This plan replaces that target contract with stored training-start targets.

[Stopped local reports](../../TMP/SYNC/RAG_AC_POC/report.tressoir.html) · [Experiment source](./rag_ac_poc/README.md) · [Extension handoff](./RAG_AC_EXTENSION_HANDOFF.md) (separate track). Cloud teardown and recovery evidence is in `IB/STATE.md` and `IB/TMP/AGENT_AC_PRETRAINING/rag_ac_poc/`.
