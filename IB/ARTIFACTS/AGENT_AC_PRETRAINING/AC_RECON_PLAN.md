# AC reconstruction night

Mass reconstruction SFT as the activation-context pretraining task, with a no-context penalty that charges the reader for anything it learns without the compressed rows. Two anchors are never touched; everything else is single-variable hill-climbing against a frozen evaluation protocol for about fourteen hours.

## Executive Summary

The RAG round ended in a null result: with a trainable reader the activation-context student landed exactly where the no-context reader landed on unseen passages (held gold NLL 0.584 vs 0.586). The diagnosis is that the reader takes the no-context shortcut — it learns the answer prior and memorizes question→answer — so the compressed rows never have to carry anything. This round removes both halves of that escape.

**The task is reconstruction.** The reader sees a passage only through its activation-context rows and reproduces it word for word (SFT on the passage tokens, no oracle, no teacher pass). Passages are tool outputs from OpenSWE and S1 deep-research trajectories — text the base model has not memorized. A cued-continuation variant keeps the first 20 % in plain text and targets the rest. There is no answer prior to fall back on: the only way to lower the loss is to decode the rows.

**The penalty is the second half.** Whenever the reader trains, the objective carries a second term on the same history with its activation-context parts stripped:

\[ \mathcal{L} = \text{task}(\text{AC view}) + \lambda \cdot \mathrm{KL}\big(\text{base}(\text{no-context view}) \,\big\|\, \text{reader}_{\text{adapted}}(\text{no-context view})\big) \]

The reference is the base model with no adapter, stored once per training example before the first update. The term is sampled on 25 % of examples per update and scaled by \(1/p\), so its expectation is \(\lambda \cdot \mathrm{KL}\) and it costs a quarter of a second forward. Default \(\lambda = 1\); raise it to 2–3 if the held no-context NLL drifts more than about 0.02 nats from the base. Skipped when the reader is frozen.

| Work | Where | Result |
| --- | --- | --- |
| Reconstruction and cued-continuation items | `activation/ac_model/ac_model_study.py` | A dense task with no oracle and no memorized answer |
| No-context view of a history | `activation/common/ac_parts.py` | `strip_ac_parts()` — the penalty's and the floor's input |
| Penalty term, sampled and unbiased | `activation/ac_model/ac_model_training.py` | Learning without the rows is charged; decoding them is free |
| Teacher-forced token accuracy | `activation/ac_model/ac_model_training.py` | A second metric SFT runs previously did not have |
| Round bench, ledger, frozen protocol | `ac_recon/` | Every run recorded before and after; one promotion rule |

Two anchors run untouched for the whole night: **A16** (ratio 1/16) and **A4** (ratio 1/4), both reconstruction SFT with a trainable reader and the penalty. Everything else is a single-variable arm judged against a same-size baseline on the frozen protocol. Up to three RTX PRO 6000 nodes; deadline about 13:30 UTC on 2026-09-16, including teardown.

## Requested Decisions

Nothing new to request. The following were accepted in chat on 2026-09-15 and are recorded here as the round's contract; they are written out in [RECONSTRUCTION_NIGHT.md](./RECONSTRUCTION_NIGHT.md).

**Accepted — the task.** Mass reconstruction SFT on text the model has not memorized (SWE-trace and deep-research tool outputs, windows of 400–2000 characters, plus LOFT/Loong for length if needed), plain assistant text with a minimal system prompt, no `submit_answer` wrapper and no tool schema. Two variants: full reconstruction and cued continuation. No oracle, no teacher pass.

**Accepted — the no-context penalty is the default when the reader trains.** \(\lambda = 1\) on a 25 % sample of examples per update, scaled by \(1/p\); the reference is the base with no adapter, cached once per dataset with the existing top-k machinery. Raise \(\lambda\) if the held no-context NLL moves more than ~0.02 nats. Skipped when the reader is frozen or \(\lambda = 0\).

**Accepted — two anchors, never touched.** A16 (ratio 1/16) and A4 (ratio 1/4), reconstruction SFT, reader trainable with the penalty, 2 epochs over the size the smoke run's budget rule selects, panels every 1/8 epoch.

**Accepted — hill-climbing with genuine methods.** Fair game: penalty weight and sampling, ratio curricula, reconstruction/continuation mix, packing, learning rates and schedules, encoder and row-scale initialization, staged freezing, data mix and chunk lengths. Not fair: train/held overlap by source, changing metrics or panels after a run starts, evaluation-time tricks, unflagged memorization.

**Accepted — operating envelope.** Up to three RTX PRO 6000 nodes for up to 14 hours with teardown without asking; product changes on a tagged snapshot, nothing committed to main; non-anchor experiments may be killed and replaced freely; old run folders moved aside, never deleted; best-recipe checkpoints pulled under a few-GB cap before teardown. A frozen evaluation protocol written before the first run, a ledger entry before and after every run, and a 15-minute monitor.

## Milestones

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title">M1 — The recipe in the product</span><span class="card-oneliner">Reconstruction items, the no-context penalty, token accuracy, the bench and its review.</span><span class="card-badge">Completed</span></summary>

#### Completion report

**What landed.** Baseline tagged `ac-recon-baseline-20260915T233324Z`; the reviewed snapshot the runs verify against is `ac-recon-20260916T000008Z`.

`ActivationContextStudyGenerator.generate_reconstruction_samples()` builds oracle-free items from plain `{"text", "doc_id", "source"}` passages: the activation-context view holds one part and the instruction, the in-context view holds the plain passage, the assistant output is the passage itself (variant `reconstruct`) or its tail after a cue that ends on a word boundary (variant `continue`). Kind is the variant name; degenerate passages are skipped.

`strip_ac_parts()` in `activation/common/ac_parts.py` returns the same messages with every activation-context part removed, collapsing a content list back to a string when only text is left. It is the no-context view for both the penalty and the validation floor.

The trainer carries the penalty end to end. `build_example()` attaches a second `_Example` on the stripped history when the reader trains and \(\lambda > 0\); its cache key is namespaced `#base` so it cannot collide with a real teacher entry. `train()` stores the base's top-k reference for those examples once, before the optimizer exists, reusing `_store_teacher_targets(..., lora=None)`. `_example_loss()` samples with a dedicated RNG, scores the stripped history under the reader adapter against the stored base distribution, and hands back `(λ/p)·KL`, which the update adds inside the same `backward()`. Evaluation never fires it: the `with_drift` guard short-circuits before the RNG is touched, so panels neither pay for it nor desynchronize the sampling stream. `stats.penalty` records the unweighted KL per update.

Teacher-forced token accuracy now comes out of both chunked losses (`_chunked_sft` and the `target_ids` branch of `_chunked_kl`) as `_last_hits`, and appears per item, per kind and in every evaluation summary. The reporter gained a `token acc` column in the evaluations and per-kind tables, a token-accuracy plot per evaluation set, and a "No-context penalty" plot with the value in the status strip.

`ac_recon/run_recon.py` (from the RAG bench) has `prepare`, `run --run <label>`, `budget` and `overview`; `ac_recon/launch.py` verifies the 111-file source manifest on the node before running anything. `PROTOCOL.md` and `LEDGER.md` were written before the first run.

**Drifts, challenges, unplanned steps.** The independent review ([IMPLEMENTATION_REVIEW.md](../../TMP/AGENT_AC_PRETRAINING/ac_recon/IMPLEMENTATION_REVIEW.md), verdict *incorrect-or-missing*) confirmed the learning machinery correct — alignment, reference, scale, unbiasedness, exclusion from evaluation, token accuracy under gradient checkpointing — and found two blocking defects in the round rather than the mechanism.

- **D1**: with the penalty on, an SFT run filled and persisted the 256-way teacher cache — about 0.5 MB per item, so roughly 11 GB of RAM and an 11 GB `teacher_targets.pt` per epoch checkpoint at the calibrated size. Fixed twice: `TOP_K` is 32 for this round (coverage is reported, so the coarsening is observable), and `_save_epoch` only writes the file for KL runs, since an SFT run's cache holds nothing but penalty references.
- **D2**: the fidelity plot's 320-token generation budget truncated about half the sampled passages, so a perfect reader would have scored near 0.65. Fixed by choosing the fixed completion samples from the panel items that fit the budget (a stable sort on `passage_tokens > completion_tokens - 32`) rather than by raising the budget and paying for it every epoch.
- Also fixed from the review's non-defect list: `result.json` now carries `held_novel` and `held_wiki` separately, since the top-level `held` is a blend of the two panels the round is designed to contrast (F3).

Left as recorded nits: the reading order of the two lazily created plots (F4), the penalty curve's carry-forward when no example is sampled (F5), unused penalty references built for panel items (F6), and `budget()` sizing from the module-level anchor (F7).

**Focused actual diffs.** Excerpts against `ac-recon-baseline-20260915T233324Z`; the full delta is `git diff ac-recon-baseline-20260915T233324Z -- activation/`.

`activation/ac_model/ac_model_training.py · build_example()`

```diff
-        key = (ac_model.name, digest, self.config.teacher_cache_top_k)
-        return _Example(item, teacher_ids, student_ids, teacher_positions, student_positions, ...)
+        key = (ac_model.name + ("#base" if reference else ""), digest, self.config.teacher_cache_top_k)
+        example = _Example(item, teacher_ids, student_ids, teacher_positions, student_positions, ...)
+        if not reference and self.config.no_context_penalty_weight > 0 and self.config.learning_rate_target_lora > 0 and requests:
+            stripped = strip_ac_parts(item.ac_messages)
+            example.penalty = self.build_example(ac_model, replace(item, item_id=item.item_id + ":no_context_reference",
+                                                 in_context_messages=stripped, ac_messages=stripped, in_context_start=item.ac_start), reference=True)
+        return example
```

`activation/ac_model/ac_model_training.py · train()` — the reference is stored before the optimizer exists

```diff
+            penalty_examples = [example.penalty for example in examples if example.penalty is not None]
+            if penalty_examples:
+                # The penalty's reference is the base with no adapter on the no-context history: fixed for the whole call.
+                stats.no_context_reference = self._store_teacher_targets(ac_model, penalty_examples, device, reporter,
+                                                                        reporting_name="no-context reference", lora=None)
```

`activation/ac_model/ac_model_training.py · _example_loss()` (the sampling guard and the term; the reference forward is omitted)

```diff
+        self._penalty_term = None
+        if with_drift and example.penalty is not None and config.no_context_penalty_weight > 0 and self._penalty_rng.random() < config.no_context_penalty_fraction:
+            # any improvement that does not need the rows (the answer prior, memorizing the question) is charged; decoding the rows is free
+            ...  reference forward under `lora`, scored against the stored base distribution ...
+            penalty, _, _ = self._chunked_kl(None, reference_states, head, cached=cached_reference)
+            self._penalty_term = (config.no_context_penalty_weight / config.no_context_penalty_fraction) * penalty
```

`activation/ac_model/ac_model_training.py · train()` — the update adds it inside the same backward

```diff
-                        loss = (objective + config.drift_term_weight * drift) * (example.item.weight / len(mini))
+                        penalty_term = self._penalty_term
+                        loss = (objective + config.drift_term_weight * drift + (penalty_term if penalty_term is not None else 0.0)) * (example.item.weight / len(mini))
                         loss.backward()
+                        if penalty_term is not None:
+                            totals["penalty"].append(float(penalty_term.detach()) * config.no_context_penalty_fraction / config.no_context_penalty_weight)
```

`activation/ac_model/ac_model_training.py · _chunked_sft()` — token accuracy out of the same chunk

```diff
-            return -token_logp.sum(), token_logp.detach()
+            return -token_logp.sum(), token_logp.detach(), (logp.argmax(dim=-1) == labels).detach()
...
+        self._last_hits = torch.cat(hits) if hits else targets.new_zeros((0,), dtype=torch.bool)
```

`activation/ac_model/ac_model_training.py · _save_epoch()` — the D1 fix

```diff
-        if self.teacher_cache:
-            self.save_teacher_targets(ac_folder / "teacher_targets.pt")     # the reference this run was trained against
+        if self.teacher_cache and self.config.loss_kind == "kl":
+            self.save_teacher_targets(ac_folder / "teacher_targets.pt")     # (an SFT run's cache only holds penalty references)
```

`activation/common/ac_parts.py · strip_ac_parts()` (new, 19 lines) and `activation/ac_model/ac_model_study.py · generate_reconstruction_samples()` (new, 53 lines) are omitted here; the study generator's item shape is one part plus the instruction in the activation-context view and the plain passage in the in-context view, with the passage (or its tail) as the assistant target.

`IB/ARTIFACTS/AGENT_AC_PRETRAINING/ac_recon/run_recon.py` — the D1/D2 fixes and the per-panel keys

```diff
+TOP_K = 32          # penalty reference only; 256 would cost ~0.5 MB/item of RAM and checkpoint
...
+        held.sort(key=lambda item: item.info['passage_tokens'] > arm['completion_tokens'] - 32)   # stable: the fixed samples fit the budget
...
+                  'held_novel': (last.get('by_kind') or {}).get('reconstruct'), 'held_wiki': (last.get('by_kind') or {}).get('wiki'),
```

**Validation.** CPU: `IB/TMP/AGENT_AC_PRETRAINING/ac_recon/test_recon.py` — 3 passed in 13 s (reconstruction and continuation item shapes against `strip_ac_parts`; a fresh `lora_name=None` forward reproduces the stored reference log-probs to 2e-2; the penalty changes the reader's gradients and its sampling is respected). The review added nineteen independent checks and five probes on the tiny fixture, including an argmax oracle for token accuracy under gradient checkpointing and a `save`/`load_teacher_targets` round trip with `#base` keys; the held split is by trajectory and the wiki panel is evaluation-only. `launch.py` recomputed all 111 manifest hashes on the node: 0 missing, 0 mismatched. GPU: the smoke run below is the first end-to-end exercise.

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title">M2 — Smoke run S and the budget</span><span class="card-oneliner">512 items × 2 epochs on one GPU, then the anchor size that fits 27,000 seconds.</span><span class="card-badge">Completed</span></summary>

#### Completion report

**What landed.** S ran 00:01–00:48 UTC on `ac-recon-1` GPU 0 from tag `ac-recon-20260916T000008Z`: 128 updates at 3.47 s each, novel gold NLL 1.905 → 1.629, wiki 2.338 → 2.261, no-context floor 1.921 → 1.855, greedy fidelity 0.07. The four things it had to establish held: end-to-end run, ~3.5 s per update of 8, the floor near the base value, novel token accuracy above the no-context value. `budget` read its per-phase rates and chose **21,888 items** (the largest multiple of 64 whose two epochs, eight panels per epoch, final seen-training evaluation and a 15 % margin fit 27,000 s); `calibration.json` was pushed to every node.

**Drifts, challenges, unplanned steps.** The base no-context floor was first measured with the passage still in the in-context view (0.016); fixed at tag 011200Z by stripping both views (runs started before 01:15 carry a wrong "base, no context" line, cosmetic). `remote.sh push-calibration` referenced a stale file name, so the calibration went by rsync. Two throughput probes followed S: S4 (4 per forward, no checkpointing) gave equivalent panels at 2.3× the speed but 92 GB; S3 (3 per forward) 1.58 s per update at 64 GB became the standard for every later arm. The anchors' budget of 27,000 s was exceeded in practice (431 and 434 min) because they ran one example per forward under the original recipe.
</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title">M3 — The two anchors</span><span class="card-oneliner">A16 and A4 launch from the calibrated size and are never touched again.</span><span class="card-badge">Completed</span></summary>

#### Completion report

**What landed.** A16 and A4 ran 00:38–07:50 UTC on `ac-recon-1` (GPUs 0 and 2) from tag `ac-recon-20260916T002847Z`, 21,888 passages × 2 epochs = 5,472 updates of 8, panels every 1/8 epoch, one example per forward, untouched throughout. **A16 (1/16): novel 0.878, wiki 1.349, token accuracy 0.78, floor 1.875 (base 1.921), seen-training 0.740, 431 min. A4 (1/4): novel 1.506, wiki 2.265, accuracy 0.68, floor 1.873, seen 1.218, 434 min.** The first question is answered: the recipe learns — A16's novel NLL fell 1.03 nats with the floor moving 0.05 (the honest gap the RAG round could not show). The second is answered the unexpected way: four times the row capacity is *worse* — A4 never transitioned in 5,472 updates, while the same rows with the reader frozen (R0F4) did.

**Drifts, challenges, unplanned steps.** The first launch failed the source-manifest check (the bench file had changed after tagging) and was relaunched from the v2 root at 00:38 (folders `A16_manifest`, `A4_manifest` kept). Both anchors used the shared checkpoint root; a mover script on the node kept later runs from colliding with their epoch folders. Their wall time (7.2 h) exceeded the 27,000 s budget because the calibration assumed the smoke run's rates and the anchors ran without the batched forward adopted afterwards.
</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title">M4 — Hill-climbing rounds</span><span class="card-oneliner">One change per arm, a same-size baseline, one promotion rule, a ledger entry before and after.</span><span class="card-badge">In progress</span></summary>

**The method.** An arm changes exactly one thing against a baseline of the same size and the same number of updates. The baseline is an anchor when the arm is anchor-sized, and a short same-size run started in the same wave otherwise — never a differently sized run, and never a run from an earlier wave with different data. Everything in `PROTOCOL.md`'s *fixed across runs* list stays fixed: seed 20260915, the shared initial adapters, 8 examples per update, 10 warm-up updates, clipping 1.0, and the panels with their item order.

**The promotion rule**, frozen before the first run: novel-panel gold NLL at the end of training, with the no-context floor unchanged from the base (drift ≤ 0.02 nats) and the wiki panel not improving faster than the novel panel. Ties break on token accuracy. Nothing is promoted on training loss, and nothing is promoted on the wiki panel — that panel exists to catch a run that is cueing memorized text rather than compressing it. A change that wins promotion goes onto node 1 as a full-size rerun before it is treated as the new recipe.

**Ledger discipline.** Every run gets a row in [LEDGER.md](./ac_recon/LEDGER.md) *before* it starts: time, label, node and GPU, the hypothesis, the one change, and the expected effect. The row is closed with the result afterwards, including runs that are killed early — a killed arm with its reason recorded is evidence; a killed arm with no row is not. Deviations from the protocol are recorded in the ledger rather than by editing the protocol.

**What is on the table**, from the accepted list: penalty weight and sampling fraction, ratio curricula, the reconstruction/continuation mix, reader-side packing, learning rates and schedules, encoder and row-scale initialization, staged freezing of the reader, and the data mix and chunk lengths. Arms run on node 2; node 3 is held for a scale run if one of these earns it. Non-anchor arms may be killed and replaced freely as results arrive.

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title">M5 — Results, checkpoints and teardown</span><span class="card-oneliner">A results document, the best checkpoints pulled, canon and state updated, every node gone.</span><span class="card-badge">Pending</span></summary>

At the 14-hour mark or earlier — the deadline is about 13:30 UTC on 2026-09-16 and teardown is inside it, not after it — the round closes in a fixed order.

The overview page and a results document in this folder record what each run did against the frozen protocol: the anchors, every promoted change with its baseline, and the arms that failed, with the ledger as the record of what was tried. The honest negative result is written as plainly as a positive one; the RAG round's value came from stating the null result precisely.

Best-recipe checkpoints are pulled before teardown under the few-GB cap. Old run folders are moved aside, never deleted. Durable outcomes — what the penalty does to the no-context floor, whether reconstruction escapes the shortcut, the operating facts that will outlive the round — are promoted into `IB/STATE.md` and the activation-context canon; the transient chronology stays in the ledger.

Then every node is torn down and `sky ls` is confirmed empty. Teardown is pre-authorized and does not wait for a reply.

</details>

## Frozen evaluation protocol

Written at 23:50 UTC on 2026-09-15, before any run of the round started, and not changed afterwards: [PROTOCOL.md](./ac_recon/PROTOCOL.md). The essentials:

| Panel | Size | What it detects |
| --- | --- | --- |
| novel | 200 windows from held trajectories, never trained on | Reconstruction fidelity on text the reader has not seen |
| wiki | 100 NQ passages | Memorization — text the base model has likely seen in pretraining |
| no-context floor | the novel panel with the rows removed | The reader's prior; the penalty monitor |
| seen-vs-unseen | 512 training passages after the last epoch | The train/held gap |

Metrics at every panel, all teacher-forced: reconstruction cross-entropy per passage token (novel and wiki separately), teacher-forced token accuracy, the no-context floor, and word-level difflib similarity of greedy reconstructions on 8 fixed novel items per epoch as a qualitative read.

One caveat the review raised and the run summary now states: the penalty's anchor and the reported no-context floor are two different references. The penalty holds the reader to the **base with no adapter**, while the validation panel measures the **current reader** without rows. The arm starts from a pre-trained initial reader adapter, so the floor is expected to move *toward* the base value early rather than to stay at its own initial value. That movement is the penalty working, not leakage.

## Monitoring and the operating envelope

A persistent monitor emits a status line every 15 minutes plus errors and result arrivals. Run folders are pulled from the nodes continuously (`remote.sh watch`), so every page under `IB/TMP/SYNC/AC_RECON/<label>/` is live locally.

Two nodes are up: `ac-recon-1` (anchors and promoted-recipe reruns) and `ac-recon-2` (single-variable arms). A third is reserved for scale and is launched only if a promoted change earns it. `sky exec` reserves every GPU of a node, so concurrent runs start over ssh, one process per GPU, with a sleeping sky job holding off the 30-minute autostop.

The bench verifies its own sources on the node before training: `launch.py` recomputes the `training_source.json` manifest against the reviewed snapshot and refuses to run on a mismatch. Product changes live on tagged snapshots (`ac-recon-baseline-20260915T233324Z` → `ac-recon-20260916T000008Z`); nothing is committed to main.

## Agent execution notes

Implementation, review, smoke, anchors and the hill-climbing waves run unattended in the authorized block; the user returns around 7–9 am their time. Do not read or edit TASK. Preserve the uncommitted co-design and RAG-round deltas in the working tree. Do not edit a helper script while a long invocation of it is alive (bash re-reads scripts mid-execution); `remote_run.sh` is the frozen copy for long-running invocations. `IB/TMP/SYNC` is large — never push it wholesale. Old run folders are moved aside, never deleted. Teardown of all nodes is part of the block; confirm with `sky ls` afterwards. Remaining review nits F4–F13 are recorded in `IB/TMP/AGENT_AC_PRETRAINING/ac_recon/IMPLEMENTATION_REVIEW.md` and were left for a later pass.
