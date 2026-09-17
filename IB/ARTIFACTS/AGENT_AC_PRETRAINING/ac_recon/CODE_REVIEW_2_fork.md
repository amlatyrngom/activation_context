# Code review 2 — results-informed (12:15 UTC 2026-09-16)

Scope: the AC reconstruction stack as it stands at tag `ac-recon-20260916T120158Z`: `activation/ac_model/{ac_model,ac_model_training,ac_model_utils,ac_model_study}.py`, `activation/harness/module_manager.py`, `activation/autotuners/*`, the bench `IB/ARTIFACTS/AGENT_AC_PRETRAINING/ac_recon/run_recon.py` + `arms.json`, read against the ledger, the results draft and the live panels (K16 0.288 novel at epoch 0.875; R4X 0.795 at 0.875; A4 flat at 1.51; G1b 0.654; A16 0.878; R0F4 0.906 ending after a 0.750 minimum).

Nothing here was tested; every finding names the cheapest experiment that would confirm it. Findings are ranked by expected value per hour of GPU. Section 3 holds the only items the user asked to be pinged about.

---

## 1. Findings (ranked)

### F1. No learning-rate decay; a 10-update warm-up — the late-run oscillations and rises are the constant 1e-4

- Where: `run_recon.py:177` (`warmup_updates=10`), `agent_training_utils.py:477` `set_learning_rates` (linear warm-up only, then `base_lr` forever), `ac_model_training.py:562`.
- What the results say: every run that transitions then wanders at the end. R0F4: 0.750 (1.25) → 0.901 → 0.813 → 0.772 → 0.828 → 0.805 → **0.906** at the end (the "encoder overfits late" reading in finding 4 of the results draft is more simply a constant LR that is too high once the loss is small). R4X: 1.005 → 1.177 → 0.795. A16: 1.163 → 1.206, 0.997 → 1.049, 0.968 → 1.001. G1b: 0.815 → 0.878 → 0.782. The panel-to-panel swings (0.05–0.15 nats) are as large as most arm effects we compare, which also explains why single-panel comparisons between arms have been unreliable.
- Change: a schedule in `set_learning_rates(optimizer, updates_done, warmup, total_updates, kind)`: cosine (or linear) from the base rate to 10 % of it over `total_updates = updates_done_at_call_start + steps × num_epochs` (resume-safe: the restored `updates_done` places the run on the same curve). Warm-up 50–100 updates instead of 10 (the fresh fp32 mixer/pooling/head modules start from random init; 10 updates of Adam at 1e-4 is not a warm-up). Config fields `lr_schedule: "constant" | "cosine"`, `lr_final_fraction`.
- Expected: monotone last third, final novel 0.03–0.10 nats lower than the run's own best panel today, and lower replicate variance of the *end* levels (the current end levels 0.77–0.91 for transitioned 1/16 arms are partly the phase of the oscillation at the last panel).
- Test: micro pair, SK recipe (512 × 2 epochs, gates off = SKn) with and without cosine decay — 15 min each on the L40S or a preemptable RTX. The micro run is short enough that the effect will be small; the real confirmation is the first 6-h ORCD arm with decay (F10 plan). Adopt for every new long arm regardless: there is no known downside to decaying to 10 %.

### F2. The no-context penalty sees only the base's top-32 — the reader's remaining freedom is the tail

- Where: `run_recon.py:29` (`TOP_K = 32`), `ac_model_training.py:1213` `chunk_kl_cached` (k+1-way coarsened KL), `_with_gold` at 1156.
- What the results say: gold-inclusive targets — closing one specific hole in the coarsened KL (mass moved onto a gold token outside the top-32) — were worth **+0.22 nats** (G1b 0.654 vs A16 0.878), the largest single effect of the night. That is direct evidence that what the penalty *fails to see* is exactly what the reader exploits. With k = 32 of a 151k vocabulary the "everything else" bucket still lets the reader reshape the tail freely: sharpen it toward the domain (URLs, markdown lists, diff syntax), which is a no-context improvement the metric charges to nothing and which competes with learning to read the rows. The floor still drifts 0.04–0.05 under λ = 1 and λ = 3 alike (P3 did not reduce it — a stronger weight on a blind penalty cannot see more), and P0 shows a freer reader gives a *worse* channel (0.912 vs 0.770).
- Change (preferred): `no_context_penalty_reference: "stored" | "live"`. "live" runs the base **without adapter** on the stripped history under `no_grad` at penalty time and takes the exact full-vocabulary KL with the existing `chunk_kl` (teacher states, no cache). Cost: one extra no-grad forward on 25 % of the examples ≈ +8 % step time; no RAM for stored targets at all (the 30-min reference pass at start disappears too — see F4). Fallback: k = 256 stored in fp16 (`_teacher_top_k` values `.half()`), ≈ 0.25 MB/item, 5.5 GB for 21,888 items, 10 GB for 40k — fits the nodes (256 GB) and the ORCD `--mem=120G` requests.
- Expected: floor drift ≤ 0.01; novel gains a fraction of the gold effect — my guess 0.03–0.08 — and less dependence on λ.
- Test: micro pair SKn vs SKn + live penalty (15 min each; watch the floor line and novel at 128 updates). Floor drift is small at micro scale, so also fold it into one 6-h ORCD arm.

### F3. Gradient clipping at 1.0 over encoder + reader together is binding at every step and couples the two

- Where: `ac_model_training.py:72` (`max_grad_norm=1.0`), 559–562 (one `clip_grad_norm_` over `ac_parameters + target_parameters`).
- What the stats say: SK's pre-clip norms: encoder 7.08 at the first update, 1.1 at the end; reader 2.06 → 3.58 (`grad_norm_ac`, `grad_norm_reader` in `training_stats.json`). The joint norm exceeds 1.0 essentially always, and the results draft records 10–40 at the transition. Adam is invariant to a *constant* gradient scale, not to a scale that changes every step: clipping to 1.0 turns the update into a per-step reweighting by 1/‖g‖, which down-weights precisely the high-gradient steps of the transition inside the Adam moments. Worse, the two groups share one norm: when the encoder's gradient spikes at the transition the reader's step is scaled down by the same factor (and the reader's 2–3.6 steadily dilutes the encoder's clip budget before it).
- Change: clip per group (encoder at 1.0, reader at 1.0, gates included in the encoder's) — `clip_grad_norm_` twice — and report both. Consider 5.0 for the encoder; keep the reader at 1.0 (its drift is what the penalty polices).
- Expected: earlier/sharper transition, possibly smaller onset variance; a risk of instability at the transition, which is why it is a micro test first.
- Test: micro pair SKn vs SKn + per-group clip (15 min each). Look at the update count of the transition and the loss spikes.

### F4. Every resume (and every start) rebuilds 40 min of examples and a 30-min reference pass — fatal for 6-h chunks

- Where: `train()` 409–412 (`build_example` for every item, single core), 456–461 (`_store_teacher_targets` for the penalty references, GPU), `run_recon.py` `resume_state`/`run` (rebuilds the study items and the examples from scratch), `save_teacher_targets` at 1134 exists but is only written for KL runs.
- What the results say: the results draft's engineering note (40 min of one core + ~30 min reference pass for 21,888 items). For 40k items that is ~2 h of a 6-h chunk *per chunk*; for 100k it is most of the chunk.
- Change: (a) build examples in a `multiprocessing.Pool` over the tokenizer (the work is pure tokenization; 8 cores → 5 min); (b) persist the no-context reference cache once per (material, arm) under the run's sync root (`teacher_targets.pt` is already the format; make `_save_epoch` write it for SFT runs when a penalty is active and let `train()` load it when present and its keys match); (c) with F2's "live" penalty the reference pass disappears entirely — the strongest argument for "live".
- Expected: resume cost from ~70 min to ~5 min; ORCD chunks become ~95 % training.
- Test: none needed beyond a resumed SR-size run; measure the resumed job's "run elapsed" at first update.

### F5. Checkpoints only per epoch — 6-h chunks need a mid-epoch checkpoint

- Where: `ActivationContextTrainingConfig.checkpoint_every_epoch` (98), `_save_epoch` (677), bench `resume_state`.
- Why: one epoch of 40k passages is 2.4 h on an RTX PRO 6000 at mb 3; of 100k, 5.5 h; the H200 rate is unknown until P8H. A chunk that dies at update 90 % of an epoch loses that epoch.
- Change: `checkpoint_every_updates: int | None`; save to `AC_MODELS/<name>/epoch_<n>_step_<s>` with `optimizer.pt` and a `partial_epoch.json` {epoch_number, step, seed}; the epoch's order is `random.Random(seed + epoch_number).shuffle` so resume replays it and skips the first `s` steps; `completed_epoch.json` semantics unchanged. Bench: `resume_state` prefers the partial record when it is newer than the completed one.
- Expected: at most `checkpoint_every_updates` × 1.6 s lost per preemption (500 updates ≈ 13 min).

### F6. Loss is a per-example token mean — a 25-token passage weighs as much as an 1,800-token one

- Where: `_batch_loss` 1002 (`cross_entropy[...].mean()` per example), `train()` 536–537 (`× weight / len(mini)`), `_evaluate` `summarize` 870–880 (item mean of per-item means).
- What the data says: training passages 21–1842 tokens (median 255, mean 279); at a fixed ratio the row count scales with length, so long passages are the ones where the encoder must actually pack (a 1,800-token passage into 112 rows) and they receive 1/70 of the per-token gradient a short one gets. The metric is item-mean too (consistent), but our claim is about compression per token.
- Change: `loss_weighting: "example" | "token"` — token: sum of CE over the update ÷ total target tokens of the update. Report both `gold_nll` (item mean, today) and `gold_nll_token` (token-weighted) in `summarize`, so the comparison with earlier runs survives.
- Expected: better long-passage reconstruction; the item-mean headline may move little either way. Medium priority; a micro pair decides it (15 min each).

### F7. Penalty sampling is i.i.d. per example — 10 % of updates carry no penalty, others four times the weight

- Where: `_batch_loss` 1008–1010 (`random() < fraction` per example), `_example_loss` 949.
- Why: with p = 0.25 and 8 examples the count per update is Binomial(8, 0.25): zero in 10 % of updates, ≥ 4 in 11 %; each sampled term is scaled 1/p = 4. The reader therefore alternates free updates and 4×-weighted ones. Unbiased, but a noisy constraint on a parameter set whose whole job is to stay still.
- Change: pick exactly `round(p × len(mini))` examples per update (a seeded `sample` over the mini-batch before it is split into micro-batches; keep the 1/p scale). Same expectation, no variance in the count. Trivial; no test needed beyond the unit test of the count.

### F8. "Replicates" share the initialization: the arm seed changes only the data order

- Where: `run_recon.py:84` (`torch.manual_seed(SEED)` in `harness()` before the fresh adapters/modules are created), `config()` (`seed=arm['seed']` → data order and penalty sampling only).
- Why it matters: K16r1/K16r2 (seeds 20260917/18) and A4r test data-order variance, not initialization variance; the results draft's "replicate variance in the onset" statements should say so. Cheap to fix (`torch.manual_seed(arm seed)` in `harness()` when `fresh_initial`); the running replicates stay valid as data-order replicates.

### F9. The panels cost ~1 h of every 7-h run

- Where: `run_recon.py` DEFAULT_ARM `reporting_interval=0.125`, `completion_samples=8`, `completion_tokens=320`; `_sample_completions` greedy.
- Why: 16 panels × (300 items forward + 8 × 320 greedy tokens) ≈ 3–4 min each. For 6-h ORCD chunks use 0.25 and 4 × 160-token samples (the fidelity metric is qualitative anyway); keep 0.125 for the node-1 headline runs whose curves we are reading.

### F10. What the metric measures: tool-output text is highly structured; the wiki panel is the honest prose number

- Where: `run_recon.py:98` `windows` (400–2000-char slices of tool outputs cut at line ends), `prepare()` 128–160 (72 % SWE, 28 % deep-research tool outputs), `ac_model_study.py:592` (single-turn "Passage:" reconstruction).
- Observation: K16 is at 0.288 nats/token on novel tool outputs (perplexity 1.33) with the base floor at 1.92: most of a search-result list or a diff is structure the reader predicts from a few cues, so "1/16 compression" on this material overstates what the rows carry per token. Wiki (short general prose, out of the training distribution) is 0.508 — not a memorisation signal (memorised text would sit *below* novel), just harder text. "Wiki above novel" should be read as the prose number.
- Change: report both panels as *gap to the floor* per panel (novel floor 1.92, wiki floor measured separately — today the wiki floor is not on the plot), and add 10–20 % general prose (Wikipedia/train split, books, docs) to the training material before the next 10× data step. The user's "one order of magnitude more data" is better spent on distribution breadth than on more tool outputs: K16 was still falling at 0.875 epoch on 21,888 items, so the 1/16 recipe is not data-limited on this distribution yet.

### F11. Smaller items (no experiment needed)
- `RowHead.scale` climbs 15× after the transition (0.0128 → 0.19 in R0F): rows enter the reader 15× louder than token embeddings. Legitimate, but record K16's `gate_trace` at the end to see whether the skip-init run also climbs; if it does not, the climb was compensating for the random projection, and the scalar could be frozen at the embedding RMS in future arms (one fewer moving part).
- The side LoRA's default dropout 0.05 (`register_lora`) is neutralised by `disable_dropout` in `train()`; fine, but say so in the config docstring — a reader of `arms.json` sees "dropout 0" for the reader only.
- `_padded_forward` relies on causality for right pads (documented); the eval path uses the same — fine. Reader and side LoRA weights are fp32 (checked in the G1b checkpoints) so the 2e-5 rate is not lost to bf16 rounding.
- Adam (0.9, 0.95), eps 1e-8, weight decay 0: fine for LoRA; weight decay 0 on the fresh fp32 modules is also fine at this scale.
- Autotuner: the Hopper fix is in; the checkpoint threshold falls back to "always recompute" when no explicit `gradient_checkpointing_min_tokens` — the bench always passes one. Publish the H200 profile as a preset once P8H completes so the 300-s cold tune is not paid per run root (the shared `AUTOTUNE` symlink already covers ORCD).

---

## 2. Explanations for the results (grounded in the code)

**Why a frozen reader wins at 1/4 (A4 flat at 1.51; R0F4/R4X transition).** The reader and the side encoder are the same 4B base with different LoRAs (`run_recon.py` NAME shared, `ac_model.py:157–160`). At init the old row head (`RowHead` without skip, `ac_model_utils.py:458`) is a random d→4d→d FFN: the rows are content-dependent but scrambled unit vectors. A *trainable* reader has two ways to lower the SFT loss: learn to read the rows (gradient must first make the encoder's rows separable, slow) or learn that the run of placeholder positions is noise and route around it (fast: a LoRA on q/k/v can learn to attend away from those positions, and the penalty does not see it because the penalty is computed on the *stripped* history where the rows do not exist). Once the reader discounts the row positions the gradient reaching the encoder through them shrinks — an attractor. At 1/4 there are four times as many noise positions (64 vs 16 for a 255-token passage), so both the incentive to suppress them and the dilution of each row's gradient are larger; at 1/16 the encoder wins the race in most runs (after 1.5–3k updates), at 1/4 it lost the one time it was tried. A frozen reader cannot take the shortcut, so the encoder's gradient stays intact (R0F4, R4X). The 1.51 plateau (below the 1.87 floor) is what the reader extracts from the rows' *count and coarse statistics* (length, topic) without decoding them. Prediction: with skip-init rows (K-recipe) the reader has no reason to suppress the rows, so **A4K should transition early**; its 0.125-epoch panel is the test and is due within the hour. If A4K also stalls, the mechanism is not the noise attractor and the next suspect is the row *count* itself (64 placeholder positions between "Passage:" and the instruction).

**Why skip-init removes the silent phase.** With `row_head_skip` the row is `RMS-normalize(x + FFN(x)) × scale` with `FFN` zero-initialised (`ac_model_utils.py:466–476`): at step 0 the rows are the side decoder's own last hidden states at the summary positions, RMS-normalised and scaled to the embedding RMS — i.e. approximately what the base model's final norm produces, which (with tied input/output embeddings, as the Qwen3 family has) is already a vector in the space the reader's embedding matrix speaks. The reader can read those rows on day one, so the gradient to the encoder is informative from the first update. K16's 0.125-epoch panel (1.105 vs 1.570) is exactly that. The random-FFN head forced the encoder to first invert a random projection while the reader was learning to ignore it — the race in the previous paragraph, whose outcome depended on the data order (R0F vs R0F3: same recipe, transition at ~3k updates vs never).

**Why gold-inclusive targets help (+0.22).** `chunk_kl_cached` (`ac_model_training.py:1213`) compares only the base's top-32 tokens and one lumped "rest"; on the stripped history the gold token is usually in the rest (the base cannot know the passage). The reader can then move mass onto gold tokens in no-context mode at no charge — i.e. it can *memorise passages through the LoRA* — and the SFT loss rewards exactly that, since the same gold tokens are the targets. `_with_gold` puts the gold token into the stored reference at its base log-prob, so the penalty now charges that move. This is why the effect is large and why F2 (the rest of the tail) is the natural continuation.

**Why the transition is sharp and variable.** The loss surface has two basins: "reader ignores rows" and "reader decodes rows". Which one wins depends on the order in which early examples arrive (the data order is the only thing that differs between R0F and R0F3, and between K16 and K16r1/r2 — F8) and on how early the encoder produces separable rows. Clipping (F3) makes the transition sharper than it needs to be: as soon as the rows become readable the encoder's gradient spikes (10–40), the joint clip scales the whole update down, and the run rides the clip boundary until the moments settle.

**Why B4 (more passes) beat A16 (more unique data).** With a constant LR (F1) and the item-mean loss, four passes over 10,944 passages give each passage four chances to shape the encoder after the transition; A16 sees each passage once, half of them before the transition. This is a statement about the *recipe's* onset, not about data value: with skip-init the transition is inside the first 500 updates, so X40K vs E4K is the first fair unique-vs-repeat comparison.

---

## 3. FUNDAMENTAL — would change the latent-context concept (the only items to ping the user about)

These alter what the rows are or how the reader consumes them. They are *not* recommended for tonight; they are the honest conclusions the numbers point toward.

**FC1. Frozen reader as the product, not a diagnostic.** The frozen-reader arms (R0F 0.880 at 1/16; R0F4/R4X the only working 1/4 runs) show the base model can read encoder rows *natively* — no reader LoRA. That is a different product: rows that any unmodified copy of the model consumes, several AC models coexisting on one served base, no penalty machinery, no reader drift to police, and the honest gap is the whole gap. The price at 1/16 today is ~0.1 nats (R0F 0.880 vs B 0.770; the K-recipe has not been run frozen yet). If the user's concept requires a reader adapter (e.g. the reader must also learn agent behaviours from the rows), this is off the table; if the concept is "compressed context the base model reads", the frozen reader is closer to it. Decision needed before the headline recipe is fixed. A test that does not need the decision: **RK16** = K16 recipe with `reader_lr=0` (frozen reader, skip init, no penalty), one full arm.

**FC2. Deep injection instead of input-embedding rows.** Rows enter at layer 0 as soft tokens (`_batch_loss` writes them over the placeholder embeddings) and must survive 36 layers of a reader tuned to read tokens. The 1/4 failure with a trainable reader and the 15× scale climb both say the layer-0 channel is narrow. Injecting per-layer rows (a KV-prefix or per-layer additive states, as in prefix-tuning/gist tokens) changes the reader's consumption path and the serving story (vLLM prompt embeddings would no longer suffice). Large concept change; do not start it in this campaign.

**FC3. Rows constrained to the token-embedding manifold.** Skip-init works because the rows start as near-soft-tokens. Going further — e.g. rows as softmax-weighted combinations of the embedding matrix, or a penalty pulling rows toward their nearest embeddings — would make rows interpretable and serving-trivial, at the cost of expressiveness. Concept-level because it redefines the row space. Not now.

---

## 4. Proposed experiment plan for the remaining budget

Fixed points: node 1 (4 × RTX PRO 6000) until ~19:00 UTC; ORCD: 4 preemptable RTX PRO 6000/H200 (2-day limit, requeue), 2 normal H200 (6-h chunks, resume implemented).

**Now (code, ~1.5 h, on a tagged snapshot):** F1 (cosine decay + warm-up 100), F3 (per-group clipping), F7 (stratified penalty count), F4a/b (parallel example building; reference cache persisted) and, if time allows, F2 "live" penalty (it removes the reference pass) and F6 token weighting behind flags. Unit tests for the schedule, the per-group clip and the count. None of these touch the running arms (they verify their manifest at start only).

**Node 1 (keep as planned):** X40K (headline 1/16), K16 → E4K at ~15:00 (repeat-vs-unique at the new recipe), A4K (the mechanism test; read its 0.125 and 0.25 panels — if it is below 1.3 by 0.25 epoch the noise-attractor story holds), R4X (1/4 frozen headline). Pull K16's epoch-2 checkpoint and X40K's before teardown.

**ORCD preemptable (RTX / H200, immediately after SR proves resume):** micro pairs against SKn (512 × 2, gates off), each ~15–20 min on RTX; run them in the 4 preemptable slots as K16r1/K16r2/A32K finish or in parallel when A8K is queued:
1. M-DECAY: SKn + cosine (F1).
2. M-CLIP: SKn + per-group clip (F3).
3. M-LIVE: SKn + live penalty (F2), if implemented; else M-K256.
4. M-TOKW: SKn + token-weighted loss (F6).
Read them by novel at 128 updates *and* by the floor line; anything ≥ 0.05 better than SKn (1.42 for SK) goes into the chunk recipe.

**ORCD normal H200 (6-h chunks, resume):** after P8H gives the H200 micro-batch (expect mb 6–8 with checkpointing ≥ 3,000 tokens; 141 GB): 
- H1: **K-recipe + decay (+ whatever the micro pairs approve) on the 100k material, 1 epoch**, 2 chunks chained with `AFTER_JOBID` — the unique-data headline the user wants, at 10× the anchors' data per epoch (~5.5 h at the RTX rate; faster on H200).
- H2: **RK16** (FC1 test: K16 recipe, reader frozen) — one 6-h chunk covers it.
Then the 1/4 line: **A4K-decay** or, if A4K stalls, **R4K** (frozen reader + skip init + gold-irrelevant) on 40k.

**Later (what fits 6 h / 2 × 6 h):** from the throughput of H1's first chunk, tabulate updates-per-6-h per GPU type; with F5 (mid-epoch checkpoints) any arm fits any number of chunks, so the question becomes purely the data size per epoch.
