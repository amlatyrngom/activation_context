# Code review 2 (fresh eyes) — AC reconstruction stack

Reviewer: independent, no session context; read-only. Written 2026-09-16 12:30–13:00 UTC against the working tree
(trainer line numbers as of 12:40 UTC; the file is being edited concurrently, so cite by the quoted expression when in doubt; the state *includes* the cosine schedule, `clip_per_group`, `penalty_count_exact` and the
`warmup` arm key that landed while this review was being written; the newest tag is `ac-recon-20260916T122307Z`).

Files read in full: `activation/ac_model/{ac_model,ac_model_training,ac_model_utils,ac_model_study}.py`,
`activation/harness/module_manager.py`, `activation/autotuners/{core,worker,fla}.py`, the bench
`ac_recon/{run_recon.py,arms.json,PROTOCOL.md,LEDGER.md}`, `AC_RECON_RESULTS.md`, `RESTART.md`, the AC canon, and
the helpers they call (`set_learning_rates`, `persistent_optimizer`, `head_logits`/`_HeadMatmul`, `decoder_forward`,
`checkpoint_adapter_scope`, `prompt_tokens`, `encode_with_part_sentinels`, `strip_ac_parts`).

Besides reading, three CPU-only analyses were run on the local mirrors (scripts in the session scratchpad; numbers
quoted below are reproducible from `IB/TMP/SYNC/AC_RECON/*/training_stats.json` and `preparation/material*.json`):
a 50-character shingle overlap between the held panel and the training pool; a per-item re-scoring of every finished
run's final panel split by that overlap and by passage length; a two-line torch probe of `Optimizer.load_state_dict`.
The learning-curve numbers (gradient norm, training loss, panels) come from the mirrored `report_data.json` files.

One correction to the brief: the recipe in the code is encoder LR 1e-4 / reader LR 2e-5 (`run_recon.py:31`), not
5e-5 / 1e-5. Everything below uses the code's values. Model facts used: Qwen3.5-4B has `tie_word_embeddings=True`,
d=2560, 32 layers of which 8 are full attention (every 4th), vocabulary 248,320 (from the cached `config.json`).

---

## 0. Summary of the ranking

| # | Finding | Kind | Confidence | Cheapest test |
|---|---|---|---|---|
| 1 | The held "novel" panel is contaminated by near-duplicate tool outputs: 12.5 % of items (40k material) and 21.5 % (100k material) share more than half their text with a training passage; the training pool is itself ~36 % near-duplicated | evaluation / data | verified (measured) | CPU re-scoring, done below |
| 2 | Constant LR after a 10-update warm-up; late-run regressions (R0F4, RL5, R4X) coincide with gradient-norm spikes of 100–500. Cosine decay landed at 12:24; it still needs the anchor-size test, and the micro arms MD/MA cannot measure it | schedule | high | K16 + cosine at anchor size (4.7 h) |
| 3 | Resume silently restores the *saved* learning rates: `optimizer.load_state_dict` overwrites `lr` and `base_lr`, so a resumed chunk ignores the arm's LR (matters now that chained ORCD jobs and schedules exist) | bug (latent) | verified (probe) | 2-line fix + CPU test |
| 4 | The mixer is dead weight that injects ~10 % random noise at init; the optimizer drives its `residual_scale` from 0.1 to 0.003 at Adam's ~1e-4 per step (~1,000 updates). 157 M fp32 parameters with Adam state for nothing | architecture / init | high (from the 07:20 checkpoint inspection + code) | midi arm `mixer_layers=0` (25 min on an RTX slot) |
| 5 | Row loudness: the reader needs rows ~15× louder than token embeddings; the single scalar `target_head.scale` walks there at 1e-4 per step, and saturated attention on loud rows is the likely source of the post-transition gradient spikes. The "gate group hurts" verdict rests on a run killed before its transition; SK vs SKn (1.423 vs 1.625) says the opposite once the skip init is in | init / optimizer | medium | midi arm: `target_head.scale` init ×4, or a group holding only that scalar |
| 6 | Loss is a per-example token *mean* averaged over examples: a 50-token passage weighs 10× a 500-token one per token; eval reports the same item mean (A4: 1.506 item-mean vs 1.364 token-weighted) | loss weighting | medium | `loss_weighting="token"` midi arm; report both metrics for free |
| 7 | The 0.22-nat gold-inclusive gain (G1b vs A16) is inside the replicate spread of that era (B vs B4-at-epoch-2: 0.24 nats, same recipe); the floors were identical. Plausible mechanism, unproven effect size | evidence | medium | K16 without gold (anchor size) |
| 8 | S-size micro arms (128 updates) can no longer resolve recipe questions: on the skip recipe the whole transition happens inside the first ~500 updates; decay/clipping effects are invisible at 128 | methodology | high | use 2,048 × 2 (512 updates) "midi" arms on ORCD preemptable RTX slots (~25 min) |
| 9 | 40 min of single-core example building + ~30 min reference pass per 40k arm; identical across arms of the same ratio/material — cache to disk | throughput | high | build once, load in later arms |
| 10 | Smaller: side LoRA is rank 128 while the protocol says 64; `recursive_adapter` (52 M fp32 + Adam) is carried and never used here; summary rows reuse the *token* position table; content enters the side model per-token RMS-normalised; the side checkpoint threshold counts one row's length while the reader's counts the batch | hygiene | verified | none needed |

Not found: any bug in the loss, the penalty alignment, the batched path, the stored references, the KL bound, the
teacher-forced hits, the gradient checkpointing scope, the optimizer persistence, or the autotuner's kernel selection.
The 12:00 review's table of those checks holds; I re-derived items 1–8 and 15 from the current code and disagree with
item 15 only in what "leakage" means (see finding 1).

---

## 1. Ranked findings

### F1. The held novel panel is contaminated by near-duplicates; the training pool is ~36 % near-duplicated

**Where.** `run_recon.py:99-112` (`windows`: random-length slices of every tool output, cut at line ends),
`:142` (dedup by sha1 of the *exact* window), `:148-149` (split by trajectory `doc_id`).

**What is wrong.** The same file is viewed in many OpenSWE trajectories of the same repository (and the same page is
fetched in several research trajectories). The trajectory split keeps those *documents* apart, and the sha1 dedup
removes only byte-identical windows — but the windows are cut at random lengths, so the same file content produces
different, heavily overlapping windows in a held trajectory and a training trajectory. The 12:00 review's item 15
("no exact leakage") is true and beside the point.

**Measured** (50-character shingles, training shingles at stride 8, held shingles at every offset; a shared substring
of 58+ characters is always detected):

| held vs | items with > 20 % of characters in shared substrings | > 50 % | > 80 % | > 95 % |
|---|---|---|---|---|
| `material.json` (40k train; all arms except X40K/R4X) | 49 / 200 | **25 (12.5 %)** | 13 | 7 |
| `material_100k.json` (X40K, R4X) | 69 / 200 | **43 (21.5 %)** | 24 | 11 |
| wiki panel vs either | 0 | 0 | 0 | 0 |

The >95 % items are the same file window under two trajectories (e.g. held #156 ≡ train #3916, byte-for-byte for the
first 160 characters). Within the training pool, of 1,500 sampled passages 36 % have more than half their text and
16 % more than 80 % of it present in *other* training passages.

**Does it change the results?** Modestly for the numbers, materially for two interpretations:

| run | novel (reported) | clean subset (151 items, < 20 % overlap) | leaked subset (25 items, > 50 %) | base floor clean / leaked |
|---|---|---|---|---|
| A16 | 0.878 | 0.922 | 0.798 | 1.944 / 1.603 |
| G1b | 0.654 | 0.691 | 0.580 | 1.949 / 1.618 |
| R0F | 0.880 | 0.922 | 0.791 | 1.978 / 1.696 |
| B | 0.770 | 0.807 | 0.693 | 1.947 / 1.625 |
| P0 | 0.912 | 0.958 | 0.830 | 1.761 / 1.466 |
| A4 | 1.506 | 1.589 | 1.227 | 1.943 / 1.603 |

The reported numbers are ~0.04 nats optimistic at 40k (the leaked items are also intrinsically easier: their base
floor is 0.3 lower), and will be ~0.08 optimistic for the 100k-material runs on the same panel. Rankings between arms
do not change. What does change: (i) "novel = never seen" is not true for 1 in 8 panel items, and the headline should
be quoted on the clean 151; (ii) the *unique data vs repeats* question (X40K vs E4K/B4) is confounded — the "unique"
pool repeats itself at the 36 % level, so a second epoch over 40k is not as different from 4 epochs over 22k as the
design assumes; (iii) the seen/held generalisation gaps (0.53 vs 0.65 for G1b) are slightly understated.

**Change.** (a) Re-score every `result.json` on the clean subset from the stored `per_item` rows (no GPU; the item
index is the last field of `item_id`, in `material['held']` order) and print both numbers in the results table.
(b) In `prepare`, after the trajectory split, drop held windows with > 20 % shingle coverage from the training pool
(or better, drop them from *held* and top up from other held trajectories, keeping the panel at 200), and record the
contamination rate in `material.json`. (c) For the "unique data" experiments, dedup the training pool by shingle
coverage (keep the first window of each near-duplicate cluster) before counting "unique passages".

**Expected effect.** Honest panel numbers (+0.04–0.08 nats on the headline); a defensible "novel" claim; a cleaner
unique-vs-repeat comparison. No effect on training.

**Cheapest confirming experiment.** Already done above (CPU, 10 s). The re-scoring of all runs is a 15-minute CPU job.

### F2. No decay in the schedule; late-run regressions track gradient-norm spikes

**Where.** `agent_training_utils.py:478-490` (`set_learning_rates`: warm-up only until 12:24, cosine now optional),
`run_recon.py:31-33` (`warmup=10`, `lr_schedule='constant'` defaults), `ac_model_training.py:595-605` (clipping and the schedule call).

**Evidence.** Every arm ran at a constant LR (1e-4 / 2e-5) after 10 updates. Panel-by-panel curves from the mirrors:
R0F4 novel 0.772 (epoch 1.625) → 0.805 → 0.906 (final) with its pre-clip gradient norm rising from ~16 to 487;
RL5 0.877 (1.75) → 0.959 → 1.044 with the norm reaching 125; R4X oscillates 1.005 → 1.177 → 0.795 → 0.80 with norms
of 40–180; A16 itself is non-monotone (1.163 → 1.206, 0.968 → 1.001). Before the transition every arm's gradient norm
sits at a flat ~1.6 (R0F, A16, A4, B, G1b, RL5, R0F3 for all 8,208 updates); after it the clip at 1.0 is always
active. K16 (skip init) has norms of 2.5–5.9 and a smooth curve, but is still at full LR with panels still falling
(0.288 at epoch 0.875), so the last 20–30 % of its updates are spent at a rate that the spiky arms could not tolerate.

**Change.** Cosine to 10 % (now in the code) plus warm-up 100 for fresh adapters (LoRA B starts at zero, the first
steps only move B; 10 updates of warm-up on fresh Adam moments produce the 12–27 gradient norms seen in A4K's first
panels), and `clip_per_group=True` so the encoder's spikes stop rescaling the reader's step. For a headline run that
already exists (X40K, K16), the cheapest use of the decay is a *resumed last epoch at a lower rate* — which needs F3
fixed first.

**Expected effect.** 0.03–0.10 nats on the final panel (typical for a run whose curve is still falling at the end and
whose late panels regress), and removal of the R0F4/RL5-style late losses. Zero risk to the transition since the
skip-init transition completes long before the decay bites.

**Cheapest confirming experiment.** Not the S-size micro arms MD/MA (128 updates: the decay window is 100 updates
of a curve that has not left the initial descent). Use the anchor-size pair **K16D = K16 + cosine + warm-up 100 +
per-group clip** on node 1 after K16 (same seed, same data order, so the panels are paired) — 4.7 h; or the
half-size pair on ORCD (2.4 h each).

### F3. Resume restores the checkpoint's learning rates, not the arm's

**Where.** `ac_model_training.py:502-510` — `optimizer.load_state_dict(state["optimizer"])` at `:506`.

**What is wrong.** `torch.optim.Optimizer.load_state_dict` replaces every param-group hyperparameter (`lr`,
`base_lr`, `betas`, `weight_decay`) with the saved one; only `params` is kept from the live groups. Probe:
a group created with `lr=3e-5, base_lr=3e-5`, loaded from a state saved at 1e-4, reads `lr 1e-4, base_lr 1e-4`
afterwards. `set_learning_rates` then scales that restored `base_lr`. Today every resumed arm uses the same LR, so it
is invisible; it becomes a real defect the moment a chained ORCD chunk is resubmitted with a different arm (a
lower-LR final epoch, an `lr_schedule` change that alters `lr_final_fraction`, a gate group added), and it silently
ignores the change. The same applies to `betas`/`weight_decay`.

**Change.** After `load_state_dict`, re-apply the configured groups:
`for group, wanted in zip(optimizer.param_groups, groups): group["base_lr"] = wanted["lr"]` (and `betas`,
`weight_decay` from the config). Add an assertion to `test_resume_restores_optimizer_and_update_count` that a
resumed trainer with a different `learning_rate_ac` reports that rate in `stats.learning_rates[0]`.

**Expected effect.** Correctness of every "resume with a changed schedule" experiment; enables the cheap "decayed
last epoch" variant of F2 on existing checkpoints.

**Cheapest confirming experiment.** The CPU test (minutes).

### F4. The mixer is a noise source at init and is switched off by the optimizer over ~1,000 updates

**Where.** `ac_model_utils.py:339, 372-373` (`residual_scale` 0.1 shared by the attention and FFN sub-blocks of each
`BlockedLocalAttentionLayer`); `ac_model.py:142` (2 layers by default), `:387-389` (applied to the unit-RMS
embeddings before the side decoder), `:401` (`unit_rows(content)` afterwards). Also `AdaptiveSummaryPooling`
(`:435, 454-455`) and `WindowedPooling` (`:389`, unused at stride 0) with the same 0.1 residual.

**What is wrong.** At init the mixer's four residual adds are 0.1 × random-projection outputs of unit-scale inputs:
roughly a 10 % RMS perturbation of every token embedding, applied *before* the side decoder sees the passage. The
07:20 checkpoint inspection found the mixers' `residual_scale` collapsed to ~0.003 ("the mixer is bypassed; the side
decoder carries the encoding"). Adam moves a scalar by about its LR per step whatever the gradient, so 0.1 → 0.003 at
1e-4 is a ~1,000-update walk during which the encoder's input is corrupted — the same silent-phase mechanism as the
row scale, in the other direction. The bidirectional local mixer was designed for the recursive/pooled setting; on a
300-token passage with stride 0 the causal side decoder (a 4B LM with a rank-128 LoRA) already sees everything.
Cost: 2 layers × (qkv 3d² + proj d² + FFN 8d²) ≈ 157 M fp32 parameters + gradients + two Adam moments ≈ 2.5 GB of
GPU memory and a forward/backward per part, for a module the optimizer removes.

**Change.** `mixer_layers=0` as an arm key (`ActivationContextModules` handles an empty `ModuleList`; the config
is structural, so this is a `fresh_initial` arm, which the K recipe already is). If the mixer is kept for the recursive
setting, initialise `residual_scale` at 0.0 (or `proj`/`down` weights at zero, as the row head now does) so it starts
as the identity. Same for the summary pooling's attention/FFN residuals.

**Expected effect.** Faster early descent (the encoder no longer has to learn around its own noise), ~2.5 GB less
memory, slightly faster steps; final NLL equal or better. If the effect on the final number is nil, the memory alone
justifies it.

**Cheapest confirming experiment.** A midi pair (2,048 × 2 epochs, K recipe) with `mixer_layers` 2 vs 0, ~25 min each
on an ORCD preemptable RTX slot; compare the epoch-0.5 and final panels.

### F5. Row loudness is the channel gain, and it lives in one scalar that Adam moves at 1e-4 per step

**Where.** `ac_model.py:227-239` (`_ensure_scales`: `target_head.scale` = target embedding RMS 0.0128),
`ac_model_utils.py:458-476` (`RowHead`: every row RMS-normalised, one scalar for all rows), `ac_model_training.py:491-498`
(the optional gate group holds *every* scalar: row scale, `input_scale`, `position_scale`, all residual scales).

**What the data say.** In R0F (frozen reader) the row scale climbed 0.013 → 0.080 → 0.191 over two epochs — the rows
ended 15× louder than token embeddings. With a frozen reader the only way to make the reader attend more to the rows is
to make them louder in the residual stream (the first RMSNorm removes magnitude at the row's own position, but the
row's share of the residual stream at deeper layers, and therefore of the keys/values the passage positions attend to,
grows with it). That is a *gain* the frozen reader cannot supply itself, and at 1e-4 per step the scalar needs
≥ 1,800 updates to cover +0.18 — the same order as the 1.5–3k silent phase. The ledger's counter-argument (SC's scale
settled at 0.03 in 25 updates at gate LR 3e-3 and "stayed flat") only shows the gradient asked for little *before*
the channel existed; it says nothing about the post-transition climb. The G16 verdict ("gates hurt") compares a
run killed at epoch 0.5 — before A16's own transition at 0.625–0.75 — inside the replicate spread. And the newest
micro pair says the opposite: SK (skip + gates 3e-3) 1.423 vs SKn (skip, no gates) 1.625 at S size.

The same loudness plausibly explains the post-transition gradient spikes (R0F4 487, R4X 179, RL5 125): attention
logits from passage positions onto 15×-scaled row keys saturate, and softmax saturation produces exactly the spiky,
example-dominated gradients that clipping then turns into a noisy direction.

**Change.** (a) Initialise `target_head.scale` at 3–4× the target embedding RMS (≈ 0.04–0.05; R0F's epoch-1 value
was 0.08) — a one-line init change, no optimizer group. (b) If a gate group is used, put *only* `target_head.scale`
in it (the `input_scale` moves the side LM's input magnitude away from what it was pretrained on and the residual
scales want to go to zero — those should not be fast). (c) Consider a soft cap (`scale = s_max · sigmoid(θ)`) so the
scale cannot run into attention saturation; with F2's decay this may be unnecessary.

**Expected effect.** Shorter or absent silent phase on non-skip recipes (already largely solved by the skip init),
a smoother post-transition gradient norm on frozen-reader arms, and a cleaner answer to the gate question than G16
gave. On the K recipe: probably small on the final number, possibly meaningful on stability at 1/4.

**Cheapest confirming experiment.** Midi arms on the K recipe: `scale_init ×4` vs control; and `gate_lr 1e-3` on the
row scale only vs control. Read the `gate_trace` (already logged every 25 updates) and `grad_norm_ac`.

### F6. Per-example token mean: short passages dominate the gradient and the metric

**Where.** `ac_model_training.py:1094` (`cross_entropy[...].mean()` per example), `:1160` (`total / num` per example),
`:570` (`× weight / len(mini)`); eval item mean at `:955-963`.

**What is wrong.** The update's loss is the mean over examples of each example's mean over its tokens: a 50-token
passage's tokens carry 10× the per-token weight of a 500-token passage's. Held tokens run 51–1,299 per item
(mean 291, median 252), so the gradient's variance is driven by the short tail, and the reported "novel NLL" is an
item mean that differs from the per-token NLL by 0.02 (A16: 0.878 vs 0.857) to 0.14 (A4: 1.506 vs 1.364). Not a bug —
it is consistent between training and eval — but it is a non-standard choice for an LM objective and it hides that A4
failed mostly on *long* passages (13 of its 200 items show no signal at all, median length 481 tokens; A16 has 2).

**Change.** `loss_weighting: "example" | "token"` (token: sum of per-token losses in the update / total target
tokens in the update). Report both the item-mean and the token-weighted NLL in every panel (free: `per_item` has
`positions`).

**Expected effect.** Lower gradient variance, a small gain on long passages, a metric that matches how the rows will
be used (long tool outputs). Direction on the item-mean metric: neutral to slightly worse on short items.

**Cheapest confirming experiment.** A midi arm; but adding the token-weighted *metric* to the reports costs nothing
and should be done now.

### F7. The gold-inclusive gain is one pair of runs inside the era's replicate spread

**Where.** `ac_model_training.py:1242-1251` (`_with_gold`), `:1198-1200` (storing).

**What the code does, and why it should help.** The coarsened KL is a lower bound of the true KL by the
data-processing inequality; separating the gold token from the remainder bucket gives a *finer* partition and hence a
tighter (≥) bound. Concretely, the plain top-32 bound is blind to mass the no-context reader moves onto the gold token
from *other* remainder tokens — which is precisely the shortcut SFT rewards at the base's high-entropy positions, where
the SFT gradient is largest. The gold-inclusive bound sees that move through the shrinking remainder. So the finer
penalty pushes back on the "gold without rows" direction of the reader's LoRA, which is nearly the same direction as
SFT-while-ignoring-rows; the surviving reader gradient is the row-dependent one. This is a sound mechanism.

**What the data do not establish.** G1b (0.654) vs A16 (0.878) is a single pair; in the same era B (0.770) vs
B4-at-epoch-2 (1.014) — identical recipe and data order — differ by 0.24, and R0F vs R0F3 differ by "transitioned
vs never". G1b's floor drift equals A16's (1.883 vs 1.875), so the effect is not visible where the mechanism says it
should act first. The gain may be real and this size, or half, or an onset accident. Under the skip recipe the
transition is deterministic enough (K16 within 500 updates) that the ablation is now cheap and decisive.

**Change.** None to the code. Run **K16 without gold** at anchor size (or half size), paired with K16.

**Expected effect.** Either a confirmed 0.1–0.2 nats attributable to the penalty targets, or the freedom to drop a
component. Either way the headline recipe gets a defensible ablation.

### F8. S-size micro arms no longer resolve anything on the skip recipe

**Where.** `arms.json` (`SC/SK/SKn/SL/SCn`, and the new `MD/MC/MP/MA`: 512 × 2 = 128 updates).

**Why.** On the K recipe the transition is *inside* the first 500 updates (K16: 1.105 at 552 updates; SK at 128
updates: 1.423 with the loss still falling steeply). 128 updates test only initialisation effects (that is why SK vs
SC was informative). A cosine decay over 128 updates (MD) or per-group clipping before any gradient spike exists
(MC) will show noise. The L40S micro node is 17× slower than an RTX slot, so it cannot host longer arms.

**Change.** Define a "midi" size: 2,048 passages × 2 epochs = 512 updates, `reporting_interval 0.25`,
`completion_samples 4`, `micro_batch 3`, `ckpt_min_tokens 1700`. On an ORCD preemptable RTX PRO 6000 that is
~14 min of training + ~10 min of loading/building/panels. Run MD/MC/MP/MA at that size, not S.

### F9. Example building and the reference pass are rebuilt per arm

**Where.** `run_recon.py:283-290` (items), `ac_model_training.py:424-426` (`build_example` for every item, single
process), `:482` (base no-context references, ~30 min for 40k items).

**What is wrong.** For a given material, ratio and TOP_K, the built examples and the stored base references are
identical across arms — every 40k arm spends ~70 min recomputing them (X40 was killed and restarted three times, each
time paying it). `teacher_cache` already has `save/load_teacher_targets`.

**Change.** Cache `{material, ratio, TOP_K, gold_targets}` → `(examples token ids, teacher_targets.pt)` under
`preparation/cache/`; parallelise `build_example` with a process pool (it is pure CPU tokenisation).

**Expected effect.** ~1 h saved per large arm; more arms in the remaining window.

### F10. Smaller items (verified, low impact)

- **Side LoRA rank.** `module_manager.py:134` registers the side adapter with `config.side_lora_rank` = 128
  (`ac_model.py:96`); the protocol and the results doc say rank 64 for both. `initial/ac/side_lora/adapter_config.json`
  confirms 128/alpha 256. Fix the documentation; the reader is 64/128.
- **`recursive_adapter`** (`ac_model.py:150`, 52 M params, d→4d→d) is created, put in the optimizer and never used by
  reconstruction items; with fp32 master + Adam that is ~0.8 GB of GPU memory. Skip registering parameters that never
  receive a gradient (or build the adapter lazily).
- **Summary position codes** (`ac_model.py:399-400`): summary *i* gets `sinusoidal_positions(num_rows)[i]`, i.e. the
  same code as content *token i*, not the centre of chunk *i*. The side decoder's RoPE supplies the real position, so
  this is only confusing; if kept, use the chunk centre.
- **Per-token normalisation of the content** (`ac_model.py:378, 401`): the side LM receives every token at the same
  RMS, a distribution shift from its pretraining that the side LoRA must undo. With the mixer gone (F4) feed the raw
  embeddings (multiply back by `embedding_scale`).
- **Checkpoint thresholds** (`ac_model.py:415-417` vs `ac_model_training.py:1119-1121`): the side compares one row's
  length with `ckpt_min_tokens`, the reader compares the padded *batch* tokens. Harmless today (side sequences are
  < 1,700) but the same knob means two things; document or unify.
- **Penalty sampling and clipping** (`ac_model_training.py:1029, :1099`): with p = 0.25 the sampled examples carry a 4× term, and the
  joint clip then rescales the whole update by a norm that depends on which examples were drawn, so the estimator is
  unbiased only before clipping. `penalty_count_exact` (landed 12:24) removes the count variance; per-group clipping
  removes the cross-talk. With both in, p = 0.5 is worth one midi arm (P1F at p = 1 never transitioned, but that was
  a pre-skip run in the onset-variance era).
- **`held.sort` (`run_recon.py:287`)** puts short items first so the 8 completion samples fit 320 tokens — the
  fidelity number is therefore measured on the short tail only. Label it as such in the report.
- **Autotuner** (`core.py:206-211`): with `gradient_checkpointing_min_tokens=None` the threshold is 0 = always
  recompute; the bench always passes a value, so fine. The Hopper `unused_kernels` fix is correct (visited ⊆ inventory,
  catalog validated against the platform's kernels). No further findings.

---

## 2. Explanations for the results, grounded in the code

**Why a frozen reader wins at 1/4 (and why RL1 < A16 < RL5).** The encoder receives gradient only *through the
reader's attention to the row positions*. At init (pre-skip) the rows are RMS-normalised outputs of a random FFN:
content-independent noise in the reader's input. The reader's LoRA has two cheap descent directions that need no
rows: learn the passage prior (format of tool outputs: the floor drops 1.921 → 1.873 in A4 exactly as in A16) and
attenuate the noisy row positions (a LoRA on q/k/v/o of the 8 full-attention layers can key on the rows' shared
signature — same norm, same subspace — and down-weight them). Attenuation kills the encoder's gradient, so the channel
never forms: A4's training loss flattened at 1.38 with a gradient norm of ~1.6–2 for 5,472 updates while its floor
drifted like A16's — the reader learned everything except the rows. At 1/4 there are four times as many uninformative
rows, so the attenuation direction is four times as rewarding early, and A4's failures concentrate on long passages
(13 of 200 items with no signal, median 481 tokens ≈ 120 rows). A frozen reader cannot attenuate, so the encoder's
gradient stays alive until the rows become useful; R0F4 transitioned at epoch 0.7. The reader-LR ordering RL1 (1e-5)
< A16 (2e-5) < RL5 (5e-5) in final NLL is the same effect: the faster the reader moves relative to the encoder, the
more likely it locks in before the channel exists. The gold-inclusive penalty (F7) opposes the prior-learning
direction specifically, and the skip init (below) makes the rows informative from step 0, so A4K's early curve
(loss 0.96 at 303 updates) is the expected outcome under this mechanism. F1 (staging: reader frozen for the first
half) did not help at 1/16 because the unfreezing came before the encoder's own plateau ended — the freeze window must
outlast the plateau, which the skip init now removes anyway.

**Why the transition is sharp and its onset seed-dependent.** The pre-transition state is a saddle: the encoder must
emit content-dependent rows *and* the reader must attend to and decode them; each is useless without the other, so the
gradient in the channel-forming direction is second-order small. All pre-skip arms sit at the same flat gradient norm
(~1.6) until escape; escape is driven by whatever small correlations the random init and the data order happen to
create, so the time to escape is a heavy-tailed random variable (R0F ~3k updates; R0F3, same recipe and data order,
never in 8.2k — GPU non-determinism alone is enough to change the outcome). Two slow scalars add a floor to that
time regardless of seed: the row scale climbing (F5) and the mixer's residual scale collapsing (F4), both at Adam's
~1e-4 per step. The identity-skip init removes the saddle: with `tie_word_embeddings=True` the side model's last
hidden state (post final norm) lives in the space where `logits = E · h`, i.e. it is aligned with the input embedding
of the token the side model predicts next; RMS-normalising it and scaling it to the embedding RMS gives the reader a
vector its frozen layer-0/1 circuits already parse as "a token-like thing whose identity depends on the chunk". The
gradient to the encoder is first-order from step 0, so K16 has no plateau (gradient norm 2.5 → 5.9 → decaying,
loss monotone), and replicate variance should collapse to ordinary noise (K16r1/K16r2 will confirm).

**Why wiki reads worse than novel.** It is mostly a floor offset, not a channel failure. The base no-context floor
is 2.34 on wiki vs 1.92 on novel (1.95 on the clean novel subset): prose from NQ has more entropy per token than
file views and logs, and the wiki passages are half as long (142 vs 291 tokens; less in-passage context for the
prior). At 1/16 each row must carry 16 tokens whichever the domain, so the same channel capacity leaves more residual
entropy on wiki. The gap after training shrinks as the channel improves — A16 0.47, G1b 0.38, R0F 0.32, K16 at
0.875 epoch 0.16 — so in *nats removed* wiki gains more than novel (K16: 1.91 vs 1.61), which is the opposite of what
memorised text being cued would look like (that would make wiki *easier* than novel). The training domain is 100 %
tool outputs; a small prose share in the training pool would close most of what remains.

**Why gold targets help (if they do).** See F7: the finer partition tightens the KL bound exactly at the positions
where the reader's no-context shortcut acts, so the reader's LoRA gradient loses its rows-independent component and
keeps the rows-dependent one. The floor did not move because the remaining drift (~0.04) comes from mass moved among
the base's top-32, which both bounds see equally. The effect size is not yet established (F7).

---

## 3. FUNDAMENTAL: would change the latent-context concept

These alter what a row is or how the reader consumes it. None is needed for tonight; they are recorded because the
observations above point at them.

**C1. Layer-0 injection fights the residual stream; inject rows deeper as well.** The 15× loudness the frozen reader
demanded, the attention saturation behind the gradient spikes, and the fact that only the 8 full-attention layers (via
LoRA on q/k/v/o) can do precise retrieval from the rows all follow from rows being *input embeddings*. The runtime
investigation of `layer_inputs`/KV substitution (`IB/ARTIFACTS/VLLM_ACTIVATION_INPUTS`) is the serving-side half of
this; the training-side half is to let the encoder emit, per row, an additional vector added to the residual stream
at one middle layer (e.g. the first full-attention layer). It changes the row contract (two vectors per row, or a
2d-wide row) and the vLLM path.

**C2. Free the row magnitude.** The canon's invariant "unit-RMS rows × one learned scalar" removes a degree of freedom
the reader evidently uses (loudness = gain). A per-row magnitude bounded by a cap (`scale_max · sigmoid`) would let
the encoder express salience per row and stop the global scalar from being the bottleneck. It changes the row
normalisation contract and the cache format's assumptions.

**C3. Rows in the (tied) embedding space by construction.** The skip init works because rows start in the logit
space of the tied head. The strong version constrains rows to that space permanently: row = a mixture of token
embeddings (`Eᵀ softmax(z/τ)`, or a bounded number of tokens per row). Interpretable, readable by any reader trained
on tokens, but the capacity per row drops from "a 2,560-dimensional vector" to "a soft token", which at 1/16 is
unlikely to be enough. Not recommended at these ratios; noted because it is the limiting case of what is now working.

**C4. Chunk-local rows versus global rows.** Today row *i* is the causal side decoder's state at summary position
*i* after seeing the whole passage: rows are global, and the reader has to find the row that covers output position
*t*. A chunk-aligned auxiliary loss (reconstruct chunk *i* from row *i* alone, or a small locality penalty) would make
rows locally faithful, which is what compaction and retrieval downstream want (a row that can be dropped or moved).
It changes what a row represents.

**C5. Reconstruction as the only objective.** Verbatim copy at 1/16 trains rows to be lossless codes and a reader
that decodes them; the intended use (QA, compaction, subagent returns) needs rows the reader can *reason over*, which
the RAG round showed the earlier recipe failing at. Before scaling the reconstruction recipe further, evaluate the
K16 checkpoint on the RAG QA panel with the reader frozen — if the reconstruction pretraining transfers, the concept
holds; if not, a mixed objective (reconstruction + QA distillation) is a concept change worth making early.

**C6. Content-derived summary queries.** The V queries are bin means of chunk embeddings plus a marker; the encoder
has to recover "which chunk" from a bag-of-words mean plus a position code. Architecture 1's learned per-ordinal
markers were retired by decision; a hybrid (bin mean + a learned ordinal embedding) is small but touches the canon
decision on markers. Low priority: the side decoder's RoPE already disambiguates ordinals.

---

## 4. Experiment plan for the remaining budget

Timing basis: it is ~12:30 UTC; node 1 (4 × RTX PRO 6000) until ~19:00 (late finishes allowed). Anchor size
(21,888 × 2) at 3/forward ≈ 4.7 h wall including building and panels; half size ≈ 2.4 h; midi (2,048 × 2) ≈ 25 min
on an RTX slot. Running: X40K (GPU 0, ends ~18:30), K16 (GPU 1, ends ~14:45), A4K (GPU 2, ends ~16:30), R4X
(GPU 3, ends ~15:30). ORCD: 2 normal H200 slots (6 h, resume-chained) + 4 preemptable RTX/H200 slots (start within
a minute so far).

**Now, CPU (do before anything else; ~45 min total).**
1. F1: re-score every `result.json` on the clean 151-item subset (and the 100k-clean subset for X40K/R4X); put
   "novel (all / clean)" and the token-weighted NLL (F6 metric) in the results table; note the contamination rates.
   Add the shingle check to `prepare` and write a deduplicated `material_40k_clean.json` for tomorrow's runs.
2. F3: fix the resume LR override + test. F4/F5/F6 arm keys: `mixer_layers`, `row_scale_init` (multiplier on the
   embedding RMS), `loss_weighting`. Snapshot, run `test_batched.py`, push to node 1 and ORCD.
3. F8: replace the S-size MD/MC/MP/MA definitions with midi-size versions (2,048 × 2) and submit them to ORCD
   preemptable slots as they free up (~25 min each; they can share one job sequentially).

**Node 1 (the decisions that need anchor-size, paired-panel evidence).**
4. GPU 1 at ~14:45 → **K16D**: K16 + `lr_schedule=cosine` (final 0.1) + `warmup=100` + `clip_per_group` (F2). Same
   seed and data order as K16, so every panel is paired. Ends ~19:30. This decides the headline schedule.
5. GPU 3 at ~15:30 → **K16G0**: K16 without gold targets (F7). Ends ~20:15 — if that is too late, run it at half
   size (10,944 × 2, ends ~18:00) and compare with K16's half-epoch panels at matched updates (the K recipe's
   curves are smooth enough for that now).
6. GPU 2 at ~16:30 → a sequence of midi arms on the K recipe, ~25 min each, control first: **MK** (control),
   **MK-nomixer** (`mixer_layers=0`), **MK-scale4** (`row_scale_init=4`), **MK-token** (`loss_weighting=token`),
   **MK-p50** (`penalty_fraction=0.5`, exact count). Five arms ≈ 2.2 h → done by ~18:45.
7. GPU 0: leave X40K; when it finishes (~18:30) pull its checkpoint. Do not restart it tonight — its panel is 21 %
   contaminated and its "unique data" premise is weakened (F1); the clean-material version belongs on ORCD.

**ORCD normal H200 (2 slots, 6 h, chained with resume).**
8. **X40KD-clean**: the headline recipe (gold + skip) + cosine + warm-up 100 + per-group clip on the deduplicated
   40k material, 2 epochs = 10k updates ≈ 8–9 h as two chained 6-h jobs (the resume path is verified live).
9. After A4K's result (~16:30): if it transitioned, **A4KD** (A4K + cosine) as the 1/4 headline candidate with a
   trainable reader; if not, **R4XD** (frozen reader, 1/4, cosine) — either fits one 6-h job at anchor size.

**ORCD preemptable (4 slots).**
10. Keep K16r1, K16r2 (onset variance under the skip recipe — this is what makes every other single-run comparison
    interpretable), A32K and A8K (ratio curve). As slots free: the midi arms of step 3/6 that did not fit on node 1,
    then anchor-size **K16-nomixer** and **K16-gate-scale-only** (row scale in its own group at 1e-3), which SK vs
    SKn (1.423 vs 1.625) makes worth one full run.

**Reporting.** Quote every headline as "novel clean / all, token-weighted / item-mean", with the base floor on the
same subset, and never a single-run onset comparison without a replicate. The K16r1/K16r2 spread is the error bar.

**Before teardown.** Pull K16, K16D, A4K (and X40K if it finishes) checkpoints; record F1's contamination figures and
the clean-subset table in `AC_RECON_RESULTS.md`; promote to canon: "held panels are shingle-deduplicated against the
training pool", "resume re-applies the configured learning rates", and the midi-arm size for recipe questions.
