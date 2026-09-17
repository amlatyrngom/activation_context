# Null review — round 8 (finding 22): are the three nulls real or artefacts?

Read-only review, 09-17. Sources: `ROUND8_ENCODER_PLAN.md`, `DESIGN_REVIEW_8.md`, `POC_IMPLEMENTATION_8.md`, `POC_REVIEW_10.md`, `poc_changes_8.patch`;
code `activation/ac_model/ac_model.py` (`_encode_prepared_batch` 741–950, `_content_surprisal` 962–1006), `ac_model_utils.py` (`mass_bins` 436, `AdaptiveSummaryPooling` 454–512,
`RefineAdapter` 534), `ac_model_training.py` (groups 615–640, `_evaluate` 1121–1190, `_example_loss` 1237–1243, `_batch_loss` 1537–1546, `_token_weights` 1308, `_rare_token_stats` 1318);
`arms.json`; run data `IB/TMP/SYNC/AC_RECON/{A32P2n,A32KVTAP2n,A32KVTAP2Rn,A32SPn,A32SWn,A32X,A32KVTAn}/{result,report_data,training_stats}.json`; the K16 epoch-2 AC checkpoint
(`CHECKPOINTS/K16_epoch2/ac/ac_modules.pt`, the `…n` arms' init). `test_encoder_passes.py`: 9 passed (64 s). No training run.

Two post-hoc computations were done here (scripts inline in the session, not committed): (i) the per-update `refine_rms` / `pass_cosine` traces from `report_data.json`;
(ii) a rare/common decomposition of every arm's held-out per-token log-probs (`training_stats.json` → `epochs[-1].reporting.per_item.target_logp`), with the rare set defined
per item as the top-20 % of the **no-context reader's** surprisal (`epochs[-1].validation.per_item`, the reader held to the base by the penalty). The proxy reproduces the
trainer's panel to 0.003–0.005 (A32SWn 0.642 vs reported 0.639; A32SPn 0.797 vs 0.793), so it is a valid control readout the round lacked.

| arm | held novel | proxy-rare (20 %) | common (80 %) |
|---|---|---|---|
| A32X (control, K16) | 0.3219 | 0.807 | 0.199 |
| A32SWn (rare_weight 1) | 0.3472 | **0.642** | **0.273** |
| A32SPn (pool_surprisal 1) | 0.3213 | 0.797 | 0.201 |
| A32P2n (2 passes) | 0.3221 | 0.797 | 0.202 |
| A32KVTAn (control) | 0.2995 | 0.750 | 0.186 |
| A32KVTAP2n / A32KVTAP2Rn | 0.3040 / 0.3041 | 0.768 / 0.765 | 0.187 / 0.188 |

Scales that matter, from the K16 checkpoint (d_side 2560): `summary_marker` RMS 0.021, `position_scale` 0.020 (positions ≈ 0.014 RMS), `input_scale` 0.0147,
`summary_pooling.residual_scale` 0.082, `target_head.scale` 0.069. A summary input is `unit_rows(pooled) + marker + positions`, i.e. ≈ 98 % the unit-RMS pooled seed.

## (a) Iterative encoder — verdict: CONFOUNDED (the null is real for the form as built; it does not test the idea)

Ranked artefact candidates:
1. **LIKELY — warm-start rejection of the perturbation, not inertness.** The ledger reads A32KVTAP2Rn as "the adapter moved (rms 0.061), the rows did not". The trace says
   otherwise: at refine_lr 1e-3 `refine_rms` rose 0.004 → 0.36 → **0.56 at update 40** with `pass_cosine` **0.95** (the rows were ~30 % different), then collapsed to 0.14 by
   update 50 and sat at 0.05–0.06 for the remaining 950 updates; held-out at 0.125 was 0.864 vs 0.849 (A32KVTAP2n) / 0.856 (A32KVTAn). A32P2n shows the same shape at 1e-4
   (peak 0.039 at update 65 → 0.016). This is finding 15/18's "grow and be driven back": the terminal loss actively pushed the refinement back to the scale where it does
   nothing, because the warm side LoRA is tuned to exactly the pass-1 summary inputs and any change of that channel costs loss before it can pay. A 1,024-update cosine pass
   from a warm init cannot leave that basin; the equilibrium 0.06 is where cost and benefit balance, and `passes_off` = full (0.3038 vs 0.3041) says the benefit at that
   scale is nil. A larger LR or a non-zero init is therefore **not** the fair test — the excursion to 0.56 was already tried by the optimizer and rejected.
2. **LIKELY (design) — no new information reaches pass 2.** Pass 2 re-reads the same content block under the same causal mask (`sequences` reuse `content_blocks`,
   `attention_mask`/`max_seq` unchanged, `ac_model.py:841–852`); content states are pass-invariant, so the only thing pass 2 adds is depth at the V row positions, and
   row j sees the final states of rows ≤ j only (design review §1). "Each row reads what the others captured" is not what was built; the rows of pass 1 are already a
   function of the whole passage, so a re-read cannot add bits, only reallocate them. The null is consistent with the architecture, not only with the optimizer.
3. **EXCLUDED — adapter output normalised away / tiny vs marker.** The refinement is added to `summary_blocks` (unit seed + marker + positions, RMS ≈ 1.0) *before*
   `unit_rows` (`:851–852`): at 0.061 it is 3× the marker and 4× the positions, and the decoder passes ~70 % of it through (angle in 0.061 → row angle 0.042 = cos 0.9991;
   at the peak 0.56 → cos 0.95). `input_scale` multiplies content and summaries alike. `pass_cosine` and `refine_rms` are mutually consistent at every point of the trace.
4. **EXCLUDED — optimizer / gradient plumbing.** Adapter parameters are trainable by `requires_grad` (console: "Fresh (zero-initialised) round-8 modules … refine_adapter.*");
   at 1e-4 they sit in group 0, at 1e-3 in their own last group (`ac_model_training.py:633–637`, `test_refine_adapter_own_learning_rate_group`); `weight_decay` 0. Pass-1
   `hidden` is not detached (only the diagnostic copy `first_rows` is, `:846`), so gradient reaches pass 1 through the adapter; at init `down` gets none (LoRA bootstrap),
   which only delays the first steps — the 1e-3 trace shows the adapter was open by update 15.
5. **EXCLUDED — `passes_off` measured the same rows.** `_evaluate` sets TRAINING mode, and the row cache is consulted in ROLLOUT mode only (`ac_model.py:668`), so the
   panel re-encodes with one pass; captures are opened on the last pass only (`:855–868`, `test_kv_transfer_capture_is_the_last_pass`). All arms ran 1,024 updates (no truncation).
6. **PLAUSIBLE, minor — shared side LoRA for both roles.** The same adapter must read the passage (pass 1) and the passage plus refined rows (pass 2), so it cannot specialise
   for the second read without disturbing the first; this is the mechanism behind candidate 1 and is why the excursion was expensive.

Settling diagnostic (one arm): **A32KVTAP2 with a separate side LoRA for pass 2** (initialised as a copy of the warm adapter, refine_lr 1e-3), read by held − passes_off
and the `refine_rms` peak/plateau. If the second read can specialise without touching the first and the plateau still returns to ≈ 0.05 with passes_off = full, the
iterative encoder is closed at 1/32; if refine_rms holds at its excursion scale and passes_off separates, candidate 1 was the whole story. (A from-scratch SK32-style
variant is the second-best test; a longer warm pass at the same design would merely re-measure the basin.)

## (b) Surprisal-weighted pooling — verdict: CONFOUNDED (the lever as built cannot move the row budget)

1. **LIKELY (design) — the bins only seed the rows; the side decoder reads the whole passage anyway.** `AdaptiveSummaryPooling` restricts the *pooling* attention to the bin
   (`key_padding_mask`, `:499–508`), but its output is `means + 0.082·update + 0.082·ffn` — 90 %+ the bin's mean embedding — and that pooled vector is only the input
   at the row's position; the row itself is the side decoder's final state after causal attention over every content token (`_encode_prepared_batch`). The design review's
   expectation ("moves bits from cheap tokens to expensive ones") assumed row = pooled bin; in this architecture a bin is a soft address, and rows' capacity is allocated by
   the decoder's attention, which the bins do not constrain. The re-allocation observed (widest bin 39.7 vs 31.5 tokens, i.e. +26 %, bounded by the `1 + s/mean` prior)
   shifts each seed by the mean of a few overlapping tokens — a small change of a weak lever. Proxy-rare 0.797 vs 0.807 on the control: the sign is right and the size is noise.
2. **LIKELY — the surprisal bias is gated to insignificance.** `surprisal_bias` ([8 heads], zero-init) sits in group 0 at 1e-4 with cosine decay: |b| ≲ 0.05 after 1,024
   updates (Adam moves a weight by ≈ its rate per step at best); the logit bias `b · s/mean(s)` reaches ≲ 0.25 on the most surprising tokens, and whatever it does to the
   pooling attention is multiplied by `residual_scale` 0.082 before entering the seed. Its value is not logged anywhere (no trace, no checkpoint pulled), so "trained" is
   unverified on the GPU runs; the CPU test only shows a nonzero gradient. It is in the optimizer (numel 8 > 1, not a gate) — the plumbing is fine, the reach is not.
3. **PLAUSIBLE — the re-allocation is partly positional.** The surprisal is the base's no-context next-token surprisal over the passage, so the passage's first tokens are
   the most surprising: 22.8 % of the proxy-rare tokens lie in the first 10 % of positions, and the bin trace shows it (`pool_first` 3.78 > mean 3.13 > `pool_last` 2.81).
   γ = 1 therefore narrows the leading bins and widens the trailing ones as much as it targets identifiers — the lever spent part of its 26 % on the passage start.
4. **EXCLUDED — bins/surprisal/mask faults.** `mass_bins` recovers the token-count bins at uniform mass, keeps every bin nonempty and ordered (tested); the surprisal is the
   base's (`lora_name=None`), computed in every mode (`:769`, no mode gate) and cached by ids, so training and panels use the same bins; the merged-mask path was traced by
   POC review 10 and ran on R8S; A32SPn's held-out curve tracks A32X's at every checkpoint (0.533/0.555 … 0.404/0.402), which a broken mask would not do.

Settling diagnostic (no training): **evaluate the trained A32X checkpoint with permuted bins** — the held-out panel at γ = 0 with the summary bins reversed (row j seeded from
bin V−1−j) or shuffled. If the loss barely moves (< 0.01), the bins are not where the budget is decided and (b) as built could never have moved the number; then the
lever needs a form where bins constrain the read (a per-row attention mask over its bin in the side decoder, or rows that *are* the pooled bins). If reversing bins
costs > 0.05, the seeds matter and a γ sweep (3, 10) plus the bias in the gate group at 1e-3 becomes worth one arm.

## (c) Surprisal-weighted terminal loss — verdict: REAL (a per-token reweighting doing what it says)

1. **EXCLUDED — weights leaking into the metric or the panels.** Per-example path: `loss = (ce · w).mean()`, `_loss_metric = ce.mean()` (`:1240–1242`); batched path:
   `metric = float(objective.detach())` binds the unweighted `objective` before the rebinding (`:1543–1546`); `_evaluate` runs under `no_grad` with `with_drift=False`
   / `with_penalty=False`, so `weights` is `None` on every panel (`:1237`, `:1538`); the recorded training loss is the metric (`:755`). The held-out 0.3472 is unweighted.
2. **EXCLUDED — normalisation and alignment.** `w = 1 + s/mean(s)` then `w / w.mean()` per example (`:1315–1316`, mean exactly 1, tested); `s` comes from the no-context
   reference built on the same `assistant_token_ids` in the same order (`build_example` `:459–466`), a length mismatch would raise at the multiply; the penalty term is a
   KL and cannot be weighted.
3. **EXCLUDED — wrong control for the rare-token number.** The ledger's "0.639 vs 0.793" compares against A32SPn (the controls had no panel; the post-hoc script was never
   written). The true control from the proxy is A32X 0.807, so the rare-token gain is −0.165, slightly larger than quoted, and the common-token cost is +0.074
   (0.199 → 0.273). Per position decile A32SWn wins only the first decile (0.300 vs 0.315) and loses all nine others — the signature of a reweighting, not of a fault.
4. **PLAUSIBLE — the cost side is inflated by global clipping.** Grad norm runs 4.4–7.3 on A32SWn vs 3.0–3.5 on A32X under a joint clip of 1.0 that binds every step
   (`clip_per_group false`); the weighted loss's larger, rare-dominated gradient takes the fixed step budget, so common tokens lose more than the 2:1 weight ratio implies
   and the seen-training loss is worse too (0.213 vs 0.175). This affects the *magnitude* of the trade-off in a 1k-update pass, not its existence.
5. **PLAUSIBLE — "rare" is front-loaded** (22.8 % of the set in the first decile), so part of the gain is on passage openings rather than identifiers; finding 11's
   digit/identifier split was not measured.

Settling diagnostic: the queued **A32SW3n** (β 0.3) read with the rare/common decomposition above (and the digit / identifier-piece split of finding 11) on the real
control; if the rare-token gain scales sub-linearly with β while the common cost scales linearly, the reweighting is a knob worth carrying at small β, otherwise it is closed.
The decomposition script should be written once (validation no-context log-probs are already in every `training_stats.json`) so no future arm quotes a proxy control.

## Bottom line

(a) CONFOUNDED: the trace shows the refinement grew to 0.56 (rows 30 % different) and was driven back by the loss — a warm-basin rejection, and the re-read carries no
information pass 1 lacked; (b) CONFOUNDED: the bins seed rows that the decoder then fills from the whole passage, and the bias is gated by 0.08 × 1e-4 — the budget was never
moved; (c) REAL: rare 0.807 → 0.642, common 0.199 → 0.273 on the true control, metric unweighted, with the clip inflating the cost. Nothing in the implementation is wrong;
two of the three experiments could not have moved the number even if the ideas were right.
