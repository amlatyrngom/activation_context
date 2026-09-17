# Delta scale parametrisation: why the scales collapse, and what to change

Scope: the deep-intervention delta path (`DeltaHead` / `KVDeltaHead`). Read-only investigation, no product
code changed; line numbers are from the current working tree.

## 1. What the code actually does

**The head's gradient is exactly proportional to the scale — confirmed.**
`DeltaHead.forward` (`activation/ac_model/ac_model_utils.py:534-535`) is
`delta = unit_rows(down(silu(up(z)))) * scale`, `scale = rms * relative` (`:529-532`); `KVDeltaHead` (`:565-567`)
is the same twice. The delta is the *only* path by which `up`/`down` reach the loss
(`activation/ac_model/ac_model.py:658-660`), so `∂L/∂W_head = scale · (…)` identically. `relative` sits outside
that factor — `∂L/∂relative = rms · ⟨g, unit_row⟩`, whose magnitude does **not** shrink as `relative` shrinks.
The scale keeps a full-strength gradient all the way to zero while everything behind it fades.

**`unit_rows` is the second half of the mechanism, and it is the deeper cause.**
`unit_rows` (`ac_model_utils.py:430-433`) is an RMS normalisation, so its Jacobian annihilates the radial
component exactly: the head weights receive *zero* gradient in the direction that would change the delta's
magnitude. Magnitude is controllable only by the single scalar `relative`. And because `down` is left at
`nn.Linear`'s default init (`:524-525`, unlike `PassageAttention.out` at `:590-591` and the `RowHead` skip FFN
at `:478`, both deliberately zero-initialised), a fresh head emits a **full-magnitude random direction** from
update 0. That is the observed +0.0008 nats. The cheapest descent direction for that damage is one scalar per
layer going to zero; rotating a `d_side → r → d_target` random map into something useful takes thousands of
steps. With Adam the scalar moves ≈ its LR per step regardless of gradient size (the trainer's own comment,
`ac_model_training.py:518-520`): 0.05 / 1e-3 ≈ **50 updates** to zero, matching the ~100 observed. The race was
structurally lost, not badly tuned.

**bf16 in the hook makes the dead state permanent — this is a real third contributor.**
The reader runs bf16 (`activation/harness/model_config.py:57,72`) and the hook adds the fp32 delta as
`rows[keep].to(dtype=hidden.dtype)` into a bf16 `index_add_` (`activation/harness/hf_utils.py:212`; K/V at
`:168`). bf16 has 8 significant bits: ulp = 2⁻⁷ ≈ 0.78 % of the residual element, half-ulp ≈ 0.39 %. With the
true row-position residual RMS of 0.40, the delta at init is 0.05 × 0.131 = **0.0065**, i.e. 1.6 % of the
residual — about **2 ulps**, so ~24 % of every element is already rounding noise *at init*. Below ≈ 0.2 % of
the residual (`relative` ≲ 0.006 in calibration units, reached after ~45 updates) the addition rounds away
entirely and the forward is bit-identical to deltas-off — exactly the reported `novel == deltas off` to three
decimals. The backward is *not* quantised: the cast and `index_add_` pass the gradient straight through, so
`relative` keeps receiving a gradient computed for a perturbation the forward never applied. That noise
gradient is what Adam turns into the ±LR walk around zero. Nothing recovers: the loss surface in the heads is
genuinely flat, so Adam's scale-invariance (which would otherwise hold head steps at ≈ LR despite tiny
gradients) buys nothing, and at these magnitudes `adam_eps=1e-8` (`ac_model_training.py:74`) also floors them.

**Nothing else contributes.** No detach on the delta path (`intervention_detach_source` defaults False,
`ac_model.py:125`); the cosine and `row_rms` diagnostics detach correctly (`ac_model.py:663`,
`hf_utils.py:206`). The scales are excluded from `ac_parameters` before clipping
(`ac_model_training.py:521-522`, clipped at `:656-661`), and a global clip is Adam-invariant anyway. One more
aggravator: `relative` has no floor and no sign constraint, so it crosses zero, flipping the sign of every head
gradient and poisoning the heads' Adam momentum.

**The calibration denominator is wrong by 3×, and it matters here.**
`calibrate_interventions` (`ac_model.py:396-441`) measures the median residual RMS on `_calibration_sequence`
histories (`ac_model_training.py:1223-1232`) — the AC parts replaced by plain text, i.e. **no row positions
exist** — via `capture_layer_input_rms` (`hf_utils.py:246-252`). The rows are written at the target embedding
RMS × `row_scale_init` (`ac_model.py:110,315`) and are louder than ordinary tokens, hence 0.40 vs 0.131 at
layer 7. Every number in the `interventions` widget and in `intervention_scale_fraction` is therefore overstated 3×:
the "5 % of the residual" init is really 1.6 %, precisely the 2-ulp band above.

## 2. Ranked fixes (training-friendliness first; cost noted)

**1. Zero-init `down`, drop `unit_rows`, let the head own its magnitude (LoRA-style).**
`ac_model_utils.py:535` → `return self.down(F.silu(self.up(rows.float()))) * self.scale`; `:567` likewise for
the two halves; add `nn.init.zeros_(self.down.weight); nn.init.zeros_(self.down.bias)` after `:525` and `:551`.
Set `learning_rate_delta_scales: float = 0.0` (`ac_model_training.py:90`) so `rms × relative` is a fixed unit
anchor and the head alone sets magnitude. The delta is then exactly 0 at init: no +0.0008 nats, no
consistent-sign gradient on anything, and the gradient at delta = 0 is the *exact* linearisation (bf16 error
is zero there), so the path grows out of the dead zone monotonically instead of dying in it. Cost: medium — two forwards, two init lines, one default, plus the doc/widget text at
`ac_model_reporter.py:144-151` and `ac_model_training.py:1242` ("scale = the applied delta RMS, rows being unit
RMS" is no longer true), and a delta-RMS series next to the cosine diagnostic (`ac_model.py:661-666`) so
magnitude stays observable. Checkpoint keys unchanged, no migration. Risk: the growth rate from zero is
unmeasured — the anchor `rms × fraction` sets it, and the fraction is capped below 1 (`ac_model.py:145-146`).

**2. Log-space scale with a floor.** Store `log_relative` and make `scale = rms * (floor + softplus(θ))`
(`ac_model_utils.py:526,529-532,552-555,557-563`). Adam then moves the scale *multiplicatively* by ≈ LR per
step (≈ 0.1 %), so 0.05 → 0.005 takes ~2,300 updates instead of 45, and the floor (≥ 1 % of the **true** row
RMS ≈ 2.5 ulps) makes a dead path impossible. Cost: low-medium — two classes, plus a checkpoint migration in
the style already present at `ac_model.py:765-784`, plus `intervention_summary` (`:385-387`) and the trace
(`ac_model_training.py:1251-1259`). Keeps the interpretable `relative` widget. Does not remove the init damage.

**3. Freeze-then-release the scale group after N updates.** Mirror the reader's staged recipe: after
`set_learning_rates` at `ac_model_training.py:662-664`, add `optimizer.param_groups[-1]["lr"] = 0.0` while
`updates_done < config.delta_scale_frozen_updates`, next to the identical reader line at `:664-666`. ~6 lines,
no checkpoint or reporting change, and the closest thing to the live mitigation that still lets the scale learn
later. But it buys the heads only N updates; on release the same descent resumes unless the delta is already
useful. `warmup_updates` (`:92`) cannot substitute: `set_learning_rates`
(`activation/agent_training/agent_training_utils.py:478-489`) scales *every* group and only slows the descent.

**4. Lower scale LR (1e-5) with a normalised gradient.** One config value (`ac_model_training.py:90`).
Cheapest and weakest: Adam already normalises the gradient, so "normalised" changes nothing, and 1e-5 merely
moves the collapse to update ~5,000 — inside a long run. Rejected except as a knob on top of 1 or 2.

**Orthogonal and cheap, pair with whichever wins: calibrate on row positions.** Run
`capture_layer_input_rms` over sequences that actually contain rows — the real `build_example` embeds rather
than `_calibration_sequence` (`ac_model_training.py:490`, `:1223-1232`) — or rescale the buffer by the measured
`row_rms` after the first panel. Without it every floor, fraction and widget reading is off by 3×.

## 3. What I would ship, and what to watch

Ship **fix 1** (zero-init `down`, no `unit_rows`, scale LR 0) with the row-position calibration folded in.
It removes the failure mode rather than slowing it: no full-magnitude random delta at init, therefore no
consistent-sign gradient, therefore nothing for Adam to walk to zero; the head regains the magnitude degree of
freedom `unit_rows` removes by construction; and zero is the one regime where the straight-through bf16
gradient is exact, so the path opens instead of dying under the ulp. It is the pattern this codebase has
already validated twice in the same file (`PassageAttention.out` `:590-591`, `RowHead` skip `:478`), needs no
checkpoint migration, and deletes a whole optimizer group. Fix 2 is the fallback if growth from zero is too
slow — the two compose.

Confirmation, on the existing widgets (raise the trace cadence at `ac_model_training.py:667` from every 25
updates to every 5 for the first few hundred — a 50-update collapse is only two points today):

- `intervention_grads` must stay **non-zero and roughly flat or rising** over the first 200–500 updates.
  Today it decays with the scale; under fix 1 it starts small (delta = 0) and must climb. A monotone decay
  toward 0 means the path is dying again.
- `interventions` becomes a flat line by design (scale LR 0), so it is now a *guard*, not a signal: if it
  moves, the scale group was not actually frozen. The magnitude signal moves to the new delta-RMS series;
  watch it cross **1 % of `intervention_row_rms`** (≈ 1.3 ulps) within ~200 updates and settle in the 2–10 %
  band. Below 0.4 % the forward is a no-op whatever the gradients say.
- `intervention_row_rms` should sit near 0.40 at layer 7 and is the denominator for every claim above.
- `intervention_cosine` should move off its init value: a delta that merely re-injects the input row
  (cosine → 1) is not new information even when the magnitude looks healthy.
- Decisive check: the `deltas_off_panel` gap (`ac_model_training.py:91`, `:708-712`). `novel` minus
  `deltas_off` must become **> 0.003 nats** and keep separating. Equality to three decimals is the signature
  of the dead path and is what all three warm-started runs reported.
