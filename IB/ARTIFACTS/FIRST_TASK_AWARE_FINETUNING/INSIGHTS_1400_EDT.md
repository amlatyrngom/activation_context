# Insights — September 22, 14:00 EDT

As of 2026-09-22 18:00 UTC / 14:00 EDT. These are development results; no new confirmation scores were inspected.

## Data amount: improvement continues, final scale remains open

The 32,000-question × 1-epoch run has completed its 60% quality panel, after 19,200 distinct questions. Held-out answer NLL is **0.626767**, down from **0.662222** at halfway and **0.781322** at 40%. This does not yet look saturated. The prior completed 16,000 × 1 run reached **0.603264**; the new run has not beaten that endpoint yet, and it uses a longer learning-rate schedule.

At matched 16,000 question presentations, the larger pool's halfway checkpoint scores **0.662222**, versus **0.695225** for the 16,000-question companion after its first epoch. The lead has changed sides across earlier panels. The descriptive paired-question interval for this difference includes zero; one seed and repeated development checks do not establish a reliable win. The next half is the direct fresh-data versus repetition comparison: 32,000 × 1 against 16,000 × 2, with the same total presentations and learning-rate schedule.

Provisional next scale: **64,000 × 1 if the final quarter and actual answers keep improving**. This is a planning option, not a required final recipe. No new synthesis was launched for it.

## What the learning NLL means

The learning curve reevaluates the same 32 training questions; it is not pre-update loss on a continually fresh stream. At the 32k run's halfway panel, 16 had been trained on once and 16 were unseen. Their NLL was **0.546040**, versus **0.624378** on the 200 held-out 27B questions, or **0.662222** on the mixed 400-question panel. Training exposure, sample difficulty and generator composition can all contribute to the gap. The gap alone does not demonstrate harmful overfitting.

The companion had seen all 32 panel questions after its first epoch and scored **0.451112** on them despite worse held-out NLL. We should compare held-out curves and actual answers when deciding data quantity.

## Separate clipping: modest NLL gain, answer check pending

The 16,000 × 1, 1/8-compression run with independent encoder/reader clipping finished at **0.629357**, versus **0.646272** under the original joint clipping. Fixed learning-panel NLL improved **0.521832 → 0.429660**. Preclip norm events above 100 fell **174 → 60**, and the maximum fell **6,602 → 540**. The paired-question interval for the final held-out difference includes zero. The fixed 100-answer readout is queued; this change has not been adopted in the main runs.

## Piecewise encoding: memory repair is being validated

The fresh-P2, 4,000 × 1 piecewise run failed after 120 questions with no resumable checkpoint. Its matched merged control is held. A 20-minute GPU probe is queued on ORCD to test activation checkpointing on every batch while retaining the original masks, source, row budget and inputs. It reconstructs the failed data batch and tests both layouts. CPU evidence confirms actual recomputation with unchanged loss and all 202 tiny-model gradients; real-GPU equivalence and memory sufficiency remain pending.

The 27B bank remains paused at 53,000 published questions. No EC2 experiment is active. At most four autonomous MIT GPU slots are committed; at least two user slots remain reserved.

At this boundary, training has processed roughly **32k × 0.69 epochs** and **16k × 1.12 epochs**; the quality panels above are the latest complete evaluations. Both auxiliary GPU jobs are waiting on scheduler priority, with no start time promised.

Update at 14:05 EDT: the main run completed its 70% panel after **22,400 distinct questions**, with held-out answer NLL **0.605952** and fixed training-panel NLL **0.432571**. It is close to the previous completed 16k × 1 endpoint (0.603264), with 30% of the new run still to go. Improvements continue but have slowed over the last two intervals.

Update at 14:17 EDT: at matched **19,200 question presentations**, fresh data scores **0.626767** versus **0.696787** for **16k × 1.2 epochs**. The exploratory paired-question difference is −0.0700 nats/token (descriptive 95% interval −0.1129 to −0.0344; one seed, repeated development checks). Only 214/400 individual question losses favor the fresh-data run, so the mean advantage is uneven. The companion’s fixed training panel improves 0.451112 → 0.439596 while held-out NLL is essentially flat. This supports watching the value of additional unique data; it is not yet a final convergence or overfitting diagnosis.

Learning-gap check: before QA training, the fixed training panel scored **1.935765** versus **2.002939** on the 27B held-out cohort (gap **0.0672**). Halfway, that same-generator gap was **0.0783**. The small halfway gap is weak evidence for substantial overfitting; some panel difficulty difference clearly predates this run. This does not prove that the entire later gap is sampling noise.

At the 80% panel (25,600 fresh questions, about14:30 EDT), held NLL reached **0.57855**, below the previous completed16k ×1 endpoint0.60326. Fixed training-panel NLL is0.42555. Held improvement continues; the old16k run had a shorter LR schedule, so this is useful scaling evidence rather than a controlled isolate of data count. The matched16k ×2 run was preempted again and is queued for an exact optimizer resume.

At the 90% panel (28,800 fresh questions, about15:00 EDT), held NLL reached **0.55806**, another0.02049 improvement from25,600 questions. The fixed training panel is0.42682, essentially flat over that interval. There is still no clear held-out plateau.

The clipping follow-up is complete on matched L40S hardware: control52/87 supported answers correct versus51/87 for separate clipping. Held400 NLL modestly favors separate clipping, but the fixed100 NLL reverses direction and actual answers show no clear win. Keep the original clipping recipe; GROUP_CLIPPING_COHORT100.tressoir.html contains all examples.

32k × 1 finished at15:30 EDT: **final heldNLL0.56883**, fixed training-panel0.39718; no dropped questions,1.50M supervised completion tokens and65.84M side-input tokens. Earlier16k ×1 finished at0.60326 under its shorter schedule. The final10% rose0.01077 from the90% panel;208/400 question losses still improved, with a descriptive paired-question interval for mean change[−0.0053,+0.0271]. This tail is noisy: no strong overfitting or saturation conclusion, and no switch from the preselected final checkpoint to the best development panel.

Atmatched22,400 presentations, fresh32k ×0.7 is0.60595 versusrepeated16k ×1.4 at0.63964 (delta−0.03369; descriptive paired-question interval[−0.0615,−0.0064]). The repeated run is still improving; complete its full second epoch before selecting a canonical data recipe.
