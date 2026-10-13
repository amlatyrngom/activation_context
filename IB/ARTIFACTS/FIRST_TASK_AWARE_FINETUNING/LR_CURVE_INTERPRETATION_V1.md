# What the observed curves say about learning rate

The terminal LR does not prevent improvement: constant5e-6/2e-6 improved held400 NLL.568831→.538511 within the first2,000 fresh questions after32k. It ended10,000 fresh questions at.532686. Most of that net gain arrived early; later progress was smaller and noisy.

Rewarming both rates to4× and decaying back to the same floor initially performed worse: +2k .572481, +4k .568430, +8k .533558, final.524844. It only beat the constant arm's recorded NLL at the last panel. Its final supported answer count was62/87 versus71/87. This is not convincing overall support for a4× restart.

The subsequent constant-floor42k→50k stage reached.518543 after1,600 fresh questions, then fluctuated around.518–.527; final.527131. The interim minimum is descriptive, not a selected checkpoint or separately reviewed answer-quality result.

The original32k cosine run continued improving during decay: held400.662222 at16k,.578553 at25.6k,.558061 at28.8k,.568831 at32k. The long-context8k stage also improved late as its LR fell (.635593 at4k→.596817 at8k). Neither establishes that a10× decay is intrinsically wrong.

Untested distinctions: keeping a moderate LR longer during initial training; using a modestly higher constant continuation LR; seeing more unique questions before final decay. Rewarming an already-decayed checkpoint does not reproduce delaying its original cooldown. Existing fresh-vs-repeat comparisons also change schedules, so they do not settle the unique-data requirement.

Proposed next discriminating comparison, **not submitted**: same start, examples/order and optimizer policy; current cosine versus longer moderate stable phase with a final decay, at equal exposure. A smaller continuation alternative is1e-5/4e-6 versus5e-6/2e-6 from the same paired checkpoint and inherited Adam, same fresh question pool. Judge held400 NLL alongside blinded/fixed answer review; do not infer improvement from train loss alone. A larger100k data budget remains an experiment, not an established convergence requirement.

Exact source panel timelines and schedules: LR_CURVE_INTERPRETATION_V1.json. No production training settings or jobs changed by this analysis.
