# What the gradient records establish about continuation LR

**The terminal LR still learns. Clipping is pervasive, but these records do not explain the rewarm regression causally.** All retained batches in all four runs exceeded the joint clipping threshold1. Rewarm did not introduce a new clipping regime or a larger typical raw gradient.

| Run, unique questions ×epochs | Batches above threshold1 | Median joint gradient norm before clipping |
|---|---:|---:|
| Original32,000 ×1, cosine to floor |4,000/4,000 (100%)|6.86|
| Same32k parent +10,000 fresh ×1, constant floor |1,250/1,250 (100%)|7.27|
| Same32k parent +same10,000 ×1, 4×peak rewarm |1,250/1,250 (100%)|7.09|
| Constant42k parent +8,000 fresh ×1, constant floor |1,000/1,000 (100%)|7.65|

These fractions are **computed from recorded pre-clip norms and the configured threshold**, not a separately logged clipping counter. Norms include gradients from the main objective and the sampled no-context penalty. Floor means AC5e-6/reader2e-6; rewarm uses400 questions of warmup to2e-5/8e-6, then cosine back to floor.

The matched10k arms use the same initial pair, question IDs/order and fixed32 learning questions. Each entry below is **held400 / fixed learning32 answer NLL**, in nats per answer token, averaged equally over questions.

| Fresh questions seen ×1 | Constant floor | 4×peak rewarm |
|---|---:|---:|
|0|.5688 / .5970|.5688 / .5970|
|1,000|.5520 / .6000|.5598 / .6217|
|2,000|.5385 / .5892|.5725 / .5962|
|5,000|.5321 / .4985|.5608 / .5638|
|10,000|.5327 / .5186|.5248 / .4407|

The floor's improvement is direct evidence that the terminal rate is not inert. Rewarm's early disadvantage appears in both held data and learning data: at3,000 fresh questions its learning NLL rises to.6486, versus.5859 at the floor. This pattern is not simply lower training error coupled to worse generalization.

Recorded changing-batch training cross-entropy also favors the floor early: mean **.2295 vs.2331** over the first1,000 questions and **.2171 vs.2309** over the next1,000. This metric covers all retained assistant tokens, including the wrapper, and excludes the separately added penalty; it must not be numerically equated with answer-only NLL. Both schedules end with similar last1,000-question training CE (.2071/.2087), while their fixed learning32 panels differ substantially. A small fixed learning panel and changing batches answer different questions.

The subsequent42k→50k floor stage improves its own fixed learning32 **.4403→.3603**, while held400 changes only **.5327→.5271** and fluctuates after early gains. Original32k training also improves late during decay: from25,600 to32,000 questions, held NLL **.5786→.5688** and learning NLL **.4256→.3972**. The learning32 questions differ between these stages, so their absolute levels cannot be compared as one continuous training curve.

AC parameters contribute roughly97–98% of the squared pre-clip group norm in the available traces; reader gradients remain nonzero. This establishes which group dominates joint clipping. It does **not** establish reader starvation or proportional Adam update sizes: adaptive moments prevent that inference from raw gradient-norm shares. Reader parameter movement, gradient alignment and whether separate clipping would help remain unmeasured. Actual Adam update norms/directions, objective-versus-penalty gradient decomposition and a matched clipping counterfactual are absent. A larger LR changing the optimization trajectory is a plausible explanation for rewarm's early harm; these logs do not identify its mechanism.

Coverage: batch counts describe the retained checkpoint prefixes plus final published suffix, **not all physical updates performed across failed/preempted attempts**. Combined-norm/loss curves reconstruct that retained trajectory. Rewarm's separate AC/reader norms are available only for its final8,000-question resumed suffix; discarded preemption work is excluded. Gradient norms are valid, but the known unreliable activation-RMS/cosine diagnostics are unused. All runs are single realizations on repeatedly inspected development data; lower final NLL is not by itself an answer-quality winner.

The modest2× LR comparison can test schedule sensitivity while keeping clipping unchanged. These measurements provide no reason to add a clipping change to those arms.

[Exact timelines, question-window summaries, source hashes and missing measurements](LR_GRADIENT_EVIDENCE_V1.json). Local reports only; no remote calls, experiments, confirmation access or frozen-code changes.
