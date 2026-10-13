# The improved continuation at8k context

**The additional10,000 short-context questions improve AC NLL at8k evidence from0.6760 to0.6219 on the same400 questions.** Both8k evaluations use independently encoded passages, the same evidence/distractors and row counts, and original batching.

| View | Original32k pair | Continued42k pair |
|---|---:|---:|
| AC | 0.6760 | 0.6219 |
| full context | 0.6135 | 0.6084 |
| no context | 1.8106 | 1.8143 |

AC improves on223/400 questions and worsens on177; the mean NLL change is-0.0541. Full/no-context rows use each checkpoint’s own reader. The much smaller changes in those controls support an improvement specific to using the AC channel, without isolating the encoder from its paired reader.

All100 selected AC answers were generated:100 normal EOS endings, no token caps. Their source review is complete: **61/87 fully correct**, versus58/87 for the parent, with9gains and6regressions. The remaining supported answers are13partial,12incorrect and1existing contextual ambiguity. [Inspect all100 answers](STRONG_LONG_COHORT100.tressoir.html). Selected100 AC NLL improves0.7352→0.6567; similarity to a full-context NLL would not establish answer-quality parity.

At this point the separate original32k→8k adaptation has a0.6422 panel after2,400fresh questions. The42k starting point is already better on this metric, which motivates preparing a gentle continuation from a stronger checkpoint. Final checkpoint choice awaits the rewarm answer/context comparisons.

This is repeatedly inspected development data. Added context consists of retrieved distractors around the same required source, not a new multi-document reasoning test. Native2k and this8k cell differ in both context length and merged/passagewise layout.
