# What answer NLL tells us in the 16k runs

NLL is useful for tracking learning within an AC system, but is not a calibrated answer-quality score across systems. The same answer can be expressed correctly in different words; teacher forcing also supplies every earlier gold token. Greedy generation measures a different behavior.

On the existing development review, lower NLL usually ranks a fully supported AC answer ahead of an incomplete or incorrect one. The table gives the fraction of all correct/other question pairs where the correct answer has lower NLL (ties count half). This is a descriptive ranking check, not an accuracy score or a chosen deployment threshold.

| AC system, all 16k × 1 | Correct /87 | Median NLL, correct /other | Correct answer has lower NLL |
|---|---:|---:|---:|
| 27B supervision · 2B encoder · 1/16 | 66 | 0.356 / 1.057 | 79.4% |
| 27B supervision · 2B encoder · 1/8 | 51 | 0.310 / 0.851 | 81.0% |
| 9B supervision · 2B encoder · 1/16 | 52 | 0.361 / 0.703 | 76.6% |
| 9B supervision · 2B encoder · 1/8 | 55 | 0.344 / 0.784 | 81.1% |

That within-system relationship does not justify comparing absolute NLL against a universal “excellent” cutoff. Frozen P2 full context has higher mean NLL on all 100 questions than the 27B/1/16 AC model (0.728 versus 0.636), yet more fully supported answers on the same87 references (78 versus 66). The full-context controls have only7–9 non-fully-supported answers here, so their analogous ranking estimates are especially imprecise.

Use gold-answer NLL and fixed learning-question NLL for frequent, inexpensive training panels. Keep frozen full/no-context controls visible; accompany promising final pairs with source-grounded generated answers. The reconstruction target below 0.2 nats/token is a different task and should not be carried over as a QA quality threshold.

All100 questions remain in the original NLL reports. This analysis uses the previously frozen 87 supported-reference subset for every system, agent-authored judgments and one training seed. It reuses a repeatedly inspected development cohort; it is not independent confirmation or a statistical claim about future questions. Partial answers count as “other.” No new model evaluation or threshold tuning was performed.

[All sources and answers](MODEL_COMPARISON_COHORT100.tressoir.html) · [Exact statistics and evidence hash](NLL_AND_GENERATED_ANSWERS.json).
