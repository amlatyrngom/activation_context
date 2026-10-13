# Learning-rate decision

Use **AC LR 5e-5, reader LR 2e-5, 1/16 compression** for the 32,000-question run and its 16,000-question × 2-epoch companion. Both start from the original P2 pair with fresh optimizers.

All comparisons below trained on **16,000 unique 27B-synthesized questions × 1 epoch**. Within each ratio, only the peak AC learning rate changed.

| Compression | AC LR | Development NLL, 400 questions | Learning NLL, fixed 32 questions |
|---|---:|---:|---:|
| 1/16 | 5e-5 | **0.6033** | **0.4244** |
| 1/16 | 2.5e-5 | 0.7071 | 0.5354 |
| 1/8 | 5e-5 | 0.6463 | 0.5218 |
| 1/8 | 2.5e-5 | 0.7781 | 0.5690 |

Both generator cohorts separately favor the original learning rate at both ratios. The paired-question bootstrap intervals also favor it; these describe one seed and do not capture source clustering or training-seed variation.

The preselected 18-question source-grounded review supports the choice: original 1/16 has 11 correct, 2 partial and 5 incorrect answers; half-LR 1/16 has 8/5/5. Both 1/8 variants have 8/5/5. This small balanced check is descriptive, not a representative accuracy estimate. The half-LR 1/16 model fixes one repetition problem without correcting that answer.

Halving AC LR did not demonstrate a stability benefit. Joint gradient norms exceeded100 on 2 versus6 batches at1/16, and174 versus170 at1/8 (original versus half LR). These are gradients before clipping, not parameter-update magnitudes. Separate AC/reader clipping remains a useful optional controlled experiment after the larger runs get their allocations.

This decision uses development evidence only. The original confirmation results did not guide it. The larger comparison keeps the selected recipe fixed and uses a fresh reserved confirmation set after both full runs finish.

[Full evidence](LR_DECISION.json) · [Gradient summaries](LR_FINAL_GRADIENTS.json) · [1/16 answer review](LR_R16_QUALITATIVE_CHECK.md) · [1/8 answer review](LR_R8_QUALITATIVE_CHECK.md)
