# Separate clipping at 1/8 compression

**Training completed: September 22, 2026, about 17:17 UTC. The 100-answer readout is pending.**

Both arms trained on the same **16,000 unique 27B questions × 1 epoch**, from the original P2 pair, with identical order, schedule, logical batch of eight, and AC/reader learning rates of 5e-5/2e-5. The only training change clips AC parameters and reader parameters separately to norm 1, instead of clipping their combined gradient to norm 1. This also changes the allowed combined norm; it does not isolate a reader-only intervention.

| Measurement | Original joint clipping | Separate clipping |
|---|---:|---:|
| Development answer NLL, all 400 | 0.6463 | **0.6294** |
| 9B-generated development cohort, 200 | 0.6672 | **0.6480** |
| 27B-generated development cohort, 200 | 0.6254 | **0.6107** |
| Fixed 32-question learning NLL | 0.5218 | **0.4297** |
| Preclip combined gradient norm >100 | 174 | **60** |
| Median AC gradient norm | 11.53 | **8.61** |
| 95th-percentile AC gradient norm | 170.61 | **67.60** |
| Maximum AC gradient norm | 6601.86 | **539.70** |

Separate clipping improves training fit and reduces gradient volatility. Its held-out gain is modest: **−0.0169 NLL**, with a descriptive paired-question bootstrap interval of **[−0.0432, +0.0091]**. Exactly 200 of the 400 questions have lower NLL. This is one seed and repeated development monitoring; the interval omits source clustering and training-seed variance.

The parameter gradient records are measured before clipping. They do not measure Adam parameter updates, and large preclip norms do not imply equally large updates. The known contaminated activation-RMS/cosine diagnostics are excluded.

Keep the ongoing 32k and repeated-data runs unchanged. Inspect the fixed-100 actual responses before deciding whether separate clipping should become the next 1/8 recipe. No reserved confirmation data informed this trial.

[Gradient evidence](GROUP_CLIPPING_FINAL_GRADIENTS.json) · [Frozen experiment design](GROUP_CLIPPING_FOLLOWUP.json) · [Live report](../../TMP/SYNC/TASK_AWARE_QA/oracle_27b_16k_r8_group_clip/report.tressoir.html).
