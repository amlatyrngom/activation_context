# Completed 4B encoder comparison

Both systems trained on 16,000 unique 27B-synthesized questions × 1 epoch at 1/16 compression. Both readers are 9B.

| System | NLL: 9B questions | NLL: 27B questions | Learning NLL | Correct / 87 supported |
|---|---:|---:|---:|---:|
| 2B encoder / P2 | .64291 | .56362 | .42437 | 66 |
| 4B encoder / P4 | .60482 | .56218 | .41310 | 60 |

The 4B system has lower mean development NLL but fewer fully supported generated answers in the fixed 100-question review: six gains and twelve losses. Frozen P4 full context gets 80/87 correct. More encoder capacity is not yet a demonstrated solution to the answer-quality gap; keep 2B for the requested 32k run.

The mean 400-question NLL difference is −.01976; its descriptive paired-question bootstrap 95% interval is [−.04793, +.00774]. This omits training-seed and source-cluster variance. P4 also has a different pretrained reader and reconstruction quality, so the comparison measures the available trained systems rather than isolated encoder size. The 4B run took 4h 14m 47s on RTX PRO 6000, including 10% panels and saves, versus roughly 2h 45m for 2B at 1/16.

[All 900 responses and source-grounded reasons](MODEL_COMPARISON_COHORT100.tressoir.html) · [Numerical analysis](ENCODER4B_FINAL_DEVELOPMENT.json).
