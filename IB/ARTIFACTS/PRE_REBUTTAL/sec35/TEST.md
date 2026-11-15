# §3.1 on the held-out test1000 (2B encoder -> 9B reader, HotpotQA)

Paired on the same 1,000 test questions: delta [95% bootstrap CI], exact McNemar p; **bold** = p < 0.05.

| System | 20I | 100I | 200I | plain k=20 | plain k=100 | plain k=200 |
|---|---|---|---|---|---|---|
| Text, untuned | 67.9 | 66.4 | 66.2 | 48.7 | 54.4 | 57.6 |
| AC KL stage 2, frozen reader (MAIN) | 68.9 | 67.1 | 66.5 | 49.9 | 58.0 | 60.5 |
| Text + LoRA (reference) | 78.5 | 77.6 | 76.8 | 56.3 | 63.6 | 65.7 |

## Main row − untuned text, same setting

| 20I | 100I | 200I | plain k=20 | plain k=100 | plain k=200 |
|---|---|---|---|---|---|
| +1.0 [-0.9, +2.9] p=0.358 | +0.7 [-1.5, +3.1] p=0.600 | +0.3 [-2.1, +2.7] p=0.870 | +1.2 [-0.9, +3.3] p=0.299 | **+3.6** [+1.4, +5.7] p=0.002 | **+2.9** [+0.6, +5.4] p=0.021 |

## Main row at k=100 / 200 − text at plain k=20

| AC k | vs untuned k=20 | vs text + LoRA k=20 |
|---|---|---|
| 100 | **+9.3** [+6.8, +11.9] p<0.001 | +1.7 [-0.9, +4.4] p=0.264 |
| 200 | **+11.8** [+9.3, +14.5] p<0.001 | **+4.2** [+1.5, +7.2] p=0.008 |

All outputs re-scored to their summary accuracy (±0.05).
