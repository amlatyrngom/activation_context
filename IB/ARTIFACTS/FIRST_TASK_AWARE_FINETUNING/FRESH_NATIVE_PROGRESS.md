# Fresh data: 42,000 → 50,000 unique questions

Inherited Adam and constant final learning rate; 8,000 fresh balanced questions × 1 epoch at 2k evidence, merged passages, compression 1/16.

Initial full-context and no-context controls stay frozen; current-reader controls are measured at 0%, 50% and 100%.

| Stage questions seen | Held 400 NLL | Fixed 32 training NLL | Current full-context NLL | Current no-context NLL |
|---:|---:|---:|---:|---:|
| 0/8,000 | 0.5327 | 0.4403 | 0.6497 | 1.8143 |
| 800/8,000 | 0.5305 | 0.4727 | — | — |
| 1,600/8,000 | 0.5185 | 0.4535 | — | — |
| 2,400/8,000 | 0.5210 | 0.4104 | — | — |
| 3,200/8,000 | 0.5214 | 0.4129 | — | — |
| 4,000/8,000 | 0.5183 | 0.3951 | 0.6446 | 1.8139 |
| 4,800/8,000 | 0.5248 | 0.4078 | — | — |
| 5,600/8,000 | 0.5258 | 0.4084 | — | — |
| 6,400/8,000 | 0.5243 | 0.3753 | — | — |
| 7,200/8,000 | 0.5183 | 0.3963 | — | — |
| 8,000/8,000 | 0.5271 | 0.3603 | 0.6460 | 1.8124 |

Training completed in 92m 14s wall time: 62.4 minutes of training and 15.6 minutes of in-epoch reporting; startup, initial controls and other work account for the rest. No examples dropped. Final held NLL improves from 0.5327 to 0.5271. The final checkpoint is used; the lower interim panels do not determine selection. The same 100-answer evaluation is queued.

Observed: 2026-09-23 06:22:14 UTC.
