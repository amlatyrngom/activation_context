# 32,000-question checkpoint → 8,000 fresh questions at 8k evidence

Fresh Adam from the 32,000-question checkpoint; 8,000 fresh questions × 1 epoch at 8k evidence, independently encoded passages, compression 1/16.

Initial full-context and no-context controls stay frozen; current-reader controls are measured at 0%, 50% and 100%.

| Stage questions seen | Held 400 NLL | Fixed 32 training NLL | Current full-context NLL | Current no-context NLL |
|---:|---:|---:|---:|---:|
| 0/8,000 | 0.6760 | 0.6492 | 0.6135 | 1.8106 |
| 800/8,000 | 0.6397 | 0.7934 | — | — |
| 1,600/8,000 | 0.6398 | 0.7108 | — | — |
| 2,400/8,000 | 0.6422 | 0.6693 | — | — |
| 3,200/8,000 | 0.6439 | 0.6389 | — | — |
| 4,000/8,000 | 0.6356 | 0.6533 | 0.5926 | 1.8118 |
| 4,800/8,000 | 0.6195 | 0.6344 | — | — |
| 5,600/8,000 | 0.6153 | 0.6257 | — | — |
| 6,400/8,000 | 0.6137 | 0.6122 | — | — |

A dash means the moving control was not measured at that panel. Final actual-answer evaluation remains pending. Interim NLL alone does not choose a checkpoint.

Observed: 2026-09-23 06:52:57 UTC.
