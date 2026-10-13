# Gentle context adaptation from 42,000 questions

Inherited Adam and constant final learning rate; 4,000 familiar questions × 1 epoch at 8k evidence, independently encoded passages, compression 1/16. The lifetime unique count stays 42,000.

Initial full-context and no-context controls stay frozen; current-reader controls are measured at 0%, 50% and 100%.

| Stage questions seen | Held 400 NLL | Fixed 32 training NLL | Current full-context NLL | Current no-context NLL |
|---:|---:|---:|---:|---:|
| 0/4,000 | 0.6219 | 0.6013 | 0.6084 | 1.8143 |
| 400/4,000 | 0.6063 | 0.5514 | — | — |
| 800/4,000 | 0.6006 | 0.5278 | — | — |
| 1,200/4,000 | 0.6076 | 0.5365 | — | — |

A dash means the moving control was not measured at that panel. Final actual-answer evaluation remains pending. Interim NLL alone does not choose a checkpoint.

Observed: 2026-09-23 06:45:27 UTC.
