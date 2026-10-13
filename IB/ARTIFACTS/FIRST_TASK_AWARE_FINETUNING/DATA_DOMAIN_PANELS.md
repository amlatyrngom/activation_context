# Final development panels by source family

All three runs completed 8,000 question presentations. Same 200 development questions; approximately 16.0M encoder tokens per run, within 0.7%. Answer-only NLL in nats/token; lower is better. Full/no-context views use each checkpoint’s current reader. Small family panels are descriptive, not firm domain rankings.

## 2,000 unique questions × 4 epochs

| Source family | Questions | AC | Full context | No context |
|---|---:|---:|---:|---:|
| code | 60 | 1.229 | 0.791 | 2.306 |
| documents | 16 | 1.223 | 0.838 | 2.336 |
| financial | 10 | 0.979 | 0.954 | 2.631 |
| legal | 12 | 1.451 | 0.706 | 2.015 |
| math_arxiv | 24 | 0.883 | 0.533 | 1.614 |
| math_textbook | 16 | 0.362 | 0.439 | 1.196 |
| office | 12 | 0.859 | 0.747 | 1.964 |
| technical_web | 10 | 1.376 | 0.507 | 2.114 |
| web | 40 | 0.751 | 0.775 | 1.801 |

## 4,000 unique questions × 2 epochs

| Source family | Questions | AC | Full context | No context |
|---|---:|---:|---:|---:|
| code | 60 | 0.998 | 0.717 | 2.314 |
| documents | 16 | 1.001 | 0.763 | 2.322 |
| financial | 10 | 0.819 | 0.929 | 2.571 |
| legal | 12 | 1.230 | 0.694 | 2.024 |
| math_arxiv | 24 | 0.765 | 0.510 | 1.610 |
| math_textbook | 16 | 0.390 | 0.433 | 1.200 |
| office | 12 | 0.574 | 0.704 | 1.963 |
| technical_web | 10 | 1.185 | 0.494 | 2.151 |
| web | 40 | 0.622 | 0.762 | 1.802 |

## 8,000 unique questions × 1 epoch

| Source family | Questions | AC | Full context | No context |
|---|---:|---:|---:|---:|
| code | 60 | 0.889 | 0.678 | 2.321 |
| documents | 16 | 1.004 | 0.687 | 2.308 |
| financial | 10 | 0.846 | 0.853 | 2.545 |
| legal | 12 | 1.279 | 0.655 | 2.018 |
| math_arxiv | 24 | 0.671 | 0.465 | 1.619 |
| math_textbook | 16 | 0.473 | 0.407 | 1.233 |
| office | 12 | 0.585 | 0.675 | 1.971 |
| technical_web | 10 | 1.073 | 0.463 | 2.187 |
| web | 40 | 0.551 | 0.719 | 1.812 |

Initial panels and the common 4,000-presentation midpoint remain in the JSON companion. The separate 200-question confirmation is complete; see DATA_BUDGET_CONFIRMATION.md. The initial full-context mean for this cohort is 0.869; raw NLL differs from the earlier 100-question cohort across multiple families, not only law/office.

Exact gold-answer masks were reconstructed from the same tool-answer template. Recomputed overall values matched stored panel scores within 1e-9. No new model scoring, gold edits or question filtering.
