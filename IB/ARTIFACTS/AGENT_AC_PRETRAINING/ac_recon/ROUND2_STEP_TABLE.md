# Round 2 — step-matched panels (2026-09-21 14:15 UTC)

Held-out gold NLL at each panel, as `held blend / mean of the seven novel sources`. All arms run the same 4,256-update short
schedule (first 34,048 passages), 2B encoder into the 9B reader, encoder LR 5e-5. F, E2, E4, N0, N1 warm-start from KEEP/P2;
**P2s5 is a cold start** (fresh adapters), so it is a floor reference, not the control. F is the control for E2/E4; N0 for N1.
N0/N1 use the nested corpus and their own held set: compare them only with each other.

| step | frac | P2s5 (cold) | E2 (stride 2) | E4 (stride 4) | F (control) | N0 (nested) | N1 (nested + child keys) |
|-----:|-----:|---|---|---|---|---|---|
|    0 | 0.000 | 1.777 / 1.687 | 1.804 / 1.666 | 1.832 / 1.680 | (lost) | (lost) | 1.056 / 1.266 |
|  532 | 0.125 | 1.509 / 1.423 | 0.622 / 0.596 | 0.734 / 0.703 | (lost) | (lost) | 0.770 / 0.881 |
| 1064 | 0.250 | 1.385 / 1.320 | 0.540 / 0.518 | 0.616 / 0.589 | (lost) | (lost) | 0.765 / 0.875 |
| 1596 | 0.375 | 1.216 / 1.148 | 0.480 / 0.457 | 0.544 / 0.518 | (lost) | (lost) | 0.751 / 0.860 |
| 1800 | 0.423 | ~1.05 | ~0.46 | ~0.52 | **0.136 / 0.132** | **0.749 / 0.865** | ~0.745 / ~0.858 |
| 2128 | 0.500 | 0.786 / 0.773 | 0.425 / 0.405 | 0.492 / 0.473 | 0.144 / 0.139 | 0.766 / 0.874 | 0.737 / 0.855 |
| 2660 | 0.625 | 0.612 / 0.606 | 0.382 / 0.365 | 0.450 / 0.426 | running | running | 0.727 / 0.845 |
| 3192 | 0.750 | 0.489 / 0.480 | 0.347 / 0.332 | 0.421 / 0.402 | running | running | 0.717 / 0.838 |
| 3724 | 0.875 | 0.439 / 0.435 | 0.330 / 0.313 | 0.402 / 0.382 | running | running | 0.717 / 0.835 |
| 4256 | 1.000 | 0.393 / 0.389 | 0.315 / 0.302 | 0.390 / 0.371 | running | running | 0.707 / 0.831 |

F and N0 lost their first-attempt panels when the run root's report data was rotated on resume; their entries at 1800 are the
resume-point panel and the next one (the resumed run's x axis is offset: x = 1.0 and 1.5 for these two points).

Final held novel (result.json): E2 0.2705 / 91.8 %, E4 0.3334 / 90.0 %, N1 0.7110 / 82.9 %, P2s5 0.3899 / 88.2 %;
paper reference P2 (full corpus, one epoch) 0.106 / 96.5 %.

## Reading

1. **Strided pooling costs more than it looked.** Against the cold P2s5 the pooled arms look strong, but against F, the matched
   warm-started control, they are far behind: at step 1800 F is at 0.136 while E2 is ~0.46 and E4 ~0.52. Even after the full
   4,256 updates E2 (0.315) and E4 (0.390) have not reached what F holds at step 1800. A fresh pooler destroys the content
   representation at step 0 (E2/E4 start at ~1.8, no better than a cold model) and the run is spent recovering.
2. **Stride 2 beats stride 4 by about 0.07 nats at every panel**, and E4's final number is the same as the cold start's (0.390 vs
   0.393): at stride 4 the adaptation budget buys back exactly what a cold run reaches without pooling.
3. **Recursive child keys do nothing measurable so far**: N0 0.749 at step 1800 against N1's ~0.745, inside noise. The nested
   curve is also nearly flat after the first panel (N1: 0.770 at 12.5 % to 0.707 at the end).
4. **F itself is flat** (0.136 to 0.144): P2 had converged on this corpus, so more of the same data adds nothing. 0.13-0.14 is the
   bar any architecture change has to clear at this budget.
5. **The efficiency gain was never measured.** Wall clock was 5.8 h (P2s5, H200), 6.2 h (E2, RTX), 6.6 h (E4, CSAIL H200) for the
   same 4,256 updates: pooling did not make training faster, because the 9B reader's forward and backward dominates the step and
   the pooler adds its own attention. The saving is an encode-time property and needs a dedicated encoder-only benchmark.
