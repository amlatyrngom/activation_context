# 9am EDT checkpoint — September 22

Latest local evidence: **2026-09-22 13:00 UTC**. The original 27B-supervised **2B encoder, 1/16 compression, AC LR 5e-5** remains the leading practical recipe. The lower-rate trials are unfinished and currently worse. The requested 32k training set is ready; launch awaits the completed LR comparison and GPU allocation.

## Learning rate: smaller has not helped

Each row compares identical question exposure, initialization, data order and schedule. Reader LR remains 2e-5; AC LR changes from 5e-5 to 2.5e-5. Development NLL averages the same 400 questions, 200 from each synthesis model. Learning NLL uses the same fixed 32 training questions within this bank. Lower is better.

| Compression | Completed quality panel | Original LR: development NLL | Half LR: development NLL | Learning NLL: original / half |
|---|---|---:|---:|---:|
| 1/16 | 16,000 × 0.7 epochs | 0.674 | 0.849 | 0.484 / 0.643 |
| 1/8 | 16,000 × 0.9 epochs | 0.655 | 0.796 | 0.467 / 0.602 |

- **1/16 half-LR trial:** 16,000 unique questions × 0.71 passes; training. Latest report: 2026-09-22 12:59:40 UTC.
- **1/8 half-LR trial:** 16,000 unique questions × 0.97 passes; training. Latest report: 2026-09-22 12:59:40 UTC.

Use the final development results and actual answer checks to settle the recipe. These repeated intermediate observations are exploratory, not independent statistical tests. Separate AC/reader clipping is prepared as a subsequent targeted experiment; it is not an additional gate for 32k. [Stability options](STABILITY_OPTIONS.md).

## What completed experiments establish

- **The channel transfers useful information.** On the fixed 87 questions with supported references, the 27B-supervised 2B/1/16 system gives 66 supported answers, versus 15 for frozen no context and 78 for frozen full context. All 100 questions remain visible and included in NLL; the same 13 reference problems are excluded only from the optional semantic denominator. These are transparent agent annotations, not independent adjudication.
- **More rows and a larger encoder are not automatic wins.** The 27B/2B/1/8 system gives 51/87 supported answers; 4B/P4 gives 60/87. The 4B system has slightly lower development NLL but changes both encoder and pretrained paired initialization. This does not isolate encoder capacity or demonstrate a need for full fine-tuning.
- **Longer evidence still challenges AC.** On the same 100 questions, expanding evidence from roughly 2k to 32k tokens raises AC NLL by .057–.075 across the three tested systems, while each full-context reader improves. No-context values replay exactly. These questions retain their required source and gain retrieved evidence; this is a distractor/long-context screen, not proof about multi-document reasoning.
- **16k is not a demonstrated convergence point.** All four original 2B systems improved during their final fifth of the epoch. The data-volume question remains open.

[All 900 reviewed responses](MODEL_COMPARISON_COHORT100.tressoir.html) · [Context transfer](CONTEXT_TRANSFER100.tressoir.html) · [Original five-arm confirmation](MODEL_COMPARISON_CONFIRMATION.tressoir.html). The original confirmation is published separately and does not guide the new LR or data-scale choices.

## Next data comparison and timing

The 27B bank has **43,000 published questions**, with balanced reusable releases and the agreed law/office coverage. The **32,000-question** training selection and a fresh separate confirmation are frozen. The requested run starts from original P2, with fresh optimizer state, one epoch and 10% quality panels. Its companion uses the original **16,000 questions × 2 epochs**, with the same selected recipe and total question presentations. Equal question counts do not guarantee equal tokens or runtime; both will be reported.

Training at 1/16 is estimated at **5½–6 hours after allocation**. The LR decision is expected around **9:30–10am EDT**. Current shared-cluster dry-run estimates are later than immediate allocation and can change as existing jobs finish. The 4pm EDT target therefore remains provisional; complete passes take priority over an arbitrary cutoff. No EC2 is running. At most four autonomous MIT GPUs remain allowed, including synthesis.

[Live overview](../../TMP/SYNC/TASK_AWARE_QA/report.tressoir.html) · [32k preparation](QA_32K_PREPARATION.md) · [Matched data comparison](DATA_AMOUNT_32K.md) · [Code handoff](IMPLEMENTATION.tressoir.md).
