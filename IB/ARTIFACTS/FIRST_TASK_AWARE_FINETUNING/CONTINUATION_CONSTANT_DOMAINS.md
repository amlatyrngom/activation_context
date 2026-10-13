# Where the constant-LR continuation improved

The additional 10,000 fresh questions improve overall answer NLL and source-reviewed correctness. These are the same fixed development questions and the existing source reviews; domain slices are descriptive.

| Domain | NLL questions | Parent NLL | Continued NLL | Correct: parent → continued | Frozen full-context correct |
|---|---:|---:|---:|---:|---:|
| code | 120 | 0.587 | 0.548 | 19/24 → 20/24 | 22/24 |
| documents | 32 | 0.489 | 0.513 | 4/8 → 6/8 | 7/8 |
| legal | 24 | 0.821 | 0.779 | 3/4 → 3/4 | 4/4 |
| numeric | 100 | 0.632 | 0.566 | 11/21 → 14/21 | 15/21 |
| office | 24 | 0.535 | 0.527 | 5/6 → 6/6 | 6/6 |
| web | 100 | 0.458 | 0.430 | 21/24 → 22/24 | 24/24 |

Correctness denominators include only the original 87 source-supported references. NLL uses all 400 questions, so the two columns measure different subsets and properties. The legal and office answer samples are particularly small. No source weighting or pruning is changed.

[Inspect all 100 actual answers](CONTINUATION_COHORT100.tressoir.html).
