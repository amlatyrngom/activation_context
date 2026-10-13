# Why the smaller encoder batches failed the numerical check

The one-minute diagnostic found **identical captured valid masks and positions** and an exactly reproducible original batch. It traced the shortest and longest passage from the affected 24-passage microbatch. It made no optimizer updates and reverified unchanged parent files.

| Traced passage | First numerical difference | Final encoder-state relative difference |
|---|---|---:|
| Shortest: 109 content tokens | First full-attention side layer; mixer, side input and first three layers exact | 1.88% |
| Longest: 2,360 content tokens | First mixer output: 0.009% relative difference | 2.15% |

For the longest passage, two output heads differ by 3.09% and 3.11%, crossing the existing 3% gate. The shortest passage's captured output heads remain within that gate. This limited forward trace does not supersede the broader failed output/gradient comparison.

The evidence is consistent with batch-shape numerical differences accumulating through the encoder. It does not isolate the exact differing operators or prove the backward behavior. TF32 flags alone cannot be assumed to explain it: the shortest passage first differs after entering full attention despite identical encoder inputs.

**Decision: retain the original 65k batching budget.** No tolerance was relaxed, no speed measurement was made, and this diagnostic is not an equivalence pass. [Numeric receipt](PIECEWISE_FORWARD_LOCALIZATION_RESULT.json).
