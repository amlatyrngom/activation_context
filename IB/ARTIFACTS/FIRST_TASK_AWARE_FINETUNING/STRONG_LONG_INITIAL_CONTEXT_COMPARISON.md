# The32k checkpoint before long-context adaptation

The same100 questions are evaluated with independently encoded passages at each evidence budget. Both reader and encoder weights are frozen at the original32k-question final checkpoint. These are starting-point measurements, not the running adaptation’s results.

| Evidence budget | AC answer NLL | Full context | No context | Mean compressed rows |
|---|---:|---:|---:|---:|
| 2k | 0.7019 | 0.6372 | 1.8396 | 124 |
| 8k | 0.7352 | 0.6032 | 1.8391 | 503 |
| 32k | 0.7537 | 0.5765 | 1.8396 | 2055 |

Adding retrieval context improves the full-context reference while modestly worsening AC NLL. There are no row caps. This setting adds distractors; it does not test questions requiring evidence spread across many documents.

Only8k has actual100-answer decoding here:58/87source-supported answers fully correct. [Inspect those answers](STRONG_LONG_COHORT100.tressoir.html). Smallno-context variation reflects different evaluation batching.

The original checkpoint’s native2k merged-passage layout scores0.6079 on these100; the0.7019 above changes to independent passage encoding. This layout change itself has a cost before adaptation.
