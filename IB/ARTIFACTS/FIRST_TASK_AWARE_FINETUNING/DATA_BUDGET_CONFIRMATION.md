# Fixed-final data-budget confirmation

Completed on the reserved 200 questions. The fixed final checkpoints all saw 8,000 question presentations and approximately equal encoder-token presentations. No checkpoint was selected or changed using these scores. This confirmation cohort is now used; further tuning uses development data and later studies reserve separate confirmation questions.

| Unique questions × epochs | AC NLL | Current-reader full context | Current-reader no context |
|---|---:|---:|---:|
| 2,000 × 4 | 1.07759 | .65187 | 1.94992 |
| 4,000 × 2 | .89501 | .62433 | 1.95209 |
| 8,000 × 1 | .84435 | .59360 | 1.97283 |

Fresh data leads on this separate cohort too. The 8k-minus-4k difference is −.05065 nats/token; a paired-question bootstrap interval is [−.10651, .01977], so that modest advantage remains uncertain. Both clearly improve over 2k×4: paired intervals [−.30226, −.16555] and [−.25298, −.12025]. These descriptive 95% intervals use 5,000 resamples, seed20260921; they exclude training-seed and source-cluster uncertainty.

Corrected GPU job23424466 completed all three views on all200 questions and12 fixed decodes per view per arm. First GPU attempt23422240 failed before scoring and was superseded.

Use `data_budget_material/confirmation.json` with the exact common 8,200-record retrieval corpus from `material_2000.json`. All arms use identical 2k/four-passage evidence, ratio1/16 and microbatch2. AC, full-context and no-context scores use each paired checkpoint's current reader. Report mean-per-question answer NLL, all per-question values, actual token/row counts, memory and time. Up to12 fixed greedy examples per view are optional and bounded by remaining time; literal NLL and generated-answer correctness stay separate.

`bench/data_confirmation.sbatch` runs three fresh processes on one RTX PRO GPU with a30-minute allocation, using `bench/evaluate_data_confirmation.py`, `bench/evaluate_long_cross.py` for checked metric helpers, and the pinned confirmation-only `bench/run_qa_confirmation_v3.py` builder. It validates all three completed final pairs before the first model load. The Python scoring/decode budget is420seconds per arm, beginning after model/input preparation. No optimizer is created.

Remote outputs: `reports/data_budget_confirmation/<arm>/{partial.json,result.json,report.tressoir.html}`, with paired comparisons in `summary.json`. Immediately before the first score, `confirmation_used.json` records the exact frozen evaluation design and marks this cohort used. After execution it is no longer an untouched confirmation set; do not tune on it. The later27B comparison has its own separate confirmation cohort.

If interrupted, rerun the same SBATCH; matching completed outputs are skipped unchanged and completed partial views are reused. An individual arm may be passed as its positional argument. Changed pair/recipe identities are rejected. Focused CPU checks passed for the three final epoch markers, incomplete-run rejection, matched per-question deltas, input consistency and immutable completed-result skips. GPU execution completed successfully; each arm took roughly three minutes after model/input preparation.

Startup failure and narrow correction (Sep21): the reserved source documents are intentionally absent from the common distractor index, but the BM25 exclusion-size estimator assumed every exclusion ID was indexed. The confirmation-only builder now intersects exclusion IDs with actual indexed documents. Each question still receives its exact own-source payload; the8,200-record retrieval pool, gold targets, evidence ordering and compression rules are unchanged. The original cross-evaluation builder remains pinned and unchanged: its100 development questions have zero exclusion IDs absent from its4,100-record retrieval corpus. No product code was edited.

The focused real-material CPU regression reproduced the exact KeyError with the old builder, then built all200 confirmation inputs successfully with the correction. It verified unchanged8,200 retrieval records /7,082 indexed documents, retained all200 external own-source documents in both AC and full-context evidence, unchanged gold answers and exact row-budget calculations. Evidence: `IB/TMP/FIRST_TASK_AWARE_FINETUNING/confirmation_retrieval_fix/cpu_regression.json`. The corrected evaluator, confirmation-only builder and SBATCH are deployed to the remote `bench/`; remote SHA256 matches local. Failed-job script copies are retained with `_failed_23422240` names. No job was submitted or cancelled during this fix. At the time of the fix the confirmation-use marker was absent; the completed retry has now marked this cohort used.
