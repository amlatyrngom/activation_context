# Qualitative comparison of fresh questions versus repeated passes

Prepared and deployed; scheduling is controlled by the parent agent. One RTX allocation, at most 45 minutes. No training or checkpoint selection occurs in this evaluation.

Compare five responses to each of 100 fixed questions:

- Frozen P2 reader before task-aware training, with full evidence and distractors.
- The same frozen reader without context.
- Paired encoder/reader after 2,000 questions × 4 passes.
- Paired encoder/reader after 4,000 questions × 2 passes.
- Paired encoder/reader after 8,000 questions × 1 pass.

Membership is frozen in [DATA_BUDGET_COHORT100_SELECTION.json](./DATA_BUDGET_COHORT100_SELECTION.json). Within each family, rank item IDs by SHA256 of the fixed selection key plus ID and take half: code 30, arXiv 12, textbook 8, financial 5, web 20, technical web 5, documents 8, legal 6, office 6. Retain the original development order. Neither model output nor loss participates in selection. These are development questions, separate from the reserved confirmation cohort and the earlier 100-item report.

NLL evaluates the complete original 200-question sequence with microbatch two, then selects the frozen 100 values. Record both **subset-100 NLL** and **full-200 NLL** with distinct labels, plus replay differences from the original panels. Free generation decodes only the selected 100: greedy, thinking disabled, 160-token cap, no gold-answer prefix.

The evaluator requires the completed final paired weights for every schedule. Frozen baselines use the initial P2 reader; trained AC views use their own paired readers. This is a whole-system comparison, not identical-reader evidence that compression beats full context.

After all four processes complete, inspect all 100 source/gold triples and all 500 responses. Produce `DATA_BUDGET_COHORT100.tressoir.html` and its evidence JSON with descriptive counts, individual reference concerns, actual responses and identity details. Link from the earlier report only once this new report is complete. Do not alter questions, golds or original metrics to improve results.

CPU validation passed: exact original/reference item IDs and ordering, common source/retrieval data across all three materials, preservation of required source and gold, identical full/AC evidence and correct row budgets. Remote completed-pair markers and all required nonempty weights/configs were checked. The GPU evaluator remains unrun by this subagent.

Private entry points: `bench/evaluate_data_budget_cohort.py`, `bench/data_budget_cohort.sbatch`. Remote outputs: `reports/data_budget_cohort100/`. Local analysis staging and validation receipts: `IB/TMP/FIRST_TASK_AWARE_FINETUNING/data_budget_cohort_review/`.
