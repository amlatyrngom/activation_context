# Separate passage encoding at 32k context

A frozen-weight screen lowers average answer NLL from **0.8758 to 0.8019**, but the result is mixed and is not yet a preferred configuration. Only 48/100 questions improve; the paired median difference is +0.0023 and a question-bootstrap 95% interval for the mean difference is [−0.1778, +0.0212]. This interval omits source clustering and training-seed variation.

Both layouts use the same earlier 4,000-question × 1.8-pass checkpoint, 100 development questions, evidence, gold targets and reader weights. There is **no additional training**. Evidence is about 32k tokens with up to 63 retrieved distractors, at 1/16 compression. Both allocate exactly 205,886 rows across the cohort. Separate passages repeat the question/template and add 2.91% encoder tokens; the largest individual encoder part is 4,865 tokens. The common full-context/no-context NLL is 0.3767/1.8782.

| Source category | Questions | Merged NLL | Separate NLL |
|---|---:|---:|---:|
| Code |30|1.1704|1.1247|
| Documents |20|0.7684|0.7420|
| Numeric |25|0.9534|0.6943|
| Web |20|0.5982|0.6154|
| Wiki |5|0.2601|0.3883|

The average gain is concentrated in numeric questions. At 2k context the same layout change slightly worsened NLL (0.5890→0.6036), so any advantage is context-dependent. Separate encoding also costs more in this implementation: 247 versus 155 seconds to score 100 AC questions, with 31.92 versus 28.31 GiB peak allocated memory. These exclude loading and decoding.

The fixed 12 decoded answers give reasons for caution. Separate encoding improves the specificity of the fainting-mechanism answer and partially recovers the asynchronous retry action, but changes a correct 1930s answer to 1990s, substitutes the wrong television series, and corrupts a previously correct biome answer. Both layouts repeat the same wrong revenue difference. These observations compare with the existing gold; they are not a new source-grounded correctness audit.

Next: retain as an experimental long-context option. A larger decoded review and a matched adaptation test are needed before promotion; the 16k fresh-question 1/16 versus 1/8 runs take priority. Job 23439119 is generating answers for all 100 questions under a 55-minute bound while the fresh-data pool fills.

Evidence: ORCD job 23436678 completed in 19m26s. Results and all per-question scores are under `IB/TMP/FIRST_TASK_AWARE_FINETUNING/passage_screen_32k/`; `analysis.json` records the paired calculation. Live raw reports are under `IB/TMP/SYNC/TASK_AWARE_QA/passage_screen_32k/`.
