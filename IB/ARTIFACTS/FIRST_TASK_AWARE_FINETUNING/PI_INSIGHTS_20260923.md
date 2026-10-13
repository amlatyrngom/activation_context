# Evidence for the September 23 PI discussion

Updated **3:20pm EDT / 19:20 UTC**. These are development results; the independent confirmation questions remain unused for tuning.

## One 32k encoder pass already works

A **2B encoder → 9B reader**, trained on **42,000 unique QA questions × 1 epoch**, can use one approximately 32k encoder sequence without extra long-context training.

| Same paired checkpoint, L40S, questions, 32k evidence and total AC rows | Answer NLL | Correct / 87 supported references |
|---|---:|---:|
| Monolithic AC, 1/16 compression | **0.6314** | **66** |
| Independently encoded passages | 0.6792 | 59 |
| Full text, including the same distractors | **0.5744** | **81** |
| No context | 1.8413 | Not decoded in this comparison |

All 300 actual answers were reviewed against their supplied sources. Thirteen reference concerns remain visible and are excluded only from the correctness denominator; headline NLL includes all 100 questions. Monolithic encoding has 14 uniquely correct answers versus seven for passagewise encoding. Full text has 16 gains and one loss versus monolithic AC. The small-looking 0.057 NLL gap to full text therefore still accompanies a material answer-quality gap.

Monolithic encoding also scored these 100 questions faster: **277.9 seconds / 28.27 GiB peak**, versus **425.9 seconds / 31.88 GiB** for separate passages on distinct L40S allocations. This is 35% less observed scoring time, not an optimized serving benchmark. The largest input contains 33,177 formatted source tokens plus 2,074 summary rows, or **35,251 encoder positions in one pass**, without row capping.

An independent RTX PRO 6000 allocation reproduces the monolithic result: **0.6305 NLL and the same 66/87 correct**, with 89 identical responses and 11 source-reviewed changes. All terminate normally. Available-memory autotuning also changes the scoring chunk size between GPU families, so this is not an isolated hardware experiment. [Replication details](PI_MERGED32K_RTX_REPLICATION_V1.md).

The strong checkpoint learned merged short inputs, so retaining that layout may explain its advantage; this is not a demonstrated mechanism. Most extra long context consists of retrieved distractors. The result establishes useful long-input handling, not yet reasoning across distributed required evidence. The NLL advantage also holds on the supported 87 alone: monolithic 0.5702, passagewise 0.6191, full text 0.5263.

[The matched comparison report](PI_SAME_L40S_COHORT100.tressoir.html) contains all 300 actual answers, source notes and token losses. The earlier mixed-device report remains a separate historical snapshot.

## Learning rate and more data

Terminal learning rate still learns. Adding 10,000 fresh questions improved held-400 NLL from 0.5688 to 0.5327 and reviewed correctness from 63/87 to 71/87. Another 8,000 fresh questions reached 0.5271 and 70/87. A 4× reheat improved NLL further but regressed to 62/87. This does not establish saturation or an optimal schedule for a new 100k-question run.

The short, matched L40S comparison has now completed **960 fresh questions × 1 epoch**, with inherited Adam state, identical inputs, matching numerical settings and no dropped questions:

| Setting | Held-100 NLL | Fixed learning-32 NLL | Reviewed correct / 87 |
|---|---:|---:|---:|
| Before this stage | 0.5569 | 0.4372 | Not decoded on this runtime |
| Terminal rate | **0.5496** | 0.3997 | 70 |
| 2× terminal rate | 0.5670 | **0.3452** | 71 |

The terminal encoder/reader rates are 5e-6 / 2e-6; the doubled rates are 1e-5 / 4e-6. Higher LR fits the small pool harder but worsens held loss. Actual answers differ by one correct question overall: four terminal-only correct versus five doubled-rate-only correct. That does not establish an answer-quality winner. All 200 responses end normally, and fresh scoring exactly replays the final panels. The initial full/no-context losses stay frozen at 0.6336 / 1.8413; final reader controls are separately labeled. [The two-column report](PI_NATIVE960_LR_COHORT100.tressoir.html) contains all 200 answers and their source-based assessments.

The larger comparison uses the same **8,000 fresh questions × 1 epoch** for all three rates, preserves Adam state, and reports on 400 fixed questions. All three training runs are complete. Their intermediate ordering varied, but mild cosine also has the lowest NLL at the preselected final:

| Fresh questions seen once | Terminal rate | Constant 2× | Cosine 2× → terminal |
|---|---:|---:|---:|
| 0 | 0.5327 | 0.5327 | 0.5327 |
| 800 | 0.5305 | **0.5179** | 0.5337 |
| 1,600 | **0.5185** | 0.5317 | 0.5227 |
| 3,200 | **0.5214** | 0.5255 | 0.5279 |
| 4,000 | **0.5183** | 0.5339 | 0.5234 |
| 4,800 | 0.5248 | 0.5419 | **0.5212** |
| 5,600 | 0.5258 | 0.5255 | **0.5130** |
| **8,000 — final** | 0.5271 | 0.5229 | **0.5118** |

These are held-400 answer losses. The mild cosine finishes 0.0153 nats/token below the terminal-rate continuation, with fixed learning-32 NLL 0.3199 versus 0.3603. This favors the modest cooldown on held loss; actual answer comparisons remain queued before making an answer-quality recommendation. The short study uses microbatch 1; these larger runs use microbatch 2, so their differing early trends do not isolate data amount alone.

## Direct long-context adaptation

The strong 42k checkpoint completed **960 fresh questions × 1 epoch with monolithic 32k inputs**, on an H200. Final held NLL is **0.6059 versus 0.6296 initially**; fixed learning-set NLL is 0.3874 versus 0.5116. It closes approximately 42% of the initial NLL gap to the frozen full-context reference (0.5734). Actual answer-quality improvement is not yet established.

Panels appeared every 96 questions. The completed final checkpoint will receive 100 decoded answers plus a short-context retention comparison against its parent on the same GPU allocation. The evaluation explicitly retains training's numerical chunk policy. The final NLL panel is available; decoded answers and retention measurements are queued.

## Remaining work and limits

Both LR runs completed around **2:29pm EDT** and long-context adaptation around **2:40pm EDT**, all with exit code zero. Their dependent answer evaluations are waiting for scheduler priority as of 3:10pm. Jobs run on ORCD; pending CSAIL duplicates were canceled. No EC2 is being used.

27B synthesis is queued on ORCD RTX (23565367) to extend the original portable bank from 53,000 published questions plus 615 buffered toward 100k, then 200k. The still-pending CSAIL duplicate was canceled before release. Automatic requeue uses three-hour segments; balanced 1,000-question releases are usable immediately. No new generation has started yet. The existing RTX rate suggests about 11 active GPU hours from 53k to 100k, before queue/setup. Source capacity is adequate for 100k; before 200k, add approximately 5k technical-web source windows if recent acceptance rates persist.

NLL is gold-answer-token loss averaged equally over questions. Correctness counts are source-grounded qualitative annotations on a repeatedly inspected cohort, not independent benchmark accuracy. The strongest current conclusion is that the latent channel carries useful information and handles one long encoder input. Better agent outcomes or comparable outcomes at lower total cost still need their corresponding task and efficiency comparisons.

Live reports: `IB/TMP/SYNC/TASK_AWARE_QA/report.tressoir.html`. [Experiment states and report paths](PI_EXPERIMENTS_20260923.md) and [presentation context](../../../RESEARCH_DESIGN.md) provide the broader framing.

## Reconstruction retention after QA — completed 19:39 UTC

Same original420 held reconstruction passages,1/16 compression,2B→9B pairs and L40S runtime: originalP2 **0.1064 NLL /96.48% token accuracy**;42k QA **1.0812 /73.57%**;50k cosine **1.0798 /73.68%**;42k plus960 monolithic32k **1.0808 /73.63%**. Every source group degrades. The loss is already present at42k; the latest continuations add almost no further penalty. Encoder versus reader contributions are not isolated. [Full reconstruction findings](RECONSTRUCTION_RETENTION.md) and [actual reconstructions](../../TMP/SYNC/TASK_AWARE_QA/reconstruction_retention_v1/report.tressoir.html).
