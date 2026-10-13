# Piecewise encoding: passagewise training finished; matched comparison pending

The passagewise arm has seen all **4,000 unique questions × 1 epoch**, with final held-out answer NLL **0.9165** and fixed training-panel NLL **0.5529**. Its last three panels were 0.9247, 0.9128 and 0.9165; the end of this small run is flattening. The merged arm still has 800 questions left after scheduler preemptions.

At the latest matched panel, **2,800 unique questions seen once**, independently encoding each passage gives **0.933 held-out answer NLL**, versus **1.315** for encoding all passages together. Both use the same roughly8k evidence, same total compressed rows at1/16, originalP2 reconstruction initialization and matched training settings. Planned total:4,000 unique questions ×1 epoch.

| Fixed400 development questions | Piecewise | Together |
|---|---:|---:|
| Before QA training |3.1729|2.1079|
| After400 fresh questions |1.2237|1.3743|
| After800 fresh questions |1.1650|1.3689|
| After1,200 fresh questions |1.1493|1.3454|
| After1,600 fresh questions |1.1314|1.3207|
| After2,000 fresh questions |1.0435|1.3194|
| After2,400 fresh questions |0.9653|1.3172|
| After2,800 fresh questions |0.9326|1.3153|
| After3,200 fresh questions |0.9247|pending resume|
| After3,600 fresh questions |0.9128|pending resume|
| After4,000 fresh questions |0.9165|pending resume|
| Current-reader full context after2,400 |0.6244|0.6126|
| Current-reader no context after2,400 |1.8216|1.8106|

Piecewise overcomes its worse initial QA score quickly. Its advantage appears in both generator cohorts. At2,400questions,312/400 had lower NLL. The current-reader full-context scores are close, so the AC difference is not accompanied by a comparable difference in the full-context control.

This is a **development result**; the final matched answer-quality and 32k-transfer results are pending. At2,400questions, the descriptive paired-question interval for piecewise minus together is[−0.4010,−0.3028]; it omits source clustering and training-seed variation, and repeated checkpoints are not independent tests. Final paired checkpoints, fixed100 answers and32k transfer remain queued. [Exact comparison](PIECEWISE_60_PERCENT.json).

Both runs have passed the old120-question memory failure. Existing activation checkpointing is enabled for all side-encoder sequences in both arms; the model architecture, training data and row budgets are unchanged by that repair.

The completed frozen9B full-context control at8k scores74/87 fully correct,8partial,4incorrect and1ambiguous, versus78/87 correct at the original~2k evidence budget. No-context answers are byte-identical on all100. The one new ambiguity comes from added evidence containing a competing software-release announcement; it is recorded separately without changing the original87/13 source-reference split. This indicates the extra retrieved context can complicate the task even for a full-context reader. The trained-layout answer comparison remains pending.

At the matched1,600-question checkpoint, the saved run timer reads127.8minutes for piecewise and97.8minutes for merged (about1.31× elapsed). These include earlier reporting and use separate GPUs of the same type; they are not isolated kernel timings. Piecewise also used more memory in the bounded production fits. The inspection found substantial side-encoder padding. The strong32k-parent batching test rejected smaller budgets on output/gradient differences, although the unchanged repeat was exact. No throughput benefit was measured or adopted. The redundant older batching sweep was canceled; a15-minute forward-localization diagnostic is queued as23515979. Active training settings remain frozen. [Probe and status](PIECEWISE_BUDGET_PROBE.md). [Timing receipt](PIECEWISE_COST_40_PERCENT.json).

The larger run from the final32k-question checkpoint starts at **0.6760 AC NLL versus 0.6135 initial-reader full-context NLL** on 8k evidence. This supports using a stronger QA checkpoint for adaptation; it combines additional prior QA training and stronger initialization, so it is not an isolated initialization ablation. It is running as23514939:8,000 fresh questions×1,8k evidence,original batching, and reduced repeated-baseline reporting. The old matched comparison remains unchanged. See [measured training and reporting costs](PIECEWISE_TRAINING_COST_BREAKDOWN.md).
