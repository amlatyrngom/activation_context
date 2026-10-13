# Continue the final32k checkpoint on10k fresh questions

Approved in chat on September22. Compare two learning-rate schedules while preserving the final32k encoder/reader weights and every Adam state. Both runs use the same10,000 fresh27B-synthesized questions for one epoch. This is an extension of the completed run; it is not equivalent to a42k-question schedule planned from the beginning.

| Arm | Encoder / reader LR | Schedule |
|---|---|---|
| `continue32k_10k_final_lr` |5e-6 /2e-6|Constant throughout the new10k questions; no warmup.|
| `continue32k_10k_rewarm` |Peak2e-5 /8e-6|Standard warmup over400 questions, then cosine decay to25% of peak: approximately5e-6 /2e-6 at completion.|

The second arm warms from nearzero using the existing scheduler. Its peak is40% of the original32k run’s peak, four times the final floor. All other optimizer hyperparameters, joint clipping, source mix, question order,1/16 compression,~2k evidence/up to4 passages, reader/encoder architecture and original-base no-context penalty are matched. Both use single CSAILH200 GPUs. Report final checkpoints; development panels do not select a substitute.

**Optimizer continuity:** all615 populated Adam parameter states, moments and parameter clocks remain unchanged at initialization. Only the new stage’s reporting/schedule counters start atzero. The original parent checkpoint is untouched. A separate CPU test exercised the frozen trainer’s actual restore branch and confirmed that the next Adam parameter clock advances from4000 to4001. Both LR arms have identical updated moments under identical gradients; a fresh optimizer gives a different weight update.

**Data:** code3000, arXiv1200, textbooks800, financial500, web2000, technical500, documents800, legal600, office600. Freshness excludes the parent’s32k question IDs and normalized questions; reserved evaluation requests stay excluded under the existing split policy. Previously seen source documents can recur. Selection is independent of model losses or responses. No additional synthesis or native benchmark questions are needed.

**Reporting:** fixed400 development questions and unchanged retrieval inputs; one quality panel per1000 new questions, plus initialization. Learning NLL uses32 fixed questions from the new10k pool. Initial input fingerprints must match the parent’s full400; any H200-versus-RTX numerical baseline difference is recorded explicitly. Live full/no-context panels use the current reader and remain labeled as such; the source-grounded answer report retains the original frozen controls. Final100 greedy answers use the existing fixed selection, with NLL replay before decoding. No new confirmation set is opened for this comparison.

The live report folders on this machine will be:

- `IB/TMP/SYNC/TASK_AWARE_QA/continue32k_10k_final_lr/`
- `IB/TMP/SYNC/TASK_AWARE_QA/continue32k_10k_rewarm/`

The overview stays at `IB/TMP/SYNC/TASK_AWARE_QA/report.tressoir.html`. Routine progress updates change JSON; the HTML shell changes only when report structure changes.

Preparation and transfer have passed: [data and inherited optimizer](CONTINUATION_10K_PREPARATION.md), [CSAIL verification](CONTINUATION_10K_CSAIL_RELOCATION.json). The frozen design and queue ledger record exact sources, initial payloads, schedules, final-checkpoint guards and job IDs. Own-stage preemption resumes the paired optimizer and stage schedule. Runtime estimates will use observed H200 pace; neither a flat low-LR result nor a transient rewarming regression alone establishes saturation.
