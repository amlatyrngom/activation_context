# 16:00 EDT checkpoint — September 22

The 32,000-question ×1-epoch run is complete. It improves development NLL over the earlier16,000-question run, but its inspected answers do not establish a strict-correctness gain. The matched16,000-question ×2-epoch control is still running. Two continuation schedules and the trained piecewise comparison will test the remaining explanations.

| Measurement | Earlier16k ×1 | Final32k ×1 |
|---|---:|---:|
| Held400 answer NLL |0.6033|0.5688|
| Correct on fixed87 supported references |66|63|
| Partial on those87 |5|10|
| Incorrect on those87 |16|14|

The earlier16k run had a shorter learning-rate schedule, so this is a practical checkpoint comparison, not an isolated data-amount effect. The fixed100 include50 questions from each generator;13 references have preexisting source concerns. All100 remain in NLL. The source review and historical annotations are unchanged. [Full100 answers and evidence](DATA_AMOUNT_COHORT100.tressoir.html).

There is no strong saturation conclusion. Held NLL fell through28,800 fresh questions to0.5581, then ended at0.5688 while the32-question fixed training panel improved. In that final interval208/400 development questions improved; larger regressions raised the mean. The descriptive paired-question95% interval for the mean change crosseszero. This could reflect optimization variability or diminishing gains; it does not prove overfitting or prove that more data will help. The final checkpoint was preselected and is retained; we did not replace it with the best intermediate panel.

At25,600 matched question presentations, fresh32k×0.8 gives0.5786 versus0.6074 for16k×1.6. The descriptive paired-question interval favors fresh questions at that boundary, but the full2-epoch control is unfinished and one seed remains a limitation. Equal question counts do not ensure equal tokens or runtime.

The learning curve is explicitly labeled32 fixed training questions. It is not a stream of fresh pre-update losses; a train/development gap cannot simply be called a sampling artifact. Original frozen controls appear in the answer report; current-reader controls in live training panels are labeled accordingly.

## Work continuing

- **Same10k fresh questions ×1, from final32k:** both arms inherit all Adam moments and per-parameter clocks. One holds encoder/reader LR at5e-6/2e-6. The other uses400-question warmup to2e-5/8e-6, then cosine decay to the same floor. The source mix, order, original400 development questions,32 fixed learning questions, evidence and preservation penalty are matched. Preparation passed; staging both arms to CSAILH200. No separate fresh-optimizer arm is requested.
- **Piecewise versus merged training:**4,000 unique questions ×1, original reconstruction pair,~8k evidence/up to16 passages and identical total compressed rows at1/16. Both ORCD training jobs and final8k-answer/32k-transfer evaluations are queued. The memory repair changes only the existing checkpointing threshold. All four measured training cases fit; controlled deterministic outputs and gradients match exactly. Whole-run memory safety and quality gains remain to be measured.
- **Matched16k ×2:** continues with its original optimizer/schedule after preemption. Final100 and reserved confirmation remain dependent on complete final checkpoints. Newly considered extensions do not inspect that reserved confirmation.

No new EC2 instances. Additional CSAIL parallelism is authorized, with room retained for supervised requests. The final32k pair is verified in S3; its optimizer remains available on ORCD. Other future task-aware work remains planning-only until the user supplies the plan here.

Live reports: `IB/TMP/SYNC/TASK_AWARE_QA/`. Exact job dependencies and frozen experiment identities: [EVALUATION_QUEUE.json](EVALUATION_QUEUE.json).
