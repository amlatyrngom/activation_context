# Overnight insights — 23 September, 07:00 UTC

Continuing at the existing final learning rate remains the best supported choice. The 32,000-question checkpoint plus 10,000 fresh questions × 1 epoch gives **71/87 fully correct answers**, versus **62/87** after rewarming on the same questions. The constant-rate model also wins when both are evaluated at 8k evidence: **61/87 versus 54/87**. These are source-reviewed answers from the predetermined final checkpoints. [Native answers](CONTINUATION_COHORT100.tressoir.html) · [Long-context answers](STRONG_LONG_COHORT100.tressoir.html).

The rewarm model nevertheless has slightly better held-400 NLL, 0.5248 versus 0.5327. On the selected 100, its NLL is worse, 0.5609 versus 0.5544; the other 300 account for the aggregate reversal. Eight of its eleven semantic regressions also worsen NLL, while three improve NLL. This is a useful warning against choosing a schedule from one aggregate alone. It does not establish overfitting or a population-wide semantic ranking. [Paired analysis](CONTINUATION_NLL_SEMANTICS_V1.md).

The constant-rate continuation has now reached **50,000 unique questions** by adding another 8,000 fresh questions × 1 epoch. Training completed in **92m 14s wall time**, with 62.4 minutes of actual training and 15.6 minutes of in-epoch reporting. Held-400 NLL improves **0.5327 → 0.5271**, a smaller gain than the earlier 32k → 42k continuation. The new fixed training sample improves **0.4403 → 0.3603**. There were no dropped examples. Its final 100-answer evaluation is still queued; semantic gains and saturation remain unresolved. No lower intermediate panel was selected. [Full learning curve](FRESH_NATIVE_PROGRESS.md).

The matched layout experiment is now complete: from the same reconstruction checkpoint, **4,000 questions × 1 epoch at 8k evidence** gives **34/87 correct with independently encoded passages versus 18/87 when merged**. Held-400 NLL is 0.9165 versus 1.3029. Passagewise encoding is the better tested layout in this setup, although both benefit substantially from a stronger task-trained starting checkpoint. [Every answer](PIECEWISE_COHORT100.tressoir.html).

The stronger long-context runs are progressing:

| Starting checkpoint and added training | Last complete panel | Held-400 NLL, initial → latest | Fixed training-32 NLL, initial → latest |
|---|---:|---:|---:|
| Original 32k; 8,000 fresh questions × 1 epoch, fresh Adam | 6,400/8,000 questions | 0.6760 → **0.6137** | 0.6492 → **0.6122** |
| Constant-rate 42k; 4,000 familiar questions × 1 epoch, inherited Adam | 1,200/4,000 questions | 0.6219 → **0.6076** | 0.6013 → **0.5365** |

Both use 8k evidence, independent passages, retrieved distractors and compression 1/16. The first still improves late in its pass; the second fluctuates between panels. Their starting weights, data familiarity and optimizer policies differ, so this is practical checkpoint progress, not a controlled schedule comparison. Final answers are pending. At 06:59 UTC the training estimates were roughly 1h 16m and 2h 33m remaining, followed by evaluations; scheduler waits can extend those times.

Without further training, the constant-rate 42k checkpoint retains most of its reviewed accuracy at **32k evidence: 58/87 versus 61/87 at 8k**, with four gains and seven regressions. Selected-100 AC NLL changes 0.6567 → 0.6785; its own full-context reader improves 0.5997 → 0.5734. All 100 AC answers stop normally. The report now has seven views and separately displays the actual 32k evidence and row counts. More retrieved context primarily tests distractor robustness here; it is not a distributed-evidence reasoning experiment. [Result and limits](CONSTANT42K_32K_QUALITATIVE_RESULT_V1.md).

Optimization work preserves the current encoder batching and precision. A new, isolated probe will test skipping reader recomputation while retaining encoder checkpointing. Tiny CPU outputs and gradients match exactly; the real GPU proof must pass before timing or adoption. Its 15-minute allocation follows the native 100-answer evaluation. That evaluation's pending reservation was shortened from 20 to 12 minutes using approximately four-minute completed sibling measurements; its scientific settings are unchanged. [Optimization candidate](READER_CHECKPOINT_OPPORTUNITY_V1.md).

Next: review the 50k native answers, finish both long-context runs and their final answer evaluations, then run the prepared context-transfer checks if their final answer quality warrants them. The original 32k and selected constant-rate 42k final pairs are verified in S3. Two ORCD training GPUs remain active, with no EC2 spend or synthesis generation. Native monitoring and stable-HTML report synchronization remain active. These are repeatedly inspected development experiments; reserved confirmation results remain excluded from hillclimbing.

Observed through 2026-09-23 06:59 UTC. New 50k and long-adaptation semantic results were unavailable at publication.
