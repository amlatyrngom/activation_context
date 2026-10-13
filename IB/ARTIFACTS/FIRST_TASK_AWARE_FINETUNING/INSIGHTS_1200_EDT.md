# Noon EDT checkpoint — September 22

Evidence checked at **15:57 UTC / 11:57am EDT**. The **32,000 unique questions × 1 epoch** run is healthy, about **27% complete**, with approximately **3h 22m remaining**. The original learning rates remain selected. The new piecewise experiment is preparing from the reconstruction checkpoint; 27B synthesis is paused at 53,000 published questions.

## Large-data comparison

These are the latest matched completed panels, after **6,400 question presentations**. Both runs still use previously unseen questions: the repeated-data part of the comparison begins in the companion's second epoch.

| Run | Completed quality panel | Development answer NLL | Learning-set answer NLL |
|---|---|---:|---:|
| 32,000 unique questions × 1 epoch | 32,000 × 0.2 epochs | **0.846** | **0.793** |
| 16,000 unique questions × 2 epochs | 16,000 × 0.4 epochs | 0.873 | 0.855 |

The larger pool's early lead has narrowed from 0.138 to **0.027 NLL**. Both generator cohorts favor it, but the descriptive paired-question interval crosses zero. This is one seed and repeated development monitoring; it does not yet answer unique data versus repeated epochs. The same original P2 pair, fresh optimizer, learning rates and 32k-presentation schedule are used. Equal question presentations do not imply equal tokens or runtime. Learning NLL uses the same fixed 32 training questions; development uses the same 400 questions.

ORCD preempted the **companion only**, at 11:40am EDT. Its published checkpoint retains 7,200 questions of progress, paired weights and optimizer state; the last 200 questions need replaying. The replacement allocation has loaded that checkpoint and is rebuilding its inputs. Original order and learning-rate schedule are preserved; downstream evaluations were requeued. The main 32k run remains uninterrupted. Its provisional finish is around **3:20pm EDT**, followed by the fixed-100 response readout. The companion's finish will be later because of the interruption.

## Learning rates and clipping

The completed learning-rate comparison favors **AC 5e-5 / reader 2e-5**. After 16,000 questions × 1 epoch, original versus half AC LR gives development NLL **0.603 versus 0.707** at 1/16, and **0.646 versus 0.778** at 1/8. The focused answer check supplies no reason to prefer half LR. [Decision and evidence](LR_DECISION.md).

Separately clipping AC and reader gradients remains promising at **1/8**: after **16,000 × 0.5 epochs**, development NLL is **0.768 versus 0.801** for the original joint clipping. Both generator cohorts improve. It has led at the last three completed panels, after starting worse. This changes the combined allowed gradient norm as well as the coupling between groups; it is not a pure reader-only intervention. Complete the pass and answer readout before adopting it. The large runs retain their agreed recipe.

## Piecewise encoding

The controlled comparison uses **4,000 unique 27B questions × 1 epoch**, **8k evidence with up to 16 passages**, and the original **P2 reconstruction pair with fresh QA optimizer**. One arm encodes each passage independently; the control merges them. Questions, full evidence, targets, total compressed rows at 1/16 and training recipe match. This is a first architecture comparison, not a data ceiling.

The existing encoder already batches passages. An optional mask/checkpoint optimization failed the real GPU numerical screen: AC outputs were close, but loss and many parameter gradients exceeded the preset equivalence limits. It is **not adopted**, and the limits are unchanged. Both arms will use the original implementation and existing batching budget; a short fit/timing run is queued before training. The piecewise arm uses the former synthesis slot; its control follows the clipping run and response readout. Final readouts include fixed-100 answers with frozen full/no-context baselines on the same 8k evidence, plus a 32k-evidence NLL transfer check.

## Operations

- **53,000** mixed 27B questions are published; accepted surplus and generation cursors are preserved. Generation and its autorequeue have stopped. The reusable 9B bank remains paused.
- Three training allocations are active; the fit/timing job can use the fourth autonomous slot. No EC2. User-reserved capacity remains available.
- Local free space is **256 GiB**; cached ORCD pool usage is **557/1024 GB**. No pruning is needed. Preservation flags and the newest approximately 250 GB remain protected.
- The overview HTML has remained unchanged since **10:40am EDT**; its JSON continues updating each minute. Actual extension rendering remains outside this environment's verification.

[Live overview](../../TMP/SYNC/TASK_AWARE_QA/report.tressoir.html) · [32k run](../../TMP/SYNC/TASK_AWARE_QA/oracle_27b_32k/report.tressoir.html) · [Matched-data design](DATA_AMOUNT_32K.md) · [Existing reviewed responses](MODEL_COMPARISON_COHORT100.tressoir.html).
