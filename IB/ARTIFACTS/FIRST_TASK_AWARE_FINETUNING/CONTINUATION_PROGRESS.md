# Continuing the32k checkpoint with10k fresh questions

**Constant final LR is complete: held-out answer NLL0.5688→0.5327 after10,000 fresh questions once.** Its source-reviewed100-answer panel improves63→71/87fullycorrect, with9gains and1loss; frozenfullcontext78/87. The rewarming arm is now also complete: **0.5248 held NLL**, below constant LR’s **0.5327**; its actual100-answer review is pending.

Both completed arms keep the original optimizer state and use the same fresh questions once. One holds the final learning rate; the other briefly warms it before decaying back to the same floor. These are repeated development measurements, not an independent confirmation.

| Fresh questions seen once | Constant final LR | Rewarm |
|---:|---:|---:|
| 0 | 0.5688 | 0.5688 |
| 1,000 | 0.5520 | 0.5598 |
| 2,000 | 0.5385 | 0.5725 |
| 3,000 | 0.5509 | 0.5726 |
| 4,000 | 0.5367 | 0.5684 |
| 5,000 | 0.5321 | 0.5608 |
| 6,000 | 0.5424 | 0.5573 |
| 7,000 | 0.5302 | 0.5465 |
| 8,000 | 0.5290 | 0.5336 |
| 9,000 | 0.5290 | 0.5395 |
| 10,000 | 0.5327 | 0.5248 |

The constant arm used77.6minutes of training and29.9minutes of in-training reporting; trainer duration122.2minutes including other work. Its fixed32-question learning-panel NLL improves0.5970→0.5186; this small panel is not the full training loss. That new-stage panel differs from the original32k run’s panel.

Both initial400-question replays exactly matched the parent. Rewarm has been preempted twice and resumes saved paired weights, optimizer and original schedule. Production backward calculations are not bitwise identical; the audit found no confirmed state or schedule defect.

[Inspect all100 before/after answers](CONTINUATION_COHORT100.tressoir.html). One run per arm and these nonmonotonic curves do not establish saturation.
