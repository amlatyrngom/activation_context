# Long-context cost and the next optimization

The current 8,000-question, 8k-evidence proposal has approximately the same useful encoder-token volume as the completed 32,000-question, 2k-evidence run. It does not currently have the same execution cost.

| Actual training logs | Mean seconds per eight questions | Pure training time at the stated data size |
|---|---:|---:|
| Original merged, 32k questions × 2k evidence | 3.73 | 4.15 hours, measured complete |
| Current merged, 8k evidence | 16.58 | 4.61 hours for 8k questions, extrapolated |
| Current passage pieces, 8k evidence | 22.62 | 6.28 hours for 8k questions, extrapolated |

The long-context means cover 3,200 merged and 2,688 piecewise training questions so far. Different nodes and data pools prevent interpreting this as an isolated causal benchmark. The original trainer's logged times include the actual forward/backward/optimizer work and exclude report evaluation and checkpoint serialization; they are rounded to 0.1 seconds. [Parsed measurements](PIECEWISE_TRAINING_COST_BREAKDOWN.json).

At 2,400 piecewise questions, the trainer had spent 112.96 minutes on updates and 187.97 minutes overall before that panel's additional callback: about 75 minutes on preparation, prior reporting, saves, and the current AC evaluation. The callback's approximately six-minute timer covers full-context/no-context/learning evaluations only; it omits the preceding AC evaluation and later example generation. It must not be presented as total panel overhead. Extrapolating the current recipe to 8k questions therefore plausibly reaches 8–10 hours, but that is an estimate of the present implementation.

Two recorded, matched training cases expose substantial waste: about 59% of passage-piece encoder positions are padding versus 2% when merged. The existing 65,536-token batching budget keeps widely different passage lengths together. Repeated question framing adds only 3.58% useful input tokens. There is already a batched encoder call, rather than a separate model forward for every passage.

The OOM repair also changed activation checkpointing from a 1,700-token threshold to always on. Its contribution is not measured. The reader's gate counts batch size × sequence length, so three of four reader microbatches in one recorded long-context case would already have checkpointed at the old threshold; this cannot explain the whole slowdown.

The proposed 65k/32k/16k encoder-budget comparison ran from the completed 32k pair with fixed examples, masks, penalty draws and optimizer state. Smaller budgets failed the output/gradient gate despite nearly unchanged scalar losses; timing did not run. The larger adaptation therefore retains the original 65k batching. A separate forward-localization diagnostic is queued to find the first differing operation. See [failed comparison](STRONG_PIECEWISE_BUDGET_FAILURE.md).

The larger adaptation now keeps 10% AC and fixed-training feedback, caches full/no-context controls from the initial reader, and measures moving-reader full/no-context diagnostics only at the start, midpoint and end. The cache is tied to the initial checkpoint and exact evaluation inputs, and resume/report checks passed. Actual sample responses and final comparisons remain. This removes repeated control evaluations; no training-kernel speed improvement has been established.

The completed 4,000-question passagewise run gives a full-run measurement: **3.14 hours of training, 1.85 hours of in-training reporting, 12.3 minutes of initial evaluation**, and 5h19m Slurm wall time. Peak GPU allocation was 53.57 GB. This confirms that repeated reporting was a substantial part of the earlier runtime estimate. The larger run reduces those repeated controls; its actual total saving remains to be measured. [Completed-run timing](PIECEWISE_FINAL_TRAINING.json).
