# Overnight insights —23Sep,03:00UTC

The clearest gain is more fresh data at the existing final learning rate. Continuing the32k checkpoint with10k new questions once, preserving Adam, improves **63→71/87 fully correct answers** on the fixed source-reviewed cohort:9 gains and1 regression. Frozen full context gets78/87. Held400 answer NLL improves0.5688→0.5327; all100 decoded answers end normally. This is the preselected final checkpoint, not the best intermediate panel. [Inspect every answer](CONTINUATION_COHORT100.tressoir.html).

The important final encoder/reader pair is now checksum-verified in S3, with its recipe and parent ancestry; optimizer and transient checkpoints are excluded. [Archive receipt](QA_CHECKPOINT_ARCHIVE_CONTINUATION_FINAL_LR_RECEIPT.json).

The learning-rate comparison remains open. Rewarming is halfway through the same10k fresh questions and has held NLL0.5608, versus the constant-rate arm’s0.5321 at5k. Two scheduler preemptions required saved-state resumes; state/order/schedule checks passed. Its final answers will decide the comparison.

The larger long-context adaptation has passed1,600 of8,000 fresh questions. Held NLL0.6760→0.6397→0.6398 at0/800/1,600, against frozen starting-reader full-context0.6135. Initial gains hold but have not grown at the second panel. The starting32k checkpoint’s actual8k-context answers are58/87correct, versus34/87for the smaller reconstruction-start passagewise run. These are distinct pre-adaptation and trained systems. [Long-context answer comparison](STRONG_LONG_COHORT100.tressoir.html).

At the same starting checkpoint, independent-passage AC scores0.7019/0.7352/0.7537 on the same100 questions at2k/8k/32k evidence. Full-context NLL improves with more evidence. Only8k has generated answers so far. The native2k merged layout scores0.6079, so changing to independent passages itself has a cost before adaptation. [Matched context table](STRONG_LONG_INITIAL_CONTEXT_COMPARISON.md).

Reporting overhead has been reduced in the new long run by caching frozen controls and measuring moving controls less often. Smaller encoder batches remain unadopted: BF16 projection rounding explains part of the discrepancy, but the targeted precision trial still fails two output checks. No tolerance was relaxed and no speed gain is claimed. The merged comparator is resuming its last800 questions on a different node after repeated preemptions.

Next: complete the rewarm and merged comparisons; evaluate the newly improved42k checkpoint at8k context as a potentially better initialization; finish the8k-question long adaptation and its8k/32k actual-answer readouts. The three active trainings continue under the existing ORCD GPU limit, with native monitoring. No EC2 or synthesis generation is running. These remain repeated development experiments; reserved confirmation results are not used for tuning.
