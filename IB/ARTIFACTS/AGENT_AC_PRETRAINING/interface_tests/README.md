# Full mainline test proposals

These are complete proposed files for the updated interfaces, prepared for review before the next stage. They are not applied to activation/tests and have not been run against product code: loss_kind, paired-history items, folder-local tensor persistence and utility loading are still planned changes.

- `test_basic_agent_ac_training.py` is the full revised existing public-data AC training test, retaining its three real-data cases.
- `test_basic_ac_self_distillation.py` is a complete four-problem HotpotQA example with a custom AgenticProgram, actual tools/model rollouts, KL/SFT training, paired checkpoint reload, fresh AC rollouts and saved-row/cache replay.

Only these two test files belong in this work's eventual mainline test delta. They contain no mocks or synthetic trajectories. Private numerical, failure-injection and synthetic tests stay in IB/TMP; their code is not part of this review. Existing unrelated tests are not deleted.

The self-distillation example chooses four short questions labeled easy from a bounded train-split prefix, using question metadata rather than model success. It runs one epoch/two updates per loss kind. Its explicit selection="all" for SFT exercises the complete four-problem pipeline even when all scored rollouts are wrong; successful-only SFT defaults are checked privately. Scores are never fabricated, and same-question post-training scores are diagnostics, not held-out quality evidence. The fixed-row AgentTrainer loss/gradient proof is private and does not depend on an accuracy increase in these mainline examples.
