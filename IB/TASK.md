______ Frozen Canonical Section ____
This section contains canonical decisions. We'll assume all subsequent work should be able to build on top off this.
This means that when selecting checkpoints to work with, we should pick things that will transfer well to checkpoints trained under this setting.

# AC Reconstruction Training
- Data: 50M reconstruction tokens from deduplicated tool outputs, code, documents, math and web text; hold out whole documents/trajectories.
- Models: Qwen3.5-2B encoder → 2B/4B/9B/27B readers; frozen bases, rank-128 LoRAs.
- Compression: cover 1/10–1/30, measured as rows/source tokens. No input/content pooling.
- Architecture: input rows + deep residuals at every full-attention layer; dedicated passage-attention readouts and zero-initialized heads.
- Training: gold-token SFT; encoder/heads LR 5e-5, reader LR 2e-5; AdamW, 8 examples/update, 50-update warmup, cosine decay to 10%, gradient clip 1.0. Preserve reader behavior with a no-context KL penalty: λ=1 on 25% of examples.
- Quality gate: held-out gold NLL <0.2 nats/token. Track training NLL, token accuracy, greedy reconstructions and no-context drift.
- Rationale: reconstruction must preserve every token; QA and trajectory compaction can select relevant information. Being the hardest task remains a hypothesis.

Unspecified in the records: exact ratio grid; whether 50M tokens is per model/ratio or across the sweep.
___________________________

New thread of work. I want to downgrade all the experiments we are currently running to the lowest priority, so that we can push for the paper. From now, if I ask you micro-optimize something, give gentle warning that I should be focusing on the paper, not on pushing numbers by 1-2%. We need relentless focus towards completing an MVP for the paper as quickly as possible. We will be, as much as possible, using 2xH200/GH200.

Take a look at IB/evaluation_section_snapshot.tex for my current snapshot. Commit and push your code right now to a branch. (my github credentials should be in .env).

Open up a new IB Artifacts folder, and make a .tressoir.md plan for the compaction work that I'll review. The first version should be intent heavy. The key beats should be:
- Train/Test dataset selection. Task limits, etc.
- Trajectory Syntheses. 27B? 9B?
- Training targets (should be clear from the aspirational writing).
- Plots. This is important to avoid deviating from the paper. The terminal goal is to produce plots that go in the paper. Result Plots in the main text. Maybe appendix plots in the target