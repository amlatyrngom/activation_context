# Golden AC Reconstruction Training

Frozen by the user for this workstream. Subsequent work must build on this setting; choose starting checkpoints for transfer compatibility. New proposals in the plan do not amend these decisions.

- Data: 50M reconstruction tokens from deduplicated tool outputs, code, documents, math and web text; hold out whole documents/trajectories.
- Models: Qwen3.5-2B encoder → 2B/4B/9B/27B readers; frozen bases, rank-128 LoRAs.
- Compression: cover 1/10–1/30, measured as rows/source tokens. No input/content pooling.
- Architecture: input rows + deep residuals at every full-attention layer; dedicated passage-attention readouts and zero-initialized heads.
- Training: gold-token SFT; encoder/heads LR 5e-5, reader LR 2e-5; AdamW, 8 examples/update, 50-update warmup, cosine decay to 10%, gradient clip 1.0. Preserve reader behavior with a no-context KL penalty: λ=1 on 25% of examples.
- Quality gate: held-out gold NLL <0.2 nats/token. Track training NLL, token accuracy, greedy reconstructions and no-context drift.
- Rationale: reconstruction must preserve every token; QA and trajectory compaction can select relevant information. Being the hardest task remains a hypothesis.

Unspecified in the records: exact ratio grid; whether 50M tokens is per model/ratio or across the sweep.

Implementation interpretation retained from existing canon: "no pooling" excludes input/content pooling, while the encoder still constructs compressed summary rows; the no-context reference is the original reader base, using stored top-32 plus remainder distributions. A trained head is loaded with its learned weights; "zero-initialized" describes the original parameterization, not resetting a trained head during fine-tuning.
