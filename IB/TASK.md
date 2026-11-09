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
- Plots. This is important to avoid deviating from the paper. The terminal goal is to produce plots that go in the paper. Result Plots in the main text. Maybe appendix plots in the target.
---------------------------------------

▎ We evaluate on HotpotQA (Yang et al., 2018). Following IRCoT (Trivedi et al., 2023), we pool every paragraph from HotpotQA's distractor-setting contexts (train and validation) into one retrieval corpus of 509,315 passages, and retrieve with BM25 using the question alone. Our test set is 500 questions sampled at random from the HotpotQA validation set (424 bridge, 76 comparison). It is disjoint from the 200 validation questions we used for development, and from all training questions, which come from the HotpotQA train split. We report the mean of 3 sampled passes, with thinking off for all readers.

I am not sure I like the first 3-4 sentences. Can you propose some 2-3 variant for me.
Agent harnesses communicate with models via text or audiovisual channels.
....
Current harness/model co-design does not optimize this channel, leading to context representation inefficiencies. 


This part is largely good: I've made some small edits.
We propose co-design at the architectural level: the harness declares specialized sub-models that turn arbitrary information into Activation Context: latent values written into the model as input-layer embeddings, hidden-layer residuals or directly ingested key-value (KV) pairs. Previously elided or summarized information can then be efficiently made visible to the model without overflowing its context window. We evaluate Activation Contexts on four workloads. In retrieval question answering, it matches or exceeds text accuracy with 9-19x fewer tokens. In corpus compression on LongHealth, it reaches 81.8% accuracy versus xx for text with 10x fewer tokens, 69% for Cartridges at the same size with 43x fewer training tokens. When resuming compacted agent trajectories, it preserves nearly as much information as full context on hardest takss while compaction 5.7x faster than text. In multi-agent synthesis, it lets an ... the most difficult DeepMatch-103K problems.





# Approach Section First Pass
the interface ..,., the different write modes, the AC encoders training, and the inference engine serving...
(the repeating "hows" feel too "mechanical").

The completions interface does mention the actual completion interface at all the typing part is underdevelopped. I'd dedicate listing 1 more this.
Listing 1: unreadble, weird line breaks, too many details.
- Move the declare call to the listing 2. Maybe with higher level Engine class that has high-level interfaces.
- then, make the shown trajectories way more user friendly:
    - The easiest thing is likely a before and after showing either:
        - minimal: text with top 20 vs AC with top200.
        - more detailed: compaction text + top 20 text, vs the two ACs.
    - It also lacks a bit of an agentic feel.
    - Smoething likes list client.chat.completions.create(...) can be elided in two ways:
        - Put ... liberally in the arguments.
        - Make the sure the key things we're showing are in the messages declaration.
    - The format should be nicer, like json, etc.
        - It listings is the wrong approach, I am fine with nice html+css -> pdf figures.

After making the background a little better, the last paragraph should somehow move before the math, you know what I mean? The current thing misreads somewhat.

"Harness-Facing Interface" is a bit too vague (would cover the completions interface). Maybe the "Activation Context Interface" or the "Activation Context Harness Interface" or something similar.

This section begins with a reference back to listing 1, with a focus on how the interface changes to add the `activation_context` type. Then, it dicusses the encoder.
V is reused for the kvs and the V.
I'd try to make this paragraph flow better and be split in two. Also make it introduce the notion of "reader" more clearly as the target model.

In table 1, remove Encoder constraint.
Reader pass over P. I'd do Reader Forward over V or Reader Prefill over V or something similar.
Remove serving.
There is missing conversaion on uniform vs non-uniform KVs. Might just belong to the text as an aside. E.g., we'd have to say the dimensions are averages.

confusions of "V positions P". Are both of these necessary?
I'd try to make the paragraph here flow better.

note: stopeed reading at the beginning of 2.3. Only skimmed the rest so far. Improve, then we go back.