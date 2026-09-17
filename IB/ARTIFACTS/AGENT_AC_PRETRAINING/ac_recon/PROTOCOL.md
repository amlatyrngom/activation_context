# AC reconstruction round — frozen evaluation protocol

Written before any run of the round started (2026-09-15, 23:50 UTC). Not changed afterwards; deviations are recorded in the ledger.

**Task.** Qwen3.5-4B (side and reader share the base; separate LoRAs, rank 64) sees a passage only through its activation-context rows and reproduces it word for word (SFT on the passage tokens). Cued-continuation items, where used, keep the first 20 % of the passage in plain text and target the rest.

**Data.** Tool outputs (file views, test logs, fetched pages) from OpenSWE and S1 deep-research trajectories, windows of 400–2000 characters, deduplicated, split by trajectory so no held passage shares a trajectory with a training passage. Training pool up to 40k windows. Held-out panels, fixed for the round:
- **novel**: 200 windows from held trajectories, never trained on by any run;
- **wiki**: 100 NQ passages (the RAG round's held passages), the memorization detector: text the base model has likely seen in pretraining.

**Metrics** (all teacher-forced, at every panel; panel every 1/8 epoch for anchors):
- gold NLL: reconstruction cross-entropy per passage token, novel and wiki separately;
- token accuracy: share of target positions whose argmax is the gold token;
- no-context floor: the current reader's NLL on the novel panel with the rows removed (the prior); the penalty monitor;
- seen-vs-unseen gap: NLL on 512 training passages after the last epoch versus the novel panel;
- fidelity: word-level difflib similarity of greedy reconstructions on 8 fixed novel items per epoch (qualitative, small sample).

**Score used to promote a change**: novel-panel gold NLL at the end of training, with the no-context floor unchanged from the base (drift ≤ 0.02 nats) and the wiki panel not improving faster than the novel panel. Ties broken by token accuracy. Nothing is promoted on training loss or on the wiki panel.

**Fixed across runs**: seed 20260915, the shared initial adapters (hash c2f0d3…), examples per update 8, warm-up 10 updates, gradient clipping 1.0, the panels and their item order.

**Anchors**: A16 (ratio 1/16) and A4 (ratio 1/4): SFT, encoder LR 1e-4, reader LR 2e-5, penalty λ = 1 on 25 % of examples, 2 epochs over the size the smoke run's budget rule selects.


**Amendment 00:37 UTC 09-16.** The anchors run from source tag `ac-recon-20260916T002847Z` (the 000008Z tag plus the batched SFT path, unused by the anchors at `micro_batch_examples=1`, plus the base-model no-context floor measurement in the bench). Reason: the first launch from the 000008Z root failed the bench manifest check after the bench gained the measurement.
