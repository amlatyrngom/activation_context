# Fresh questions versus repeat training

**For the same32,000 question presentations, fresh questions give lower held-out NLL:**32,000 unique questions ×1 epoch scores0.5688, versus16,000 ×2 epochs at0.6017. Both full training runs are complete. Final answer review and a separate reserved evaluation are pending.

| Final training recipe | All400 held questions | 9B-generated200 |27B-generated200 |
|---|---:|---:|---:|
|32,000 unique ×1 |0.5688|0.6059|0.5317|
|16,000 unique ×2 |0.6017|0.6470|0.5564|

Both start from the sameP2 reconstruction pair with fresh QA optimizers,2B encoder→9B reader,1/16 compression, roughly2k evidence, the same learning-rate schedule over32,000 presentations, and the same fixed development questions. The16k pool is a subset of the32k pool. Repeated training was recovered from paired optimizer checkpoints after two preemptions; the original schedule was preserved.

Fresh data is better on218/400 questions, with a mean difference of−0.0329nats/answer token. A descriptive paired-question95% interval is[−0.0619,−0.0049]. Both generator-cohort means favorfresh, although their separate intervals crosszero. This is one seed on a repeatedly examined development set; those intervals omit source clustering and seed variation.

**Working recommendation:** prefer additional fresh questions over a second pass when a sufficiently good mixed bank is already available. This supports a32k×1 shape for a larger run, not a claim that32k is enough for convergence. The queued32k→fresh10k continuations will test remaining gains and how to extend the learning-rate schedule.

NLL alone does not establish an answer-quality gain. The already published32k review gave63/87 fully correct versus66/87 for the earlier16k×1 run with a shorter schedule. That earlier comparison is not matched for presentations. The matched16k×2 final100 answers are queued and will be added to [the shared answer report](DATA_AMOUNT_COHORT100.tressoir.html).

Each run has its own32 fixed training questions. Their learning NLLs are0.3972 forfresh32k and0.3117 for16k×2; these are not paired panels and should not be interpreted as a controlled estimate of the generalization gap. Whole-run compute accounting must also distinguish resumed-attempt counters from total training.

[Exact final panels and paired comparison](DATA_AMOUNT_MATCHED_FINAL.json). Reserved evaluation is kept separate from tuning and will be reported as confirmation, not used to choose a continuation schedule.
