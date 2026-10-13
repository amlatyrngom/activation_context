# Fresh questions versus a second epoch

Compare **32,000 unique 27B-generated questions × 1 epoch** with **16,000 original questions × 2 epochs**. The second pool is an unchanged subset of the first. This tests whether more unique supervision helps beyond the same number of question presentations.

Both begin from the original P2 pair with fresh QA optimizers. Use the selected **AC LR 5e-5, reader LR 2e-5 and 1/16 compression**, a 2B encoder and 9B reader, 8 examples per update, 50-update warmup and cosine decay over all 32,000 presentations. The same 400 development questions, frozen distractor index and roughly 2k evidence budget are retained. Each arm has its own fixed32-question learning panel, selected from its training pool; those learning scores are not paired across the two pools. One train call per run; report every 10%.

The requested 32k run starts first. The companion uses another available MIT slot after the main allocation, within four autonomous GPUs including synthesis. No EC2. It takes priority over the optional group-clipping experiment. Both use the existing private runner; no product-code change is required.

Compare the fixed final pairs, their full/no-context reader controls and generated answers. The fresh confirmation reserved with the 32k material stays unused until both final pairs are fixed. Also report actual answer/source tokens and time: equal question presentations are not necessarily equal compute. This is one seed and one selected schedule, not a proof of convergence.

Main32k job **23479433** is allocated after an input-only [Unicode assertion repair](ANSWER_MASK_REPAIR.md); first attempt23478943 never reached training. Companion **23478982** continues unchanged. The [LR decision](LR_DECISION.md) and [evaluation design](DATA_AMOUNT_32K_EVALUATION_V2.json) are frozen. Exact settings and hashes: [DATA_AMOUNT_32K.json](DATA_AMOUNT_32K.json).

Both launchers accept an optional `--resume` after the run name, AC learning rate and compression ratio. This uses the existing paired optimizer/order/schedule recovery; it does not start a new training schedule or reset the optimizer. No other trailing recipe overrides are accepted. Shell syntax and eight parser checks passed; launchers staged on ORCD at13:05 UTC.

At the90% development panel (22September20:30UTC), **28,800fresh questions ×1** gives **0.5581NLL**, versus **16,000questions ×1.8** at **0.6067**. Both generator cohorts favorfresh;210/400 questions have lower NLL. The descriptive paired-question95% interval for the difference is[−0.0770,−0.0222], excluding source clustering and seed variation. This is an interim repeated development measurement; use the preselected complete final pairs and answer review for the final comparison.
