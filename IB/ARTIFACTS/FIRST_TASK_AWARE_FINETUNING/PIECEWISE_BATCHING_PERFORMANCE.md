# Piecewise encoder: batching cost and next small test

**The encoder is already batched, but its65,536-token limit permits substantial padding waste.** The frozen V2 trainer collects every passage from each two-question microbatch into one `encode_batch` call. The encoder sorts these requests by content length and packs them up to its padded-token budget. Recorded microbatches contain8–32 passages in one side-encoder forward; there is no accidental one-forward-per-passage loop.

At1,600 questions seen once (40% of the4,000-question pass), piecewise development answer NLL is1.1314 versus merged1.3207. Trainer elapsed time at that checkpoint is127.8 versus97.8minutes, a1.31× difference on separate RTX PRO nodes. These times include preparation/reporting overhead. The callback’s reported “panel seconds” excludes the preceding AC evaluation and subsequent sample decoding, so subtracting it would not recover pure training time.

Two existing real training cases give the following evidence. Both use logical batches of8, reader microbatches of2, checkpoint threshold0, identical total compressed rows, original masks, and original frozen source.

| Same eight questions | Piecewise | Merged |
|---|---:|---:|
| Formerly failing batch: side-encoder useful/padded positions |66,523 /162,572|63,031 /64,286|
| Training-length quantiles: side-encoder useful/padded positions |69,358 /168,963|67,451 /69,134|
| Padding fraction |59% /59%|2.0% /2.4%|
| Measured steady seconds per8 questions |23.1 /24.1|14.1 /15.0|
| Measured peak GPU allocation |49.8 /52.3GiB|37.4 /37.5GiB|

The worst recorded piecewise microbatch pads10 sequences spanning107–6,559positions to65,590 positions;74% are padding. Sorting helps only when a budget boundary actually splits the group. The current limit often keeps short and long pieces together. It also counts content before appended summary rows, so the side encoder can slightly exceed the nominal budget. Across the complete4,000-question material, repeated query/framing adds only3.58% encoder input tokens; that alone does not explain the measured slowdown. Reader lengths and allocated row counts are identical in these paired cases.

The source also loops over pieces for summary construction and the nine readout/head paths. Those smaller calls add launch/graph overhead; their contribution is unmeasured. There was no phase profiler in these receipts, so padding is a concrete inefficiency, not a measured decomposition of all elapsed time. The side encoder, mixer and their backward recomputation all operate on the padded shapes. The original mask-removal proposal failed its numerical gate and remains unused.

Two bounded follow-ups, after the current runs:

1. **Test the existing encoder budget knob first:**65,536 versus32,768 and16,384, retaining checkpoint threshold0 and all current masks. Replaying the exact recorded geometry predicts27–31% fewer padded side-encoder positions at32,768 (six forwards instead of four), and37–39% fewer at16,384 (eight or nine forwards). These are shape calculations, not promised speedups. Keep the merged control at its measured suitable65,536 budget; lowering it adds calls with almost no padding benefit.
2. **Only if head time remains significant, concatenate the pointwise heads across pieces.** Keep each passage-attention readout separate, concatenate its row outputs, apply the shared target/delta head once per layer, then split in original order. `RowHead` and `DeltaHead` act independently on rows. This avoids changing attention masks, per-passage positions, parameter names or the model interface. It can reduce many small head launches; the speed benefit needs measurement. Broad batching of variable-length passage attention is unnecessary for this first test.

Use the two existing training cases and fixed penalty draws, starting from the completed piecewise pair so the trained readouts are exercised. First compare outputs, joint loss and all trainable gradients with the established deterministic diagnostic and **unchanged** strict gates (loss absolute≤0.001; gradient/output RMS3% with2e-6 floor). Include an unchanged repeat. Then separately time ordinary production kernels for one warmup plus three real SFT/penalty updates, recording padding, calls and peak memory. Reload the same paired weights/optimizer between variants. No training-arm change follows from shape estimates alone, and no held questions are needed for this speed choice.

Source: frozen `ac_model_training.py:1140` batches all requests; `ac_model.py:419` sorts/packs; `ac_model.py:465` pads/mixes/encodes; `ac_model.py:535` loops over readouts. Measured inputs and reproducible packing calculations are retained in [the evidence JSON](PIECEWISE_BATCHING_PERFORMANCE.json). No active source/design or GPU jobs were changed.
