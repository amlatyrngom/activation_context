# BF16 reduction helped, but smaller batches remain unvalidated

**Stop this precision-probing avenue for now. Keep the original 65,536 encoder budget and production flags.** The predeclared stopping condition was reached: selected forward failures remain with BF16 reduced-precision reduction disabled.

The flag removed the shortest passage's batching differences entirely. For the longest passage, both mixer outputs and the side-encoder input became bitwise identical, so it addressed a real source of numerical differences. Both unchanged repeats were exact; masks/positions matched, and precision settings and parent payloads were restored.

The longest passage still first differs at the output of side-encoder layer 0: 0.0138% relative L2 there, growing to 2.0748% at final normalization. Two final deep-delta outputs fail the unchanged 3% gate: **layer 11 at 3.1722% and layer 19 at 3.1150%**. These compare 32,768 against 65,536 with the reduction flag disabled in both. Comparing disabled/32,768 against the original enabled/65,536 happens to pass all 18 selected final tensors (maximum 2.8149%); that is a different comparison and cannot substitute for the failed packing gate. There was no loss, backward, optimizer or speed test.

There is no measured operator-level basis for another specific precision change. Qwen's layer 0 includes base/LoRA projections, causal convolution, recurrent gated-delta attention, gated normalization and an MLP. Float32 LoRA GEMMs could be TF32-sensitive, but their layer-0 outputs were not captured. Earlier evidence that the shortest passage's layer-3 LoRA branches agree does not establish what happens in the longest passage's layer 0. Disabling BF16 reductions also does not guarantee shape-invariant GEMMs. Selecting TF32, convolution or FLA next would therefore be another hypothesis, not an evidence-backed repair.

This is useful attribution, not equivalence or a reason to change a live run. Preserve the failed receipts and return effort to the ongoing training/readouts. Revisit batching only if it remains a priority with a separately bounded question; no additional probe or implementation is prepared here.

[Measured receipt](PIECEWISE_BF16_REDUCTION_RESULT_V1.json), from GPU job 23522375.
