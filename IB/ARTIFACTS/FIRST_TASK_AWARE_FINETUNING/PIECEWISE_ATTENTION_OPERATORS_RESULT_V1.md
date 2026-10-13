# The first difference is in BF16 base projections

The 117-second RTX diagnostic found the first changed operator at layer 3's **base key projection**, before full attention. The selected passage's input, normalized input, query projection and all Q/K/V LoRA branches are bitwise identical across packing budgets. Unchanged-batch repetition is exact, and masks, positions and scaling match.

| Captured value | Relative L2 difference, 32,768 versus 65,536 |
|---|---:|
| Base key projection | 0.1812% |
| Base value projection | 0.1950% |
| Float32 Q/K/V LoRA branches | 0 exactly |
| Native attention output | 0.2443% |
| FP32 reference attention on each captured Q/K/V | 0.1498% |
| Whole layer output | 0.3172% |

The base weights and outputs are BF16; LoRA weights are float32. The resolved attention implementation is SDPA, with the same efficient CUTLASS BF16 CUDA kernel in every variant. The FP32 attention references differ because their captured K/V inputs already differ. This identifies an upstream BF16 projection difference; it does not establish that attention adds no further rounding. TF32 was enabled, but it is not the direct explanation for the first BF16 base-projection difference when the corresponding float32 adapter branches agree.

**One targeted follow-up is warranted:** disable `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`, preserving every other setting. The receipt confirms it was enabled. This can affect the BF16 base GEMMs and the mixer's autocast BF16 GEMMs together. It is a plausible test of accumulation precision, not a guarantee: final BF16 rounding and shape-dependent algorithm choices remain, and the flag does not govern every attention/recurrent kernel.

The prepared five-minute worker `bench/probe_piecewise_bf16_reduction_v1.py` reuses the existing full-forward localizer. It runs enabled and disabled reduction, each at original budget, unchanged repeat and 32,768; records the same shortest/longest passage stages; and compares disabled-policy final rows against the original-policy baseline. No reader forward, backward, optimizer update or checkpoint write occurs. Parent files are rehashed afterward. In installed PyTorch 2.13, assigning a bare boolean also resets the separate split-K option to `True`; the worker uses `(False, original_split_k)` and restores both options, including on exceptions. TF32 stays unchanged.

Keep the original budget until the unchanged full loss/output/all-gradient gates and ordinary performance measurement pass. If the flag does not resolve the selected forward failures, stop this flag avenue rather than relaxing thresholds or trying a series of unrelated precision changes. Passing these selected forward comparisons would only justify the existing full proof; it would not authorize deployment.

[Measured receipt](PIECEWISE_ATTENTION_OPERATORS_RESULT_V1.json). The original failed packing receipts and current runs are unchanged. The follow-up is prepared locally, not staged or submitted.
