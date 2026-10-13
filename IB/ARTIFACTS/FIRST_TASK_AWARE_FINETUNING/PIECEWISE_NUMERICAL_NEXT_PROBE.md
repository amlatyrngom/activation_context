# Smallest useful next numerical probe

Keep the production encoder budget at 65,536. The failed 32,768/16,384 equivalence gates remain failures. One short **first-full-attention operator capture and reference replay** is warranted before choosing a precision setting; a global TF32 toggle is not yet justified.

The existing localization gives an unusually clean case. For the shortest passage (109 content + 4 compressed positions), the mixer, side-encoder input and layers 0–2 are bitwise identical across packing budgets. The first difference is the **whole layer 3 output**, relative L2 0.00317156. That hook includes projections, attention and the feed-forward network, so it does not isolate attention itself. The valid masks and positions match. Unchanged-batch repetition is bitwise exact.

The original side-encoder batch is `[24, 2507, 2048]`; 32,768 splits it into `[21, 1215, 2048]` and `[3, 2507, 2048]`. The longest passage already differs at the first mixer output (relative L2 0.00008971). A fix confined to full attention might therefore leave another source of numerical differences.

## What the source establishes

- Frozen `ac_model.py:494–514` runs mixer projections and local attention under **BF16 autocast**. Float32 residual/output tensors do not establish float32 GEMM execution. TF32 does not control those BF16 products.
- The Qwen side encoder receives BF16 embeddings outside that autocast scope. The PEFT projection path computes the base branch, casts its input to each adapter's weight dtype, adds the adapter branch, then casts to the base result dtype. Actual runtime adapter dtypes must be recorded: float32 adapter GEMMs can use TF32, while BF16 base GEMMs have separate precision behavior.
- Qwen layer 3 performs Q/K/V projections, Q/K RMS normalization, RoPE, a selected attention implementation, gated output projection, and an MLP. The whole-layer hook cannot identify which first differs. The frozen model loader does not explicitly select an attention implementation; the localization receipt did not record the resolved implementation or CUDA operator. Do not label it FlashAttention or SDPA based only on the Qwen class name.
- Final readouts/heads run per passage with the same shapes in both layouts. They can amplify earlier differences but cannot explain the observed first layer-3 difference.

Source anchors: frozen `TMP/FIRST_TASK_AWARE_FINETUNING/train_ws_night_v1_snapshot/activation/{ac_model/ac_model.py,ac_model/ac_model_utils.py,harness/loaded_model.py}`; installed `transformers/models/qwen3_5/modeling_qwen3_5.py:626–718`, which matches the recorded remote source SHA `67cf849081143a998f0e189c551e4b5f532365a055805570747385e400372abb`. Local PEFT source explains the branch dispatch, but its exact remote hash and runtime dtypes should be included in the next receipt.

## One proposed diagnostic

Reuse the exact parent, affected 24-passage microbatch, original masks/positions, grad-enabled checkpoint policy and baseline/repeat/32,768 sequence. Capture **only the shortest passage through layer 3**, while preserving the real batch shapes:

1. Record normalized layer input; Q/K/V base-branch and active-adapter branch outputs and dtypes; post-normalization/RoPE Q/K/V; attention output before gating; output projection; MLP output. Record the actual attention implementation and CUDA attention operator once. Compare valid prefixes, not padding.
2. Replay the captured shortest-passage Q/K/V through a small explicit float32, TF32-disabled causal attention reference, preserving the actual scaling, grouped-query mapping and valid mask. This is only 113 positions, so no full-batch quadratic FP32 allocation is needed. Compare each native valid attention output with its own reference and compare the two references with each other. The reference is a diagnostic calculation; it does not replace any model output or constitute a production timing.
3. If Q/K/V are equal but native attention differs, while the FP32 references agree, target attention accumulation/backend in the next equivalence attempt. If projections differ first, identify BF16 base versus float32 adapter branch; only the latter supports a targeted TF32-off hypothesis. If attention output is equal, the output projection/MLP captures locate the remaining cause. Do not change several precision controls together.

This is a single forward diagnostic, roughly a 5–10 minute allocation including model loading, with no optimizer, reader scoring, checkpoint save or new data. Retain parent payload rehashes and explicit `equivalence_not_tested` labeling. No strict gate is relaxed. A favorable result would identify a candidate intervention; smaller production batches would still require the existing full output/loss/all-gradient proof and ordinary-kernel speed/memory measurement. It is not yet evidence that one precision switch will make smaller batches trustworthy.

## Prepared worker

`bench/probe_piecewise_attention_operators_v1.py` and its same-name `.sbatch` are ready for review, **not staged or submitted**. The launcher requests one RTX PRO6000, five minutes, eight CPU cores and 64 GiB host memory. It uses the existing guarded frozen loader and exact case, stops native execution after layer 3, retains the actual captured tensors in `operator_tensors.pt`, then rehashes the parent payloads. Output folder: `reports/piecewise_attention_operators_v1`.

The small real Qwen3.5 CPU check passed against the pinned AC/trainer/utils: 38 captured stages, nonzero LoRA branches, exact unchanged repetition, changed batch shapes, grouped-query reference shape, three rejected invalid masks, and restored hooks/methods/attention registry. There is no reader forward, backward or parameter update. These are protocol checks, not evidence of GPU numerical equivalence. Hashes and scope are in `PIECEWISE_ATTENTION_OPERATORS_PREPARATION_V1.json`.
