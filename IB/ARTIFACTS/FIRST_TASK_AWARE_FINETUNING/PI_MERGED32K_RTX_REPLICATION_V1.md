The completed RTX monolithic-32k readout reproduces the L40S result's **66/87 correct supported answers**, with no paired correctness changes. It also has the same 4 partial and 17 incorrect supported answers. This independently completed allocation used the same constant-floor 42k paired checkpoint; there was no additional training.

| Monolithic AC, same 100 questions | RTX PRO 6000 | L40S |
|---|---:|---:|
| Gold-answer NLL | 0.630480 | 0.631402 |
| Correct /87 supported | 66 | 66 |
| EOS termination /100 | 100 | 100 |
| Token-cap hits /100 | 0 | 0 |
| Nonempty text after answer wrapper | 0 | 0 |
| AC scoring seconds | 152.06 | 277.90 |
| Peak allocated GPU memory, GiB | 28.30 | 28.27 |

Eighty-nine complete replies match byte for byte. The 11 changed replies were checked against their actual supplied sources; most are paraphrases. Both versions miss the event sum at index 57: the source gives 228.1 + 66.3 + 39.4 = 333.8, while RTX answers 262.1 and L40S answers 62.5. At index 89, RTX additionally invents a two-year timing premise despite the source's more-than-three-and-a-half years. That reply is incorrect, versus the L40S reply's attribution ambiguity; the question remains among the unchanged 13 reference concerns and does not affect the supported-87 count. All 100 judgments remain recorded. Indices are zero-based.

Checkpoint payload hashes, model revisions/config hashes, scientific source and builder pins, material/selection identities, all three input/row fingerprints, generation settings, and every row's question/gold/source/evidence/target metadata match exactly. The original 87 supported and 13 reference-concern flags were preserved.

This is not bitwise numerical equivalence. The mean RTX-minus-L40S loss difference is −0.000922 nats, mean absolute per-question difference 0.011281, and maximum absolute difference 0.107367 at index 57. PyTorch/CUDA versions and recorded TF32/BF16 flags match; GPU families and available-memory autotuning differ. The pinned resolver plus cached vocabulary size and recorded available memory imply logits chunks of 512 on RTX and 256 on L40S. Those chunk sizes are derived from provenance rather than directly stored operator traces. They affect teacher-forced scoring; they do not explain the greedy reply changes. NLL differences cannot be attributed to GPU hardware alone.

RTX full/no-context NLL controls were reused from its earlier exact same-input, same-checkpoint RTX readout; L40S controls were freshly measured there. This supplement reviews AC responses only. The existing [same-L40S primary report](PI_SAME_L40S_COHORT100.tressoir.html) remains unchanged, so its three columns remain the matched-device comparison. The timing numbers above describe single scoring passes, not a serving benchmark.

[The durable JSON receipt](PI_MERGED32K_RTX_REPLICATION_V1.json) includes all 100 judgments, changed-response source evidence, exact identities, runtime provenance and termination checks. The frozen raw RTX input and local review receipt are in `IB/TMP/FIRST_TASK_AWARE_FINETUNING/constant42k_merged32k_rtx_replication/`. No new jobs, remote access, confirmation reads, source changes or HTML publication were needed.
