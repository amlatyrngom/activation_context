# Budget proof failure: shape dependence is real; cause unresolved

RTX job 23511648 failed its unchanged strict gates. Keep the original 65536 budget for useful long-context training. No smaller budget is validated, and no production timing ran. This analysis used only the saved receipt, log and frozen source; it did not change source, submit work or inspect confirmation data.

Only the first eight-question case ran. It contains 84 independently encoded passages and 756 output tensors (one input-row tensor plus eight deep deltas per passage). The quantile case was never reached.

| Budget | Change in joint objective | Failed populated gradient tensors / 615 | Failed output tensors / 756 | Padded side-encoder positions |
|---|---:|---:|---:|---:|
| 65536 unchanged repeat | 0 exactly | 0 | 0 | 162,572 |
| 32768 | −0.00048736 | 413 | 55 | 118,355 |
| 16384 | +0.00058051 | 474 | 123 | 102,948 |

Baseline joint objective is 0.12334547; it includes the sampled no-context penalty and is not held answer NLL. Both loss differences pass the 0.001 absolute loss gate, but the gradient/output gates fail. Useful side-encoder positions remain 66,523; passage lengths, row counts, penalty selections, weights and parameter metadata are unchanged.

**The strongest localization:** budget 32768 leaves the first 32-passage and last eight-passage batches unchanged. All 360 of their output tensors are bitwise identical. The middle batches split from `[24,2507,2048]` into `[21,1215,2048]` plus `[3,2507,2048]`, and from `[20,2751,2048]` into `[17,1746,2048]` plus `[3,2751,2048]`. Every output tensor for those 44 passages changes; 55 exceed the gate. This proves a forward dependence on the changed batching geometry. It is not explained solely by backward accumulation, shuffled targets, or the old nondeterministic scalar-backward problem. Every batch in this 32768 comparison still has more than one passage, so switching between a missing mask and a padding mask is not its cause. The 16384 comparison additionally creates singleton batches and therefore has that extra branch difference.

Gradient differences are substantial, including nontrivial values:

| Parameter group | 32768 failures | Median per-tensor relative L2 difference | 16384 failures | Median per-tensor relative L2 difference |
|---|---:|---:|---:|---:|
| Encoder LoRA | 192 / 192 | 40.0% | 192 / 192 | 82.2% |
| AC modules | 79 / 167 | 18.2% | 102 / 167 | 39.8% |
| Reader LoRA | 142 / 256 | 8.0% | 180 / 256 | 14.7% |

These are unweighted tensor summaries, not concatenated-gradient norms or direction cosines. For example, the input-scale scalar's baseline gradient magnitude can be reconstructed as 0.36519; its absolute difference is 0.02140 at32768 and 0.68490 at16384. The receipt omits tensor sizes, candidate norms, signed gradients and dot products. Whole-group gradient magnitudes/directions and the resulting Adam update direction cannot be recovered honestly from it.

Frozen-source inspection found no obvious intended semantic change. `ac_model.py:414–453` preserves request order after length-sorted grouping; `:465–548` embeds each passage independently, masks padded positions, slices each passage at its own length and returns its original row count. The local mixer (`ac_model_utils.py:327–380`) keeps batch examples separate and masks padding in its neighboring attention windows. Summary pooling and heads run per passage. `loaded_model.py:485–523` disables the side model's cache. There is no cross-example batch normalization or deliberate content mixing. The proof disables dropout and verifies checkpoint recomputation, fixed penalty IDs and parameter ownership.

The leading hypothesis is numerical sensitivity to changed BF16/TF32 matrix and attention shapes, amplified by the trained encoder and reader loss gradient. Mixer projections/SDPA and the hybrid side encoder's projections, causal convolution, recurrent attention and full attention are candidates. Deterministic execution makes the same shape repeatable; it does not make distinct shapes perform identical floating-point reductions. The saved receipt cannot locate the first divergent operator or establish that all differences are ordinary rounding. Padding-mask values and hybrid-kernel behavior still need a direct check before adopting an optimization. Locally installed Qwen implementation was consulted for candidate mechanisms; its dependency version was not recorded in this remote proof and is not claimed to be an exact remote-source match.

The smallest useful next diagnostic is one affected reader microbatch, forward localization only, with the same paired weights, grad-enabled/checkpointed mode and fixed inputs. Compare 65536, its unchanged repeat and32768. Track original request IDs, valid lengths, actual masks/position IDs, then capture valid activations for two predetermined passages (shortest and longest in that affected group): before/after the mixer, side-encoder inputs, each side block, and final rows/deltas. Keep comparisons per request, excluding padding, and record the first divergence; do not load another dataset or run an optimizer. Capturing only two passages bounds host copies. If inputs/masks/positions differ semantically, repair that precise issue. If divergence first appears in a shape-sensitive arithmetic operation, rerun only that localized operation with identical inputs in FP32/reference math and the original gates to test whether it contracts. Persist baseline/candidate norms, dot products and error sums if a subsequent backward is justified. This diagnoses a cause; it does not turn the existing failed gate into a pass or delay the authorized larger run.

Machine-readable measurements and source/proof hashes: `STRONG_PIECEWISE_BUDGET_FAILURE.json`.
