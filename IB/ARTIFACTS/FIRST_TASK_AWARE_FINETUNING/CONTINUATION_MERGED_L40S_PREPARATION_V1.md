# L40S queue fallback: constant42k merged32k

This independent worker uses the same completed ORCD constant42k pair, original100 questions and exact32k merged encoder inputs as the reviewed RTX parent worker. Own1250-stage/615 actual Adam5250 provenance, full required sources, shared prompts/targets and total rows keep the same strict guards. Packing stays65536, microbatch1 and the original precision policy. No source, input, model or numerical-gate changes are used to make it fit.

All AC/full/no-context NLL measurements are freshly computed on L40S. The prior RTX piecewise readout binds only checkpoint/input/row identities; its score values are never copied. AC NLL and100 actual greedy answers publish first, then fresh full/no-context NLL. Optional full-context answers follow after the complete core result is saved. A timeout during that optional phase preserves the complete core and marks unfinished full decoding explicitly. Partial resume binds the same hardware/precision policy and each response's actual generated IDs.

Candidate files: `bench/evaluate_continuation_merged_l40s_v1.py` and `.sbatch`. Allocation: one `mit_normal_gpu` L40S,64GiB host,40minutes; application2250seconds. Separate output: `reports/continuation_merged_l40s_cohort100/continue32k_10k_final_lr__merged_32k_l40s/result.json`. Original RTX, H200 and final960 files remain unchanged.

Prior32k inference peaks around32GiB make the46GB device a plausible fit; this candidate has not run, and monolithic inference fit is not guaranteed. An OOM fails visibly; there is no fallback that changes inputs, batches or precision. GPU speed will differ from RTX/H200, so the40-minute bound is a scheduling allowance, not a measured completion estimate.

Seven focused CPU checks passed. Parent/input/row guard functions are structurally identical to the parent worker already tested against all100 real-tokenizer merged requests; actual ORCD parent receipts bind. The new checks verify fresh scorer calls for every view, explicit L40S runtime, phase ordering and isolated report destination. No remote calls, staging or GPU jobs were performed.
