# Piecewise V2 preflight

**Verdict: pass.** The existing checkpoint threshold can be set to0 for both matched arms. No product/source patch or numerical-tolerance change is adopted. No jobs were submitted or remote files changed during this review.

The frozen amendment is [PIECEWISE_TRAINING_DESIGN_V2.json](PIECEWISE_TRAINING_DESIGN_V2.json), identically copied under `bench/`; SHA256 `86fcaad5db4b24d8b5f4972567a78e7e2f3d4e821b18b3d5ca4e55ae4ed98e08`. It pins the deterministic proof/settings, each raw memory receipt, all diagnostic workers/launchers, the V2 training runner/launcher, final evaluator and restart archiver. V1 and failed ordinary-kernel proofs remain unchanged.

The deterministic GPU diagnostic passes **without using its tolerance allowance**: loss,18 AC outputs, and all615 present gradients are bitwise identical for the retained original graph, fresh original graph and checkpointed graph. Fifteen unused gradients are consistently absent. Eleven parameter-version/trainability checks pass. Actual encoder/reader recomputation counts change from1→1/2→2 to1→2/2→4. Source, original paired P2 payloads, builder and material identities match V1. Deterministic settings are explicitly diagnostic-only; production continues with its ordinary kernels. This demonstrates checkpoint-only mathematics under controlled execution, not bitwise reproducibility of ordinary production backward.

All four ordinary-kernel fits complete three real SFT/25%-penalty updates from fresh P2:

| Layout / training geometry | Peak allocated GPU memory | Steady seconds per8 questions |
|---|---:|---:|
| Separate parts / previously failing batch |49.81GiB|23.11s|
| Separate parts / source-length quantiles |52.28GiB|24.13s|
| Merged / previously failing batch |37.38GiB|14.12s|
| Merged / source-length quantiles |37.55GiB|15.00s|

Question IDs, full evidence, target hashes and total compressed rows match across layouts in each geometry. The exact failed batch contains an encoder call with32 short parts, maximum593 positions and18,976 padded positions; the previous1700 maximum-length gate disabled checkpointing there. Threshold0 enables it. The measured batches fit on the same RTX PRO6000 hardware selected by both training launchers. These bounded fits do not prove every remaining batch will fit, nor replay the lost15 updates/moments.

The runner diff only exposes the existing checkpoint-threshold field, validates it and passes it to the trainer. AST comparison confirms identical input builder, row allocator, harness, answer masks and resume functions. The original source hashes match the frozen snapshot. Both launchers retain4,000 unique questions ×1, fresh original P2 optimizer, 8k evidence/up to16 passages,1/16 equal total rows, encoder budget65,536, reader microbatch2/logical8, LRs5e-5/2e-5, original joint clipping,50-update warmup/500-update cosine schedule, fixed bare-base no-context penalty and10% panels. Final readout still requires the arm’s own complete pair and500 updates, original400 NLL/token replay, unchanged development100 and separate32k transfer; no confirmation is read.

One genuine finding was fixed: the restart archiver hardlinked its preserved reference cache. The frozen trainer can overwrite that path directly when filling references, so a later write could also alter the archive. It now makes an independent `shutil.copy2` copy and verifies its SHA256. Only the frozen base reference is reused; the failed run has no paired optimizer checkpoint and the restart is explicitly fresh P2.

Focused validation: seven evaluator rejection cases plus valid-final case passed; Python compilation and all three shell syntax checks passed; source/input-builder/resume AST checks passed; a tiny filesystem check confirms the copied cache is independent and preserves the source when rewritten. GPU evidence was inspected, not rerun. The parent owns the inactive-V1-job check, archive execution, same-name local report archival, staging and resource scheduling. Keep at most four active autonomous GPUs; continuations/readouts must remain in their complete dependency chains.
