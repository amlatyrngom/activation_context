# Reader checkpointing: GPU result

The isolated probe passed exactly and warrants a bounded memory check before use in a future run. **No setting has been adopted.** Job23533352 completed on an RTX PRO6000 in9m41s. The only execution change was reader checkpoint threshold0→4096; the side encoder remained at0, with original65536 packing, masks, precision settings and frozen source.

| Same eight questions at8k evidence | Reader0 | Reader4096 |
|---|---:|---:|
| Median of three updates after one warmup |24.6125s|20.1668s|
| Individual measured updates |24.9537 /23.3832 /24.6125s|20.1668 /20.0112 /20.2098s|
| Peak allocated GPU memory |49.8128GiB|62.7945GiB|

This is **18.1% less update time**, or1.220× throughput, at **12.98GiB additional peak allocation** on this batch. It is not a whole-run speed estimate. Measurements ran control then candidate, with only three steady updates each; they use the frozen trainer's production wall-clock timing, not CUDA-event kernel timing. Memory is PyTorch allocated memory, not total device reservation.

The deterministic diagnostic compared the original, unchanged repeat and reader4096. Both comparisons had **zero recorded error** for all764 captured AC-output/gold-log-probability tensors, all615 active gradients (359 AC and256 reader), and joint loss. Fifteen unused recursive parameters consistently had no gradient. The original3% relative/2e-6 absolute-floor tensor gates and1e-3 loss gate were unchanged. First-layer hooks verified four side-encoder recomputations in every condition and **seven reader recomputations→zero**. Reader padded batches ranged up to2×1004 positions;4096 counts the whole padded reader batch, so this is a conditional threshold, not an unconditional disabling of checkpointing.

All three processes matched frozen source, builder, material, question/input hashes, parent weights, optimizer and publication hashes. Each timing process independently restored615 Adam states at4000 with group sizes374/256, then made four in-memory updates at unchanged terminal LRs5e-6/2e-6. The trainer clock reached4004. Exact penalty question IDs matched at every update, including warmup; counts were3,1,0,1. Packing/geometry were identical:16 side-encoder forward batches,266,092 useful and650,288 padded positions over four updates. Parent payloads and optimizer rehashed unchanged; no checkpoint was saved.

The timing runs used ordinary production kernels, with deterministic algorithms off, CUBLAS workspace unset, TF32 on and the same shared FLA settings. Their first SFT loss matched exactly, but later losses differed by up to0.000452 and group gradient norms differed. The exact proof therefore does **not** establish bitwise-identical ordinary training trajectories. These receipts cannot isolate ordinary backward nondeterminism from checkpoint execution order. The proof's joint loss includes the penalty; timing `loss` records SFT alone. Final per-parameter Adam clocks were not separately dumped: incoming4000 clocks, actual optimizer steps and final trainer4004 were verified. No blocking defect was found in this bounded comparison.

Before future use, make one CPU geometry pass over **every actual planned training microbatch**. Select only the few largest memory-risk cases: uncheckpointed reader B×length just below4096, the largest concurrent encoder padding/row payload, and the largest actual penalty group. Check those cases once with ordinary forward/backward, mature optimizer resident and fixed real penalty draws; stop on OOM and retain the parent unchanged. No threshold sweep is needed. The62.79GiB result cannot establish safety for all inputs and already exceeds L40S memory;32k inputs may cross4096 and regain reader checkpointing. Current runs stay unchanged.

Machine-readable audit and raw receipt hashes: [READER_CHECKPOINT_GPU_RESULT_V1.json](READER_CHECKPOINT_GPU_RESULT_V1.json). Raw evidence: `IB/TMP/SYNC/TASK_AWARE_QA/reader_checkpoint_probe_v1/`. Receipt-only audit: `IB/TMP/FIRST_TASK_AWARE_FINETUNING/audit_reader_checkpoint_gpu_v1.py`.
