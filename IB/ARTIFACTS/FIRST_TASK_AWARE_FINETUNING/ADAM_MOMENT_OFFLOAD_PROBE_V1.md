# Adam moment-placement probe — completed

**ORCD job23555733 passed all five workers on L40S.** Exact native-context numerical comparisons passed, and both tested merged32k training batches fit. This is a private feasibility result; offloading has **not** been adopted in production.

| Test | Peak allocated memory | Seconds per8-question update |
|---|---:|---:|
| Native2k resident, two updates |32.603GiB|29.55 /26.98|
| Native2k offloaded, same two updates |31.882GiB|30.81 /28.58|
| Merged32k offloaded, actual first8 |38.042GiB|84.08|
| Merged32k offloaded, worst-geometry batch |38.023GiB|81.71|

The unchanged resident repeat and offloaded native2k workers matched exactly on raw/clipped gradients, updated parameters, moments, clocks, groups, gold log-probabilities and actual logits-call traces across two real updates. All615 active gradient IDs matched the615 populated states among630 group positions; every populated clock advanced5250→5252. The32k workers independently began at the same parent and advanced all615 clocks5250→5251. Optimizer serialization/load-state roundtrips passed, and the frozen parent payload hashes remained unchanged.

All workers used the frozen42k pair, actual inherited Adam, logical8/micro1, checkpoint threshold0, packing65536, ratio1/16 and constant5e-6/2e-6. Moments were on CPU before baseline/forward/backward, restored synchronously around the original AdamW step; step tensors and arithmetic were unchanged. The original pre-optimizer autotuning point resolved the same256-token logits chunks and FLA/checkpoint policy throughout. No decoding ran. The32k tests used ordinary production numerical flags and included initial/final AC/full/no-context evaluation of the longest full-context member of the fixed100.

**This is not fast enough for a complete960-question32k run before the PI deadline.** Extrapolating the two measured32k updates to960×1 gives roughly163–168minutes of training alone (about2h45). Ten subsequent held100/learning32 panels, the initial panel, setup and checkpoints would likely push the total beyond3hours. Full-run throughput has not been measured. This may enable later long-context training if larger-GPU queues remain blocked.

Limits remain explicit: exact numerical equivalence was proved only for the same-GPU native2k diagnostic, not a resident-versus-offloaded32k comparison or another device. The largest-geometry proxy batch is not an exhaustive fit guarantee for all960 items; it starts from the parent, not a replayed training prefix. The smaller native peak reduction also shows that4.355GiB of movable moments does not imply that much reduction in overall peak. Reported trainer-step timings include copies and instrumentation; the separate original-step stopwatch excludes the preceding restore. **Optimizer-state serialization/load_state_dict is not a fresh-process full paired-checkpoint resume.** That integration remains untested.

Durable evidence: [result and input hashes](ADAM_MOMENT_OFFLOAD_PROBE_V1.result.json), [validation receipt](ADAM_MOMENT_OFFLOAD_PROBE_V1.receipt.json), [submission](ADAM_MOMENT_OFFLOAD_PROBE_SUBMISSION_V1.json). Local raw results: `IB/TMP/SYNC/TASK_AWARE_QA/adam_moment_offload_probe_v1/`. Frozen worker, launcher and design remain unchanged in `bench/`;37 CPU checks preceded this GPU proof. No further implementation or submission accompanies this result.
