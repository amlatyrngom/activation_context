# Rewarm continuation replay: keep running; exact numerical replay is not established

Read-only audit of original job23511289 and resumed23516714. No source/queue changes, GPU probes, or large checkpoint reads were performed. Small logs were read through the guarded ORCD connection; historical/current raw reports and the frozen trainer source supplied the remaining evidence.

The first replayed update, after800 questions, matches more closely than the rounded log suggests:

| Measurement for update101 | Original | Resumed |
|---|---:|---:|
| Training SFT loss | 0.36597520718351007 | 0.36597520718351007 |
| Sampled no-context penalty | 0.010891259647905827 | 0.010891259647905827 |
| Combined preclip gradient norm | 11.972111701965332 | 11.969069480895996 |

The first forward objectives agree exactly at stored precision; the backward norm already differs by−0.0254%. Delta-head gradient norms also differ slightly at this same pre-update point. At the next update, the SFT loss differs by0.00163713; by update120 its difference is−0.03624986. The training loss logged here excludes the no-context penalty; that penalty was compared separately. These observations fit small production backward/update differences followed by divergence of the parameter trajectory. They do not alone prove the specific kernel responsible or exclude every possible optimizer-state discrepancy.

The source and available metadata support correct operational resume:

- Published partial pair is epoch1/step100, with AC/saved optimizer epoch0; the runner restarts original epoch1 and skips exactly100 logical batches. Frozen material IDs/order are checked before loading. The trainer reshuffles with the same `seed + epoch_number` and sorts each unchanged eight-question logical batch by reader length.
- Original versus effective training configuration differs only in `resume_skip_steps:0→100`. The horizon remains1250 updates; the50-update warmup is not restarted. The frozen helper gives update101 LRs1.993583646030358e-5 and7.974334584121430e-6. The optimizer loader restores state, then resets configured base rates before applying that original clock.
- The runner validates both parameter-group lengths and individual moment shapes before the trainer's load. The log confirms100 updates restored with no layout-discard warning. Adam moments and per-parameter clocks are loaded rather than deliberately reset. This bounded audit did not independently deserialize the multi-gigabyte saved optimizer; the guards and logs are evidence, not a tensor-by-tensor equality proof.
- Every RAG item carries an AC/no-context candidate, and the resumed reference cache loaded all10000 keys. The runner advances the dedicated Python penalty RNG by800 draws. Reconstructing the next168 draws from the frozen seed exactly matches the uninterrupted stream; reporting calls use `with_penalty=False`, so the added resume baseline does not consume those draws. Expected update101 has one selected penalty example. Actual selected IDs were not persisted live, so this is source-derived sampling verification.
- There is no gradient scaler in this BF16 training path. Dropout is disabled, greedy sample count is zero, and no intentional random training operation besides the separate penalty RNG was found. Torch RNG is reseeded on resume rather than serialized; exact replay should therefore not be promised if future stochastic operations are introduced.
- Frozen source/builder/material receipts match. The original job ran on node5104 and the resumed job on node5101. Production runs enable TF32 and do not enable deterministic algorithms. Prior same-graph GPU diagnostics already demonstrated nondeterministic backward behavior. Historical resolved autotune choices and all dependency/kernel versions were not captured for both attempts, so their exact equality cannot be established here.

Recommendation: keep the resumed run. There is no concrete order, schedule, penalty-stream or state-load defect justifying interruption. Preserve the original abandoned branch and label the continuing run as resumed from its saved pair; do not describe its production trajectory as bitwise deterministic. The small first backward difference followed by subsequent model divergence is the leading explanation, with the precise kernel/update cause unproven.

No new GPU test is warranted solely by these scalar differences. If exact replay becomes necessary, the smallest useful diagnostic is one logical update executed twice from the same verified pair/Adam state with identical selected penalty IDs: record actual LRs, parameter-step/moment identity, gradient norms/dot products and resulting parameter updates. This measures an ordinary-production repeat difference before attributing anything specifically to resume. A fixed-gradient CPU save/load comparison can separate serialization from GPU backward behavior; prior inherited-Adam CPU checks already cover the basic mechanism.

A reporting-only detail: `intervention_grads` uses the pre-update clock. The old branch's point at100 represents gradients for unsaved update101; resume trimming keeps that point and appends a second100. This audit used the last100 entry for the resumed branch. Future display cleanup should distinguish or remove the abandoned point; it does not affect training.

Exact observations, hashes, configuration differences, source-derived LR values and penalty counts: `CONTINUATION_REWARM_RESUME_AUDIT.json`.
