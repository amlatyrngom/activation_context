# Reader checkpointing: actual-pool memory cases

The CPU census confirms that reader threshold4096 would remove **all reader recomputation in both existing8k-context training pools**. This makes the measured speed improvement relevant to these input lengths, but does not establish their memory safety. Encoder checkpointing0, packing65536, masks and precision remain unchanged; no GPU test or production adoption has occurred.

| Exact original epoch | Strong8k questions | Gentle4k familiar questions |
|---|---:|---:|
| Logical batches of8 / reader microbatches of2 |1000 /4000|500 /2000|
| Largest padded reader positions |2454 (2×1227)|2454 (2×1227)|
| Reader microbatches at or above4096 |0|0|
| Largest actual no-context penalty positions |1226|1218|
| Largest concurrent encoder padded positions |72,999|70,740|
| Largest single prepared encoder batch |69,540|69,606|

The encoder's packing test uses source lengths; appended compressed rows explain prepared batches above65,536. This is the unchanged frozen implementation. Reconstruction matched actual production totals exactly: strong8k **65,979,986 encoder tokens /3,986,645 rows**; gentle4k **33,014,865 tokens /1,995,278 rows**. Both saved training orders, seed-plus-epoch shuffle, length-sorted reader microbatches and actual25% penalty draws were used. Gentle inputs reuse the same frozen per-question builder and retrieval corpus; its saved4000 IDs select the subset. No confirmation was opened and no model weights were loaded. The full CPU rebuild took29m05s; its result is retained for reuse.

The fixed rule selects at most three batches by separate shape extremes, without losses or answer quality:

- **Reader:** original strong pass questions1,753–1,760 (`strong8k_0220`), reaching2454 padded reader positions.
- **Concurrent encoder:** questions6,569–6,576 (`strong8k_0822`), reaching72,999 encoder positions across prepared batches. This exact logical batch happened to select no penalties.
- **Penalty:** questions6,409–6,416 (`strong8k_0802`), reaching1226 no-context penalty positions, with four penalties across its eight questions.

`bench/probe_reader_memory_v1.py/.sbatch` is a candidate-only check. It requires the completed exact615-gradient numerical proof, then uses a fresh process and the same original32k final weights/Adam4000 for each case. Each process performs **one full private update**, including clipping and Adam scratch, at unchanged terminal rates; verifies actual Adam4001, exact input hashes/penalty IDs/observed encoder geometry, and rehashes the parent. No checkpoint is written. An OOM or other failure stops the probe; it never falls back to another policy. This is memory measurement, not a new QA training arm or another equivalence/speed matrix.

Proposed allocation: **15min, one RTX PRO6000,64GiB host;840s application bound**, expected roughly5–10min for three loads and updates. No L40S fit is implied. Reusing ordinary production kernels keeps this distinct from the earlier deterministic correctness proof. Even if all three fit, axis-wise extremes do not prove every combination or unseen pool safe; use the observed device margin to decide a future bounded production trial. No whole-run speed or quality claim follows.

Ready inputs are `IB/TMP/FIRST_TASK_AWARE_FINETUNING/reader_checkpoint_geometry_v1/{selected_cases.json,summary.json}`; the launcher expects copies under `QA_ROOT/bench/reader_checkpoint_geometry_v1/`. Full census stays in `all_batches.json`. Runtime hashes and13 focused CPU checks are recorded in [READER_CHECKPOINT_MEMORY_PREPARATION_V1.json](READER_CHECKPOINT_MEMORY_PREPARATION_V1.json). Root review is required before staging or submission.
