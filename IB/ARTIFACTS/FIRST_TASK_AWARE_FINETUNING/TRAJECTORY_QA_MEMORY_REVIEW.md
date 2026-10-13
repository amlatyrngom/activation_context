# QA memory correction

Reviewer: `compaction_teacher_review`. Source examined: frozen V9 product code and the failed QA traceback. No product edit is needed.

The first 2,000-question QA run on L40S completed baseline scoring and decoded examples, then failed during the first logical batch of eight examples. The reader forward took Transformers' non-checkpointed branch. Its compressed sequences often contain only 500–1,555 tokens, below the configured 1,700-token checkpoint threshold, while the encoder still processes a long history.

This was allocated-memory exhaustion: 43.69 GiB allocated on a 44.39 GiB device, with only approximately 179 MiB reserved but unused. Adam moment parking cannot help before Adam creates its states. The earlier long compaction gate used one example per update; it did not establish the same memory bound as QA's eight-example accumulation. Existing gradient buffers can raise later micro-example peaks before the first optimizer update.

The retry uses the existing `--checkpoint-tokens 0` setting. Zero remains zero through execution-config resolution; it checkpoints every gradient-bearing encoder, reader and no-context-penalty forward. Teacher and evaluation forwards remain unchanged. The retry has a new output/cache identity and a byte-identical question/history selection. Data, targets, logical batch, optimizer and learning schedule are unchanged.

Validation completed on the active retry: the full QA run records its first 16 micro-forwards, first two completed eight-example updates, and a subsequent approximately 32k-history example after optimizer buffers exist. It records actual reader checkpoint selection, finite nonzero encoder/reader gradient norms and peak GPU memory. It does not shorten inputs or targets. The private telemetry wrapper changes no arithmetic.

Private source V10 also fixes immediate phase/status JSON updates. A check using the real HTML and training reporters passed current phase, examples/epochs, loss, inactive-counter cleanup and unchanged HTML bytes. The complete product handoff remains V4.

At06:25 UTC, the first two full eight-example updates passed. Encoder gradient norms were10.49 and19.06; reader norms1.17 and1.62, all finite/nonzero. Peak allocated memory was40,530,310,656 bytes (37.75GiB), including histories up to32,213 tokens. The subsequent 32,735-token history passed after 12 completed updates, with the same 37.75 GiB peak. At 06:33 UTC, 176 questions had completed training without another memory failure. This verifies the previously failing accumulation shape and a near-maximum history; continued monitoring covers the remaining data.
