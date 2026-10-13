# Reusable synthesized QA bank

Accepted questions are plain reusable data. They do not expire when the producing prompt, model settings or code changes. Raw corpus text is allowed; native benchmark questions, answers and task solutions never supply synthesis supervision.

The new producer writes to `/workspace/IB/TMP/SYNC/TASK_AWARE_QA/synthesis_bank_v2/`. The original `synthesis_bank/` stays intact and readable. V2 job23404276 is running. Legacy job23401910 was stopped after its eighth published batch (8,526 questions). The target is200,000 accepted, published questions in v2; legacy questions remain extra reusable data.

## Mix per10,000 published questions

| Family | Questions | Raw sources |
|---|---:|---|
| Code |3,000|Pinned repository source files|
| Mathematical papers |1,200|Formula-rich arXiv text|
| Mathematical exposition |800|LibreTexts math/statistics and tagged applied quantitative exposition|
| Financial reasoning |500|Raw annual filings and tables|
| Web/search |2,000|MS MARCO passages only|
| Technical web |500|BRIGHT document fields only|
| Technical documents |800|Repository manuals, Python PEPs and official documentation|
| Law |600|Regulations and judicial opinions|
| Office |600|Office automation docs, business exposition and LibreOffice procedures|

Balance is enforced on accepted questions, not attempted sources. Each immutable1,000-question release has these proportions. Office source attempts are interleaved roughly60% automation/30% business/10% procedural help; accepted internal proportions may differ and remain visible in source metadata. Wiki is omitted. Existing raw capacity does not guarantee200k accepted; limiting families are being expanded through immutable source-extension files.

## Publication and reuse

`manifest.json` lists complete `batches/*.jsonl` releases. Each line contains `id`, `domain`, `source_family`, `question`, `answer`, exact source text/units, supporting-unit IDs, token counts and provenance. Read directly with `json.loads(line)`. No generator, model or prompt-cache lookup is needed.

`bench/select_bank.py` verifies published hashes, selects a reproducible mixed subset and freezes its IDs/hash into ordinary training material. It reads both new portable JSONL and original nested JSON batches. Recipe metadata describes provenance; it never controls data eligibility. A growing bank cannot alter an active training run's frozen selection.

The producer commits accepted microchunks and attempted-source IDs together, including rejected attempts. Surplus accepted rows remain available for later balanced releases. The manifest is the sole publication point; orphan files after interruption are ignored. Stable source-ID receipts prevent repeated committed attempts after raw pool or code changes. Append new raw sources under `bank_source_extensions/<family>__<unique>.jsonl` with an atomic rename; files become immutable once published.

Three-hour Slurm segments stop at a small source-block boundary and requeue automatically. The target can run across many allocations. Unexpected errors or source exhaustion stop visibly for agent repair/expansion; cancellation cannot trigger a self-requeue. A real tiny CPU self-requeue succeeded (job23403726, restart1). The actual GPU producer also passed a graceful signal/requeue onto another node, preserving2,000 published and505 buffered questions and loading all8 immutable source extensions. Focused fixtures cover exact10k quotas, crash/restart, orphan files, rejected-source receipts, changed recipes/pools, immutable extensions and selector hashes. Narrow correctness review passed.

Ordinary source windows produce one question. Substantial contexts of at least1,800 tokens may produce up to two mathematical/financial or three document/legal/office questions with distinct evidence. This producer uses document mode over explicitly recorded raw windows; related-source generation is available in the product but is not claimed as part of this bank's current mix. One batched grounding pass plus schema/nonempty and exact generated-question checks suffice.

The pilot's100 development questions and200 reserved confirmation questions are excluded by exact question text. Raw corpus reuse is allowed. This is reservation of our own generated questions, not a benchmark overlap audit.

## Runtime and preservation

The first legacy batch yielded1,098 accepted questions in444seconds including startup, before accepted-mix balancing. That is not a measured200k campaign rate. Reaching200k requires2.31 published questions/second over24GPU-hours or1.16 over48. Measure the new producer after several balanced releases.

The accepted bank and raw source roots carry `PRESERVE`; local and ORCD copies survive transient cache cleanup. Agent monitoring includes quota/headroom and retention. S3 is used only for important checkpoint selections; this bank has not been uploaded there.


## Scalable 27B bank

A separate `synthesis_bank_27b/` uses Qwen3.5-27B dynamic FP8 with reasoning disabled in generation and checking. Its initial32k target can be extended live: atomically replace `~/orcd/pool/activation/taskaware/qa27b_target.json` with `{"target_questions": 200000}` (or another positive multiple of1000). The producer reads the target between microbatches. If it already completed, resubmit the same bank job; prior accepted data and source cursors are retained. Balanced1k releases are immediately consumable while it grows. No regeneration or cache-key matching is required to read accepted records.

The revised27B recipe checks premises, arithmetic direction/units, exact code branches and missing definitions. Temporary Source-number references are rejected in both question and answer, with citations kept in metadata. Both prompt strings and the pinned model revision are recorded per recipe. The live9B producer remains on its prior recipe, so a comparison against it measures the revised27B supervision pipeline rather than only model size. The current27B job is23412107, queued behind the2k×4 training arm. No EC2 allocation is required.

Before either new synthesis job started, the generator/checker presence penalty was set explicitly to0. The inherited1.5 discourages output-token reuse (including repeated JSON boolean decisions); removing it avoids that decoding bias without adding reasoning. This is a justified configuration choice, not a measured diagnosis of the earlier reference defects. Original9B production and all accepted data remain unchanged.
