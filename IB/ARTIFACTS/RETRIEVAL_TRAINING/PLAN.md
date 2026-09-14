# Plan — BM25 index, study labeling, single-vector retrieval training

Agent-facing source of truth. The human projection is `PLAN.tressoir.md`.

## Context (researched)

- `/workspace/activation` `.py` files are identical to `/source/activation` (only `__pycache__` differs).
- `DatasetIndex` (`activation/dataset/dataset_index.py`): chunks documents in `__init__`, then
  `_build_index()` embeds with the harness embedding model into a `faiss.IndexFlatIP`.
  Query surface: `query_many_frozen`, `query_docs_many_frozen`, `get_top_documents`,
  `get_chunk_section`, `_build_excluded_sets`.
- `DatasetManager.build_dataset_indexes()` constructs `DatasetIndex` per dataset (chunk + dense build).
- `DatasetStudyGenerator` (`dataset_study.py`): samples chunks, one vLLM structured-output call
  per chunk → `LabeledRetrievalQAExample(origin=SYNTHETIC, split=TRAIN, positive_doc_ids=[doc],
  positive_chunk_ids=[chunk])`. `hard_negative_doc_ids/chunk_ids` exist on the dataclass but are
  never populated.
- Callers of the renamed surface: `tests/test_basic_dataset_loading.py` (several),
  `tests/test_basic_dataset_study.py`, `tests/probe_family_ab.py` (build only).
- Environment: CPU only (no `nvidia-smi`), 8 cores, 30 GB RAM. `rank_bm25`, `scipy`, `peft`,
  `accelerate` are NOT installed. numpy 2.4, torch 2.13, transformers 5.15, faiss-cpu, vllm 0.28.
  HF cache has `Qwen3-Embedding-0.6B`, `Qwen3-1.7B`, BRIGHT/SciFact/SciQ/NQ/MS-MARCO datasets.
- `readout_embedding(last_hidden_state, attention_mask, readout_type)` in `hf_utils.py` does
  EOS/last-token or mean pooling + L2 normalize; reusable by the trainable encoder.
- `canonical_embedding_readout_type` keys on `"embed" in model_id` → a plain causal LM is loaded
  as `AutoModelForCausalLM` by the harness; the training package therefore owns its own encoder
  wrapper (`AutoModel` base tower + EOS readout) and does not go through `LoadedModel`.

## Boundaries

| Concern | Where |
| --- | --- |
| M1 + M2 source changes | `/workspace/activation/{dataset,harness,tests}` and mirrored into the existing staged tree `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/activation/` |
| M1 + M2 diff cards (vs `/source`) | `IB/ARTIFACTS/RETRIEVAL_TRAINING/BM25_STUDY_LABELS_ROUND.tressoir.md` |
| M3 training package | `IB/ARTIFACTS/RETRIEVAL_TRAINING/activation/training/` (source-shaped, new package) + `IB/ARTIFACTS/RETRIEVAL_TRAINING/activation/tests/test_basic_contrastive_training.py` |
| Disposable validation | `IB/TMP/RETRIEVAL_TRAINING/` |
| New dependencies | none (BM25 hand-rolled on numpy; full fine-tune, no peft) |

## Decisions

Accepted (from `PLAN.interactions.json`, round 1):

1. Labeling pool is BM25-only. Do NOT implement a random-chunk knob.
2. Base model `Qwen/Qwen3-0.6B` (public tuned baselines to compare against).
3. Real code paths only: no stubbing/mocking in product code or handoff tests (private IB/TMP
   checks may). Write report files first, then one SkyPilot smoke at the very end, as small as
   possible while executing every stage; no benchmark/micro-benchmark, no recall tables in the smoke.

4. (round 2) Multiple positives: form A (mean of per-positive log-softmax, the query's other
   positives masked from each denominator) with NO cap and NO reduction switch. Flat ragged
   candidate layout: `candidates [N,d]`, `owner[N]`, `is_positive[N]`, doc ids. Average over a
   query's positives, then over queries. B (log-sum) is only a docstring sentence.

## M1 — BM25 code path (Review: landed as planned; test 15.77s; private brute-force check OK)

Renames (mechanical, all call sites updated):

| Before | After |
| --- | --- |
| `DatasetIndex._build_index` | `DatasetIndex.build_dense_index` (public, idempotent) |
| `DatasetIndex.query_many_frozen` | `dense_query_many_frozen` |
| `DatasetIndex.query_docs_many_frozen` | `dense_query_docs_many_frozen` |
| `DatasetIndex.faiss_index` | `dense_faiss_index` |
| `DatasetManager.build_dataset_indexes` | `build_dense_indexes` |
| `DatasetStats.index_size_mb` | `dense_index_size_mb` |
| `DatasetStats.query_search_latencies` | `dense_query_search_latencies` |

Additions:

- `activation/dataset/bm25.py` — `Bm25Index` (numpy only): `tokenize(text)` (lowercase
  `\w+`), build from list of token lists: vocabulary, per-term postings `(doc_idx int32,
  weight float32)` where weight is the full precomputed BM25 term-document score
  `idf(t) * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))`, `idf = ln(1 + (N - df + 0.5)/(df + 0.5))`.
  `search(queries: list[str], k) -> (scores [Q,k] float32, indices [Q,k] int64)` mirroring faiss
  (`-1` padding when fewer than k non-zero hits). Query score = sum over unique query terms.
  Top-k via `np.argpartition`. Memory: sum of postings ≈ number of (term, chunk) pairs.
- `DatasetIndex.__init__` chunks only. `build_bm25_index()` and `build_dense_index()` are explicit,
  idempotent builders. `bm25_query_many_frozen`, `bm25_query_docs_many_frozen`. Shared
  `_collect_chunk_results(all_indices, excluded_sets, top_k)` used by both paths.
- `DatasetManager`: `_get_or_create_index(dataset_id)` (chunk once), `build_bm25_indexes()`,
  `build_dense_indexes()`. The study generator calls `build_bm25_indexes()` itself if missing? No —
  keep explicit: the study generator asserts the bm25 index exists (clear error).
- `HarnessRuntimeConfig`: `doc_bm25_k1: float = 1.5`, `doc_bm25_b: float = 0.75`.
- `DatasetStats`: `bm25_build_latency`, `bm25_index_num_postings`, `bm25_query_search_latencies`.
- Tests: rename call sites; `test_basic_dataset_loading` gains bm25 exact-match self-retrieval
  asserts and exclusion respect on the bm25 path; public loading test prints bm25 top-k alongside
  dense.

## M2 — Study labeling (Review: landed; generator returns examples; private plumbing check OK; GPU via smoke)

- Config: `dataset_study_label_top_k: int = 10`, `dataset_study_label_pool_chars: int = 12_000`,
  reuse `dataset_study_chunk_input_limit`
  per snippet, reuse `dataset_study_batch_size`.
- Flow in `DatasetStudyGenerator`: `_generate_study_material()` (existing) then
  `_label_study_material(examples)`:
  1. `bm25_query_many_frozen([q for examples], top_k=label_top_k)` in one call.
  2. Pool per example: iterate ranked chunks, skip chunks already in `positive_chunk_ids`
     (the source chunk), truncate with `safe_truncate_embedding_chunk(chunk_input_limit)`,
     append while running chars ≤ pool_chars (append-then-break like `take_to_budget`). BM25-only.
  3. Prompt (system + user): question, reference answer, numbered snippets. Schema
     `{"positives": [int], "negatives": [int]}` (arrays of integers); anything not listed, or
     listed in both, is ambiguous. Structured outputs via `StructuredOutputsParams(json=...)`.
  4. Apply: positives → `positive_chunk_ids` (+ `positive_doc_ids` dedup); negatives →
     `hard_negative_chunk_ids` (+ `hard_negative_doc_ids` only for docs with no positive chunk).
     Out-of-range indices dropped, counted as parse failures.
  5. Stats: `study_label_batch_latencies`, `study_label_prompt_tokens`, `study_label_output_tokens`,
     `study_num_label_positives/negatives/ambiguous`, `study_num_label_parse_failures`.
- `synthesize_study_examples(dataset_id)` unchanged entry point; labeling runs inside the
  generator after generation (engine stays resident, one `engine_to_device` cycle).
- Explainer on in-batch negatives lives in the projection (short) and as a docstring in the
  training loss module.

## M3 — Single-vector retrieval training (Review: landed; CPU test 2 passed in 64s with real Qwen3-0.6B; fixed-batch loss 2.60→2.53)

Package `activation/training/`:

- `contrastive_data.py`
  - `TrainingExample(query, positive_chunk_ids, hard_negative_chunk_ids, positive_doc_ids)`.
  - `build_training_pool(loaded_dataset, dataset_index, label_source="study"|"bm25_pseudo",
    split=TRAIN, min_positives=1)`. `study` uses the labeled fields; `bm25_pseudo` derives hard
    negatives as bm25 top-k minus positive docs (lets the pipeline run without the study model —
    noisier labels, documented).
  - `ContrastiveBatchSampler`: shuffles examples per epoch (random order ⇒ in-batch negatives
    are approximately random negatives), takes all positive chunks and up to H random
    hard negatives per query, returns `ContrastiveBatch(queries, positives[B][≤P],
    negatives[B][≤H], positive_doc_ids[B] (all labeled positive docs), candidate doc ids)`.
- `encoder.py` — `SingleVectorEncoder(nn.Module)`: `AutoModel.from_pretrained("Qwen/Qwen3-0.6B")`,
  appends EOS token, `readout_embedding(..., EOS_TOKEN)`. `encode(texts, max_len)` batched with
  grads. Query instruction prefix `"Instruct: Retrieve passages that answer the question\nQuery: "`
  (Qwen-Embedding style, config knob), documents raw.
- `losses.py` — `contrastive_loss(q[B,d], candidates[N,d], owner[N], is_positive[N],
  candidate_doc_ids, positive_doc_ids, temperature)`: logits `q @ cand.T / τ`; base mask = same-doc
  collisions (candidate doc ∈ query's positive docs, not an own positive column). Expand to one row
  per (query, own positive k) pair: additionally mask the query's other own positives, cross-entropy
  with target k; mean over k per query, then mean over queries. Docstring explains explicit/implicit
  and why log-sum is not implemented.
- `train_retriever.py` — `ContrastiveTrainConfig` dataclass (model_id Qwen/Qwen3-0.6B, lr 2e-5,
  batch_size 16, num_hard_negatives 3, temperature 0.02, max_steps, max_query_len 128, max_doc_len 512, warmup
  0.1, bf16 autocast on CUDA, grad checkpointing flag, seed, output_dir) and
  `train_contrastive(config, pool) -> TrainReport(losses, steps, saved_path)`. Plain torch loop,
  AdamW, linear warmup + linear decay, prints every N steps, `save_pretrained` at the end.
- `retrieval_eval.py` — `recall_at_k(encoder, dataset_index, examples, ks=(1,5,10))` at chunk
  and doc level using a temporary `faiss.IndexFlatIP` over the index chunks. For the user's real
  runs; NOT invoked by the smoke (no benchmark).
- `scripts/train_bright_retriever.py` — end-to-end GPU entry: load BRIGHT (sizes as args), build
  bm25, study generation + labeling, pool, train, print one retrieval sample. This is the smoke.
- Test `test_basic_contrastive_training.py` (CPU, real Qwen/Qwen3-0.6B, batch 2, a few steps,
  tiny synthetic corpus, `bm25_pseudo` labels): asserts masking math on a hand-built batch
  (same-doc candidate and other own positive get `-inf`; two-positive query averages two terms) and loss decreases.
  No stubs.

## Validation plan

- M1: `pytest activation/tests/test_basic_dataset_loading.py` (CPU); brute-force BM25 cross-check
  in IB/TMP (private).
- M2: no stub in handoff. Exercised only by the final Sky smoke.
- M3: the CPU test above with the real 0.6B model; `py_compile` of the entry script.
- Reports first: round doc with `git diff --no-index` cards, plan → Review, STATE.
- Last: Sky smoke on a fresh cluster: `train_bright_retriever.py` with ~6 examples / 30-doc corpus /
  4 study chunks / labeling / ~10 steps batch 4 / one retrieval print. Study model = validated
  `test_basic_dataset_study` formula (SUPPORTS_FP4 branch). Teardown after; log to IB/TMP. No retry loop.

## Time budget (autonomous, 1–2 h)

M1 30 min · M2 30 min · M3 40 min · reports 15 min · Sky smoke 20–40 min wall clock (mostly engine setup).
