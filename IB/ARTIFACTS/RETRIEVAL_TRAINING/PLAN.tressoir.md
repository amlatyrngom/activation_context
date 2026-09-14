# BM25, study labeling, and single-vector retrieval training

Plan for three milestones: a BM25 code path beside the existing dense index, a study-time
labeling pass that turns BM25 candidates into explicit positive/negative labels, and a separate
contrastive-training package that fine-tunes a small Qwen causal LM into a single-vector
retriever. All four decisions are accepted and recorded below. Implementation is complete and CPU-validated;
each milestone card leads with its completion report. The handoff with exact diff cards is
`BM25_STUDY_LABELS_ROUND.tressoir.md` in this folder.

## Executive Summary

### Goal

After this round you can build a BM25 index for any loaded dataset without a GPU, have the study
model turn each synthetic question into explicit positive and hard-negative chunk labels, and run
a small, readable contrastive training loop that shows exactly how those labels and the implicit
in-batch negatives become a loss.

### Approach

| Milestone | What lands | Where |
| --- | --- | --- |
| M1 BM25 | `dense_*` renames + `Bm25Index` (numpy only, no new dependency) + `bm25_*` query methods + explicit `build_bm25_indexes` / `build_dense_indexes` | `activation/dataset/{bm25,dataset_index,dataset_manager,dataset}.py`, `harness/runtime_config.py`, tests |
| M2 Labeling | second study pass: BM25 top-10 pool → labeler prompt → positives/negatives applied to the example | `activation/dataset/dataset_study.py`, `harness/runtime_config.py`, `dataset.py` stats |
| M3 Training | new `activation/training/` package: pool builder, encoder, InfoNCE loss with masking, plain torch loop, recall@k eval, GPU entry script, CPU test | new folder `IB/ARTIFACTS/RETRIEVAL_TRAINING/` |

Handoff shape: M1 and M2 are edits to your files, mirrored into the existing staged tree
`IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/activation/` with per-file diff cards (against `/source`)
in a new round document in this folder. M3 is a fresh package staged only in this folder.

### Data flow

`DatasetIndex` → `DatasetStudyGenerator` → `activation/training`

```
LoadedDataset ──chunk──▶ DatasetIndex.chunks
                             ├── build_bm25_index()   → Bm25Index (postings, CPU, seconds)
                             └── build_dense_index()  → faiss.IndexFlatIP (embedding model)

Study:  chunk ──gen──▶ (question, answer, positive=source chunk)
        question ──bm25 top-10──▶ pool (≤ pool_chars) ──labeler──▶ positives / negatives / ambiguous
                                                                   │            │
                                                positive_chunk_ids ◀┘            └▶ hard_negative_chunk_ids

Train:  example ──▶ (query, all positives, ≤H hard negatives)
        batch of B ──encode──▶ q[B,d], candidates[N,d] flat, owner[N], is_positive[N]
        logits = q · candidatesᵀ / τ ; own positives are targets ; mask own other positives + same-doc collisions
```

### In-batch negatives, briefly

Your reading is right. With a batch of \(B\) (query, positive) pairs, query \(i\) scores every
candidate in the batch; only its own positive is the target, so the other \(B-1\) positives act as
negatives at no extra encoding cost. They are **implicit** labels: nobody asserted "chunk \(j\) is
irrelevant to query \(i\)", we rely on the batch being a random draw from the corpus so that a
collision is rare. Two consequences shape the plan:

- **Random order matters.** If examples were grouped by document, in-batch negatives would be
  near-duplicates of the positive and the loss would be poisoned with false negatives. The sampler
  therefore reshuffles every epoch, and the loss masks any candidate whose document is among the
  query's labeled positive documents (cheap insurance for small corpora).
- **Explicit hard negatives are a different signal.** Random in-batch negatives are mostly easy.
  The BM25-mined, labeler-confirmed negatives are lexically close but wrong, which is what the
  model has to learn to reject. Each query's explicit negatives also serve as extra in-batch
  negatives for the other queries, so the candidate pool per step is \(B + B\cdot H\) documents.

Loss, for query \(i\) with one positive \(p_i\) and candidate set \(C\) (all positives and
all hard negatives in the batch):

\[
\mathcal{L}_i = -\log \frac{\exp(q_i \cdot p_i / \tau)}{\sum_{c \in C,\; \text{not masked}} \exp(q_i \cdot c / \tau)}
\]

### Multiple positives per query (your question)

You asked whether the numerator should sum the positives instead of using one random positive.
Both exist in the literature, they are different objectives, and the choice matters once M2
labels give a query several positive chunks. Let \(P_i\) be the positives of query \(i\) and
\(\pi_i(c) = \operatorname{softmax}_c(q_i \cdot c / \tau)\) over the unmasked candidates.

| Form | Per-query loss | Gradient on positive \(p_k\) | Character |
| --- | --- | --- | --- |
| **A. Mean of per-positive log-softmax** (SupCon "out" form) | \(-\frac{1}{\lvert P_i \rvert}\sum_{p \in P_i} \log \pi_i^{(p)}(p)\), where \(\pi_i^{(p)}\) is computed with the *other* positives masked out of the denominator | each positive is pulled toward the query independently, roughly \(\propto 1 - \pi_i^{(p)}(p)\) | every labeled positive gets trained; strongest, most literal use of the labels |
| **B. Sum inside the log** (multi-positive InfoNCE, MIL-NCE) | \(-\log \sum_{p \in P_i} \pi_i(p)\) | \(\propto \pi_i(p_k)\big(\tfrac{1}{\sum_{P_i}\pi_i} - 1\big)\): proportional to how well \(p_k\) already scores | winner-take-all: once one positive dominates, the others receive almost no gradient. Only asks that *some* positive win. By Jensen, B \(\le\) A |
| **C. One random positive per step** (what the first draft said) | \(-\log \pi_i(p)\) for one \(p \sim \mathrm{Uniform}(P_i)\); other positives are not encoded at all | same as A for the sampled positive | an unbiased Monte-Carlo estimate of A with the cheapest batches; noisier per step, converges to the same objective over epochs |

So "one random positive" was not a different objective from the averaged form, only a cheaper
estimator of it. The genuinely different choice is B, and B is weaker for our purpose: after M2
we deliberately collected *several* positives per question and want each to rank above the
BM25-mined negatives, not just the easiest one.

The one rule shared by all forms is the important part: **a query's other positives must never
sit in its denominator as negatives.** In A that is explicit (masked). In C it holds because they
are not encoded. In B it holds because they are in the numerator. This is also why the loss masks
same-document collisions among in-batch candidates.

Accepted: **A with no cap.** Every labeled positive of every query in the batch is encoded.
The pool design already bounds this (at most the source chunk plus the BM25 top-10 labeled
positives), so there is nothing to cap. Implementation detail that falls out of "no cap": the
candidates are one flat ragged list rather than a padded `[B, P, d]` tensor, with two small
index vectors (`owner[j]` = which query candidate `j` belongs to, `is_positive[j]`). No padding
slots, no `p_mask`, no `max_positives_per_query`, and no `log_sum` switch (B stays a sentence in
the docstring). The loss averages over a query's positives first and then over queries, so a
query with many positives does not outweigh one with a single positive.

## Accepted Decisions

- **Labeling pool is BM25-only.** No random chunks, and no random-chunk knob is implemented. Random
  negatives come free as in-batch negatives during training.
- **Base model is `Qwen/Qwen3-0.6B`.** Chosen for its public tuned baselines (Qwen3-Embedding-0.6B)
  that make comparisons easy. Classic dense attention; the encoder wrapper needs no unwrapping.
- **Multiple positives use form A with no cap.** Mean of per-positive log-softmax, the query's
  other positives masked from each denominator, all labeled positives encoded, flat ragged
  candidate layout. No `max_positives_per_query`, no `positive_reduction` switch.
- **Validation: real code paths only, plus one small SkyPilot run at the end.** No stubbing or
  mocking in product code or in the handoff tests. Private throwaway checks may live in `IB/TMP`.
  Report files are written first; the SkyPilot run is the last step and is as small as possible
  while still executing every stage end to end. It is a smoke run, not a benchmark: it prints
  losses and a retrieval sanity sample, and it computes no recall tables.

## Milestones

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M1 — BM25 code path beside dense</span>
    <span class="card-oneliner">Rename the embedding path to dense_*, add a numpy BM25 index with the same query surface.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `activation/dataset/bm25.py` (new `Bm25Index`, numpy only), `DatasetIndex` with
`build_bm25_index` / `build_dense_index`, `bm25_query_many_frozen` / `bm25_query_docs_many_frozen`
beside the renamed `dense_*` methods and a shared `_collect_chunk_results`, `DatasetManager` with
`build_bm25_indexes` / `build_dense_indexes` over one `_get_or_create_index`, two BM25 knobs in
`HarnessRuntimeConfig`, renamed and extended `DatasetStats`, and renamed call sites in the tests.

**Drifts.** `_build_index` became public `build_dense_index` (the manager calls it from outside
now). `probe_family_ab.py` is IB-only, so only its call was renamed. **Review pass:** the hand-rolled
BM25 was replaced by a ten-line adapter over the `bm25s` library with library defaults, the `k1` /
`b` knobs were removed, and the loading tests were split into `_bm25` / `_dense` variants (the
planned changes below still show the original hand-rolled sketch).

**Validation.** `test_basic_dataset_loading` on CPU: `1 passed in 15.77s`, including exact-text BM25
self-retrieval, exclusion respect, an unmatched query returning an empty list, and document-level
BM25 results. A private brute-force BM25 reference on 200 random documents agreed on scores and
top-k membership to 1e-5.

#### Planning Overview

`DatasetIndex.__init__` keeps chunking only. Index building becomes two explicit, idempotent
builders so a BM25-only workflow (study labeling, training-pool construction) never touches the
embedding model. Both query paths share the exclusion and chunk-collection tail.

| Before | After |
| --- | --- |
| `DatasetIndex._build_index` | `build_dense_index()` (public, skips if built) |
| `query_many_frozen` / `query_docs_many_frozen` | `dense_query_many_frozen` / `dense_query_docs_many_frozen` |
| `faiss_index` | `dense_faiss_index` |
| `DatasetManager.build_dataset_indexes` | `build_dense_indexes()`, new `build_bm25_indexes()` |
| `DatasetStats.index_size_mb`, `query_search_latencies` | `dense_index_size_mb`, `dense_query_search_latencies`, plus `bm25_*` |

BM25 is hand-rolled on numpy (about 80 lines) rather than adding `rank_bm25` or `bm25s`: the
environment has neither and no scipy, the corpus sizes here are at most a few hundred thousand
chunks, and a readable reference beats a dependency for adaptation. Tokenization is lowercase
`\w+`; the per-term, per-chunk BM25 weight is precomputed at build time so a query is a sum of
postings vectors plus one `argpartition`. The `search` signature mirrors faiss so the two paths
read alike.

Edge cases: empty query or zero-hit queries return fewer than `top_k` (padded with `-1` like
faiss); exclusions use the same flat +10 over-fetch; `k1=1.5`, `b=0.75` are harness config knobs.

#### Planned Changes

`activation/dataset/bm25.py · Bm25Index` (new)

```python
class Bm25Index:
    def __init__(self, documents_tokens: list[list[str]], k1=1.5, b=0.75): ...
        # vocabulary: term -> postings (doc_idx: int32[], weight: float32[])
        # weight = idf(t) * tf*(k1+1) / (tf + k1*(1 - b + b*len/avg_len))
    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        # scores [Q, k] float32, indices [Q, k] int64 (-1 padded), faiss-shaped
```

`activation/dataset/dataset_index.py · DatasetIndex`

```diff-python
@@ class DatasetIndex: __init__ @@
-        self.faiss_index: faiss.IndexFlatIP | None = None
+        self.dense_faiss_index: faiss.IndexFlatIP | None = None
+        self.bm25_index: Bm25Index | None = None
         print(f"{self.dataset_id} - Chunking.")
         self._chunk_documents()
         print(f"{self.dataset_id} - Chunked to {len(self.chunk_ids)} chunks.")
-        print(f"{self.dataset_id} - Building index.")
-        self._build_index()
-        print(f"{self.dataset_id} - Built index.")
+
+    def build_bm25_index(self):
+        """Cheap CPU index; safe to call before any model is loaded."""
+        ...
+
+    def build_dense_index(self):
+        """Batch-embed every chunk with the harness embedding model (was _build_index)."""
+        ...
⋯ unchanged embedding loop ⋯
-    def query_many_frozen(self, queries, top_k=20, excluded_doc_ids=None):
+    def dense_query_many_frozen(self, queries, top_k=20, excluded_doc_ids=None):
⋯ embedding loop unchanged, then: ⋯
-        _all_scores, all_indices = self.faiss_index.search(query_embeddings, k=...)
-        ... inline exclusion loop ...
+        _all_scores, all_indices = self.dense_faiss_index.search(query_embeddings, k=...)
+        return self._collect_chunk_results(all_indices, all_excluded_sets, top_k)
+
+    def bm25_query_many_frozen(self, queries, top_k=20, excluded_doc_ids=None):
+        all_excluded_sets, extra_fetches = self._build_excluded_sets(len(queries), excluded_doc_ids)
+        _scores, all_indices = self.bm25_index.search(queries, k=min(top_k + extra_fetches, len(self.chunk_ids)))
+        return self._collect_chunk_results(all_indices, all_excluded_sets, top_k)
+
+    def bm25_query_docs_many_frozen(...):  # mirrors dense_query_docs_many_frozen
```

`activation/dataset/dataset_manager.py · DatasetManager`

```diff-python
-    def build_dataset_indexes(self):
-        """Build all dataset indexes"""
-        for loaded_dataset in self.loaded_datasets.values():
-            self.dataset_indexes[loaded_dataset.dataset_id] = DatasetIndex(self.harness, loaded_dataset)
+    def _get_or_create_index(self, dataset_id) -> DatasetIndex:  # chunks once
+    def build_bm25_indexes(self):   # for every loaded dataset
+    def build_dense_indexes(self):  # for every loaded dataset
```

`activation/harness/runtime_config.py · HarnessRuntimeConfig`

```diff-python
     doc_embedding_input_limit_chars: int|None = None
+    doc_bm25_k1: float = 1.5
+    doc_bm25_b: float = 0.75
```

Tests: call sites renamed in `test_basic_dataset_loading.py`, `test_basic_dataset_study.py`,
`probe_family_ab.py`; the basic loading test additionally asserts BM25 exact-text self-retrieval
and exclusion respect; the public loading test prints BM25 top-k next to dense.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M2 — Study labeling pass</span>
    <span class="card-oneliner">BM25 top-10 pool per synthetic question, labeled by the study model into explicit positives and hard negatives.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `make_label_prompt` + `LABEL_JSON_SCHEMA`, `_build_label_pool` (rank order, source
chunk skipped, per-snippet truncation, append-then-break on the char budget), `_apply_labels`
(positives → positive lists, negatives → hard-negative lists, doc-level negatives only for
documents with no positive chunk, both-sides indices treated as ambiguous), and
`_label_study_material` running right after generation while the engine is resident. Two knobs
(`dataset_study_label_top_k`, `dataset_study_label_pool_chars`) and nine stats fields.
`DatasetStudyGenerator` asserts that the BM25 index exists.

**Drifts.** `_generate_study_material` now returns the examples it created so the labeling pass
works on exactly this run's questions. **Review pass:** the label prompt no longer receives the
exam context, and the study test builds only the bm25 index with no embedding model. Out-of-range indices are dropped and counted as parse
failures while the valid ones are still applied.

**Validation.** CPU: a private throwaway script under `IB/TMP` replaced the engine call with canned
JSON and confirmed the pool, parsing, label application, and counters (nothing of it is in the
handoff). GPU: the prompt ran for real in the SkyPilot smoke on `unsloth/Qwen3.8-27B-NVFP4`:
4 questions, 4 labeling requests, 1 positive / 12 negatives / 1 ambiguous, 0 parse failures.

#### Planning Overview

`DatasetStudyGenerator` gains a second phase that runs right after question generation while the
engine is still resident. For every synthetic example it takes the BM25 top-10 chunks for the
question (one batched call), skips the chunk that produced the question (already a positive),
truncates each snippet with the existing `dataset_study_chunk_input_limit`, and gathers snippets
in rank order until `dataset_study_label_pool_chars` (default 12,000) is reached, append-then-break
like `take_to_budget`. The pool is BM25-only (accepted decision); no random chunks are appended.

The labeler prompt shows the question, the reference answer, and the numbered snippets, and asks
for a JSON object `{"positives": [ints], "negatives": [ints]}` under vLLM structured outputs.
A snippet is **positive** when it contains information that answers the question or supports the
reference answer, **negative** when it is off-topic or does not help answer it, and anything not
listed (or listed on both sides) is **ambiguous** and goes nowhere.

Application: positives extend `positive_chunk_ids` and `positive_doc_ids` (deduplicated);
negatives extend `hard_negative_chunk_ids`, and `hard_negative_doc_ids` only for documents that
have no positive chunk for that example. Out-of-range indices count as parse failures.

#### Planned Changes

`activation/harness/runtime_config.py · HarnessRuntimeConfig`

```diff-python
     dataset_study_seed: int = 0
+    dataset_study_label_top_k: int = 10
+    dataset_study_label_pool_chars: int = 12_000
```

`activation/dataset/dataset_study.py · make_label_prompt() + DatasetStudyGenerator._label_study_material()`

```diff-python
+LABEL_JSON_SCHEMA = {"type": "object",
+    "properties": {"positives": {"type": "array", "items": {"type": "integer"}},
+                   "negatives": {"type": "array", "items": {"type": "integer"}}},
+    "required": ["positives", "negatives"], "additionalProperties": False}
+
+def make_label_prompt(study_context: str | None) -> tuple[str, str, dict]:
+    """System prompt defining positive / negative / ambiguous, user instructions, schema."""
+
 class DatasetStudyGenerator:
     def __init__(...):
         ...
         self._generate_study_material()
+        self._label_study_material()
+
+    def _build_label_pool(self, example, ranked_chunks) -> list[DatasetDocumentChunk]:
+        # skip known positives, truncate, append-then-break on pool chars
+
+    def _label_study_material(self):
+        examples = [synthetic TRAIN examples created above]
+        ranked = self.dataset_index.bm25_query_many_frozen([e.query for e in examples], top_k=self.label_top_k)
+        for batch in batches(examples):  # dataset_study_batch_size
+            conversations = [system + "# Question\n...\n# Reference Answer\n...\n# Snippets\n[0] ...\n[1] ..."]
+            outputs = loaded_model.engine_chat_many(conversations, chat_kwargs=label_chat_kwargs)
+            for example, pool, output in zip(...):
+                labels = json.loads(output.text)  # positives / negatives index lists
+                _apply_labels(example, pool, labels)  # + stats counters
```

`activation/dataset/dataset.py · DatasetStats`

```diff-python
+    study_label_batch_latencies: list[float]
+    study_label_prompt_tokens / study_label_output_tokens: list[int]
+    study_num_label_positives: int = 0
+    study_num_label_negatives: int = 0
+    study_num_label_ambiguous: int = 0
+    study_num_label_parse_failures: int = 0
```

`activation/tests/test_basic_dataset_study.py`: builds the BM25 index before the study call and
asserts that at least one example carries a hard negative and that no chunk id appears on both
sides of any example. It runs in the final SkyPilot smoke with a tiny sample count.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M3 — Single-vector contrastive training</span>
    <span class="card-oneliner">Fresh activation/training package: pool → batches → InfoNCE with explicit and implicit negatives → plain torch loop → recall@k.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `activation/training/` with `contrastive_data.py`, `losses.py`, `encoder.py`,
`train_retriever.py`, `retrieval_eval.py`, `scripts/train_bright_retriever.py`, and
`tests/test_basic_contrastive_training.py`, staged only under `IB/ARTIFACTS/RETRIEVAL_TRAINING/`.
The loss is form A with no cap over a flat candidate list (`owner`, `is_positive`), with same-doc
collisions and a query's other positives masked, mean per query then mean over queries.

**Drifts.** The CPU test measures learning on one fixed batch before and after training rather than
comparing per-step losses, because six steps over reshuffled batches are too noisy to order. The
encoder truncates text before appending the end-of-text token so the readout token always
survives. `recall_at_k` exists but the smoke does not call it (no benchmark, as decided).

**Validation.** `test_basic_contrastive_training --slow` on CPU with the real `Qwen/Qwen3-0.6B`:
`2 passed` in 64 s; masking asserts pass; fixed-batch loss 2.60 → 2.53 after 6 steps at batch 4.
The entry script also ran end to end on CPU without the study model (`--label-source bm25_pseudo
--split any`, 2 steps) and on the GPU smoke with study labels: 10 steps at batch 4, loss 3.69 → 0.79
in 4 s, encoder saved, labeled positive ranked first in the retrieval sample. Cluster torn down.

#### Planning Overview

A new package, staged only in this folder, that depends on the dataset layer but not on
`LoadedModel` (training needs gradients; the harness runs inference). Five small modules with
one obvious job each:

| Module | Job |
| --- | --- |
| `contrastive_data.py` | `build_training_pool(dataset, index, label_source)` and an epoch-shuffling sampler that emits `(query, all positives, ≤H hard negatives)` and flattens a batch into one candidate list with `owner` / `is_positive` / doc-id vectors |
| `encoder.py` | `SingleVectorEncoder`: `Qwen/Qwen3-0.6B` base tower via `AutoModel`, EOS-token readout through the existing `readout_embedding`, L2-normalized; query instruction prefix knob (Qwen3-Embedding style, so the public baseline is directly comparable) |
| `losses.py` | `contrastive_loss(...)` with the masking rules and per-positive averaging; docstring is the in-batch and multi-positive explainer in code |
| `train_retriever.py` | `ContrastiveTrainConfig` dataclass + `train_contrastive(config, pool)`; AdamW, warmup + linear decay, bf16 autocast on CUDA, optional gradient checkpointing, `save_pretrained` |
| `retrieval_eval.py` | `recall_at_k` (chunk and doc level) over the index chunks with a temporary flat faiss index, for your later real runs; the smoke only prints one retrieval sample |

`label_source="study"` reads the M2 labels. `label_source="bm25_pseudo"` derives hard negatives
as BM25 top-k minus the example's positive documents, so the whole pipeline runs without the
27B study model (noisier: an unlabeled BM25 hit can be a true positive). Full fine-tune, no
LoRA: `peft` is not installed and a 0.6B model in bf16 with AdamW fits a single L40S with
room to spare, which keeps the reference short.

Interface choice: a dataclass config plus one function, no trainer class. You said you will
adapt it to your style; this shape has the fewest moving parts to read.

#### Planned Changes

`activation/training/losses.py · contrastive_loss()`

```python
def contrastive_loss(q, candidates, owner, is_positive, candidate_doc_ids, positive_doc_ids,
                     temperature=0.02):
    """
    q: [B, d] queries. candidates: [N, d], the flat list of every positive and hard negative
    in the batch; owner[j] is the query index candidate j was drawn for, is_positive[j] says
    whether it is that query's positive or its labeled hard negative.
    logits = q @ candidates.T / tau                       # [B, N]
    Explicit: for query i, columns with owner==i are its labeled positives / hard negatives.
    Implicit: every other column is an in-batch negative (another query's positive or negative).
    Masked with -inf for query i: columns whose doc id is among query i's positive docs but
    which are not query i's own positive columns (same-doc collisions).
    Loss: for each own positive k of query i, mask query i's *other* positives too, take
    -log softmax at k; average over k, then over queries.
    (The multi-positive InfoNCE alternative, -log of the summed positive probability, is
    winner-take-all and deliberately not implemented.)
    """
```

`activation/training/train_retriever.py · ContrastiveTrainConfig / train_contrastive()`

```python
@dataclass
class ContrastiveTrainConfig:
    model_id: str = "Qwen/Qwen3-0.6B"
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_hard_negatives: int = 3
    temperature: float = 0.02
    max_steps: int = 200
    max_query_tokens: int = 128
    max_doc_tokens: int = 512
    warmup_fraction: float = 0.1
    gradient_checkpointing: bool = False
    query_instruction: str = "Instruct: Retrieve passages that answer the question\nQuery: "
    seed: int = 0
    output_dir: str | None = None

def train_contrastive(config, pool, encoder=None) -> TrainReport:
    # sampler → encode queries/positives/negatives with grads → contrastive_loss → AdamW step
    # prints step/loss/lr every N steps; returns losses and the saved path
```

`activation/training/scripts/train_bright_retriever.py`: GPU entry, BRIGHT biology, builds
BM25, runs study generation + labeling, builds the pool, trains, prints one retrieval sample.
Sizes are arguments; the final smoke uses the smallest that still runs every stage.

`activation/tests/test_basic_contrastive_training.py` (CPU, real `Qwen/Qwen3-0.6B`, a few
steps at batch 2 on a tiny synthetic corpus with `bm25_pseudo` labels): asserts the masking math
on a hand-built batch (same-doc collision and other-own-positive columns get `-inf`, a query
with two positives contributes the mean of two terms) and that the loss decreases. No stubs.

</details>

## Validation and handoff

- M1: the real `test_basic_dataset_loading` on CPU, plus a brute-force BM25 cross-check in `IB/TMP`.
- M3: the CPU training test with the real 0.6B model (a few steps).
- Report files first: `BM25_STUDY_LABELS_ROUND.tressoir.md` with M1/M2 diff cards regenerated by
  `git diff --no-index` against `/source`, this plan moved to Review, `IB/STATE.md` updated.
- Last: one SkyPilot smoke on a fresh cluster running `train_bright_retriever.py` at the smallest
  shape that executes every stage (about 6 BRIGHT examples and a 30-document corpus, 4 study
  chunks, labeling, roughly 10 training steps at batch 4, one retrieval sample print). The study
  model is the already-validated engine formula from `test_basic_dataset_study`, so engine
  setup (5 to 13 minutes cold) dominates the wall clock. The cluster is torn down afterward and
  the log lands in `IB/TMP`. A failure is reported as such; no retry loop.
- Outcome: the smoke ran once and succeeded (`Job finished (status: SUCCEEDED)`); wall clock was
  dominated by the 647 s cold engine load. Details in the round document.
