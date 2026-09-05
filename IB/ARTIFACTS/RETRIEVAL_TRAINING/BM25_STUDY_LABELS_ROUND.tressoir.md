# BM25, study labeling, and retrieval training round

Handoff for the three milestones of `PLAN.tressoir.md`: the `dense_*` rename plus a BM25 code path
in `DatasetIndex`, the study-time labeling pass that turns BM25 candidates into explicit positive and
hard-negative chunk labels, and a new `activation/training/` package with a readable contrastive
training loop. M1 and M2 are staged in `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/activation/` (the
existing handoff tree) and their exact per-file deltas against `/source` are the cards below. M3 lives
only in `IB/ARTIFACTS/RETRIEVAL_TRAINING/activation/`.

## Validation summary

| Check | Result |
| --- | --- |
| `test_basic_dataset_loading.py` (CPU; bm25 and dense variants, basic and public) | `4 passed in 323.42s` after the review pass |
| BM25 adapter contract (unmatched query → empty, exclusions, exact-text self-retrieval) | asserted in the bm25 loading tests |
| Labeling plumbing with a replaced engine call (private, `IB/TMP`, not in the handoff) | positives applied, out-of-range index counted as parse failure, no chunk on both sides |
| `test_basic_contrastive_training` (CPU, real `Qwen/Qwen3-0.6B`, 6 steps, batch 4) | `2 passed in 63.75s` after the adapter swap; fixed-batch loss 2.60 → 2.53; masking asserts pass |
| SkyPilot end-to-end smoke (`train_bright_retriever.py`, fresh RTX PRO 6000 node) | `Job finished (status: SUCCEEDED)`, exit 0; details in **Smoke run** |

## How to read the training package

Start at `activation/training/losses.py`: its module docstring is the in-batch and multi-positive
explainer, and `contrastive_logit_rows` is the whole label flow in about fifteen lines. Then
`train_retriever.py::batch_loss` shows how a `ContrastiveBatch` becomes those tensors, and
`contrastive_data.py` shows where the batch's `owner` / `is_positive` vectors come from.

| File | Role |
| --- | --- |
| `training/contrastive_data.py` | `build_training_pool` (`study` labels or `bm25_pseudo` fallback), `ContrastiveBatchSampler` (epoch reshuffle, all positives, ≤H random hard negatives, flat candidate list) |
| `training/losses.py` | `contrastive_logit_rows` (masks) and `contrastive_loss` (per-positive rows, mean per query, mean over queries) |
| `training/encoder.py` | `SingleVectorEncoder`: `AutoModel` tower, EOS readout via the harness `readout_embedding`, truncation before the EOS token |
| `training/train_retriever.py` | `ContrastiveTrainConfig`, `train_contrastive` (AdamW, warmup + linear decay, bf16 autocast on CUDA, `save_pretrained`) |
| `training/retrieval_eval.py` | `search_chunks`, `recall_at_k` for your real runs |
| `training/scripts/train_bright_retriever.py` | The end-to-end entry: BRIGHT → bm25 → study + labels → pool → train → one retrieval sample |
| `tests/test_basic_contrastive_training.py` | Masking unit test plus a slow CPU training test with the real 0.6B model |

`activation/training/losses.py` (complete, for reading here):

````python
"""
Contrastive (InfoNCE-style) loss over a flat candidate list.

Explicit and implicit labels
----------------------------
For a batch of B queries, the candidate list holds every positive and a few labeled hard
negatives of each query (see contrastive_data.py). With logits = q @ candidates.T / tau:

- Explicit labels: for query i, the columns with owner == i are its own labeled positives
  (targets) and its own labeled hard negatives (bm25-mined, confirmed by the study model).
- Implicit labels (in-batch negatives): every other column, i.e. the positives and negatives the
  other queries brought, counts as a negative for query i. Nobody asserted those are irrelevant;
  we rely on the batch being a random draw from the corpus, which is why the sampler reshuffles
  each epoch. As cheap insurance we mask any candidate whose document is among query i's
  positive documents but which is not one of its own positive columns (same-document collision).

Multiple positives (form A, no cap)
-----------------------------------
Each own positive k of query i gets its own softmax row in which the query's *other* positives
are also masked (a query's positives must never act as its negatives). The per-query loss is the
mean over its positives, and the batch loss the mean over queries, so a query with many positives
does not outweigh a query with one. The multi-positive InfoNCE alternative, -log of the *summed*
positive probability, is winner-take-all (its gradient on a positive scales with how well that
positive already scores) and is deliberately not implemented.
"""

import torch
import torch.nn.functional as F


def contrastive_logit_rows(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    owner: torch.Tensor,
    is_positive: torch.Tensor,
    candidate_doc_ids: list[str],
    positive_doc_ids: list[set[str]],
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (row_logits [R, N], row_targets [R], row_query [R]): one row per (query, own positive)
    pair with the masks applied as -inf. Exposed separately so the masking is testable.
    """
    num_queries, num_candidates = query_embeddings.shape[0], candidate_embeddings.shape[0]
    logits = query_embeddings @ candidate_embeddings.T / temperature                  # [B, N]
    query_ids = torch.arange(num_queries, device=logits.device)
    own_positive = (owner[None, :] == query_ids[:, None]) & is_positive[None, :]       # [B, N]
    collision = torch.tensor(
        [[doc_id in positive_doc_ids[i] for doc_id in candidate_doc_ids] for i in range(num_queries)],
        dtype=torch.bool, device=logits.device,
    ).reshape(num_queries, num_candidates) & ~own_positive
    logits = logits.masked_fill(collision, float("-inf"))
    row_query, row_targets = own_positive.nonzero(as_tuple=True)                       # [R], [R]
    other_own_positives = own_positive[row_query].clone()
    other_own_positives[torch.arange(row_query.shape[0], device=logits.device), row_targets] = False
    row_logits = logits[row_query].masked_fill(other_own_positives, float("-inf"))     # [R, N]
    return row_logits, row_targets, row_query


def contrastive_loss(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    owner: torch.Tensor,
    is_positive: torch.Tensor,
    candidate_doc_ids: list[str],
    positive_doc_ids: list[set[str]],
    temperature: float = 0.02,
) -> torch.Tensor:
    """
    query_embeddings [B, d], candidate_embeddings [N, d] (both L2-normalized), owner [N] long,
    is_positive [N] bool, candidate_doc_ids (N strings), positive_doc_ids (B sets).
    """
    row_logits, row_targets, row_query = contrastive_logit_rows(
        query_embeddings, candidate_embeddings, owner, is_positive,
        candidate_doc_ids, positive_doc_ids, temperature,
    )
    row_loss = F.cross_entropy(row_logits.float(), row_targets, reduction="none")      # [R]
    num_queries = query_embeddings.shape[0]
    per_query_sum = torch.zeros(num_queries, device=row_loss.device).index_add_(0, row_query, row_loss)
    per_query_count = torch.zeros(num_queries, device=row_loss.device).index_add_(
        0, row_query, torch.ones_like(row_loss),
    )
    has_positive = per_query_count > 0
    return (per_query_sum[has_positive] / per_query_count[has_positive]).mean()
````

## Review pass (after the first handoff)

Four changes from your review, all folded into the cards below:

- **BM25 comes from the `bm25s` library**, not custom code. `Bm25Index` is now a ten-line adapter (library defaults, zero-score slots masked to -1). The `k1` / `b` knobs are gone; `pyproject.toml` gains `bm25s`.
- **The study test builds only the bm25 index** and configures no embedding model.
- **The label prompt ignores the exam context**; relevance is judged on the question and reference answer alone.
- **Loading tests are split** into `_bm25` and `_dense` variants that share one body, so a bm25 run never loads a model.

Validation after the pass: `test_basic_dataset_loading.py` `4 passed in 323.42s` (both basic and both public variants on CPU); `test_basic_contrastive_training.py --slow` re-run against the new adapter (result in the summary table).

## Diffs to adapt (M1 + M2)

The probe scripts are IB-only by decision; `tests/probe_family_ab.py` in the staged tree only had
its `build_dataset_indexes()` call renamed to `build_dense_indexes()` and carries no card.

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">pyproject.toml</span>
    <span class="card-oneliner">Adds the bm25s dependency (refresh uv.lock with `uv lock`).</span>
    <span class="card-badge">Diff</span>
  </summary>

One dependency line (plus a trailing newline). Refresh `uv.lock` with `uv lock`; the workspace lock already resolved it. Exact delta vs `/source/pyproject.toml`:

````diff-toml
diff --git a/source/pyproject.toml b/pyproject.toml
index 72781e8..20ff3ee 100644
--- a/source/pyproject.toml
+++ b/pyproject.toml
@@ -11,6 +11,7 @@ dependencies = [
     "vllm",
     "datasets",
     "faiss-cpu",
+    "bm25s>=0.3.11",
 ]
 
 [project.scripts]
@@ -34,4 +35,4 @@ markers = [
 
 [build-system]
 requires = ["hatchling"]
-build-backend = "hatchling.build"
\ No newline at end of file
+build-backend = "hatchling.build"
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/bm25.py</span>
    <span class="card-oneliner">New: thin adapter over bm25s with a faiss-shaped search.</span>
    <span class="card-badge">Diff</span>
  </summary>

New file. Library defaults (Lucene BM25, k1=1.5, b=0.75, English stop words, no stemming); zero-score slots are masked to -1 so an unmatched query yields an empty result. Exact delta vs `/source/activation/dataset/bm25.py`:

````diff-python
diff --git a/activation/activation/dataset/bm25.py b/activation/activation/dataset/bm25.py
new file mode 100644
index 0000000..28a8c30
--- /dev/null
+++ b/activation/activation/dataset/bm25.py
@@ -0,0 +1,39 @@
+"""
+Thin adapter around the bm25s library so the sparse path in DatasetIndex mirrors faiss:
+build once, then search(queries, k) -> (scores, indices) with -1 padding.
+
+Library defaults are used on purpose (Lucene-style BM25 with k1=1.5, b=0.75, English stop
+words, no stemming): retrieval papers report BM25 baselines with default parameters and the
+chunker already bounds document length.
+"""
+
+import time
+
+import bm25s
+import numpy as np
+
+
+class Bm25Index:
+    def __init__(self, texts: list[str]):
+        self.num_documents = len(texts)
+        build_start_time = time.time()
+        self.retriever = bm25s.BM25()
+        self.retriever.index(
+            bm25s.tokenize(texts, stopwords="en", show_progress=False),
+            show_progress=False,
+        )
+        self.build_time = time.time() - build_start_time
+
+    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
+        """
+        faiss-shaped search: (scores [Q, k] float32, indices [Q, k] int64), sorted by decreasing
+        score. Slots past the last document that matched any query term hold -1 (bm25s fills
+        them with arbitrary zero-score documents).
+        """
+        k = min(k, self.num_documents)
+        query_tokens = bm25s.tokenize(queries, stopwords="en", return_ids=False, show_progress=False)
+        indices, scores = self.retriever.retrieve(query_tokens, k=k, show_progress=False)
+        indices = indices.astype(np.int64)
+        scores = scores.astype(np.float32)
+        indices[scores <= 0] = -1
+        return scores, indices
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_index.py</span>
    <span class="card-oneliner">dense_* renames, explicit bm25/dense builders, shared result collection, bm25_* queries.</span>
    <span class="card-badge">Diff</span>
  </summary>

The constructor now only chunks. `build_bm25_index` / `build_dense_index` are idempotent. Both query paths end in `_collect_chunk_results`, which is the old inline exclusion loop hoisted out unchanged. Exact delta vs `/source/activation/dataset/dataset_index.py`:

````diff-python
diff --git a/activation/source/activation/dataset/dataset_index.py b/activation/activation/dataset/dataset_index.py
index c2eee7f..0258ffa 100644
--- a/activation/source/activation/dataset/dataset_index.py
+++ b/activation/activation/dataset/dataset_index.py
@@ -14,6 +14,7 @@ from .dataset import (
     DatasetDocumentChunk,
 )
 from .dataset_utils import safe_truncate_embedding_chunk
+from .bm25 import Bm25Index
 
 if t.TYPE_CHECKING:
     from activation.harness import HarnessRuntime
@@ -34,13 +35,11 @@ class DatasetIndex:
         self.chunks: dict[str, DatasetDocumentChunk] = dict()
         self.chunk_ids: list[str] = list() # cached computation of list(self.chunks.keys())
         self.chunk_id_indexes: dict[str, int] = dict() # map from chunk id to int index.
-        self.faiss_index: faiss.IndexFlatIP | None = None
+        self.dense_faiss_index: faiss.IndexFlatIP | None = None
+        self.bm25_index: Bm25Index | None = None
         print(f"{self.dataset_id} - Chunking.")
         self._chunk_documents()
         print(f"{self.dataset_id} - Chunked to {len(self.chunk_ids)} chunks.")
-        print(f"{self.dataset_id} - Building index.")
-        self._build_index()
-        print(f"{self.dataset_id} - Built index.")
 
     def _chunk_documents(self):
         """
@@ -73,10 +72,25 @@ class DatasetIndex:
         for idx, chunk_id in enumerate(self.chunk_ids):
             self.chunk_id_indexes[chunk_id] = idx
 
-    def _build_index(self):
+    def build_bm25_index(self):
         """
-        Build index by batch embedding.
+        Cheap CPU index over the raw chunk texts; needs no model. Idempotent.
         """
+        if self.bm25_index is not None:
+            return
+        stats = self.loaded_dataset.stats
+        print(f"{self.dataset_id} - Building bm25.")
+        self.bm25_index = Bm25Index([chunk.chunk_text for chunk in self.chunks.values()])
+        stats.bm25_build_latency = self.bm25_index.build_time
+        print(f"{self.dataset_id} - Built bm25 in {stats.bm25_build_latency:.2f}s.")
+
+    def build_dense_index(self):
+        """
+        Build index by batch embedding. Idempotent.
+        """
+        if self.dense_faiss_index is not None:
+            return
+        print(f"{self.dataset_id} - Building dense index.")
         stats = self.loaded_dataset.stats
         embedding_model = self.harness.loaded_models[self.embedding_model_name]
         chunk_texts = [
@@ -102,11 +116,11 @@ class DatasetIndex:
         print(f"{self.dataset_id} - Building faiss.")
         faiss_build_start_time = time.time()
         embeddings = torch.cat(embeddings).float().numpy()
-        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[-1])
-        self.faiss_index.add(embeddings)
-        stats.index_size_mb = embeddings.nbytes / 2**20
+        self.dense_faiss_index = faiss.IndexFlatIP(embeddings.shape[-1])
+        self.dense_faiss_index.add(embeddings)
+        stats.dense_index_size_mb = embeddings.nbytes / 2**20
         elapsed = time.time() - faiss_build_start_time
-        print(f"{self.dataset_id} - Built faiss in {elapsed:.2f}s. Size={stats.index_size_mb:.2f}MB.")
+        print(f"{self.dataset_id} - Built faiss in {elapsed:.2f}s. Size={stats.dense_index_size_mb:.2f}MB.")
 
 
     def _build_excluded_sets(self, num_queries: int, excluded_doc_ids: list[list[str] | None] | None = None) -> tuple[list[set], int]:
@@ -119,15 +133,41 @@ class DatasetIndex:
         if any(len(excluded) > 0 for excluded in excluded_sets):
             return (excluded_sets, 10)
         return (excluded_sets, 0)
-        
-    def query_many_frozen(
+
+    def _collect_chunk_results(
+        self,
+        all_indices: np.ndarray,
+        all_excluded_sets: list[set],
+        top_k: int,
+    ) -> list[list[DatasetDocumentChunk]]:
+        """
+        Shared tail of both query paths: map ranked chunk indices to chunks,
+        dropping -1 padding and excluded documents, up to top_k per query.
+        """
+        all_results = []
+        for indices, excluded in zip(all_indices, all_excluded_sets):
+            results = []
+            for index in indices:
+                if index < 0:
+                    continue
+                chunk_id = self.chunk_ids[index]
+                chunk = self.chunks[chunk_id]
+                if chunk.doc_id in excluded:
+                    continue
+                results.append(chunk)
+                if len(results) >= top_k:
+                    break
+            all_results.append(results)
+        return all_results
+
+    def dense_query_many_frozen(
         self,
         queries: list[str],
         top_k: int = 20,
         excluded_doc_ids: list[list[str] | None] | None = None,
     ) -> list[list[DatasetDocumentChunk]]:
         """
-        Helper to perform batched frozen queries.
+        Helper to perform batched frozen dense queries.
         Returns, for each query, up to top_k chunks.
         """
         stats = self.loaded_dataset.stats
@@ -135,7 +175,7 @@ class DatasetIndex:
         stats.query_batch_sizes.append(len(queries))
         # Batch query
         queries_sorted = sorted(enumerate(queries), key=lambda x: len(x[1]))
-        query_embeddings = np.empty((len(queries), self.faiss_index.d), dtype=np.float32)
+        query_embeddings = np.empty((len(queries), self.dense_faiss_index.d), dtype=np.float32)
         total_embedding_time = 0
         reporting_interval = 0
         for start in range(0, len(queries_sorted), self.batch_size):
@@ -162,29 +202,14 @@ class DatasetIndex:
         all_excluded_sets, extra_fetches = self._build_excluded_sets(
             len(queries), excluded_doc_ids
         )
-        _all_scores, all_indices = self.faiss_index.search(
+        _all_scores, all_indices = self.dense_faiss_index.search(
             query_embeddings,
             k=min(top_k + extra_fetches, len(self.chunk_ids)), # +10 handles most cases of exclusion.
         )
-        stats.query_search_latencies.append(time.time() - search_start_time)
-        all_results = []
-        for indices, excluded in zip(all_indices, all_excluded_sets):
-            results = []
-            for index in indices:
-                if index < 0:
-                    continue
-                chunk_id = self.chunk_ids[index]
-                chunk = self.chunks[chunk_id]
-                if chunk.doc_id in excluded:
-                    continue
-                results.append(chunk)
-                if len(results) >= top_k:
-                    break
-            all_results.append(results)
-        return all_results
-
+        stats.dense_query_search_latencies.append(time.time() - search_start_time)
+        return self._collect_chunk_results(all_indices, all_excluded_sets, top_k)
 
-    def query_docs_many_frozen(
+    def dense_query_docs_many_frozen(
         self,
         queries: list[str],
         top_k: int = 20,
@@ -195,7 +220,47 @@ class DatasetIndex:
         For domains that necessitate full documents. The 5x over-fetch can
         undershoot top_k documents when results concentrate in few documents.
         """
-        all_chunk_results = self.query_many_frozen(
+        all_chunk_results = self.dense_query_many_frozen(
+            queries,
+            top_k=5 * top_k,
+            excluded_doc_ids=excluded_doc_ids,
+        )
+        return [
+            self.get_top_documents(chunks)[:top_k]
+            for chunks in all_chunk_results
+        ]
+
+    def bm25_query_many_frozen(
+        self,
+        queries: list[str],
+        top_k: int = 20,
+        excluded_doc_ids: list[list[str] | None] | None = None,
+    ) -> list[list[DatasetDocumentChunk]]:
+        """
+        BM25 counterpart of dense_query_many_frozen: same exclusion semantics,
+        no model involved. Queries with no matching term return an empty list.
+        """
+        stats = self.loaded_dataset.stats
+        stats.query_batch_sizes.append(len(queries))
+        search_start_time = time.time()
+        all_excluded_sets, extra_fetches = self._build_excluded_sets(
+            len(queries), excluded_doc_ids
+        )
+        _all_scores, all_indices = self.bm25_index.search(
+            queries,
+            k=min(top_k + extra_fetches, len(self.chunk_ids)),
+        )
+        stats.bm25_query_search_latencies.append(time.time() - search_start_time)
+        return self._collect_chunk_results(all_indices, all_excluded_sets, top_k)
+
+    def bm25_query_docs_many_frozen(
+        self,
+        queries: list[str],
+        top_k: int = 20,
+        excluded_doc_ids: list[list[str] | None] | None = None,
+    ) -> list[list[DatasetDocument]]:
+        """BM25 counterpart of dense_query_docs_many_frozen (same 5x over-fetch)."""
+        all_chunk_results = self.bm25_query_many_frozen(
             queries,
             top_k=5 * top_k,
             excluded_doc_ids=excluded_doc_ids,
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_manager.py</span>
    <span class="card-oneliner">build_dataset_indexes → build_dense_indexes, plus build_bm25_indexes.</span>
    <span class="card-badge">Diff</span>
  </summary>

One `_get_or_create_index` chunks each dataset once, so the two builders can be called in either order. Exact delta vs `/source/activation/dataset/dataset_manager.py`:

````diff-python
diff --git a/activation/source/activation/dataset/dataset_manager.py b/activation/activation/dataset/dataset_manager.py
index 8f768db..09f1057 100644
--- a/activation/source/activation/dataset/dataset_manager.py
+++ b/activation/activation/dataset/dataset_manager.py
@@ -20,13 +20,26 @@ class DatasetManager:
         self.loaded_datasets[loaded_dataset.dataset_id] = loaded_dataset
         return
 
-    def build_dataset_indexes(self):
-        """Build all dataset indexes"""
-        print(f"Building Indexes: {list(self.loaded_datasets.keys())}")
-        for loaded_dataset in self.loaded_datasets.values():
-            self.dataset_indexes[loaded_dataset.dataset_id] = DatasetIndex(
-                self.harness, loaded_dataset,
+    def _get_or_create_index(self, dataset_id: str) -> DatasetIndex:
+        """Chunk a dataset once; the bm25 and dense indexes are built on top explicitly."""
+        if dataset_id not in self.dataset_indexes:
+            self.dataset_indexes[dataset_id] = DatasetIndex(
+                self.harness, self.loaded_datasets[dataset_id],
             )
+        return self.dataset_indexes[dataset_id]
+
+    def build_bm25_indexes(self):
+        """Build the bm25 index of every loaded dataset (CPU only, no model needed)."""
+        print(f"Building bm25 indexes: {list(self.loaded_datasets.keys())}")
+        for dataset_id in self.loaded_datasets:
+            self._get_or_create_index(dataset_id).build_bm25_index()
+        return
+
+    def build_dense_indexes(self):
+        """Build the dense (embedding) index of every loaded dataset."""
+        print(f"Building dense indexes: {list(self.loaded_datasets.keys())}")
+        for dataset_id in self.loaded_datasets:
+            self._get_or_create_index(dataset_id).build_dense_index()
         return
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset.py</span>
    <span class="card-oneliner">Stats: dense_* renames, bm25 build latency and search latencies, study-label counters.</span>
    <span class="card-badge">Diff</span>
  </summary>

`index_size_mb` → `dense_index_size_mb`, `query_search_latencies` → `dense_query_search_latencies`, plus `bm25_*` and `study_label_*` / `study_num_label_*` fields surfaced in `summarize()`. Exact delta vs `/source/activation/dataset/dataset.py`:

````diff-python
diff --git a/activation/source/activation/dataset/dataset.py b/activation/activation/dataset/dataset.py
index d1e573b..b5a8116 100644
--- a/activation/source/activation/dataset/dataset.py
+++ b/activation/activation/dataset/dataset.py
@@ -135,19 +135,30 @@ class DatasetStats:
     total_num_documents: int = 0
     avg_document_chars: int = 0
     # Computed at index build time.
-    index_size_mb: float = 0.0
+    dense_index_size_mb: float = 0.0
+    bm25_build_latency: float = 0.0
     build_embedding_latencies: list[float] = field(default_factory=list)
     build_embedding_input_chars: list[int] = field(default_factory=list)
     # Computed/Updated at query time.
     query_embedding_latencies: list[float] = field(default_factory=list)
     query_embedding_input_chars: list[int] = field(default_factory=list)
-    query_search_latencies: list[float] = field(default_factory=list)
+    dense_query_search_latencies: list[float] = field(default_factory=list)
+    bm25_query_search_latencies: list[float] = field(default_factory=list)
     query_batch_sizes: list[int] = field(default_factory=list)
     # Study statistics
     study_prompt_tokens: list[int] = field(default_factory=list)
     study_output_tokens: list[int] = field(default_factory=list)
     study_batch_latencies: list[float] = field(default_factory=list)
     study_num_parse_failures: int = 0
+    # Study labeling statistics (bm25 pool -> positives / negatives / ambiguous).
+    study_label_prompt_tokens: list[int] = field(default_factory=list)
+    study_label_output_tokens: list[int] = field(default_factory=list)
+    study_label_batch_latencies: list[float] = field(default_factory=list)
+    study_label_pool_sizes: list[int] = field(default_factory=list)
+    study_num_label_positives: int = 0
+    study_num_label_negatives: int = 0
+    study_num_label_ambiguous: int = 0
+    study_num_label_parse_failures: int = 0
 
     def summarize(self) -> dict:
         # Returns average statistics.
@@ -160,7 +171,8 @@ class DatasetStats:
             "total_document_chars": self.total_document_chars,
             "total_num_documents": self.total_num_documents,
             "avg_document_chars": self.avg_document_chars,
-            "index_size_mb": self.index_size_mb,
+            "dense_index_size_mb": self.dense_index_size_mb,
+            "bm25_build_latency": self.bm25_build_latency,
             "num_build_embedding_batches": len(self.build_embedding_latencies),
             "avg_build_embedding_latency": average(self.build_embedding_latencies),
             "total_build_embedding_latency": total(self.build_embedding_latencies),
@@ -172,8 +184,10 @@ class DatasetStats:
             "total_query_embedding_latency": total(self.query_embedding_latencies),
             "avg_query_embedding_input_chars": average(self.query_embedding_input_chars),
             "total_query_embedding_input_chars": total(self.query_embedding_input_chars),
-            "avg_query_search_latency": average(self.query_search_latencies),
-            "total_query_search_latency": total(self.query_search_latencies),
+            "avg_dense_query_search_latency": average(self.dense_query_search_latencies),
+            "total_dense_query_search_latency": total(self.dense_query_search_latencies),
+            "avg_bm25_query_search_latency": average(self.bm25_query_search_latencies),
+            "total_bm25_query_search_latency": total(self.bm25_query_search_latencies),
             "num_study_requests": len(self.study_prompt_tokens),
             "avg_study_prompt_tokens": average(self.study_prompt_tokens),
             "avg_study_output_tokens": average(self.study_output_tokens),
@@ -183,6 +197,15 @@ class DatasetStats:
                 if self.study_batch_latencies else 0.0
             ),
             "study_num_parse_failures": self.study_num_parse_failures,
+            "num_study_label_requests": len(self.study_label_prompt_tokens),
+            "avg_study_label_prompt_tokens": average(self.study_label_prompt_tokens),
+            "avg_study_label_output_tokens": average(self.study_label_output_tokens),
+            "total_study_label_latency": total(self.study_label_batch_latencies),
+            "avg_study_label_pool_size": average(self.study_label_pool_sizes),
+            "study_num_label_positives": self.study_num_label_positives,
+            "study_num_label_negatives": self.study_num_label_negatives,
+            "study_num_label_ambiguous": self.study_num_label_ambiguous,
+            "study_num_label_parse_failures": self.study_num_label_parse_failures,
         }
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_study.py</span>
    <span class="card-oneliner">Labeling pass: bm25 pool → labeler prompt → positives / hard negatives.</span>
    <span class="card-badge">Diff</span>
  </summary>

`make_label_prompt()` (no exam context, by review) and `LABEL_JSON_SCHEMA`; `_generate_study_material` now returns the created examples; `_build_label_pool`, `_apply_labels`, `_label_study_material` run while the engine is resident. The constructor asserts the bm25 index exists. Exact delta vs `/source/activation/dataset/dataset_study.py`:

````diff-python
diff --git a/activation/source/activation/dataset/dataset_study.py b/activation/activation/dataset/dataset_study.py
index 8d641f2..1f55f90 100644
--- a/activation/source/activation/dataset/dataset_study.py
+++ b/activation/activation/dataset/dataset_study.py
@@ -7,6 +7,12 @@ Study generation is simple. Until study budget is reached:
 - Select random chunk.
 - Prompt.
 - Parse.
+
+Study labeling then turns each question into explicit retrieval labels:
+- Take the bm25 top-k chunks for the question (the source chunk is already a positive).
+- Gather them in rank order until the pool char budget is reached.
+- Ask the study model which snippets are positives / negatives; anything it leaves out is ambiguous.
+- Positives extend the positive labels, negatives the hard-negative labels, ambiguous goes nowhere.
 """
 import json
 import random
@@ -40,6 +46,49 @@ STUDY_JSON_SCHEMA = {
 }
 
 
+LABEL_JSON_SCHEMA = {
+    "type": "object",
+    "properties": {
+        "positives": {"type": "array", "items": {"type": "integer"}},
+        "negatives": {"type": "array", "items": {"type": "integer"}},
+    },
+    "required": ["positives", "negatives"],
+    "additionalProperties": False,
+}
+
+
+def make_label_prompt() -> tuple[str, str, dict]:
+    """
+    Returns system prompt, instructions, json schema for the snippet labeling task.
+    Deliberately independent of the exam context: relevance is judged on the question alone.
+    """
+    system_prompt = """
+# Core Guidelines
+- You are a relevance judge building a retrieval training set.
+- You are given a study question, its reference answer, and numbered corpus snippets.
+- Label each snippet index:
+    - **positive**: the snippet contains information that answers the question or directly supports the reference answer.
+    - **negative**: the snippet does not help answer the question (off-topic, or only shares vocabulary/entities with it).
+    - Leave out any snippet you cannot decide on (partial, tangential, contradictory); those are treated as ambiguous.
+- Be strict: sharing the topic or the same named entities without containing the answer is a negative.
+- Every listed index must come from the snippet numbering; never list the same index on both sides.
+"""
+    system_prompt += """
+# Response Format
+Your answer must be formatted as a json object of snippet indices like:
+```json
+{
+    "positives": [0, 3],
+    "negatives": [1, 2, 5]
+}
+```
+"""
+    instructions = """
+Label the snippets as positives / negatives for this question. Leave out ambiguous ones.
+"""
+    return system_prompt, instructions, LABEL_JSON_SCHEMA
+
+
 def make_study_prompt(study_context: str | None) -> tuple[str, str, dict]:
     """
     Returns system prompt, instructions, json schema.
@@ -97,7 +146,11 @@ class DatasetStudyGenerator:
         self.chunk_input_limit = harness.harness_config.dataset_study_chunk_input_limit
         self.chat_kwargs = harness.harness_config.dataset_study_chat_kwargs
         self.study_seed = harness.harness_config.dataset_study_seed
-        self._generate_study_material()
+        self.label_top_k = harness.harness_config.dataset_study_label_top_k
+        self.label_pool_chars = harness.harness_config.dataset_study_label_pool_chars
+        assert self.dataset_index.bm25_index is not None, "Study labeling needs the bm25 index (build_bm25_indexes)."
+        examples = self._generate_study_material()
+        self._label_study_material(examples)
 
 
     def _sample_study_chunks(self) -> list[DatasetDocumentChunk]:
@@ -117,7 +170,7 @@ class DatasetStudyGenerator:
 
 
 
-    def _generate_study_material(self):
+    def _generate_study_material(self) -> list[LabeledRetrievalQAExample]:
         loaded_model = self.harness.loaded_models[self.study_model_name]
         # vllm batches a whole conversation list inside one chat() call, so a
         # simple single-threaded loop needs no locks.
@@ -126,6 +179,7 @@ class DatasetStudyGenerator:
         system_prompt, instructions, json_schema = make_study_prompt(self.study_context)
         chat_kwargs = self._make_engine_chat_kwargs(json_schema)
         stats = self.loaded_dataset.stats
+        examples: list[LabeledRetrievalQAExample] = []
         example_count = 0
         total_generation_time = 0
         reporting_interval = 0
@@ -177,6 +231,7 @@ class DatasetStudyGenerator:
                     positive_chunk_ids=[chunk.chunk_id],
                 )
                 self.loaded_dataset.labeled_retrieval_examples[example.example_id] = example
+                examples.append(example)
                 example_count += 1
             if reporting_interval <= 0:
                 print(
@@ -185,4 +240,131 @@ class DatasetStudyGenerator:
                 )
                 reporting_interval = max(1, len(study_samples) // 20)
             reporting_interval -= len(batch)
+        return examples
+
+    def _build_label_pool(
+        self,
+        example: LabeledRetrievalQAExample,
+        ranked_chunks: list[DatasetDocumentChunk],
+    ) -> list[tuple[DatasetDocumentChunk, str]]:
+        """
+        (chunk, snippet text) pairs in bm25 rank order, skipping chunks already labeled
+        positive (the source chunk), gathered append-then-break until the pool char budget.
+        """
+        known_positives = set(example.positive_chunk_ids or [])
+        pool = []
+        total_chars = 0
+        for chunk in ranked_chunks:
+            if chunk.chunk_id in known_positives:
+                continue
+            snippet = safe_truncate_embedding_chunk(chunk.chunk_text, self.chunk_input_limit)
+            pool.append((chunk, snippet))
+            total_chars += len(snippet)
+            if total_chars >= self.label_pool_chars:
+                break
+        return pool
+
+    def _apply_labels(
+        self,
+        example: LabeledRetrievalQAExample,
+        pool: list[tuple[DatasetDocumentChunk, str]],
+        positives: list[int],
+        negatives: list[int],
+    ):
+        """
+        Positives extend positive_chunk_ids / positive_doc_ids; negatives extend
+        hard_negative_chunk_ids, and hard_negative_doc_ids only for documents that have no
+        positive chunk for this example. Indices listed on both sides are ambiguous.
+        """
+        stats = self.loaded_dataset.stats
+        positive_set = set(positives) - set(negatives)
+        negative_set = set(negatives) - set(positives)
+        example.positive_chunk_ids = list(example.positive_chunk_ids or [])
+        example.positive_doc_ids = list(example.positive_doc_ids or [])
+        example.hard_negative_chunk_ids = list(example.hard_negative_chunk_ids or [])
+        example.hard_negative_doc_ids = list(example.hard_negative_doc_ids or [])
+        for index in sorted(positive_set):
+            chunk, _ = pool[index]
+            if chunk.chunk_id not in example.positive_chunk_ids:
+                example.positive_chunk_ids.append(chunk.chunk_id)
+            if chunk.doc_id not in example.positive_doc_ids:
+                example.positive_doc_ids.append(chunk.doc_id)
+        positive_docs = set(example.positive_doc_ids)
+        for index in sorted(negative_set):
+            chunk, _ = pool[index]
+            if chunk.chunk_id not in example.hard_negative_chunk_ids:
+                example.hard_negative_chunk_ids.append(chunk.chunk_id)
+            if chunk.doc_id not in positive_docs and chunk.doc_id not in example.hard_negative_doc_ids:
+                example.hard_negative_doc_ids.append(chunk.doc_id)
+        stats.study_num_label_positives += len(positive_set)
+        stats.study_num_label_negatives += len(negative_set)
+        stats.study_num_label_ambiguous += len(pool) - len(positive_set) - len(negative_set)
+
+    def _label_study_material(self, examples: list[LabeledRetrievalQAExample]):
+        """
+        Second study pass: label the bm25 candidate pool of every synthetic question.
+        Runs while the engine is still resident from generation.
+        """
+        if not examples:
+            return
+        loaded_model = self.harness.loaded_models[self.study_model_name]
+        loaded_model.engine_to_device(TARGET_DEVICE)
+        system_prompt, instructions, json_schema = make_label_prompt()
+        chat_kwargs = self._make_engine_chat_kwargs(json_schema)
+        stats = self.loaded_dataset.stats
+        all_ranked = self.dataset_index.bm25_query_many_frozen(
+            [example.query for example in examples],
+            top_k=self.label_top_k,
+        )
+        pools = [
+            self._build_label_pool(example, ranked)
+            for example, ranked in zip(examples, all_ranked)
+        ]
+        total_label_time = 0
+        reporting_interval = 0
+        for batch_start in range(0, len(examples), self.batch_size):
+            batch = list(zip(examples, pools))[batch_start:batch_start + self.batch_size]
+            conversations = []
+            for example, pool in batch:
+                snippets = "\n".join(
+                    f"[{index}]\n```\n{snippet}\n```"
+                    for index, (_, snippet) in enumerate(pool)
+                )
+                user_text = (
+                    f"# Question\n{example.query}\n\n"
+                    f"# Reference Answer\n{example.gold_answers[0]}\n\n"
+                    f"# Snippets\n{snippets}\n{instructions}"
+                )
+                conversations.append([
+                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
+                    {"role": "user", "content": [{"type": "text", "text": user_text}]},
+                ])
+            batch_start_time = time.time()
+            outputs = loaded_model.engine_chat_many(conversations, chat_kwargs=chat_kwargs)
+            elapsed = time.time() - batch_start_time
+            total_label_time += elapsed
+            stats.study_label_batch_latencies.append(elapsed)
+            for (example, pool), output in zip(batch, outputs):
+                stats.study_label_prompt_tokens.append(output.prompt_token_count)
+                stats.study_label_output_tokens.append(output.output_token_count)
+                stats.study_label_pool_sizes.append(len(pool))
+                try:
+                    labels = json.loads(output.text)
+                    positives = [int(index) for index in labels["positives"]]
+                    negatives = [int(index) for index in labels["negatives"]]
+                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
+                    stats.study_num_label_parse_failures += 1
+                    continue
+                in_range = lambda indexes: [index for index in indexes if 0 <= index < len(pool)]
+                if len(in_range(positives)) + len(in_range(negatives)) < len(positives) + len(negatives):
+                    stats.study_num_label_parse_failures += 1 # Out-of-range index: count, keep the valid ones.
+                self._apply_labels(example, pool, in_range(positives), in_range(negatives))
+            if reporting_interval <= 0:
+                print(
+                    f"{self.dataset_id} - Study labels: {batch_start + len(batch)}/{len(examples)} questions, "
+                    f"+{stats.study_num_label_positives} pos / {stats.study_num_label_negatives} neg / "
+                    f"{stats.study_num_label_ambiguous} ambiguous. {total_label_time:.2f}s."
+                )
+                reporting_interval = max(1, len(examples) // 20)
+            reporting_interval -= len(batch)
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/runtime_config.py</span>
    <span class="card-oneliner">Two labeling knobs (top-k, pool chars). No BM25 knobs.</span>
    <span class="card-badge">Diff</span>
  </summary>

Library defaults are used for BM25 by review. Exact delta vs `/source/activation/harness/runtime_config.py`:

````diff-python
diff --git a/activation/source/activation/harness/runtime_config.py b/activation/activation/harness/runtime_config.py
index 7b91137..b3d3f8e 100644
--- a/activation/source/activation/harness/runtime_config.py
+++ b/activation/activation/harness/runtime_config.py
@@ -32,6 +32,8 @@ class HarnessRuntimeConfig:
     dataset_study_chunk_input_limit: int|None = None # Limit for fast testing.
     dataset_study_chat_kwargs: dict|None = None
     dataset_study_seed: int = 0 # Seeds chunk sampling and generation sampling.
+    dataset_study_label_top_k: int = 10 # BM25 candidates considered per synthetic question.
+    dataset_study_label_pool_chars: int = 12_000 # Char budget of the snippet pool shown to the labeler.
 
 @dataclass
 class HarnessStats:
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/tests/test_basic_dataset_loading.py</span>
    <span class="card-oneliner">Split into bm25 and dense variants sharing one body.</span>
    <span class="card-badge">Diff</span>
  </summary>

`test_basic_dataset_loading_{bm25,dense}` and `test_public_dataset_loading_{bm25,dense}`; the bm25 variants configure no model at all. All four passed on CPU after the review pass (`4 passed in 323.42s`). Exact delta vs `/source/activation/tests/test_basic_dataset_loading.py`:

````diff-python
diff --git a/activation/source/activation/tests/test_basic_dataset_loading.py b/activation/activation/tests/test_basic_dataset_loading.py
index fb2350e..9e54174 100644
--- a/activation/source/activation/tests/test_basic_dataset_loading.py
+++ b/activation/activation/tests/test_basic_dataset_loading.py
@@ -28,15 +28,47 @@ TEST_DOCUMENTS = [
 
 EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
 
-def test_basic_dataset_loading():
+def _embedding_model_config(retrieval: str) -> dict:
+    """The dense variant needs the embedding model; bm25 runs with no model at all."""
+    if retrieval == "dense":
+        return dict(
+            model_configs={EMBEDDING_MODEL_ID: ModelConfig(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_ID)},
+            doc_embedding_model_name=EMBEDDING_MODEL_ID,
+        )
+    return dict()
+
+
+def _build_indexes(harness: HarnessRuntime, retrieval: str):
+    if retrieval == "dense":
+        harness.dataset_manager.build_dense_indexes()
+    else:
+        harness.dataset_manager.build_bm25_indexes()
+
+
+def _query_many(doc_index, retrieval: str, *args, **kwargs):
+    query_fn = doc_index.dense_query_many_frozen if retrieval == "dense" else doc_index.bm25_query_many_frozen
+    return query_fn(*args, **kwargs)
+
+
+def _query_docs_many(doc_index, retrieval: str, *args, **kwargs):
+    query_fn = doc_index.dense_query_docs_many_frozen if retrieval == "dense" else doc_index.bm25_query_docs_many_frozen
+    return query_fn(*args, **kwargs)
+
+
+def test_basic_dataset_loading_bm25():
+    _run_basic_dataset_loading("bm25")
+
+
+def test_basic_dataset_loading_dense():
+    _run_basic_dataset_loading("dense")
+
+
+def _run_basic_dataset_loading(retrieval: str):
     harness_config = HarnessRuntimeConfig(
-        model_configs={
-            EMBEDDING_MODEL_ID: ModelConfig(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_ID),
-        },
         doc_chunk_size_chars=8,
         doc_chunk_overlap_chars=2,
         doc_chunk_max_atomic_size=16,
-        doc_embedding_model_name=EMBEDDING_MODEL_ID,
+        **_embedding_model_config(retrieval),
     )
     harness = HarnessRuntime(harness_config)
     dataset_id: str = "test_basic"
@@ -53,7 +85,7 @@ def test_basic_dataset_loading():
     )
     harness.dataset_manager.register_dataset(dataset)
     assert dataset_id in harness.dataset_manager.loaded_datasets
-    harness.dataset_manager.build_dataset_indexes()
+    _build_indexes(harness, retrieval)
     doc_index = harness.dataset_manager.dataset_indexes[dataset_id]
     # Some chunking assertions.
     num_docs = len(TEST_DOCUMENTS)
@@ -65,18 +97,27 @@ def test_basic_dataset_loading():
         for chunk in doc_index.chunks.values()
         if not dataset.documents[chunk.doc_id].atomic
     )
-    assert doc_index.faiss_index.ntotal == len(doc_index.chunks)
+    if retrieval == "dense":
+        assert doc_index.dense_faiss_index.ntotal == len(doc_index.chunks)
+    else:
+        assert doc_index.bm25_index.num_documents == len(doc_index.chunks)
     # Some retrieval asserions.
     # Exact-mathch queries.
     queries = [doc_index.chunks["0:0"].chunk_text, doc_index.chunks["1:0"].chunk_text]
     top_k = 5
-    all_results = doc_index.query_many_frozen(queries, top_k)
+    all_results = _query_many(doc_index, retrieval, queries, top_k)
     assert all_results[0][0].doc_id == "0"
     assert all_results[1][0].doc_id == "1"
+    # Exclusions are respected.
+    excluded_results = _query_many(doc_index, retrieval, queries, top_k, excluded_doc_ids=[["0"], None])
+    assert all(chunk.doc_id != "0" for chunk in excluded_results[0])
     # Document-level retrieval dedups chunks into documents.
-    all_doc_results = doc_index.query_docs_many_frozen(queries, top_k)
+    all_doc_results = _query_docs_many(doc_index, retrieval, queries, top_k)
     assert all_doc_results[0][0].doc_id == "0"
     assert len({doc.doc_id for doc in all_doc_results[0]}) == len(all_doc_results[0])
+    if retrieval == "bm25":
+        # A query sharing no term with the corpus returns nothing rather than arbitrary chunks.
+        assert doc_index.bm25_query_many_frozen(["zzzz unmatched"], top_k) == [[]]
 
 
 def _elide(text: str, max_chars: int) -> str:
@@ -84,7 +125,15 @@ def _elide(text: str, max_chars: int) -> str:
     return text[:max_chars] + ("..." if len(text) > max_chars else "")
 
 
-def test_public_dataset_loading():
+def test_public_dataset_loading_bm25():
+    _run_public_dataset_loading("bm25")
+
+
+def test_public_dataset_loading_dense():
+    _run_public_dataset_loading("dense")
+
+
+def _run_public_dataset_loading(retrieval: str):
     # Test public dataset loading.
     BRIGHT_DOMAINS = ["biology", "economics"]
     TESTED_DATASETS = [
@@ -100,12 +149,9 @@ def test_public_dataset_loading():
     EMBEDDING_INPUT_LIMIT_CHARS = None if TARGET_DEVICE == "cuda" else 64
     TEST_BATCH_SIZE = 32
     harness_config = HarnessRuntimeConfig(
-        model_configs={
-            EMBEDDING_MODEL_ID: ModelConfig(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_ID),
-        },
-        doc_embedding_model_name=EMBEDDING_MODEL_ID,
         doc_embedding_batch_size=TEST_BATCH_SIZE,
         doc_embedding_input_limit_chars=EMBEDDING_INPUT_LIMIT_CHARS,
+        **_embedding_model_config(retrieval),
     )
     harness = HarnessRuntime(harness_config)
     loaders = {
@@ -126,7 +172,7 @@ def test_public_dataset_loading():
     loaded_datasets: list[LoadedDataset] = []
     for dataset_name in TESTED_DATASETS:
         loaded_datasets.extend(loaders[dataset_name]())
-    harness.dataset_manager.build_dataset_indexes()
+    _build_indexes(harness, retrieval)
     for dataset in loaded_datasets:
         doc_index = harness.dataset_manager.dataset_indexes[dataset.dataset_id]
         # Size assertions.
@@ -135,14 +181,16 @@ def test_public_dataset_loading():
         for example in dataset.labeled_retrieval_examples.values():
             assert example.positive_doc_ids
             assert all(doc_id in dataset.documents for doc_id in example.positive_doc_ids)
-        # Same-doc-embedding: querying with an exact stored chunk returns that chunk text first.
+        # Querying with an exact stored chunk returns that chunk text first (same embedding /
+        # strongest lexical match).
         probe_chunk_id, probe_chunk = next(iter(doc_index.chunks.items()))
         probe_text = probe_chunk.chunk_text
-        probe_results = doc_index.query_many_frozen([probe_text], top_k=1)
+        probe_results = _query_many(doc_index, retrieval, [probe_text], top_k=1)
         top_chunk = probe_results[0][0]
         assert top_chunk.chunk_text == probe_text, f"exact-chunk probe failed for {probe_chunk_id}"
         # Exclusions are respected: excluding the probe's own document removes it.
-        excluded_results = doc_index.query_many_frozen(
+        excluded_results = _query_many(
+            doc_index, retrieval,
             [probe_text],
             top_k=5,
             excluded_doc_ids=[[probe_chunk.doc_id]],
@@ -150,12 +198,13 @@ def test_public_dataset_loading():
         assert all(chunk.doc_id != probe_chunk.doc_id for chunk in excluded_results[0])
         # Manual-inspection printout: 2 labeled queries with their top 5 chunks.
         inspected = list(dataset.labeled_retrieval_examples.values())[:2]
-        all_results = doc_index.query_many_frozen(
+        all_results = _query_many(
+            doc_index, retrieval,
             [example.query for example in inspected],
             top_k=5,
             excluded_doc_ids=[example.excluded_doc_ids for example in inspected],
         )
-        print(f"\n=== {dataset.dataset_id} ===")
+        print(f"\n=== {dataset.dataset_id} [{retrieval}] ===")
         for example, results in zip(inspected, all_results):
             print(f"\nquery [{example.example_id}]: {_elide(example.query, 250)}")
             for rank, chunk in enumerate(results):
````

</details>
<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/tests/test_basic_dataset_study.py</span>
    <span class="card-oneliner">Builds only the bm25 index; no embedding model; asserts label integrity.</span>
    <span class="card-badge">Diff</span>
  </summary>

The study path needs no dense index. Exercised for real by the SkyPilot smoke (before the adapter swap; the adapter change does not touch the study code). Exact delta vs `/source/activation/tests/test_basic_dataset_study.py`:

````diff-python
diff --git a/activation/source/activation/tests/test_basic_dataset_study.py b/activation/activation/tests/test_basic_dataset_study.py
index e59f152..eb20bb8 100644
--- a/activation/source/activation/tests/test_basic_dataset_study.py
+++ b/activation/activation/tests/test_basic_dataset_study.py
@@ -22,7 +22,6 @@ if SUPPORTS_FP4:
 else:
     STUDY_MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"
 
-EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
 BRIGHT_DOMAIN = "biology"
 
 # Our canonical formula.
@@ -40,14 +39,12 @@ STUDY_ENGINE_KWARGS = {
 def test_basic_dataset_study():
     harness_config = HarnessRuntimeConfig(
         model_configs={
-            EMBEDDING_MODEL_ID: ModelConfig(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_ID),
             STUDY_MODEL_NAME: ModelConfig(
                 STUDY_MODEL_NAME,
                 STUDY_MODEL_NAME,
                 engine_kwargs=STUDY_ENGINE_KWARGS,
             ),
         },
-        doc_embedding_model_name=EMBEDDING_MODEL_ID,
         dataset_study_model_name=STUDY_MODEL_NAME,
         dataset_study_num_samples=STUDY_NUM_QUESTIONS,
         dataset_study_batch_size=STUDY_BATCH_SIZE,
@@ -68,9 +65,8 @@ def test_basic_dataset_study():
         domain=BRIGHT_DOMAIN,
         max_corpus_documents=100,
     )
-    harness.dataset_manager.build_dataset_indexes()
-    # Free the embedding model before the engine claims its memory fraction.
-    harness.loaded_models[EMBEDDING_MODEL_ID].model_to_device(FREE_DEVICE)
+    # The study path (question generation + bm25-pool labeling) needs only the bm25 index.
+    harness.dataset_manager.build_bm25_indexes()
     study_model = harness.loaded_models[STUDY_MODEL_NAME]
     try:
         harness.dataset_manager.synthesize_study_examples(dataset.dataset_id)
@@ -84,11 +80,17 @@ def test_basic_dataset_study():
         for example in synthetic:
             assert example.gold_answers and example.gold_answers[0]
             assert example.positive_doc_ids and example.positive_chunk_ids
+            # Labeling never puts one chunk on both sides.
+            assert not set(example.positive_chunk_ids) & set(example.hard_negative_chunk_ids or [])
+        # The bm25 pool yields at least some explicit hard negatives across the study set.
+        assert any(example.hard_negative_chunk_ids for example in synthetic)
         print(f"\n=== {dataset.dataset_id}: {len(synthetic)} synthetic questions ===")
         for example in synthetic[:4]:
             print(f"\n[{example.example_id}] (doc {example.positive_doc_ids[0]})")
             print(f"  Q: {example.query}")
             print(f"  A: {example.gold_answers[0]}")
+            print(f"  positives: {example.positive_chunk_ids}")
+            print(f"  hard negatives: {example.hard_negative_chunk_ids}")
         print("\n=== dataset stats ===")
         print(json.dumps(dataset.stats.summarize(), indent=2))
     finally:
````

</details>
## Smoke run

One fresh cluster (`ac-retriever-smoke`, g7e.2xlarge, RTX PRO 6000, us-east-1c after a capacity
miss in 1a), one command, torn down afterward. This is a smoke, not a measurement: it proves every
stage executes on the GPU with the real study model and the real 0.6B encoder.

`activation/cloud/sky.py · exec` (run from the project root):

```
python -m activation.cloud.sky exec ac-retriever-smoke -- env MAX_JOBS=3 NVCC_THREADS=1 \
    uv run python -m activation.training.scripts.train_bright_retriever --max-steps 10
```

Shape: BRIGHT biology, 6 examples, 30-document corpus (31 chunks), 4 study chunks, labeling,
10 training steps at batch 4. Full log: `IB/TMP/RETRIEVAL_TRAINING/sky_smoke.log`.

| Stage | Observed |
| --- | --- |
| Engine (`unsloth/Qwen3.8-27B-NVFP4`, validated formula) | cold load 647 s (dominates the wall clock), freed in 3.8 s before training |
| Study generation | 4/4 chunks → 4 questions in 57.3 s, 0 parse failures |
| Study labeling | 4 requests, avg pool 3.5 snippets, avg 4,700 prompt tokens, 1.8 s total: **1 positive / 12 negatives / 1 ambiguous**, 0 parse failures |
| Pool (`label_source=study`, TRAIN split) | 4 examples, 5 positives, 12 hard negatives |
| Training (`Qwen/Qwen3-0.6B`, bf16, cuda) | 15 candidates per batch; loss 3.69 → 0.79 over 10 steps in 4.0 s; encoder saved to `~/activation_artifacts/retriever_smoke` on the node (destroyed with it) |
| Retrieval sample | the labeled positive chunk ranked first for the first study question (not a metric) |

Small-corpus caveat worth knowing before real runs: with a 30-document corpus the BM25 top-10 is
most of the corpus, so the labeler saw few genuinely close candidates and marked most of the pool
negative. The 12,000-char pool budget was never the binding limit here (pools were 3–4 snippets).
