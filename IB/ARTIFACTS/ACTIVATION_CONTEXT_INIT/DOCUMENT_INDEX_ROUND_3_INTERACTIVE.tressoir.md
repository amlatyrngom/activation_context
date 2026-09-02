# Document index — public benchmark loaders round

This round reviews your applied round-2 source, implements the five public dataset loaders, batches query embedding, wires up the consolidated dataset statistics, and iterates the public dataset test to a clean CPU pass with inspectable retrieval output. Unlike earlier rounds, the actual code files are staged next to this document and linked below; the diffs shown are exact against your current source.

## Validation summary

- `test_basic_dataset_loading` passes in 37s with the corrected chunking and FAISS index.
- `test_public_dataset_loading` passes in 29m24s on CPU with all five datasets enabled (BRIGHT biology + economics, NQ, MS-MARCO, SciFact, SciQ), and its printed retrieval output is sensible: gold documents rank first or dominate the top 5 for every inspected query.
- Wall clock is dominated by long-sequence embedding batches, not document count — see the timing decision at the end.

Run `activation/tests/test_basic_dataset_loading.py` yourself after applying:

```bash
uv run pytest activation/tests/test_basic_dataset_loading.py -s
```

## Review of your applied round-2 source

All of these are corrected in the staged files; each maps to a hunk in the exact delta below.

- **Atomic chunk windows were skipping text.** In `_chunk_documents`, the slice used `self.chunk_size_chars` while the step used the per-document `chunk_size`, so atomic documents produced ordinary-width pieces advancing at the atomic step — the text between 8 and 16 characters of each window was silently dropped. The slice now uses `chunk_size`.
- **The empty-corpus and empty-query guards were lost.** An empty corpus reached `torch.cat([])` and an empty query list reached the tokenizer. Both guards are restored: an empty corpus leaves the index unbuilt and queries against it return empty result sets.
- **MS-MARCO deduplication crashed on the first duplicate.** The `else` branch read `documents_dedup[doc.text]` before `doc` was bound; it now reads `documents_dedup[text]`.
- **`LoadedDataset` declared `labeled_retrieval_examples` twice.** The second, messages-typed declaration silently shadowed the first. It is renamed to `labeled_messages_examples`.
- **The GPU budget branch could never trigger.** `TARGET_DEVICE.startswith("gpu")` is never true because the values are `"cuda"`/`"cpu"`; the test now checks `== "cuda"`.
- **Two small import fixes.** `activation/dataset/__init__.py` re-imported `DatasetDocument` from `document_index` (now exports `DocumentIndex`), and `dataset_manager.py`'s type-checking import said `from harness import …` (now `from activation.harness import …`).
- **`DatasetStats.to_json` referenced `np` without an import and a nonexistent field.** Completed as part of the statistics work below.

## The five loaders

Each loader is one class with a `load(harness, max_examples, …)` classmethod following your MS-MARCO shape: build documents and labeled examples, compute load stats, register with the dataset manager, return the `LoadedDataset`. Every document is loaded `atomic=True` because each source ships pre-sized retrieval units.

| Loader | Source | Documents | Labels |
| --- | --- | --- | --- |
| [bright.py](./activation/dataset/loaders/bright.py) | `xlangai/BRIGHT`, one domain per load (domain is part of the dataset id) | pre-chunked web passages | native `gold_ids` + `gold_answer` |
| [nq.py](./activation/dataset/loaders/nq.py) | `sentence-transformers/natural-questions` | positive Wikipedia passage per query, deduped by text | query → its passage |
| [msmarco.py](./activation/dataset/loaders/msmarco.py) | `microsoft/ms_marco` v1.1 (yours; two fixes) | ~10 candidate passages per record | `is_selected` positives/negatives |
| [scifact.py](./activation/dataset/loaders/scifact.py) | `BeIR/scifact` + `BeIR/scifact-qrels` | paper abstracts (title + text) | claim → gold abstracts |
| [sciq.py](./activation/dataset/loaders/sciq.py) | `allenai/sciq` | support passage per question, deduped; empty supports skipped | question → support, gold answer |

Two policies are shared and worth your sign-off:

- **Label integrity under a document budget.** For BRIGHT and SciFact, where the corpus and labels are separate, examples are accepted in order while their gold documents fit within a `max_examples` document budget; the remaining budget fills with distractor documents. An example whose gold document is missing (filtered or over budget) is dropped entirely, so every registered example's `positive_doc_ids` always resolve. This is why BRIGHT keeps only 2–4 examples at the CPU budget of 20 — gold sets are 2–7 documents each.
- **`max_doc_chars` load filter.** All five loaders accept `max_doc_chars` (default `None`); a longer document is not loaded, and examples depending on it drop with it. The test passes `8192`, which excludes for example a 31k-char BRIGHT biology document that alone would dominate CPU embedding time.

## Statistics lifecycle

`DatasetStats` now has proper defaults and fills in at three points:

- **Load time** — `initialize_dataset_stats` (implemented in [dataset_utils.py](./activation/dataset/dataset_utils.py)) computes load latency and document totals; every loader calls it.
- **Index build** — `_build_index` records per-batch embedding latency and input chars, plus `index_size_mb` from the raw embedding bytes. Directly registered datasets without a load phase get empty load stats created here.
- **Query time** — `query_many_frozen` records the incoming batch size, per-embedding-batch latency and chars, and FAISS search latency.

`to_json` reports the load totals plus counts and averages for each recorded list. The test prints it per dataset at the end.

## Query embedding is now batched

`query_many_frozen` embeds queries in `doc_embedding_batch_size` batches, sorted by length for tighter padding (your `@AI` note), and scatters embeddings back to original positions; FAISS search remains one coarse call. `_build_index` also resolves your other `@AI` note: it captures the embedding model's starting placement and restores it after building. One consequence to be aware of: at harness startup that placement is the disk sentinel, so each index build currently reloads weights (seconds from cache on CPU) — see the residency decision below.

## Staged files

New implementations (complete files, no diff shown):
[loaders/\_\_init\_\_.py](./activation/dataset/loaders/__init__.py) ·
[loaders/bright.py](./activation/dataset/loaders/bright.py) ·
[loaders/nq.py](./activation/dataset/loaders/nq.py) ·
[loaders/scifact.py](./activation/dataset/loaders/scifact.py) ·
[loaders/sciq.py](./activation/dataset/loaders/sciq.py)

Modified files (exact deltas below):
[dataset.py](./activation/dataset/dataset.py) ·
[dataset_utils.py](./activation/dataset/dataset_utils.py) ·
[dataset_manager.py](./activation/dataset/dataset_manager.py) ·
[document_index.py](./activation/dataset/document_index.py) ·
[\_\_init\_\_.py](./activation/dataset/__init__.py) ·
[loaders/msmarco.py](./activation/dataset/loaders/msmarco.py) ·
[tests/test_basic_dataset_loading.py](./activation/tests/test_basic_dataset_loading.py)

## Exact delta — modified files

`activation/dataset/dataset.py`

```diff-python
@@
 from dataclasses import dataclass, field
 from enum import StrEnum, auto
 import typing as t
+import numpy as np
@@
 @dataclass
 class DatasetStats:
-    # @AI: Set the right defaults here.
     # Computed at load time.
-    initial_load_latency: float
-    total_document_chars: int
-    total_num_documents: int
-    avg_document_chars: int
+    initial_load_latency: float = 0.0
+    total_document_chars: int = 0
+    total_num_documents: int = 0
+    avg_document_chars: int = 0
     # Computed at index build time.
-    index_size_mb: float
-    build_embedding_latencies: list[float]
-    build_embedding_input_chars: list[int]
+    index_size_mb: float = 0.0
+    build_embedding_latencies: list[float] = field(default_factory=list)
+    build_embedding_input_chars: list[int] = field(default_factory=list)
     # Computed/Updated at query time.
-    query_embedding_latencies: list[float]
-    query_embedding_input_chars: list[int]
-    query_search_latencies: list[float]
-    query_batch_sizes: list[int]
+    query_embedding_latencies: list[float] = field(default_factory=list)
+    query_embedding_input_chars: list[int] = field(default_factory=list)
+    query_search_latencies: list[float] = field(default_factory=list)
+    query_batch_sizes: list[int] = field(default_factory=list)


     def to_json(self) -> dict:
         # Returns average statistics.
+        def average(values: list) -> float:
+            return float(np.average(values)) if values else 0.0
         return {
+            "initial_load_latency": self.initial_load_latency,
+            "total_document_chars": self.total_document_chars,
+            "total_num_documents": self.total_num_documents,
+            "avg_document_chars": self.avg_document_chars,
             "index_size_mb": self.index_size_mb,
-            "avg_embedding_latency": np.average(self.embedding_latencies),
-            "avg_embedding_input_chars": ..., # @AI: Complete me...
+            "num_build_embedding_batches": len(self.build_embedding_latencies),
+            "avg_build_embedding_latency": average(self.build_embedding_latencies),
+            "avg_build_embedding_input_chars": average(self.build_embedding_input_chars),
+            "num_query_batches": len(self.query_batch_sizes),
+            "avg_query_batch_size": average(self.query_batch_sizes),
+            "avg_query_embedding_latency": average(self.query_embedding_latencies),
+            "avg_query_embedding_input_chars": average(self.query_embedding_input_chars),
+            "avg_query_search_latency": average(self.query_search_latencies),
         }
@@
     labeled_retrieval_examples: dict[str, LabeledRetrievalQAExample] = field(default_factory=dict)
     """List of labeled retrieval examples"""

-    labeled_retrieval_examples: dict[str, LabeledMessagesExample] = field(default_factory=dict)
+    labeled_messages_examples: dict[str, LabeledMessagesExample] = field(default_factory=dict)
     """List of labeled messages."""
```

`activation/dataset/dataset_utils.py`

```diff-python
@@
 import typing as t
-from .dataset import DataSplit, LoadedDataset
+from .dataset import DataSplit, DatasetStats, LoadedDataset
@@
-def initialize_dataset_stats(loaded_dataset: LoadedDataset, load_time: float):
-    ... # @AI: Implement me.
+def initialize_dataset_stats(loaded_dataset: LoadedDataset, load_time: float) -> DatasetStats:
+    """
+    Compute the load-time statistics; index/query statistics fill in later.
+    """
+    total_chars = sum(len(document.text) for document in loaded_dataset.documents.values())
+    num_documents = len(loaded_dataset.documents)
+    return DatasetStats(
+        initial_load_latency=load_time,
+        total_document_chars=total_chars,
+        total_num_documents=num_documents,
+        avg_document_chars=total_chars // num_documents if num_documents else 0,
+    )
```

`activation/dataset/dataset_manager.py`

```diff-python
@@
 if t.TYPE_CHECKING:
-    from harness import HarnessRuntime
+    from activation.harness import HarnessRuntime
```

`activation/dataset/document_index.py`

```diff-python
@@
+import time
 import typing as t
 import torch
 import faiss
 import numpy as np
@@
 )
+from .dataset_utils import initialize_dataset_stats
@@
             chunk_size = self.chunk_max_size if document.atomic else self.chunk_size_chars
             step_size = chunk_size - self.chunk_overlap_chars
             pieces = [
-                document.text[start:start + self.chunk_size_chars]
+                document.text[start:start + chunk_size]
                 for start in range(0, len(document.text), step_size)
             ]
@@
     def _build_index(self):
         """
-        Batch embed.
-        @AI: Ensure model is in same device before/after, moving to source in-between.
-        @AI: No need for try/finally logic. Just fail if something goes wrong.
+        Batch embed. The embedding model returns to its starting device afterwards;
+        each batch already lands on the source device. No try/finally: just fail.
         """
+        if self.loaded_dataset.stats is None:
+            # Directly registered datasets have no load phase; start empty load stats.
+            self.loaded_dataset.stats = initialize_dataset_stats(self.loaded_dataset, load_time=0.0)
+        stats = self.loaded_dataset.stats
+        if not self.chunks:
+            return # Empty corpus: leave the index unbuilt; queries return empty results.
         embedding_model = self.harness.loaded_models[self.embedding_model_name]
+        starting_device = embedding_model.current_model_device
         chunk_texts = [text for text, _ in self.chunks.values()]
         embeddings = []
         for start in range(0, len(chunk_texts), self.batch_size):
-            embeddings.append(
-                embedding_model.simple_vector_embed_many(
-                    chunk_texts[start:start + self.harness.harness_config.doc_embedding_batch_size],
-                )
-            )
+            batch = chunk_texts[start:start + self.batch_size]
+            batch_start_time = time.time()
+            embeddings.append(embedding_model.simple_vector_embed_many(batch))
+            stats.build_embedding_latencies.append(time.time() - batch_start_time)
+            stats.build_embedding_input_chars.append(sum(len(text) for text in batch))
+        embedding_model.model_to_device(starting_device)
         embeddings = torch.cat(embeddings).float().numpy()
         self.faiss_index = faiss.IndexFlatIP(embeddings.shape[-1])
         self.faiss_index.add(embeddings)
+        stats.index_size_mb = embeddings.nbytes / 2**20
@@
     def query_many_frozen(self, queries: list[str], top_k: int = 20) -> list[list[tuple[str, DatasetDocument]]]:
         """
         Helper to perform batched frozen queries.
         Returns, for each query, up to top_k (chunk_id, doc).
         """
+        if not queries:
+            return []
+        if top_k <= 0 or self.faiss_index is None:
+            return [[] for _query in queries]
         embedding_model = self.harness.loaded_models[self.embedding_model_name]
+        stats = self.loaded_dataset.stats
+        stats.query_batch_sizes.append(len(queries))
+        # Batch + embed in sorted order for tighter padding. Search remains one coarse call.
         queries_sorted = sorted(enumerate(queries), key=lambda x: len(x[1]))
-        # @AI: batch + embed in sorted order. Search remains coarse though.
-        query_embeddings = embedding_model.simple_vector_embed_many(queries).float().numpy()
+        query_embeddings = np.empty((len(queries), self.faiss_index.d), dtype=np.float32)
+        for start in range(0, len(queries_sorted), self.batch_size):
+            batch = queries_sorted[start:start + self.batch_size]
+            batch_texts = [query for _, query in batch]
+            batch_start_time = time.time()
+            batch_embeddings = embedding_model.simple_vector_embed_many(batch_texts).float().numpy()
+            stats.query_embedding_latencies.append(time.time() - batch_start_time)
+            stats.query_embedding_input_chars.append(sum(len(text) for text in batch_texts))
+            for (original_position, _), embedding in zip(batch, batch_embeddings):
+                query_embeddings[original_position] = embedding
         # Search.
+        search_start_time = time.time()
         chunk_ids = list(self.chunks.keys())
         _all_scores, all_indices = self.faiss_index.search(
             query_embeddings,
             k=min(top_k, len(chunk_ids)),
         )
+        stats.query_search_latencies.append(time.time() - search_start_time)
         all_results = []
```

`activation/dataset/__init__.py`

```diff-python
@@
 from .document_index import (
-    DatasetDocument,
+    DocumentIndex,
 )
```

`activation/dataset/loaders/msmarco.py`

```diff-python
@@
         version: str = "v1.1",
         split: str = "train",
+        max_doc_chars: int | None = None,
     ) -> LoadedDataset:
@@
             for index, (text, is_selected) in enumerate(zip(texts, selected_flags)):
                 text = str(text)
+                if max_doc_chars is not None and len(text) > max_doc_chars:
+                    continue # Filter out overlong passages.
                 if text not in documents_dedup:
@@
                 else:
                     # MS-Marco, I think, has no notion of stable ids. So we dedup by text.
-                    doc = documents_dedup[doc.text]
+                    doc = documents_dedup[text]
                     doc_id = doc.doc_id
```

`activation/tests/test_basic_dataset_loading.py` — `test_public_dataset_loading` is a new implementation; read it in [the staged file](./activation/tests/test_basic_dataset_loading.py). The configuration heart of it:

```python
    MAX_EXAMPLES_PER_DATASET = 1000 if TARGET_DEVICE == "cuda" else 20
    MAX_DOC_CHARS = 8192 # Overlong documents are filtered at load to keep embedding tractable.
    TEST_BATCH_SIZE = 32
```

It loads every enabled dataset, builds all indexes, asserts only reliable invariants (non-empty sizes within budget bounds, stats totals matching documents, FAISS count matching chunks, every example's positive docs present, doc lengths under the cap, and exact-chunk self-retrieval), prints 2 labeled queries with their top-5 chunks (gold hits marked, elided to 250 chars), and prints each dataset's stats. `test_basic_dataset_loading` is unchanged apart from the shared imports.

## Sample of the printed output

`test_public_dataset_loading()` prints this with `-s`:
```text
query [bright:biology:1]: Why do I only breathe out of one nostril? …
  1. GOLD [breathe_out_of_one_nostril/Nasal_cycle_2.txt:0] Benefits for breathing[edit] It has been shown that the cilia of the …
  2. GOLD [breathe_out_of_one_nostril/Nasal_cycle_5.txt:0] Research on the effects[edit] A 1994 study suggested that breathing …

query [sciq:train:0]: What type of organism is commonly used in preparation of foods such as cheese and yogurt?
  1. GOLD [sciq:train:0:0] Mesophiles grow best in moderate temperature, typically between 25°C and 40°C …
```

All twelve inspected queries rank a gold document first except scifact claim `0`, where the gold is rank 3 of 20 documents — reasonable for a claim-verification corpus.

## Timing profile and the CPU budget

Per-index build cost at the CPU budget (20 examples, 8192-char cap, batch 32):

| Index | Docs | Chars | Build batches | Build time |
| --- | ---: | ---: | ---: | ---: |
| bright/biology | 18 | 27k | 1 | 402s |
| bright/economics | 19 | 72k | 1 | 655s |
| nq | 20 | 12k | 1 | 89s |
| ms_marco | 175 | 75k | 6 | 260s |
| scifact | 20 | 37k | 1 | 236s |
| sciq | 20 | 10k | 1 | 84s |

Long sequences dominate: BRIGHT's ~7k-char documents cost minutes per batch on CPU while MS-MARCO's 175 short passages embed faster than 19 BRIGHT documents. Lowering `MAX_DOC_CHARS` is the strong lever; lowering counts further is the weak one.

<article class="decision" data-tressoir-decision data-decision-state="unresolved"
  aria-labelledby="round3-cpu-budget">
  <header class="decision-header">
    <div>
      <h3 class="decision-title" id="round3-cpu-budget">CPU run budget: is ~30 minutes acceptable, or should the cap drop?</h3>
      <p class="decision-context">Applies only to the CPU test configuration; GPU keeps 1000 examples.</p>
    </div>
    <span class="decision-state" data-decision-indicator role="status" aria-live="polite">Unresolved</span>
  </header>
  <fieldset class="decision-options">
    <legend class="visually-hidden">Decision answers</legend>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="round3.cpu_budget.accept">
      <span><strong>Keep 8192 chars / 20 examples</strong><small>~30 min CPU; documents stay closest to their native sizes.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="round3.cpu_budget.lower_cap">
      <span><strong>Drop MAX_DOC_CHARS to ~2000–3000 on CPU</strong><small>Estimated ~5–8 min; BRIGHT loses its longest gold documents and a few more examples.</small></span>
    </label>
  </fieldset>
  <div class="field decision-feedback">
    <label for="round3-cpu-budget-response">Free Response</label>
    <textarea id="round3-cpu-budget-response" rows="2" data-tressoir-input="round3.cpu_budget.feedback"
      data-tressoir-autogrow="2:6" placeholder="Another cap value, per-dataset budgets, anything else…"></textarea>
  </div>
</article>

<article class="decision" data-tressoir-decision data-decision-state="unresolved"
  aria-labelledby="round3-residency">
  <header class="decision-header">
    <div>
      <h3 class="decision-title" id="round3-residency">Embedding model placement between index builds?</h3>
      <p class="decision-context">Restoring the starting placement means each build reloads weights when the model started at the disk sentinel (seconds from cache on CPU, but a real policy question for GPU rounds).</p>
    </div>
    <span class="decision-state" data-decision-indicator role="status" aria-live="polite">Unresolved</span>
  </header>
  <fieldset class="decision-options">
    <legend class="visually-hidden">Decision answers</legend>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="round3.residency.keep_restore">
      <span><strong>Keep restore-to-starting-placement</strong><small>Each build leaves the model exactly where it found it; reloads are the accepted cost.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="round3.residency.resident_during_builds">
      <span><strong>Keep the model resident across build_document_indexes</strong><small>The dataset manager restores placement once after all indexes build.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="round3.residency.defer">
      <span><strong>Defer</strong><small>Decide with the GPU round, where residency actually costs memory.</small></span>
    </label>
  </fieldset>
  <div class="field decision-feedback">
    <label for="round3-residency-response">Free Response</label>
    <textarea id="round3-residency-response" rows="2" data-tressoir-input="round3.residency.feedback"
      data-tressoir-autogrow="2:6" placeholder="Different policy…"></textarea>
  </div>
</article>

## Cleanups done this round

- `IB/.gitignore` now ignores per-artifact copies of `tressoir-linear.css` / `tressoir-linear.js` (the canonical template copies under `IB/skills/` stay tracked, matching how `vendor/` caches are already handled). The authored `input-embeds-report.css` remains tracked.
- The superseded harness handoff snapshots (`activation/harness/loaded_model.py`, `activation/tests/test_basic_harness.py`) were removed from this folder — your source has since applied and evolved past them. The staged `activation/` tree now holds exactly this round's dataset handoff.

## Deferred

- GPU runs and performance A/B comparisons (explicitly out of scope this round).
- NQ against a full BEIR-style corpus with qrels (the pair dataset keeps the CPU round light; swapping sources later only touches `nq.py`).
- Embedding-input token caps inside `simple_vector_embed_many` (currently truncates only at the tokenizer's model max length).
