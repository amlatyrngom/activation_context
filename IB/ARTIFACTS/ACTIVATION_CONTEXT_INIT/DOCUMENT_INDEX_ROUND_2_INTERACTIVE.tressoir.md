# Document index — FAISS and atomic-window round

This next slice replaces temporary torch ranking with a CPU FAISS inner-product index and changes atomic documents from truncation to max-sized overlapping windows. It is relative to the source you have already applied; it does not repeat the first-round changes.

## Review of the current slice

The current test is a good end-to-end proof: it constructs real documents, embeds them through Qwen, and retrieves exact stored chunks. Keeping it unmarked matches your request.

Three corrections are needed before the next run:

- `doc_embedding_model_name` is a model-name string, and `doc_chunk_max_atomic_size` is an integer. Their current annotations are reversed.
- The current atomic path truncates. The requested behavior is to chunk with `doc_chunk_max_atomic_size` and the ordinary overlap.
- Empty corpora and empty query batches currently reach `torch.cat([])` or the tokenizer with an empty list. The FAISS change gives both a natural empty result.

## Exact proposed delta

`pyproject.toml`

```diff-toml
@@
 dependencies = [
+    "faiss-cpu",
     "torch",
```

Then regenerate the resolved dependency file with:

```bash
uv lock
```

`activation/harness/runtime_config.py`

```diff-python
@@
-    doc_embedding_model_name: int|None = None
+    doc_embedding_model_name: str|None = None
@@
-    doc_chunk_max_atomic_size: str = 32000 # For long atomic passages.
+    doc_chunk_max_atomic_size: int = 32000 # For long atomic passages.
```

`activation/dataset/dataset.py`

```diff-python
@@
     atomic: bool = False
-    """Whether document should be chunked"""
+    """Whether this document uses the larger atomic chunk size."""
```

`activation/dataset/document_index.py`

```diff-python
@@
 import typing as t
+import faiss
 import torch
@@
-        self.chunk_embeddings: torch.Tensor | None = None # Placeholder for later faiss-based solution.
+        self.faiss_index: faiss.IndexFlatIP | None = None
@@
     def _chunk_documents(self):
         """
         Chunk the documents.
         """
+        if self.chunk_size_chars <= self.chunk_overlap_chars:
+            raise ValueError("doc_chunk_size_chars must be greater than doc_chunk_overlap_chars")
+        if self.chunk_max_size <= self.chunk_overlap_chars:
+            raise ValueError("doc_chunk_max_atomic_size must be greater than doc_chunk_overlap_chars")
         for document in self.loaded_dataset.documents.values():
-            if document.atomic:
-                pieces = [document.text[:self.chunk_max_size]]
-            else:
-                step_size = self.chunk_size_chars - self.chunk_overlap_chars
-                pieces = [
-                    document.text[start:start + self.chunk_size_chars]
-                    for start in range(0, len(document.text), step_size)
-                ]
+            chunk_size = self.chunk_max_size if document.atomic else self.chunk_size_chars
+            step_size = chunk_size - self.chunk_overlap_chars
+            pieces = [
+                document.text[start:start + chunk_size]
+                for start in range(0, len(document.text), step_size)
+            ]
@@
-        self.chunk_embeddings = torch.cat(embeddings)
+        if not embeddings:
+            return
+        embeddings = torch.cat(embeddings).float().numpy()
+        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[-1])
+        self.faiss_index.add(embeddings)
@@
-        if top_k <= 0 or self.chunk_embeddings is None:
+        if not queries:
+            return []
+        if top_k <= 0 or self.faiss_index is None:
             return [[] for _query in queries]
@@
         embedding_model = self.harness.loaded_models[self.embedding_model_name]
-        query_embeddings = embedding_model.simple_vector_embed_many(queries)
+        query_embeddings = embedding_model.simple_vector_embed_many(queries).float().numpy()
         chunk_ids = list(self.chunks)
-        result_sets = []
-        for query_embedding in query_embeddings:
-            scores = self.chunk_embeddings @ query_embedding
-            best_indices = torch.topk(scores, k=min(top_k, len(chunk_ids))).indices.tolist()
-            result_sets.append([
-                (chunk_ids[index], self.chunks[chunk_ids[index]][1])
-                for index in best_indices
-            ])
-        return result_sets
+        _scores, result_indices = self.faiss_index.search(
+            query_embeddings,
+            k=min(top_k, len(chunk_ids)),
+        )
+        return [
+            [
+                (chunk_ids[index], self.chunks[chunk_ids[index]][1])
+                for index in indices
+                if index >= 0
+            ]
+            for indices in result_indices
+        ]
```

`activation/tests/test_dataset_loading.py`

```diff-python
@@
     # Some chunking assertions.
     num_docs = len(TEST_DOCUMENTS)
     assert doc_index.chunks[f"{num_docs-1}:0"][0] == TEST_DOCUMENTS[num_docs-1][:16]
+    assert doc_index.chunks[f"{num_docs-1}:1"][0] == TEST_DOCUMENTS[num_docs-1][14:30]
     assert doc_index.chunks[f"{num_docs-2}:0"][0] == TEST_DOCUMENTS[num_docs-2]
@@
         if not document.atomic
     )
+    assert doc_index.faiss_index.ntotal == len(doc_index.chunks)
```

## Behavior after this round

| Document kind | Chunk width | Advance | Long-document outcome |
| --- | ---: | ---: | --- |
| ordinary | `doc_chunk_size_chars` | width minus overlap | overlapping ordinary chunks |
| atomic | `doc_chunk_max_atomic_size` | width minus overlap | overlapping max-sized chunks, never truncation |

Embeddings remain on CPU once created, as they do today. `IndexFlatIP` performs exact inner-product retrieval, which is cosine similarity for the already normalized model vectors. There is still no cache or persistence layer.

## Run after applying

```bash
uv lock
uv run pytest activation/tests/test_dataset_loading.py
```

## Deferred

- GPU FAISS: this CPU index is exact and keeps the first version simple. Moving only the index to GPU is a separate residency/throughput decision.
- Approximate FAISS indexes, persistence, filtering, reranking, and semantic-quality evaluation.
