# First working document index

This is a minimal end-to-end retrieval slice: construct typed documents, split ordinary text into overlapping character chunks, embed them in batches through the configured embedding model, and rank with cosine similarity. It deliberately adds neither FAISS nor persistence.

## Apply order

1. Make document construction and the harness-to-dataset-manager link valid.
2. Add batched embedding to the existing `LoadedModel` surface, then build the in-memory index.
3. Run the GPU-gated end-to-end test below.

## Exact proposed delta

`activation/dataset/dataset.py`

```diff-python
@@
 from dataclasses import dataclass, field
@@
+@dataclass
 class DatasetDocument:
```

`activation/harness/runtime.py`

```diff-python
@@
-        self.dataset_manager = DatasetManager()
+        self.dataset_manager = DatasetManager(self)
```

`activation/harness/runtime_config.py`

```diff-python
@@
-    doc_chunk_max_atomic_size: str = 32000 # For long atomic passages.
+    doc_chunk_max_atomic_size: int = 32000 # For long atomic passages.
```

`activation/harness/loaded_model.py`

```diff-python
@@
-    def simple_vector_embed(self, text: str) -> torch.Tensor:
-        \"\"\"Simply function to test embeddings\"\"\"
+    def simple_vector_embed_many(self, texts: list[str]) -> torch.Tensor:
+        \"\"\"Return one normalized embedding per text.\"\"\"
         self.model_to_device(TARGET_DEVICE)
         assert self.model_config.model_description.is_embedding_model
         inputs = self.tokenizer(
-            text,
+            texts,
+            padding=True,
             truncation=True,
             return_tensors=\"pt\",
         ).to(self.model.device)
         with torch.inference_mode():
             outputs = self.model(**inputs)
             embedding = readout_embedding(
                 last_hidden_state=outputs.last_hidden_state,
                 attention_mask=inputs[\"attention_mask\"],
                 readout_type=self.model_config.model_description.embedding_readout_type,
-            )[0]
+            )
         return embedding.detach().to(device=SOURCE_DEVICE)
+
+    def simple_vector_embed(self, text: str) -> torch.Tensor:
+        \"\"\"Return one normalized embedding.\"\"\"
+        return self.simple_vector_embed_many([text])[0]
```

`activation/dataset/document_index.py`

```diff-python
@@
 import typing as t
+import torch
 from .dataset import (
     LoadedDataset,
     DatasetDocument,
 )
@@
         self.chunk_overlap_chars = harness.harness_config.doc_chunk_overlap_chars
         self.chunk_max_size = harness.harness_config.doc_chunk_max_atomic_size
-        self.chunks: dict[str, tuple[str, DatasetDocument]] = dict # chunk_id -> (chunk content, doc)
-        # @AI: Additional fields for a FAISS index. No caching in first version of code.
+        self.chunks: dict[str, tuple[str, DatasetDocument]] = {}
+        self.chunk_embeddings: torch.Tensor | None = None
         self._chunk_documents()
         self._build_index()
 
-    def _chunk_documents():
+    def _chunk_documents(self):
         \"\"\"
-        @AI: Chunk.
-        For non-atomic passes, use size/overlap logic.
-        For atomic passes truncate by max size.
+        Create stable chunk ids of the form `document-id:chunk-number`.
+
+        Atomic documents contribute one capped chunk. Ordinary documents use
+        fixed-width overlapping character windows.
         \"\"\"
-        pass
+        if self.chunk_size_chars <= 0:
+            raise ValueError(\"doc_chunk_size_chars must be positive\")
+        if not 0 <= self.chunk_overlap_chars < self.chunk_size_chars:
+            raise ValueError(\"doc_chunk_overlap_chars must be smaller than doc_chunk_size_chars\")
+        if self.chunk_max_size <= 0:
+            raise ValueError(\"doc_chunk_max_atomic_size must be positive\")
+
+        for document in self.loaded_dataset.documents.values():
+            if document.atomic:
+                pieces = [document.text[:self.chunk_max_size]]
+            else:
+                step = self.chunk_size_chars - self.chunk_overlap_chars
+                pieces = [
+                    document.text[start:start + self.chunk_size_chars]
+                    for start in range(0, len(document.text), step)
+                ]
+            for chunk_number, text in enumerate(piece for piece in pieces if piece):
+                self.chunks[f\"{document.doc_id}:{chunk_number}\"] = (text, document)
 
     def _build_index(self):
         \"\"\"
-        Batch embed.
-        @AI: Ensure model is in same device before/after, moving to source in-between.
-        @AI: No need for try/finally logic. Just fail if something goes wrong.
+        Embed every frozen chunk once and retain CPU vectors for cosine search.
         \"\"\"
-        pass
+        model_name = self.harness.harness_config.doc_embedding_model_name
+        if model_name is None:
+            raise ValueError(\"doc_embedding_model_name must be configured\")
+        embedding_model = self.harness.loaded_models[model_name]
+        chunk_texts = [text for text, _document in self.chunks.values()]
+        embeddings = []
+        for start in range(0, len(chunk_texts), self.harness.harness_config.doc_embedding_batch_size):
+            embeddings.append(
+                embedding_model.simple_vector_embed_many(
+                    chunk_texts[start:start + self.harness.harness_config.doc_embedding_batch_size]
+                )
+            )
+        if embeddings:
+            self.chunk_embeddings = torch.cat(embeddings)
 
@@
     def query_many(self, queries: list[str], top_k: int = 20) -> list[list[tuple[str, DatasetDocument]]]:
         \"\"\"
         Helper to perform batched frozen queries.
         Returns, for each query, up to top_k (chunk_id, doc).
         \"\"\"
+        if top_k <= 0 or self.chunk_embeddings is None:
+            return [[] for _query in queries]
+
+        model_name = self.harness.harness_config.doc_embedding_model_name
+        assert model_name is not None
+        query_embeddings = self.harness.loaded_models[model_name].simple_vector_embed_many(queries)
+        chunk_ids = list(self.chunks)
+        result_sets = []
+        for query_embedding in query_embeddings:
+            scores = self.chunk_embeddings @ query_embedding
+            best_indices = torch.topk(scores, k=min(top_k, len(chunk_ids))).indices.tolist()
+            result_sets.append([
+                (chunk_ids[index], self.chunks[chunk_ids[index]][1])
+                for index in best_indices
+            ])
+        return result_sets
```

`activation/tests/test_dataset_loading.py`

```diff-python
@@
-# @AI: Create 10 short science-related statements across physics, biology, etc.
-# Have one, the last one, larger than the atomic size.
-# Have one less than the chunk size.
 TEST_DOCUMENTS = [
-    \"...\",
+    \"Gravity pulls masses together.\",
+    \"Photosynthesis makes sugar.\",
+    \"Sound travels as a wave.\",
+    \"Electrons have negative charge.\",
+    \"DNA stores cell instructions.\",
+    \"Volcanoes release molten rock.\",
+    \"Magnets attract iron.\",
+    \"Whales are mammals.\",
+    \"Light bends through glass.\",
+    \"Oxygen.\",
+    \"Atomic passage larger than sixteen characters.\",
 ]
@@
+@pytest.mark.gpu
 def test_basic_dataset_loading():
@@
         doc_chunk_size_chars=8,
         doc_chunk_overlap_chars=2,
         doc_chunk_max_atomic_size=16,
+        doc_embedding_model_name=EMBEDDING_MODEL_ID,
@@
     dataset = LoadedDataset(
         dataset_id=dataset_id,
-        documents={str(i): content for i, content in enumerate(TEST_DOCUMENTS)}
+        documents={
+            str(i): DatasetDocument(
+                doc_id=str(i),
+                dataset_id=dataset_id,
+                text=content,
+                atomic=i == len(TEST_DOCUMENTS) - 1,
+            )
+            for i, content in enumerate(TEST_DOCUMENTS)
+        },
@@
     harness.dataset_manager.build_document_indexes()
     doc_index = harness.dataset_manager.document_indexes[dataset_id]
-    # @AI: some basic assertions around the 
-    # ...
+    assert doc_index.chunks[\"9:0\"][0] == \"Oxygen.\"
+    assert doc_index.chunks[\"10:0\"][0] == TEST_DOCUMENTS[10][:16]
+    assert all(
+        len(text) <= 8
+        for text, document in doc_index.chunks.values()
+        if not document.atomic
+    )
 
     # Retrieval assertions.
-    queries = [\"...\", \"...\"] # @AI: query identical to a document
-    top_k = 5 # document should be 1; related topics in top 5. This is a more hit or miss assertion.
+    queries = [doc_index.chunks[\"0:0\"][0], doc_index.chunks[\"1:0\"][0]]
+    top_k = 5
     result_sets = doc_index.query_many(queries, top_k)
-    result_set_TOPICX = result_sets[0]
-    result_set_TOPICY = result_sets[1]
-    # ... assertions.
+    assert result_sets[0][0][1].doc_id == \"0\"
+    assert result_sets[1][0][1].doc_id == \"1\"
```

Add the missing import at the top of the test:

```diff-python
@@
+import pytest
 from activation.dataset import DatasetDocument, LoadedDataset
```

## Why this first version is enough

- `simple_vector_embed_many` retains the existing one-text API while allowing index creation and query encoding to make one model call per configured batch.
- The model's existing embedding readout normalizes each vector, so a dot product is cosine similarity.
- The retrieval assertions query the exact stored chunks. That validates the ranking contract deterministically without pretending an eight-character semantic chunk has a stable topical neighborhood.
- The test is explicitly GPU-gated because it loads and runs Qwen 3 Embedding. Run it with `uv run pytest --gpu activation/tests/test_dataset_loading.py`.

## Follow-up, deliberately deferred

- FAISS, persistence/caching, metadata filtering, and reranking.
- Word/sentence-aware chunking and a semantic-relevance benchmark.
- Releasing the embedding model after indexing. The present harness keeps it available for queries.
