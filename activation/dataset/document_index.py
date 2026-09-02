"""
Represents a frozen document index. Different from warm query/question embedding.
"""

import time
import typing as t
import torch
import faiss
import numpy as np
from dataclasses import dataclass
from .dataset import (
    LoadedDataset,
    DatasetDocument,
)
from .dataset_utils import safe_truncate_embedding_chunk

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime




class DocumentIndex:
    def __init__(self, harness: "HarnessRuntime", loaded_dataset: LoadedDataset):
        self.harness = harness
        self.loaded_dataset = loaded_dataset
        self.dataset_id = loaded_dataset.dataset_id
        self.chunk_size_chars = harness.harness_config.doc_chunk_size_chars
        self.chunk_overlap_chars = harness.harness_config.doc_chunk_overlap_chars
        self.chunk_max_size = harness.harness_config.doc_chunk_max_atomic_size
        self.embedding_model_name = harness.harness_config.doc_embedding_model_name
        self.batch_size = harness.harness_config.doc_embedding_batch_size
        self.embedding_input_limit = harness.harness_config.doc_embedding_input_limit_chars
        self.chunks: dict[str, tuple[str, DatasetDocument]] = dict() # chunk_id -> (chunk content, doc)
        self.chunk_ids: list[str] = list() # cached computation of list(self.chunks.key())
        self.faiss_index: faiss.IndexFlatIP | None = None
        print(f"{self.dataset_id} - Chunking.")
        self._chunk_documents()
        print(f"{self.dataset_id} - Chunked to {len(self.chunk_ids)} chunks.")
        print(f"{self.dataset_id} - Building index.")
        self._build_index()
        print(f"{self.dataset_id} - Built index.")

    def _chunk_documents(self):
        """
        Chunk the documents.
        """
        chunks_list = [] # pre sort.
        for document in self.loaded_dataset.documents.values():
            chunk_size = self.chunk_max_size if document.atomic else self.chunk_size_chars
            step_size = chunk_size - self.chunk_overlap_chars
            pieces = [
                document.text[start:start + chunk_size]
                for start in range(0, len(document.text), step_size)
            ]
            pieces = [x for x in pieces if x] # filter our empty strings
            for chunk_num, chunk_text in enumerate(pieces):
                chunk_id = f"{document.doc_id}:{chunk_num}"
                chunks_list.append((chunk_id, chunk_text, document))
        chunks_list.sort(key=lambda x: len(x[1])) # Sort by text length
        for (chunk_id, chunk_text, document) in chunks_list:
            self.chunks[chunk_id] = (chunk_text, document)
        self.chunk_ids = list(self.chunks.keys())

    def _build_index(self):
        """
        Build index by batch embedding.
        """
        stats = self.loaded_dataset.stats
        embedding_model = self.harness.loaded_models[self.embedding_model_name]
        chunk_texts = [
            safe_truncate_embedding_chunk(text, self.embedding_input_limit)
            for text, _ in self.chunks.values()
        ]
        embeddings = []
        reporting_interval = 0
        total_embedding_time = 0
        for start in range(0, len(chunk_texts), self.batch_size):
            batch = chunk_texts[start:start + self.batch_size]
            batch_start_time = time.time()
            embeddings.append(embedding_model.simple_vector_embed_many(batch))
            elapsed = time.time() - batch_start_time
            stats.build_embedding_latencies.append(elapsed)
            total_embedding_time += elapsed
            stats.build_embedding_input_chars.append(sum(len(text) for text in batch))
            if reporting_interval <= 0:
                print(f"{self.dataset_id} - Index Embedding: {start + len(batch)}/{len(chunk_texts)}. {total_embedding_time:.2f}s.")
                reporting_interval = max(1, len(chunk_texts) // 20)
            reporting_interval -= len(batch)

        print(f"{self.dataset_id} - Building faiss.")
        faiss_build_start_time = time.time()
        embeddings = torch.cat(embeddings).float().numpy()
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[-1])
        self.faiss_index.add(embeddings)
        stats.index_size_mb = embeddings.nbytes / 2**20
        elapsed = time.time() - faiss_build_start_time
        print(f"{self.dataset_id} - Built faiss in {elapsed:.2f}s. Size={stats.index_size_mb:.2f}MB.")


    def _build_excluded_sets(self, num_queries: int, excluded_doc_ids: list[list[str] | None] | None = None) -> tuple[list[set], int]:
        """
        Return (excluded sets, additional fetch count)
        """
        if not excluded_doc_ids:
            excluded_doc_ids = [[] for _ in range(num_queries)]
        excluded_sets = [set(excluded or []) for excluded in excluded_doc_ids]
        if any(len(excluded) > 0 for excluded in excluded_sets):
            return (excluded_sets, 10)
        return (excluded_sets, 0)
        
    def query_many_frozen(
        self,
        queries: list[str],
        top_k: int = 20,
        excluded_doc_ids: list[list[str] | None] | None = None,
    ) -> list[list[tuple[str, DatasetDocument]]]:
        """
        Helper to perform batched frozen queries.
        Returns, for each query, up to top_k (chunk_id, doc).
        """
        stats = self.loaded_dataset.stats
        embedding_model = self.harness.loaded_models[self.embedding_model_name]
        stats.query_batch_sizes.append(len(queries))
        # Batch query
        queries_sorted = sorted(enumerate(queries), key=lambda x: len(x[1]))
        query_embeddings = np.empty((len(queries), self.faiss_index.d), dtype=np.float32)
        total_embedding_time = 0
        reporting_interval = 0
        for start in range(0, len(queries_sorted), self.batch_size):
            batch = queries_sorted[start:start + self.batch_size]
            batch_texts = [
                safe_truncate_embedding_chunk(query, self.embedding_input_limit)
                for _, query in batch
            ]
            batch_start_time = time.time()
            batch_embeddings = embedding_model.simple_vector_embed_many(batch_texts).float().numpy()
            elapsed = time.time() - batch_start_time
            total_embedding_time += elapsed
            stats.query_embedding_latencies.append(elapsed)
            stats.query_embedding_input_chars.append(sum(len(text) for text in batch_texts))
            for (original_position, _), embedding in zip(batch, batch_embeddings):
                query_embeddings[original_position] = embedding
            if reporting_interval <= 0:
                print(f"{self.dataset_id} - Query Embedding: {start + len(batch)}/{len(queries)}. {total_embedding_time:.2f}s.")
                reporting_interval = max(1, len(queries) // 20)
            reporting_interval -= len(batch)

        # Search.
        search_start_time = time.time()
        all_excluded_sets, extra_fetches = self._build_excluded_sets(
            len(queries), excluded_doc_ids
        )
        _all_scores, all_indices = self.faiss_index.search(
            query_embeddings,
            k=min(top_k + extra_fetches, len(self.chunk_ids)), # +10 handles most cases of exclusion.
        )
        stats.query_search_latencies.append(time.time() - search_start_time)
        all_results = []
        for indices, excluded in zip(all_indices, all_excluded_sets):
            results = []
            for index in indices:
                if index < 0:
                    continue
                chunk_id = self.chunk_ids[index]
                _, doc = self.chunks[chunk_id]
                if doc.doc_id in excluded:
                    continue
                results.append((chunk_id, doc))
                if len(results) >= top_k:
                    break
            all_results.append(results)
        return all_results
