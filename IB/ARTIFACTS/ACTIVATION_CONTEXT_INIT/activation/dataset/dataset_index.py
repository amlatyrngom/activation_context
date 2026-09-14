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
    DatasetDocumentChunk,
)
from .dataset_utils import safe_truncate_embedding_chunk
from ..common.bm25 import BM25Index

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime



class DatasetIndex:
    def __init__(self, harness: "HarnessRuntime", loaded_dataset: LoadedDataset):
        self.harness = harness
        self.loaded_dataset = loaded_dataset
        self.dataset_id = loaded_dataset.dataset_id
        self.chunk_size_chars = harness.harness_config.doc_chunk_size_chars
        self.chunk_overlap_chars = harness.harness_config.doc_chunk_overlap_chars
        self.embedding_model_name = harness.harness_config.doc_embedding_model_name
        self.batch_size = harness.harness_config.doc_embedding_batch_size
        self.embedding_input_limit = harness.harness_config.doc_embedding_input_limit_chars
        self.chunks: dict[str, DatasetDocumentChunk] = dict()
        self.chunk_ids: list[str] = list() # cached computation of list(self.chunks.keys())
        self.chunk_id_indexes: dict[str, int] = dict() # map from chunk id to int index.
        self.dense_faiss_index: faiss.IndexFlatIP | None = None
        self.bm25_index: BM25Index|None = None
        self._chunk_documents()



    def _chunk_documents(self):
        """
        Chunk the documents.
        """
        print(f"{self.dataset_id} - Chunking.")
        chunks_list = [] # pre sort.
        for document in self.loaded_dataset.documents.values():
            step_size = self.chunk_size_chars - self.chunk_overlap_chars
            pieces = [
                (document.text[start:start + self.chunk_size_chars], start)
                for start in range(0, len(document.text), step_size)
            ]
            pieces = [x for x in pieces if x[0]] # filter our empty strings
            if len(pieces) > 1 and pieces[-1][1] + len(pieces[-1][0]) <= pieces[-2][1] + len(pieces[-2][0]):
                pieces.pop() # last piece is a subset of previous one.
            for chunk_num, (chunk_text, chunk_start) in enumerate(pieces):
                chunk_id = f"{document.doc_id}:{chunk_num}"
                chunks_list.append((chunk_id, chunk_text, chunk_start, document))
        chunks_list.sort(key=lambda x: len(x[1])) # Sort by text length
        for (chunk_id, chunk_text, chunk_start, document) in chunks_list:
            self.chunks[chunk_id] = DatasetDocumentChunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                dataset_id=self.dataset_id,
                chunk_text=chunk_text,
                chunk_start=chunk_start,
            )
            document.chunks[chunk_id] = self.chunks[chunk_id]
        self.chunk_ids = list(self.chunks.keys())
        for idx, chunk_id in enumerate(self.chunk_ids):
            self.chunk_id_indexes[chunk_id] = idx
        print(f"{self.dataset_id} - Chunked to {len(self.chunk_ids)} chunks.")
        return

    def build_bm25_index(self):
        """Build a bm25 index."""
        if self.bm25_index is not None:
            return
        stats = self.loaded_dataset.stats
        print(f"{self.dataset_id} - Building bm25.")
        self.bm25_index = BM25Index([chunk.chunk_text for chunk in self.chunks.values()])
        stats.bm25_build_latency = self.bm25_index.build_time
        print(f"{self.dataset_id} - Built bm25 in {stats.bm25_build_latency:.2f}s.")


    def build_dense_index(self):
        """
        Build a dense index by batch embedding.
        """
        if self.dense_faiss_index is not None:
            return
        print(f"{self.dataset_id} - Building dense index.")
        stats = self.loaded_dataset.stats
        embedding_model = self.harness.loaded_models[self.embedding_model_name]
        chunk_texts = [
            safe_truncate_embedding_chunk(chunk.chunk_text, self.embedding_input_limit)
            for chunk in self.chunks.values()
        ]
        embeddings = []
        reporting_interval = 0
        total_embedding_time = 0
        for start in range(0, len(chunk_texts), self.batch_size):
            batch = chunk_texts[start:start + self.batch_size]
            batch_start_time = time.time()
            embeddings.append(embedding_model.simple_vector_embed_many(batch))
            elapsed = time.time() - batch_start_time
            stats.dense_build_embedding_latencies.append(elapsed)
            total_embedding_time += elapsed
            stats.dense_build_embedding_input_chars.append(sum(len(text) for text in batch))
            if reporting_interval <= 0:
                print(f"{self.dataset_id} - Index Embedding: {start + len(batch)}/{len(chunk_texts)}. {total_embedding_time:.2f}s.")
                reporting_interval = max(1, len(chunk_texts) // 20)
            reporting_interval -= len(batch)

        print(f"{self.dataset_id} - Building faiss index.")
        faiss_build_start_time = time.time()
        embeddings = torch.cat(embeddings).float().numpy()
        self.dense_faiss_index = faiss.IndexFlatIP(embeddings.shape[-1])
        self.dense_faiss_index.add(embeddings)
        stats.dense_index_size_mb = embeddings.nbytes / 2**20
        elapsed = time.time() - faiss_build_start_time
        print(f"{self.dataset_id} - Built faiss index in {elapsed:.2f}s. Size={stats.dense_index_size_mb:.2f}MB.")


    def estimate_extra_fetches(self, all_excluded_sets: list[set]) -> int:
        """
        Estimate the number of extra fetches to avoid exclusion crowding.
        """
        max_chunks = 0
        for excluded in all_excluded_sets:
            for doc_id in excluded:
                document = self.loaded_dataset.documents[doc_id]
                step_size = self.chunk_size_chars - self.chunk_overlap_chars
                num_chunks = -(-len(document.text) // step_size)  # ceil division
                max_chunks = max(max_chunks, num_chunks)
        return 5 + max_chunks if max_chunks else 0

    def _build_excluded_sets(self, num_queries: int, excluded_doc_ids: list[list[str] | None] | None = None) -> tuple[list[set], int]:
        """
        Return (excluded sets, additional fetch count)
        """
        if not excluded_doc_ids:
            excluded_doc_ids = [[] for _ in range(num_queries)]
        excluded_sets = [set(excluded or []) for excluded in excluded_doc_ids]
        extra_fetches = self.estimate_extra_fetches(excluded_sets)
        return (excluded_sets, extra_fetches)

    
    def _collect_chunk_results(
        self,
        all_indices: np.ndarray,
        all_excluded_sets: list[set],
        top_k: int,
    ) -> list[list[DatasetDocumentChunk]]:
        """
        Used by bm25/dense retrieval to select the result chunks.
        """
        all_results = []
        for indices, excluded in zip(all_indices, all_excluded_sets):
            results = []
            for index in indices:
                if index < 0:
                    continue
                chunk_id = self.chunk_ids[index]
                chunk = self.chunks[chunk_id]
                if chunk.doc_id in excluded:
                    continue
                results.append(chunk)
                if len(results) >= top_k:
                    break
            all_results.append(results)
        return all_results
        
    def dense_query_many_frozen(
        self,
        queries: list[str],
        top_k: int = 20,
        excluded_doc_ids: list[list[str] | None] | None = None,
    ) -> list[list[DatasetDocumentChunk]]:
        """
        Helper to perform batched frozen dense queries.
        Returns, for each query, up to top_k chunks.
        """
        stats = self.loaded_dataset.stats
        embedding_model = self.harness.loaded_models[self.embedding_model_name]
        stats.query_batch_sizes.append(len(queries))
        # Batch query
        queries_sorted = sorted(enumerate(queries), key=lambda x: len(x[1]))
        query_embeddings = np.empty((len(queries), self.dense_faiss_index.d), dtype=np.float32)
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
            stats.dense_query_embedding_latencies.append(elapsed)
            stats.dense_query_embedding_input_chars.append(sum(len(text) for text in batch_texts))
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
        _all_scores, all_indices = self.dense_faiss_index.search(
            query_embeddings,
            k=min(top_k + extra_fetches, len(self.chunk_ids)), # +10 handles most cases of exclusion.
        )
        stats.dense_query_search_latencies.append(time.time() - search_start_time)
        return self._collect_chunk_results(all_indices, all_excluded_sets, top_k)

    def dense_query_docs_many_frozen(
        self,
        queries: list[str],
        top_k: int = 20,
        excluded_doc_ids: list[list[str] | None] | None = None,
    ) -> list[list[DatasetDocument]]:
        """
        Document-level retrieval: over-fetch chunks, then dedup to documents.
        For domains that necessitate full documents. The 5x over-fetch can
        undershoot top_k documents when results concentrate in few documents.
        """
        all_chunk_results = self.dense_query_many_frozen(
            queries,
            top_k=5 * top_k,
            excluded_doc_ids=excluded_doc_ids,
        )
        return [
            self.get_top_documents(chunks)[:top_k]
            for chunks in all_chunk_results
        ]

    def bm25_query_many_frozen(
        self,
        queries: list[str],
        top_k: int = 20,
        excluded_doc_ids: list[list[str] | None] | None = None,
    ) -> list[list[DatasetDocumentChunk]]:
        """
        BM25 counterpart of dense_query_many_frozen: same exclusion semantics,
        no model involved. Queries with no matching term return an empty list.
        """
        stats = self.loaded_dataset.stats
        stats.query_batch_sizes.append(len(queries))
        search_start_time = time.time()
        all_excluded_sets, extra_fetches = self._build_excluded_sets(
            len(queries), excluded_doc_ids
        )
        _all_scores, all_indices = self.bm25_index.search(
            queries,
            k=min(top_k + extra_fetches, len(self.chunk_ids)),
        )
        stats.bm25_query_search_latencies.append(time.time() - search_start_time)
        return self._collect_chunk_results(all_indices, all_excluded_sets, top_k)

    def bm25_query_docs_many_frozen(
        self,
        queries: list[str],
        top_k: int = 20,
        excluded_doc_ids: list[list[str] | None] | None = None,
    ) -> list[list[DatasetDocument]]:
        """BM25 counterpart of dense_query_docs_many_frozen (same 5x over-fetch)."""
        all_chunk_results = self.bm25_query_many_frozen(
            queries,
            top_k=5 * top_k,
            excluded_doc_ids=excluded_doc_ids,
        )
        return [
            self.get_top_documents(chunks)[:top_k]
            for chunks in all_chunk_results
        ]


    def bm25_rank_doc_chunks(
        self,
        query: str,
        doc_ids: list[str],
    ) -> dict[str, list[tuple[DatasetDocumentChunk, float]]]:
        """
        Rank the chunks in the given documents by their bm25 score.
        """
        assert self.bm25_index is not None, "bm25_rank_doc_chunks needs the bm25 index."
        scores = self.bm25_index.score_all(query)
        ranked: dict[str, list[tuple[DatasetDocumentChunk, float]]] = {}
        for doc_id in doc_ids:
            document = self.loaded_dataset.documents[doc_id]
            scored = [
                (chunk, float(scores[self.chunk_id_indexes[chunk_id]]))
                for chunk_id, chunk in document.chunks.items()
            ]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            ranked[doc_id] = scored
        return ranked


    def get_top_documents(self, chunks: list[DatasetDocumentChunk]) -> list[DatasetDocument]:
        seen = set()
        docs = list()
        for chunk in chunks:
            if chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            docs.append(self.loaded_dataset.documents[chunk.doc_id])
        return docs

    def get_chunk_section(self, chunk: DatasetDocumentChunk) -> str:
        if not self.harness.harness_config.doc_chunk_enable_extension:
            return chunk.chunk_text
        if len(chunk.chunk_text) > 3 * self.chunk_size_chars:
            # Already long
            return chunk.chunk_text
        max_extended_length = 3 * self.chunk_size_chars
        extension_length = (max_extended_length - len(chunk.chunk_text)) // 2
        extension_start = max(0, chunk.chunk_start - extension_length)
        doc = self.loaded_dataset.documents[chunk.doc_id]
        return doc.text[extension_start:extension_start+max_extended_length]
