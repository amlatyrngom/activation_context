"""
Represents a frozen document index: the chunks of every document (text documents by characters,
trajectories by messages) and a bm25 index over their renderings. Dense retrieval and the faiss
index were archived with the retrieval trainer (IB/ARTIFACTS/RETRIEVAL_TRAINING/archive_slice3).
"""

import time
import typing as t
import numpy as np
from .dataset import (
    DataModality,
    LoadedDataset,
    DatasetDocument,
    DatasetDocumentChunk,
)
from .dataset_utils import message_text, render_message, render_messages
from ..common.bm25 import BM25Index

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime


MESSAGE_SEPARATOR = "\n\n"
"""What render_messages puts between messages; the chunker's offsets count it."""


class DatasetIndex:
    def __init__(self, harness: "HarnessRuntime", loaded_dataset: LoadedDataset):
        self.harness = harness
        self.loaded_dataset = loaded_dataset
        self.dataset_id = loaded_dataset.dataset_id
        self.chunk_size_chars = harness.harness_config.doc_chunk_size_chars
        self.chunk_overlap_chars = harness.harness_config.doc_chunk_overlap_chars
        self.chunks: dict[str, DatasetDocumentChunk] = dict()
        self.chunk_ids: list[str] = list() # cached computation of list(self.chunks.keys())
        self.chunk_id_indexes: dict[str, int] = dict() # map from chunk id to int index.
        self.bm25_index: BM25Index|None = None
        self._chunk_documents()


    def _text_pieces(self, text: str) -> list[tuple[str, int]]:
        """(chunk text, start offset) pieces of a text: chunk_size_chars long, chunk_overlap_chars overlapping."""
        step_size = self.chunk_size_chars - self.chunk_overlap_chars
        pieces = [
            (text[start:start + self.chunk_size_chars], start)
            for start in range(0, len(text), step_size)
        ]
        pieces = [x for x in pieces if x[0]] # filter our empty strings
        if len(pieces) > 1 and pieces[-1][1] + len(pieces[-1][0]) <= pieces[-2][1] + len(pieces[-2][0]):
            pieces.pop() # last piece is a subset of previous one.
        return pieces

    def _trajectory_pieces(self, messages: list[dict]) -> list[tuple[str, int, list[dict], list[tuple[int, int, int, bool]]]]:
        """
        (chunk text, start offset, message slice) pieces of a trajectory: whole messages packed up to
        chunk_size_chars of rendering (no overlap: a message belongs to one chunk); a single message
        longer than the chunk size is split by characters like a text, each piece a one-message slice
        holding that piece as its content. Offsets are into render_messages(messages).
        """
        pieces: list[tuple[str, int, list[dict], list[tuple[int, int, int, bool]]]] = []
        current: list[dict] = []
        current_spans: list[tuple[int, int, int, bool]] = []
        current_text = ""
        current_start = 0
        offset = 0
        for message_index, message in enumerate(messages):
            rendered = render_message(message)
            if len(rendered) > self.chunk_size_chars:
                if current:
                    pieces.append((current_text, current_start, current, current_spans))
                    current, current_text, current_spans = [], "", []
                # Split the message text itself (not the rendering), so every slice is a real message
                # holding a piece of the content; the calls ride on the last slice.
                role = message.get("role", "")
                prefix = rendered[:len(rendered) - len(rendered.split(": ", 1)[1])] if ": " in rendered else f"{role}: "
                slices = list(self._text_pieces(message_text(message))) or [("", 0)]
                for slice_index, (text, start) in enumerate(slices):
                    slice_message = {"role": role, "content": text}
                    if slice_index == len(slices) - 1 and message.get("tool_calls"):
                        slice_message["tool_calls"] = message["tool_calls"]
                    pieces.append((render_message(slice_message), offset + len(prefix) + start, [slice_message],
                                   [(message_index, start, start + len(text), slice_index == len(slices) - 1)]))
            else:
                joined = rendered if not current else current_text + MESSAGE_SEPARATOR + rendered
                if current and len(joined) > self.chunk_size_chars:
                    pieces.append((current_text, current_start, current, current_spans))
                    current, current_text, current_spans = [], "", []
                    joined = rendered
                if not current:
                    current_start = offset
                current.append(message)
                current_spans.append((message_index, 0, len(message_text(message)), True))
                current_text = joined
            offset += len(rendered) + len(MESSAGE_SEPARATOR)
        if current:
            pieces.append((current_text, current_start, current, current_spans))
        return pieces

    def _chunk_documents(self) -> None:
        """
        Chunk the documents.
        """
        print(f"{self.dataset_id} - Chunking.")
        chunks_list = [] # pre sort.
        for document in self.loaded_dataset.documents.values():
            if document.modality == DataModality.TRAJECTORY or document.text is None:
                pieces = self._trajectory_pieces(document.trajectory or [])
            else:
                pieces = [(text, start, None, None) for text, start in self._text_pieces(document.text)]
            for chunk_num, (chunk_text, chunk_start, chunk_messages, source_spans) in enumerate(pieces):
                chunk_id = f"{document.doc_id}:{chunk_num}"
                chunks_list.append((chunk_id, chunk_text, chunk_start, chunk_messages, source_spans, document))
        chunks_list.sort(key=lambda x: len(x[1])) # Sort by text length
        for (chunk_id, chunk_text, chunk_start, chunk_messages, source_spans, document) in chunks_list:
            self.chunks[chunk_id] = DatasetDocumentChunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                dataset_id=self.dataset_id,
                chunk_text=chunk_text,
                chunk_start=chunk_start,
                chunk_messages=chunk_messages,
                source_spans=source_spans,
            )
            document.chunks[chunk_id] = self.chunks[chunk_id]
        self.chunk_ids = list(self.chunks.keys())
        for idx, chunk_id in enumerate(self.chunk_ids):
            self.chunk_id_indexes[chunk_id] = idx
        if self.loaded_dataset.stats is not None:
            self.loaded_dataset.stats.num_chunks = len(self.chunk_ids)
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


    def estimate_extra_fetches(self, all_excluded_sets: list[set]) -> int:
        """
        Estimate the number of extra fetches to avoid exclusion crowding.
        """
        max_chunks = 0
        for excluded in all_excluded_sets:
            for doc_id in excluded:
                document = self.loaded_dataset.documents[doc_id]
                max_chunks = max(max_chunks, len(document.chunks))
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
        Used by bm25 retrieval to select the result chunks.
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

    def bm25_query_many_frozen(
        self,
        queries: list[str],
        top_k: int = 20,
        excluded_doc_ids: list[list[str] | None] | None = None,
    ) -> list[list[DatasetDocumentChunk]]:
        """
        Batched frozen bm25 queries: for each query, up to top_k chunks outside the excluded
        documents. Queries with no matching term return an empty list.
        """
        assert self.bm25_index is not None, "bm25_query_many_frozen needs the bm25 index (build_bm25_index)."
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
        """
        Document-level retrieval: over-fetch chunks (5x), then dedup to documents. The over-fetch
        can undershoot top_k documents when results concentrate in few documents.
        """
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
        """The chunk text, extended to 3 x the chunk size within its document when enabled (text documents; trajectory chunks as they are)."""
        doc = self.loaded_dataset.documents[chunk.doc_id]
        if not self.harness.harness_config.doc_chunk_enable_extension or doc.text is None:
            return chunk.chunk_text
        if len(chunk.chunk_text) > 3 * self.chunk_size_chars:
            # Already long
            return chunk.chunk_text
        max_extended_length = 3 * self.chunk_size_chars
        extension_length = (max_extended_length - len(chunk.chunk_text)) // 2
        extension_start = max(0, chunk.chunk_start - extension_length)
        return doc.text[extension_start:extension_start+max_extended_length]
