"""
Thin adapter around the bm25s library so the sparse path in DatasetIndex mirrors faiss:
build once, then search(queries, k) -> (scores, indices) with -1 padding.

Library defaults are used on purpose (Lucene-style BM25 with k1=1.5, b=0.75, English stop
words, no stemming): retrieval papers report BM25 baselines with default parameters and the
chunker already bounds document length.
"""

import time

import bm25s
import numpy as np


class BM25Index:
    def __init__(self, texts: list[str]):
        self.num_documents = len(texts)
        build_start_time = time.time()
        self.retriever = bm25s.BM25()
        self.retriever.index(
            bm25s.tokenize(texts, stopwords="en", show_progress=False),
            show_progress=False,
        )
        self.build_time = time.time() - build_start_time

    def score_all(self, query: str) -> np.ndarray:
        """bm25 score of one query against every indexed document, as a float32 vector."""
        query_tokens = bm25s.tokenize([query], stopwords="en", return_ids=False, show_progress=False)[0]
        return self.retriever.get_scores(query_tokens).astype(np.float32)

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        faiss-shaped search: (scores [Q, k] float32, indices [Q, k] int64), sorted by decreasing
        score. Slots past the last document that matched any query term hold -1 (bm25s fills
        them with arbitrary zero-score documents).
        """
        k = min(k, self.num_documents)
        query_tokens = bm25s.tokenize(queries, stopwords="en", return_ids=False, show_progress=False)
        indices, scores = self.retriever.retrieve(query_tokens, k=k, show_progress=False)
        indices = indices.astype(np.int64)
        scores = scores.astype(np.float32)
        indices[scores <= 0] = -1
        return scores, indices
