"""
NQ Loader.
Uses sentence-transformers/natural-questions: query/answer-passage pairs
derived from Google Natural Questions, so passages never separate from labels.
"""

import time
import typing as t
from datasets import load_dataset

from ..dataset_utils import (
    make_dataset_id,
    take_to_budget,
    extra_corpus_budget,
    normalize_split_name,
    initialize_dataset_stats,
)

from ..dataset import (
    DatasetDocument,
    NATIVE,
    DatasetQAExample,
    LoadedDataset,
)

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime


class NqDataset:
    """
    Loader for Natural Questions queries with their positive Wikipedia passages.
    """

    @classmethod
    def load(
        cls,
        harness: "HarnessRuntime",
        max_examples: int|None,
        split: str = "train",
        max_corpus_documents: int|None = None,
    ) -> LoadedDataset:
        start_time = time.time()
        dataset_id = make_dataset_id("nq", split=split, n=max_examples, corpus=max_corpus_documents)
        print(f"{dataset_id} - Loading")
        rows = load_dataset(
            "sentence-transformers/natural-questions",
            split=split,
            streaming=True,
        )
        row_iter = enumerate(rows)
        selected = take_to_budget(
            row_iter,
            budget=max_examples,
        )
        documents: dict[str, DatasetDocument] = {}
        documents_dedup: dict[str, DatasetDocument] = {}
        examples: dict[str, DatasetQAExample] = {}
        for ordinal, row in selected:
            text = str(row["answer"])
            if text not in documents_dedup:
                doc = DatasetDocument(
                    doc_id=f"nq:{split}:{ordinal}",
                    dataset_id=dataset_id,
                    text=text,
                )
                documents[doc.doc_id] = doc
                documents_dedup[text] = doc
            else:
                doc = documents_dedup[text]
            example = DatasetQAExample(
                example_id=f"nq:{split}:{ordinal}",
                dataset_id=dataset_id,
                query=str(row["query"]),
                origin=NATIVE,
                split=normalize_split_name(split),
                positive_doc_ids=[doc.doc_id],
            )
            examples[example.example_id] = example
        # Distractor pool: keep scanning rows past max_examples, docs only.
        extra_budget = extra_corpus_budget(max_corpus_documents, len(documents))
        if extra_budget is None or extra_budget > 0:
            for ordinal, row in row_iter:
                text = str(row["answer"])
                if text in documents_dedup:
                    continue
                doc = DatasetDocument(
                    doc_id=f"nq:{split}:{ordinal}",
                    dataset_id=dataset_id,
                    text=text,
                )
                documents[doc.doc_id] = doc
                documents_dedup[text] = doc
                if extra_budget is not None:
                    extra_budget -= 1
                    if extra_budget <= 0:
                        break
        loaded_dataset = LoadedDataset(
            dataset_id=dataset_id,
            documents=documents,
            labeled_qa_examples=examples,
        )
        loaded_dataset.stats = initialize_dataset_stats(loaded_dataset, time.time() - start_time)
        harness.dataset_manager.register_dataset(loaded_dataset)
        return loaded_dataset
