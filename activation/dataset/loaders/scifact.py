"""
SciFact Loader.
Uses BeIR/scifact (paper abstracts + claims) with BeIR/scifact-qrels labels.
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
    DataOrigin,
    LabeledRetrievalQAExample,
    LoadedDataset,
)

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime


class SciFactDataset:
    """
    Loader for SciFact claims with their gold evidence abstracts.
    """

    @classmethod
    def load(
        cls,
        harness: "HarnessRuntime",
        max_examples: int|None,
        split: str = "train",
        max_corpus_documents: int|None = None,
    ) -> LoadedDataset:
        """
        Take up to max_examples claims (all of them when None) and keep every
        gold abstract they reference, plus up to
        max(0, max_corpus_documents - golds) non-gold abstracts in corpus order
        (None = the whole 5k-abstract reference corpus).
        """
        start_time = time.time()
        dataset_id = make_dataset_id("scifact", split=split, n=max_examples, corpus=max_corpus_documents)
        print(f"{dataset_id} - Loading")
        query_texts = {
            str(row["_id"]): str(row["text"])
            for row in load_dataset("BeIR/scifact", "queries", split="queries")
        }
        golds_by_query: dict[str, list[str]] = {}
        for row in load_dataset("BeIR/scifact-qrels", split=split):
            query_id = str(row["query-id"])
            if query_id not in query_texts:
                continue # Defensive: a label without its claim text is unusable.
            golds_by_query.setdefault(query_id, []).append(str(row["corpus-id"]))
        selected = take_to_budget(
            golds_by_query.items(),
            budget=max_examples,
        )
        wanted_doc_ids: set[str] = set()
        for _query_id, gold_ids in selected:
            wanted_doc_ids.update(gold_ids)
        documents: dict[str, DatasetDocument] = {}
        extra_budget = extra_corpus_budget(max_corpus_documents, len(wanted_doc_ids))
        for row in load_dataset("BeIR/scifact", "corpus", split="corpus"):
            doc_id = str(row["_id"])
            if doc_id not in wanted_doc_ids:
                if extra_budget is not None and extra_budget <= 0:
                    continue
                if extra_budget is not None:
                    extra_budget -= 1
            title = str(row["title"]).strip()
            text = str(row["text"]).strip()
            documents[doc_id] = DatasetDocument(
                doc_id=doc_id,
                dataset_id=dataset_id,
                text=f"{title}\n\n{text}" if title else text,
            )
        examples: dict[str, LabeledRetrievalQAExample] = {}
        for query_id, gold_ids in selected:
            if not all(gold_id in documents for gold_id in gold_ids):
                continue # Defensive: drop claims whose gold abstracts were not found.
            example = LabeledRetrievalQAExample(
                example_id=f"scifact:{split}:{query_id}",
                dataset_id=dataset_id,
                query=query_texts[query_id],
                origin=DataOrigin.NATIVE,
                split=normalize_split_name(split),
                positive_doc_ids=gold_ids,
            )
            examples[example.example_id] = example
        loaded_dataset = LoadedDataset(
            dataset_id=dataset_id,
            documents=documents,
            labeled_retrieval_examples=examples,
        )
        loaded_dataset.stats = initialize_dataset_stats(loaded_dataset, time.time() - start_time)
        harness.dataset_manager.register_dataset(loaded_dataset)
        return loaded_dataset
