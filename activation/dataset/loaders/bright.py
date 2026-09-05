"""
Loader for the BRIGHT retrieval benchmark (xlangai/BRIGHT).
Each load covers one domain, and the domain is part of its id.
"""

import ast
import time
import typing as t
from datasets import load_dataset

from ..dataset_utils import (
    make_dataset_id,
    take_to_budget,
    extra_corpus_budget,
    initialize_dataset_stats,
)

from ..dataset import (
    DatasetDocument,
    DataOrigin,
    DataSplit,
    LabeledRetrievalQAExample,
    LoadedDataset,
)

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime


def _parse_id_list(value) -> list[str]:
    """gold_ids / excluded_ids arrive as a list, or as one stringified Python literal."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return [str(item) for item in value if str(item) != "N/A"]


class BrightDataset:
    """
    Loader for one BRIGHT domain: pre-chunked passages plus labeled queries.
    """

    @classmethod
    def load(
        cls,
        harness: "HarnessRuntime",
        max_examples: int|None,
        domain: str = "biology",
        max_corpus_documents: int|None = None,
    ) -> LoadedDataset:
        """
        Take up to max_examples labeled examples (all of them when None) and keep
        every gold passage they reference. The corpus additionally holds up to
        max(0, max_corpus_documents - golds) non-gold passages in corpus order;
        None means the domain's whole reference corpus.
        """
        start_time = time.time()
        dataset_id = make_dataset_id("bright", domain=domain, n=max_examples, corpus=max_corpus_documents)
        print(f"{dataset_id} - Loading")
        selected = take_to_budget(
            load_dataset("xlangai/BRIGHT", "examples", split=domain),
            budget=max_examples,
            filter_fn=lambda row: bool(_parse_id_list(row["gold_ids"])),
        )
        wanted_doc_ids: set[str] = set()
        for row in selected:
            wanted_doc_ids.update(_parse_id_list(row["gold_ids"]))
        documents: dict[str, DatasetDocument] = {}
        extra_budget = extra_corpus_budget(max_corpus_documents, len(wanted_doc_ids))
        for row in load_dataset("xlangai/BRIGHT", "documents", split=domain):
            doc_id = str(row["id"])
            if doc_id not in wanted_doc_ids:
                if extra_budget is not None and extra_budget <= 0:
                    continue
                if extra_budget is not None:
                    extra_budget -= 1
            documents[doc_id] = DatasetDocument(
                doc_id=doc_id,
                dataset_id=dataset_id,
                text=str(row["content"]),
            )
        examples: dict[str, LabeledRetrievalQAExample] = {}
        for row in selected:
            gold_ids = _parse_id_list(row["gold_ids"])
            if not all(gold_id in documents for gold_id in gold_ids):
                continue # Defensive: drop examples whose gold passages were not found.
            excluded_ids = _parse_id_list(row["excluded_ids"])
            example = LabeledRetrievalQAExample(
                example_id=f"bright:{domain}:{row['id']}",
                dataset_id=dataset_id,
                query=str(row["query"]),
                gold_answers=[str(row["gold_answer"])] if row["gold_answer"] else None,
                origin=DataOrigin.NATIVE,
                split=DataSplit.TEST, # BRIGHT is an evaluation benchmark.
                positive_doc_ids=gold_ids,
                excluded_doc_ids=excluded_ids or None,
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
