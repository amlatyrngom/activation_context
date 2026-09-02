"""
SciQ Loader.
Uses allenai/sciq: crowdsourced science exam questions with a support passage.
"""

import time
import typing as t
from datasets import load_dataset

from ..dataset_utils import (
    make_dataset_id,
    take_to_budget,
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


class SciQDataset:
    """
    Loader for SciQ questions with their supporting passage and gold answer.
    """

    @classmethod
    def load(
        cls,
        harness: "HarnessRuntime",
        max_examples: int,
        split: str = "train",
    ) -> LoadedDataset:
        start_time = time.time()
        dataset_id = make_dataset_id("sciq", split=split, n=max_examples)
        print(f"{dataset_id} - Loading")
        rows = load_dataset("allenai/sciq", split=split)
        selected = take_to_budget(
            enumerate(rows),
            budget=max_examples,
            # Questions without a support passage are not retrievable.
            filter_fn=lambda item: bool(str(item[1]["support"]).strip()),
        )
        documents: dict[str, DatasetDocument] = {}
        documents_dedup: dict[str, DatasetDocument] = {}
        examples: dict[str, LabeledRetrievalQAExample] = {}
        for ordinal, row in selected:
            text = str(row["support"]).strip()
            if text not in documents_dedup:
                doc = DatasetDocument(
                    doc_id=f"sciq:{split}:{ordinal}",
                    dataset_id=dataset_id,
                    text=text,
                    atomic=True, # One support passage is one retrieval unit.
                )
                documents[doc.doc_id] = doc
                documents_dedup[text] = doc
            else:
                doc = documents_dedup[text]
            example = LabeledRetrievalQAExample(
                example_id=f"sciq:{split}:{ordinal}",
                dataset_id=dataset_id,
                query=str(row["question"]),
                gold_answers=[str(row["correct_answer"])],
                origin=DataOrigin.NATIVE,
                split=normalize_split_name(split),
                positive_doc_ids=[doc.doc_id],
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
