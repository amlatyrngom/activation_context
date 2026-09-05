"""
MS-Marco Loader.
"""

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
    DatasetStats,
)
import time

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime


class MsMarcoDataset:
    """
    Loader for microsoft/ms_marco candidate passages and native labels.
    """

    @classmethod
    def load(
        cls,
        harness: "HarnessRuntime",
        max_examples: int|None,
        version: str = "v1.1",
        split: str = "train",
        max_corpus_documents: int|None = None,
    ) -> LoadedDataset:
        """
        Load whole query records so passages never separate from their labels.
        """
        start_time = time.time()
        dataset_id = make_dataset_id("ms_marco", version=version, split=split, n=max_examples, corpus=max_corpus_documents)
        print(f"{dataset_id} - Loading")
        rows = load_dataset(
            "microsoft/ms_marco",
            version,
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
        examples: dict[str, LabeledRetrievalQAExample] = {}
        for ordinal, row in selected:
            prefix = f"msmarco:{split}:{ordinal}"
            positives = []
            negatives = []
            texts = row["passages"]["passage_text"]
            selected_flags = row["passages"]["is_selected"]
            for index, (text, is_selected) in enumerate(zip(texts, selected_flags)):
                text = str(text)
                if text not in documents_dedup:
                    doc_id = f"{prefix}:{index}"
                    doc = DatasetDocument(
                        doc_id=doc_id,
                        dataset_id=dataset_id,
                        text=text,
                    )
                    documents[doc_id] = doc
                    documents_dedup[text] = doc
                else:
                    # MS-Marco, I think, has no notion of stable ids. So we dedup by text.
                    doc = documents_dedup[text]
                    doc_id = doc.doc_id
                if is_selected:
                    positives.append(doc_id)
                else:
                    negatives.append(doc_id)
            answers = [str(answer).strip() for answer in row.get("answers", [])]
            answers = [
                answer for answer in answers
                if answer and not answer.lower().startswith("no answer")
            ]
            if positives:
                example = LabeledRetrievalQAExample(
                    example_id=prefix,
                    dataset_id=dataset_id,
                    query=str(row["query"]),
                    origin=DataOrigin.NATIVE,
                    split=normalize_split_name(split),
                    gold_answers=answers or None,
                    positive_doc_ids=positives,
                    hard_negative_doc_ids=negatives or None,
                )
                examples[example.example_id] = example
        # Distractor pool: keep scanning rows past max_examples, passages only.
        extra_budget = extra_corpus_budget(max_corpus_documents, len(documents))
        if extra_budget is None or extra_budget > 0:
            for ordinal, row in row_iter:
                prefix = f"msmarco:{split}:{ordinal}"
                for index, text in enumerate(row["passages"]["passage_text"]):
                    text = str(text)
                    if text in documents_dedup:
                        continue
                    doc = DatasetDocument(
                        doc_id=f"{prefix}:{index}",
                        dataset_id=dataset_id,
                        text=text,
                    )
                    documents[doc.doc_id] = doc
                    documents_dedup[text] = doc
                    if extra_budget is not None:
                        extra_budget -= 1
                if extra_budget is not None and extra_budget <= 0:
                    break
        loaded_dataset = LoadedDataset(
            dataset_id=dataset_id,
            documents=documents,
            labeled_retrieval_examples=examples,
        )
        elapsed = time.time() - start_time
        loaded_dataset.stats = initialize_dataset_stats(loaded_dataset, elapsed)
        harness.dataset_manager.register_dataset(loaded_dataset)
        return loaded_dataset
