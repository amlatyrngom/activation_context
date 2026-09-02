"""
SciFact Loader.
Uses BeIR/scifact (paper abstracts + claims) with BeIR/scifact-qrels labels.
"""

import random
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


class SciFactDataset:
    """
    Loader for SciFact claims with their gold evidence abstracts.
    """

    @classmethod
    def load(
        cls,
        harness: "HarnessRuntime",
        max_examples: int,
        split: str = "train",
    ) -> LoadedDataset:
        """
        Take up to max_examples claims, keep every gold abstract they reference
        (overfetching documents rather than dropping claims), and add up to
        max_examples seed-sampled distractor abstracts.
        """
        start_time = time.time()
        dataset_id = make_dataset_id("scifact", split=split, n=max_examples)
        print(f"{dataset_id} - Loading")
        golds_by_query: dict[str, list[str]] = {}
        for row in load_dataset("BeIR/scifact-qrels", split=split):
            golds_by_query.setdefault(str(row["query-id"]), []).append(str(row["corpus-id"]))
        query_texts = {
            str(row["_id"]): str(row["text"])
            for row in load_dataset("BeIR/scifact", "queries", split="queries")
        }
        selected = take_to_budget(
            golds_by_query.items(),
            budget=max_examples,
        )
        wanted_doc_ids: set[str] = set()
        for _query_id, gold_ids in selected:
            wanted_doc_ids.update(gold_ids)
        corpus_rows = load_dataset("BeIR/scifact", "corpus", split="corpus")
        doc_ids = [str(doc_id) for doc_id in corpus_rows["_id"]]
        gold_positions = [p for p, doc_id in enumerate(doc_ids) if doc_id in wanted_doc_ids]
        candidate_positions = [p for p, doc_id in enumerate(doc_ids) if doc_id not in wanted_doc_ids]
        # Seeded uniform sample avoids any ordering bias in the corpus file.
        sampled_distractors = random.Random(0).sample(
            candidate_positions,
            min(max_examples, len(candidate_positions)),
        )
        documents: dict[str, DatasetDocument] = {}
        for position in sorted(gold_positions + sampled_distractors):
            row = corpus_rows[position]
            doc_id = str(row["_id"])
            title = str(row["title"]).strip()
            text = str(row["text"]).strip()
            documents[doc_id] = DatasetDocument(
                doc_id=doc_id,
                dataset_id=dataset_id,
                text=f"{title}\n\n{text}" if title else text,
                atomic=True, # One abstract is one retrieval unit.
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
