"""
NarrativeQA (`deepmind/narrativeqa`): free-form questions over whole books and film scripts (tens to hundreds of
thousands of words). The full text is written to /workspace/document.txt by a WriteFilesSetup; the prompt names the
path. Token F1 against the two reference answers. `max_document_chars` skips the longest documents; rows repeat a
document for each of its questions and `questions_per_document` bounds the take.
"""
from __future__ import annotations

import random
import time
import typing as t

from datasets import load_dataset

from ..dataset import DatasetDocument, DatasetTask, DatasetTaskMetricsKind, LoadedDataset
from ..dataset_utils import initialize_dataset_stats, make_dataset_id
from ..dataset import ANSWER_RULES, DatasetTaskKind, bare_prompt
from .quality import DOCUMENT_PATH, document_setup

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime

HF_DATASET = "deepmind/narrativeqa"


class NarrativeQADataset:
    @classmethod
    def load(cls, harness: "HarnessRuntime", max_examples: int | None, seed: int = 0, split: str = "validation",
             questions_per_document: int = 2, max_document_chars: int = 600_000) -> LoadedDataset:
        start = time.time()
        rng = random.Random(seed)
        dataset_id = make_dataset_id("narrativeqa", split=split, n=max_examples, seed=seed)
        documents: dict[str, DatasetDocument] = {}
        tasks: dict[str, DatasetTask] = {}
        per_document: dict[str, int] = {}
        for row in load_dataset(HF_DATASET, split=split, streaming=True):
            document = row["document"]
            text = str(document["text"])
            doc_key = str(document["id"])
            if len(text) > max_document_chars or per_document.get(doc_key, 0) >= questions_per_document:
                continue
            per_document[doc_key] = per_document.get(doc_key, 0) + 1
            documents.setdefault(f"narrativeqa/{doc_key}", DatasetDocument(doc_id=f"narrativeqa/{doc_key}", dataset_id=dataset_id, text=text))
            answers = [str(answer["text"]) for answer in row["answers"]]
            kind = str(document.get("kind", "document"))
            noun = {"gutenberg": "book", "movie": "film script"}.get(kind, "document")
            question = (f"The {noun} is at {DOCUMENT_PATH} ({len(text):,} characters). Read it as needed and answer this question "
                        f"about it.\n\nQuestion: {row['question']['text']}")
            task_id = f"{doc_key}-{per_document[doc_key]}"
            tasks[task_id] = DatasetTask(
                task_id=task_id, dataset_id=dataset_id,
                task_datum={"article": text, "question": row["question"]["text"], "kind": kind,
                            "summary": str((document.get("summary") or {}).get("text", ""))},
                reference_metrics_kind=DatasetTaskMetricsKind.F1, gold_answer=answers[0], gold_answer_aliases=answers[1:],
                agent_prompt=bare_prompt(question, ANSWER_RULES["f1"]), task_kind=DatasetTaskKind.FILE_SEARCH,
                env_setups=document_setup(),
            )
            if max_examples is not None and len(tasks) >= max_examples:
                break
        loaded = LoadedDataset(dataset_id=dataset_id, documents=documents, scorable_tasks=tasks)
        loaded.stats = initialize_dataset_stats(loaded, load_time=time.time() - start)
        return loaded
