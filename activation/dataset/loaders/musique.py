"""
MuSiQue (`dgslibisey/MuSiQue`) as scorable multi-hop tasks (2 to 4 hops with distractor paragraphs). Same shape as the
HotpotQA loader: the corpus is the union of every selected question's paragraphs; token F1 against the answer and its
aliases; bare prompts (the teacher study adds the approach).
"""
from __future__ import annotations

import random
import time
import typing as t

from datasets import load_dataset

from ..dataset import DatasetDocument, DatasetTask, DatasetTaskMetricsKind, LoadedDataset
from ..dataset_utils import initialize_dataset_stats, make_dataset_id
from ..dataset import ANSWER_RULES, DatasetTaskKind, bare_prompt
from .hotpotqa import SEARCH_HINT

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime

HF_DATASET = "dgslibisey/MuSiQue"


class MuSiQueDataset:
    @classmethod
    def load(cls, harness: "HarnessRuntime", max_examples: int | None, seed: int = 0, split: str = "validation",
             answerable_only: bool = True) -> LoadedDataset:
        start = time.time()
        rng = random.Random(seed)
        dataset_id = make_dataset_id("musique", split=split, n=max_examples, seed=seed)
        documents: dict[str, DatasetDocument] = {}
        tasks: dict[str, DatasetTask] = {}
        for row in load_dataset(HF_DATASET, split=split, streaming=True):
            if answerable_only and not row.get("answerable", True):
                continue
            for paragraph in row["paragraphs"]:
                doc_id = f"wiki/{paragraph['title']}"
                documents.setdefault(doc_id, DatasetDocument(doc_id=doc_id, dataset_id=dataset_id,
                                                             text=f"{paragraph['title']}\n{paragraph['paragraph_text']}"))
            task_id = str(row["id"])
            hops = task_id.split("__")[0]
            tasks[task_id] = DatasetTask(
                task_id=task_id, dataset_id=dataset_id,
                task_datum={"question": row["question"], "hops": hops,
                            "supporting_titles": [p["title"] for p in row["paragraphs"] if p.get("is_supporting")]},
                reference_metrics_kind=DatasetTaskMetricsKind.F1, gold_answer=str(row["answer"]),
                gold_answer_aliases=[str(alias) for alias in (row.get("answer_aliases") or [])],
                agent_prompt=bare_prompt("Question: " + row["question"] + SEARCH_HINT, ANSWER_RULES["f1"]), task_kind=DatasetTaskKind.SEMANTIC_SEARCH,
            )
            if max_examples is not None and len(tasks) >= max_examples:
                break
        loaded = LoadedDataset(dataset_id=dataset_id, documents=documents, scorable_tasks=tasks)
        loaded.stats = initialize_dataset_stats(loaded, load_time=time.time() - start)
        return loaded
