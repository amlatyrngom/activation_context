"""
HotpotQA (`hotpotqa/hotpot_qa`, distractor setting) as scorable multi-hop tasks. The corpus is the union of the ten
context paragraphs of every selected question (two gold, eight distractors each), so the semantic_search tool has
natural distractors without the whole of Wikipedia. Answers are scored by token F1; the prompt is the bare question plus the answer rule (the teacher study adds the approach).
"""
from __future__ import annotations

import random
import time
import typing as t

from datasets import load_dataset

from ..dataset import DatasetDocument, DatasetTask, DatasetTaskMetricsKind, LoadedDataset
from ..dataset_utils import initialize_dataset_stats, make_dataset_id
from ..dataset import ANSWER_RULES, DatasetTaskKind, bare_prompt

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime

HF_DATASET = "hotpotqa/hotpot_qa"
SEARCH_HINT = ("\n\nAnswer the question using the corpus behind semantic_search (Wikipedia paragraphs; some are unrelated "
               "distractors). The answer is usually an entity, a date, a number, or yes/no.")


class HotpotQADataset:
    @classmethod
    def load(cls, harness: "HarnessRuntime", max_examples: int | None, seed: int = 0, config: str = "distractor",
             split: str = "validation") -> LoadedDataset:
        start = time.time()
        rng = random.Random(seed)
        dataset_id = make_dataset_id("hotpotqa", config=config, split=split, n=max_examples, seed=seed)
        documents: dict[str, DatasetDocument] = {}
        tasks: dict[str, DatasetTask] = {}
        for row in load_dataset(HF_DATASET, config, split=split, streaming=True):
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
                doc_id = f"wiki/{title}"
                documents.setdefault(doc_id, DatasetDocument(doc_id=doc_id, dataset_id=dataset_id, text=f"{title}\n{''.join(sentences)}"))
            task_id = str(row["id"])
            tasks[task_id] = DatasetTask(
                task_id=task_id, dataset_id=dataset_id,
                task_datum={"question": row["question"], "type": row["type"], "level": row["level"],
                            "supporting_titles": list(row["supporting_facts"]["title"])},
                reference_metrics_kind=DatasetTaskMetricsKind.F1, gold_answer=str(row["answer"]),
                agent_prompt=bare_prompt("Question: " + row["question"] + SEARCH_HINT, ANSWER_RULES["f1"]), task_kind=DatasetTaskKind.SEMANTIC_SEARCH,
            )
            if max_examples is not None and len(tasks) >= max_examples:
                break
        loaded = LoadedDataset(dataset_id=dataset_id, documents=documents, scorable_tasks=tasks)
        loaded.stats = initialize_dataset_stats(loaded, load_time=time.time() - start)
        return loaded
