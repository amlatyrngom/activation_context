"""
QuALITY (`emozilla/quality`): multiple-choice questions over long articles (about 5k words). The article is written to
/workspace/document.txt by a WriteFilesSetup at env creation and the prompt names that path; it is never inlined. Every
article is also one corpus document, so semantic_search works within and across articles. Scored by exact match on the
option letter (the option text is an alias). Rows repeat an article for each of its questions; `max_examples` counts
questions, `questions_per_article` bounds how many questions of one article are taken.
"""
from __future__ import annotations

import hashlib
import random
import time
import typing as t

from datasets import load_dataset

from ..dataset import DatasetDocument, DatasetTask, DatasetTaskMetricsKind, LoadedDataset
from ..dataset_utils import initialize_dataset_stats, make_dataset_id
from ..dataset import ANSWER_RULES, DatasetTaskKind, bare_prompt

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime

HF_DATASET = "emozilla/quality"
DOCUMENT_PATH = "/workspace/document.txt"
LETTERS = "ABCD"
WRITE_FILES_SETUP = "activation.agent.agent_env:WriteFilesSetup"


def document_setup(datum_key: str = "article", path: str = DOCUMENT_PATH) -> dict[str, dict]:
    return {"document": {"class": WRITE_FILES_SETUP, "kwargs": {"files": {path: {"datum_key": datum_key}}}}}


class QualityDataset:
    @classmethod
    def load(cls, harness: "HarnessRuntime", max_examples: int | None, seed: int = 0, split: str = "validation",
             questions_per_article: int = 2, hard_only: bool = False) -> LoadedDataset:
        start = time.time()
        rng = random.Random(seed)
        dataset_id = make_dataset_id("quality", split=split, n=max_examples, seed=seed)
        documents: dict[str, DatasetDocument] = {}
        tasks: dict[str, DatasetTask] = {}
        per_article: dict[str, int] = {}
        for row in load_dataset(HF_DATASET, split=split, streaming=True):
            if hard_only and not row.get("hard"):
                continue
            article = str(row["article"])
            article_id = hashlib.sha256(article.encode()).hexdigest()[:12]
            if per_article.get(article_id, 0) >= questions_per_article:
                continue
            per_article[article_id] = per_article.get(article_id, 0) + 1
            documents.setdefault(f"quality/{article_id}", DatasetDocument(doc_id=f"quality/{article_id}", dataset_id=dataset_id, text=article))
            options = [str(option) for option in row["options"]]
            answer = int(row["answer"])
            question = (f"The document is at {DOCUMENT_PATH} ({len(article):,} characters). Read it as needed and answer this "
                        f"question about it.\n\nQuestion: {row['question']}\n" + "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(options)))
            task_id = f"{article_id}-{per_article[article_id]}"
            tasks[task_id] = DatasetTask(
                task_id=task_id, dataset_id=dataset_id,
                task_datum={"article": article, "question": row["question"], "options": options, "hard": bool(row.get("hard"))},
                reference_metrics_kind=DatasetTaskMetricsKind.EXACT_MATCH, gold_answer=LETTERS[answer],
                gold_answer_aliases=[options[answer], f"{LETTERS[answer]}. {options[answer]}"],
                agent_prompt=bare_prompt(question, ANSWER_RULES["letter"]), task_kind=DatasetTaskKind.FILE_SEARCH,
                env_setups=document_setup(),
            )
            if max_examples is not None and len(tasks) >= max_examples:
                break
        loaded = LoadedDataset(dataset_id=dataset_id, documents=documents, scorable_tasks=tasks)
        loaded.stats = initialize_dataset_stats(loaded, load_time=time.time() - start)
        return loaded
