"""
DAPO-Math-17k loader (`BytedTsinghua-SIA/DAPO-Math-17k`). Unlike the retrieval loaders this populates
`scorable_tasks`: one DatasetTask per unique problem, scored by exact numeric match against
`reward_model.ground_truth`. The Hugging Face file holds each of the 17,917 problems 100 times, so
the loader streams and dedupes on `extra_info.index` until `max_examples` unique problems.
"""
from __future__ import annotations

import re
import time
import typing as t

from datasets import load_dataset

from ..dataset import DatasetTask, DatasetTaskMetricsKind, LoadedDataset
from ..dataset_utils import initialize_dataset_stats

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime

HF_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
DATASET_ID = "dapo_math"

# The dataset's own answer-format instructions, replaced so the agent uses submit_answer instead.
_HEAD_INSTRUCTION = re.compile(
    r"Solve the following math problem step by step\.\s*The last line of your response should be of the form"
    r"\s*Answer:\s*\$Answer\s*\(without quotes\)\s*where\s*\$Answer is the answer to the problem\.\s*",
    re.S,
)
_TAIL_INSTRUCTION = re.compile(r"\s*Remember to put your answer on its own line after\s*\"Answer:\"\.?\s*$", re.S)
AGENT_HEAD = ("Solve the following math problem. Use the python tool for any computation you are not certain "
              "about, then call submit_answer with the final answer as a single number.\n\n")
AGENT_TAIL = "\n\nWhen you are done, call submit_answer with only the final number."


def agent_prompt_from_dapo(prompt_text: str) -> str:
    """The problem with the dataset's "Answer:" instructions pattern-replaced by submit_answer instructions."""
    problem = _TAIL_INSTRUCTION.sub("", _HEAD_INSTRUCTION.sub("", prompt_text)).strip()
    return AGENT_HEAD + problem + AGENT_TAIL


class DapoMathDataset:
    """
    Loader for DAPO-Math-17k problems as scorable agent tasks.
    """

    @classmethod
    def load(cls, harness: "HarnessRuntime", max_examples: int | None) -> LoadedDataset:
        start = time.time()
        rows = load_dataset(HF_DATASET, split="train", streaming=True)
        dataset_id = f"{DATASET_ID}_{max_examples if max_examples is not None else 'all'}"
        tasks: dict[str, DatasetTask] = {}
        for row in rows:
            index = row["extra_info"]["index"]
            if index in tasks:
                continue
            prompt_text = row["prompt"][0]["content"]
            tasks[index] = DatasetTask(
                task_id=index,
                dataset_id=dataset_id,
                task_datum=row,
                reference_metrics_kind=DatasetTaskMetricsKind.NUMERIC_EXACT,
                gold_answer=str(row["reward_model"]["ground_truth"]),
                agent_prompt=agent_prompt_from_dapo(prompt_text),
            )
            if max_examples is not None and len(tasks) >= max_examples:
                break
        loaded = LoadedDataset(dataset_id=dataset_id, scorable_tasks=tasks)
        loaded.stats = initialize_dataset_stats(loaded, load_time=time.time() - start)
        return loaded
