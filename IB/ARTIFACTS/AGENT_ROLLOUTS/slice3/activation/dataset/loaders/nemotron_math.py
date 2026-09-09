"""
Loader for NVIDIA's Nemotron-SFT-Math-v4 (`nvidia/Nemotron-SFT-Math-v4`): math solutions as `cot`
rows (one assistant turn whose `reasoning_content` holds the solution) or `tir` rows (tool-integrated
reasoning with `stateful_python_code_exec(code)`, mapped onto our `python(code)`). The dataset cards
name no benchmark for their decontamination, so the loader takes `excluded_problems` (e.g. the AIME
2024/2025 statements from loaders/aime.py) and drops any problem sharing a 13-word window with one of
them (`stats.num_decontaminated`). The single `train` split is sorted by subset (cot then tir), so each
subset reads only its shards by default.
"""
from __future__ import annotations

import time
import typing as t

from datasets import load_dataset

from ..dataset import EXTERNAL, DataModality, DatasetDocument, LoadedDataset
from ..dataset_utils import initialize_dataset_stats, make_dataset_id, take_to_budget
from .trajectory_utils import (
    ngram_windows,
    parity_split,
    parse_tool_definitions,
    reformat_trajectory,
    structured_calls,
    trajectory_chars,
)

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime

HF_DATASET = "nvidia/Nemotron-SFT-Math-v4"
NUM_SHARDS = 12
SUBSET_SHARDS = {"cot": range(0, 8), "tir": range(7, NUM_SHARDS)}   # v4 rows are cot then tir; the boundary (row 285,516) lies inside shard 7 (rows 264,625-305,802), so both subsets read it
SYSTEM_PROMPT = "You are a math problem solver. Use the python tool for computations and end with the final answer."
DECONTAMINATION_NGRAM = 13


def normalize_nemotron_row(row: dict) -> tuple[list[dict], list[dict]]:
    """One source row -> (normalized messages, tool definitions), see trajectory_utils."""
    normalized = []
    for message in row["messages"]:
        role = message["role"]
        if role == "assistant":
            normalized.append({"role": role, "content": message.get("content") or "",
                               "reasoning": message.get("reasoning_content") or None,
                               "tool_calls": structured_calls(message.get("tool_calls"))})
        else:
            normalized.append({"role": role, "content": message.get("content") or ""})
    return normalized, parse_tool_definitions(row.get("tools"))


class NemotronMathDataset:
    """Loader for one subset of Nemotron-SFT-Math-v4 as trajectory documents."""

    @classmethod
    def load(
        cls,
        harness: "HarnessRuntime",
        max_examples: int | None,
        subset: str | None = "tir",
        excluded_problems: list[str] | None = None,
        min_chars: int = 0,
        min_tool_calls: int = 0,
        hf_dataset: str = HF_DATASET,
        data_files: list[str] | None = None,
    ) -> LoadedDataset:
        """
        Streams the subset's shards (`data_files` overrides the shard choice; `hf_dataset` an earlier
        version as the leak fallback) and keeps the first max_examples problems that pass the filters:
        not sharing a 13-word window with an excluded problem, at least `min_chars` in our rendering, and
        at least `min_tool_calls` calls (some `tir` rows never call the tool).
        """
        start_time = time.time()
        dataset_id = make_dataset_id("nemotron_math", subset=subset or "all", n=max_examples, min=min_chars)
        print(f"{dataset_id} - Loading")
        excluded_windows = ngram_windows(excluded_problems or [], DECONTAMINATION_NGRAM)
        if data_files is None and subset in SUBSET_SHARDS and hf_dataset == HF_DATASET:
            data_files = [f"data/train-{index:05d}-of-{NUM_SHARDS:05d}.parquet" for index in SUBSET_SHARDS[subset]]
        documents: dict[str, DatasetDocument] = {}
        num_decontaminated = 0

        def convert(row: dict) -> bool:
            nonlocal num_decontaminated
            if subset is not None and row.get("subset") != subset:
                return False
            if excluded_windows and ngram_windows([row["problem"]], DECONTAMINATION_NGRAM) & excluded_windows:
                num_decontaminated += 1
                return False
            normalized, tools = normalize_nemotron_row(row)
            messages, kwargs = reformat_trajectory(normalized, tools)
            if trajectory_chars(messages) < min_chars or not any(message["role"] == "assistant" for message in messages):
                return False
            if sum(len(message.get("tool_calls") or []) for message in messages) < min_tool_calls:
                return False
            doc_id = str(row["uuid"])
            kwargs.update({"system_prompt": SYSTEM_PROMPT, "problem": row["problem"], "answer": row.get("expected_answer"),
                           "subset": row.get("subset"), "source": row.get("source")})
            documents[doc_id] = DatasetDocument(doc_id=doc_id, dataset_id=dataset_id, trajectory=messages, trajectory_kwargs=kwargs,
                                                modality=DataModality.TRAJECTORY, origin=EXTERNAL, split=parity_split(doc_id))
            return True

        rows = load_dataset(hf_dataset, split="train", streaming=True, data_files=data_files) if data_files else \
            load_dataset(hf_dataset, split="train", streaming=True)
        take_to_budget(rows, budget=max_examples, filter_fn=convert)
        loaded_dataset = LoadedDataset(dataset_id=dataset_id, documents=documents)
        loaded_dataset.stats = initialize_dataset_stats(loaded_dataset, time.time() - start_time)
        loaded_dataset.stats.num_decontaminated = num_decontaminated
        if num_decontaminated:
            print(f"{dataset_id} - {num_decontaminated} problems dropped by decontamination")
        harness.dataset_manager.register_dataset(loaded_dataset)
        return loaded_dataset
