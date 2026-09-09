"""
Loader for NVIDIA's Open-SWE-Traces (`nvidia/Open-SWE-Traces`): coding-agent trajectories over SWE
tasks, reformatted into our dialect (trajectory_utils). Rows carry structured `tool_calls` with JSON
string arguments, tool results as JSON `{"returncode", "output"}`, and tool definitions as JSON
strings; the single tool `bash(command)` maps onto our `shell(script)`.
"""
from __future__ import annotations

import time
import typing as t

from datasets import load_dataset

from ..dataset import EXTERNAL, DataModality, DatasetDocument, LoadedDataset
from ..dataset_utils import initialize_dataset_stats, make_dataset_id, take_to_budget
from .trajectory_utils import (
    flatten_tool_output,
    parse_tool_definitions,
    parity_split,
    reformat_trajectory,
    structured_calls,
    trajectory_chars,
)

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime

HF_DATASET = "nvidia/Open-SWE-Traces"
SYSTEM_PROMPT = "You are a software engineering agent. Solve the task in the repository using the shell tool."


def normalize_open_swe_row(row: dict) -> tuple[list[dict], list[dict]]:
    """One source row -> (normalized messages, tool definitions), see trajectory_utils."""
    normalized = []
    for message in row["messages"]:
        role = message["role"]
        if role == "assistant":
            normalized.append({"role": role, "content": message.get("content") or "",
                               "reasoning": message.get("reasoning_content") or None,
                               "tool_calls": structured_calls(message.get("tool_calls"))})
        elif role == "tool":
            normalized.append({"role": role, "content": flatten_tool_output(message.get("content"))})
        else:
            normalized.append({"role": role, "content": message.get("content") or ""})
    return normalized, parse_tool_definitions(row.get("tools"))


class OpenSweTracesDataset:
    """Loader for one config/split of Open-SWE-Traces as trajectory documents."""

    @classmethod
    def load(
        cls,
        harness: "HarnessRuntime",
        max_examples: int | None,
        config: str = "v1.2",
        split: str = "minisweagent",
        resolved_only: bool = False,
        min_chars: int = 0,
        hf_dataset: str = HF_DATASET,
    ) -> LoadedDataset:
        """
        Streams the split and keeps the first max_examples trajectories (all when None) that pass the
        filters: `resolved_only` keeps rows the harness marked resolved (1), `min_chars` the trajectories
        at least that long in our rendering (content plus call arguments).
        """
        start_time = time.time()
        dataset_id = make_dataset_id("open_swe", config=config, split=split, n=max_examples, min=min_chars)
        print(f"{dataset_id} - Loading")
        documents: dict[str, DatasetDocument] = {}

        def convert(row: dict) -> bool:
            if resolved_only and int(row.get("resolved") or 0) != 1:
                return False
            normalized, tools = normalize_open_swe_row(row)
            messages, kwargs = reformat_trajectory(normalized, tools)
            if trajectory_chars(messages) < min_chars or not any(message["role"] == "assistant" for message in messages):
                return False
            doc_id = str(row.get("trajectory_id") or row["instance_id"])
            kwargs.update({"system_prompt": SYSTEM_PROMPT, "instance_id": row.get("instance_id"), "repo": row.get("repo"),
                           "resolved": row.get("resolved")})
            documents[doc_id] = DatasetDocument(doc_id=doc_id, dataset_id=dataset_id, trajectory=messages, trajectory_kwargs=kwargs,
                                                modality=DataModality.TRAJECTORY, origin=EXTERNAL, split=parity_split(doc_id))
            return True

        take_to_budget(load_dataset(hf_dataset, config, split=split, streaming=True), budget=max_examples, filter_fn=convert)
        loaded_dataset = LoadedDataset(dataset_id=dataset_id, documents=documents)
        loaded_dataset.stats = initialize_dataset_stats(loaded_dataset, time.time() - start_time)
        harness.dataset_manager.register_dataset(loaded_dataset)
        return loaded_dataset
