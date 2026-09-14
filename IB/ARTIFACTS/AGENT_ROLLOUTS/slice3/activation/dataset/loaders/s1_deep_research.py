"""
Loader for S1-DeepResearch-15k (`ScienceOne-AI/S1-DeepResearch-15k`): web-research trajectories in
Qwen's Hermes style (tool definitions in a <tools> block of the system prompt, <think> blocks and
<tool_call> JSON blocks in the assistant text, results in `tool` messages, the final answer in
<answer> tags). The `search` and `visit` tools have no counterpart of ours and keep their names.
"""
from __future__ import annotations

import re
import time
import typing as t

from datasets import load_dataset

from ..dataset import EXTERNAL, DataModality, DatasetDocument, LoadedDataset
from ..dataset_utils import initialize_dataset_stats, make_dataset_id, take_to_budget
from .trajectory_utils import (
    parity_split,
    parse_inline_calls,
    reformat_trajectory,
    split_think,
    tools_from_system_prompt,
    trajectory_chars,
)

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime

HF_DATASET = "ScienceOne-AI/S1-DeepResearch-15k"
SYSTEM_PROMPT = ("You are a deep research assistant. Investigate the question with the search and visit tools "
                 "and enclose the final answer within <answer></answer> tags.")
ANSWER_BLOCK = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)


def normalize_s1_row(row: dict) -> tuple[list[dict], list[dict]]:
    """One source row -> (normalized messages, tool definitions), see trajectory_utils."""
    normalized, tools = [], []
    for message in row["messages"]:
        role, content = message["role"], message.get("content") or ""
        if role == "system":
            tools = tools_from_system_prompt(content)
            normalized.append({"role": role, "content": content})
        elif role == "assistant":
            reasoning, rest = split_think(content)
            text, calls = parse_inline_calls(rest)
            normalized.append({"role": role, "content": text.strip(), "reasoning": reasoning, "tool_calls": calls})
        else:
            normalized.append({"role": role, "content": content})
    return normalized, tools


class S1DeepResearchDataset:
    """Loader for S1-DeepResearch-15k as trajectory documents."""

    @classmethod
    def load(
        cls,
        harness: "HarnessRuntime",
        max_examples: int | None,
        language: str | None = "en",
        min_chars: int = 0,
        hf_dataset: str = HF_DATASET,
    ) -> LoadedDataset:
        """Streams the rows and keeps the first max_examples of the language (None: any) at least `min_chars` long."""
        start_time = time.time()
        dataset_id = make_dataset_id("s1_deep_research", lang=language or "all", n=max_examples, min=min_chars)
        print(f"{dataset_id} - Loading")
        documents: dict[str, DatasetDocument] = {}

        def convert(row: dict) -> bool:
            meta = row.get("meta") or {}
            if language is not None and meta.get("language") != language:
                return False
            normalized, tools = normalize_s1_row(row)
            messages, kwargs = reformat_trajectory(normalized, tools)
            if trajectory_chars(messages) < min_chars or not any(message["role"] == "assistant" for message in messages):
                return False
            doc_id = str(meta.get("id") or len(documents))
            kwargs.update({"system_prompt": SYSTEM_PROMPT, "question": meta.get("question"), "answer": meta.get("answer"),
                           "task_type": meta.get("task_type") or meta.get("type")})
            documents[doc_id] = DatasetDocument(doc_id=doc_id, dataset_id=dataset_id, trajectory=messages, trajectory_kwargs=kwargs,
                                                modality=DataModality.TRAJECTORY, origin=EXTERNAL, split=parity_split(doc_id))
            return True

        take_to_budget(load_dataset(hf_dataset, split="train", streaming=True), budget=max_examples, filter_fn=convert)
        loaded_dataset = LoadedDataset(dataset_id=dataset_id, documents=documents)
        loaded_dataset.stats = initialize_dataset_stats(loaded_dataset, time.time() - start_time)
        harness.dataset_manager.register_dataset(loaded_dataset)
        return loaded_dataset
