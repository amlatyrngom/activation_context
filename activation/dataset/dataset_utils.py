"""
Utilities to load and split datasets.
"""

import typing as t
from .dataset import DataSplit, LoadedDataset, DatasetStats

def make_dataset_id(base_name: str, **kwargs):
    parts: list[str] = [
        f"{k}_{v}" for k, v in kwargs.items()
    ]
    parts = "__".join(parts)
    if not parts:
        return base_name
    return f"{base_name}___{parts}"


def take_to_budget(
    rows: t.Iterable,
    budget: int | None,
    cost_fn: t.Callable[[t.Any], int]|None = None,
    filter_fn: t.Callable[[t.Any], int]|None=None
) -> list:
    """
    Take items up to a given budget.
    Uses streaming in case the dataset is very large.
    """
    if cost_fn is None:
        # Assume a count-based cost.
        cost_fn = lambda _x: 1
    if filter_fn is None:
        # Assume all true
        filter_fn = lambda _x: True
    if budget is None:
        # Effectively infinite.
        budget = 2**50
    selected = []
    total = 0
    for row in rows:
        if not filter_fn(row):
            continue
        row_cost = cost_fn(row)
        selected.append(row)
        total += row_cost
        if total >= budget:
            break
    return selected



def normalize_split_name(split: str) -> DataSplit:
    """
    Map ordinary upstream split names to the typed experiment role.
    """
    normalized = split.strip().lower()
    if normalized in {"dev", "train"}:
        return DataSplit.TRAIN
    if normalized in {"validation", "val"}:
        return DataSplit.VAL
    if normalized == "test":
        return DataSplit.TEST
    raise ValueError(f"unknown labeled-example source split {split!r}")


def initialize_dataset_stats(loaded_dataset: LoadedDataset, load_time: float) -> DatasetStats:
    """
    Compute the load-time statistics; index/query statistics fill in later.
    """
    total_chars = sum(len(document.text) for document in loaded_dataset.documents.values())
    num_documents = len(loaded_dataset.documents)
    return DatasetStats(
        initial_load_latency=load_time,
        total_document_chars=total_chars,
        total_num_documents=num_documents,
        avg_document_chars=total_chars // num_documents if num_documents else 0,
    )


def safe_truncate_embedding_chunk(chunk: str, limit: int|None):
    if limit is None or len(chunk) <= limit:
        return chunk
    else:
        half = limit // 2
        return chunk[:half] + chunk[-(limit - half):]