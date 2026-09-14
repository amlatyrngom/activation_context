"""
DeepMath-103K (`zwhe99/DeepMath-103K`, MIT) as scorable math tasks: problems at `min_difficulty` and above (the
dataset's own 1 to 10 scale), scored by math-verify equivalence against `final_answer` (numbers, LaTeX expressions,
sets, intervals; Yes/No by normalized string match). The parquet shards are read once with only the columns the
tasks need (no R1 solutions) and the filtered rows are cached as JSONL under the synced folder; later loads read the cache.
"""
from __future__ import annotations

import json
import random
import time
import typing as t

from ..dataset import ANSWER_RULES, DatasetTask, DatasetTaskKind, DatasetTaskMetricsKind, LoadedDataset, bare_prompt
from ..dataset_utils import initialize_dataset_stats, make_dataset_id
from activation.common.data_syncing import resolve_path

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime

HF_DATASET = "zwhe99/DeepMath-103K"
CACHE_FOLDER = "DEEPMATH"
COLUMNS = ["question", "final_answer", "difficulty", "topic"]


def _cached_rows(min_difficulty: float) -> list[dict]:
    """Rows at or above the difficulty, from `DEEPMATH/difficulty_<n>.jsonl`, built from the Hub parquet shards on first use."""
    path = resolve_path(CACHE_FOLDER, create=True) / f"difficulty_{min_difficulty:g}.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    fs = HfFileSystem()
    rows: list[dict] = []
    for shard in sorted(fs.glob(f"datasets/{HF_DATASET}/**/*.parquet")):
        with fs.open(shard, "rb") as handle:
            table = pq.read_table(handle, columns=COLUMNS)
        for row in table.to_pylist():
            if float(row["difficulty"]) >= min_difficulty:
                rows.append({"question": row["question"], "final_answer": str(row["final_answer"]),
                             "difficulty": float(row["difficulty"]), "topic": row["topic"]})
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    tmp.replace(path)
    return rows


class DeepMathDataset:
    @classmethod
    def load(cls, harness: "HarnessRuntime", max_examples: int | None, seed: int = 0, min_difficulty: float = 6.0) -> LoadedDataset:
        start = time.time()
        dataset_id = make_dataset_id("deepmath", min_difficulty=f"{min_difficulty:g}", n=max_examples, seed=seed)
        rows = _cached_rows(min_difficulty)
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        tasks: dict[str, DatasetTask] = {}
        for index in order:
            row = rows[index]
            task_id = f"d{row['difficulty']:g}-{index}"
            tasks[task_id] = DatasetTask(
                task_id=task_id, dataset_id=dataset_id,
                task_datum={"difficulty": row["difficulty"], "topic": row["topic"]},
                reference_metrics_kind=DatasetTaskMetricsKind.MATH_VERIFY, gold_answer=row["final_answer"],
                agent_prompt=bare_prompt("Solve the following math problem.\n\n" + row["question"].strip(), ANSWER_RULES["math"]),
                task_kind=DatasetTaskKind.MATH,
            )
            if max_examples is not None and len(tasks) >= max_examples:
                break
        loaded = LoadedDataset(dataset_id=dataset_id, scorable_tasks=tasks)
        loaded.stats = initialize_dataset_stats(loaded, load_time=time.time() - start)
        return loaded
