"""
Very simple caching of the oracle study steps (generation and labeling), so an hour-long pass is
not repeated when a run is restarted or a later run asks for the same or fewer samples.

One JSONL file per key (caching id, dataset, model, kind, seed; plus the label model for labels),
rows in generation order. A request for n samples is served from the first n rows when the file
holds at least n: the study sampler is a seeded shuffle, so the first n chunks of a larger request
are the same chunks. A request beyond the cached rows reuses them and generates the tail, which is
appended. Labels are keyed by example id. Nothing is checked beyond the key and, for generation,
the chunk id of each row; a stable caching id from the caller is trusted. Cheap steps (programmatic
examples, selection) are not cached.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StudyCacheKey:
    caching_id: str
    dataset_id: str
    model_id: str
    kind: str                       # "qa", "description", "labels"
    seed: int
    label_model_id: str | None = None

    def file_name(self) -> str:
        parts = [self.caching_id, self.dataset_id, self.kind, self.model_id, f"seed{self.seed}"]
        if self.label_model_id:
            parts.append(self.label_model_id)
        return "__".join(re.sub(r"[^0-9A-Za-z_.-]+", "_", part) for part in parts) + ".jsonl"


class StudyCache:
    """JSONL rows under a root folder; `None` root disables every method (reads return None, appends are no-ops)."""

    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: StudyCacheKey) -> Path | None:
        return None if self.root is None else self.root / key.file_name()

    def read(self, key: StudyCacheKey) -> list[dict]:
        path = self.path(key)
        if path is None or not path.exists():
            return []
        rows = []
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        break                                                      # a torn last line from a killed run
        return rows

    def append(self, key: StudyCacheKey, rows: list[dict]) -> None:
        """Flushed per call, so a killed run keeps the rows it produced."""
        path = self.path(key)
        if path is None or not rows:
            return
        with open(path, "a") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    def read_by_id(self, key: StudyCacheKey, id_field: str) -> dict[str, dict]:
        """Rows keyed by one field (labels by example id); a later row for the same id wins."""
        return {row[id_field]: row for row in self.read(key) if id_field in row}
