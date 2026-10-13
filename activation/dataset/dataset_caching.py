"""
Very simple caching of the oracle study steps (question generation and labeling), so an hour-long
pass is not repeated when a run is restarted or a later run asks for the same or fewer samples.

One JSONL file per key (caching id, dataset, model, kind, seed; plus the label model for labels),
rows in generation order, under `STUDY/` of the synced folder. A request for n samples is served from
the first n rows when the file holds at least n: the study sampler is a seeded shuffle, so the first n
chunks of a larger request are the same chunks (each row records its chunk id and the reader checks
it). A request beyond the cached rows reuses them and generates the tail, which is appended. Labels
are keyed by example id. A stable caching id from the caller is trusted; nothing else is checked.
"""
import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

STUDY_KINDS = ("qa", "traj_qa", "labels")


@dataclass(frozen=True)
class StudyCacheKey:
    caching_id: str
    dataset_id: str
    model_id: str
    kind: str                       # one of STUDY_KINDS
    seed: int
    label_model_id: str | None = None

    def file_name(self) -> str:
        parts = [self.caching_id, self.dataset_id, self.kind, self.model_id, f"seed{self.seed}"]
        if self.label_model_id:
            parts.append(self.label_model_id)
        stem = "__".join(re.sub(r"[^0-9A-Za-z_.-]+", "_", part) for part in parts)
        if len(stem) + len(".jsonl") > 255:
            # Local model snapshots can make a cache key exceed filesystem limits.
            # Keep valid existing names; hash the complete original key when shortening.
            suffix = hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()
            stem = stem[:180] + "__" + suffix
        return stem + ".jsonl"


class StudyCache:
    """JSONL rows under a root folder; `None` root disables every method (reads return [], appends are no-ops)."""

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


def freeze_text_collection(root: str | Path, destination: str | Path) -> dict:
    """Freeze the available committed prefix of a portable collection, without copying payloads.

    Growing manifests and in-flight transfers are safe: only listed, present,
    checksum-verified files enter the snapshot. A listed corrupt file is an error.
    Relative payload paths let the same snapshot move with its collection.
    """
    import os
    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Frozen snapshot already exists: {destination}")
    raw = (root / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    entries = []
    if 'releases' in manifest:
        for release in manifest['releases']:
            relative = Path(release['manifest'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe collection manifest path')
            path = root / relative
            if not path.exists():
                continue
            child = json.loads(path.read_bytes())
            entries.extend({**row, 'file': str(relative.parent / row['file'])} for row in child['shards'])
    elif 'files' in manifest:
        entries = manifest['files']
    elif 'file' in manifest:
        entries = [manifest]
    else:
        raise ValueError('Portable collection has no committed file list')
    files, pending, seen = [], [], set()
    for entry in entries:
        relative = Path(entry['file'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:
            raise ValueError('Unsafe or repeated collection payload path')
        seen.add(str(relative))
        path = root / relative
        if not path.is_file():
            pending.append(str(relative)); continue
        with path.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        if checksum != entry['sha256']:
            raise ValueError(f'Committed collection payload changed: {relative}')
        files.append({'file': str(relative), 'sha256': checksum, 'bytes': path.stat().st_size,
                      'rows': entry.get('rows', entry.get('questions'))})
    controls = {}
    # Existing evaluation questions remain reserved even when new runs freeze a fresh held panel.
    if manifest.get('schema') == 'self_contained_rag_qa_v1' and not (root / 'reserved_questions.json').is_file():
        raise FileNotFoundError('RAG collection has not transferred reserved_questions.json yet; retry after it arrives')
    if (root / 'reserved_questions.json').exists():
        control = (root / 'reserved_questions.json').read_bytes()
        controls['reserved_questions'] = {'sha256': hashlib.sha256(control).hexdigest(), 'data': json.loads(control)}
    result = {'schema': 'frozen-portable-text-v1', 'collection_schema': manifest['schema'],
              'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'files': files,
              'pending_transfer_files': pending, 'controls': controls}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f'.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    # Publish without overwriting another launch's already committed selection.
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink()
    return result


def frozen_text_rows(root: str | Path, snapshot: dict):
    """Read an immutable snapshot; new producer commits never change this iterator."""
    root = Path(root)
    for entry in snapshot['files']:
        path = root / entry['file']
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != entry['sha256']:
                raise ValueError(f'Frozen payload changed: {path}')
            stream.seek(0)
            if path.suffix == '.jsonl':
                for ordinal, line in enumerate(stream):
                    yield json.loads(line), {'file': entry['file'], 'ordinal': ordinal}
            else:
                payload = json.load(stream)
                for split in ('train', 'held'):
                    for ordinal, row in enumerate(payload[split]):
                        yield dict(row, split=split), {'file': entry['file'], 'split': split, 'ordinal': ordinal}
