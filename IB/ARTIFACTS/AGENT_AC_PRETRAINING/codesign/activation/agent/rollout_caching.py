"""
Rollout results as JSONL under the synced folder (`resolve_path("ROLLOUTS/<caching_id>")`), one
serialized AgentRunResult per line, flushed per append. Keyed by (config key, seed): a dataset task's
key is "<dataset_id>/<task_id>", a free-form config hashes its prompts and model. A killed run keeps
every finished agent; a group of 6 after a cached group of 4 runs members 4 and 5 only. Rows written
on a Sky node land here through `sky exec --sync` and a later node starts with them.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass

from activation.common.data_syncing import resolve_path

from .agent_config import AgentConfig, AgentRunResult, _class_spec
from .agent_utils import release_ac_rows

CACHE_FILENAME = "rollouts.jsonl"


@dataclass(frozen=True)
class RedoPolicy:
    """
    Which cached rows a run rolls out again: those whose finish reason is listed and, when teacher kinds are listed,
    whose teacher kind (config metadata) is too. The new row is appended and supersedes the old one from the next
    load on (the last row per key wins); the old rows stay in the file as history. `only`: the run consists of the
    rows to redo alone, tasks with no cached row are skipped.
    """
    finish_reasons: frozenset[str] = frozenset()
    teacher_kinds: frozenset[str] = frozenset()
    only: bool = False

    def matches(self, row: dict) -> bool:
        if not self.finish_reasons or row.get("finish_reason") not in self.finish_reasons:
            return False
        if self.teacher_kinds:
            kind = ((row.get("agent_config") or {}).get("metadata") or {}).get("teacher_kind")
            return kind in self.teacher_kinds
        return True


def config_key(config: AgentConfig) -> str:
    """
    A labelled dataset task (`metadata["template"]`, set by the teacher study) is keyed "<dataset_id>/<task_id>/<template>"
    (plus "/<program>" when `metadata["program"]` names an agentic program):
    the caching id is the version of the run, so rewording a prompt, changing the model or the sampling does not
    invalidate its rows (like an autotune id). An unlabelled dataset task is keyed by a digest of its prompts, model,
    adapter and call kwargs; a free-form config hashes the same.
    """
    task = config.dataset_task
    label = config.metadata.get("template") if config.metadata else None
    program = config.metadata.get("program") if config.metadata else None
    if task is not None and label:
        return f"{task.dataset_id}/{task.task_id}/{label}" + (f"/{program}" if program else "")   # a programmed variant never collides with the plain one
    program_spec = None if config.agentic_program is None else _class_spec(*config.agentic_program)
    digest = hashlib.sha256(json.dumps([config.system_prompt, config.user_prompt, config.model_name, config.lora_name,
                                        config.messages_input, config.ac_model_name, config.call_kwargs, program_spec],
                                       default=repr, sort_keys=True).encode()).hexdigest()
    if task is not None:
        return f"{task.dataset_id}/{task.task_id}/{digest[:12]}"
    return f"prompt/{digest[:16]}"


class RolloutCache:
    """`None` caching id disables the cache (reads return None, appends are no-ops). With a `RedoPolicy`, the rows it
    matches read as absent (`get` returns None, `is_stale` True) so the run rolls them out again and appends the new row."""

    def __init__(self, caching_id: str | None, redo: RedoPolicy | None = None):
        self.redo = redo
        self.path = None if caching_id is None else resolve_path(f"ROLLOUTS/{caching_id}", create=False) / CACHE_FILENAME
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)   # resolve_path would take a dotted id for a file name
        self.lock = threading.Lock()
        self.rows: dict[tuple[str, int], dict] = {}
        if self.path is not None and self.path.exists():
            with open(self.path) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        break                                                      # a torn last line from a killed run
                    self.rows[(row["config_key"], int(row["seed"]))] = row
            if self.rows:
                stale = sum(self.is_stale(key, seed) for key, seed in self.rows)
                print(f"RolloutCache - {len(self.rows)} cached rollouts in {self.path}" + (f"; {stale} to redo ({redo})." if redo else "."), flush=True)

    def get(self, key: str, seed: int) -> dict | None:
        row = self.rows.get((key, seed))
        return None if row is None or (self.redo is not None and self.redo.matches(row)) else row

    def is_stale(self, key: str, seed: int) -> bool:
        """A cached row exists but the redo policy sends it back to the rollout."""
        row = self.rows.get((key, seed))
        return row is not None and self.redo is not None and self.redo.matches(row)

    def append(self, result: AgentRunResult) -> None:
        if self.path is None:
            return
        row = result.serialize(base_dir=self.path.parent) | {"config_key": config_key(result.agent_config)}
        with self.lock:
            with open(self.path, "a") as handle:
                offset = handle.tell()
                try:
                    handle.write(json.dumps(row, ensure_ascii=False, default=repr) + "\n")
                    handle.flush()
                except Exception:
                    handle.seek(offset)
                    handle.truncate()
                    raise
            self.rows[(row["config_key"], int(result.seed))] = row
            release_ac_rows(result, row, base_dir=self.path.parent)
