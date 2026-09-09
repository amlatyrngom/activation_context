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

from activation.common.data_syncing import resolve_path

from .agent_config import AgentConfig, AgentRunResult

CACHE_FILENAME = "rollouts.jsonl"


def config_key(config: AgentConfig) -> str:
    task = config.dataset_task
    if task is not None:
        return f"{task.dataset_id}/{task.task_id}"
    digest = hashlib.sha256(json.dumps([config.system_prompt, config.user_prompt, config.model_name, config.lora_name,
                                        config.messages_input, config.ac_model_name], default=repr).encode()).hexdigest()
    return f"prompt/{digest[:16]}"


class RolloutCache:
    """`None` caching id disables the cache (reads return None, appends are no-ops)."""

    def __init__(self, caching_id: str | None):
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
                print(f"RolloutCache - {len(self.rows)} cached rollouts in {self.path}.", flush=True)

    def get(self, key: str, seed: int) -> dict | None:
        return self.rows.get((key, seed))

    def append(self, result: AgentRunResult) -> None:
        if self.path is None:
            return
        row = result.serialize() | {"config_key": config_key(result.agent_config)}
        with self.lock:
            self.rows[(row["config_key"], int(result.seed))] = row
            with open(self.path, "a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=repr) + "\n")
                handle.flush()
