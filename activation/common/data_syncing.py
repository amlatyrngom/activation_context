"""
Coarse-grained data syncing between this machine and a Sky node.

One folder, `IB/TMP/SYNC/`, is the synced working area: `uv run sky exec --sync` pushes it to the
node before the job (as `~/activation_artifacts/SYNC/`), pulls it back every few seconds while the
job runs and once more when the job ends, so caches and reports written on the node land here as
they are written and a later job on a fresh node starts with everything a previous one wrote.
Code never cares where it runs: `resolve_path("BRIGHT_ECON_ABLATION/CACHE")` is the local folder on
this machine and the remote folder on a node (the Sky wrapper exports `ACTIVATION_SYNC_ROOT`).
A fully local run resolves to the local folder and has no watcher.
"""
import os
from pathlib import Path

SYNC_ROOT_ENV = "ACTIVATION_SYNC_ROOT"
"""Set by the Sky wrapper on the node; unset here."""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SYNC_ROOT = PROJECT_ROOT / "IB" / "TMP" / "SYNC"
REMOTE_SYNC_ROOT = "/root/activation_artifacts/SYNC"


def sync_root() -> Path:
    """The synced folder on this machine: the local root here, the artifacts folder on a node."""
    return Path(os.environ.get(SYNC_ROOT_ENV) or LOCAL_SYNC_ROOT)


def resolve_path(relative: str, create: bool = True) -> Path:
    """A folder (or file path) under the synced area, created on request; `relative` may not escape it."""
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"sync paths are relative to the synced folder: {relative!r}")
    resolved = sync_root() / relative_path
    if create:
        (resolved if not resolved.suffix else resolved.parent).mkdir(parents=True, exist_ok=True)
    return resolved
