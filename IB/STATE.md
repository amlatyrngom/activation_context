# Project State

## Agent container

- A dedicated Agent Container is configured at `IB/isolation/agent.devcontainer.json`, exposed to
  VS Code through `.devcontainer/agent/devcontainer.json`.
- It keeps host source read-only at `/source`, uses disposable copied source at `/workspace`, and
  mounts only host `IB/` read-write at `/workspace/IB`.
- Its shared build image bakes Node 24/npm, uv, CPython 3.14, and the selected Claude, Codex, and
  Pi CLIs. Startup only syncs the project environment from committed `uv.lock`.
- A fresh Agent Workspace build was validated end to end: `/source` is read-only, `/workspace` is scratch-writable, and `IB/` write-through works.
- The baked toolchain reports Node 24.18.0/npm 11.16.0, uv 0.12.5, Python 3.14.7, and all selected agent CLIs.
- `uv run pytest` passes from `/workspace` (1 test). The scratch copy excludes the host virtual environment and test caches, then creates a fresh `.venv` from `uv.lock`.
