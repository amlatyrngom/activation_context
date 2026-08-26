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
- `IB/isolation/tmux-run.sh` creates or reattaches to a named tmux session while preserving
  command arguments and the current working directory. The shared image now installs tmux; existing
  containers require a rebuild to receive it.
- A real tmux 3.3a check validated terminal detachment, session survival, argument preservation, and
  reattachment without rerunning the supplied command. tmux survives client disconnection only while
  the container remains alive; local laptop sleep pauses it, and container stop/rebuild removes it.
- Commands now run through an in-pane result holder: output and exit status remain visible until Enter.
  A status-17 failure was validated across detach and reattach, then closed cleanly on acknowledgement.
- The wrapper enables mouse scrolling and a 50,000-line history buffer for new and reattached sessions.
  A live `ac-init` reattachment verified the settings without rerunning its supplied command.
- Both Dev Container definitions now keep terminal keybindings in VS Code with
  `terminal.integrated.sendKeybindingsToShell = false`; the current scratch workspace has the same
  setting in `.vscode/settings.json`. All three JSON files parse successfully.
- The active `ac-init` CLI rollout confirms that its interactive permissions switch applied
  `approval_policy = never` with the `danger-full-access` profile. No persistent full-access default
  was added to the Dev Container setup.


## Active work — Activation Context harness

- The active review surface is `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/INTERACTIVE.tressoir.md`; source-shaped proposals are adjacent under `activation/harness/` and `activation/tests/test_harness/`, with a staged `pyproject.toml`.
- Accepted direction: no full-checkpoint override; `ModelConfig.model_id` names the shared base; ordinary dataclasses remain unslotted; model output retains its case; smoke tests remain in the default suite; weights initially reside on CPU for later park/unpark work.
- The staged follow-up is a minimal four-file delta from the latest user-edited source. It preserves
  `adapt_message_format`, existing naming and assertions, user-added no-thinking chat behavior, and
  BF16 embedding dtype; `simple_embed` returns a detached BF16 vector on CPU.
- The runtime supports multimodal generation and Qwen embeddings. Hugging Face-specific
  parsing/adaptation lives in `hf_utils.py`; layer vocabulary uses `LINEAR_ATTENTION`; Qwen
  embedding uses terminal-token pooling plus L2 normalization; duplicate names share one base load
  by `model_id` and propagate discovered metadata to every alias config.
- Accepted: embedding status and readout are auto-populated through explicit Hugging Face family rules. Unknown embedding families fail until their rule is added.
- Qwen direct embedding tokenization with special tokens appends `<|endoftext|>` (ID 151643)
  for both the 0.6B and 4B checkpoints. The 4B generic config instead declares
  `<|im_end|>` (ID 151645) as EOS, so embedding metadata derives the actual direct-input terminal
  token for recognized EOS-readout families rather than trusting that inconsistent field.
- Text-only configs use `AutoTokenizer` and multimodal configs use `AutoProcessor`, matching the
  official Gemma 3 1B text-only loading contract. All four generative and both embedding model rows
  remain active in the default smoke-test matrices.
- Validated: all staged Python compiles; pytest collects both staged smoke tests; public/cached Qwen
  config and processor checks cover Qwen3.5-0.8B, Qwen3-1.7B, and both Qwen3-Embedding sizes; direct
  terminal IDs, right-padding pooling, shared aliases, distinct dataclass lists, and the no-thinking
  template pass focused checks. Gemma metadata requires authentication in this session, and full
  weights were not downloaded or run.
- The interactive artifact now shows four independently collapsible exact file deltas and its
  checker passes. The prior `simple_embed` output/device question is resolved as CPU BF16.
