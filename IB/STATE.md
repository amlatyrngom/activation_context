# Project State

## Agent container

- The dedicated Agent Container is configured through `IB/isolation/agent.devcontainer.json` and `.devcontainer/agent/devcontainer.json`.
- `/source` is read-only user source, `/workspace` is disposable scratch, and `/workspace/IB` is the shared write-through boundary.
- The image bakes Node 24/npm, uv, CPython 3.14, tmux, and the selected agent CLIs; startup syncs the project environment from committed `uv.lock`.
- Fresh-build isolation, tool versions, tmux detach/reattach and result retention, mouse/history settings, and VS Code terminal keybindings were validated.
- Full-access agent CLI behavior was confirmed for the active initialization workflow without adding a persistent global full-access default.

## Active work — Activation Context harness and rollout engine

### Review surfaces

- Current source-shaped handoff: `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/activation/harness/{loaded_model.py,runtime.py}`.
- Current interactive review: `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/INTERACTIVE.tressoir.md`.
- Deferred LoRA explainer: `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/LORA_EXPLAINER/LORA_EXPLAINER.tressoir.html`.

### Accepted direction

- The latest user source remains the baseline: the engine/Hugging Face placement logic is merged into one `LoadedModel`, with harness-level LoRA limits and local sampling defaults preserved.
- Concurrency is deliberately undecided; no locking behavior is proposed.
- Harness startup loads metadata/tokenizer/processor while Transformers weights remain at the disk sentinel.
- Each `LoadedModel` owns the optional full Transformers model, independent primary embedding-module copy, and optional vLLM engine with separate placement state.
- `HarnessRuntime` remains the name-based registry and exposes thin `simple_chat` / `simple_embed` delegates; generation and embedding behavior stay in `LoadedModel`.
- Freeing vLLM invokes the pinned internal engine-core shutdown before reference release, GC, and CUDA allocator cleanup. There is no intermediate sleep state; process isolation remains the strongest hard-cleanup boundary.
- LoRA routing remains unsupported.

### Current narrow handoff and validation

- `loaded_model.py` supplies `model_id`/`dtype` when reloading, makes embedding-copy freeing stop instead of falling through into recreation, checks the embedding device for CUDA cleanup, restores a temporarily loaded full model in `finally`, records the engine free state, and accepts the test's unused `lora_name=None` keyword.
- `runtime.py` adds only the two thin name-based delegates expected by the unchanged basic harness tests.
- No tests, concurrency behavior, LoRA routing, sampling settings, or source formatting were changed.
- Local preflight: both staged files parse; all three test functions collect; a fake-model lifecycle fixture passes.
- L40S release run: `activation/tests/test_basic_engine.py` plus `activation/tests/test_basic_harness.py` completed with `3 passed in 350.92s`.
- The disposable `ac-loaded-model-tests` cluster was torn down; final SkyPilot state reported no clusters, jobs, or services.
- Next step: apply the exact two-file handoff shown in `INTERACTIVE.tressoir.md`.

## Completed work — SkyPilot GPU workflow

### Current handoff

- Source-shaped implementation: `IB/ARTIFACTS/SKYPILOT/{.skyignore,README.md,activation/cloud,activation/tests}`.
- Detailed current review: `IB/ARTIFACTS/SKYPILOT/INTERACTIVE.tressoir.md`.
- Completed plan pair: `SYNC_INSTALL_PLAN.md` and `SYNC_INSTALL_PLAN.tressoir.md`.
- The handoff is clean: no `orig.py`, patch rejects, bytecode, or superseded source snapshots.

### Accepted architecture

- `~/sky_workdir` is an exact, disposable, local-owned mirror. `exec` resumes stopped clusters with `--retry-until-up`, uploads with deletion, then submits only after success.
- `upload` means local source to remote. `download` means remote artifacts to local.
- Hugging Face and FlashInfer caches live beneath `/root/.cache`; checkpoints and results live beneath `~/activation_artifacts`, all outside the mirror.
- Downloads are confined to the artifact root, never delete local files, preserve existing files by default, and require in-project destinations to be under `IB/TMP`.
- Runtime dependencies are baked from `uv.lock` into `/opt/activation/.venv` on a digest-pinned CUDA 13 development image. Changing code does not require an image rebuild.
- Remote exec adds `PYTHONPATH=/root/sky_workdir` and shell-serializes argv with `shlex.join`.
- An existing project-root `.env` is passed to trusted remote jobs through `sky exec --secret-file`; it remains excluded from the exact source mirror, image, caches, and artifacts.
- Remote exec also sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`, preventing CUDA re-initialization after a CUDA-aware parent has started the vLLM worker.
- Private ECR is the image transport; `ACTIVATION_SKY_IMAGE` overrides the project default for another registry/account.
- The existing `pyproject.toml` console entry already exposes `uv run sky`; no project configuration change was needed.

### Validation

- Earlier focused wrapper suite: `13 passed in 0.05s`, including `sky start --retry-until-up`; later live runs exercised corrected resume/capacity fallback behavior.
- The latest combined wrapper/helper regression run reports `19 passed in 8.18s`.
- Published ECR image: `814218043106.dkr.ecr.us-east-1.amazonaws.com/activation-context/sky:uv-de16520b51ef997e`.
- Image digest: `sha256:2d5a7e6be7724eed9404b5a475cffbcbf5177ad9732c549e8b4f91186e797127`.
- Live AWS runtime: `g6e.xlarge`, one L40S, driver 580.159.04, CUDA 13.0.
- Live basic engine command passed: `1 passed in 307.94s`.
- Live secret probe: the job environment received the variable while `/root/sky_workdir/.env` remained absent; no value was printed or retained.
- Live Gemma 3 project-engine smoke passed with the wrapper's spawn-safe worker environment.
- Exact upload deleted a remote-only workdir sentinel.
- An external artifact sentinel survived upload and downloaded into `IB/TMP` with exact content.
- Temporary GPU and image-builder clusters were torn down. Final SkyPilot state contains no clusters/jobs/services; final AWS `us-east-2` active-instance query returned `[]`.
- Private ECR repository/image are intentionally retained for reuse.
- Pause/resume persistence was not separately exercised; durable checkpoints must still be exported before teardown.

### Next meaningful step

Copy the updated staged `activation/cloud/sky.py` into the user source. The validated ECR image can be reused immediately while `uv.lock` is unchanged.

## Completed research — Message schemas and vLLM input embeddings

### Deliverables

- Final report: `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/INPUT_EMBEDS_REPORT.tressoir.html`.
- Completed plan pair: `INPUT_EMBEDS_RESEARCH_PLAN.md` and `INPUT_EMBEDS_RESEARCH_PLAN.tressoir.md`.
- Report-local evidence: `input-embeds-live-probe.json` and `input-embeds-message-matrix.json`.

### Validated findings — cumulative

- Pinned vLLM 0.28 detects Qwen 3.5 and Gemma 4 templates as `openai`, Qwen 3 and Nemotron 3 as `string`, and normalizes direct-string and typed-text inputs to equal conversations before template rendering.
- Qwen 3.5 live generation produced identical prompt IDs and output IDs for direct-string and typed-text chat without project adaptation.
- Native hybrid chat works through a `prompt_embeds` content part when `enable_prompt_embeds=True`; the exact seven-row Qwen 3.5 user span and the 29-row full prompt each matched ordinary greedy generation.
- The reserved sentinel is positional alignment, not semantic equivalent text. Chat controls remain template-owned; user-span rows come from content tokenization without added special tokens.
- A pooled single row passed shape validation but changed output, confirming that pooled semantic embeddings are not causal-LM token-row substitutes.
- Authenticated Gemma 3 1B string/typed messages, exact full-prompt rows, exact hybrid user rows, and the project engine all passed live. Its input module applies model-defined scaling; the copied module reproduced vLLM rows exactly.
- A live Qwen 3 Embedding 0.6B → Qwen 3.5 0.8B transfer held seven rows and `d_model=1024` constant. vLLM accepted direct side activations but generated different output; exact target rows reproduced the baseline. The mismatch is learned coordinate space.
- Gemma 4 E2B consumes a token-conditioned per-layer embedding path in addition to main `d_model` rows. Full `EmbedsPrompt` companion target token IDs can preserve it for text-derived rows; genuinely non-text side rows need an explicit trained PLE/sentinel policy or a richer carrier.
- For audited 8B–32B fixtures, primary BF16 tables occupy 1.16–2.62 GiB and 2.37–10.54% of checkpoint parameters. Pinning is plausible on large GPUs but remains a deliberate residency tradeoff.
- The follow-up `ac-embed-next` cluster was torn down after artifact retrieval; final SkyPilot status reported no clusters, jobs, or services.

### Proposed future engine work

- Let vLLM own ordinary message normalization instead of using multimodality as the capability proxy.
- Widen the future engine message contract for trusted `prompt_embeds` parts and enable the feature only on engines that need it.
- Treat a side-to-target projector and per-family validation as part of the model contract; Gemma 4 E2B additionally needs an explicit PLE policy.

### Completed follow-up — secrets, Gemma 3, and side-model embeddings

- Completed plan pair: `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/INPUT_EMBEDS_NEXT_ROUND_PLAN.md` and `INPUT_EMBEDS_NEXT_ROUND_PLAN.tressoir.md`.
- Updated cumulative report: `INPUT_EMBEDS_REPORT.tressoir.html`; new report-local evidence covers Gemma 3, side-space transfer, Gemma 4 E2B, the project-engine smoke, and exact embedding sizes.
- Apply callouts: copy the staged Sky `activation/cloud/sky.py`; separately review/adapt the staged `hf_utils.py` helper and `LoadedBaseModel` lifecycle method.
