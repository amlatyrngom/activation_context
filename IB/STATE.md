# Project State

## Agent container

- The dedicated Agent Container is configured through `IB/isolation/agent.devcontainer.json` and `.devcontainer/agent/devcontainer.json`.
- `/source` is read-only user source, `/workspace` is disposable scratch, and `/workspace/IB` is the shared write-through boundary.
- The image bakes Node 24/npm, uv, CPython 3.14, tmux, and the selected agent CLIs; startup syncs the project environment from committed `uv.lock`.
- Fresh-build isolation, tool versions, tmux detach/reattach and result retention, mouse/history settings, and VS Code terminal keybindings were validated.
- Full-access agent CLI behavior was confirmed for the active initialization workflow without adding a persistent global full-access default.

## Active work — Document index and public dataset loaders

### Review surfaces

- Current interactive review: `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/DOCUMENT_INDEX_ROUND_4_INTERACTIVE.tressoir.md`.
- Staged source-shaped handoff: `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/activation/{dataset,tests,harness}` (dataset module, five loaders, public dataset test, runtime config).
- Earlier rounds (applied): `DOCUMENT_INDEX_INTERACTIVE`, `DOCUMENT_INDEX_ROUND_2_INTERACTIVE`, `DOCUMENT_INDEX_ROUND_3_INTERACTIVE` `.tressoir.md`.

### Status

- Rounds 1–4 are fully applied in user source, including the user's progress-print UX pass and the three review fixes from the final chat round (print typo, latency-stat consistency, size formatting). The staged handoff mirrors the applied source.
- Durable policies were promoted to `IB/CANON/ROOT_CANON.md` ("Dataset loading and indexing decisions"); the SciFact-distractor question remains open in the round-4 doc.
- The user is prototyping the next major steps and will return with direction. Likely next rounds: retrieval-quality metrics (recall@k, where `excluded_doc_ids` does real work) and GPU runs.

### Round 4 — unskewed loaders, truncator, exclusions

- Skew fixes agreed in chat: examples are first-class (`take_to_budget` with the user's `filter_fn`; `cost_fn` documented as the budget tracker for fan-out cases), `max_doc_chars` removed everywhere, and the user's `safe_truncate_embedding_chunk` head+tail truncator wired through `doc_embedding_input_limit_chars` on both build and query paths (512 on CPU test, None on GPU).
- `LabeledRetrievalQAExample.excluded_doc_ids` (user renamed from chunk-level) is populated by BRIGHT from native `excluded_ids` and respected in `query_many_frozen` via per-query sets with a flat +10 over-fetch.
- BRIGHT: `max_examples: int|None` (None = all labeled examples), corpus = gold passages only; distractor sampling removed at user request. SciFact still carries seeded distractors — open decision in the round-4 doc.
- Review of the user's partial application found and fixed: truncate calls missing the limit argument (build crash), exclusion filtering testing the wrong variable (never filtered), `_build_excluded_sets` None/empty handling and returning raw lists, and the unused `chunk_ids` cache.
- Validation: basic test passed post-fixes; BRIGHT loads at n=20 and n=None; full CPU public run of the final code reported in chat.

### Round 3 — implemented and validated

- Reviewed and corrected the applied round-2 source: atomic chunk slice width, restored empty-corpus/empty-query guards, MS-MARCO dedup crash, duplicate `labeled_retrieval_examples` field, `TARGET_DEVICE` GPU check, two import mistakes, and `DatasetStats.to_json`.
- Loaders for BRIGHT (per-domain), NQ (sentence-transformers pairs), MS-MARCO (user's, fixed), SciFact (BeIR + qrels), SciQ — all follow the MS-MARCO shape; documents load `atomic=True`; examples are dropped whenever a gold document is absent so labels always resolve.
- `max_doc_chars` load filter in every loader (test uses 8192) plus CPU example budget 20 / GPU 1000; chosen after a 100-example uncapped run showed long-sequence batches dominating CPU time.
- `query_many_frozen` embeds in length-sorted batches and restores order; `_build_index` restores the embedding model's starting placement; `DatasetStats` fills at load, build, and query time.
- Validation: `test_basic_dataset_loading` passed (37s); `test_public_dataset_loading` passed on CPU (29m24s) with sensible printed retrieval (golds rank first nearly everywhere) and populated stats.
- `IB/.gitignore` now ignores per-artifact `tressoir-linear.css/js` copies (skill template copies stay tracked); superseded harness handoff snapshots were removed from the artifact folder.

### Open decisions (widgets in the round-3 doc)

- CPU budget: keep 8192 chars/20 examples (~30 min) or lower the char cap to ~2000–3000 (~5–8 min).
- Embedding-model residency across `build_document_indexes` (currently restore-per-index, which reloads weights each build).

### Next meaningful step

User reviews `DOCUMENT_INDEX_ROUND_3_INTERACTIVE.tressoir.md`, resolves the two decisions, and applies the staged `activation/dataset` + `activation/tests` handoff.

## Completed work — Activation Context harness engine

- The two-file harness handoff (`loaded_model.py`, `runtime.py`) was applied and has since evolved in user source (e.g. `simple_vector_embed_many` with padding); the staged copies were removed as superseded.
- Historical review: `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/INTERACTIVE.tressoir.md`.
- Deferred LoRA explainer: `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/LORA_EXPLAINER/LORA_EXPLAINER.tressoir.html`.

### Accepted architecture (still current)

- One `LoadedModel` owns the optional full Transformers model, independent primary embedding-module copy, and optional vLLM engine with separate placement state; harness startup loads metadata/tokenizer/processor while weights stay at the disk sentinel.
- `HarnessRuntime` remains the name-based registry with thin `simple_*` delegates; behavior lives in `LoadedModel`. Concurrency deliberately undecided; LoRA routing unsupported; vLLM freeing uses the pinned engine-core shutdown, with process isolation as the strongest cleanup boundary.
- Validation history: local preflight plus L40S release run (`3 passed in 350.92s`); the disposable cluster was torn down with clean final SkyPilot state.

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
