# Secrets, Gemma 3, and side-model embeddings — completed plan

Status: complete. The user approved the recommended batch on 30 August 2026.

## Objective

Complete one bounded implementation-and-validation round that:

1. injects the local project `.env` into trusted SkyPilot jobs without placing it in the exact source mirror;
2. live-tests `google/gemma-3-1b-it` through both vLLM prompt embeddings and the project engine;
3. adds an independently owned copy of the target model's primary input-embedding module while preserving the full model's starting placement;
4. measures main embedding-table residency for the paper's Qwen 3, Qwen 3.5, Gemma 3, and Gemma 4 fixtures, including the requested 8B–32B range; and
5. gives a family-scoped verdict on direct side-model activations.

## Accepted scope and boundaries

- Automatically pass an existing project-root `.env` to wrapped `sky exec` through SkyPilot `--secret-file`.
- Keep `.env` excluded from rsync, images, caches, and artifacts.
- Add a small policy-free embedding-module copy helper; do not add a permanent GPU cache.
- Assume side tensors already have the target `d_model`; investigate representation compatibility rather than restating structural shape requirements.
- Do not design concurrency, train a projector, or broaden the engine API in this round.
- Retrieve sanitized evidence and tear down every experiment cluster.

## Completed implementation

### SkyPilot

- `activation/cloud/sky.py` detects the project-root `.env` and supplies it as `--secret-file` for `exec` only when present.
- Remote commands receive `VLLM_WORKER_MULTIPROC_METHOD=spawn` as well as the live-source `PYTHONPATH`; this fixes vLLM engine construction after the parent process has inspected CUDA.
- The exact source mirror remains unchanged and contains no `.env`.

### Embedding extraction

- `activation/harness/hf_utils.py::copy_input_embedding_layer()` deep-copies `model.get_input_embeddings()`, freezes it, switches it to evaluation mode, and moves the independent module to the requested device.
- `LoadedBaseModel.extract_input_embedding_layer()` records the full model's starting placement. A model that starts free is temporarily loaded and restored to free in `finally`; an already loaded model remains where it started.
- The helper intentionally returns the primary input module. Gemma 4 E2B's separate per-layer embedding path is an architecture-level concern, not hidden inside this generic helper.

## Completed experiments

- Gemma 3 direct-string and typed-text messages produced identical prompt and output token IDs.
- Exact Gemma 3 target rows reproduced ordinary generation for both a full prompt and a hybrid seven-row user span.
- Gemma 3's real input module applies the model's required scaling; the cloned module reproduced vLLM rows exactly and survived freeing the full model.
- A Qwen 3 Embedding 0.6B side model and Qwen 3.5 0.8B target produced the same seven rows at `d_model=1024`. vLLM accepted both. Exact target rows reproduced the baseline, while direct side activations produced a different response. The mismatch is learned coordinate space, not tensor shape.
- Gemma 4 E2B uses token-conditioned per-layer embeddings in addition to its main `d_model` rows. Main rows alone are therefore not the complete text-equivalent carrier; meaningful companion target token IDs can preserve the PLE path for text-derived rows.
- The audited 8B–32B main embedding tables occupy 1.16–2.62 GiB in BF16 and 2.37–10.54% of checkpoint parameters. Absolute residency, rather than percentage alone, governs pinning.

## Validation and cleanup

- Focused local suite: `19 passed in 8.18s`.
- Gemma 3 project-engine smoke: `Hello, World!`; the Transformers model remained free after vLLM generation.
- Secret probe: the variable was present in the job environment and the mirrored `.env` was absent; no secret value was printed or retained.
- Sanitized experiment JSON is stored beside the cumulative HTML report.
- `ac-embed-next` was terminated after retrieval. Final SkyPilot status: no clusters, managed jobs, or services.

## Deliverables

- Cumulative report: `INPUT_EMBEDS_REPORT.tressoir.html`.
- Staged Sky wrapper: `../SKYPILOT/activation/cloud/sky.py`.
- Staged helper: `activation/harness/hf_utils.py` plus the lifecycle method in `activation/harness/runtime.py`.
- Current review projections: both subsystem `INTERACTIVE.tressoir.md` files.

