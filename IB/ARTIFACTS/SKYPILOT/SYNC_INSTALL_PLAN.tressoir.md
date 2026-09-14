# SkyPilot exact sync and install: completion report

The approved source-mirror, artifact-transfer, and baked CUDA environment are implemented and proven on a live AWS L40S node.

## Executive Summary

| Remote location | Purpose | Ownership and lifetime |
| --- | --- | --- |
| `~/sky_workdir` | runnable source | exact local-owned mirror; disposable |
| `/opt/activation/.venv` | locked Python environment | baked into the image |
| `/root/.cache/huggingface` and `/root/.cache/flashinfer` | model and JIT reuse | outside upload deletion |
| `~/activation_artifacts` | checkpoints and results | explicit download; deleted by teardown |

The wrapper publishes or reuses a private ECR image based on a pinned CUDA 13 development image. `setup` validates `nvcc` and the baked lock. Every `exec` resumes with `--retry-until-up` if needed, exact-uploads source, shell-quotes the command safely, and runs against live-synced project imports.

The live basic engine test passed on one NVIDIA L40S. Exact deletion and artifact download also passed, and all temporary cloud instances were removed.

## Requested Decisions

All requested decisions are resolved and implemented:

- `upload` is the local-to-remote exact source operation and runs before every `exec`.
- `download` is an explicit remote-to-local artifact operation.
- `~/sky_workdir` is a strict disposable mirror.
- Caches and retained outputs live outside the mirror.
- The runtime uses a baked CUDA-devel image in private ECR.
- Live AWS setup, testing, and teardown were authorized and completed.

## Milestones

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M1 — Exact source mirror</span>
    <span class="card-oneliner">Delete stale remote code while protecting state by location.</span>
    <span class="card-badge">Completed</span>
  </summary>

#### What landed

`.skyignore` defines the transport namespace. `sky upload` targets only `/root/sky_workdir/` and uses `--delete --delete-excluded`. `sky exec` retries transient capacity failures while resuming, then performs that upload and before submission.

`activation/cloud/sky.py · exact upload`

```diff-python
+args = [
+    "-azP", "--itemize-changes", "--protect-args",
+    "--no-owner", "--no-group",
+    "--delete", "--delete-excluded",
+    "--filter=dir-merge,- .skyignore",
+]
+args.extend([f"{PROJECT_ROOT}/", f"{record['name']}:/root/sky_workdir/"])
```

#### Drift and challenge

The planned `copy` name was split into directional `upload` and `download` commands after user clarification. Live compound-command testing also exposed SkyPilot's shell reconstruction; `exec` now submits one `shlex.join`-serialized command. A later stopped-cluster run exposed single-attempt resume; `--retry-until-up` now keeps retrying transient capacity failures in the existing zone.

#### Validation

A remote-only workdir file was visibly deleted by the next automatic upload. The remote command ran afterward and confirmed the external artifact sentinel still existed.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M2 — Baked CUDA environment</span>
    <span class="card-oneliner">Supply nvcc and immutable locked dependencies without baking source.</span>
    <span class="card-badge">Completed</span>
  </summary>

#### What landed

`activation/cloud/Dockerfile` uses the digest-pinned `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` base, uv 0.12.5, managed Python 3.14, and `uv sync --frozen --no-install-project` into `/opt/activation/.venv`.

`activation/cloud/sky.py` adds ECR login, image build/push/reuse, image-tracked cluster labels, `nvcc` validation, and baked/source lock comparison. `ACTIVATION_SKY_IMAGE` supports another registry or account.

`activation/cloud/Dockerfile`

```diff-dockerfile
+FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04@sha256:0230b7f...
+ENV UV_PROJECT_ENVIRONMENT=/opt/activation/.venv \
+    CUDA_HOME=/usr/local/cuda
+COPY pyproject.toml uv.lock ./
+RUN uv python install 3.14 \
+    && uv sync --frozen --no-install-project
+ENV UV_NO_SYNC=1
```

#### Drift and challenge

The first image was built on a temporary SkyPilot CPU builder because this agent container lacked a Docker daemon. The builder was torn down. The immutable ECR image remains for reuse.

The first live pytest collection showed that source needed an explicit import boundary. `exec` now sets `PYTHONPATH=/root/sky_workdir`, leaving dependencies baked while code stays live.

#### Validation

- ECR tag: `uv-de16520b51ef997e`
- Digest: `sha256:2d5a7e6be7724eed9404b5a475cffbcbf5177ad9732c549e8b4f91186e797127`
- Live runtime: driver 580.159.04, CUDA 13.0, one L40S
- Setup passed CUDA compiler and lock checks
- `sky image` correctly reused the published immutable tag

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M3 — Safe download and live proof</span>
    <span class="card-oneliner">Retrieve retained artifacts and prove the engine call end to end.</span>
    <span class="card-badge">Completed</span>
  </summary>

#### What landed

`sky download` accepts sources only below `~/activation_artifacts`, rejects traversal and unsupported characters, never deletes local data, and defaults to preserving existing files. An in-project destination must be under `IB/TMP`.

`activation/cloud/sky.py · reliable exec`

```diff-python
+_upload_record(record)
+remote_command = shlex.join(
+    ["env", "PYTHONPATH=/root/sky_workdir", *command]
+)
 return _run_skypilot(
-    "exec", "--workdir", str(PROJECT_ROOT), ...,
+    "exec", "--gpus", requested_gpus, name, "--", remote_command,
 )
```

#### Validation

- Focused wrapper suite: `13 passed in 0.05s`.
- Live command: `uv run sky exec ac-test uv run pytest activation/tests/test_basic_engine.py --gpu`.
- Live result: `1 passed in 307.94s`.
- The test retrieved `Qwen/Qwen3.5-0.8B`, initialized vLLM, generated the expected greeting, and cleaned up.
- A retained 15-byte sentinel downloaded successfully into `IB/TMP`.
- SkyPilot and AWS both confirmed that the L40S cluster was gone after teardown.

Pause/resume persistence was not separately exercised. The upload boundary itself was proven; data that must survive teardown still requires durable object storage.

</details>

## Current Handoff

- [`.skyignore`](./.skyignore)
- [`activation/cloud/Dockerfile`](./activation/cloud/Dockerfile)
- [`activation/cloud/sky.py`](./activation/cloud/sky.py)
- [`activation/tests/test_sky.py`](./activation/tests/test_sky.py)
- [`README.md`](./README.md)
- [Detailed interactive review](./INTERACTIVE.tressoir.md)

No `pyproject.toml` change was needed because the existing console entry already exposes `uv run sky`.

