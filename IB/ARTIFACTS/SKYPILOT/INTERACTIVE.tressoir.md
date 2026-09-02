# SkyPilot GPU workflow: live source, remote secrets, and retained artifacts

The staged handoff now provides a baked CUDA environment, exact source uploads, job-scoped `.env` injection outside that mirror, safe artifact downloads, live command output, and verified Qwen and Gemma completions-style vLLM tests on AWS.

## Executive Summary

The remote filesystem now has explicit ownership boundaries:

| Location | Owner | Transfer behavior | Lifetime |
| --- | --- | --- | --- |
| `~/sky_workdir` | local repository | replaced exactly before every `exec` | disposable |
| `/opt/activation/.venv` | baked image | never source-synced | image lifetime |
| `/root/.cache/{huggingface,flashinfer}` | remote runtime | never source-synced | survives upload/pause |
| `~/activation_artifacts` | remote jobs/user | downloaded only on request | survives upload/pause |

The wrapper publishes or reuses a private ECR image based on a digest-pinned NVIDIA CUDA 13 development image. It bakes Python 3.14 and the `uv.lock` dependency environment, validates `nvcc` and lock compatibility at setup, and keeps changing project code out of the image.

`sky exec` resumes a stopped managed cluster with `--retry-until-up`, performs an exact `rsync --delete --delete-excluded` upload, and submits only if upload succeeds. Commands are shell-serialized safely and receive `PYTHONPATH=/root/sky_workdir` plus vLLM's spawn-safe worker setting. If the project-root `.env` exists, SkyPilot injects it into the trusted job environment with `--secret-file`; the file never enters `~/sky_workdir`.

## Latest completed increment

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="./activation/cloud/sky.py">activation/cloud/sky.py</a></span>
    <span class="card-oneliner">Inject `.env` outside rsync and make remote vLLM startup spawn-safe.</span>
    <span class="card-badge">Copy this file</span>
  </summary>

~~~diff-python
+ENV_FILE = PROJECT_ROOT / ".env"

 remote_command = shlex.join([
     "env",
+    "VLLM_WORKER_MULTIPROC_METHOD=spawn",
     f"PYTHONPATH={REMOTE_WORKDIR}",
     *command,
 ])
+secret_file_args = (
+    ["--secret-file", str(ENV_FILE)] if ENV_FILE.is_file() else []
+)
 return _run_skypilot(
     "exec",
+    *secret_file_args,
     "--gpus",
     ...
 )
~~~

The live check observed the secret variable in the job environment and confirmed the mirrored `.env` was absent, without printing the value. The real Gemma 3 engine smoke then returned `Hello, World!` with `spawn`. Focused local validation now reports `19 passed` across the wrapper and helper checks.

</details>

## Live Outcome

- Published image: `814218043106.dkr.ecr.us-east-1.amazonaws.com/activation-context/sky:uv-de16520b51ef997e`
- Image digest: `sha256:2d5a7e6be7724eed9404b5a475cffbcbf5177ad9732c549e8b4f91186e797127`
- Runtime: AWS `g6e.xlarge` with one NVIDIA L40S in `us-east-2` after automatic capacity fallback from `us-east-1`.
- Driver/runtime check: NVIDIA driver 580.159.04, CUDA 13.0, and `nvcc` available from the image.
- Basic engine test: `1 passed in 307.94s`, including model retrieval, vLLM startup, chat generation, assertion, and cleanup.
- Gemma 3 engine smoke: `Hello, World!`; the separate Transformers model remained free after generation.
- Secret transport smoke: injected variable present; mirrored `.env` absent; no value retained in evidence.
- Wrapper tests: `13 passed in 0.05s`.
- Exact-upload smoke: a remote-only file under `~/sky_workdir` was visibly deleted.
- Persistence smoke: an artifact outside the mirror survived the next upload and downloaded into `IB/TMP` with exact content.
- Cleanup: `ac-test` was torn down; SkyPilot reports no clusters and AWS `us-east-2` reports no active instances.

The private ECR repository and validated image remain intentionally available for future clusters.

## Accepted Interface

`activation/cloud/sky.py` exposes:

```text
uv run sky image
uv run sky setup --name ac-test --gpu l40s --gpu-count 1
uv run sky upload ac-test [--dry-run]
uv run sky exec ac-test COMMAND...
uv run sky download ac-test REMOTE_PATH LOCAL_PATH [--dry-run] [--overwrite]
uv run sky pause ac-test
uv run sky teardown ac-test
uv run sky ls
uv run sky status ac-test
```

`upload` is local-to-remote and always exact. `download` is remote-to-local, accepts only sources under `~/activation_artifacts`, never deletes local files, and preserves existing local files unless `--overwrite` is explicit. A destination inside the repository must be under `IB/TMP`.

## Completed Changes

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M1 — Baked CUDA environment</span>
    <span class="card-oneliner">Replace mutable Ubuntu setup with an immutable CUDA-devel dependency image.</span>
    <span class="card-badge">Completed</span>
  </summary>

#### What landed

[Open `activation/cloud/Dockerfile`](./activation/cloud/Dockerfile)

The new image pins the CUDA base digest and uv release, installs the minimal SSH/rsync/build tools, installs managed Python 3.14, and freezes dependencies into `/opt/activation/.venv` without baking project source.

`activation/cloud/Dockerfile`

```diff-dockerfile
+FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04@sha256:0230b7f...
+ENV UV_PROJECT_ENVIRONMENT=/opt/activation/.venv \
+    CUDA_HOME=/usr/local/cuda \
+    HF_HOME=/root/.cache/huggingface
+COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/
+COPY pyproject.toml uv.lock ./
+RUN uv python install 3.14 \
+    && uv sync --frozen --no-install-project \
+    && uv cache clean
+ENV UV_NO_SYNC=1 \
+    PATH=/opt/activation/.venv/bin:/usr/local/bin:$PATH
```

[Open `activation/cloud/sky.py`](./activation/cloud/sky.py)

The wrapper derives the default ECR tag from `uv.lock`, obtains a temporary ECR login token from the standard AWS credential chain, validates CUDA and the baked lock at setup, and offers `sky image` for build/push or reuse.

`activation/cloud/sky.py · image and setup policy`

```diff-python
-IMAGE_ID = "docker:ubuntu:26.04"
+DEFAULT_IMAGE_REPOSITORY = (
+    "814218043106.dkr.ecr.us-east-1.amazonaws.com/activation-context/sky"
+)
+LOCK_DIGEST = hashlib.sha256((PROJECT_ROOT / "uv.lock").read_bytes()).hexdigest()[:16]
+DEFAULT_IMAGE_ID = f"docker:{DEFAULT_IMAGE_REPOSITORY}:uv-{LOCK_DIGEST}"
+IMAGE_ID = os.environ.get("ACTIVATION_SKY_IMAGE", DEFAULT_IMAGE_ID)

 setup: |
-  apt-get update
-  ... install uv and sync dependencies remotely ...
+  test -x /usr/local/cuda/bin/nvcc
+  cmp /opt/activation/image-lock/uv.lock uv.lock || {
+    echo "uv.lock differs from the baked SkyPilot image; rebuild it" >&2
+    exit 1
+  }
+  mkdir -p /root/activation_artifacts /root/.cache/huggingface /root/.cache/flashinfer
```

#### Drift and challenge

The agent container had no usable local Docker daemon, so the first image was built and pushed from a temporary SkyPilot CPU builder. That builder was torn down. The public `sky image` command then correctly recognized the immutable ECR tag as already published.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M2 — Exact upload and retained artifacts</span>
    <span class="card-oneliner">Give source and generated state separate, testable ownership boundaries.</span>
    <span class="card-badge">Completed</span>
  </summary>

#### What landed

[Open `.skyignore`](./.skyignore)

The committed transport policy excludes Git state, environments, secrets, generated Python/test caches, and `IB/` from the remote source mirror.

`.skyignore`

```diff
+.git/
+.venv/
+.env
+__pycache__/
+*.py[cod]
+.pytest_cache/
+.ruff_cache/
+.mypy_cache/
+IB/
```

`activation/cloud/sky.py · upload() and download()`

```diff-python
+def _upload_record(record: dict, *, dry_run: bool = False) -> int:
+    args = [
+        "-azP", "--itemize-changes", "--protect-args",
+        "--no-owner", "--no-group",
+        "--delete", "--delete-excluded",
+        "--filter=dir-merge,- .skyignore",
+    ]
+    ...
+    args.extend([f"{PROJECT_ROOT}/", f"{record['name']}:/root/sky_workdir/"])
+    return _run_rsync(*args).returncode
+
+def download(..., overwrite: bool = False) -> int:
+    source = _remote_artifact_path(remote_path)
+    destination = _local_download_path(local_path)
+    args = ["-azP", "--itemize-changes", "--protect-args", ...]
+    if not overwrite:
+        args.append("--ignore-existing")
+    ...
```

The public names intentionally distinguish direction: `upload` mirrors local code to the node; `download` retrieves remote artifacts.

#### Validation

A remote `remote-only-stale.txt` under the workdir was created after one upload. The following `exec` visibly reported `*deleting remote-only-stale.txt` before its command ran. A sentinel under `/root/activation_artifacts/download-smoke/result.txt` survived and downloaded as the exact 15-byte `sky-download-ok` payload.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M3 — Reliable live exec</span>
    <span class="card-oneliner">Keep source editable while preserving arbitrary command grouping and live output.</span>
    <span class="card-badge">Completed</span>
  </summary>

#### What landed

`activation/cloud/sky.py · _start_if_stopped(), exec_cmd()`

```diff-python
 def _start_if_stopped(record: dict) -> None:
     ...
     _run_skypilot(
         "start",
         "--yes",
+        "--retry-until-up",
         "--idle-minutes-to-autostop",
         ...
     )

 def exec_cmd(name: str, cmd: str | list[str]) -> int:
-    ... SkyPilot additive workdir sync ...
+    record, gpu_type, gpu_count = _managed_record(name)
+    _start_if_stopped(record)
+    _upload_record(record)
+
+    remote_command = shlex.join(
+        ["env", "PYTHONPATH=/root/sky_workdir", *command]
+    )
     return _run_skypilot(
         "exec",
-        "--workdir", str(PROJECT_ROOT),
         "--gpus", f"{GPU_TYPES[gpu_type]}:{gpu_count}",
-        name, "--", *command,
+        name, "--", remote_command,
     ).returncode
```

#### Live-discovered fixes

The first GPU test reached pytest but failed collection because the project itself is deliberately absent from the baked environment. Injecting the fixed workdir into `PYTHONPATH` preserves the desired split: dependencies are immutable, source remains live.

The first compound storage-smoke command then exposed SkyPilot's shell reconstruction behavior. `shlex.join` now protects arguments such as `bash -lc '...'`. A focused unit assertion covers the exact serialized command.

A later stopped-cluster run reached `sky start` but failed on one AWS `InsufficientInstanceCapacity` response. Resume now uses `--retry-until-up`, so transient availability errors remain attached and retry until the existing instance starts. SkyPilot cannot move a stopped cluster or its disk to another zone; teardown plus setup remains the relocation path.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M4 — Tests and documentation</span>
    <span class="card-oneliner">Cover safety contracts and document the end-to-end operator workflow.</span>
    <span class="card-badge">Completed</span>
  </summary>

#### What landed

[Open `activation/tests/test_sky.py`](./activation/tests/test_sky.py)

Thirteen focused tests cover persistent capacity retry, the baked task, fixed exact-upload target, resume/upload/exec ordering, shell-safe serialization, remote artifact confinement, local `IB/TMP` enforcement, non-deleting/non-overwriting downloads, and unambiguous parser names.

[Open `README.md`](./README.md)

The README now documents image publication, setup, automatic and explicit upload, download, persistence boundaries, pause, teardown, and the `ACTIVATION_SKY_IMAGE` registry override. The existing `pyproject.toml` console entry already supports `uv run sky`, so no dependency or project configuration change was needed.

#### Residual boundary

Pause/resume persistence was not separately exercised in the final live run. Persistence across exact uploads was proven directly, and SkyPilot's pause command remains non-destructive by design. Anything that must survive teardown or instance loss still belongs in durable object storage rather than only on the node disk.

</details>

## Apply Surface

The current source-shaped handoff remains these five files; for this increment, copy the updated `activation/cloud/sky.py` and adapt tests/docs only if useful in your branch:

- [`.skyignore`](./.skyignore)
- [`activation/cloud/Dockerfile`](./activation/cloud/Dockerfile)
- [`activation/cloud/sky.py`](./activation/cloud/sky.py)
- [`activation/tests/test_sky.py`](./activation/tests/test_sky.py)
- [`README.md`](./README.md)

The handoff folder contains no `orig.py`, patch rejects, Python bytecode, or superseded source snapshots.
