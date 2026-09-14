# SkyPilot exact-sync and install plan

Status: completed. The approved implementation was staged, tested locally, exercised on a live AWS L40S cluster, and the temporary cloud resources were torn down.

## Executive summary

The implementation separates mutable source from retained runtime state:

| Zone | Path | Owner | Behavior |
| --- | --- | --- | --- |
| Mirrored source | `~/sky_workdir` | local repository | exact replacement before every `exec` |
| Python environment | `/opt/activation/.venv` | project image | baked from `uv.lock` |
| Reusable caches | `/root/.cache/huggingface`, `/root/.cache/flashinfer` | remote runtime | outside upload deletion |
| Retained outputs | `~/activation_artifacts` | jobs/user | explicit remote-to-local download |

The runtime image uses a digest-pinned NVIDIA CUDA 13 development base, so FlashInfer and other JIT paths have `nvcc` and `/usr/local/cuda`. It is published in private ECR because SkyPilot pulls Docker images on the remote VM rather than uploading local image bytes.

## Accepted decisions

- Name local-to-remote source transfer `upload`.
- Run exact upload automatically after resume and before every `exec`; block execution on upload failure.
- Name remote-to-local artifact transfer `download` and make it explicit.
- Make `~/sky_workdir` a strict disposable mirror.
- Keep model caches, checkpoints, and results outside the mirror.
- Use a baked CUDA-devel project image in private ECR.
- Permit live AWS launch, validation, and teardown.

## M1 — Exact source ownership

Status: Completed.

### What landed

- Added root `.skyignore` as the explicit transport policy.
- Added `upload NAME [--dry-run]`.
- Upload uses a fixed `/root/sky_workdir/` target with `--delete` and `--delete-excluded`.
- Managed-cluster validation and stopped-cluster resume with `--retry-until-up` happen before transfer.
- `exec` invokes the same upload before job submission and no longer asks SkyPilot for a second additive workdir transfer.
- Added shell-safe command joining so compound command arguments retain their boundaries.

### Drift, challenges, and unplanned steps

The original plan used the ambiguous public name `copy`. User interaction clarified transfer direction, so the implementation uses `upload` for source-to-node and `download` for node-to-local artifacts.

Live validation also exposed that passing argv elements directly through `sky exec` loses grouping when SkyPilot reconstructs a shell command. The wrapper now uses `shlex.join` and has a regression assertion.

A later user run showed that automatic resume made only one start attempt and returned on AWS `InsufficientInstanceCapacity`. Resume now passes `--retry-until-up`; it waits for capacity in the stopped cluster's original zone and cannot relocate the attached disk.

### Validation

- A remote-only `~/sky_workdir/remote-only-stale.txt` was created.
- The next `exec` upload visibly emitted `*deleting remote-only-stale.txt`.
- The command ran only after transfer completed.
- Focused unit coverage verifies exact flags, fixed target, resume retry flags, upload/exec order, and serialized command text.

## M2 — CUDA and dependency image

Status: Completed.

### What landed

- Added `activation/cloud/Dockerfile`.
- Base image: `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` at digest `sha256:0230b7f...`.
- uv is pinned to 0.12.5 and installs managed Python 3.14.
- `uv sync --frozen --no-install-project` bakes dependencies into `/opt/activation/.venv`.
- `UV_NO_SYNC=1` keeps runtime commands from mutating that environment.
- Setup validates `/usr/local/cuda/bin/nvcc` and compares the uploaded `uv.lock` to the baked copy.
- The wrapper obtains temporary ECR credentials through AWS CLI and passes them to SkyPilot as task secrets.
- Added `sky image` to create/reuse the ECR repository, build, and publish the image.
- `ACTIVATION_SKY_IMAGE` can override the project default for another account or registry.

### Drift, challenges, and unplanned steps

The agent container did not expose a usable local Docker daemon. A disposable SkyPilot CPU builder performed the initial Docker build and ECR push, then was torn down. The resulting image is intentionally retained in ECR.

The baked environment does not install changing project source. The first live pytest collection therefore needed `PYTHONPATH=/root/sky_workdir` on every exec; this preserves the intended dependency/source split without a per-command install.

### Validation

- Published image: `814218043106.dkr.ecr.us-east-1.amazonaws.com/activation-context/sky:uv-de16520b51ef997e`.
- Manifest digest: `sha256:2d5a7e6be7724eed9404b5a475cffbcbf5177ad9732c549e8b4f91186e797127`.
- Compressed image size reported by ECR: 8,281,288,642 bytes.
- `sky image` subsequently recognized the immutable tag and skipped rebuilding.
- The live container reported NVIDIA driver 580.159.04 and CUDA 13.0; the setup `nvcc` and lock checks passed.

## M3 — Artifact download and end-to-end validation

Status: Completed with one noted validation omission.

### What landed

- Added `download NAME REMOTE_PATH LOCAL_PATH [--dry-run] [--overwrite]`.
- Remote sources are confined to `~/activation_artifacts` or its normalized `/root` form.
- Paths containing traversal segments, colons, or control characters are rejected.
- Downloads never use deletion and default to `--ignore-existing`.
- Destinations outside the project are allowed; destinations inside it must be beneath `IB/TMP`.
- Added focused wrapper tests and README operator documentation.

### Live validation

- Local wrapper suite: `13 passed in 0.05s`.
- SkyPilot launched an AWS `g6e.xlarge` with one L40S in `us-east-2` after capacity fallback from `us-east-1`.
- Command:
  `uv run sky exec ac-test uv run pytest activation/tests/test_basic_engine.py --gpu`
- Result: `1 passed in 307.94s`.
- The test downloaded `Qwen/Qwen3.5-0.8B`, initialized vLLM, generated a chat response containing “hello,” and executed model cleanup.
- An artifact sentinel outside the mirror survived a later exact upload.
- `sky download` retrieved the sentinel into `IB/TMP` with the exact 15-byte content `sky-download-ok`.
- `ac-test` was torn down.
- Final SkyPilot status: no clusters, managed jobs, or services.
- Final AWS `us-east-2` query: no pending, running, stopping, or stopped instances.

### Validation omission

Pause/resume persistence was not separately exercised before teardown. Persistence across repeated exact uploads was proven directly. Checkpoints that must survive teardown or instance loss still require S3 or another durable store.

## Acceptance criteria

| Criterion | Result |
| --- | --- |
| Remote-only workdir file deleted by next upload | Passed live |
| Artifact outside workdir survives upload | Passed live |
| `exec` waits for successful upload | Passed by construction and focused tests |
| CUDA compiler available | Passed during setup |
| Baked lock matches source lock | Passed during setup |
| Basic completions-style engine call | Passed live |
| Explicit artifact download into `IB/TMP` | Passed live |
| Temporary GPU cluster removed | Passed; verified in SkyPilot and AWS |
| Pause/resume persistence | Not separately exercised |

## External state intentionally retained

- Private ECR repository `activation-context/sky` in `us-east-1`.
- Validated immutable image tag `uv-de16520b51ef997e` and its layers.

No temporary SkyPilot cluster or builder remains.

## Apply surface

- `.skyignore`
- `activation/cloud/Dockerfile`
- `activation/cloud/sky.py`
- `activation/tests/test_sky.py`
- `README.md`

No `pyproject.toml` change was necessary because the existing `sky = "activation.cloud.sky:main"` entry already exposes the CLI.

