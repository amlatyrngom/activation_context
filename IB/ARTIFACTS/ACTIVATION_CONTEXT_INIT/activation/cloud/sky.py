"""Small, project-specific SkyPilot command wrapper.

SkyPilot currently supports Python through 3.13 while this project uses
Python 3.14. Each SkyPilot or AWS CLI command therefore runs through an
isolated ``uv tool run`` environment instead of importing it into the project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile


SKYPILOT_COMMAND = (
    "uv",
    "tool",
    "run",
    "--no-config",
    "--python",
    "3.13",
    "--with",
    "pip",
    "--from",
    "skypilot[aws]==0.13.0",
    "sky",
)
AWS_COMMAND = (
    "uv",
    "tool",
    "run",
    "--no-config",
    "--python",
    "3.13",
    "--from",
    "awscli==1.46.1",
    "aws",
)

GPU_TYPES = {
    "l40s": "L40S",
    "rtx6000pro": "RTXPRO6000",
}

AUTOSTOP_MINUTES = 30
DISK_SIZE_GB = 500
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_IMAGE_REPOSITORY = (
    "814218043106.dkr.ecr.us-east-1.amazonaws.com/activation-context/sky"
)
LOCK_DIGEST = hashlib.sha256((PROJECT_ROOT / "uv.lock").read_bytes()).hexdigest()[:16]
DEFAULT_IMAGE_ID = f"docker:{DEFAULT_IMAGE_REPOSITORY}:uv-{LOCK_DIGEST}"
IMAGE_ID = os.environ.get("ACTIVATION_SKY_IMAGE", DEFAULT_IMAGE_ID)
CONFIG_VERSION = "2"
LABEL_PREFIX = "activation-sky"
REMOTE_WORKDIR = "/root/sky_workdir"
REMOTE_ARTIFACTS = "/root/activation_artifacts"
PUBLIC_REMOTE_ARTIFACTS = "~/activation_artifacts"
LOCAL_DOWNLOAD_ROOT = PROJECT_ROOT / "IB" / "TMP"
ECR_REGISTRY = re.compile(
    r"^[0-9]+\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com$"
)
LOCAL_API_SERVER_ENDPOINTS = {
    "http://0.0.0.0:46580": "0.0.0.0",
    "http://localhost:46580": "localhost",
    "http://127.0.0.1:46580": "127.0.0.1",
}


def _run_skypilot(
    *args: str,
    capture_stdout: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*SKYPILOT_COMMAND, *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture_stdout else None,
    )


def _run_aws(
    *args: str,
    capture_stdout: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*AWS_COMMAND, *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture_stdout else None,
    )


def _run_rsync(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["rsync", *args],
        check=True,
        text=True,
    )


def _run_docker(
    *args: str,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=True,
        text=True,
        input=input_text,
    )


def _ensure_local_api_server() -> None:
    endpoint = os.environ.get(
        "SKYPILOT_API_SERVER_ENDPOINT",
        "http://127.0.0.1:46580",
    ).rstrip("/")
    host = LOCAL_API_SERVER_ENDPOINTS.get(endpoint)
    if host is None:
        return

    try:
        with socket.create_connection((host, 46580), timeout=0.2):
            return
    except OSError:
        _run_skypilot("api", "start")


def _parse_json_records(output: str) -> list[dict]:
    decoder = json.JSONDecoder()
    offset = 0
    for line in output.splitlines(keepends=True):
        if line.startswith("["):
            try:
                records, _ = decoder.raw_decode(output[offset:])
            except json.JSONDecodeError:
                pass
            else:
                return records
        offset += len(line)

    raise json.JSONDecodeError(
        "SkyPilot did not return a JSON record list",
        output,
        0,
    )


def _cluster_record(name: str, *, refresh: bool = False) -> dict | None:
    args = ["status", "--output", "json"]
    if refresh:
        args.append("--refresh")
    args.append(name)

    result = _run_skypilot(*args, capture_stdout=True)
    records = _parse_json_records(result.stdout)
    return next((record for record in records if record["name"] == name), None)


def _labels(gpu_type: str, gpu_count: int) -> dict[str, str]:
    return {
        f"{LABEL_PREFIX}-config": CONFIG_VERSION,
        f"{LABEL_PREFIX}-cloud": "aws",
        f"{LABEL_PREFIX}-gpu": gpu_type,
        f"{LABEL_PREFIX}-gpu-count": str(gpu_count),
        f"{LABEL_PREFIX}-disk-gb": str(DISK_SIZE_GB),
        f"{LABEL_PREFIX}-autostop-minutes": str(AUTOSTOP_MINUTES),
        f"{LABEL_PREFIX}-spot": "false",
    }


def _require_matching_config(
    record: dict,
    gpu_type: str,
    gpu_count: int,
) -> None:
    actual = record.get("labels") or {}
    expected = _labels(gpu_type, gpu_count)
    if all(actual.get(key) == value for key, value in expected.items()):
        return

    current = record.get("accelerators") or "unknown resources"
    requested = f"{GPU_TYPES[gpu_type]}:{gpu_count}"
    raise RuntimeError(
        f"cluster {record['name']!r} already exists with {current}, not the "
        f"tracked {requested} configuration; run "
        f"`uv run sky teardown {record['name']}` explicitly before replacing it"
    )


def _managed_gpu(record: dict) -> tuple[str, int]:
    labels = record.get("labels") or {}
    gpu_type = labels.get(f"{LABEL_PREFIX}-gpu")
    gpu_count_text = labels.get(f"{LABEL_PREFIX}-gpu-count")

    if gpu_type not in GPU_TYPES or gpu_count_text is None:
        raise RuntimeError(
            f"cluster {record['name']!r} was not created by `uv run sky setup`; "
            "refusing to guess its GPU configuration"
        )

    try:
        gpu_count = int(gpu_count_text)
    except ValueError as error:
        raise RuntimeError(
            f"cluster {record['name']!r} has an invalid tracked GPU count"
        ) from error

    _require_matching_config(record, gpu_type, gpu_count)
    return gpu_type, gpu_count


def _managed_record(
    name: str,
    *,
    refresh: bool = True,
) -> tuple[dict, str, int]:
    record = _cluster_record(name, refresh=refresh)
    if record is None:
        raise RuntimeError(
            f"cluster {name!r} does not exist; create it with `uv run sky setup`"
        )
    gpu_type, gpu_count = _managed_gpu(record)
    return record, gpu_type, gpu_count


def _start_if_stopped(record: dict) -> None:
    if record.get("status") != "STOPPED":
        return
    _run_skypilot(
        "start",
        "--yes",
        "--retry-until-up",
        "--idle-minutes-to-autostop",
        str(AUTOSTOP_MINUTES),
        record["name"],
    )


def _docker_registry_secrets() -> dict[str, str]:
    image = IMAGE_ID.removeprefix("docker:")
    registry = image.split("/", 1)[0]
    match = ECR_REGISTRY.fullmatch(registry)
    if match is None:
        return {}

    password = _run_aws(
        "ecr",
        "get-login-password",
        "--region",
        match.group("region"),
        capture_stdout=True,
    ).stdout.strip()
    return {
        "SKYPILOT_DOCKER_USERNAME": "AWS",
        "SKYPILOT_DOCKER_PASSWORD": password,
        "SKYPILOT_DOCKER_SERVER": registry,
    }


def _image_published() -> bool:
    """Whether IMAGE_ID already exists in its ECR repository."""
    image = IMAGE_ID.removeprefix("docker:")
    registry, repository_tag = image.split("/", 1)
    match = ECR_REGISTRY.fullmatch(registry)
    if match is None:
        return False
    repository, tag = repository_tag.rsplit(":", 1)
    try:
        _run_aws(
            "ecr",
            "describe-images",
            "--region",
            match.group("region"),
            "--repository-name",
            repository,
            "--image-ids",
            f"imageTag={tag}",
            capture_stdout=True,
        )
    except subprocess.CalledProcessError:
        return False
    return True


def _latest_published_image() -> str | None:
    """The most recently pushed image in the ECR repository, if any."""
    image = IMAGE_ID.removeprefix("docker:")
    registry, repository_tag = image.split("/", 1)
    match = ECR_REGISTRY.fullmatch(registry)
    if match is None:
        return None
    repository, _tag = repository_tag.rsplit(":", 1)
    try:
        result = _run_aws(
            "ecr",
            "describe-images",
            "--region",
            match.group("region"),
            "--repository-name",
            repository,
            "--query",
            "reverse(sort_by(imageDetails,&imagePushedAt))[0].imageTags[0]",
            "--output",
            "text",
            capture_stdout=True,
        )
    except subprocess.CalledProcessError:
        return None
    tag = result.stdout.strip()
    if not tag or tag == "None":
        return None
    return f"docker:{registry}/{repository}:{tag}"


def build_image() -> int:
    image = IMAGE_ID.removeprefix("docker:")
    registry, repository_tag = image.split("/", 1)
    match = ECR_REGISTRY.fullmatch(registry)
    if match is not None:
        region = match.group("region")
        repository, tag = repository_tag.rsplit(":", 1)
        try:
            _run_aws(
                "ecr",
                "describe-repositories",
                "--region",
                region,
                "--repository-names",
                repository,
                capture_stdout=True,
            )
        except subprocess.CalledProcessError:
            _run_aws(
                "ecr",
                "create-repository",
                "--region",
                region,
                "--repository-name",
                repository,
                "--image-scanning-configuration",
                "scanOnPush=true",
                "--image-tag-mutability",
                "IMMUTABLE",
            )
        try:
            _run_aws(
                "ecr",
                "describe-images",
                "--region",
                region,
                "--repository-name",
                repository,
                "--image-ids",
                f"imageTag={tag}",
                capture_stdout=True,
            )
        except subprocess.CalledProcessError:
            pass
        else:
            print(f"Image already published: {image}")
            return 0

        password = _run_aws(
            "ecr",
            "get-login-password",
            "--region",
            region,
            capture_stdout=True,
        ).stdout
        _run_docker(
            "login",
            "--username",
            "AWS",
            "--password-stdin",
            registry,
            input_text=password,
        )

    _run_docker(
        "build",
        "--pull",
        "--file",
        str(PROJECT_ROOT / "activation" / "cloud" / "Dockerfile"),
        "--tag",
        image,
        str(PROJECT_ROOT),
    )
    _run_docker("push", image)
    print(f"Published image: {image}")
    return 0


def _task_yaml(
    gpu_type: str,
    gpu_count: int,
    *,
    image_id: str = IMAGE_ID,
    docker_secrets: dict[str, str] | None = None,
) -> str:
    labels = "\n".join(
        f"    {key}: {json.dumps(value)}"
        for key, value in _labels(gpu_type, gpu_count).items()
    )
    secrets = docker_secrets if docker_secrets is not None else {}
    secrets_yaml = ""
    if secrets:
        secret_lines = "\n".join(
            f"  {key}: {json.dumps(value)}" for key, value in secrets.items()
        )
        secrets_yaml = f"secrets:\n{secret_lines}\n"
    workdir = json.dumps(str(PROJECT_ROOT))

    return f"""\
resources:
  infra: aws
  accelerators: {GPU_TYPES[gpu_type]}:{gpu_count}
  use_spot: false
  disk_size: {DISK_SIZE_GB}
  image_id: {image_id}
  labels:
{labels}
{secrets_yaml}workdir: {workdir}
setup: |
  test -x /usr/local/cuda/bin/nvcc
  mkdir -p {REMOTE_ARTIFACTS} /root/.cache/huggingface /root/.cache/flashinfer
"""


def setup(
    name: str,
    gpu_type: str,
    gpu_count: int = 1,
    rebuild_image: bool = True,
) -> int:
    gpu_type = gpu_type.lower()
    if gpu_type not in GPU_TYPES:
        choices = ", ".join(GPU_TYPES)
        raise ValueError(f"unknown GPU {gpu_type!r}; choose one of: {choices}")
    if gpu_count < 1:
        raise ValueError("gpu_count must be at least 1")

    record = _cluster_record(name)
    if record is not None:
        _require_matching_config(record, gpu_type, gpu_count)

    _run_skypilot("check", "aws")
    image_id = IMAGE_ID
    if not _image_published():
        if rebuild_image and shutil.which("docker") is not None:
            build_image()
        else:
            # The image is only a warm cache: any published image works
            # because `uv run` syncs the env to the uploaded lock on the node.
            fallback = _latest_published_image()
            if fallback is None:
                raise RuntimeError(
                    f"image {IMAGE_ID!r} is not published and no fallback "
                    "image exists; run `uv run sky image` on a machine with "
                    "docker"
                )
            print(f"Image {IMAGE_ID} not published; using {fallback}")
            image_id = fallback
    docker_secrets = _docker_registry_secrets()
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        encoding="utf-8",
    ) as task_file:
        task_file.write(
            _task_yaml(
                gpu_type,
                gpu_count,
                image_id=image_id,
                docker_secrets=docker_secrets,
            )
        )
        task_file.flush()
        _run_skypilot(
            "launch",
            "--cluster",
            name,
            "--idle-minutes-to-autostop",
            str(AUTOSTOP_MINUTES),
            "--yes",
            task_file.name,
        )
    return upload(name)


def teardown(name: str) -> int:
    return _run_skypilot("down", "--yes", name).returncode


def list() -> int:
    return _run_skypilot("status", "--refresh").returncode


def status(name: str) -> int:
    return _run_skypilot("status", "--refresh", name).returncode


def pause(name: str) -> int:
    return _run_skypilot("stop", "--yes", name).returncode


def _upload_record(record: dict, *, dry_run: bool = False) -> int:
    args = [
        "-azP",
        "--itemize-changes",
        "--protect-args",
        "--no-owner",
        "--no-group",
        "--delete",
        "--delete-excluded",
        "--filter=dir-merge,- .skyignore",
    ]
    if dry_run:
        args.append("--dry-run")
    args.extend(
        [
            f"{PROJECT_ROOT}/",
            f"{record['name']}:{REMOTE_WORKDIR}/",
        ]
    )
    return _run_rsync(*args).returncode


def upload(name: str, *, dry_run: bool = False) -> int:
    record, _, _ = _managed_record(name)
    _start_if_stopped(record)
    return _upload_record(record, dry_run=dry_run)


def _remote_artifact_path(path: str) -> str:
    if any(character in path for character in ("\x00", "\n", "\r", ":")):
        raise ValueError("remote artifact path contains an unsupported character")

    trailing_slash = path.endswith("/")
    prefixes = (PUBLIC_REMOTE_ARTIFACTS, REMOTE_ARTIFACTS)
    for prefix in prefixes:
        if path == prefix or path.startswith(f"{prefix}/"):
            suffix = path[len(prefix) :]
            parts = suffix.split("/")
            if any(part in (".", "..") for part in parts):
                raise ValueError("remote artifact path may not contain . or ..")
            normalized = f"{REMOTE_ARTIFACTS}{suffix}"
            if trailing_slash and not normalized.endswith("/"):
                normalized += "/"
            return normalized

    raise ValueError(
        f"remote downloads must come from {PUBLIC_REMOTE_ARTIFACTS!r}"
    )


def _local_download_path(path: str) -> Path:
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    destination = destination.resolve(strict=False)

    project_root = PROJECT_ROOT.resolve()
    download_root = LOCAL_DOWNLOAD_ROOT.resolve(strict=False)
    if destination.is_relative_to(project_root) and not destination.is_relative_to(
        download_root
    ):
        raise ValueError(
            "downloads inside the project must target IB/TMP; choose an "
            "outside path or an IB/TMP destination"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def download(
    name: str,
    remote_path: str,
    local_path: str,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> int:
    record, _, _ = _managed_record(name)
    _start_if_stopped(record)
    source = _remote_artifact_path(remote_path)
    destination = _local_download_path(local_path)

    args = [
        "-azP",
        "--itemize-changes",
        "--protect-args",
        "--no-owner",
        "--no-group",
    ]
    if not overwrite:
        args.append("--ignore-existing")
    if dry_run:
        args.append("--dry-run")
    args.extend([f"{record['name']}:{source}", str(destination)])
    return _run_rsync(*args).returncode


def exec_cmd(name: str, cmd: str | list[str]) -> int:
    command = [cmd] if isinstance(cmd, str) else cmd.copy()
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ValueError("exec requires a command after `--`")

    record, gpu_type, gpu_count = _managed_record(name)
    _start_if_stopped(record)
    _upload_record(record)

    remote_command = shlex.join(
        [
            "env",
            "VLLM_WORKER_MULTIPROC_METHOD=spawn",
            # The baked env is a warm cache: sync it to the uploaded lock
            # (old images set UV_NO_SYNC=1) without re-resolving on the node.
            "UV_NO_SYNC=0",
            "UV_FROZEN=1",
            f"PYTHONPATH={REMOTE_WORKDIR}",
            *command,
        ]
    )
    secret_file_args = (
        ["--secret-file", str(ENV_FILE)] if ENV_FILE.is_file() else []
    )
    return _run_skypilot(
        "exec",
        *secret_file_args,
        "--gpus",
        f"{GPU_TYPES[gpu_type]}:{gpu_count}",
        name,
        "--",
        remote_command,
    ).returncode


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sky",
        description="Manage the project's small AWS GPU workhorses with SkyPilot.",
        epilog=(
            "Authentication: SkyPilot and ECR read standard AWS credentials, "
            "including ~/.aws/credentials. Set AWS_PROFILE to select a "
            "non-default profile. Quote remote ~/activation_artifacts paths "
            "so the local shell does not expand them."
        ),
    )
    commands = parser.add_subparsers(dest="action", required=True)

    commands.add_parser(
        "image",
        help="build and publish the lock-tagged SkyPilot image",
    )

    setup_parser = commands.add_parser("setup", help="create or resume a node")
    setup_parser.add_argument("--name", required=True, help="cluster name")
    setup_parser.add_argument(
        "--gpu",
        required=True,
        choices=GPU_TYPES,
        help="GPU type",
    )
    setup_parser.add_argument(
        "--gpu-count",
        type=_positive_int,
        default=1,
        help="number of GPUs (default: 1)",
    )
    setup_parser.add_argument(
        "--no-image-rebuild",
        action="store_true",
        help="do not build/publish a missing lock-tagged image before launch",
    )

    exec_parser = commands.add_parser(
        "exec",
        help="upload current code and run a command on a node",
    )
    exec_parser.add_argument("name", help="cluster name")
    exec_parser.add_argument("command", nargs=argparse.REMAINDER)

    upload_parser = commands.add_parser(
        "upload",
        help="exactly mirror local code into the remote workdir",
    )
    upload_parser.add_argument("name", help="cluster name")
    upload_parser.add_argument("--dry-run", action="store_true")

    download_parser = commands.add_parser(
        "download",
        help="copy an artifact from the node to a local path",
    )
    download_parser.add_argument("name", help="cluster name")
    download_parser.add_argument(
        "remote_path",
        help="path under ~/activation_artifacts (quote paths beginning with ~)",
    )
    download_parser.add_argument("local_path", help="local destination")
    download_parser.add_argument("--dry-run", action="store_true")
    download_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing local files (default: keep them)",
    )

    teardown_parser = commands.add_parser(
        "teardown",
        help="permanently delete a node and its disk",
    )
    teardown_parser.add_argument("name", help="cluster name")

    pause_parser = commands.add_parser("pause", help="stop a node but keep its disk")
    pause_parser.add_argument("name", help="cluster name")

    commands.add_parser("ls", help="list nodes")

    status_parser = commands.add_parser("status", help="show one node")
    status_parser.add_argument("name", help="cluster name")

    commands.add_parser("help", help="show this help")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.action not in ("help", "image"):
            _ensure_local_api_server()

        if args.action == "image":
            return build_image()
        if args.action == "setup":
            return setup(
                args.name,
                args.gpu,
                args.gpu_count,
                rebuild_image=not args.no_image_rebuild,
            )
        if args.action == "exec":
            return exec_cmd(args.name, args.command)
        if args.action == "upload":
            return upload(args.name, dry_run=args.dry_run)
        if args.action == "download":
            return download(
                args.name,
                args.remote_path,
                args.local_path,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
        if args.action == "teardown":
            return teardown(args.name)
        if args.action == "pause":
            return pause(args.name)
        if args.action == "ls":
            return list()
        if args.action == "status":
            return status(args.name)

        parser.print_help()
        return 0
    except subprocess.CalledProcessError as error:
        if error.stdout:
            sys.stdout.write(error.stdout)
        if error.stderr:
            sys.stderr.write(error.stderr)
        return error.returncode or 1
    except (RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"sky: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
