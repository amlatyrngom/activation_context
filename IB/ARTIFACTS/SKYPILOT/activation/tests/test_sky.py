from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from activation.cloud import sky


def _record(*, status: str = "UP") -> dict:
    return {
        "name": "ac-test",
        "status": status,
        "accelerators": "L40S:1",
        "labels": sky._labels("l40s", 1),
    }


def _completed(args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args or [], 0, "", "")


def test_task_uses_baked_cuda_environment() -> None:
    task = sky._task_yaml(
        "l40s",
        1,
        docker_secrets={
            "SKYPILOT_DOCKER_USERNAME": "AWS",
            "SKYPILOT_DOCKER_PASSWORD": "temporary-token",
            "SKYPILOT_DOCKER_SERVER": "example.dkr.ecr.us-east-1.amazonaws.com",
        },
    )

    assert f"image_id: {sky.IMAGE_ID}" in task
    assert "test -x /usr/local/cuda/bin/nvcc" in task
    assert "cmp /opt/activation/image-lock/uv.lock uv.lock" in task
    assert "mkdir -p /root/activation_artifacts" in task
    assert "SKYPILOT_DOCKER_PASSWORD: \"temporary-token\"" in task


def test_upload_is_exact_and_fixed_target(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_rsync(*args: str) -> subprocess.CompletedProcess[str]:
        seen.extend(args)
        return _completed(list(args))

    monkeypatch.setattr(sky, "_run_rsync", fake_rsync)

    assert sky._upload_record(_record(), dry_run=True) == 0
    assert "--delete" in seen
    assert "--delete-excluded" in seen
    assert "--dry-run" in seen
    assert seen[-1] == "ac-test:/root/sky_workdir/"


def test_stopped_cluster_retries_until_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        sky,
        "_run_skypilot",
        lambda *args, **kwargs: calls.append(args) or _completed(list(args)),
    )

    sky._start_if_stopped(_record(status="STOPPED"))

    assert calls == [
        (
            "start",
            "--yes",
            "--retry-until-up",
            "--idle-minutes-to-autostop",
            str(sky.AUTOSTOP_MINUTES),
            "ac-test",
        )
    ]


def test_exec_uploads_after_resume_before_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    exec_args: list[str] = []
    record = _record(status="STOPPED")
    monkeypatch.setattr(
        sky,
        "_managed_record",
        lambda name: (record, "l40s", 1),
    )
    monkeypatch.setattr(
        sky,
        "_start_if_stopped",
        lambda actual: events.append("start"),
    )
    monkeypatch.setattr(
        sky,
        "_upload_record",
        lambda actual: events.append("upload") or 0,
    )

    def fake_exec(*args: str, **kwargs: object) -> subprocess.CompletedProcess[str]:
        events.append("exec")
        exec_args.extend(args)
        return _completed(list(args))

    monkeypatch.setattr(sky, "_run_skypilot", fake_exec)

    assert sky.exec_cmd("ac-test", ["bash", "-lc", "echo hello world"]) == 0
    assert events == ["start", "upload", "exec"]
    assert exec_args[-1] == (
        "env VLLM_WORKER_MULTIPROC_METHOD=spawn "
        "PYTHONPATH=/root/sky_workdir bash -lc 'echo hello world'"
    )


@pytest.mark.parametrize(
    ("provided", "expected"),
    [
        ("~/activation_artifacts/run-1/model.pt", "/root/activation_artifacts/run-1/model.pt"),
        ("~/activation_artifacts/run-1/", "/root/activation_artifacts/run-1/"),
        ("/root/activation_artifacts/model.pt", "/root/activation_artifacts/model.pt"),
    ],
)
def test_remote_artifact_path_is_confined(provided: str, expected: str) -> None:
    assert sky._remote_artifact_path(provided) == expected


@pytest.mark.parametrize(
    "provided",
    [
        "~/other/file",
        "~/activation_artifacts/../secret",
        "~/activation_artifacts/bad:name",
    ],
)
def test_remote_artifact_path_rejects_escape(provided: str) -> None:
    with pytest.raises(ValueError):
        sky._remote_artifact_path(provided)


def test_local_download_allows_ib_tmp_and_rejects_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(sky, "PROJECT_ROOT", project)
    monkeypatch.setattr(sky, "LOCAL_DOWNLOAD_ROOT", project / "IB" / "TMP")

    allowed = sky._local_download_path(str(project / "IB" / "TMP" / "run-1"))
    assert allowed == project / "IB" / "TMP" / "run-1"

    with pytest.raises(ValueError, match="must target IB/TMP"):
        sky._local_download_path(str(project / "activation" / "model.pt"))


def test_download_never_deletes_or_overwrites_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[str] = []
    record = _record()
    monkeypatch.setattr(
        sky,
        "_managed_record",
        lambda name: (record, "l40s", 1),
    )
    monkeypatch.setattr(sky, "_start_if_stopped", lambda actual: None)
    monkeypatch.setattr(
        sky,
        "_run_rsync",
        lambda *args: seen.extend(args) or _completed(list(args)),
    )

    destination = tmp_path / "download"
    assert (
        sky.download(
            "ac-test",
            "~/activation_artifacts/run-1/",
            str(destination),
            dry_run=True,
        )
        == 0
    )
    assert "--ignore-existing" in seen
    assert "--dry-run" in seen
    assert "--delete" not in seen
    assert seen[-2] == "ac-test:/root/activation_artifacts/run-1/"
    assert seen[-1] == str(destination)


def test_parser_uses_unambiguous_transfer_names() -> None:
    parser = sky._build_parser()

    upload = parser.parse_args(["upload", "ac-test", "--dry-run"])
    download = parser.parse_args(
        [
            "download",
            "ac-test",
            "~/activation_artifacts/run-1/",
            "IB/TMP/run-1/",
        ]
    )

    assert upload.action == "upload"
    assert upload.dry_run is True
    assert download.action == "download"
    assert not hasattr(download, "copy")
