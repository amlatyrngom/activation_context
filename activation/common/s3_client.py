"""
Very simple wrapper around S3. Not used anywhere yet.

Where the settings come from: the client reads only `os.environ`, the names being AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN (optional), AWS_DEFAULT_REGION and ACTIVATION_S3_BUCKET (default
DEFAULT_BUCKET).
- Locally those live in the project's .env file: the first client built in a process loads it into the environment
  with python-dotenv (a project dependency; skipped quietly when it is not importable), without printing anything and
  without overriding variables that are already set.
- On Sky nodes there is no .env file. The `activation/cloud/sky.py` wrapper passes the local .env as `--secret-file`,
  which injects the same names as environment variables, so the client works unchanged there. Make sure the sky nodes
  get that file.

boto3 is not a project dependency (see pyproject.toml) and is imported lazily in the constructor, which raises a
RuntimeError naming the missing package, or the missing credential variables, so a misconfigured node fails early.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_BUCKET = "activation-context-artifacts-bucket"
_MISSING_KEY_CODES = {"404", "NoSuchKey", "NotFound"}
_dotenv_loaded = False


def _load_dotenv_once() -> None:
    """Load the project's .env into os.environ (existing variables win). Quiet, and a no-op without python-dotenv."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


class S3Client:
    def __init__(self, bucket: str | None = None, region: str | None = None):
        _load_dotenv_once()
        env = os.environ
        self.bucket = bucket or env.get("ACTIVATION_S3_BUCKET") or DEFAULT_BUCKET
        self.region = region or env.get("AWS_DEFAULT_REGION") or None
        key_id = env.get("AWS_ACCESS_KEY_ID")
        secret = env.get("AWS_SECRET_ACCESS_KEY")
        missing = [name for name, value in (("AWS_ACCESS_KEY_ID", key_id), ("AWS_SECRET_ACCESS_KEY", secret)) if not value]
        if missing:
            raise RuntimeError(f"S3Client: missing credentials in the environment: {', '.join(missing)} "
                               "(put them in the project .env locally; on Sky nodes pass the .env as --secret-file)")
        try:
            import boto3
        except ImportError as error:
            raise RuntimeError("S3Client needs the `boto3` package, which is not a project dependency; install it in the "
                               "environment that uses S3 (e.g. `uv pip install boto3`)") from error
        self._client = boto3.client("s3", region_name=self.region, aws_access_key_id=key_id, aws_secret_access_key=secret,
                                    aws_session_token=env.get("AWS_SESSION_TOKEN") or None)

    # ----------------------------------------------------------------------------- single objects
    def upload_file(self, local_path: str | Path, key: str) -> None:
        self._client.upload_file(str(local_path), self.bucket, key)

    def download_file(self, key: str, local_path: str | Path) -> None:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, key, str(local_path))

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
        except Exception as error:                              # botocore.exceptions.ClientError, without importing botocore
            response = getattr(error, "response", None)
            code = str(response.get("Error", {}).get("Code", "")) if isinstance(response, dict) else ""
            if code in _MISSING_KEY_CODES:
                return False
            raise
        return True

    # ----------------------------------------------------------------------------- folders
    def list_keys(self, prefix: str) -> list[str]:
        return sorted(self._list_sizes(prefix))

    def sync_folder_up(self, local_folder: str | Path, prefix: str) -> list[str]:
        """Upload every file under `local_folder` whose key is missing under `prefix` or whose size differs. Returns the uploaded keys."""
        local_folder = Path(local_folder)
        prefix = prefix.strip("/")
        remote = self._list_sizes(prefix + "/" if prefix else "")
        uploaded: list[str] = []
        for path in sorted(p for p in local_folder.rglob("*") if p.is_file()):
            relative = path.relative_to(local_folder).as_posix()
            key = f"{prefix}/{relative}" if prefix else relative
            if remote.get(key) != path.stat().st_size:
                self.upload_file(path, key)
                uploaded.append(key)
        return uploaded

    def _list_sizes(self, prefix: str) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for page in self._client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []) or []:
                sizes[item["Key"]] = int(item.get("Size", 0))
        return sizes
