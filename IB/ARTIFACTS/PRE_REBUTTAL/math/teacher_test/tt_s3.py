"""Small S3 mirror for the teacher-test node (boto3; credentials from the standard chain, never printed).
    python tt_s3.py get  s3://bucket/prefix/ LOCAL_DIR      # download every object under the prefix (missing or size-changed files)
    python tt_s3.py put  LOCAL_DIR s3://bucket/prefix/      # upload every file whose size differs from the S3 copy
"""
import sys
from pathlib import Path

import boto3


def split(url: str) -> tuple[str, str]:
    bucket, _, prefix = url[len("s3://"):].partition("/")
    return bucket, prefix


def listing(client, bucket: str, prefix: str) -> dict[str, int]:
    out = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            out[item["Key"]] = item["Size"]
    return out


def main() -> int:
    action, a, b = sys.argv[1:4]
    client = boto3.client("s3")
    if action == "get":
        bucket, prefix = split(a)
        moved = 0
        for key, size in listing(client, bucket, prefix).items():
            local = Path(b) / key[len(prefix):].lstrip("/")
            if key.endswith("/") or (local.exists() and local.stat().st_size == size):
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local))
            moved += 1
        print(f"tt_s3: {moved} files fetched from {a}", flush=True)
    else:
        bucket, prefix = split(b)
        prefix = prefix.rstrip("/") + "/"
        remote = listing(client, bucket, prefix)
        moved = 0
        for local in sorted(Path(a).rglob("*")):
            if local.is_file():
                key = prefix + str(local.relative_to(a))
                if remote.get(key) != local.stat().st_size:
                    client.upload_file(str(local), bucket, key)
                    moved += 1
        print(f"tt_s3: {moved} files uploaded to {b}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
