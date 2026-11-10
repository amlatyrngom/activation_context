"""
Minimal S3 transfer for the section 4.5 jobs (boto3, default credential chain: environment, ~/.aws, instance profile; nothing is
printed about credentials). A transfer skips a file only when the other side has the same size AND is at least as new (2026-11-10 fix:
size alone left same-size files stale on S3, e.g. a recovery slot's state.json / partial_epoch.json / optimizer.pt, so a reclaimed spot
node resumed from an old slot). NEVER uploads optimizer.pt unless S3X_ALLOW_OPTIMIZER=1; putdir skips *.tmp / *.part files and
folders, uploads the slot pointers (state.json, partial_epoch.json) after everything else and a DONE file last.

  python s3x.py get    s3://BUCKET/KEY        LOCAL_FILE
  python s3x.py getdir s3://BUCKET/PREFIX/    LOCAL_DIR          (every object under the prefix, same relative paths)
  python s3x.py putdir LOCAL_DIR              s3://BUCKET/PREFIX/ [--exclude NAME ...]   (relative paths kept; skips symlinks)
  python s3x.py put    LOCAL_FILE             s3://BUCKET/KEY
  python s3x.py check                         (exit 0 when boto3 and credentials are usable)
Exit code 3: no boto3 / no credentials (the caller falls back or keeps outputs local).
"""
import os
import sys
from pathlib import Path

NEVER = set() if os.environ.get("S3X_ALLOW_OPTIMIZER") == "1" else {"optimizer.pt"}   # EC2 spot resume uploads the recovery slot


def split(url):
    assert url.startswith("s3://"), url
    bucket, _, key = url[5:].partition("/")
    return bucket, key


def client():
    try:
        import boto3
        from boto3.s3.transfer import TransferConfig
    except ImportError:
        print("s3x: boto3 is not installed", file=sys.stderr)
        sys.exit(3)
    session = boto3.session.Session()
    if session.get_credentials() is None:
        print("s3x: no AWS credentials on this machine", file=sys.stderr)
        sys.exit(3)
    return session.client("s3"), TransferConfig(multipart_threshold=64 * 2**20, multipart_chunksize=64 * 2**20, max_concurrency=8)


POINTERS = {"state.json", "partial_epoch.json"}                  # recovery-slot pointers: uploaded after the slot they point to


def head(s3, bucket, key):
    """(size, last-modified epoch seconds) of an object, or (None, None)."""
    try:
        h = s3.head_object(Bucket=bucket, Key=key)
        return h["ContentLength"], h["LastModified"].timestamp()
    except Exception:
        return None, None


def size_of(s3, bucket, key):
    return head(s3, bucket, key)[0]


def get(s3, config, url, local):
    bucket, key = split(url)
    local = Path(local)
    remote, modified = head(s3, bucket, key)
    if remote is None:
        raise SystemExit(f"s3x: missing {url}")
    if local.is_file() and local.stat().st_size == remote and local.stat().st_mtime >= modified:
        return 0
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_name(local.name + ".part")
    s3.download_file(bucket, key, str(temporary), Config=config)
    temporary.rename(local)
    return 1


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    command, args = sys.argv[1], sys.argv[2:]
    s3, config = client()
    if command == "check":
        return
    if command == "get":
        get(s3, config, args[0], args[1])
    elif command == "getdir":
        bucket, prefix = split(args[0].rstrip("/") + "/")
        count = 0
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                if Path(item["Key"]).name in NEVER:
                    continue
                count += get(s3, config, f"s3://{bucket}/{item['Key']}", Path(args[1]) / item["Key"][len(prefix):])
        print(f"s3x: {count} files fetched from {args[0]}")
    elif command in ("put", "putdir"):
        exclude = set(NEVER)
        if "--exclude" in args:
            exclude |= set(args[args.index("--exclude") + 1:])
            args = args[:args.index("--exclude")]
        source, (bucket, prefix) = Path(args[0]), split(args[1])
        files = [(source, prefix)] if command == "put" else []
        if command == "putdir":
            prefix = prefix.rstrip("/") + "/"
            for folder, dirs, names in os.walk(source):
                dirs[:] = [d for d in dirs if d not in exclude and not d.endswith((".tmp", ".part"))]     # half-written folders (link_pair's NAME.tmp)
                for name in names:
                    path = Path(folder) / name
                    if name in exclude or path.is_symlink() or not path.is_file() or name.endswith((".part", ".tmp")):
                        continue
                    files.append((path, prefix + str(path.relative_to(source))))
        files.sort(key=lambda entry: (entry[0].name == "DONE", entry[0].name in POINTERS))   # pointers after their slots; DONE last (its presence implies the rest)
        sent = 0
        for path, key in files:
            if path.name in NEVER:
                continue
            remote, modified = head(s3, bucket, key)
            if remote == path.stat().st_size and modified >= path.stat().st_mtime:
                continue
            s3.upload_file(str(path), bucket, key, Config=config)
            sent += 1
        print(f"s3x: {sent} of {len(files)} files uploaded to s3://{bucket}/{prefix}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
