"""
Canonical exporter of teacher rollouts as reusable text-only SFT segments. Every recorded run is unrolled into its
segments (earlier compacted segments, the final segment, and every subagent run beneath it, recursively), so each row is
one agent's conversation exactly as the teacher saw it: messages (system, prompt, then every assistant and tool message),
the index of the first recorded turn (`target_start`), the tool definitions recovered from the recorded prompt tokens, and
the root task's labels, outcome and split. Nothing is filtered: unfinished and wrong runs are kept with their finish
reason and score, so selection happens when a mixture is built.

Held-out split: by the root task (benchmark, `task_id`; dataset ids embed loader counts and are not stable across
campaigns), so every segment of a held run, and every later sample of the same task, stays held. Rows written for a cache are not written again: a later export of a growing cache adds a release
with the new runs only.

    python -m activation.bench.canonical_training.export_agent_teacher_segments \\
        --cache IB/TMP/AGENT_AC_PRETRAINING/node_camp/ROLLOUTS/teacher_campaign_12h/rollouts.jsonl --release campaign_12h \\
        --tokenizer Qwen/Qwen3.8-27B-FP8 --model-id unsloth/Qwen3.8-27B-NVFP4 --output IB/TMP/SYNC/REUSABLE_DATA/agent_teacher_regular \\
        --s3-prefix activation-context-data/datasets/agent_teacher_regular
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from activation.common.ac_parts import is_ac_part

SCHEMA = "agent-teacher-segment-v1"
HELD_PERCENT = 4
ROWS_PER_SHARD = 500
BUCKET = "general-bucket-aln"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    temporary.replace(path)


def task_split(benchmark: str, task_id: str) -> str:
    return "held" if int(digest(["agent_teacher_regular_held", benchmark, str(task_id)])[:8], 16) % 100 < HELD_PERCENT else "train"


def last_rows(cache: Path, skip: set[str]):
    """
    The cache's rows, the last one per config key (the rollout cache's own rule: a redo supersedes the earlier row), minus
    the keys in `skip`. Streams twice (key offsets first, then only the winning lines) so a multi-GB cache fits in RAM.
    """
    offsets: dict[str, int] = {}
    with cache.open("rb") as handle:
        while True:
            offset, line = handle.tell(), handle.readline()
            if not line:
                break
            marker = line.rfind(b'"config_key": ')
            if marker < 0 or not line.rstrip().endswith(b"}"):
                continue                                   # a torn final line of a live cache
            try:
                key = json.JSONDecoder().raw_decode(line[marker + len(b'"config_key": '):].decode())[0]
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            offsets[key] = offset
    with cache.open("rb") as handle:
        for key, offset in sorted(offsets.items(), key=lambda item: item[1]):
            if key in skip:
                continue
            handle.seek(offset)
            yield json.loads(handle.readline())


class ToolDefinitions:
    """The tool definitions a segment's model saw, parsed from the <tools> block of its recorded prompt tokens."""

    def __init__(self, tokenizer_name: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, prompt_token_ids: list[int]) -> list[dict]:
        for limit in (8192, None):
            text = self.tokenizer.decode(prompt_token_ids[:limit] if limit else prompt_token_ids)
            start, end = text.find("<tools>"), text.find("</tools>")
            if start >= 0 and end > start:
                return [json.loads(line) for line in text[start + len("<tools>"):end].strip().splitlines() if line.strip()]
        return []


def segments(run: dict, path: str = "main", depth: int = 0):
    """(path, depth, segment) for the run's compacted segments, the run, and every subagent run beneath each, depth first."""
    for index, segment in enumerate(list(run.get("compactions") or []) + [run]):
        here = f"{path}/compaction_{index}" if segment is not run else path
        yield here, depth, segment
        for child_index, child in enumerate(segment.get("subagent_results") or []):
            yield from segments(child, f"{here}/subagent_{child_index}", depth + 1)


def segment_rows(row: dict, tools: ToolDefinitions, release: str, model_id: str) -> list[dict]:
    config = row["agent_config"]
    task, metadata = config.get("dataset_task") or {}, config.get("metadata") or {}
    root = {"config_key": row["config_key"], "benchmark": metadata.get("benchmark"), "task_kind": metadata.get("task_kind"),
            "teacher_kind": metadata.get("teacher_kind"), "template": metadata.get("template"), "dataset_id": task.get("dataset_id"),
            "task_id": task.get("task_id"), "gold_answer": task.get("gold_answer"), "finish_reason": row.get("finish_reason"),
            "score": row.get("score"), "answer": row.get("answer")}
    split = task_split(root["benchmark"], root["task_id"])
    call_kwargs = config.get("call_kwargs") or {}
    out = []
    for path, depth, segment in segments(row):
        messages = [dict(message) for message in segment["prompt_messages"]]
        target_start = len(messages)
        for step in segment["trajectory"]:
            messages.extend(step["messages"])
        if not any(message.get("role") == "assistant" for message in messages[target_start:]):
            continue                                       # no recorded turn: nothing to learn from this segment
        if any(is_ac_part(part) for message in messages if isinstance(message.get("content"), list) for part in message["content"]):
            raise ValueError(f"{row['config_key']} {path}: an AC part in what the agent saw (text-only teachers expected)")
        agent_config = segment.get("agent_config") or {}
        out.append({
            "schema": SCHEMA, "segment_id": digest([row["config_key"], path]), "root_id": digest(row["config_key"]), "path": path,
            "split": split, "root": root,
            "segment": {"agent_name": agent_config.get("agent_name"), "role": "main" if depth == 0 else "subagent", "depth": depth,
                        "finish_reason": segment.get("finish_reason"), "num_turns": segment.get("num_turns"),
                        "num_output_tokens": segment.get("num_output_tokens"), "answer": segment.get("answer")},
            "messages": messages, "target_start": target_start, "tools": tools(segment["prompt_token_ids"]),
            "chat_template_kwargs": dict(call_kwargs.get("chat_template_kwargs") or {}),
            "provenance": {"release": release, "model_id": model_id, "sampling_params": call_kwargs.get("sampling_params"),
                           "seed": row.get("seed"), "ac_model_name": row.get("ac_model_name")},
        })
    return out


def exported_keys(collection: Path, manifest: dict, source: str) -> set[str]:
    keys = set()
    for release in manifest["releases"]:
        if release.get("source") != source:
            continue
        child = json.loads((collection / release["manifest"]).read_text())
        for shard in child["shards"]:
            with (collection / release["name"] / shard["file"]).open() as handle:
                keys.update(json.loads(line)["root"]["config_key"] for line in handle)
    return keys


def export(cache: Path, release: str, tokenizer: str, model_id: str, collection: Path) -> dict | None:
    collection.mkdir(parents=True, exist_ok=True)
    manifest_path = collection / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema": "agent_teacher_segment_collection_v1", "data_label": "regular_agent_tasks",
        "description": ("Teacher rollouts of the regular agent campaign (MuSiQue, DAPO, DeepMath, LCA bug localization, NarrativeQA, "
                        "LOFT, Loong, BRIGHT), unrolled into text-only segments: compacted segments, final segments and subagent runs. "
                        "Nothing filtered; root finish reason and score recorded. Split by root task."),
        "held_percent": HELD_PERCENT, "releases": []}
    if any(entry["name"] == release for entry in manifest["releases"]):
        raise SystemExit(f"release {release!r} already exists in {collection}")
    source = str(cache.resolve())
    done = exported_keys(collection, manifest, source)
    definitions = ToolDefinitions(tokenizer)
    folder = collection / release
    folder.mkdir()
    shards, buffer, counts = [], [], {"runs": 0, "segments": 0, "held_segments": 0, "output_tokens": 0}

    def flush():
        if not buffer:
            return
        path = folder / f"shard_{len(shards):05d}.jsonl"
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in buffer))
        shards.append({"file": path.name, "sha256": sha256(path), "rows": len(buffer)})
        buffer.clear()

    for row in last_rows(cache, done):
        counts["runs"] += 1
        for item in segment_rows(row, definitions, release, model_id):
            buffer.append(item)
            counts["segments"] += 1
            counts["held_segments"] += item["split"] == "held"
            counts["output_tokens"] += item["segment"]["num_output_tokens"] or 0
            if len(buffer) >= ROWS_PER_SHARD:
                flush()
    flush()
    if not counts["runs"]:
        folder.rmdir()
        print(f"{cache}: no new runs since the last export", flush=True)
        return None
    child = {"schema": SCHEMA, "name": release, "source": source, "model_id": model_id, "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "counts": counts, "shards": shards, "complete": True}
    write_json(folder / "manifest.json", child)
    manifest["releases"].append({"name": release, "manifest": f"{release}/manifest.json", "source": source, "counts": counts})
    manifest["updated_utc"] = child["exported_utc"]
    write_json(manifest_path, manifest)                 # published after the release's payload and manifest
    print(json.dumps({"release": release, **counts}), flush=True)
    return child


def upload(collection: Path, prefix: str, bucket: str = BUCKET) -> dict:
    """Uploads every payload with an S3-checked SHA-256, then the release manifests, then the collection manifest last."""
    import base64
    import boto3
    from botocore.exceptions import ClientError
    client = boto3.client("s3")
    files = sorted(p for p in collection.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    ordered = [p for p in files if p.name != "manifest.json"] + [p for p in files if p.name == "manifest.json" and p.parent != collection] \
        + [collection / "manifest.json"]
    uploaded = []
    for path in ordered:
        key = f"{prefix.rstrip('/')}/{path.relative_to(collection)}"
        checksum = base64.b64encode(bytes.fromhex(sha256(path))).decode()
        try:
            head = client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
            if head.get("ChecksumSHA256") == checksum and path.name != "manifest.json":
                continue
        except ClientError:
            pass
        with path.open("rb") as body:
            client.put_object(Bucket=bucket, Key=key, Body=body, ChecksumSHA256=checksum)
        uploaded.append(key)
    receipt = {"bucket": bucket, "prefix": prefix, "uploaded": len(uploaded), "files": len(ordered), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps(receipt), flush=True)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, action="append", default=[], help="a rollouts.jsonl; repeat with --release")
    parser.add_argument("--release", action="append", default=[], help="release name per --cache")
    parser.add_argument("--tokenizer", required=True, help="the teacher's tokenizer (decodes the recorded prompts' tool block)")
    parser.add_argument("--model-id", required=True, help="the teacher model id recorded in provenance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--s3-prefix", default=None, help="upload the collection under this prefix after exporting")
    args = parser.parse_args(argv)
    if len(args.cache) != len(args.release):
        raise SystemExit("give one --release per --cache")
    for cache, release in zip(args.cache, args.release):
        export(cache, release, args.tokenizer, args.model_id, args.output)
    if args.s3_prefix:
        write_json(args.output / "s3_receipt.json", upload(args.output, args.s3_prefix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
