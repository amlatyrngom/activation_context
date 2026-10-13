"""
Fidelity check of exported teacher segments against the recorded tokens (MIXED_RUN_PLAN.md M1): for a seeded sample of
segments per benchmark, every text an exported message carries must appear verbatim in the tokens the teacher actually
read or sampled at that point: the segment's `prompt_token_ids` for its prompt messages, and each step's own `token_ids`
(the sampled turn, or the template wrapper holding a tool/user message) for the step's messages. Texts are compared after
NFC (the tokenizer's normalizer). A miss means the export carries text the teacher never saw (for example a full tool
output where the model read a truncated one) or dropped a rewrite; the report counts misses per benchmark with examples.

    python -m activation.bench.canonical_training.check_exported_segments --cache .../rollouts.jsonl \\
        --collection .../agent_teacher_regular --release teacher_regular_v2_20260927_s0 --tokenizer Qwen/Qwen3.8-27B-FP8 \\
        --per-benchmark 25 --output check.json
"""
from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from activation.bench.canonical_training.export_agent_teacher_segments import last_rows, segments


def texts(message: dict) -> list[str]:
    """The strings a message carries: its text content (string or text parts), reasoning, and tool-call names and string arguments."""
    out = []
    content = message.get("content")
    if isinstance(content, str):
        out.extend(piece for piece in content.split("</think>"))
    elif isinstance(content, list):
        out.extend(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    if message.get("reasoning_content"):
        out.append(message["reasoning_content"])
    for call in message.get("tool_calls") or []:
        function = call.get("function") or call
        out.append(function.get("name") or "")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"": arguments}
        out.extend(value for value in (arguments or {}).values() if isinstance(value, str))
    return [piece.replace("<think>", "").strip() for piece in out if piece and piece.replace("<think>", "").strip()]


def check_segment(tokenizer, exported: dict, segment: dict) -> dict:
    """Misses of `exported`'s message texts against the recorded tokens of `segment` (the rollout's run/compaction/subagent dict)."""
    def decoded(ids):
        return unicodedata.normalize("NFC", tokenizer.decode(ids, skip_special_tokens=False))
    messages, start = exported["messages"], exported["target_start"]
    windows = [(messages[:start], decoded(segment["prompt_token_ids"]), "prompt")]
    position = start
    for index, step in enumerate(segment["trajectory"]):
        count = len(step["messages"])
        windows.append((messages[position:position + count], decoded(step["token_ids"]), f"step_{index}:{step['role']}"))
        position += count
    misses, checked, chars = [], 0, 0
    if position != len(messages):
        misses.append({"where": "layout", "detail": f"{len(messages)} exported messages, {position} recorded"})
    for window, text, where in windows:
        for message in window:
            for piece in texts(message):
                checked += 1
                chars += len(piece)
                if unicodedata.normalize("NFC", piece) not in text:
                    misses.append({"where": where, "role": message.get("role"), "chars": len(piece), "recorded_chars": len(text),
                                   "text_head": piece[:160]})
    return {"checked": checked, "chars": chars, "misses": misses}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, required=True, help="the release's source rollouts.jsonl")
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--per-benchmark", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    child = json.loads((args.collection / args.release / "manifest.json").read_text())
    by_benchmark = defaultdict(list)
    for shard in child["shards"]:
        with (args.collection / args.release / shard["file"]).open() as handle:
            for line in handle:
                row = json.loads(line)
                by_benchmark[row["root"]["benchmark"]].append(row)
    rng = random.Random(args.seed)
    chosen = {}
    for benchmark, rows in sorted(by_benchmark.items(), key=lambda item: str(item[0])):
        for row in rng.sample(rows, min(args.per_benchmark, len(rows))):
            chosen.setdefault(row["root"]["config_key"], []).append(row)
    results, found = defaultdict(lambda: Counter()), set()
    examples = defaultdict(list)
    for run in last_rows(args.cache, set()):
        wanted = chosen.get(run["config_key"])
        if not wanted:
            continue
        found.add(run["config_key"])
        paths = {path: segment for path, _, segment in segments(run)}
        for exported in wanted:
            benchmark = exported["root"]["benchmark"]
            outcome = check_segment(tokenizer, exported, paths[exported["path"]])
            tally = results[benchmark]
            tally["segments"] += 1
            tally["texts"] += outcome["checked"]
            tally["chars"] += outcome["chars"]
            tally["texts_missing"] += len(outcome["misses"])
            tally["segments_with_miss"] += bool(outcome["misses"])
            if outcome["misses"] and len(examples[benchmark]) < 5:
                examples[benchmark].append({"segment_id": exported["segment_id"], "path": exported["path"], "misses": outcome["misses"][:3]})
    missing = sorted(set(chosen) - found)
    report = {"release": args.release, "cache": str(args.cache), "tokenizer": args.tokenizer, "per_benchmark": args.per_benchmark,
              "seed": args.seed, "runs_not_in_cache": len(missing),
              "by_benchmark": {str(k): dict(v) for k, v in sorted(results.items(), key=lambda item: str(item[0]))},
              "segments": sum(v["segments"] for v in results.values()),
              "segments_with_miss": sum(v["segments_with_miss"] for v in results.values()), "examples": dict(examples)}
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("release", "segments", "segments_with_miss", "runs_not_in_cache")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
