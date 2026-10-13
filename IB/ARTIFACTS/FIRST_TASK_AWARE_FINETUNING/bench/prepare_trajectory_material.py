"""Question-independent trajectory windows and complete compaction cuts, in portable JSONL.

Run on a compute node. Sources use the project's row normalization and ordinary HF
cache. Published shards are immutable; manifest.json is the atomic commit point.
QA uses original recorded messages. Compaction alone may use the existing public
assistant continuation conversion; optional compaction_* fields identify that view.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import time

SOURCES = {
    "coding": ("nvidia/Open-SWE-Traces", "v1.2", "minisweagent", 5),
    "research": ("ScienceOne-AI/S1-DeepResearch-15k", None, "train", 3),
    "math": ("nvidia/Nemotron-SFT-Math-v4", None, "train", 2),
}
STOP = False
WORKER_TOKENIZER = None
WORKER_ARGS = None


def digest(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def dump_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def iter_rows(source, fixture_dir=None):
    if fixture_dir:
        names = {"coding": "open_swe_rows.json", "research": "s1_rows.json", "math": "nemotron_tir_rows.json"}
        data = json.loads((fixture_dir / names[source]).read_text())
        # Dataset-viewer fixtures explicitly say when an individual cell is cut.
        for entry in data["rows"]:
            if not entry.get("truncated_cells"):
                yield entry["row"]
        return
    from datasets import load_dataset
    dataset, config, split, _ = SOURCES[source]
    kwargs = {"split": split, "streaming": True}
    if source == "math":
        from activation.dataset.loaders.nemotron_math import SUBSET_SHARDS, NUM_SHARDS
        kwargs["data_files"] = [f"data/train-{i:05d}-of-{NUM_SHARDS:05d}.parquet" for i in SUBSET_SHARDS["tir"]]
    yield from load_dataset(dataset, config, **kwargs)


def math_exclusion_windows(fixture_dir=None):
    """The existing Nemotron loader's AIME 2024/2025 13-word exclusion, loaded once."""
    from activation.dataset.loaders.trajectory_utils import ngram_windows
    from activation.dataset.loaders.nemotron_math import DECONTAMINATION_NGRAM
    if fixture_dir:
        problems = [entry["row"]["problem"] for year in (24, 25)
                    for entry in json.loads((fixture_dir / f"aime{year}_rows.json").read_text())["rows"]]
    else:
        from activation.dataset.loaders.aime import aime_problem_statements
        problems = aime_problem_statements()
    return ngram_windows(problems, DECONTAMINATION_NGRAM)


def excluded_math_problem(problem, excluded_windows):
    from activation.dataset.loaders.trajectory_utils import ngram_windows
    from activation.dataset.loaders.nemotron_math import DECONTAMINATION_NGRAM
    return bool(ngram_windows([str(problem)], DECONTAMINATION_NGRAM) & excluded_windows)


def normalize_row(source, row, ordinal):
    from activation.dataset.loaders.trajectory_utils import reformat_trajectory
    if source == "coding":
        from activation.dataset.loaders.open_swe_traces import normalize_open_swe_row, SYSTEM_PROMPT
        normalized, tools = normalize_open_swe_row(row)
        identity = str(row.get("trajectory_id") or row["instance_id"])
        root = str(row.get("instance_id") or identity)
    elif source == "research":
        from activation.dataset.loaders.s1_deep_research import normalize_s1_row, SYSTEM_PROMPT
        meta = row.get("meta") or {}
        if meta.get("language") != "en":
            return None, "non_english"
        normalized, tools = normalize_s1_row(row)
        identity = str(meta.get("id") or ordinal)
        root = meta.get("question") or next((m["content"] for m in normalized if m["role"] == "user"), identity)
        root = digest(" ".join(str(root).split()))
    else:
        from activation.dataset.loaders.nemotron_math import normalize_nemotron_row, SYSTEM_PROMPT
        if row.get("subset") != "tir":
            return None, "not_tir"
        normalized, tools = normalize_nemotron_row(row)
        identity = str(row["uuid"])
        root = digest(" ".join(str(row["problem"]).split()))
    original_system = "\n\n".join(m.get("content") or "" for m in normalized if m["role"] == "system")
    messages, kwargs = reformat_trajectory(normalized, tools)
    # Never pass source gold, expected_answer, reference patches or success labels.
    dataset, config, split, _ = SOURCES[source]
    return {"schema_version": 1, "id": f"{source}:{identity}", "source": source,
            "root_id": f"{source}:{root}", "system_prompt": original_system or SYSTEM_PROMPT,
            "tools": kwargs.get("tools") or [], "messages": messages,
            "source_meta": {"dataset": dataset, "config": config, "split": split, "source_row_id": identity}}, None


def complete_history(messages):
    """Keep complete turns; explicitly mark an unusable terminal call rather than fabricating its reply."""
    from activation.ac_model.ac_model_utils import complete_public_turns
    first = next((i for i, m in enumerate(messages) if m.get("role") == "assistant"), len(messages))
    if first == len(messages) or not any(m.get("role") == "user" for m in messages[:first]):
        raise ValueError("missing_original_task_or_assistant")
    try:
        groups = complete_public_turns(messages[first:])
        omitted = 0
    except ValueError:
        last = max(i for i, m in enumerate(messages) if m.get("role") == "assistant")
        if last == first:
            raise
        groups = complete_public_turns(messages[first:last])
        omitted = len(messages) - last
    # complete_public_turns also drops final input with no subsequent assistant.
    kept = deepcopy(messages[:first]) + [deepcopy(m) for group in groups for m in group]
    return kept, first, groups, omitted + max(0, len(messages) - omitted - len(kept))


def views_for(row, tokenizer, args, *, compaction=False):
    from activation.ac_model.ac_model_utils import assistant_target_ids, prepare_training_history, tokenize_with_parts
    messages = row.get("compaction_messages", row["messages"]) if compaction else row["messages"]
    tools = row.get("compaction_tools", row["tools"]) if compaction else row["tools"]
    first = row.get("compaction_initial_end", row["initial_end"]) if compaction else row["initial_end"]
    boundaries = [i for i in range(first, len(messages)) if messages[i].get("role") == "assistant"] + [len(messages)]
    frame = [{"role": "system", "content": row["system_prompt"]}] + messages[:first]
    cache, assistant_targets = {}, {}
    def count(end):
        if end not in cache:
            ids, _ = tokenize_with_parts(tokenizer, frame + messages[first:end], [], tokenizer.pad_token_id or 0,
                                        tools=tools, chat_template_kwargs={"enable_thinking": True})
            cache[end] = len(ids)
        return cache[end]
    windows, cuts, selected = [], [], set()
    # Prefix selection is independent of both future questions and their evidence.
    # A source shorter than a bucket appears only once, with its actual length.
    for budget in args.lengths:
        options = boundaries[1:-1] if compaction else boundaries[1:]
        low, high, best = 0, len(options) - 1, None
        while low <= high:
            middle = (low + high) // 2
            if count(options[middle]) <= budget:
                best, low = options[middle], middle + 1
            else:
                high = middle - 1
        if best is None or best in selected or count(best) < args.min_tokens:
            continue
        selected.add(best)
        window_id = f"{row['id']}:{'compaction' if compaction else 'qa'}:{first}:{best}"
        window = {"id": window_id, "start": first, "end": best, "target_tokens": budget, "source_tokens": count(best)}
        if not compaction:
            windows.append(window)
            continue
        suffix_ends = [end for end in boundaries if end > best]
        assistant_positions = [i for i in range(best, len(messages)) if messages[i].get("role") == "assistant"]
        for i in assistant_positions:
            if i not in assistant_targets:
                assistant_targets[i] = assistant_target_ids(tokenizer, messages[i], tools, {"enable_thinking": True})
        target_ids = [assistant_targets[i] for i in assistant_positions]
        prefix = frame + messages[first:best]
        measured = {}
        def measure(end):
            if end not in measured:
                targets = target_ids[:sum(i < end for i in assistant_positions)]
                ids, _, positions = prepare_training_history(tokenizer, prefix + messages[best:end], len(prefix), targets, [],
                                    tools=tools, chat_template_kwargs={"enable_thinking": True})
                measured[end] = (len(ids) - positions[0] - 1, len(ids), sum(map(len, targets)), len(targets))
            return measured[end]
        low, high, tail = 0, len(suffix_ends) - 1, None
        while low <= high:
            middle = (low + high) // 2
            length, total, _, _ = measure(suffix_ends[middle])
            if length <= args.max_suffix_tokens and total <= args.max_total_tokens:
                tail, low = suffix_ends[middle], middle + 1
            else:
                high = middle - 1
        if tail is None:
            continue
        length, total, targets, turns = measure(tail)
        cuts.append(window | {"window_id": window_id, "continuation_end": tail, "continuation_tokens": length,
                              "assistant_target_tokens": targets, "assistant_turns": turns, "full_history_tokens": total})
    return windows, cuts


def prepare_row(source, raw, ordinal, tokenizer, args):
    from activation.ac_model.ac_model_utils import normalize_public_assistants
    if source == "math" and excluded_math_problem(raw.get("problem", ""), args._excluded_math_windows):
        return None, "aime_2024_2025_overlap"
    row, reason = normalize_row(source, raw, ordinal)
    if row is None:
        return None, reason
    row["split"] = "held" if int(digest(f"{args.seed}:{row['root_id']}")[:12], 16) % args.held_every == 0 else "train"
    try:
        messages, first, _, omitted = complete_history(row["messages"])
    except ValueError:
        return None, "incomplete_history"
    row.update(messages=messages, initial_end=first, omitted_incomplete_tail_messages=omitted)
    row["qa_windows"], _ = views_for(row, tokenizer, args)
    try:
        compact, tools = normalize_public_assistants(tokenizer, messages, row["tools"], args.max_assistant_tokens)
        compact, first, _, _ = complete_history(compact)
        original_calls = {call.get("id") for message in messages for call in message.get("tool_calls") or []}
        row["compaction_synthetic_continuation_turns"] = sum(
            str(call.get("id", "")).startswith("ac_continue_") and call.get("id") not in original_calls
            for message in compact for call in message.get("tool_calls") or [])
        if compact != messages or (tools or []) != row["tools"]:
            row.update(compaction_messages=compact, compaction_tools=tools or [], compaction_initial_end=first)
        _, row["compaction_cuts"] = views_for(row, tokenizer, args, compaction=True)
    except ValueError as error:
        row["compaction_cuts"] = []
        row["compaction_ineligible"] = str(error)
    if not row["qa_windows"] and not row["compaction_cuts"]:
        return None, "no_window_or_complete_continuation"
    return row, None


def prepare_timed(source, raw, ordinal, tokenizer, args):
    started = time.monotonic()
    row, reason = prepare_row(source, raw, ordinal, tokenizer, args)
    if row is not None:
        row["preparation_seconds"] = round(time.monotonic() - started, 3)
    return row, reason


def initialize_worker(args):
    global WORKER_TOKENIZER, WORKER_ARGS
    from transformers import AutoTokenizer
    WORKER_TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=bool(args.fixture_dir))
    WORKER_ARGS = args


def worker_prepare(source, raw, ordinal):
    return prepare_timed(source, raw, ordinal, WORKER_TOKENIZER, WORKER_ARGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000, help="Total accepted trajectories, including earlier committed releases")
    parser.add_argument("--release-size", type=int, default=100)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32768])
    parser.add_argument("--min-tokens", type=int, default=1024)
    parser.add_argument("--max-suffix-tokens", type=int, default=8192)
    parser.add_argument("--max-total-tokens", type=int, default=72000)
    parser.add_argument("--max-assistant-tokens", type=int, default=1024)
    parser.add_argument("--held-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--fixture-dir", type=Path, help="Local dataset-viewer fixtures; no source downloads")
    parser.add_argument("--seconds", type=int, default=10200)
    parser.add_argument("--workers", type=int, default=1, help="CPU processes; changing this never changes material selection")
    args = parser.parse_args()
    if min(args.limit, args.release_size, args.min_tokens, args.max_assistant_tokens, args.workers) < 1 or args.held_every < 2:
        parser.error("Counts must be positive and held-every at least two")
    args.lengths = sorted(set(args.lengths))
    args.output.mkdir(parents=True, exist_ok=True)
    # A second producer would otherwise replay the same cursors and overwrite a release.
    import fcntl
    lock = (args.output / ".producer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGUSR1, stop)
    recipe = {k: v for k, v in vars(args).items() if k not in {"output", "limit", "release_size", "seconds", "fixture_dir", "workers"}}
    recipe["fixture_dir"] = str(args.fixture_dir) if args.fixture_dir else None
    recipe["math_exclusion"] = "AIME_2024_2025_13_words_v1"
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": 1, "recipe": recipe, "shards": [], "cursors": {s: 0 for s in SOURCES},
        "sources": {}, "splits": {}, "skipped": {}, "trajectories": 0, "qa_windows": 0,
        "compaction_examples": 0, "assistant_target_tokens": 0, "exhausted_sources": [],
    }
    if manifest["recipe"] != recipe:
        raise ValueError("Changed preparation recipe: select a new output directory; existing JSONL remains reusable")
    args._excluded_math_windows = math_exclusion_windows(args.fixture_dir)
    sources, splits, skipped = (Counter(manifest[key]) for key in ("sources", "splits", "skipped"))
    seen, roots = set(), set()
    for shard in manifest["shards"]:
        for line in (args.output / shard["index"]).read_text().splitlines():
            entry = json.loads(line)
            seen.add(entry["id"])
            roots.add(entry["root_id"])
    tokenizer, pool = None, None
    if args.workers == 1:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=bool(args.fixture_dir))
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=initialize_worker, initargs=(args,))
    streams = {}
    exhausted = set(manifest["exhausted_sources"])
    source_finished = set(exhausted)
    prefetched = {source: deque() for source in SOURCES}
    read_cursors = dict(manifest["cursors"])
    def fill(source):
        # Per-source queues preserve the exact serial consumption order and successful
        # source mixture. Fetching ahead never advances the published/resume cursor.
        if source not in streams:
            streams[source] = itertools.islice(iter_rows(source, args.fixture_dir), read_cursors[source], None)
        count = max(1, min(args.release_size, math.ceil(args.workers * SOURCES[source][3] / 10)))
        while len(prefetched[source]) < count and source not in source_finished:
            try:
                raw = next(streams[source])
            except StopIteration:
                source_finished.add(source)
                break
            ordinal = read_cursors[source]
            read_cursors[source] += 1
            prefetched[source].append((ordinal, pool.submit(worker_prepare, source, raw, ordinal)))
    pending = []
    started = time.monotonic()
    def publish():
        if pending:
            number = len(manifest["shards"])
            name = f"trajectories-{number:05d}.jsonl"
            index_name = f"trajectories-{number:05d}.index.jsonl"
            for filename, values in [(name, pending), (index_name, [
                    {"id": r["id"], "root_id": r["root_id"], "source": r["source"], "split": r["split"],
                     "qa_windows": len(r["qa_windows"]), "compaction_examples": len(r["compaction_cuts"])} for r in pending])]:
                temp = args.output / (filename + ".tmp")
                with temp.open("w") as handle:
                    for value in values:
                        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                temp.replace(args.output / filename)
            manifest["shards"].append({"path": name, "index": index_name, "trajectories": len(pending)})
            pending.clear()
        phase = "complete" if manifest["trajectories"] >= args.limit else "source_exhausted" if len(exhausted) == len(SOURCES) else "preparing"
        manifest.update(sources=dict(sources), splits=dict(splits), skipped=dict(skipped), distinct_roots=len(roots),
                        exhausted_sources=sorted(exhausted), updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        requested_trajectories=args.limit, phase=phase, preparation_workers=args.workers)
        dump_atomic(manifest_path, manifest)
        print(json.dumps({k: manifest[k] for k in ["phase", "trajectories", "distinct_roots", "qa_windows", "compaction_examples", "assistant_target_tokens", "sources", "skipped"]}), flush=True)
    try:
        while manifest["trajectories"] < args.limit and not STOP and time.monotonic() - started < args.seconds:
            if pool is not None:
                for source in SOURCES:
                    if source not in exhausted:
                        fill(source)
            available = [source for source in SOURCES if source not in exhausted]
            if not available:
                break
            source = min(available, key=lambda key: sources[key] / SOURCES[key][3])
            if pool is None:
                if source not in streams:
                    streams[source] = itertools.islice(iter_rows(source, args.fixture_dir), manifest["cursors"][source], None)
                try:
                    raw = next(streams[source])
                except StopIteration:
                    exhausted.add(source)
                    continue
                ordinal = manifest["cursors"][source]
                row, reason = prepare_timed(source, raw, ordinal, tokenizer, args)
            else:
                if not prefetched[source]:
                    exhausted.add(source)
                    continue
                ordinal, future = prefetched[source].popleft()
                row, reason = future.result()
            assert ordinal == manifest["cursors"][source]
            manifest["cursors"][source] = ordinal + 1
            if row is None:
                skipped[f"{source}:{reason}"] += 1
                continue
            if row["id"] in seen:
                skipped[f"{source}:duplicate_source_id"] += 1
                continue
            row["material_ordinal"] = manifest["trajectories"]
            seen.add(row["id"])
            roots.add(row["root_id"])
            pending.append(row)
            sources[source] += 1
            splits[row["split"]] += 1
            manifest["trajectories"] += 1
            manifest["qa_windows"] += len(row["qa_windows"])
            manifest["compaction_examples"] += len(row["compaction_cuts"])
            manifest["assistant_target_tokens"] += sum(c["assistant_target_tokens"] for c in row["compaction_cuts"])
            if len(pending) >= args.release_size:
                publish()
    finally:
        publish()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    if manifest["trajectories"] < args.limit and len(exhausted) < len(SOURCES):
        raise SystemExit(75)


if __name__ == "__main__":
    main()
