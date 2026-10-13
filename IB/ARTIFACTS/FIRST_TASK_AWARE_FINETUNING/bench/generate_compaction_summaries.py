"""Portable 27B compaction summaries of fixed pre-cut histories, with held100 published first."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import time

from run_qa import write

MODEL = "Qwen/Qwen3.5-27B"
STOP = False
SYSTEM = ("Write a handoff from the quoted conversation for the agent resuming the task. "
          "User messages define the task; tool results record observations. Assistant reasoning, proposed answers and search "
          "queries are unverified beliefs, even when confidently phrased. Mark facts as confirmed only when tool evidence "
          "establishes them; preserve contradictory evidence. DOs/DON'Ts contains only constraints from the original task.")


def digest(value):
    """The same canonical source-row identity used by run_trajectory_tasks."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summary_id(entry):
    return "summary:" + digest({"cut_id": entry["choice"]["id"], "row_sha256": entry["row_sha256"]})


def summary_history(row, entry):
    """Exactly the encoder's original frame and selected compaction prefix; no suffix or summary."""
    if row["id"] != entry["trajectory_id"] or digest(row) != entry["row_sha256"]:
        raise ValueError("Selected source payload changed")
    choice = entry["choice"]
    if choice not in row["compaction_cuts"]:
        raise ValueError("Selected compaction cut changed")
    messages = row.get("compaction_messages", row["messages"])
    first = row.get("compaction_initial_end", row["initial_end"])
    if not first <= choice["start"] < choice["end"] <= len(messages):
        raise ValueError("Invalid pre-cut history bounds")
    prefix = ([{"role": "system", "content": row["system_prompt"]}] if row["system_prompt"] else [])
    prefix += messages[:first] + messages[choice["start"]:choice["end"]]
    return {"messages": prefix, "tools": row.get("compaction_tools", row["tools"])}


def summary_chat(history, prompt):
    text = json.dumps(history, ensure_ascii=False)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
        "<conversation_evidence>\n" + text + "\n</conversation_evidence>\n\n" + prompt}]


def normalize_summary(text, prompt):
    """Canonical headings; harmless Markdown level/spacing differences do not waste a sample."""
    text = text.strip()
    if re.match(r"^```(?:markdown|md)?\s*\n", text) and text.endswith("```"):
        text = text[text.index("\n") + 1:text.rfind("```")].strip()
    headings = re.findall(r"^# .+$", prompt, re.M)
    canonical = {" ".join(line[2:].split()).casefold(): line for line in headings}
    lines = []
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        key = " ".join(match.group(1).split()).casefold() if match else None
        lines.append(canonical.get(key, line))
    return "\n".join(lines).strip()


def extract_summary_text(output, tokenizer, thinking_budget, *, thinking=False):
    """Use sampled IDs, since the engine's text can already have hidden reasoning delimiters."""
    ids = list(output.token_ids)
    end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    start_ids = tokenizer.encode("<think>", add_special_tokens=False)
    if not end_ids or not start_ids:
        raise ValueError("Tokenizer has no reasoning delimiters")
    def locate(sequence, marker):
        return next((i for i in range(len(sequence) - len(marker) + 1)
                     if sequence[i:i + len(marker)] == marker), -1)
    end = locate(ids, end_ids)
    if end < 0:
        if thinking or thinking_budget or locate(ids, start_ids) >= 0:
            raise ValueError("No generated reasoning end marker; summary is not safe to publish")
        return output.text, 0, output.output_token_count
    boundary = end + len(end_ids)
    if thinking_budget and boundary > thinking_budget + len(end_ids):
        raise ValueError("Generated reasoning exceeded its configured token budget")
    summary_ids = ids[boundary:]
    if locate(summary_ids, start_ids) >= 0 or locate(summary_ids, end_ids) >= 0:
        raise ValueError("Unexpected second reasoning section in summary")
    return tokenizer.decode(summary_ids, skip_special_tokens=True), boundary, len(summary_ids)


def validate_summary(text, finish_reason, output_tokens, maximum, prompt):
    text = normalize_summary(text, prompt)
    if finish_reason != "stop" or output_tokens > maximum:
        return "truncated or unfinished generation"
    if not text.strip():
        return "empty summary"
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return "invalid UTF-8"
    expected = re.findall(r"^# .+$", prompt, re.M)
    if len(expected) != 5 or re.findall(r"^# .+$", text.strip(), re.M) != expected:
        return "expected five exact headings in order"
    sections = re.split(r"^# .+$", text.strip(), flags=re.M)
    if sections[0].strip() or any(not part.strip() for part in sections[1:]):
        return "empty section or text before first heading"
    return None


def ordered_entries(selection, seed):
    """Original held panel, then a healthy mixed training prefix; retain every selected cut."""
    held = [("held", entry) for entry in selection["held"]]
    queues = defaultdict(list)
    train_roots = set()
    for entry in selection["train"]:
        if entry["root_id"] in train_roots:
            raise ValueError("Expected one selected training cut per root")
        train_roots.add(entry["root_id"])
        queues[entry["source"]].append(entry)
    held_roots = {entry["root_id"] for _, entry in held}
    if len(held_roots) != len(held) or train_roots & held_roots:
        raise ValueError("Original train/held roots are not disjoint")
    rng = random.Random(seed)
    for source in sorted(queues):
        rng.shuffle(queues[source])
    weights = {"coding": .5, "research": .3, "math": .2}
    train = []
    # Largest deficit gives well-mixed small prefixes, then naturally reweights exhausted sources.
    credit = Counter()
    while queues:
        mass = sum(weights.get(source, 1.) for source in queues)
        for source in queues:
            credit[source] += weights.get(source, 1.) / mass
        source = max(sorted(queues), key=lambda key: credit[key])
        credit[source] -= 1
        train.append(("train", queues[source].pop()))
        if not queues[source]:
            del queues[source]
    return held + train


def load_rows(folder, entries, offsets):
    """Lazily index line offsets once per immutable shard; keep only the current batch in RAM."""
    rows = {}
    by_shard = defaultdict(list)
    for entry in entries:
        by_shard[entry["shard"]].append(entry)
    for name, selected in by_shard.items():
        path = folder / name
        with path.open("rb") as handle:
            if name not in offsets:
                starts = []
                while True:
                    start = handle.tell()
                    if not handle.readline():
                        break
                    starts.append(start)
                offsets[name] = starts
            for entry in selected:
                handle.seek(offsets[name][entry["ordinal"]])
                row = json.loads(handle.readline())
                if row["id"] != entry["trajectory_id"]:
                    raise ValueError("Source shard or ordinal changed")
                rows[row["id"]] = row
    return rows


def publish_batch(folder, manifest, accepted, failures, cursor, timing):
    """Commit rows first and the cursor last. Unreferenced crash remnants are overwritten on resume."""
    number = len(manifest["shards"])
    name = f"summaries-{number:06d}.jsonl"
    temporary = folder / (name + ".tmp")
    with temporary.open("w") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(folder / name)
    manifest["shards"].append({"path": name, "summaries": len(accepted),
        "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest()})
    manifest["cursor"] = cursor
    manifest["pending"] = failures
    manifest["total_summaries"] += len(accepted)
    for key, field in [("sources", "source"), ("splits", "split")]:
        counts = Counter(manifest[key]); counts.update(row[field] for row in accepted); manifest[key] = dict(counts)
    manifest["updated_utc"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    manifest["last_batch_timing"] = dict(timing, shard=name, accepted=len(accepted), pending=len(failures))
    write(folder / "manifest.json", manifest)


def stop_requested(signum, frame):
    global STOP
    STOP = True


def main(args):
    from activation.common.ac_parts import COMPACTION_SUMMARY_PROMPT
    from activation.common.reporting import HtmlReporter
    from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig

    thinking_budget = getattr(args, "thinking_budget", 0)
    reasoning_effort = getattr(args, "reasoning_effort", None)
    if thinking_budget < 0:
        raise ValueError("Thinking budget must be nonnegative")
    if reasoning_effort and thinking_budget:
        raise ValueError("Choose native reasoning effort or a forced token budget, not both")
    thinking = bool(reasoning_effort) or thinking_budget > 0
    template_kwargs = {"enable_thinking": thinking}
    if reasoning_effort:
        template_kwargs["reasoning_effort"] = reasoning_effort
    # Qwen3.5's end-of-thinking delimiter is one token; reserve it separately from the summary budget.
    generation_budget = 8192 if reasoning_effort else args.max_output_tokens + (thinking_budget + 1 if thinking else 0)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.output / "PRESERVE").write_text("Portable oracle summaries; accepted rows remain reusable across producer/prompt changes.\n")
    if not (args.material / "PRESERVE").exists():
        (args.material / "PRESERVE").write_text("Original histories referenced by portable compaction summaries.\n")
    selection = json.loads(args.selection.read_text())
    if selection["task"] != "compaction":
        raise ValueError("Expected original compaction selection")
    order = ordered_entries(selection, args.seed)
    identity = digest([(split, summary_id(entry)) for split, entry in order])
    path = args.output / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {
        "schema_version": 1, "shards": [], "cursor": 0, "pending": [], "total_summaries": 0,
        "sources": {}, "splits": {}, "recipes": {}, "generation_batches": 0, "selection_identity": identity}
    if manifest["selection_identity"] != identity:
        raise ValueError("Selection/order changed; keep this portable bank and use a new producer bank for different cuts")
    target = min(args.target, len(order))
    prompt_sha = hashlib.sha256(COMPACTION_SUMMARY_PROMPT.encode()).hexdigest()
    recipe = {"model": getattr(args, "model_id", MODEL), "model_path": str(args.model_path), "model_revision": args.model_path.name,
        "reasoning": thinking, "thinking_token_budget": thinking_budget,
        "reasoning_effort": reasoning_effort,
        "reasoning_parser": "qwen3" if thinking else None, "max_generation_tokens": generation_budget,
        "reasoning_extraction": "sampled IDs after </think>; reasoning content never saved as a summary",
        "summary_prompt": COMPACTION_SUMMARY_PROMPT, "summary_prompt_sha256": prompt_sha,
        "system_prompt": SYSTEM, "temperature": args.temperature, "max_output_tokens": args.max_output_tokens,
        "selection_file_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "material": str(args.material.resolve()), "seed": args.seed, "source_view": "compaction prefix only; no continuation",
        "order": "original held first, then seeded within-source shuffle and weighted50/30/20 deficit mix; all original cuts retained"}
    recipe_id = digest(recipe)
    manifest["recipes"][recipe_id] = recipe
    manifest.update(target_requested=args.target, target_summaries=target, eligible_cuts=len(order),
                    held_required=len(selection["held"]), material=str(args.material.resolve()))
    seen = set()
    for shard in manifest["shards"]:
        with (args.output / shard["path"]).open() as handle:
            for line in handle:
                identifier = json.loads(line)["id"]
                if identifier in seen:
                    raise ValueError("Duplicate accepted summary id")
                seen.add(identifier)
    if len(seen) != manifest["total_summaries"]:
        raise ValueError("Summary manifest count disagrees with committed shards")
    write(path, manifest)
    reporter = HtmlReporter(str(args.output), "27B compaction summaries", "Fixed histories only; five-section handoffs saved separately from encoder inputs.",
                            page_rewrite_seconds=float("inf"))
    reporter.set_text("recipe", "Source and output", "Original held100 first, then mixed coding/research/math training histories. "
        "One summary per original cut. No future continuation, judge pass or source truncation. Accepted summaries remain portable.")
    reporter.set_status(phase="loading oracle", published_summaries=manifest["total_summaries"], target_summaries=target,
                        held=f"{manifest['splits'].get('held', 0)}/{len(selection['held'])}")
    reporter.render(force=True)
    held_goal = min(target, len(selection["held"]))
    if len(seen) >= target and manifest["splits"].get("held", 0) >= held_goal:
        reporter.finish()
        return 0
    started = time.monotonic()
    h = HarnessRuntime(HarnessRuntimeConfig(model_configs={"oracle": ModelConfig("oracle", str(args.model_path), engine_kwargs={
        "quantization": "fp8", "enable_lora": False, "max_model_len": args.context, "gpu_memory_utilization": args.memory_fraction,
        "max_num_seqs": args.microbatch, "speculative_config": None, "enable_prefix_caching": True,
        **({"reasoning_parser": "qwen3"} if thinking else {})})}))
    model = h.loaded_models["oracle"]
    if reasoning_effort and "reasoning_effort" not in (model.tokenizer.chat_template or ""):
        raise ValueError("This model template does not implement native reasoning effort")
    if len(model.tokenizer.encode("</think>", add_special_tokens=False)) != 1:
        raise ValueError("This Qwen3.5 producer expects a single end-of-thinking token")
    offsets, attempts = {}, Counter()
    reason, code = "target reached", 0
    while len(seen) < target or manifest["splits"].get("held", 0) < held_goal:
        if STOP or time.monotonic() - started > args.seconds - args.stop_margin:
            reason, code = "segment complete; resume committed bank", 75
            break
        pending = list(manifest["pending"])
        cursor = manifest["cursor"]
        retryable = [index for index in pending if attempts[index] < args.max_attempts]
        if len(seen) >= target:
            indices = [index for index in retryable if index < len(selection["held"])][:args.microbatch]
        else:
            budget = min(args.microbatch, target - len(seen))
            # At most one retry occupies a batch while untouched histories remain.
            indices = retryable[:min(budget, 1 if cursor < len(order) else budget)]
            while len(indices) < budget and cursor < len(order):
                index, cursor = cursor, cursor + 1
                if summary_id(order[index][1]) not in seen:
                    indices.append(index)
        if not indices:
            reason, code = "remaining cuts need format review; pending identities preserved", 2
            break
        entries = [order[index][1] for index in indices]
        rows = load_rows(args.material, entries, offsets)
        histories = [summary_history(rows[entry["trajectory_id"]], entry) for entry in entries]
        chats = [summary_chat(history, COMPACTION_SUMMARY_PROMPT) for history in histories]
        for chat in chats:
            tokenized = model.tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, **template_kwargs)
            count = len(tokenized["input_ids"] if hasattr(tokenized, "keys") else tokenized)
            if count + generation_budget > args.context:
                raise ValueError(f"Selected history needs {count + generation_budget} oracle tokens; raise context, never truncate")
        reporter.status.clear()
        reporter.set_status(phase="generating summaries", published_summaries=len(seen), target_summaries=target,
            held=f"{manifest['splits'].get('held', 0)}/{len(selection['held'])}", training=manifest["splits"].get("train", 0),
            pending_format_retries=len(pending), source_counts=json.dumps(manifest["sources"]))
        reporter.render(force=True)
        batch_started = time.monotonic()
        outputs = model.engine_chat_many(chats, chat_kwargs={"chat_template_kwargs": template_kwargs, "sampling_params": {
            "max_tokens": generation_budget, "temperature": args.temperature, "top_p": .8, "top_k": 20,
            "presence_penalty": 0., "seed": args.seed + manifest["generation_batches"],
            **({"thinking_token_budget": thinking_budget} if thinking_budget else {})}})
        if len(outputs) != len(entries):
            raise RuntimeError("Generator changed batch cardinality")
        manifest["generation_batches"] += 1
        attempted = set(indices)
        accepted, failures, errors = [], [index for index in pending if index not in attempted], []
        reasoning_cost, summary_cost = 0, 0
        for index, entry, history, output in zip(indices, entries, histories, outputs):
            attempts[index] += 1
            summary, reasoning_tokens, summary_completion_tokens = "", 0, 0
            try:
                summary, reasoning_tokens, summary_completion_tokens = extract_summary_text(output, model.tokenizer, thinking_budget, thinking=thinking)
                error = validate_summary(summary, output.finish_reason, summary_completion_tokens,
                                         args.max_output_tokens, COMPACTION_SUMMARY_PROMPT)
            except ValueError as exc:
                error = str(exc)
            reasoning_cost += reasoning_tokens
            summary_cost += summary_completion_tokens
            if error:
                failures.append(index)
                errors.append({"index": index, "id": summary_id(entry), "reason": error,
                    "output_tokens": output.output_token_count, "finish_reason": output.finish_reason, "text": summary})
                continue
            summary = normalize_summary(summary, COMPACTION_SUMMARY_PROMPT)
            row = {"id": summary_id(entry), "trajectory_id": entry["trajectory_id"], "root_id": entry["root_id"],
                "source": entry["source"], "split": order[index][0], "cut_id": entry["choice"]["id"],
                "entry": entry, "choice": entry["choice"], "row_sha256": entry["row_sha256"],
                "history_sha256": digest(history), "source_tokens": entry["choice"]["source_tokens"],
                "summary": summary, "recipe_id": recipe_id, "summary_prompt_sha256": prompt_sha,
                "prompt_tokens": output.prompt_token_count, "completion_tokens": output.output_token_count,
                "reasoning_completion_tokens": reasoning_tokens, "summary_completion_tokens": summary_completion_tokens,
                "summary_tokens": len(model.tokenizer.encode(summary, add_special_tokens=False)),
                "finish_reason": output.finish_reason, "selection_order": index}
            accepted.append(row)
            seen.add(row["id"])
        timing = {"recipe_id": recipe_id, "histories": len(entries), "seconds": round(time.monotonic() - batch_started, 3),
            "prompt_tokens": sum(o.prompt_token_count for o in outputs), "max_prompt_tokens": max(o.prompt_token_count for o in outputs),
            "cached_prompt_tokens": sum(o.cached_prompt_token_count for o in outputs),
            "completion_tokens": sum(o.output_token_count for o in outputs), "reasoning_completion_tokens": reasoning_cost,
            "summary_completion_tokens": summary_cost,
            "unclassified_rejected_tokens": sum(o.output_token_count for o in outputs) - reasoning_cost - summary_cost,
            "rejections": len(errors)}
        publish_batch(args.output, manifest, accepted, failures, cursor, timing)
        with (args.output / "batch_timings.jsonl").open("a") as handle:
            handle.write(json.dumps(timing) + "\n")
        if errors:
            with (args.output / "rejections.jsonl").open("a") as handle:
                for error in errors:
                    handle.write(json.dumps(error, ensure_ascii=False) + "\n")
        reporter.set_text("throughput", "Latest batch", f"{len(accepted)}/{len(entries)} summaries accepted in {timing['seconds']:.1f}s; "
            f"{timing['prompt_tokens']:,} prompt tokens and {timing['completion_tokens']:,} output tokens "
            f"({reasoning_cost:,} reasoning, {summary_cost:,} summary, {timing['unclassified_rejected_tokens']:,} unclassified rejected). "
            "The first batch includes engine startup. Format checks only; no judge pass.")
        print("COMPACTION_SUMMARIES", json.dumps({k: manifest[k] for k in ["total_summaries", "splits", "sources", "last_batch_timing"]}), flush=True)
    manifest["stop_reason"] = reason
    write(path, manifest)
    reporter.status.clear()
    reporter.set_status(phase=reason, published_summaries=len(seen), target_summaries=target,
        held=f"{manifest['splits'].get('held', 0)}/{len(selection['held'])}", training=manifest["splits"].get("train", 0))
    reporter.render(force=True)
    if not code:
        reporter.finish()
    return code


if __name__ == "__main__":
    signal.signal(signal.SIGUSR1, stop_requested); signal.signal(signal.SIGTERM, stop_requested)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL, help="Model identity recorded with portable targets")
    parser.add_argument("--target", type=int, default=20000, help="Total accepted summaries including held; capped at eligible cuts")
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--max-output-tokens", type=int, default=1536)
    parser.add_argument("--thinking-budget", type=int, default=0,
                        help="Optional bounded reasoning tokens, stripped before storing summary targets")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"),
                        help="Native model effort, without a forced reasoning cap; 8192 total output-token ceiling")
    parser.add_argument("--temperature", type=float, default=.2)
    parser.add_argument("--memory-fraction", type=float, default=.88)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=21000)
    parser.add_argument("--stop-margin", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260924)
    raise SystemExit(main(parser.parse_args()))
