"""
Stage 1 of the continuation training data: a `full` resume cache -> material.jsonl.

The resumption benchmark's `full` variant (run_resume.py --variants full) lets the untuned reader continue a prefix another
model wrote, with the whole prefix in view. This script turns each usable run of such a cache into one self-contained line:
the task head, the prefix, and the continuation the reader wrote, with the reader's own sampled tokens as training targets.
Stage 2 (prepare_agent_continuation.py) builds trainer packets from these lines.

This stage imports nothing from the research tree (standard library + the reader's tokenizer from `transformers`), so it
runs wherever the cache is and its output does not depend on any version of the tree.

    python export_continuation_material.py --rollouts ROLLOUTS/pc_resume_v1_full/rollouts.jsonl \
        --notes PAPER_COMPACTION/pc_resume_v1_full/notes --prefixes prefixes_train_v1 --out material.jsonl

How a run's recorded prompt is split. In a `full` run the recorded prompt (`prompt_messages`) is head + prefix and the
recorded trajectory is the continuation alone (resume_program.ResumeFromPrefix.install). The head is every message before
the first assistant message: the system message and the first user message(s). The prefix is the rest of the prompt. The
split is checked at token level: the head, rendered by the reader's chat template with the tool definitions and an open
generation prompt, must be exactly the first tokens of the recorded prompt, and their number must equal the run note's
`head_tokens` when there is a note. A run that fails this check is left out and counted.

Tool definitions come from the run note's `tools` when it has them; otherwise they are read back from the <tools> block of
the recorded prompt tokens (older runs). Either way the token check above confirms them.

Chat template settings: the run's recorded `call_kwargs.chat_template_kwargs`, reduced to the names the reader's template
actually reads (the 9B's template reads `enable_thinking` and ignores `reasoning_effort`); the full recorded settings are
kept beside them. The token check above confirms the choice.

One line of material.jsonl:
    task_id, config_key, benchmark, repo, language, side, official_test       where the run comes from
    solved, score, finish_reason, answer                                      how the reader's continuation ended
    head_messages         system + first user message(s), as recorded
    prefix_messages       the prefix (assistant / tool / user messages), as recorded
    continuation_messages the reader's continuation (assistant / tool / user messages after the cut, in order)
    target_token_ids      one list per continuation assistant message: the tokens the reader sampled for it
    tools                 tool definitions as the reader's chat template takes them;  tools_source: note | prompt_tokens
    chat_template_kwargs, chat_template_kwargs_recorded
    prefix_tokens         reader tokens of the prefix (prompt tokens - head tokens): the length sizes are fractions of
    tokens                {head, prefix, prompt, continuation_written, continuation_context, continuation_turns}
    teacher               the prefix writer's own outcome on the task, when --prefixes is given
    reader_tokenizer, source

Left out (each counted): runs that are not variant `full`; finish_reason error / task_unavailable; runs with compactions,
subagents, injected input, a reader adapter or an AC model (the target must be the untuned reader's plain continuation);
unsolved runs (--solved-only, the default); fewer than --min-target-tokens target tokens. When a cache holds several rows
for one run, the last one wins.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

DROPPED_FINISH = ("error", "task_unavailable")


def tools_from_prompt(tokenizer, prompt_token_ids: list[int]) -> list[dict]:
    """The tool definitions the reader saw: the <tools> block of its recorded prompt, one JSON definition per line."""
    for limit in (8192, None):
        text = tokenizer.decode(prompt_token_ids[:limit] if limit else prompt_token_ids)
        start, end = text.find("<tools>"), text.find("</tools>")
        if start >= 0 and end > start:
            return [json.loads(line) for line in text[start + len("<tools>"):end].strip().splitlines() if line.strip()]
    return []


def head_ids(tokenizer, head: list[dict], tools: list[dict], kwargs: dict) -> list[int]:
    """The head as the reader's first prompt: chat template, tool definitions, generation prompt open."""
    text = tokenizer.apply_chat_template(head, tools=tools or None, add_generation_prompt=True, tokenize=False, **kwargs)
    return tokenizer.encode(text, add_special_tokens=False)


def material_of(row: dict, tokenizer, note: dict | None, teacher: dict | None, args) -> tuple[dict | None, str]:
    """(material line, "") for a usable run, (None, why) otherwise."""
    config = row["agent_config"]
    metadata = config.get("metadata") or {}
    task = config.get("dataset_task") or {}
    if metadata.get("variant") != "full":
        return None, f"variant {metadata.get('variant')!r}, not full"
    if row.get("finish_reason") in DROPPED_FINISH:
        return None, f"finish_reason {row['finish_reason']}"
    if row.get("compactions") or row.get("subagent_results") or row.get("injected_input"):
        return None, "compaction, subagent or injected input in the run"
    if row.get("lora_name") or row.get("ac_model_name") or config.get("lora_name") or config.get("ac_model_name"):
        return None, "reader ran with an adapter or an AC model"
    solved = bool(row.get("score") and row["score"] >= 1.0)
    if args.solved_only and not solved:
        return None, "not solved"

    prompt = row["prompt_messages"]
    first = next((index for index, message in enumerate(prompt) if message.get("role") == "assistant"), None)
    if first is None:
        return None, "no prefix in the recorded prompt"
    head, prefix = prompt[:first], prompt[first:]
    if not head or any(message.get("role") not in ("system", "user") for message in head):
        return None, "unexpected head messages"

    continuation, targets, context_tokens = [], [], 0
    for step in row["trajectory"]:
        messages = step.get("messages") or []
        assistants = [message for message in messages if message.get("role") == "assistant"]
        if step.get("role") == "assistant":
            if len(assistants) != 1 or len(messages) != 1 or not step.get("token_ids"):
                return None, "an assistant step without exactly one message and its sampled tokens"
            targets.append(list(step["token_ids"]))
        elif assistants:
            return None, "an assistant message inside a tool or user step"
        else:
            context_tokens += len(step.get("token_ids") or [])
        continuation.extend(messages)
    written = sum(map(len, targets))
    if written < max(1, args.min_target_tokens):
        return None, f"fewer than {max(1, args.min_target_tokens)} target tokens"

    recorded_kwargs = dict((config.get("call_kwargs") or {}).get("chat_template_kwargs") or {})
    read_by_template = {key: value for key, value in recorded_kwargs.items() if key in (tokenizer.chat_template or "")}
    candidates = [read_by_template] if "enable_thinking" in read_by_template else [dict(read_by_template, enable_thinking=flag) for flag in (True, False)]
    if note and note.get("tools"):
        tools, tools_source = note["tools"], "note"
    else:
        tools, tools_source = tools_from_prompt(tokenizer, row["prompt_token_ids"]), "prompt_tokens"
    prompt_ids = row["prompt_token_ids"]
    kwargs, head_tokens = None, None
    for candidate in candidates:
        ids = head_ids(tokenizer, head, tools, candidate)
        if prompt_ids[:len(ids)] == ids:
            kwargs, head_tokens = candidate, len(ids)
            break
    if kwargs is None:
        return None, "the head does not render to the first tokens of the recorded prompt"
    if note and note.get("head_tokens") is not None and note["head_tokens"] != head_tokens:
        return None, "head tokens differ from the run note"
    if note and note.get("prompt_tokens") is not None and note["prompt_tokens"] != len(prompt_ids):
        return None, "prompt tokens differ from the run note"

    task_id = task.get("task_id") if isinstance(task, dict) else None
    return {
        "task_id": task_id, "config_key": row.get("config_key"), "benchmark": metadata.get("benchmark"), "repo": metadata.get("repo"),
        "language": metadata.get("language"), "side": metadata.get("side"), "official_test": metadata.get("official_test"),
        "solved": solved, "score": row.get("score"), "finish_reason": row.get("finish_reason"), "answer": row.get("answer"),
        "head_messages": head, "prefix_messages": prefix, "continuation_messages": continuation, "target_token_ids": targets,
        "tools": tools, "tools_source": tools_source,
        "chat_template_kwargs": kwargs, "chat_template_kwargs_recorded": recorded_kwargs,
        "prefix_tokens": len(prompt_ids) - head_tokens,
        "tokens": {"head": head_tokens, "prefix": len(prompt_ids) - head_tokens, "prompt": len(prompt_ids), "continuation_written": written,
                   "continuation_context": context_tokens, "continuation_turns": len(targets)},
        "teacher": teacher, "reader_tokenizer": args.reader_tokenizer,
        "source": {"rollouts": str(args.rollouts), "note": note is not None, "seed": row.get("seed")},
    }, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", required=True, help="rollouts.jsonl of a `full` resume cache")
    parser.add_argument("--notes", default=None, help="folder of run notes (<task_id>.json) written by the resume program; optional")
    parser.add_argument("--prefixes", default=None, help="prefix folder (cut_prefixes.py) whose index adds the prefix writer's outcome; optional")
    parser.add_argument("--out", required=True, help="material.jsonl to write (a summary goes beside it)")
    parser.add_argument("--reader-tokenizer", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--solved-only", action=argparse.BooleanOptionalAction, default=True, help="keep only runs the reader solved (score >= 1)")
    parser.add_argument("--min-target-tokens", type=int, default=1, help="leave out runs whose continuation holds fewer sampled tokens")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.reader_tokenizer)
    notes_folder = Path(args.notes) if args.notes else None
    teachers = {}
    if args.prefixes:
        for entry in json.loads((Path(args.prefixes) / "index.json").read_text())["prefixes"]:
            teachers[entry["task_id"]] = {**entry.get("teacher", {}), "prefix_tokens": entry.get("prefix_tokens"), "head_tokens": entry.get("head_tokens")}

    latest: dict[str, tuple[dict | None, str]] = {}                                # per run: the last row of the cache wins
    with open(args.rollouts) as handle:
        for line in handle:                                                        # streamed: one run per line
            if not line.strip():
                continue
            row = json.loads(line)
            task = row["agent_config"].get("dataset_task") or {}
            task_id = task.get("task_id") if isinstance(task, dict) else None
            note = None
            if notes_folder is not None and task_id:
                path = notes_folder / f"{task_id}.json".replace("/", "__")
                note = json.loads(path.read_text()) if path.exists() else None
            latest[row.get("config_key") or task_id] = material_of(row, tokenizer, note, teachers.get(task_id), args)

    kept = sorted((record for record, _ in latest.values() if record is not None), key=lambda record: (record["benchmark"] or "", record["task_id"] or ""))
    why = Counter(reason for record, reason in latest.values() if record is None)
    if len({record["task_id"] for record in kept}) != len(kept):
        raise ValueError("two usable runs share a task id: one cache per variant and seed is expected")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def mean(name: str) -> float | None:
        return round(sum(record["tokens"][name] for record in kept) / len(kept), 1) if kept else None

    summary = {
        "rollouts": str(args.rollouts), "runs_read": len(latest), "usable": len(kept), "left_out": dict(why),
        "solved_only": args.solved_only, "min_target_tokens": args.min_target_tokens,
        "solved": sum(record["solved"] for record in kept), "repositories": len({record["repo"] for record in kept}),
        "by_language": dict(Counter(record["language"] for record in kept)), "by_side": dict(Counter(record["side"] for record in kept)),
        "with_note": sum(record["source"]["note"] for record in kept), "tools_source": dict(Counter(record["tools_source"] for record in kept)),
        "chat_template_kwargs": sorted({json.dumps(record["chat_template_kwargs"], sort_keys=True) for record in kept}),
        "mean_tokens": {name: mean(name) for name in ("head", "prefix", "continuation_written", "continuation_context", "continuation_turns")},
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
