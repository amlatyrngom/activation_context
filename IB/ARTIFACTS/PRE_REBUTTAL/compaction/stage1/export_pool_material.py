"""
Stage 1 of the continuation training data from a POOL cache (run_pool.py): one pool run -> one material.jsonl line in the
format export_continuation_material.py writes, so prepare_agent_continuation.py builds packets from it unchanged.

A pool run is the 9B solving a task from the start. It is cut here, and the 9B's own tail after the cut is the target: the
continuation the 9B wrote with the whole prefix in view, which is what a `full` resume run gives, without the resume pass.
The prefix is the 9B's own (the evaluation's 9B-written prefixes are cut the same way from the same kind of pool).

Where a run is cut: the first step boundary that is not an assistant turn at or past min(--lower, f x run length), the run
length counting the head tokens as cut_prefixes.cut_row does. f is drawn per task, uniform in --fraction-range, from a
generator seeded by (--seed, task id). Long runs are thus cut at 20,000 tokens as the evaluation's window rule, short ones
at a share of their length as the fraction scan (user, Oct 14 ≈20:40Z). With --window-min-total T, every run of at least T tokens is
cut at --lower (closer to the evaluation's prefixes, median ≈20k; H5's material had a median prefix of 7.8k).

Checks and outputs as export_continuation_material.py: the head (system + first user message) rendered by the reader's chat
template with the tool definitions and an open generation prompt must be exactly the run's recorded prompt tokens.
prefix_tokens are the template's tokens of head + prefix (generation prompt open) minus the head's. `teacher` holds the
run's own outcome and the cut.

    python export_pool_material.py --rollouts ROLLOUTS/pc_pool9_eval_v3/rollouts.jsonl --out material_pool9_v3.jsonl \
        [--exclude-tasks prefixes_eval9c_200/index.json prefixes_dev9_75/index.json prefixes_eval_100/index.json]

Stage 1 of the clean compaction recipe (2026-11-15) adds: --upper (runs of at least this many tokens are left out, the
evaluation's 50,000 rule) and one line per (task, pool seed, cut): `run_key` = "<task id>#s<seed>#c<lower>", so the
same task from another seed's cache (a new caching id) or cut at another --lower is a separate line. Each --rollouts file
is one pool cache; a run seen twice under the same key (a rewritten cache line, or a cache listed twice) keeps its last line.

2026-11-17: each line also holds `prefix_assistant_token_ids`, the recorded sampled tokens of every prefix assistant turn (in order):
the packet builder gives them to the teacher view (info["in_context_history_ids"]) so the full-context teacher reads the prompt the
agent read, with real "</think>" delimiters (TEACHER_VIEW_FIX.md).

Left out (each counted): error / task_unavailable / trajectory_cap runs; compactions, subagents, injected input, an adapter
or an AC model; unsolved runs (--solved-only, the default); runs under --min-total tokens; nothing to predict after the cut;
fewer than --min-target-tokens target tokens; --exclude-tasks tasks (the packet build excludes them again).
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from export_continuation_material import DROPPED_FINISH, head_ids, tools_from_prompt


def material_of(row: dict, tokenizer, args) -> tuple[dict | None, str]:
    config = row["agent_config"]
    metadata = config.get("metadata") or {}
    task = config.get("dataset_task") or {}
    task_id = task.get("task_id") if isinstance(task, dict) else None
    if row.get("finish_reason") in DROPPED_FINISH + ("trajectory_cap",):
        return None, f"finish_reason {row['finish_reason']}"
    if row.get("compactions") or row.get("subagent_results") or row.get("injected_input"):
        return None, "compaction, subagent or injected input in the run"
    if row.get("lora_name") or row.get("ac_model_name") or config.get("lora_name") or config.get("ac_model_name"):
        return None, "ran with an adapter or an AC model"
    solved = bool(row.get("score") and row["score"] >= 1.0)
    if args.solved_only and not solved:
        return None, "not solved"
    head = row["prompt_messages"]
    if not head or any(message.get("role") not in ("system", "user") for message in head):
        return None, "unexpected head messages"
    steps = row["trajectory"]
    total = len(row["prompt_token_ids"]) + sum(len(step["token_ids"]) for step in steps)
    if total < args.min_total:
        return None, f"run under {args.min_total} tokens"
    if args.upper is not None and total >= args.upper:
        return None, f"run of {args.upper} tokens or more"

    fraction = random.Random(f"{args.seed}:cut:{task_id}").uniform(*args.fraction_range)
    window = args.window_min_total is not None and total >= args.window_min_total
    cut_at, running, cut = args.lower if window else min(args.lower, fraction * total), len(row["prompt_token_ids"]), None
    for position, step in enumerate(steps):
        running += len(step["token_ids"])
        if running >= cut_at and step["role"] != "assistant":
            cut = position + 1
            break
    if cut is None or not any(step["role"] == "assistant" for step in steps[cut:]):
        return None, "nothing to predict after the cut"

    prefix, continuation, targets, context_tokens, prefix_assistant_ids = [], [], [], 0, []
    for position, step in enumerate(steps):
        messages = step.get("messages") or []
        assistants = [message for message in messages if message.get("role") == "assistant"]
        if step["role"] == "assistant" and (len(assistants) != 1 or len(messages) != 1 or not step.get("token_ids")):
            return None, "an assistant step without exactly one message and its sampled tokens"
        if step["role"] != "assistant" and assistants:
            return None, "an assistant message inside a tool or user step"
        if position < cut:
            prefix.extend(messages)
            if step["role"] == "assistant":
                prefix_assistant_ids.append(list(step["token_ids"]))
            continue
        if step["role"] == "assistant":
            targets.append(list(step["token_ids"]))
        else:
            context_tokens += len(step["token_ids"])
        continuation.extend(messages)
    written = sum(map(len, targets))
    if written < max(1, args.min_target_tokens):
        return None, f"fewer than {max(1, args.min_target_tokens)} target tokens"

    recorded_kwargs = dict((config.get("call_kwargs") or {}).get("chat_template_kwargs") or {})
    read_by_template = {key: value for key, value in recorded_kwargs.items() if key in (tokenizer.chat_template or "")}
    candidates = [read_by_template] if "enable_thinking" in read_by_template else [dict(read_by_template, enable_thinking=flag) for flag in (True, False)]
    tools = tools_from_prompt(tokenizer, row["prompt_token_ids"])
    kwargs = head_tokens = None
    for candidate in candidates:
        ids = head_ids(tokenizer, head, tools, candidate)
        if row["prompt_token_ids"] == ids:
            kwargs, head_tokens = candidate, len(ids)
            break
    if kwargs is None:
        return None, "the head does not render to the recorded prompt tokens"
    prompt_tokens = len(head_ids(tokenizer, head + prefix, tools, kwargs))
    return {
        "task_id": task_id, "run_key": f"{task_id}#s{row.get('seed')}#c{args.lower}", "seed": row.get("seed"), "config_key": row.get("config_key"), "benchmark": metadata.get("benchmark"), "repo": metadata.get("repo"),
        "language": metadata.get("language"), "side": metadata.get("side"), "official_test": metadata.get("official_test"),
        "solved": solved, "score": row.get("score"), "finish_reason": row.get("finish_reason"), "answer": row.get("answer"),
        "head_messages": head, "prefix_messages": prefix, "continuation_messages": continuation, "target_token_ids": targets,
        "prefix_assistant_token_ids": prefix_assistant_ids,
        "tools": tools, "tools_source": "prompt_tokens",
        "chat_template_kwargs": kwargs, "chat_template_kwargs_recorded": recorded_kwargs,
        "prefix_tokens": prompt_tokens - head_tokens,
        "tokens": {"head": head_tokens, "prefix": prompt_tokens - head_tokens, "prompt": prompt_tokens, "continuation_written": written,
                   "continuation_context": context_tokens, "continuation_turns": len(targets)},
        "teacher": {"writer": "pool run (the reader itself)", "finish_reason": row.get("finish_reason"), "score": row.get("score"),
                    "total_tokens": total, "cut_fraction": round(fraction, 4), "cut_rule": "window" if window or args.lower <= fraction * total else "fraction",
                    "prefix_steps": cut, "recorded_prefix_tokens": running},
        "reader_tokenizer": args.reader_tokenizer,
        "source": {"rollouts": str(args.rollouts), "note": False, "seed": row.get("seed"), "kind": "pool"},
    }, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", nargs="+", required=True, help="rollouts.jsonl of pool cache(s)")
    parser.add_argument("--out", required=True, help="material.jsonl to write (a summary goes beside it)")
    parser.add_argument("--exclude-tasks", nargs="+", default=[], help="index.json file(s) of evaluation/development prefixes: their tasks are left out")
    parser.add_argument("--lower", type=int, default=20_000, help="a run is cut at most this far in (the evaluation's window cut)")
    parser.add_argument("--fraction-range", type=float, nargs=2, default=(0.5, 0.8), help="or at this share of its length, if that comes first")
    parser.add_argument("--window-min-total", type=int, default=None, help="runs at least this long are cut at --lower whatever f is (the evaluation's window cut)")
    parser.add_argument("--min-total", type=int, default=6_000, help="leave out runs shorter than this (head included)")
    parser.add_argument("--upper", type=int, default=None, help="leave out runs of at least this many tokens (head included; the evaluation's 50,000)")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--reader-tokenizer", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--solved-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-target-tokens", type=int, default=1)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.reader_tokenizer)
    excluded = {entry["task_id"] for path in args.exclude_tasks for entry in json.loads(Path(path).read_text())["prefixes"]}
    latest: dict[str, tuple[dict | None, str]] = {}
    for path in args.rollouts:
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                task = row["agent_config"].get("dataset_task") or {}
                task_id = task.get("task_id") if isinstance(task, dict) else None
                key = (task_id, row.get("seed"))                       # one run per task and pool seed (config_key is the same for every seed)
                latest[key] = (None, "an evaluation or development task") if task_id in excluded else material_of(row, tokenizer, args)

    kept = sorted((record for record, _ in latest.values() if record is not None), key=lambda record: (record["benchmark"] or "", record["task_id"] or ""))
    why = Counter(reason for record, reason in latest.values() if record is None)
    if len({record["run_key"] for record in kept}) != len(kept):
        raise ValueError("two usable runs share a run key")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def mean(name: str) -> float | None:
        return round(sum(record["tokens"][name] for record in kept) / len(kept), 1) if kept else None

    summary = {
        "rollouts": args.rollouts, "runs_read": len(latest), "usable": len(kept), "left_out": dict(why), "excluded_task_files": args.exclude_tasks, "upper": args.upper,
        "by_seed": dict(Counter(str(record["seed"]) for record in kept)), "solved": sum(record["solved"] for record in kept),
        "lower": args.lower, "window_min_total": args.window_min_total, "fraction_range": args.fraction_range, "min_total": args.min_total, "seed": args.seed, "solved_only": args.solved_only,
        "repositories": len({record["repo"] for record in kept}), "by_language": dict(Counter(record["language"] for record in kept)),
        "by_side": dict(Counter(record["side"] for record in kept)), "cut_rule": dict(Counter(record["teacher"]["cut_rule"] for record in kept)),
        "chat_template_kwargs": sorted({json.dumps(record["chat_template_kwargs"], sort_keys=True) for record in kept}),
        "mean_tokens": {name: mean(name) for name in ("head", "prefix", "continuation_written", "continuation_context", "continuation_turns")},
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
