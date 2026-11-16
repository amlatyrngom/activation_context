"""
Long-context reconstruction material for the compaction recipe (2026-11-16): the phase between the shared warm-up pair
R_res_rq9fr (FineWeb passages of 1-2k tokens) and the KL stage 1. CPU only (CUDA_VISIBLE_DEVICES= ).

One item per pool run. The encoder reads EXACTLY the AC part a stage-1 item reads (prepare_agent_continuation.build_item:
`activation_part("compaction", flattened head + prefix view, tool definitions)`), and the frozen reader must reproduce that
view as text. The reader's prompt is the warm-up's reconstruction prompt (generate_reconstruction_samples, variant
"reconstruct"): system RECONSTRUCTION_SYSTEM_PROMPT, user ["Passage:\\n", part, "\\n\\n" + RECONSTRUCTION_INSTRUCTION]. SFT.

    target text   the encoder's own rendering of the view (render_with_part_sentinels: the side chat template, inline <think>
                  escaped to [Reasoning], as the encoder tokenizes the part), without the tool definitions, as plain text:
                  the empty "<think>\\n\\n</think>\\n\\n" blocks the template adds are dropped, "<|im_start|>" becomes "### "
                  (so "### assistant", "### user" head each message) and "<|im_end|>" is dropped. Tool calls and tool
                  responses keep their template tags (<tool_call>, <tool_response>). No chat-control token, no <think> tag.
    size          r drawn per run from --r-values (seeded by run key; default 1/10 and 1/5 half and half), rows_wanted =
                  round(prefix_tokens x r) with prefix_tokens = the reader template's tokens of head + prefix (the material's
                  `prompt` count: what the full-context teacher reads); the part's compression target gives exactly that
                  many rows (ac_sizes.py); past max_view_rows (4,096) the encoder clamps (1/5 of a prefix over 20,480 tokens)
                  and the item is marked `clamped`.
    cut rule      the eval's window (kit/cut_prefixes.py, export_pool_material.py): the first step boundary that is not an
                  assistant turn at or past --lower (20,000) tokens, head included; the prefix must stay under --upper
                  (50,000). Unlike the eval, nothing has to follow the cut, so runs that hit the trajectory cap are kept
                  (no continuation is needed here). Runs that ended in error / task_unavailable, or ran with compactions,
                  subagents, injected input or an adapter are left out (as export_pool_material.py).
    exclusion     the 322 eval-200 / dev-75 / frozen-100 tasks: the three --exclude-tasks index files must be exactly
                  --exclude-ids (kit/exclude_h7.json), and no item of the output may hold one of them (asserted).
    held panel    --held runs drawn (seeded) among the training-side runs, each at 1/10 and 1/5 (groups lcrec/r10, lcrec/r5):
                  the in-run reconstruction panel. They never train.

Outputs (--out, a new folder): items.pkl {"train": [...], "held": []} (what sec45_train --kind packet reads), held_panel.pkl
{"items": [...]} (its --held-items), prepared.json {"training_examples", ...} (what cmp_stage1_job.sh reads), build.json (counts,
lengths, exclusion checks, drops). Every item is built once by the trainer's own example builder (run_channel_mixture
checked_example: the 72,000-token reader limit on both views, the 65,536-position encoder limit), and its row count is checked.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<repo> CONTINUATION_BENCH_DIR=<repo>/IB/TMP/PAPER_CORPUS_QA/kit/bench python build_lcrec.py \
        --rollouts POOLS/*/rollouts.jsonl --exclude-tasks DEV75 EVAL200 FROZEN100 --exclude-ids kit/exclude_h7.json \
        --initial-ac PAIR/ac --initial-reader PAIR/reader --out PACKETS/lcrec_v1
"""
import argparse
import json
import math
import os
import random
import statistics
import unicodedata
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import prepare_agent_continuation as P                       # sets up the bench path (CONTINUATION_BENCH_DIR) and the shared helpers
from ac_sizes import compression_target_for, rows_wanted_for
from export_continuation_material import DROPPED_FINISH, head_ids, tools_from_prompt

import run_channel_mixture as mix                                                                          # noqa: E402
from run_channel_mixture import ForkPool, dump, file_sha, nested_parts, write                              # noqa: E402
from run_qa import AC, harness                                                                             # noqa: E402
from activation.ac_model import (ActivationContextStudyGenerator, ActivationContextTrainer,              # noqa: E402
                                 ActivationContextTrainingConfig, ActivationContextTrainingItem)
from activation.ac_model.ac_model_study import RECONSTRUCTION_INSTRUCTION, RECONSTRUCTION_SYSTEM_PROMPT    # noqa: E402
from activation.ac_model.ac_model_utils import assistant_target_ids, render_with_part_sentinels            # noqa: E402
from activation.common.ac_parts import AC_FLAT_MAX_TOKENS, activation_part, resolve_activation_part        # noqa: E402

FAMILY = "long_reconstruction"
HELD_SIZES = (("r10", 1 / 10), ("r5", 1 / 5))
FORBIDDEN = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>")
CHUNK = 4


def cut_run(row: dict, tokenizer, args) -> tuple[dict | None, str]:
    """The run's head + prefix at the window cut (no continuation needed), or (None, why it is left out)."""
    config = row["agent_config"]
    metadata = config.get("metadata") or {}
    task_id = (config.get("dataset_task") or {}).get("task_id")
    if row.get("finish_reason") in DROPPED_FINISH:
        return None, f"finish_reason {row['finish_reason']}"
    if row.get("compactions") or row.get("subagent_results") or row.get("injected_input"):
        return None, "compaction, subagent or injected input in the run"
    if row.get("lora_name") or row.get("ac_model_name") or config.get("lora_name") or config.get("ac_model_name"):
        return None, "ran with an adapter or an AC model"
    head = row["prompt_messages"]
    if not head or any(message.get("role") not in ("system", "user") for message in head):
        return None, "unexpected head messages"
    running, cut = len(row["prompt_token_ids"]), None
    for position, step in enumerate(row["trajectory"]):
        running += len(step["token_ids"])
        if running >= args.lower and step["role"] != "assistant":
            cut = position + 1
            break
    if cut is None:
        return None, f"run never reaches {args.lower} tokens at a non-assistant step boundary"
    if running >= args.upper:
        return None, f"prefix of {args.upper} tokens or more"
    prefix = [message for step in row["trajectory"][:cut] for message in step.get("messages") or []]
    recorded = dict((config.get("call_kwargs") or {}).get("chat_template_kwargs") or {})
    read = {key: value for key, value in recorded.items() if key in (tokenizer.chat_template or "")}
    candidates = [read] if "enable_thinking" in read else [dict(read, enable_thinking=flag) for flag in (True, False)]
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
    return {"task_id": task_id, "run_key": f"{task_id}#s{row.get('seed')}#c{args.lower}", "seed": row.get("seed"), "repo": metadata.get("repo"),
            "language": metadata.get("language"), "side": metadata.get("side"), "benchmark": metadata.get("benchmark"),
            "finish_reason": row.get("finish_reason"), "solved": bool(row.get("score") and row["score"] >= 1.0),
            "head_messages": head, "prefix_messages": prefix, "tools": tools, "chat_template_kwargs": kwargs,
            "prompt_tokens": prompt_tokens, "head_tokens": head_tokens, "recorded_prefix_tokens": running, "prefix_steps": cut}, ""


def target_text(side_tokenizer, view: list[dict]) -> str:
    """The view as the encoder renders it (no tool definitions), as plain text (see the module doc)."""
    text = render_with_part_sentinels(side_tokenizer, view, tools=None)
    text = text.replace("<think>\n\n</think>\n\n", "").replace("<|im_start|>", "### ").replace("<|im_end|>", "").strip()
    bad = [token for token in FORBIDDEN if token in text]
    if bad:
        raise ValueError(f"target text holds {bad}")
    return text


def build_item(record: dict, r: float, item_id: str, group: str, ac, ac_name: str, side_tokens: int, text: str, reader_tokenizer) -> ActivationContextTrainingItem:
    rows_wanted = rows_wanted_for(record["prompt_tokens"], r)
    ratio = compression_target_for(rows_wanted, side_tokens)
    rows = ac.num_view_rows(side_tokens, ratio)
    if rows != min(ac.config.max_view_rows, max(ac.config.min_view_rows, rows_wanted)):
        raise ValueError(f"{item_id}: the encoder gives {rows} rows for {rows_wanted} wanted")
    view = P.flattened_view(record)
    tools = deepcopy(record["tools"]) or None
    part = resolve_activation_part(activation_part("compaction", view, tools), SimpleNamespace(ac_model_name=ac_name, ac_compaction_ratio=ratio))
    output = {"role": "assistant", "content": text}
    target_ids = assistant_target_ids(reader_tokenizer, output)
    if unicodedata.normalize("NFC", reader_tokenizer.decode(target_ids)) != unicodedata.normalize("NFC", text + "<|im_end|>"):   # the tokenizer NFC-normalizes (e.g. U+037E -> ";")
        raise ValueError(f"{item_id}: the target tokens do not decode to the target text + <|im_end|>")
    system = [{"role": "system", "content": RECONSTRUCTION_SYSTEM_PROMPT}]
    content = [{"type": "text", "text": "Passage:\n"}, part, {"type": "text", "text": f"\n\n{RECONSTRUCTION_INSTRUCTION}"}]
    return ActivationContextTrainingItem(
        item_id=item_id, kind="reconstruct",
        in_context_messages=system + [{"role": "user", "content": f"Passage:\n{text}\n\n{RECONSTRUCTION_INSTRUCTION}"}, output],
        ac_messages=system + [{"role": "user", "content": content}, deepcopy(output)],
        in_context_start=len(system) + 1, ac_start=len(system) + 1, assistant_token_ids=[target_ids], tools=None,
        dataset_id=FAMILY, doc_ids=[record["task_id"]],
        info={"family": FAMILY, "source": FAMILY, "panel_group": group, "variant": "reconstruct", "task_id": record["task_id"],
              "run_key": record["run_key"], "seed": record["seed"], "repo": record["repo"], "language": record["language"], "side": record["side"],
              "finish_reason": record["finish_reason"], "solved": record["solved"], "r": r, "ratio": ratio, "rows_wanted": rows_wanted, "rows": rows,
              "clamped": rows_wanted > ac.config.max_view_rows, "side_tokens": side_tokens, "prefix_tokens": record["prompt_tokens"],
              "head_tokens": record["head_tokens"], "recorded_prefix_tokens": record["recorded_prefix_tokens"], "text_target_tokens": len(target_ids)})


def quantiles(values: list[int]) -> dict:
    values = sorted(values)
    return {"n": len(values), "min": values[0], "median": statistics.median(values), "p90": values[int(0.9 * (len(values) - 1))],
            "max": values[-1], "mean": round(statistics.fmean(values), 1)} if values else {"n": 0}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", type=Path, nargs="+", required=True, help="pool caches (one rollouts.jsonl each)")
    p.add_argument("--exclude-tasks", type=Path, nargs="+", required=True, help="index.json files of the eval-200, dev-75 and frozen-100 prefixes")
    p.add_argument("--exclude-ids", type=Path, required=True, help="kit/exclude_h7.json: the same 322 task ids (checked equal)")
    p.add_argument("--initial-ac", required=True); p.add_argument("--initial-reader", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lower", type=int, default=20_000); p.add_argument("--upper", type=int, default=50_000)
    p.add_argument("--r-values", type=float, nargs="+", default=[0.1, 0.2])
    p.add_argument("--held", type=int, default=50, help="training-side runs held out as the reconstruction panel")
    p.add_argument("--seed", type=int, default=20261116)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists: use a new name")

    excluded = {entry["task_id"] for path in a.exclude_tasks for entry in json.loads(path.read_text())["prefixes"]}
    listed = set(json.loads(a.exclude_ids.read_text())["task_ids"])
    if excluded != listed:
        raise ValueError(f"the index files give {len(excluded)} tasks, {a.exclude_ids} {len(listed)}: not the same set")

    h = harness(a.initial_ac, a.initial_reader)
    study = ActivationContextStudyGenerator(h, AC)
    ac = study.ac_model
    reader_tokenizer, side_tokenizer = study.tokenizer, ac.side.tokenizer

    latest, read_runs, excluded_seen = {}, Counter(), 0
    for path in a.rollouts:
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = (row["agent_config"].get("dataset_task") or {}).get("task_id")
                read_runs[path.parent.name] += 1
                if task_id in excluded:
                    excluded_seen += 1
                    latest[(task_id, row.get("seed"))] = (None, "an eval-200 / dev-75 / frozen-100 task")
                    continue
                latest[(task_id, row.get("seed"))] = cut_run(row, reader_tokenizer, a)
    left_out = Counter(reason for record, reason in latest.values() if record is None)
    records = sorted((record for record, _ in latest.values() if record is not None), key=lambda record: record["run_key"])
    if len({record["run_key"] for record in records}) != len(records):
        raise ValueError("two runs share a run key")

    # ---- encoder-side length (as part_view_rows counts it) and the target text
    usable = []
    for record in records:
        view = P.flattened_view(record)
        ids, _ = P.tokenize_with_parts(side_tokenizer, view, [], ac.pad_id, tools=record["tools"] or None)
        if len(ids) > AC_FLAT_MAX_TOKENS:
            left_out["prefix view over the flattened-view budget"] += 1
            continue
        usable.append((record, len(ids), target_text(side_tokenizer, view)))

    training_side = sorted(record["run_key"] for record, _, _ in usable if record["side"] == "training")
    held_keys = set(random.Random(f"{a.seed}:held").sample(training_side, a.held))
    items, held_items = [], []
    for record, side_tokens, text in usable:
        key = record["run_key"]
        if key in held_keys:
            for label, r in HELD_SIZES:
                held_items.append(build_item(record, r, f"panel:lcrec/{label}:{key}", f"lcrec/{label}", ac, study.ac_name, side_tokens, text, reader_tokenizer))
        else:
            r = a.r_values[random.Random(f"{a.seed}:r:{key}").randrange(len(a.r_values))]
            items.append(build_item(record, r, f"{FAMILY}:{key}", "lcrec/train", ac, study.ac_name, side_tokens, text, reader_tokenizer))

    # ---- every item through the trainer's own builder and limits
    config = ActivationContextTrainingConfig(loss_kind="sft", learning_rate_ac=5e-5, learning_rate_target_lora=0.0, examples_per_update=8, num_epochs=1,
                                             completion_samples=0, logits_chunk_tokens=64, gradient_checkpointing_min_tokens=0, seed=mix.SEED,
                                             deterministic=False, schedule_total_updates=max(1, len(items) // 8), shuffle_examples=True)
    trainer = ActivationContextTrainer(h, config)
    everything = items + held_items
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    mix.BUILD_CONTEXT = {"trainer": trainer, "study": study, "scan": True, "output": everything}
    failures, built = [], {}
    with ForkPool(a.workers) as pool:
        for results in pool.map(mix.example_chunk, [range(i, min(i + CHUNK, len(everything))) for i in range(0, len(everything), CHUNK)]):
            for value, failure in results:
                if failure:
                    failures.append(failure)
                else:
                    item, example = value[:2]
                    (request,) = example.part_requests
                    (start, end), = example.spans
                    if request.compression_ratio != item.info["ratio"] or end - start != item.info["rows"]:
                        raise ValueError(f"{item.item_id}: {end - start} rows at target {request.compression_ratio}, the item says {item.info['rows']} at {item.info['ratio']}")
                    if math.ceil(item.info["side_tokens"] * item.info["ratio"]) != item.info["rows_wanted"]:
                        raise ValueError(f"{item.item_id}: ceil(side tokens x target) is not the wanted row count")
                    if nested_parts(item):
                        raise ValueError(f"{item.item_id}: a nested part")
                    built[item.item_id] = item
    mix.BUILD_CONTEXT = None
    if any(failure["id"].startswith("panel:") for failure in failures):
        raise ValueError(f"held items failed to build: {failures[:3]}")
    items = [built[item.item_id] for item in items if item.item_id in built]
    held_items = [built[item.item_id] for item in held_items]
    if len({item.item_id for item in items + held_items}) != len(items) + len(held_items):
        raise ValueError("duplicate item ids")
    trained_keys = {item.info["run_key"] for item in items}
    if trained_keys & held_keys or len(trained_keys) != len(items):
        raise ValueError("a held run trains, or a run has two training items")
    tasks = {item.info["task_id"] for item in items + held_items}
    if tasks & listed or tasks & excluded:
        raise ValueError(f"{len(tasks & listed)} excluded tasks in the output")

    a.out.mkdir(parents=True)
    dump(a.out / "items.pkl", {"train": items, "held": []})
    dump(a.out / "held_panel.pkl", {"items": held_items})
    updates = len(items) // 8
    report = {
        "rule": "one item per pool run: the stage-1 AC part of head + prefix at the eval window cut (trajectory-capped runs kept), reconstructed as text by the frozen reader",
        "rollouts": {str(path): read_runs[path.parent.name] for path in a.rollouts}, "runs_read": len(latest),
        "excluded_tasks": {"index_files": [str(path) for path in a.exclude_tasks], "ids_file": str(a.exclude_ids), "count": len(listed),
                           "runs_of_excluded_tasks_in_the_pools": excluded_seen, "excluded_tasks_in_output": len(tasks & listed)},
        "left_out": dict(left_out), "build_failures": failures, "lower": a.lower, "upper": a.upper, "seed": a.seed, "r_values": a.r_values,
        "training_items": len(items), "training_runs": len(trained_keys), "training_tasks": len({item.info["task_id"] for item in items}),
        "training_repositories": len({item.info["repo"] for item in items}), "updates_per_pass": updates, "leftover_items": len(items) - 8 * updates,
        "training_by_side": dict(Counter(item.info["side"] for item in items)), "training_by_finish": dict(Counter(item.info["finish_reason"] for item in items)),
        "training_by_language": dict(Counter(item.info["language"] for item in items)),
        "training_r_counts": {str(r): n for r, n in sorted(Counter(item.info["r"] for item in items).items())},
        "training_clamped": sum(item.info["clamped"] for item in items),
        "training_prefix_tokens": quantiles([item.info["prefix_tokens"] for item in items]),
        "training_recorded_prefix_tokens": quantiles([item.info["recorded_prefix_tokens"] for item in items]),
        "training_target_tokens": quantiles([item.info["target_tokens"] for item in items]),
        "training_rows": quantiles([item.info["rows"] for item in items]), "training_reader_tokens": quantiles([item.info["reader_tokens"] for item in items]),
        "held_runs": len(held_keys), "held_items": len(held_items), "held_groups": dict(Counter(item.info["panel_group"] for item in held_items)),
        "held_run_keys": sorted(held_keys), "held_target_tokens": sum(item.info["target_tokens"] for item in held_items),
        "initial_pair": [a.initial_ac, a.initial_reader], "items_sha256": file_sha(a.out / "items.pkl"), "held_panel_sha256": file_sha(a.out / "held_panel.pkl"),
    }
    write(a.out / "build.json", report)
    write(a.out / "prepared.json", {"training_examples": len(items), "held_examples": len(held_items), "items_sha256": report["items_sha256"],
                                    "held_panel_sha256": report["held_panel_sha256"], "builder": "build_lcrec.py"})
    print(json.dumps({key: value for key, value in report.items() if key not in ("held_run_keys", "build_failures")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
