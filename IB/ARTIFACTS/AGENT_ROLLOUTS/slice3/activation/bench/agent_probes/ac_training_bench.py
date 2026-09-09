"""
AC training validity run: items of one kind (compaction, traj_qa, rag_qa; or the three caches mixed),
three forward-only reference points on the held-out set, N epochs of training with a held-out KL
after every epoch (and every `--eval-every` updates in epoch 1), a secondary metric (QA answer
accuracy or compaction next-action agreement) from greedy HF decoding, a live page and a summary
JSON under `AC_BENCH/<tag>/`. Items are generated once (the study questions need the QA engine)
and cached under `AC_ITEMS/<kind>_<seed>.jsonl`; the engine is asleep for the whole training.

  uv run sky exec --sync <node> -- uv run python -m activation.bench.agent_probes.ac_training_bench --kind compaction --items 2000 --held 200 --epochs 2 --tag compaction_r1

References (same target adapter as the teacher, i.e. the base at run start):
  no_context   the AC prefix with every part removed (task + instructions only)
  recent_text  every part replaced by verbatim text of the same token budget as its rows (the last V
               tokens of a compacted span; the gold chunk's first V tokens for the QA kinds)
  untrained_ac the AC prefix through the untrained AC model
  in_context   the teacher itself (KL 0, agreement 1, by construction)
"""
import argparse
import dataclasses
import json
import random
import re
import time
from dataclasses import replace
from pathlib import Path

import torch

from activation.ac_model import (
    ActivationContextModelConfig,
    ActivationContextStudyGenerator,
    ActivationContextTrainer,
    ActivationContextTrainingConfig,
    ActivationContextTrainingItem,
    ActivationContextTrainingReporter,
    ActivationContextTrainingStats,
)
from activation.ac_model.ac_model_utils import direct_parts
from activation.agent.agent_utils import ModelDialect
from activation.common.data_syncing import resolve_path
from activation.dataset.dataset_utils import render_messages
from activation.dataset.loaders import BrightDataset, NemotronMathDataset, OpenSweTracesDataset, S1DeepResearchDataset, aime_problem_statements
from activation.harness import FREE_DEVICE, SOURCE_DEVICE, SUPPORTS_FP4, HarnessRuntime, HarnessRuntimeConfig, ModelConfig

KINDS = ("compaction", "traj_qa", "rag_qa")
SIDE_NAME, TARGET_NAME, AC_NAME, SIDE_LORA, TARGET_LORA = "side", "target", "ac_bench", "ac_side", "ac_target"
DEFAULT_RATIOS = {"compaction": (1 / 16, 1 / 8), "traj_qa": (1 / 32, 1 / 16), "rag_qa": (1 / 32, 1 / 16)}


# ------------------------------------------------------------------------------------------------ items
def item_cache_path(kind: str, seed: int) -> Path:
    return resolve_path(f"AC_ITEMS/{kind}_{seed}.jsonl")


def write_items(path: Path, items: list[ActivationContextTrainingItem]) -> None:
    with open(path, "w") as handle:
        for item in items:
            handle.write(json.dumps(dataclasses.asdict(item), ensure_ascii=False) + "\n")


def read_items(path: Path) -> list[ActivationContextTrainingItem]:
    with open(path) as handle:
        return [ActivationContextTrainingItem(**json.loads(line)) for line in handle if line.strip()]


def generate_items(harness: HarnessRuntime, kind: str, total: int, seed: int, ratios: tuple[float, float], qa_model: str | None,
                   thresholds: tuple[int, int] = (8192, 24576), max_total_tokens: int = 72_000) -> list[ActivationContextTrainingItem]:
    generator = ActivationContextStudyGenerator(harness, AC_NAME)
    half = -(-total // 2)
    if kind == "compaction":
        swe = OpenSweTracesDataset.load(harness, max_examples=half + 50, min_chars=40_000)
        math = NemotronMathDataset.load(harness, max_examples=half + 50, subset="tir", excluded_problems=aime_problem_statements(), min_chars=40_000, min_tool_calls=1)
        items = (generator.generate_compaction_samples(swe.dataset_id, half, seed=seed, threshold_range_tokens=thresholds, ratios_range=ratios, max_total_tokens=max_total_tokens)
                 + generator.generate_compaction_samples(math.dataset_id, half, seed=seed, threshold_range_tokens=thresholds, ratios_range=ratios, max_total_tokens=max_total_tokens))
    elif kind == "traj_qa":
        s1 = S1DeepResearchDataset.load(harness, max_examples=half + 50)
        swe = OpenSweTracesDataset.load(harness, max_examples=half + 50)
        harness.dataset_manager.build_bm25_indexes()
        items = (generator.generate_trajectory_qa_samples(s1.dataset_id, half, seed=seed, ratios_range=ratios, caching_id=f"ac_bench_{seed}")
                 + generator.generate_trajectory_qa_samples(swe.dataset_id, half, seed=seed, ratios_range=ratios, caching_id=f"ac_bench_{seed}"))
    elif kind == "rag_qa":
        items = []
        for domain in ("biology", "economics"):
            bright = BrightDataset.load(harness, max_examples=None, domain=domain, max_corpus_documents=20_000)
            harness.dataset_manager.build_bm25_indexes()
            items += generator.generate_rag_qa_samples(bright.dataset_id, half, seed=seed, ratios_range=ratios, caching_id=f"ac_bench_{seed}")
    else:
        raise ValueError(kind)
    if qa_model and qa_model in harness.loaded_models:
        harness.loaded_models[qa_model].engine_to_device(FREE_DEVICE)
    random.Random(seed).shuffle(items)
    return items


# ------------------------------------------------------------------------------------------------ references
def text_only(messages: list[dict]) -> list[dict]:
    out = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            message = dict(message, content="".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"))
        out.append(message)
    return out


def split_items(items: list[ActivationContextTrainingItem], held: int, train: int) -> tuple[list, list, int]:
    """
    Held-out items in exact proportion to the sources (the generated list is shuffled once by seed, so a
    head slice is mixed only in expectation), then training items among the rest whose origin (the
    trajectory of a compaction item, the study question of a QA item) is not held out. Returns
    (held, train, dropped for sharing an origin).
    """
    by_source: dict[str, list] = {}
    for item in items:
        by_source.setdefault(item.dataset_id, []).append(item)
    held_items: list = []
    for index, (source, source_items) in enumerate(by_source.items()):
        quota = round(held * len(source_items) / len(items)) if index < len(by_source) - 1 else held - len(held_items)
        held_items += source_items[:quota]
    held_ids = {item.item_id for item in held_items}
    held_origins = {item_origin(item) for item in held_items}
    train_items, shared = [], 0
    for item in items:
        if item.item_id in held_ids:
            continue
        if item_origin(item) in held_origins:
            shared += 1
            continue
        train_items.append(item)
    return held_items, train_items[:train], shared


def item_origin(item: ActivationContextTrainingItem) -> str:
    """What a held-out item must not share with a training item: its trajectory (compaction) or its study question (QA kinds)."""
    if item.kind == "compaction":
        return f"doc:{item.doc_ids[0] if item.doc_ids else item.item_id}"
    return f"question:{item.info.get('example_id', item.item_id)}"


def no_context_item(item: ActivationContextTrainingItem) -> ActivationContextTrainingItem:
    return replace(item, item_id=item.item_id + ":no_context", ac_prefix=text_only(item.ac_prefix))


def recent_text_item(item: ActivationContextTrainingItem, ac_model, tokenizer) -> ActivationContextTrainingItem:
    """
    Every part becomes verbatim text of V tokens (V = the rows it would have become): the tail of the
    compacted span for compaction; for the QA kinds the gold chunk's first V tokens in the gold part's
    place (rag_qa has one part; traj_qa's distractor parts contribute nothing).
    """
    gold_position = item.info.get("gold_position", 0)
    gold_text = item.info.get("gold_chunk_text") or ""
    messages = []
    part_index = 0
    for message in item.ac_prefix:
        content = message.get("content")
        if isinstance(content, list):
            pieces = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "activation_context":
                    rows = ac_model.part_view_rows(part["messages"], part.get("compression_target"))
                    if item.kind == "compaction":
                        ids = tokenizer.encode(render_messages(part["messages"]), add_special_tokens=False)
                        pieces.append(tokenizer.decode(ids[-rows:]))
                    elif item.kind == "rag_qa" or part_index == gold_position:
                        pieces.append(tokenizer.decode(tokenizer.encode(gold_text, add_special_tokens=False)[:rows]))
                    part_index += 1
                elif isinstance(part, dict):
                    pieces.append(part.get("text", ""))
            message = dict(message, content="".join(pieces))
        messages.append(message)
    return replace(item, item_id=item.item_id + ":recent_text", ac_prefix=messages)


# ------------------------------------------------------------------------------------------------ secondary metric
def normalize_answer(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", text.lower()).strip()


def greedy_text(harness, ac_model, trainer, item: ActivationContextTrainingItem, side: str, max_new_tokens: int) -> str:
    """Greedy HF decoding (KV cache, the target adapter, rows for the student) of the teacher ('teacher') or the student's prefix."""
    example = trainer.build_example(ac_model, item)
    target = ac_model.target
    base = target.model
    embedding = base.get_input_embeddings()
    device = next(base.parameters()).device
    lora = ac_model.config.target_model_lora_name
    if side == "teacher":
        ids = torch.tensor(example.teacher_ids[:len(example.teacher_ids) - example.num_completion], device=device)
        embeds = embedding(ids)
    else:
        prefix_len = len(example.student_ids) - example.num_completion
        ids = torch.tensor(example.student_ids[:prefix_len], device=device)
        embeds = embedding(ids)
        if example.part_requests:
            rows = ac_model.encode_batch(example.part_requests)
            pieces, cursor = [], 0
            for (start, end), part_rows in zip(example.spans, rows):
                pieces.extend([embeds[cursor:start], part_rows.to(embeds.dtype)])
                cursor = end
            pieces.append(embeds[cursor:])
            embeds = torch.cat(pieces, dim=0)
    peft_model = harness.module_manager.ensure_lora(lora)
    with harness.module_manager.lora_context(target.model_config.model_name, lora), torch.inference_mode():
        out = peft_model.generate(inputs_embeds=embeds[None], max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                  pad_token_id=target.tokenizer.pad_token_id or target.tokenizer.eos_token_id)
    return target.tokenizer.decode(out[0], skip_special_tokens=True)


def first_call(dialect: ModelDialect, text: str) -> tuple[str, str] | None:
    """(tool name, normalized argument text) of the first call in a generated text, or None."""
    _, calls = dialect.parse(text)
    if not calls:
        return None
    call = calls[0]
    arguments = call["arguments"] if isinstance(call["arguments"], dict) else {"raw": call["arguments"]}
    return call["name"], " ".join(normalize_answer(str(value)) for _, value in sorted(arguments.items()))


def call_agreement(reference: tuple[str, str] | None, candidate: tuple[str, str] | None) -> dict[str, float]:
    """Against the teacher's first call: made a call at all, same tool name, same arguments, and the token overlap (Jaccard) of the arguments."""
    if reference is None:
        return {}
    if candidate is None:
        return {"call": 0.0, "name": 0.0, "exact": 0.0, "args_overlap": 0.0}
    same_name = float(candidate[0] == reference[0])
    a, b = set(reference[1].split()), set(candidate[1].split())
    overlap = len(a & b) / len(a | b) if (a | b) else 1.0
    return {"call": 1.0, "name": same_name, "exact": float(same_name and candidate[1] == reference[1]), "args_overlap": same_name * overlap}


def secondary_metric(harness, ac_model, trainer, items, variants: dict[str, list[ActivationContextTrainingItem]], max_new_tokens: int) -> dict:
    """
    QA: answer accuracy (normalized contains) per variant and the teacher. Compaction: agreement of each variant's first
    tool call with the teacher's greedy continuation over the items where the teacher calls a tool (`<variant>_name`,
    `<variant>_exact`, `<variant>_args_overlap`, `<variant>_call`; `teacher_calls` is that share of items).
    """
    dialect = ModelDialect.for_tokenizer(ac_model.target.tokenizer)
    results: dict[str, float] = {}
    counted = 0
    samples = []
    started = time.time()
    for index, item in enumerate(items):
        teacher_text = greedy_text(harness, ac_model, trainer, item, "teacher", max_new_tokens)
        texts = {name: greedy_text(harness, ac_model, trainer, variant_items[index], "student", max_new_tokens) for name, variant_items in variants.items()}
        if item.kind == "compaction":
            reference = first_call(dialect, teacher_text)
            results["teacher_calls"] = results.get("teacher_calls", 0.0) + float(reference is not None)
            if reference is not None:
                counted += 1
                for name, text in texts.items():
                    for key, value in call_agreement(reference, first_call(dialect, text)).items():
                        results[f"{name}_{key}"] = results.get(f"{name}_{key}", 0.0) + value
        else:
            counted += 1
            gold = normalize_answer(item.completion_text)
            results["teacher"] = results.get("teacher", 0.0) + float(gold in normalize_answer(teacher_text))
            for name, text in texts.items():
                results[name] = results.get(name, 0.0) + float(gold in normalize_answer(text))
        if len(samples) < 6:
            samples.append({"item_id": item.item_id, "kind": item.kind, "reference": item.completion_text[:200], "teacher": teacher_text[:200],
                            "student": texts.get("student", "")[:200]})
    print(f"secondary metric on {len(items)} items in {time.time() - started:.0f}s", flush=True)
    out = {"items": len(items), "counted": counted}
    for key, value in results.items():
        out[key] = value / max(1, len(items)) if key == "teacher_calls" else value / max(1, counted)
    return {**out, "samples": samples}


def throughput_summary(records: list[dict]) -> dict:
    """Examples per hour and in-context (teacher) tokens per hour over the whole run, by teacher-length bucket and by whether the teacher was cached."""
    out = {}
    for label, selected in (("all", records), ("teacher_live", [r for r in records if not r["teacher_cached"]]), ("teacher_cached", [r for r in records if r["teacher_cached"]])):
        seconds = sum(record["seconds"] for record in selected)
        if not selected or seconds <= 0:
            continue
        stats = ActivationContextTrainingStats(example_records=selected)
        out[label] = {"examples": len(selected), "examples_per_hour": round(3600 * len(selected) / seconds),
                      "teacher_tokens_per_hour": round(3600 * sum(record["teacher_tokens"] for record in selected) / seconds),
                      "mean_teacher_tokens": round(sum(record["teacher_tokens"] for record in selected) / len(selected)), "by_bucket": stats.seconds_by_bucket()}
    return out


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=KINDS + ("mixed",), required=True)
    parser.add_argument("--items", type=int, default=2000, help="training items (mixed: per kind)")
    parser.add_argument("--held", type=int, default=200, help="held-out items (mixed: per kind)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--examples-per-update", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=50, help="held-out eval every this many updates in epoch 1 (0: per epoch only)")
    parser.add_argument("--side", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--target", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--qa-model", default="unsloth/Qwen3.8-27B-NVFP4" if SUPPORTS_FP4 else "Qwen/Qwen3.8-27B-FP8", help="study question engine (QA kinds, when the item cache is missing)")
    parser.add_argument("--ratio-range", default=None, help="compression ratios low,high (default per kind)")
    parser.add_argument("--threshold-range", default="8192,24576", help="compaction segment thresholds low,high in tokens (item generation only)")
    parser.add_argument("--max-total-tokens", type=int, default=72_000, help="compaction items: bound on the segments' total (item generation only)")
    parser.add_argument("--max-example-tokens", type=int, default=None, help="trainer cap on the teacher sequence (default: the trainer's)")
    parser.add_argument("--warmup-updates", type=int, default=10, help="linear learning-rate warm-up over this many updates (0: none)")
    parser.add_argument("--teacher-cache-top-k", type=int, default=0, help="> 0: cached top-k teacher after an item's first pass (see ActivationContextTrainingConfig)")
    parser.add_argument("--side-lora-rank", type=int, default=128)
    parser.add_argument("--target-lora-rank", type=int, default=64)
    parser.add_argument("--lr-ac", type=float, default=5e-4)
    parser.add_argument("--lr-target", type=float, default=2e-5)
    parser.add_argument("--secondary-items", type=int, default=100, help="held items for the greedy secondary metric (0: off)")
    parser.add_argument("--secondary-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--generate-only", action="store_true", help="generate and cache the items, then stop (the profile probe reads the cache)")
    args = parser.parse_args()
    tag = args.tag or f"{args.kind}_s{args.seed}"
    kinds = list(KINDS) if args.kind == "mixed" else [args.kind]
    ratios = tuple(float(value) for value in args.ratio_range.split(",")) if args.ratio_range else None

    # Which caches are missing decides whether the QA engine is needed.
    missing = [kind for kind in kinds if not item_cache_path(kind, args.seed).exists()]
    needs_qa = any(kind in ("traj_qa", "rag_qa") for kind in missing)
    model_configs = {SIDE_NAME: ModelConfig(SIDE_NAME, args.side), TARGET_NAME: ModelConfig(TARGET_NAME, args.target)}
    if needs_qa:
        model_configs[args.qa_model] = ModelConfig(args.qa_model, args.qa_model)
    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs=model_configs, dataset_study_qa_model_name=args.qa_model if needs_qa else None))
    harness.module_manager.register_lora(TARGET_LORA, TARGET_NAME, rank=args.target_lora_rank)
    ac_model = harness.module_manager.register_ac_model(ActivationContextModelConfig(AC_NAME, SIDE_NAME, SIDE_LORA, TARGET_NAME, TARGET_LORA, side_lora_rank=args.side_lora_rank))
    train_items, held_items = [], []
    for kind in kinds:
        path = item_cache_path(kind, args.seed)
        if path.exists():
            items = read_items(path)
            print(f"{kind}: {len(items)} items from {path}")
        else:
            items = generate_items(harness, kind, args.items + args.held, args.seed, ratios or DEFAULT_RATIOS[kind], args.qa_model if needs_qa else None,
                                   thresholds=tuple(int(value) for value in args.threshold_range.split(",")), max_total_tokens=args.max_total_tokens)
            write_items(path, items)
            print(f"{kind}: {len(items)} items generated and cached at {path}")
        assert len(items) > args.held, f"{kind}: {len(items)} items, {args.held} held"
        held, train, shared = split_items(items, args.held, args.items)
        held_items += held
        train_items += train
        print(f"{kind}: {len(held)} held-out, {len(train)} training ({shared} items sharing a document with a held-out item dropped)", flush=True)
    random.Random(args.seed).shuffle(train_items)
    print(f"{len(train_items)} training items, {len(held_items)} held-out", flush=True)
    if args.generate_only:
        return

    report_folder = resolve_path(f"AC_BENCH/{tag}")
    reporter = ActivationContextTrainingReporter(str(report_folder), f"AC training: {tag}", f"{args.kind}, {len(train_items)} items, {args.epochs} epochs, side {args.side}, target {args.target}")
    training_config = ActivationContextTrainingConfig(examples_per_update=args.examples_per_update, learning_rate_ac=args.lr_ac, learning_rate_target_lora=args.lr_target,
                                                      seed=args.seed, teacher_cache_top_k=args.teacher_cache_top_k, warmup_updates=args.warmup_updates)
    if args.max_example_tokens:
        training_config.max_example_tokens = args.max_example_tokens
    trainer = ActivationContextTrainer(harness, training_config)
    tokenizer = ac_model.target.tokenizer
    variants = {"no_context": [no_context_item(item) for item in held_items],
                "recent_text": [recent_text_item(item, ac_model, tokenizer) for item in held_items]}
    summary = {"tag": tag, "kind": args.kind, "args": vars(args), "train_items": len(train_items), "held_items": len(held_items), "references": {}, "epochs": [], "evals": []}
    started = time.time()
    for name, items in list(variants.items()) + [("untrained_ac", held_items)]:
        result = trainer.eval(AC_NAME, items, release=False)
        summary["references"][name] = {"kl": result["kl"], "agreement": result["agreement"], "by_kind": result["by_kind"]}
        reporter.report_reference(name, result["kl"])
        reporter.report_eval(0, name, result)
        print(f"reference {name}: kl {result['kl']:.4f} agreement {result['agreement']:.3f}", flush=True)
    summary["references"]["in_context"] = {"kl": 0.0, "agreement": 1.0}
    secondary_items = held_items[:args.secondary_items] if args.secondary_items else []
    if secondary_items:
        secondary = secondary_metric(harness, ac_model, trainer, secondary_items, {"student": secondary_items, **{name: items[:len(secondary_items)] for name, items in variants.items()}}, args.secondary_tokens)
        summary["evals"].append({"epoch": 0, "secondary": {key: value for key, value in secondary.items() if key != "samples"}})
        reporter.report_completions(secondary["samples"])
        print(f"secondary before training: { {k: round(v, 3) for k, v in secondary.items() if isinstance(v, float)} }", flush=True)
    (report_folder / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    updates = 0
    example_records: list[dict] = []
    for epoch in range(args.epochs):
        chunk = args.eval_every * args.examples_per_update if (epoch == 0 and args.eval_every) else len(train_items)
        for start in range(0, len(train_items), chunk):
            stats = trainer.train(AC_NAME, train_items[start:start + chunk], reporting_data=held_items, reporter=reporter)
            updates += stats.steps
            round_summary = stats.summarize()
            summary["evals"].append({"epoch": epoch + 1, "updates": updates, "held": {"kl": stats.reporting["kl"], "agreement": stats.reporting["agreement"], "by_kind": stats.reporting["by_kind"]},
                                     "train_kl_last": stats.kl[-1], "tokens_per_second": round_summary["tokens_per_second"], "peak_memory_gb": round_summary["peak_memory_gb"],
                                     "seconds_by_bucket": round_summary["seconds_by_bucket"], "teacher_cache_hits": round_summary["teacher_cache_hits"],
                                     "dropped_too_long": stats.dropped_too_long, "train_seconds": round(sum(stats.step_seconds), 1)})
            example_records.extend(stats.example_records)
            (report_folder / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        epoch_summary = {"epoch": epoch + 1, "updates": updates, "held_kl": summary["evals"][-1]["held"]["kl"], "held_agreement": summary["evals"][-1]["held"]["agreement"],
                         "checkpoint": stats.checkpoint_path, "target_lora_checkpoint": stats.target_lora_checkpoint_path}
        if secondary_items:
            trainer.eval(AC_NAME, held_items[:1], release=False)                       # places the models again
            secondary = secondary_metric(harness, ac_model, trainer, secondary_items, {"student": secondary_items, **{name: items[:len(secondary_items)] for name, items in variants.items()}}, args.secondary_tokens)
            epoch_summary["secondary"] = {key: value for key, value in secondary.items() if key != "samples"}
            reporter.report_completions(secondary["samples"])
            ac_model.release()
            ac_model.target.model_to_device(SOURCE_DEVICE)
            ac_model.side.model_to_device(SOURCE_DEVICE)
        summary["epochs"].append(epoch_summary)
        (report_folder / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        print(f"epoch {epoch + 1}: {json.dumps(epoch_summary, default=str)}", flush=True)
    held_kls = [entry["held"]["kl"] for entry in summary["evals"] if "held" in entry]
    summary["final_held_kl"] = held_kls[-1] if held_kls else None
    summary["best_held_kl"] = min(held_kls) if held_kls else None
    summary["duration_s"] = time.time() - started
    summary["throughput"] = throughput_summary(example_records)
    (report_folder / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    reporter.finish()
    print(json.dumps({key: value for key, value in summary.items() if key not in ("evals", "args")}, indent=1, default=str))


if __name__ == "__main__":
    main()
