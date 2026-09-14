"""
AC training profile: what one training example costs, per phase, against the card's measured GEMM
ceiling. Items come from a bench item cache (`AC_ITEMS/<kind>_<seed>.jsonl`) bucketed by teacher
length, plus synthetic compaction items lengthened to a target (a real trajectory's messages cycled,
split into `depth` nested segments) for the lengths the cache lacks. For every example it times
the teacher forward, the AC encode (embedding, mixer, pooling, side decoder, heads), the student
forward, the head chunks and the backward, records the peak memory of each, counts the FLOPs of
each from the token counts and the model shapes (dense 2·params·tokens, plus the full-attention
layers' T² term; the linear-attention layers' chunked scans are omitted, well under 1% at these
lengths) and reports the achieved TFLOPS next to the ceiling. Also: the exact-vs-cached-teacher KL
gap for `--teacher-cache-top-k` and the shared autotuner config provenance.
Use `python -m activation.autotuners inspect` or `promote` to inspect or version validated results.

  TRITON_PRINT_AUTOTUNING=1 uv run python -m activation.bench.agent_probes.ac_training_profile \\
      --items-cache AC_ITEMS/compaction_0.jsonl --steps 3 --synthetic 72000:2,104000:2 --teacher-cache-top-k 64
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from activation.ac_model import ActivationContextModelConfig, ActivationContextTrainer, ActivationContextTrainingConfig, ActivationContextTrainingItem
from activation.ac_model.ac_model import TRAINING
from activation.ac_model.ac_model_study import COMPACTION_INSTRUCTIONS, ac_part
from activation.agent_training.agent_training_utils import disable_dropout, set_checkpointing
from activation.autotuners import configure_fla_runtime
from activation.common.data_syncing import resolve_path
from activation.dataset.dataset_utils import message_text
from activation.harness import SOURCE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig

SIDE_NAME, TARGET_NAME, SIDE_LORA, TARGET_LORA = "side", "target", "ac_side", "ac_target"
DEFAULT_BUCKET_EDGES = "8192,16384,32768,49152,65536,73728"


# ------------------------------------------------------------------------------------------------ device helpers
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak(device) -> float:
    return torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def gemm_ceiling_tflops(device, size: int = 8192, repeats: int = 10) -> float:
    """bf16 GEMM throughput of the card (size³ matmuls): the ceiling every phase is measured against."""
    if device.type != "cuda":
        return 0.0
    a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    sync(device)
    started = time.time()
    for _ in range(repeats):
        a @ b
    sync(device)
    seconds = time.time() - started
    del a, b
    return repeats * 2 * size**3 / seconds / 1e12


# ------------------------------------------------------------------------------------------------ FLOP model
def text_config(model):
    return model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config


def decoder_flops(model):
    """tokens -> forward FLOPs of the decoder (no head): 2·params·T plus the full-attention layers' causal T² term."""
    params = sum(p.numel() for name, p in model.named_parameters() if "embed_tokens" not in name and "lm_head" not in name and "lora" not in name.lower())
    config = text_config(model)
    layer_types = list(getattr(config, "layer_types", None) or [])
    full_layers = layer_types.count("full_attention") if layer_types else int(config.num_hidden_layers)
    attention_dim = int(config.num_attention_heads) * int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))

    def flops(tokens: int) -> float:
        return 2.0 * params * tokens + full_layers * 2.0 * tokens * tokens * attention_dim
    flops.params = params  # type: ignore[attr-defined]
    return flops


def module_params(module) -> int:
    return sum(p.numel() for p in module.parameters()) if module is not None else 0


def items_ac_name(items: list[ActivationContextTrainingItem], default: str = "ac_profile") -> str:
    """The AC model name the items' parts are addressed to (parts of another model's name are refused by the encoder)."""
    for item in items:
        for message in item.ac_prefix:
            for part in (message.get("content") if isinstance(message.get("content"), list) else []):
                if isinstance(part, dict) and part.get("type") == "activation_context" and part.get("ac_name"):
                    return str(part["ac_name"])
    return default


# ------------------------------------------------------------------------------------------------ synthetic items
def synthetic_item(item: ActivationContextTrainingItem, tokenizer, target_tokens: int, depth: int, ac_name: str, scale: float = 1.0) -> ActivationContextTrainingItem:
    """
    A compaction item of about `target_tokens` teacher tokens: the item's in-context messages after the task
    are cycled until the budget is met, then split into `depth` equal segments nested as the generator nests them.
    The per-message estimate ignores the chat template's rendering of calls, so `scale` (target / measured,
    see `build_synthetic`) corrects the budget on a second pass.
    """
    prefix = list(item.in_context_prefix)
    system = [message for message in prefix[:1] if message.get("role") == "system"]
    rest = prefix[len(system):]
    assert rest and rest[0].get("role") == "user", "the first message after the system prompt is the task"
    task, tail = rest[0], rest[1:]
    assert tail, "an item with messages after the task"

    def tokens(message: dict) -> int:
        text = message_text(message)
        for call in message.get("tool_calls") or []:
            text += json.dumps(call.get("function", call).get("arguments", {}))
        return len(tokenizer.encode(text, add_special_tokens=False)) + 8

    budget = (target_tokens - tokens(task) - len(tokenizer.encode(item.completion_text, add_special_tokens=False)) - 64) * scale
    body: list[dict] = []
    total = 0
    cursor = 0
    while total < budget:
        message = tail[cursor % len(tail)]
        body.append(message)
        total += tokens(message)
        cursor += 1
    # Segments of equal token share, cut at message boundaries.
    segments: list[list[dict]] = []
    share = total / depth
    current: list[dict] = []
    current_tokens = 0
    for message in body:
        current.append(message)
        current_tokens += tokens(message)
        if current_tokens >= share and len(segments) < depth - 1:
            segments.append(current)
            current, current_tokens = [], 0
    if current:
        segments.append(current)
    ratio = float(item.info.get("ratio") or 1 / 16)
    nested = None
    for segment in segments:
        inner = [{"role": "user", "content": [nested]}] if nested is not None else [task]
        nested = ac_part(inner + segment, ac_name, ratio)
    ac_user = {"role": "user", "content": [nested, {"type": "text", "text": "\n\n" + COMPACTION_INSTRUCTIONS}]}
    return ActivationContextTrainingItem(
        item_id=f"synthetic:{target_tokens}:{depth}:{item.item_id}", kind="compaction", in_context_prefix=system + [task] + body,
        ac_prefix=system + [task, ac_user], completion_text=item.completion_text, completion_complete=item.completion_complete,
        teacher_partial_text="", tools=item.tools, dataset_id=item.dataset_id, doc_ids=list(item.doc_ids),
        info={"depth": len(segments), "ratio": ratio, "synthetic_target": target_tokens})


def build_synthetic(trainer, ac_model, item: ActivationContextTrainingItem, target_tokens: int, depth: int, ac_name: str):
    """The example of a synthetic item within 5% of the target: one build to measure the template overhead, one corrected."""
    example = trainer.build_example(ac_model, synthetic_item(item, ac_model.target.tokenizer, target_tokens, depth, ac_name))
    measured = len(example.teacher_ids)
    if abs(measured - target_tokens) > 0.05 * target_tokens:
        example = trainer.build_example(ac_model, synthetic_item(item, ac_model.target.tokenizer, target_tokens, depth, ac_name, scale=target_tokens / measured))
    return example


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items-cache", default="AC_ITEMS/compaction_0.jsonl", help="under the synced folder")
    parser.add_argument("--side", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--target", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--steps", type=int, default=3, help="examples per bucket")
    parser.add_argument("--bucket-edges", default=DEFAULT_BUCKET_EDGES, help="teacher-length bucket upper edges")
    parser.add_argument("--synthetic", default="", help="synthetic compaction items as teacher_tokens:depth, comma separated (e.g. 72000:2,104000:2)")
    parser.add_argument("--synthetic-steps", type=int, default=2, help="examples per synthetic length (different source items)")
    parser.add_argument("--max-example-tokens", type=int, default=None, help="cap on the teacher sequence (default: the trainer's; synthetic items are exempt)")
    parser.add_argument("--logits-chunk", type=int, default=2048)
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--checkpointing-min-tokens", type=int, default=None)
    parser.add_argument("--teacher-cache-top-k", type=int, default=0, help="> 0: also compute the cached-teacher KL and report the gap to the exact KL")
    parser.add_argument("--tag", default="profile")
    args = parser.parse_args()

    with open(resolve_path(args.items_cache, create=False)) as handle:
        items = [ActivationContextTrainingItem(**json.loads(line)) for line in handle if line.strip()]
    ac_name = items_ac_name(items)
    print(f"{len(items)} items from {args.items_cache} (AC model {ac_name!r})")
    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={SIDE_NAME: ModelConfig(SIDE_NAME, args.side), TARGET_NAME: ModelConfig(TARGET_NAME, args.target)}))
    harness.module_manager.register_lora(TARGET_LORA, TARGET_NAME, rank=64)
    ac_model = harness.module_manager.register_ac_model(ActivationContextModelConfig(ac_name, SIDE_NAME, SIDE_LORA, TARGET_NAME, TARGET_LORA))
    config = ActivationContextTrainingConfig(logits_chunk_tokens=args.logits_chunk, gradient_checkpointing=not args.no_checkpointing,
                                             gradient_checkpointing_min_tokens=args.checkpointing_min_tokens)
    if args.max_example_tokens:
        config.max_example_tokens = args.max_example_tokens
    trainer = ActivationContextTrainer(harness, config)

    variables = trainer._autotuning_variables(ac_model)
    print("Autotuning:", variables["provenance"])
    with configure_fla_runtime(variables["fla_config"]):
        # Placement as in train().
        harness.module_manager.ensure_lora(TARGET_LORA)
        device = ac_model.prepare()
        ac_model.set_mode(TRAINING)
        ac_model.profile_phases = True
        target, side = ac_model.target.model, ac_model.side.model
        for base in (target, side):
            base.train()
            disable_dropout(base)
        min_tokens = variables["target_gradient_checkpointing_min_tokens"]
        set_checkpointing(target, config, min_tokens, min_tokens)
        set_checkpointing(side, config, min_tokens, min_tokens)
        embedding = target.get_input_embeddings()
        head = target.get_output_embeddings()
        base_dtype = next(target.parameters()).dtype
        parameters = ac_model.trainable_parameters() + harness.module_manager.lora_parameters(TARGET_LORA)
        target_flops, side_flops = decoder_flops(target), decoder_flops(side)
        d_target, vocab = int(text_config(target).hidden_size), int(head.weight.shape[0])
        head_flops_per_position = 2.0 * d_target * vocab
        mixer_params = module_params(ac_model.modules.mixer) + module_params(ac_model.modules.pooling) + module_params(ac_model.modules.summary_pooling)
        head_module_params = module_params(ac_model.modules.target_head) + module_params(ac_model.modules.recursive_adapter)
        ceiling = gemm_ceiling_tflops(device)
        print(f"GEMM ceiling {ceiling:.0f} TFLOPS (bf16 8192³); target {target_flops.params / 1e9:.2f}B decoder params, side {side_flops.params / 1e9:.2f}B, "
              f"vocab {vocab}, d {d_target}", flush=True)

        # Examples: the cache bucketed by teacher length (built lazily: tokenizing thousands of items takes minutes), then synthetic lengths.
        edges = [int(value) for value in args.bucket_edges.split(",")]
        buckets = [(0 if index == 0 else edges[index - 1], edge) for index, edge in enumerate(edges)]
        examples_by_bucket: dict[tuple[int, int], list] = {bucket: [] for bucket in buckets}
        for item in items:
            if all(len(examples) >= args.steps for examples in examples_by_bucket.values()):
                break
            example = trainer.build_example(ac_model, item)
            if len(example.teacher_ids) > config.max_example_tokens:
                continue
            for bucket in buckets:
                if bucket[0] <= len(example.teacher_ids) < bucket[1] and len(examples_by_bucket[bucket]) < args.steps:
                    examples_by_bucket[bucket].append(example)
        labels = {bucket: f"{bucket[0] // 1024}k-{bucket[1] // 1024}k" for bucket in buckets}
        if args.synthetic:
            sources = [item for item in items if item.kind == "compaction" and not item.teacher_partial_text][:args.synthetic_steps]
            for spec in args.synthetic.split(","):
                target_tokens, depth = (int(value) for value in spec.split(":"))
                label = f"synthetic {target_tokens // 1000}k depth {depth}"
                examples_by_bucket[(target_tokens, depth)] = [build_synthetic(trainer, ac_model, item, target_tokens, depth, ac_name) for item in sources]
                labels[(target_tokens, depth)] = label

        rows_out = []
        for bucket, examples in examples_by_bucket.items():
            label = labels[bucket]
            if not examples:
                print(f"bucket {label}: no items")
                continue
            for example in examples:
                timings, peaks, flops = {}, {}, {}
                num = example.num_completion
                teacher_tokens, student_tokens = len(example.teacher_ids), len(example.student_ids)
                for parameter in parameters:
                    parameter.grad = None
                sequence_start = len(ac_model.stats.decoder_sequences)
                side_tokens_before, rows_before = ac_model.stats.side_tokens, ac_model.stats.view_rows
                phases_before = dict(ac_model.stats.phase_seconds)
                try:
                    # Teacher forward.
                    teacher_ids = torch.tensor(example.teacher_ids, device=device)
                    reset_peak(device); sync(device); started = time.time()
                    with torch.no_grad():
                        teacher_states = ac_model.target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=TARGET_LORA)[0][-num - 1:-1]
                    sync(device); timings["teacher_forward"] = time.time() - started; peaks["teacher_forward"] = peak(device)
                    flops["teacher_forward"] = target_flops(teacher_tokens)
                    # Teacher top-k extraction (the cache write) when asked.
                    cached = None
                    if args.teacher_cache_top_k:
                        reset_peak(device); sync(device); started = time.time()
                        cached = trainer._teacher_top_k(teacher_states, head, args.teacher_cache_top_k)
                        sync(device); timings["teacher_topk"] = time.time() - started; peaks["teacher_topk"] = peak(device)
                        flops["teacher_topk"] = head_flops_per_position * num
                    # AC encode (grad).
                    reset_peak(device); sync(device); started = time.time()
                    rows = ac_model.encode_batch(example.part_requests)
                    sync(device); timings["ac_encode"] = time.time() - started; peaks["ac_encode"] = peak(device)
                    side_tokens = ac_model.stats.side_tokens - side_tokens_before
                    view_rows = ac_model.stats.view_rows - rows_before
                    side_sequences = ac_model.stats.decoder_sequences[sequence_start:]
                    side_sequence = sum(side_sequences)  # actual content + V per part; recursive parts included
                    side_decoder_flops = sum(side_flops(length) for length in side_sequences)
                    flops["ac_encode"] = 2.0 * mixer_params * side_tokens + side_decoder_flops + 2.0 * head_module_params * view_rows
                    # Student forward.
                    reset_peak(device); sync(device); started = time.time()
                    student_ids = torch.tensor(example.student_ids, device=device)
                    with torch.no_grad():
                        student_embeds = embedding(student_ids)
                    checkpointed = set_checkpointing(target, config, student_tokens, min_tokens)
                    pieces, cursor = [], 0
                    for (start, end), part_rows in zip(example.spans, rows):
                        pieces.extend([student_embeds[cursor:start], part_rows.to(base_dtype)])
                        cursor = end
                    pieces.append(student_embeds[cursor:])
                    student_embeds = torch.cat(pieces, dim=0)
                    student_states = ac_model.target.decoder_forward(student_embeds[None], None, lora_name=TARGET_LORA)[0][-num - 1:-1]
                    sync(device); timings["student_forward"] = time.time() - started; peaks["student_forward"] = peak(device)
                    flops["student_forward"] = target_flops(student_tokens)
                    # Head chunks (KL), exact.
                    reset_peak(device); sync(device); started = time.time()
                    kl, agreement = trainer._chunked_kl(teacher_states, student_states, head)
                    sync(device); timings["head_chunks"] = time.time() - started; peaks["head_chunks"] = peak(device)
                    flops["head_chunks"] = 2.0 * head_flops_per_position * num
                    kl_cached_gap = None
                    if cached is not None:
                        with torch.no_grad():
                            kl_cached, _ = trainer._chunked_kl(None, student_states.detach(), head, cached=cached)
                        kl_cached_gap = float(kl) - float(kl_cached)
                    # Backward (student + side + AC modules).
                    reset_peak(device); sync(device); started = time.time()
                    kl.backward()
                    sync(device); timings["backward"] = time.time() - started; peaks["backward"] = peak(device)
                    side_checkpointed = config.gradient_checkpointing and ac_model.config.side_gradient_checkpointing
                    flops["backward"] = (target_flops(student_tokens) * (2 if checkpointed else 1)             # recompute + activation gradients
                                         + side_decoder_flops * (2 if side_checkpointed else 1)
                                         + 4.0 * mixer_params * side_tokens + 4.0 * head_module_params * view_rows   # trainable: activation + weight gradients
                                         + 2.0 * head_flops_per_position * num)                              # student head: recompute + gradients
                except torch.OutOfMemoryError as error:
                    print(json.dumps({"bucket": label, "item": example.item.item_id, "teacher_tokens": teacher_tokens, "student_tokens": student_tokens,
                                      "oom_in": next((name for name in ("backward", "head_chunks", "student_forward", "ac_encode", "teacher_topk", "teacher_forward") if name not in timings), "?"),
                                      "timings_s": {key: round(value, 3) for key, value in timings.items()}, "peaks_gb": {key: round(value, 2) for key, value in peaks.items()},
                                      "error": str(error)[:200]}), flush=True)
                    rows_out.append({"bucket": label, "item": example.item.item_id, "teacher_tokens": teacher_tokens, "student_tokens": student_tokens, "oom": True,
                                     "timings_s": {key: round(value, 3) for key, value in timings.items()}, "peaks_gb": {key: round(value, 2) for key, value in peaks.items()}})
                    for parameter in parameters:
                        parameter.grad = None
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                total = sum(timings.values())
                total_flops = sum(flops.values())
                encode_phases = {key: round(value - phases_before.get(key, 0.0), 3) for key, value in ac_model.stats.phase_seconds.items()}
                row = {"bucket": label, "item": example.item.item_id, "kind": example.item.kind, "depth": example.item.info.get("depth"),
                       "teacher_tokens": teacher_tokens, "student_tokens": student_tokens, "side_tokens": side_tokens, "side_sequence": side_sequence, "side_sequences": side_sequences,
                       "parts": len(example.part_requests), "rows": view_rows, "completion": num, "kl": round(float(kl), 4), "agreement": round(agreement, 4),
                       "kl_cached_gap": None if kl_cached_gap is None else round(kl_cached_gap, 5),
                       "timings_s": {key: round(value, 3) for key, value in timings.items()}, "encode_phases_s": encode_phases,
                       "peaks_gb": {key: round(value, 2) for key, value in peaks.items()},
                       "tflops": {key: round(flops[key] / max(timings[key], 1e-6) / 1e12, 1) for key in timings},
                       "total_s": round(total, 3), "total_tflop": round(total_flops / 1e12, 1), "achieved_tflops": round(total_flops / total / 1e12, 1),
                       "roofline_s": round(total_flops / max(ceiling, 1e-6) / 1e12, 3) if ceiling else None,
                       "checkpointed": bool(checkpointed), "tokens_per_s": round((teacher_tokens + student_tokens) / total)}
                rows_out.append(row)
                print(json.dumps(row), flush=True)
                del teacher_states, student_states, rows, kl, cached
                for parameter in parameters:
                    parameter.grad = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # Aggregates.
        measured = [row for row in rows_out if not row.get("oom")]
        by_bucket = {}
        for row in measured:
            by_bucket.setdefault(row["bucket"], []).append(row)
        table = []
        for label, rows in by_bucket.items():
            mean_total = sum(row["total_s"] for row in rows) / len(rows)
            mean_teacher = sum(row["teacher_tokens"] for row in rows) / len(rows)
            table.append({"bucket": label, "examples": len(rows), "mean_teacher_tokens": round(mean_teacher), "mean_s": round(mean_total, 2),
                          "examples_per_hour": round(3600 / mean_total), "teacher_tokens_per_hour": round(3600 * mean_teacher / mean_total),
                          "achieved_tflops": round(sum(row["total_tflop"] for row in rows) / sum(row["total_s"] for row in rows), 1),
                          "roofline_fraction": round(sum(row["roofline_s"] or 0 for row in rows) / sum(row["total_s"] for row in rows), 3) if ceiling else None,
                          "phase_share": {key: round(sum(row["timings_s"][key] for row in rows) / sum(row["total_s"] for row in rows), 3) for key in rows[0]["timings_s"]},
                          "peak_gb": round(max(max(row["peaks_gb"].values()) for row in rows), 1),
                          "kl_cached_gap": round(sum(row["kl_cached_gap"] for row in rows) / len(rows), 5) if all(row.get("kl_cached_gap") is not None for row in rows) else None})
        print("\nbucket | n | teacher tok | s/example | examples/h | teacher tok/h | TFLOPS | of ceiling | peak GB | shares")
        for entry in table:
            shares = " ".join(f"{key[:7]} {value:.2f}" for key, value in entry["phase_share"].items())
            print(f"{entry['bucket']} | {entry['examples']} | {entry['mean_teacher_tokens']} | {entry['mean_s']} | {entry['examples_per_hour']} | {entry['teacher_tokens_per_hour']} | "
                  f"{entry['achieved_tflops']} | {entry['roofline_fraction']} | {entry['peak_gb']} | {shares}")
        print("AC model stats:", ac_model.stats.summarize())
        ac_model.profile_phases = False
        ac_model.release()
        ac_model.target.model_to_device(SOURCE_DEVICE)
        ac_model.side.model_to_device(SOURCE_DEVICE)
        out = resolve_path("AC_BENCH/profile") / f"{args.tag}.json"
        out.write_text(json.dumps({"flops_note": "Approximate decoder/dense parameter FLOPs; summary FFN is conservatively charged at N rather than V. Sequence lengths are measured per part; padding/attention overhead is not exact.", "args": vars(args), "gemm_ceiling_tflops": round(ceiling, 1), "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                                   "target_params": target_flops.params, "side_params": side_flops.params, "autotuning": variables["provenance"],
                                   "table": table, "rows": rows_out}, indent=1))
        print("written", out)


if __name__ == "__main__":
    main()
