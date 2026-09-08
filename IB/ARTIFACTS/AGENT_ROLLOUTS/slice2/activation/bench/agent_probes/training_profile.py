"""
Training throughput probe on cached rollouts: the trainer's own step (collate -> decoder with the
adapter -> fp32 head -> clipped surrogate -> backward) under a few variants, plus a torch.profiler table
for the baseline and the best variant. Nothing is sampled and no optimizer step is taken, so the numbers
are the forward/backward cost per token that the round time is made of.

usage: python -m activation.bench.agent_probes.training_profile --cache agent_training_test_single_r0
       [--model Qwen/Qwen3.5-4B] [--variants baseline,single,...] [--micro-batches 6]
"""
import argparse
import json
import time
from collections import defaultdict

import torch

from activation.agent import AgentRunResult
from activation.agent.rollout_caching import RolloutCache
from activation.agent_training import AgentTrainer, AgentTrainingConfig, group_mean_advantage
from activation.agent_training.agent_training_utils import build_example, collate, disable_dropout, micro_batches
from activation.common.data_syncing import resolve_path
from activation.harness import TARGET_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig

MODEL_NAME, LORA = "probe", "probe_lora"

# name -> overrides: gradient checkpointing, micro-batch token budget (0 = one example per micro-batch), logits chunk,
# attention implementation, length multiple (sequence rounded up), mask (True: the padding mask goes to the decoder)
VARIANTS = {
    "padded_32k":      dict(ckpt=True,  budget=32_768, chunk=2_048, attn=None, multiple=1,    mask=True),   # the slice-2 test's configuration
    "single":          dict(ckpt=True,  budget=0,      chunk=2_048, attn=None, multiple=1,    mask=False),  # no padding: sdpa causal (flash) path
    "single_512":      dict(ckpt=True,  budget=0,      chunk=2_048, attn=None, multiple=512,  mask=False),  # + few distinct shapes for the Triton kernels
    "single_2048":     dict(ckpt=True,  budget=0,      chunk=2_048, attn=None, multiple=2048, mask=False),
    "single_512_8k":   dict(ckpt=True,  budget=0,      chunk=8_192, attn=None, multiple=512,  mask=False),
    "single_no_ckpt":  dict(ckpt=False, budget=0,      chunk=2_048, attn=None, multiple=512,  mask=False),
    "single_eager":    dict(ckpt=True,  budget=0,      chunk=2_048, attn="eager", multiple=512, mask=False),
}


def load_rows(caching_id: str | None, cache_file: str | None) -> list[dict]:
    """Rows of one rollout cache: by caching id under the synced root, or a jsonl file given directly."""
    if cache_file:
        rows = []
        with open(cache_file) as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    cache = RolloutCache(caching_id)
    return list(cache.rows.values())


def load_examples(caching_id: str | None, model_name: str, lora_name: str, max_tokens: int, cache_file: str | None = None):
    rows = load_rows(caching_id, cache_file)
    assert rows, f"no cached rollouts under {cache_file or caching_id}"
    groups = defaultdict(list)
    for row in rows:
        groups[row["config_key"]].append(AgentRunResult.deserialize(row, harness=None))
    items = []
    for group in groups.values():
        items += group_mean_advantage(group, model_name=model_name, lora_name=lora_name)
    examples = [example for example in (build_example(item, index) for index, item in enumerate(items))
                if example is not None and len(example.token_ids) <= max_tokens]
    return groups, items, examples


def bins_for(examples, budget: int) -> list[list[int]]:
    return micro_batches(examples, budget, pad_free=budget <= 0)


def collate_for(examples, group, spec, pad_token_id, device):
    return collate(examples, group, pad_token_id, device, spec["multiple"], causal_only=not spec["mask"])


def run_variant(trainer, loaded, base, examples, spec, count, pad_token_id, device, profile_rows=None):
    if spec["attn"] is not None:
        base.set_attn_implementation(spec["attn"])
    if spec["ckpt"]:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        base.gradient_checkpointing_disable()
    trainer.config.logits_chunk_tokens = spec["chunk"]
    bins = bins_for(examples, spec["budget"])[:count + 1]
    parameters = [p for p in base.parameters() if p.requires_grad]
    torch.cuda.reset_peak_memory_stats(device)
    real = padded = 0
    times = []
    for index, group in enumerate(bins):
        batch = collate_for(examples, group, spec, pad_token_id, device)
        torch.cuda.synchronize()
        started = time.time()
        loss, _, _ = trainer._objective(loaded, LORA, batch, len(group))
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.time() - started
        for p in parameters:
            p.grad = None
        if index == 0:
            continue                                                   # warm-up: kernel compile and autotune
        times.append(elapsed)
        real += int(batch.attention_mask.sum())
        padded += batch.input_ids.numel()
    if spec["attn"] is not None:
        base.set_attn_implementation("sdpa")
    total = sum(times)
    return {
        "micro_batches": len(times), "seconds": round(total, 2), "real_tokens": real, "padded_tokens": padded,
        "real_tokens_per_s": round(real / total, 1) if total else None, "padding_fraction": round(1 - real / max(1, padded), 3),
        "seconds_per_micro_batch": [round(t, 2) for t in times],
        "peak_memory_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 1),
        "micro_batch_shapes": [[len(examples[i].token_ids) for i in group] for group in bins[1:]],
    }


def profile_variant(trainer, loaded, base, examples, spec, pad_token_id, device, label: str):
    from torch.profiler import ProfilerActivity, profile
    if spec["attn"] is not None:
        base.set_attn_implementation(spec["attn"])
    if spec["ckpt"]:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        base.gradient_checkpointing_disable()
    trainer.config.logits_chunk_tokens = spec["chunk"]
    group = bins_for(examples, spec["budget"])[1]
    batch = collate_for(examples, group, spec, pad_token_id, device)
    loss, _, _ = trainer._objective(loaded, LORA, batch, len(group)); loss.backward()          # warm
    for p in base.parameters():
        p.grad = None
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        loss, _, _ = trainer._objective(loaded, LORA, batch, len(group))
        loss.backward()
        torch.cuda.synchronize()
    for p in base.parameters():
        p.grad = None
    if spec["attn"] is not None:
        base.set_attn_implementation("sdpa")
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70)
    print(f"\n===== profile: {label} (micro-batch {[len(examples[i].token_ids) for i in group]} tokens) =====\n{table}", flush=True)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="agent_training_test_single_r0", help="caching id under the synced ROLLOUTS folder")
    parser.add_argument("--cache-file", default=None, help="a rollouts.jsonl path instead of --cache")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--micro-batches", type=int, default=6, help="timed micro-batches per variant (plus one warm-up)")
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--profile", default="padded_32k,single", help="variants to run under torch.profiler ('' for none)")
    args = parser.parse_args()

    groups, items, examples = load_examples(args.cache, MODEL_NAME, LORA, args.max_tokens, args.cache_file)
    lengths = sorted(len(e.token_ids) for e in examples)
    print(f"{len(groups)} groups, {len(items)} items, {len(examples)} examples; tokens {sum(lengths)}, lengths min/median/max "
          f"{lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}, loss tokens {sum(e.num_loss_tokens for e in examples)}", flush=True)

    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={MODEL_NAME: ModelConfig(MODEL_NAME, args.model)}))
    harness.module_manager.register_lora(LORA, MODEL_NAME, rank=args.rank, dropout=0.0)
    loaded = harness.loaded_models[MODEL_NAME]
    loaded.model_to_device(TARGET_DEVICE)
    harness.module_manager.ensure_lora(LORA)
    base = loaded.model
    base.train()
    disable_dropout(base)
    device = next(base.parameters()).device
    pad_token_id = loaded.tokenizer.pad_token_id if loaded.tokenizer.pad_token_id is not None else loaded.tokenizer.eos_token_id
    trainer = AgentTrainer(harness, AgentTrainingConfig())
    print(f"attention implementation: {base.config._attn_implementation}; torch {torch.__version__}", flush=True)

    results = {}
    names = [v for v in args.variants.split(",") if v]
    for label in ("cold", "warm"):                 # cold: the Triton kernels compile for every new shape; warm: the same shapes again
        for name in names:
            spec = VARIANTS[name]
            key = f"{name}/{label}"
            try:
                results[key] = run_variant(trainer, loaded, base, examples, spec, args.micro_batches, pad_token_id, device)
            except torch.OutOfMemoryError:
                results[key] = {"error": "out of memory"}
                for p in base.parameters():
                    p.grad = None
                torch.cuda.empty_cache()
            print(f"[{key}] {json.dumps(results[key])}", flush=True)
    tables = {}
    for name in [v for v in args.profile.split(",") if v]:
        tables[name] = profile_variant(trainer, loaded, base, examples, VARIANTS[name], pad_token_id, device, name)

    out = resolve_path("TRAINING_PROFILE")
    (out / "results.json").write_text(json.dumps({"cache": args.cache_file or args.cache, "model": args.model, "examples": len(examples), "results": results}, indent=1))
    (out / "profiles.txt").write_text("\n\n".join(f"== {name}\n{table}" for name, table in tables.items()))
    print("\n| variant | real tok/s | padding | peak GB | micro-batches |\n| --- | --- | --- | --- | --- |")
    for name, r in results.items():
        if "error" in r:
            print(f"| {name} | {r['error']} | | | |")
        else:
            print(f"| {name} | {r['real_tokens_per_s']} | {r['padding_fraction']} | {r['peak_memory_gb']} | {r['micro_batch_shapes']} |")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
