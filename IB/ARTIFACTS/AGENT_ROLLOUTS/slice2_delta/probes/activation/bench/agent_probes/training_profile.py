"""
Training throughput probe on cached rollouts: the trainer's own step (collate -> decoder with the
adapter -> fp32 head -> clipped surrogate -> backward) under a few variants, plus a torch.profiler table
for the baseline and the best variant. Nothing is sampled and no optimizer step is taken, so the numbers
are the forward/backward cost per token that the round time is made of.

usage: python -m activation.bench.agent_probes.training_profile --cache agent_training_test_single_r0
       [--model Qwen/Qwen3.5-4B] [--variants single,packed_flex,...] [--micro-batches 6]
       [--check-packing packed_flex,packed_sdpa] [--synthetic 52000 --synthetic-variants single]
"""
import argparse
import json
import time
from collections import defaultdict

import torch

from activation.agent import AgentRunResult
from activation.agent.rollout_caching import RolloutCache
from activation.agent_training import AgentTrainer, AgentTrainingConfig, group_mean_advantage
from activation.agent_training.agent_training_utils import (TrainingExample, build_example, collate, disable_dropout, micro_batches,
                                                             packing_spacer_tokens, set_attention_implementation)
from activation.common.data_syncing import resolve_path
from activation.agent_training.fla_cache import configure_fla_cache
from activation.harness import TARGET_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig

MODEL_NAME, LORA = "probe", "probe_lora"

# name -> overrides: gradient checkpointing (True / False / the minimum micro-batch tokens for it), micro-batch token
# budget (0 = one example per micro-batch), logits chunk, attention implementation, length multiple (sequence rounded
# up), mask (True: the padding mask goes to the decoder), packed (several examples in one row, per-example attention
# and recurrent state), flex_dynamic (compile flex_attention with dynamic shapes instead of one graph per length)
def spec(ckpt=True, budget=0, chunk=2_048, attn=None, multiple=1, mask=False, packed=False, flex_dynamic=False, kwargs=None, spacers=True,
         fp32_head=False):
    return dict(ckpt=ckpt, budget=budget, chunk=chunk, attn=attn, multiple=multiple, mask=mask, packed=packed, flex_dynamic=flex_dynamic,
                kwargs=kwargs or {}, spacers=spacers, fp32_head=fp32_head)


# flex attention's backward has no valid Triton config for head_dim 256 on sm_120 (the 64/64/64/64 default needs 114 KB of shared
# memory, the RTX 6000 Pro has 101 KB); smaller blocks through kernel_options, which reach the attention layers as kwargs
FLEX_B32 = {"kernel_options": {"BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32}}
FLEX_B32_64 = {"kernel_options": {"BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32}}
FLEX_B16 = {"kernel_options": {"BLOCK_M1": 16, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 16}}


VARIANTS = {
    "padded_32k":      spec(budget=32_768, mask=True),                                   # the original slice-2 configuration
    "single":          spec(),                                                            # one example per micro-batch: sdpa causal (flash) path; the slice-2 handoff default
    "single_ckpt_8k":  spec(ckpt=8_192),                                                  # no recompute below 8k tokens
    "single_ckpt_16k": spec(ckpt=16_384),
    "single_ckpt_24k": spec(ckpt=24_576),
    "single_no_ckpt":  spec(ckpt=False),
    "single_fp32head": spec(fp32_head=True),                                              # the head's GEMM on the fp32 SIMT kernels (the previous code)
    "single_8k_chunks": spec(chunk=8_192),
    "single_flex":     spec(attn="flex_attention"),                                       # flex on single rows (its own cost vs sdpa)
    "packed_sdpa":     spec(budget=32_768, packed=True),                                  # sdpa with the dense block mask: correctness reference, expected slow
    "packed_flex":     spec(budget=32_768, packed=True, attn="flex_attention", multiple=1_024),   # BlockMask; few shapes for the static compile
    "packed_flex_dyn": spec(budget=32_768, packed=True, attn="flex_attention", flex_dynamic=True),
    "packed_flex_16k": spec(budget=16_384, packed=True, attn="flex_attention", multiple=1_024),
    "packed_flex_52k": spec(budget=52_000, packed=True, attn="flex_attention", multiple=1_024),
    "packed_flash":    spec(budget=32_768, packed=True, attn="flash_attention_2"),        # needs flash-attn installed (cu_seqlens varlen path)
    "packed_flex_ckpt_8k": spec(ckpt=8_192, budget=32_768, packed=True, attn="flex_attention", multiple=1_024),
    "packed_flex_b32":     spec(budget=32_768, packed=True, attn="flex_attention", multiple=1_024, kwargs=FLEX_B32),
    "packed_flex_b32_64":  spec(budget=32_768, packed=True, attn="flex_attention", multiple=1_024, kwargs=FLEX_B32_64),
    "packed_flex_b16":     spec(budget=32_768, packed=True, attn="flex_attention", multiple=1_024, kwargs=FLEX_B16),
    "packed_flex_b32_dyn": spec(budget=32_768, packed=True, attn="flex_attention", flex_dynamic=True, kwargs=FLEX_B32),
    "packed_flex_b32_52k": spec(budget=52_000, packed=True, attn="flex_attention", multiple=1_024, kwargs=FLEX_B32),
    "single_flex_b32":     spec(attn="flex_attention", kwargs=FLEX_B32),
    "packed_sdpa_nospacer": spec(budget=32_768, packed=True, spacers=False),               # the conv leak, for the check
}
DEFAULT_VARIANTS = "single,single_fp32head,single_no_ckpt,single_ckpt_16k,single_8k_chunks,packed_flex_b16"


def apply_attention(base, name: str | None):
    """The attention implementation for a variant; None restores sdpa."""
    set_attention_implementation(base, name or "sdpa")


def set_flex_dynamic(dynamic: bool):
    """transformers compiles flex_attention once with dynamic=False (one graph per sequence length); optionally dynamic."""
    from torch.nn.attention.flex_attention import flex_attention
    from transformers.integrations import flex_attention as integration
    wrapper = integration.WrappedFlexAttention
    if dynamic:
        wrapper._compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
        wrapper._is_flex_compiled = True
        wrapper.training = True
    else:
        wrapper._is_flex_compiled = False
        wrapper._compiled_flex_attention = None


def set_checkpointing(base, ckpt, num_tokens: int):
    wanted = bool(ckpt) if isinstance(ckpt, bool) else num_tokens >= int(ckpt)
    if wanted != bool(base.is_gradient_checkpointing):
        if wanted:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            base.gradient_checkpointing_disable()


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


def bins_for(examples, spec) -> list[list[int]]:
    return micro_batches(examples, spec["budget"], pad_free=spec["budget"] <= 0, packed=spec["packed"])


SPACERS = {"count": 0}          # set from the model at start-up


def collate_for(examples, group, spec, pad_token_id, device):
    return collate(examples, group, pad_token_id, device, spec["multiple"], causal_only=not spec["mask"], packed=spec["packed"],
                   spacer_tokens=SPACERS["count"] if spec["spacers"] else 0)


def prepare(base, trainer, spec):
    apply_attention(base, spec["attn"])
    if spec["attn"] == "flex_attention":
        set_flex_dynamic(spec["flex_dynamic"])
    trainer.config.logits_chunk_tokens = spec["chunk"]
    trainer.config.pack_micro_batches = spec["packed"]
    trainer.config.decoder_kwargs = dict(spec["kwargs"])
    trainer.config.fp32_head_matmul = spec["fp32_head"]


def restore(base):
    apply_attention(base, None)
    set_flex_dynamic(False)


def clear_grads(base):
    for p in base.parameters():
        p.grad = None


def run_variant(trainer, loaded, base, examples, spec, count, pad_token_id, device, bins=None):
    prepare(base, trainer, spec)
    bins = (bins if bins is not None else bins_for(examples, spec))[:count + 1]
    torch.cuda.reset_peak_memory_stats(device)
    real = padded = 0
    times = []
    try:
        for index, group in enumerate(bins):
            batch = collate_for(examples, group, spec, pad_token_id, device)
            set_checkpointing(base, spec["ckpt"], int(batch.input_ids.numel()))
            torch.cuda.synchronize()
            started = time.time()
            loss, _, _ = trainer._objective(loaded, LORA, batch, len(group))
            loss.backward()
            torch.cuda.synchronize()
            elapsed = time.time() - started
            clear_grads(base)
            if index == 0:
                continue                                                   # warm-up: kernel compile and autotune
            times.append(elapsed)
            real += int(batch.attention_mask.sum())
            padded += batch.input_ids.numel()
    finally:
        restore(base)
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
    prepare(base, trainer, spec)
    bins = bins_for(examples, spec)
    group = bins[min(1, len(bins) - 1)]
    batch = collate_for(examples, group, spec, pad_token_id, device)
    set_checkpointing(base, spec["ckpt"], int(batch.input_ids.numel()))
    loss, _, _ = trainer._objective(loaded, LORA, batch, len(group)); loss.backward()          # warm
    clear_grads(base)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        loss, _, _ = trainer._objective(loaded, LORA, batch, len(group))
        loss.backward()
        torch.cuda.synchronize()
    clear_grads(base)
    restore(base)
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70)
    print(f"\n===== profile: {label} (micro-batch {[len(examples[i].token_ids) for i in group]} tokens) =====\n{table}", flush=True)
    return table


@torch.no_grad()
def check_packing(trainer, loaded, base, examples, spec, pad_token_id, device, label: str, bins_to_check: int = 2) -> dict:
    """
    The log-probs of a packed row against the same examples one at a time under sdpa (the reference the
    engine's log-probs were checked against). Reports the deviation on the loss tokens; anything beyond the
    bf16 noise between two causal passes (measured alongside: the same example as a prefix of a longer row)
    is the gated-delta conv leak at the boundaries or a wrong mask.
    """
    reference = VARIANTS["single"]
    bins = [group for group in bins_for(examples, spec) if len(group) >= 2][:bins_to_check]
    report = {"bins": []}
    for group in bins:
        prepare(base, trainer, spec)
        batch = collate_for(examples, group, spec, pad_token_id, device)
        packed_lp, segments, _ = trainer._hidden_and_logprobs(loaded, LORA, batch)
        loss_packed = float(trainer._objective(loaded, LORA, batch, len(group))[0])
        restore(base)
        prepare(base, trainer, reference)
        diffs, loss_single = [], 0.0
        for ordinal, index in enumerate(group):
            single = collate_for(examples, [index], reference, pad_token_id, device)
            single_lp = trainer._hidden_and_logprobs(loaded, LORA, single)[0]
            diffs.append((packed_lp[segments == ordinal] - single_lp).abs())
            loss_single += float(trainer._objective(loaded, LORA, single, len(group))[0])
        restore(base)
        diff = torch.cat(diffs)
        report["bins"].append({
            "lengths": [len(examples[i].token_ids) for i in group], "loss_tokens": int(diff.numel()),
            "max_abs": round(float(diff.max()), 4), "mean_abs": round(float(diff.mean()), 5),
            "fraction_over_0.05": round(float((diff > 0.05).float().mean()), 4),
            "per_example_mean_abs": [round(float(d.mean()), 5) for d in diffs],          # the first example has no boundary before it
            "loss_packed": round(loss_packed, 5), "loss_single": round(loss_single, 5),
        })
        print(f"[check {label}] {json.dumps(report['bins'][-1])}", flush=True)
    # bf16 noise floor: the first example alone vs as the prefix of itself + the second, under the reference (causal, no packing)
    if bins:
        prepare(base, trainer, reference)
        first, second = examples[bins[0][0]], examples[bins[0][1]]
        joined = TrainingExample(first.token_ids + second.token_ids, first.loss_mask + [False] * len(second.token_ids),
                                 first.old_logprobs + [0.0] * len(second.token_ids), first.weight, first.item_index, False)
        lp_joined = trainer._hidden_and_logprobs(loaded, LORA, collate_for([joined], [0], reference, pad_token_id, device))[0]
        lp_alone = trainer._hidden_and_logprobs(loaded, LORA, collate_for([first], [0], reference, pad_token_id, device))[0]
        noise = (lp_joined - lp_alone).abs()
        report["bf16_prefix_noise"] = {"max_abs": round(float(noise.max()), 4), "mean_abs": round(float(noise.mean()), 5)}
        print(f"[check {label}] bf16 prefix noise floor {json.dumps(report['bf16_prefix_noise'])}", flush=True)
        restore(base)
    return report


def check_head(trainer, loaded, base, examples, pad_token_id, device) -> dict:
    """The bf16-operand / fp32-accumulation head against the fp32 GEMM: log-probs and the gradient at the states."""
    example = max(range(len(examples)), key=lambda i: len(examples[i].token_ids))
    batch = collate_for(examples, [example], VARIANTS["single"], pad_token_id, device)
    out = {}
    base.gradient_checkpointing_disable()
    with torch.no_grad():
        inputs_embeds = base.get_input_embeddings()(batch.input_ids)
        hidden = loaded.decoder_forward(inputs_embeds, None, batch.position_ids, LORA)
    values, grads = {}, {}
    for name, fp32 in (("fp32", True), ("bf16acc", False)):
        states = hidden.detach().requires_grad_(True)
        from activation.agent_training.agent_training_utils import sampled_logprobs
        logprobs = sampled_logprobs(states, base.get_output_embeddings(), batch, 2_048, fp32)[0]
        logprobs.sum().backward()
        values[name], grads[name] = logprobs.detach(), states.grad.detach().clone()
    diff = (values["fp32"] - values["bf16acc"]).abs()
    gdiff = (grads["fp32"] - grads["bf16acc"]).abs()
    out = {"logprob_max_abs": round(float(diff.max()), 6), "logprob_mean_abs": round(float(diff.mean()), 7),
           "grad_max_abs": round(float(gdiff.max()), 6), "grad_rel_fro": round(float(gdiff.float().norm() / grads["fp32"].float().norm()), 6)}
    print(f"[check head] {json.dumps(out)}", flush=True)
    return out


def synthetic_example(examples, num_tokens: int) -> TrainingExample:
    """One example of `num_tokens` tokens made of the cached ones concatenated (real token statistics, loss where they had it)."""
    ids, loss, old = [], [], []
    order = sorted(range(len(examples)), key=lambda i: -len(examples[i].token_ids))
    while len(ids) < num_tokens:
        for index in order:
            example = examples[index]
            ids += example.token_ids; loss += example.loss_mask; old += example.old_logprobs
            if len(ids) >= num_tokens:
                break
    return TrainingExample(ids[:num_tokens], loss[:num_tokens], old[:num_tokens], 1.0, 0, False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="agent_training_test_single_r0", help="caching id under the synced ROLLOUTS folder")
    parser.add_argument("--cache-file", default=None, help="a rollouts.jsonl path instead of --cache")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--micro-batches", type=int, default=6, help="timed micro-batches per variant (plus one warm-up)")
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--profile", default="single,single_no_ckpt", help="variants to run under torch.profiler ('' for none)")
    parser.add_argument("--check-packing", default="packed_flex_b16", help="packed variants whose log-probs are compared with single rows ('' for none)")
    parser.add_argument("--no-check-head", action="store_true")
    parser.add_argument("--synthetic", type=int, default=0, help="also time one synthetic example of this many tokens (memory ceiling)")
    parser.add_argument("--synthetic-variants", default="single,single_flex")
    parser.add_argument("--warm", action="store_true", help="repeat every variant once more with the kernels warm")
    args = parser.parse_args()
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 128)          # flex: one graph per length
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 1024)

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
    print(f"fla configs: {configure_fla_cache(device)}", flush=True)
    try:
        import fla  # noqa: F401
        fla_ok = True
    except ImportError:
        fla_ok = False
    SPACERS["count"] = packing_spacer_tokens(base)
    print(f"attention implementation: {base.config._attn_implementation}; torch {torch.__version__}; fla {fla_ok}; "
          f"spacer tokens {SPACERS['count']}", flush=True)

    results = {}
    names = [v for v in args.variants.split(",") if v]
    for label in (("cold", "warm") if args.warm else ("cold",)):      # cold: the kernels compile for every new shape; warm: the same shapes again
        for name in names:
            spec = VARIANTS[name]
            key = f"{name}/{label}"
            try:
                results[key] = run_variant(trainer, loaded, base, examples, spec, args.micro_batches, pad_token_id, device)
            except torch.OutOfMemoryError:
                results[key] = {"error": "out of memory"}
            except Exception as error:                                   # a missing kernel package, an unsupported route
                results[key] = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
                restore(base)
            clear_grads(base)
            torch.cuda.empty_cache()
            print(f"[{key}] {json.dumps(results[key])}", flush=True)
    checks = {}
    if not args.no_check_head:
        try:
            checks["head"] = check_head(trainer, loaded, base, examples, pad_token_id, device)
        except Exception as error:
            checks["head"] = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
            print(f"[check head] {checks['head']['error']}", flush=True)
        clear_grads(base)
        torch.cuda.empty_cache()
    for name in [v for v in args.check_packing.split(",") if v]:
        try:
            checks[name] = check_packing(trainer, loaded, base, examples, VARIANTS[name], pad_token_id, device, name)
        except Exception as error:
            checks[name] = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
            restore(base)
            print(f"[check {name}] {checks[name]['error']}", flush=True)
        torch.cuda.empty_cache()
    if args.synthetic:
        big = [synthetic_example(examples, args.synthetic)]
        for name in [v for v in args.synthetic_variants.split(",") if v]:
            key = f"synthetic_{args.synthetic}/{name}"
            try:
                results[key] = run_variant(trainer, loaded, base, big, VARIANTS[name], 1, pad_token_id, device, bins=[[0], [0]])
            except torch.OutOfMemoryError:
                results[key] = {"error": "out of memory"}
                restore(base)
            except Exception as error:
                results[key] = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
                restore(base)
            clear_grads(base)
            torch.cuda.empty_cache()
            print(f"[{key}] {json.dumps(results[key])}", flush=True)
    tables = {}
    for name in [v for v in args.profile.split(",") if v]:
        try:
            tables[name] = profile_variant(trainer, loaded, base, examples, VARIANTS[name], pad_token_id, device, name)
        except Exception as error:
            tables[name] = f"{type(error).__name__}: {str(error)[:300]}"
            restore(base)
            clear_grads(base)
            torch.cuda.empty_cache()
            print(f"[profile {name}] {tables[name]}", flush=True)

    out = resolve_path("TRAINING_PROFILE")
    (out / "results.json").write_text(json.dumps({"cache": args.cache_file or args.cache, "model": args.model, "examples": len(examples),
                                                  "results": results, "packing_checks": checks}, indent=1))
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
