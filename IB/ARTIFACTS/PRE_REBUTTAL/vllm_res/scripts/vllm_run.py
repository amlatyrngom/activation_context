"""
vLLM side of the Hidden Residuals check, one engine per process (run twice):

  python vllm_run.py --engine stock --ref REF_DIR --out OUT_DIR      upstream Qwen3.5-9B, rows-only prompts (the harness's path today)
  python vllm_run.py --engine res   --ref REF_DIR --out OUT_DIR      activation.vllm_residuals (hf_overrides), packed rows

REF_DIR is hf_reference.py's output. Per engine:
  equivalence  every reference item alone (prefix cache reset before each), greedy, first-position top-20 log-probs:
               stock: rows only; res: zero-delta pack first (compared bit for bit with stock), then rows + deltas.
  cache (res)  the longest item: cold, repeated, deltas changed everywhere, deltas changed in the last quarter of the rows only
               (RequestOutput.num_cached_tokens).
  throughput   32 concurrent requests, each ~4,000 distinct rows (rows/deltas of the reference items, rolled per request) inside
               the longest item's template: prefill only (max_tokens 1) and end to end (max_tokens 256, ignore_eos), prefix cache reset
               before each pass; payload bytes per request; CPU cost of hashing/serialising one payload.
Writes OUT_DIR/<engine>.json and OUT_DIR/<engine>_equiv.pt.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch

HIDDEN = 4096


def load_items(ref: Path):
    manifest = json.loads((ref / "manifest.json").read_text())
    items = [torch.load(ref / "items" / f"{entry['index']:02d}.pt", weights_only=False) for entry in manifest["items"]]
    return manifest, items


def prompt(ids, positions, rows):
    """Mixed EmbedsPrompt: full-length tensor, rows at the AC positions, zeros at token positions."""
    embeds = torch.zeros(len(ids), rows.shape[1], dtype=torch.bfloat16)
    embeds[positions] = rows
    mask = [True] * len(ids)
    for position in positions:
        mask[position] = False
    return {"prompt_embeds": embeds, "prompt_token_ids": list(ids), "prompt_is_token_ids": mask}


def packed(item, layers, scale=1.0, zero=False, from_row=0, to_row=None):
    from activation.vllm_residuals import pack_rows
    deltas = {}
    for layer in layers:
        delta = item["deltas"][layer].clone()
        if zero:
            delta.zero_()
        elif scale != 1.0:
            delta[from_row:to_row] = (delta[from_row:to_row].float() * scale).to(torch.bfloat16)
        deltas[layer] = delta
    return pack_rows(item["row_embeds"], deltas)


def first_logprobs(output) -> dict:
    step = output.outputs[0].logprobs[0]
    return {int(token): float(entry.logprob) for token, entry in step.items()}


def run_one(llm, request, params):
    llm.reset_prefix_cache()
    output = llm.generate([request], params, use_tqdm=False)[0]
    return {"tokens": list(output.outputs[0].token_ids), "logprobs": first_logprobs(output), "cached": output.num_cached_tokens}


def synthetic_requests(items, layers, engine, count=32, rows_per_request=4000):
    """`count` distinct prompts: the longest item's template with its row run replaced by `rows_per_request` rows (+ deltas)."""
    from activation.vllm_residuals import pack_rows
    base = max(items, key=lambda item: len(item["positions"]))
    first, last = base["positions"][0], base["positions"][-1]
    pool_rows = torch.cat([item["row_embeds"] for item in items])
    pool_deltas = {layer: torch.cat([item["deltas"][layer] for item in items]) for layer in layers}
    requests = []
    for index in range(count):
        take = (torch.arange(rows_per_request) + 97 * index) % pool_rows.shape[0]
        rows = pool_rows[take]
        if engine == "res":
            rows = pack_rows(rows, {layer: pool_deltas[layer][take] for layer in layers})
        ids = base["ids"][:first] + [base["ids"][first]] * rows_per_request + base["ids"][last + 1:]
        requests.append(prompt(ids, list(range(first, first + rows_per_request)), rows))
    return requests


def timed_pass(llm, requests, params):
    llm.reset_prefix_cache()
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(requests, params, use_tqdm=False)
    seconds = time.perf_counter() - started
    return seconds, sum(len(output.outputs[0].token_ids) for output in outputs), [output.num_cached_tokens for output in outputs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=("stock", "res"), required=True)
    p.add_argument("--ref", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--enable-lora", action="store_true", help="construct the engine as the harness does (LoRA slots, rank 128)")
    p.add_argument("--skip-throughput", action="store_true")
    p.add_argument("--eager", action="store_true", help="enforce_eager (no torch.compile, no CUDA graphs); outputs get the suffix _eager")
    a = p.parse_args()
    from vllm import LLM, SamplingParams
    from activation.vllm_residuals import engine_overrides
    manifest, items = load_items(a.ref)
    layers = sorted(int(layer) for layer in manifest["layers"])
    kwargs = dict(model="Qwen/Qwen3.5-9B", dtype="bfloat16", enable_prompt_embeds=True, kv_cache_dtype="auto", max_model_len=16384,
                  gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0, "video": 0}, max_num_seqs=128, enable_prefix_caching=True,
                  seed=0, max_logprobs=20)
    if a.enable_lora:
        kwargs.update(enable_lora=True, max_loras=2, max_lora_rank=128)
    if a.eager:
        kwargs["enforce_eager"] = True
    tag = a.engine + ("_eager" if a.eager else "")
    if a.engine == "res":
        kwargs["hf_overrides"] = engine_overrides(HIDDEN, layers)
    started = time.perf_counter()
    llm = LLM(**kwargs)
    report = {"engine": a.engine, "kwargs": {k: v for k, v in kwargs.items()}, "load_seconds": round(time.perf_counter() - started, 1)}
    config = llm.llm_engine.vllm_config
    report["arch"] = config.model_config.architectures
    report["inputs_embeds_size"] = config.model_config.get_inputs_embeds_size()
    report["block_size"] = config.cache_config.block_size
    report["mamba_cache_mode"] = config.cache_config.mamba_cache_mode
    report["max_num_batched_tokens"] = config.scheduler_config.max_num_batched_tokens
    report["cudagraph_mode"] = str(config.compilation_config.cudagraph_mode)
    print(f"vllm_run[{tag}]: {json.dumps(report)}", flush=True)

    greedy = SamplingParams(temperature=0.0, max_tokens=manifest["max_new"], logprobs=20, stop_token_ids=manifest["eos_ids"])
    equiv = {}
    modes = ["rows"] if a.engine == "stock" else ["zero", "res"]
    for mode in modes:
        equiv[mode] = []
        for item in items:
            rows = item["row_embeds"] if mode == "rows" else packed(item, layers, zero=(mode == "zero"))
            equiv[mode].append(run_one(llm, prompt(item["ids"], item["positions"], rows), greedy))
        print(f"vllm_run[{tag}]: equivalence {mode} done", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(equiv, a.out / f"{tag}_equiv.pt")

    if a.engine == "res":                                    # prefix caching: hits on a repeat, misses on changed deltas
        long = max(items, key=lambda item: len(item["positions"]))
        quarter = len(long["positions"]) * 3 // 4
        one = SamplingParams(temperature=0.0, max_tokens=1)
        cache = {"prompt_tokens": len(long["ids"]), "rows": len(long["positions"]), "first_row_position": long["positions"][0],
                 "changed_from_position": long["positions"][quarter]}
        llm.reset_prefix_cache()

        def ask(rows):
            return llm.generate([prompt(long["ids"], long["positions"], rows)], one, use_tqdm=False)[0].num_cached_tokens
        for name, rows in (("cold", packed(long, layers)), ("repeat", packed(long, layers)),
                           ("deltas_changed_all", packed(long, layers, scale=1.01)),
                           ("deltas_changed_last_quarter", packed(long, layers, scale=1.02, from_row=quarter)),
                           ("repeat_original_again", packed(long, layers))):
            cache[name] = ask(rows)
        # Hybrid Qwen3.5 keeps its recurrent state only at the cached boundary of each prompt (mamba "align"): a hit is all of
        # that boundary or nothing. Deltas changed only past the boundary keep the hit; one changed block before it loses it.
        boundary, block = cache["repeat"], report["block_size"]
        after = next(index for index, position in enumerate(long["positions"]) if position >= boundary)
        inside = next(index for index, position in enumerate(long["positions"]) if position >= boundary - block)
        cache["cached_boundary"] = boundary
        cache["deltas_changed_only_after_boundary"] = ask(packed(long, layers, scale=1.02, from_row=after))
        cache["deltas_changed_only_in_last_block_before_boundary"] = ask(packed(long, layers, scale=1.02, from_row=inside, to_row=after))
        cache["rows_changed_deltas_kept"] = ask(packed(dict(long, row_embeds=(long["row_embeds"].float() * 1.01).to(torch.bfloat16)), layers))
        report["cache"] = cache
        print(f"vllm_run[{tag}]: cache {json.dumps(cache)}", flush=True)

    if not a.skip_throughput:
        requests = synthetic_requests(items, layers, a.engine)
        payload = requests[0]["prompt_embeds"]
        t0 = time.perf_counter(); hashlib.sha256(payload.view(torch.uint8).numpy()).digest(); sha_seconds = time.perf_counter() - t0
        timings = {}
        try:                                                 # the engine-core IPC path for one payload, on the CPU
            from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
            t0 = time.perf_counter(); frames = MsgpackEncoder().encode(payload); timings["cpu_msgpack_encode_seconds"] = round(time.perf_counter() - t0, 4)
            t0 = time.perf_counter(); joined = [bytes(frame) for frame in frames]; timings["cpu_frame_copy_seconds"] = round(time.perf_counter() - t0, 4)
            t0 = time.perf_counter(); MsgpackDecoder(torch.Tensor).decode(joined); timings["cpu_msgpack_decode_seconds"] = round(time.perf_counter() - t0, 4)
            del frames, joined
        except Exception as error:                           # noqa: BLE001 - a diagnostic only
            timings["cpu_msgpack_error"] = repr(error)
        throughput = {"requests": len(requests), "prompt_tokens": len(requests[0]["prompt_token_ids"]),
                      "rows": requests[0]["prompt_is_token_ids"].count(False), "payload_width": payload.shape[1],
                      "payload_bytes_per_request": payload.numel() * payload.element_size(),
                      "cpu_sha256_seconds_per_request": round(sha_seconds, 4), **timings}
        gc.collect()
        timed_pass(llm, requests[:2], SamplingParams(temperature=0.0, max_tokens=4))            # warm-up
        for name, params in (("prefill_only", SamplingParams(temperature=0.0, max_tokens=1)),
                             ("end_to_end_256", SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True))):
            seconds, generated, cached = timed_pass(llm, requests, params)
            throughput[name] = {"seconds": round(seconds, 2), "generated_tokens": generated, "cached_tokens_max": max(cached),
                                "requests_per_second": round(len(requests) / seconds, 3),
                                "prompt_tokens_per_second": round(len(requests) * throughput["prompt_tokens"] / seconds),
                                "generated_tokens_per_second": round(generated / seconds, 1)}
            print(f"vllm_run[{tag}]: throughput {name} {json.dumps(throughput[name])}", flush=True)
        report["throughput"] = throughput
    (a.out / f"{tag}.json").write_text(json.dumps(report, indent=1))
    print(f"vllm_run[{tag}]: done", flush=True)


if __name__ == "__main__":
    main()
