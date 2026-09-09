"""
Slice 3b M0 payload probe (IB-only, not product code): the unpatched cost of shipping activation-context
rows as prompt embeddings through the harness engine interface.

For prompt lengths L and concurrencies C it submits `max_tokens=1` requests through
`LoadedModel.engine_submit_tokens` in three forms: token-only (control), a full-length zero tensor with
one nonzero span (the mixed prompt the agent ships), and the same mixed prompt a second time (prefix-cache
hit). It records wall time per request, aggregate throughput, `cached_prompt_token_count`, and host RSS.

    uv run python IB/ARTIFACTS/AGENT_ROLLOUTS/slice3b/probes/ac_payload_probe.py --model Qwen/Qwen3.5-4B --out probe.json

Gate (plan M0): engine-side overhead above ~20% of a typical turn at 32 agents, or host memory pressure,
triggers the vLLM plugin relaxations (short tensor to the last part; hash only AC-bearing blocks).
"""
import argparse
import json
import os
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch

sys.modules.setdefault("fla", None)
from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--lengths", default="8000,16000,32000,50000")
    parser.add_argument("--concurrency", default="1,8,32,64")
    parser.add_argument("--rows", type=int, default=2048, help="rows in the nonzero span")
    parser.add_argument("--out", default="ac_payload_probe.json")
    parser.add_argument("--max-tokens", type=int, default=1, help="sampled tokens per request (1 = prefill cost only)")
    args = parser.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    concurrencies = [int(x) for x in args.concurrency.split(",")]
    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={"probe": ModelConfig("probe", args.model)}))
    loaded = harness.loaded_models["probe"]
    loaded.model_config.engine_kwargs = dict(loaded.model_config.engine_kwargs or {}) | {"enable_prompt_embeds": True}
    loaded.ensure_engine_loaded()
    tokenizer = loaded.tokenizer
    d_model = loaded.model_config.model_description.d_model
    vocab = [i for i in tokenizer.encode("the quick brown fox jumps over the lazy dog " * 8, add_special_tokens=False)]
    chat = {"sampling_params": {"max_tokens": args.max_tokens, "temperature": 0.0}}
    records = []

    def request(agent_index: int, length: int, mixed: bool, salt: int):
        ids = [vocab[(i * 7 + salt + agent_index * 13) % len(vocab)] for i in range(length)]
        embeds = mask = None
        if mixed:
            embeds = torch.zeros(length, d_model, dtype=torch.bfloat16)
            start = length // 3
            embeds[start:start + args.rows] = 0.01
            mask = [True] * length
            mask[start:start + args.rows] = [False] * args.rows
        started = time.time()
        output = loaded.engine_submit_tokens(ids, seed=agent_index, agent_id=f"probe-{agent_index}", chat_kwargs=chat,
                                             prompt_embeds=embeds, prompt_is_token_ids=mask)
        return time.time() - started, output.cached_prompt_token_count

    for length in lengths:
        for concurrency in concurrencies:
            for form in ("tokens", "mixed", "mixed_repeat"):
                salt = 1000 * length + concurrency if form != "mixed_repeat" else 1000 * length + concurrency   # the repeat reuses the mixed prompt
                if form == "tokens":
                    salt += 1
                rss_before = rss_mb()
                started = time.time()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    results = list(pool.map(lambda i: request(i, length, form != "tokens", salt), range(concurrency)))
                elapsed = time.time() - started
                latencies = sorted(r[0] for r in results)
                record = {
                    "length": length, "concurrency": concurrency, "form": form, "wall_seconds": round(elapsed, 3),
                    "latency_median": round(latencies[len(latencies) // 2], 3), "latency_max": round(latencies[-1], 3),
                    "cached_tokens_mean": round(sum(r[1] for r in results) / len(results), 1),
                    "rss_mb_before": round(rss_before, 1), "rss_mb_after": round(rss_mb(), 1),
                }
                records.append(record)
                print(json.dumps(record), flush=True)
    with open(args.out, "w") as handle:
        json.dump({"model": args.model, "rows": args.rows, "max_tokens": args.max_tokens, "records": records}, handle, indent=1)
    print(f"wrote {args.out}")
    loaded.engine_to_device(FREE_DEVICE)


if __name__ == "__main__":
    main()
