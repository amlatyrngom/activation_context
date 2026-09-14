"""
Does the sleeping engine slow the trainer? The key test's trainer reached 3,160 tok/s where the profile
probe (no engine in the process) reached 4,250 to 6,300 on the same rows. This probe times the same
training micro-batches in one process: before the engine exists, with the engine built, used and put
to sleep (the test's state at training time), and after the engine is shut down. Alongside each pass
a pure-Python loop is timed (GIL contention shows there directly) and the live thread count printed.

usage: python -m activation.bench.agent_probes.engine_contention_probe --cache-file TMP/profile_cache/agent_training_test_r0/rollouts.jsonl
"""
import argparse
import json
import threading
import time

import torch

from activation.agent_training import AgentTrainer, AgentTrainingConfig
from activation.agent_training.agent_training_utils import checkpointing_min_tokens, collate, disable_dropout, micro_batches, set_checkpointing
from activation.bench.agent_probes.training_profile import LORA, MODEL_NAME, load_examples
from activation.common.data_syncing import resolve_path
from activation.agent_training.fla_cache import configure_fla_cache
from activation.harness import FREE_DEVICE, SOURCE_DEVICE, TARGET_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig


def python_loop_seconds(n: int = 20_000_000) -> float:
    started = time.time()
    total = 0
    for i in range(n):
        total += i
    return time.time() - started


def timed_pass(label, harness, loaded, examples, groups, pad_token_id):
    loaded.model_to_device(TARGET_DEVICE)
    harness.module_manager.ensure_lora(LORA)
    base = loaded.model
    base.train()
    disable_dropout(base)
    device = next(base.parameters()).device
    trainer = AgentTrainer(harness, AgentTrainingConfig())
    print(f"fla configs: {configure_fla_cache(device)}", flush=True)
    min_tokens = checkpointing_min_tokens(trainer.config, device)
    parameters = [p for p in base.parameters() if p.requires_grad]
    torch.cuda.reset_peak_memory_stats(device)
    times, tokens = [], 0
    for index, group in enumerate(groups):
        batch = collate(examples, group, pad_token_id, device)
        set_checkpointing(base, trainer.config, int(batch.input_ids.numel()), min_tokens)
        torch.cuda.synchronize()
        started = time.time()
        loss, _, _ = trainer._objective(loaded, LORA, batch, len(group))
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.time() - started
        for p in parameters:
            p.grad = None
        if index == 0:
            continue                                                    # warm-up
        times.append(elapsed)
        tokens += int(batch.attention_mask.sum())
    base.eval()
    result = {
        "tokens_per_s": round(tokens / sum(times), 1), "seconds_per_micro_batch": [round(t, 2) for t in times],
        "python_loop_s": round(python_loop_seconds(), 2), "threads": threading.active_count(),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 1),
    }
    print(f"[{label}] {json.dumps(result)}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-file", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--micro-batches", type=int, default=8, help="timed micro-batches per pass: half the longest rows, half around the median")
    parser.add_argument("--engine-requests", type=int, default=32, help="chat requests through the engine before it sleeps")
    args = parser.parse_args()

    _, _, examples = load_examples(None, MODEL_NAME, LORA, args.max_tokens, args.cache_file)
    order = micro_batches(examples, 0, pad_free=True)                     # longest first, one example each
    half = args.micro_batches // 2
    middle = len(order) // 2
    groups = order[:1] + order[1:1 + half] + order[middle:middle + args.micro_batches - half]
    print(f"{len(examples)} examples; micro-batches {[len(examples[g[0]].token_ids) for g in groups]} (first is warm-up)", flush=True)

    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={MODEL_NAME: ModelConfig(MODEL_NAME, args.model)}, agent_max_concurrent=64))
    harness.module_manager.register_lora(LORA, MODEL_NAME, rank=64, dropout=0.0)
    loaded = harness.loaded_models[MODEL_NAME]
    pad_token_id = loaded.tokenizer.pad_token_id if loaded.tokenizer.pad_token_id is not None else loaded.tokenizer.eos_token_id
    results = {}
    print(f"threads at start: {threading.active_count()}; python loop {python_loop_seconds():.2f}s", flush=True)
    results["no_engine_1"] = timed_pass("no_engine_1", harness, loaded, examples, groups, pad_token_id)
    results["no_engine_2"] = timed_pass("no_engine_2", harness, loaded, examples, groups, pad_token_id)

    # The engine as the test has it: built, used for a batch of requests, then asleep for training.
    loaded.model_to_device(SOURCE_DEVICE)
    loaded.engine_to_device(TARGET_DEVICE)
    conversations = [[{"role": "user", "content": f"Compute {i} * {i + 1} and answer with the number only."}] for i in range(args.engine_requests)]
    outputs = loaded.engine_chat_many(conversations, chat_kwargs={"sampling_params": {"max_tokens": 64, "temperature": 0.0}})
    print(f"engine answered {len(outputs)} requests; threads {threading.active_count()}", flush=True)
    loaded.engine_to_device(SOURCE_DEVICE)
    torch.cuda.empty_cache()
    print(f"engine asleep; threads {threading.active_count()}; python loop {python_loop_seconds():.2f}s", flush=True)
    results["engine_asleep_1"] = timed_pass("engine_asleep_1", harness, loaded, examples, groups, pad_token_id)
    results["engine_asleep_2"] = timed_pass("engine_asleep_2", harness, loaded, examples, groups, pad_token_id)

    loaded.model_to_device(SOURCE_DEVICE)
    loaded.engine_to_device(FREE_DEVICE)
    torch.cuda.empty_cache()
    print(f"engine freed; threads {threading.active_count()}", flush=True)
    results["engine_freed"] = timed_pass("engine_freed", harness, loaded, examples, groups, pad_token_id)

    out = resolve_path("TRAINING_PROFILE")
    (out / "engine_contention.json").write_text(json.dumps(results, indent=1))
    print("\n| pass | tok/s | python loop s | threads | peak GB |\n| --- | --- | --- | --- | --- |")
    for name, r in results.items():
        print(f"| {name} | {r['tokens_per_s']} | {r['python_loop_s']} | {r['threads']} | {r['peak_memory_gb']} |")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
