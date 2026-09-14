"""
Where do the 15-second first-encounter stalls come from? Runs the cold single-row micro-batches of the
training step under torch.profiler with Triton's autotune/compile logging on, and prints the top CPU
self-time entries for every micro-batch slower than --stall-seconds.
"""
import os
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")
import argparse
import json
import time

import torch

from activation.agent_training import AgentTrainer, AgentTrainingConfig
from activation.agent_training.agent_training_utils import checkpointing_min_tokens, collate, disable_dropout, micro_batches, set_checkpointing
from activation.bench.agent_probes.training_profile import LORA, MODEL_NAME, load_examples
from activation.agent_training.fla_cache import configure_fla_cache
from activation.harness import TARGET_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-file", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--micro-batches", type=int, default=9)
    parser.add_argument("--stall-seconds", type=float, default=6.0)
    args = parser.parse_args()
    from torch.profiler import ProfilerActivity, profile

    _, _, examples = load_examples(None, MODEL_NAME, LORA, args.max_tokens, args.cache_file)
    groups = micro_batches(examples, 0, pad_free=True)[:args.micro_batches]
    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={MODEL_NAME: ModelConfig(MODEL_NAME, args.model)}))
    harness.module_manager.register_lora(LORA, MODEL_NAME, rank=64, dropout=0.0)
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
    min_tokens = checkpointing_min_tokens(trainer.config, device)
    print(f"cudnn.benchmark {torch.backends.cudnn.benchmark}; cudnn.deterministic {torch.backends.cudnn.deterministic}; "
          f"tf32 matmul {torch.backends.cuda.matmul.allow_tf32}", flush=True)
    for group in groups:
        batch = collate(examples, group, pad_token_id, device)
        n = int(batch.input_ids.numel())
        set_checkpointing(base, trainer.config, n, min_tokens)
        torch.cuda.synchronize()
        started = time.time()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            loss, _, _ = trainer._objective(loaded, LORA, batch, 1)
            loss.backward()
            torch.cuda.synchronize()
        elapsed = time.time() - started
        for p in base.parameters():
            p.grad = None
        loss_tokens = int(batch.loss_mask.sum())
        print(f"[micro-batch] tokens {n} loss_tokens {loss_tokens} chunks {-(-loss_tokens // 2048)} last_chunk {loss_tokens % 2048} "
              f"T%16 {n % 16} seconds {elapsed:.2f}", flush=True)
        if elapsed > args.stall_seconds:
            table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=14, max_name_column_width=80)
            print(table, flush=True)


if __name__ == "__main__":
    main()
