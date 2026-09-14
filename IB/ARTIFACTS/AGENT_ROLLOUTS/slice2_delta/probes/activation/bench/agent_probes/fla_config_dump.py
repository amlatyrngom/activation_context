"""
Tune fla's kernels once on this GPU type and write the config files (see agent_training/fla_cache.py):
one training pass over long, median and a synthetic 52k row so every kernel and length bucket is seen,
then `{kernel_name}.json` under FLA_CONFIGS/<gpu> in the synced root (copy into
activation/agent_training/fla_configs/<gpu>/ to check them in).
"""
import argparse
import json
import time

import torch

from activation.agent_training import AgentTrainer, AgentTrainingConfig
from activation.agent_training.agent_training_utils import checkpointing_min_tokens, collate, disable_dropout, micro_batches, set_checkpointing
from activation.agent_training.fla_cache import dump_fla_configs
from activation.bench.agent_probes.training_profile import LORA, MODEL_NAME, load_examples, synthetic_example
from activation.common.data_syncing import resolve_path
from activation.harness import TARGET_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-file", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-tokens", type=int, default=52_000)
    parser.add_argument("--synthetic", type=int, default=52_000)
    args = parser.parse_args()
    from fla.ops.utils.cache import get_gpu_info

    _, _, examples = load_examples(None, MODEL_NAME, LORA, args.max_tokens, args.cache_file)
    order = micro_batches(examples, 0, pad_free=True)
    step = max(1, len(order) // 12)
    groups = order[::step][:12] + [[len(examples)]]                                  # spread over the lengths, plus the synthetic row
    examples = examples + [synthetic_example(examples, args.synthetic)]
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
    min_tokens = checkpointing_min_tokens(trainer.config, device)
    print(f"gpu {get_gpu_info()}; tuning over rows {[len(examples[g[0]].token_ids) for g in groups]}", flush=True)
    for group in groups:
        batch = collate(examples, group, pad_token_id, device)
        set_checkpointing(base, trainer.config, int(batch.input_ids.numel()), min_tokens)
        started = time.time()
        loss, _, _ = trainer._objective(loaded, LORA, batch, 1)
        loss.backward()
        torch.cuda.synchronize()
        for p in base.parameters():
            p.grad = None
        print(f"[tuned] {int(batch.input_ids.numel())} tokens in {time.time() - started:.1f}s", flush=True)
    out = resolve_path("FLA_CONFIGS") / get_gpu_info()
    written = dump_fla_configs(out)
    print(f"wrote {out}: {json.dumps(written)}", flush=True)


if __name__ == "__main__":
    main()
