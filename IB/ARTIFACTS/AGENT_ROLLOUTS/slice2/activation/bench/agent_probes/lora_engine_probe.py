"""
LoRA-on-the-engine probe (one GPU, a few minutes): does the serving path do what training needs?
1. sample a fixed prompt at temperature 0 on the base (token ids, log-probs and the end-of-turn token recorded)
2. inject a fresh adapter (zero-init B), checkpoint it, exchange -> the engine must reproduce the base output
   (the model was loaded for the checkpoint, so this also exercises engine sleep -> train-side load -> wake)
3. perturb the adapter's B matrices, checkpoint, exchange -> the output must change
4. sleep / wake once more and sample again -> unchanged
Prints a JSON verdict and exits non-zero on a failed expectation.

  uv run sky exec <node> -- uv run python -m activation.bench.agent_probes.lora_engine_probe --model Qwen/Qwen3.5-4B
"""
import argparse
import json
import sys

import torch

from activation.agent.agent_utils import ModelDialect
from activation.harness import SOURCE_DEVICE, TARGET_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--lora", default="probe_lora")
    parser.add_argument("--rank", type=int, default=64)
    args = parser.parse_args()
    model_name = args.model.split("/")[-1].lower()
    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={model_name: ModelConfig(model_name, args.model)}))
    mm = harness.module_manager
    mm.register_lora(args.lora, model_name, rank=args.rank, dropout=0.0)
    loaded = harness.loaded_models[model_name]
    dialect = ModelDialect.for_tokenizer(loaded.tokenizer)
    messages = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "List the first ten primes, then say why 91 is not prime."}]
    chat_kwargs = {"sampling_params": {"max_tokens": 120, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "presence_penalty": 0.0}}
    prompt = dialect.prompt_token_ids(loaded.tokenizer, messages, None, loaded.chat_template_kwargs(chat_kwargs))

    def sample(tag):
        out = loaded.engine_submit_tokens(prompt, seed=0, agent_id="probe", lora_name=args.lora, chat_kwargs=chat_kwargs, record_sampling=True)
        print(f"[{tag}] lora={out.lora_name} finish={out.finish_reason} tokens={len(out.token_ids)} logprobs={None if out.logprobs is None else len(out.logprobs)}\n  {out.text[:160]!r}", flush=True)
        return out

    verdict = {}
    base = sample("base")
    verdict["logprobs_recorded"] = base.logprobs is not None and len(base.logprobs) == len(base.token_ids)
    verdict["eos_in_token_ids"] = bool(base.token_ids) and base.token_ids[-1] == loaded.tokenizer.eos_token_id and base.finish_reason == "stop"
    verdict["base_lora_none"] = base.lora_name is None

    # A zero adapter through the training side: engine sleeps, model loads, checkpoint written, model to host RAM.
    mm.ensure_lora(args.lora)
    verdict["engine_slept_for_training"] = loaded.current_engine_device == SOURCE_DEVICE
    zero_path = mm.save_lora(model_name, args.lora)
    loaded.model_to_device(SOURCE_DEVICE)
    mm.exchange_lora(model_name, args.lora)
    zero = sample("zero adapter")                                   # wakes the engine
    verdict["engine_woke"] = loaded.current_engine_device == TARGET_DEVICE
    verdict["zero_adapter_named"] = zero.lora_name == args.lora
    verdict["zero_adapter_matches_base"] = zero.token_ids == base.token_ids
    if base.logprobs and zero.logprobs and zero.token_ids == base.token_ids:
        verdict["zero_adapter_max_abs_logprob_diff"] = max(abs(a - b) for a, b in zip(base.logprobs, zero.logprobs))

    # A perturbed adapter must change the output.
    peft_model = mm.ensure_lora(args.lora)
    with torch.no_grad():
        for name, parameter in peft_model.named_parameters():
            if "lora_B" in name and parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * 0.003)   # small: the output must stay coherent, so temperature-0 sampling stays reproducible
    perturbed_path = mm.save_lora(model_name, args.lora)
    loaded.model_to_device(SOURCE_DEVICE)
    mm.exchange_lora(model_name, args.lora)
    perturbed = sample("perturbed adapter")
    common = min(len(perturbed.token_ids), len(base.token_ids))
    verdict["perturbed_adapter_differs"] = perturbed.token_ids != base.token_ids or (
        base.logprobs is not None and perturbed.logprobs is not None
        and max(abs(a - b) for a, b in zip(base.logprobs[:common], perturbed.logprobs[:common])) > 1e-3)
    verdict["engine_lora_request_id"] = mm.engine_lora_request(args.lora).lora_int_id

    # Sleep and wake without any training in between.
    loaded.engine_to_device(SOURCE_DEVICE)
    loaded.engine_to_device(TARGET_DEVICE)
    again = sample("after sleep/wake")
    verdict["sleep_wake_stable"] = again.token_ids == perturbed.token_ids
    if again.token_ids == perturbed.token_ids and again.logprobs and perturbed.logprobs:
        verdict["sleep_wake_max_abs_logprob_diff"] = max(abs(a - b) for a, b in zip(again.logprobs, perturbed.logprobs))
    verdict["checkpoints"] = [zero_path, perturbed_path]
    print(json.dumps(verdict, indent=1))
    failed = [key for key, value in verdict.items() if value is False]
    loaded.engine_to_device("disk")
    if failed:
        print(f"PROBE FAILED: {failed}")
        sys.exit(1)
    print("PROBE OK")


if __name__ == "__main__":
    main()
