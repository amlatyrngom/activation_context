"""
Speed A/B probe: official FP8 vs NVFP4 Qwen3.8-27B, default vs recommended
engine configs. Prints one table; not a benchmark suite.
Run: uv run python activation/tests/probe_speed_ab.py
"""
import time
import traceback

from vllm import SamplingParams

from activation.harness import (
    HarnessRuntimeConfig,
    HarnessRuntime,
    ModelConfig,
    FREE_DEVICE,
)
from activation.harness.hf_utils import TARGET_DEVICE

LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. "
)
BATCH = 8
PREFILL_CHARS = 6000  # "artificially larger" prompts for the prefill side.
DECODE_CHARS = 100
DECODE_TOKENS = 128
ROUNDS = 3

BASE_ARGS = {
    "max_model_len": 8192,
    "max_num_seqs": 64,  # Hybrid: one Mamba cache block per decode sequence.
    "limit_mm_per_prompt": {"image": 0, "video": 0},
}
def rec_args(num_speculative_tokens: int) -> dict:
    # Per https://unsloth.ai/docs/models/qwen3.8: MTP spec decode + fp8 KV cache.
    return {
        **BASE_ARGS,
        "kv_cache_dtype": "fp8",
        "gpu_memory_utilization": 0.95,
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
        },
        "enable_lora": False,  # Spec decode and LoRA don't need to coexist here.
    }


ARMS = [
    ("fp8", "Qwen/Qwen3.8-27B-FP8", BASE_ARGS),
    ("fp8+rec1", "Qwen/Qwen3.8-27B-FP8", rec_args(1)),
    ("fp8+rec2", "Qwen/Qwen3.8-27B-FP8", rec_args(2)),
    ("fp4", "unsloth/Qwen3.8-27B-NVFP4", BASE_ARGS),
    ("fp4+rec1", "unsloth/Qwen3.8-27B-NVFP4", rec_args(1)),
    ("fp4+rec2", "unsloth/Qwen3.8-27B-NVFP4", rec_args(2)),
]

_uniq = 0


def make_prompts(chars: int) -> list[str]:
    # Unique prefixes defeat prefix caching so every run does real work.
    global _uniq
    _uniq += 1
    body = (LOREM * (chars // len(LOREM) + 2))[:chars]
    return [f"[probe {_uniq}-{i}] {body}" for i in range(BATCH)]


def timed_generate(llm, prompts: list[str], max_tokens: int):
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return elapsed, prompt_tokens, output_tokens


def measure_arm(llm) -> tuple[float, float]:
    # Warmup: absorb compile/capture for both shapes.
    timed_generate(llm, make_prompts(PREFILL_CHARS), 8)
    timed_generate(llm, make_prompts(DECODE_CHARS), 32)
    prefill_rates = []
    decode_rates = []
    for _ in range(ROUNDS):
        elapsed, prompt_tokens, _ = timed_generate(llm, make_prompts(PREFILL_CHARS), 1)
        prefill_rates.append(prompt_tokens / elapsed)
        # Decode rate via subtraction: (prefill+N tokens) minus (prefill+1 token).
        e1, _, o1 = timed_generate(llm, make_prompts(DECODE_CHARS), 1)
        e2, _, o2 = timed_generate(llm, make_prompts(DECODE_CHARS), DECODE_TOKENS + 1)
        decode_rates.append((o2 - o1) / (e2 - e1))
    return (
        sum(prefill_rates) / len(prefill_rates),
        sum(decode_rates) / len(decode_rates),
    )


def main():
    import sys
    arms = ARMS
    if len(sys.argv) > 1:  # Optional comma-separated arm filter.
        wanted = set(sys.argv[1].split(","))
        arms = [arm for arm in ARMS if arm[0] in wanted]
    rows = []
    for arm_name, model_id, engine_args in arms:
        print(f"=== {arm_name}: {model_id} {engine_args}", flush=True)
        model_config = ModelConfig(
            model_name=arm_name,
            model_id=model_id,
            engine_kwargs=dict(engine_args),
        )
        harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={arm_name: model_config}))
        loaded_model = harness.loaded_models[arm_name]
        try:
            loaded_model.engine_to_device(TARGET_DEVICE)
            prefill_rate, decode_rate = measure_arm(loaded_model.vllm_model)
            rows.append((arm_name, f"{prefill_rate:,.0f}", f"{decode_rate:,.1f}"))
            print(f"=== {arm_name}: prefill {prefill_rate:,.0f} tok/s, decode {decode_rate:,.1f} tok/s", flush=True)
        except Exception:
            traceback.print_exc()
            rows.append((arm_name, "FAILED", "FAILED"))
        finally:
            loaded_model.engine_to_device(FREE_DEVICE)

    print("\nRESULTS (batch=8, greedy, ignore_eos)")
    print(f"{'arm':<10} {'prefill tok/s':>14} {'decode tok/s':>14}")
    for arm_name, prefill_rate, decode_rate in rows:
        print(f"{arm_name:<10} {prefill_rate:>14} {decode_rate:>14}")


if __name__ == "__main__":
    main()
