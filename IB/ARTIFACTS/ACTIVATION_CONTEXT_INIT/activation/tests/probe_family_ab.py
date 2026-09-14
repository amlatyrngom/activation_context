"""
Family A/B probe: FP8 vs NVFP4 across Qwen3.8-27B, Nemotron 3.5 Lightning
30B-A3B, and Qwen3.6-35B-A3B, non-reasoning mode. Prints a speed table
(prefill/decode, probe_speed_ab methodology) plus 4 seeded study QA samples per
arm on identical BRIGHT-biology chunks for qualitative judgment.
Run (RTX/Blackwell node): uv run python activation/tests/probe_family_ab.py [arm,arm]
"""
import json
import time
import traceback

from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from activation.dataset.dataset_study import STUDY_JSON_SCHEMA, make_study_prompt
from activation.dataset.loaders import BrightDataset
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
PREFILL_CHARS = 6000
DECODE_CHARS = 100
DECODE_TOKENS = 128
ROUNDS = 2  # Keep the probe small; variance was low in the previous A/B.

STUDY_SEED = 0
N_STUDY_CHUNKS = 4
EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

MM_OFF = {"limit_mm_per_prompt": {"image": 0, "video": 0}}


def formula_args(multimodal: bool) -> dict:
    # The good formula from the quantized-engine benchmarks.
    return {
        "max_model_len": 8192,
        "max_num_seqs": 64,
        "kv_cache_dtype": "fp8",
        "gpu_memory_utilization": 0.9,
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 2},
        "enable_lora": False,
        **(MM_OFF if multimodal else {}),
    }


def base_args(multimodal: bool) -> dict:
    # Fallback when a family rejects part of the formula (kv fp8 / MTP).
    return {
        "max_model_len": 8192,
        "max_num_seqs": 64,
        "enable_lora": False,
        **(MM_OFF if multimodal else {}),
    }


# (arm, model_id, multimodal) - engine kwargs attempts are formula then base.
ARMS = [
    ("qwen3.8-fp8", "Qwen/Qwen3.8-27B-FP8", True),
    ("qwen3.8-fp4", "unsloth/Qwen3.8-27B-NVFP4", True),
    ("nemotron-fp8", "RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8", False),
    ("nemotron-fp4", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", False),
    ("qwen3.6moe-fp8", "Qwen/Qwen3.6-35B-A3B-FP8", True),
    ("qwen3.6moe-fp4", "nvidia/Qwen3.6-35B-A3B-NVFP4", True),
]

_uniq = 0


def make_prompts(chars: int) -> list[str]:
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
    timed_generate(llm, make_prompts(PREFILL_CHARS), 8)
    timed_generate(llm, make_prompts(DECODE_CHARS), 32)
    prefill_rates = []
    decode_rates = []
    for _ in range(ROUNDS):
        elapsed, prompt_tokens, _ = timed_generate(llm, make_prompts(PREFILL_CHARS), 1)
        prefill_rates.append(prompt_tokens / elapsed)
        e1, _, o1 = timed_generate(llm, make_prompts(DECODE_CHARS), 1)
        e2, _, o2 = timed_generate(llm, make_prompts(DECODE_CHARS), DECODE_TOKENS + 1)
        decode_rates.append((o2 - o1) / (e2 - e1))
    return (
        sum(prefill_rates) / len(prefill_rates),
        sum(decode_rates) / len(decode_rates),
    )


def build_study_conversations() -> list[tuple[str, list]]:
    """Seeded, model-independent (chunk_id, conversation) pairs, built once."""
    import random

    harness = HarnessRuntime(HarnessRuntimeConfig(
        model_configs={
            EMBEDDING_MODEL_ID: ModelConfig(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_ID),
        },
        doc_embedding_model_name=EMBEDDING_MODEL_ID,
    ))
    dataset = BrightDataset.load(
        harness, max_examples=20, domain="biology", max_corpus_documents=100,
    )
    harness.dataset_manager.build_dense_indexes()
    index = harness.dataset_manager.dataset_indexes[dataset.dataset_id]
    harness.loaded_models[EMBEDDING_MODEL_ID].model_to_device(FREE_DEVICE)

    rng = random.Random(STUDY_SEED)
    chunks = [index.chunks[rng.choice(index.chunk_ids)] for _ in range(N_STUDY_CHUNKS)]
    system_prompt, instructions, _ = make_study_prompt(dataset.study_context)
    pairs = []
    for chunk in chunks:
        snippet = index.get_chunk_section(chunk)
        pairs.append((chunk.chunk_id, [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": f"# Corpus Snippet\n```\n{snippet}\n```\n{instructions}"}]},
        ]))
    return pairs


def run_study(loaded_model, pairs) -> tuple[list[tuple[str, str, str]], float, float]:
    """Returns (qa list, latency, output tok/s) for the fixed seeded inputs."""
    chat_kwargs = {
        "sampling_params": {
            # Pipeline sampling (Qwen instruct recommendation) held constant
            # across arms so differences are the models, not the sampler.
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
            "max_tokens": 2048,
            "seed": STUDY_SEED,
            "structured_outputs": StructuredOutputsParams(json=STUDY_JSON_SCHEMA),
        },
    }
    start = time.perf_counter()
    outputs = loaded_model.engine_chat_many(
        [conversation for _, conversation in pairs], chat_kwargs=chat_kwargs,
    )
    elapsed = time.perf_counter() - start
    output_tokens = sum(o.output_token_count for o in outputs)
    qa_list = []
    for (chunk_id, _), output in zip(pairs, outputs):
        try:
            qa_pair = json.loads(output.text)
            qa_list.append((
                chunk_id,
                str(qa_pair["question"]).strip(),
                str(qa_pair["answer"]).strip(),
            ))
        except (json.JSONDecodeError, KeyError, TypeError):
            qa_list.append((chunk_id, "<PARSE FAILURE>", output.text[:200]))
            continue
    return qa_list, elapsed, output_tokens / elapsed


def main():
    import sys
    arms = ARMS
    if len(sys.argv) > 1:
        wanted = set(sys.argv[1].split(","))
        arms = [arm for arm in ARMS if arm[0] in wanted]

    pairs = build_study_conversations()
    print(f"=== study inputs: {[chunk_id for chunk_id, _ in pairs]}", flush=True)

    rows = []
    for arm_name, model_id, multimodal in arms:
        result = None
        for config_name, engine_args in [
            ("formula", formula_args(multimodal)),
            ("base", base_args(multimodal)),
        ]:
            print(f"=== {arm_name} [{config_name}]: {model_id} {engine_args}", flush=True)
            loaded_model = None
            try:
                model_config = ModelConfig(
                    model_name=arm_name, model_id=model_id, engine_kwargs=engine_args,
                )
                harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={arm_name: model_config}))
                loaded_model = harness.loaded_models[arm_name]
                loaded_model.engine_to_device(TARGET_DEVICE)
                prefill_rate, decode_rate = measure_arm(loaded_model.vllm_model)
                qa_list, study_latency, study_rate = run_study(loaded_model, pairs)
                result = (config_name, prefill_rate, decode_rate, study_latency, study_rate)
                print(f"\n=== {arm_name} [{config_name}]: prefill {prefill_rate:,.0f} tok/s, "
                      f"decode {decode_rate:,.1f} tok/s, study {study_latency:.1f}s "
                      f"({study_rate:,.1f} out tok/s), {len(qa_list)} QAs", flush=True)
                for chunk_id, question, answer in qa_list[:4]:
                    print(f"\n[{arm_name}] (chunk {chunk_id})")
                    print(f"  Q: {question}")
                    print(f"  A: {answer}")
                print("", flush=True)
            except Exception:
                traceback.print_exc()
            finally:
                if loaded_model is not None:
                    loaded_model.engine_to_device(FREE_DEVICE)
            if result:
                break
        rows.append((arm_name, *(result or ("FAILED", 0, 0, 0, 0))))

    print(f"\nRESULTS (batch={BATCH}, greedy speed probe; study: {N_STUDY_CHUNKS} seeded chunks, seed={STUDY_SEED})")
    print(f"{'arm':<16} {'config':<8} {'prefill tok/s':>14} {'decode tok/s':>14} {'study s':>9} {'study tok/s':>12}")
    for arm_name, config_name, prefill_rate, decode_rate, study_latency, study_rate in rows:
        print(f"{arm_name:<16} {config_name:<8} {prefill_rate:>14,.0f} {decode_rate:>14,.1f} "
              f"{study_latency:>9.1f} {study_rate:>12,.1f}")


if __name__ == "__main__":
    main()
