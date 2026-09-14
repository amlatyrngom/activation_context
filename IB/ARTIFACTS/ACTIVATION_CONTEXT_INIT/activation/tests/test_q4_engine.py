"""
Probe: Q4 Qwen3.8-27B variants through the engine-only LoadedModel path.
"""
import pytest

from activation.harness import (
    HarnessRuntimeConfig,
    HarnessRuntime,
    ModelConfig,
    FREE_DEVICE,
)

pytestmark = pytest.mark.gpu

Q4_MODELS = [
    # compressed-tensors pack-quantized W4A16; vision tower stays bf16 (~21GB total).
    ("cyankiwi/Qwen3.8-27B-AWQ-INT4", {}),
    # GPTQ 4-bit (gptq_marlin path), second-quantizer alternative.
    # NOTE: unsloth Q4 artifacts (bnb-4bit, GGUF) cannot load: vllm 0.28
    # removed both the bitsandbytes and gguf quantization paths upstream.
    # NOTE: amd/...-Quark-AWQ-INT4-W4A16 hits a vllm 0.28 quark-loader bug
    # (dict passed to _map_name_with_shard) at engine init.
    ("btbtyler09/Qwen3.8-27B-GPTQ-4bit", {}),
]

TEST_MESSAGES = [
    {
        "role": "system", "content": [
            {"type": "text", "text": "You are a helpful assistant"},
        ]
    },
    {
        "role": "user", "content": [
            {"type": "text", "text": "Say: 'Hello, World!'"},
        ]
    }
]


def test_q4_engine():
    model_configs = {
        model_id: ModelConfig(
            model_name=model_id,
            model_id=model_id,
            engine_kwargs={
                # The native 262k context would blow the KV-cache budget.
                "max_model_len": 8192,
                # Each decode seq needs one Mamba cache block for the
                # linear-attention layers; the default 256 overflows 48GB.
                "max_num_seqs": 64,
                **extra_engine_args,
            },
        )
        for model_id, extra_engine_args in Q4_MODELS
    }
    harness_config = HarnessRuntimeConfig(model_configs=model_configs)
    harness = HarnessRuntime(harness_config)
    for model_config in model_configs.values():
        loaded_model = harness.loaded_models[model_config.model_name]
        try:
            response = loaded_model.simple_engine_chat(TEST_MESSAGES, lora_name=None)
            print(f"{model_config.model_name} - Q4 engine response: {response!r}")
            assert "hello" in response.lower()
        finally:
            loaded_model.engine_to_device(FREE_DEVICE)
