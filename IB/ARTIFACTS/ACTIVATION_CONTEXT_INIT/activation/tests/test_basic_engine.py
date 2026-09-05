from activation.harness import (
    HarnessRuntimeConfig,
    HarnessRuntime,
    ModelConfig,
    FREE_DEVICE,
)
BASIC_SMALL_MODELS = [
    ("Qwen/Qwen3.5-0.8B", 24, 1024),
    # ("Qwen/Qwen3-1.7B", 28, 2048),
    # ("google/gemma-3-1b-it", 26, 1152),
    # ("google/gemma-4-E2B-it", 35, 1536),
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

def test_basic_engine():
    model_configs = {
        model_id: ModelConfig(model_name=model_id, model_id=model_id)
        for model_id, _, _ in BASIC_SMALL_MODELS
    }
    harness_config = HarnessRuntimeConfig(model_configs=model_configs)
    harness = HarnessRuntime(harness_config)
    for model_config in model_configs.values():
        loaded_model = harness.loaded_models[model_config.model_name]
        try:
            response = loaded_model.simple_engine_chat(TEST_MESSAGES, lora_name=None) 
            assert "hello" in response.lower()       
        finally:
            loaded_model.engine_to_device(FREE_DEVICE)

