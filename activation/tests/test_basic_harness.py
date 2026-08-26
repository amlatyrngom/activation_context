import torch
from activation.harness import (
    HarnessRuntimeConfig,
    HarnessRuntime,
    ModelConfig,
    EmbeddingReadoutType,
)

# Model ID, Num Layers, D Model
BASIC_SMALL_MODELS = [
    ("Qwen/Qwen3.5-0.8B", 24, 1024),
    # ("Qwen/Qwen3-1.7B", 28, 2048),
    # ("google/gemma-3-1b-it", 26, 1152),
    # ("google/gemma-4-E2B-it", 35, 1536),
]


# Model ID, Num Layers, D Model, Readout Type
# Only use qwen for now.
BASIC_SMALL_EMBEDDING_MODELS = [
    # ("Qwen/Qwen3-Embedding-0.6B", 28, 1024, EmbeddingReadoutType.EOS_TOKEN, "<|endoftext|>"),
    ("Qwen/Qwen3-Embedding-4B", 36, 2560, EmbeddingReadoutType.EOS_TOKEN, "<|endoftext|>"),
]

def test_basic_model_loading():
    """Load diverse small models and make sure they work well."""
    model_configs: dict[str, ModelConfig] = dict()
    for model_id, _, _ in BASIC_SMALL_MODELS:
        model_configs[model_id] = ModelConfig(
            model_name=model_id,
            model_id=model_id,
        )
        assert model_configs[model_id].model_description is None, "Model description is empty by default."
    harness_config = HarnessRuntimeConfig(
        model_configs=model_configs
    )
    harness = HarnessRuntime(harness_config)
    for model_id, expected_num_layers, expected_d_model in BASIC_SMALL_MODELS:
        model_config = model_configs[model_id]
        print(f"--------------- Model {model_id} ----------------")
        print(model_config.pretty_format_description())
        print(f"--------------------------------------------------")
        assert model_config.model_description is not None, "Model description should be populated."
        assert model_config.model_description.d_model == expected_d_model
        assert len(model_config.model_description.layer_descriptions) == expected_num_layers
        response = harness.simple_chat(model_config.model_name, "Say: 'Hello, World'")
        assert 'hello' in response.lower()
        print(f"Generated Response: {response}")



def test_basic_embedding_models():
    text1 = "planet earth is blue and green. it is the home of mankind."
    text2 = "planet terra is blue and green. it is the home of men."
    expected_max_cosine_distance_delta = 0.2
    model_configs: dict[str, ModelConfig] = dict()
    for (model_id, _, _, _, _) in BASIC_SMALL_EMBEDDING_MODELS:
        model_configs[model_id] = ModelConfig(
            model_name=model_id,
            model_id=model_id,
        )
        assert model_configs[model_id].model_description is None, "Model description is empty by default."
    harness_config = HarnessRuntimeConfig(
        model_configs = model_configs
    )
    harness = HarnessRuntime(harness_config)
    for (model_id, expected_num_layers, expected_d_model, expected_readout_type, expected_readout_token) in BASIC_SMALL_EMBEDDING_MODELS:
        model_config = model_configs[model_id]
        # Check description.
        description = model_config.model_description
        assert description is not None
        assert len(description.layer_descriptions) == expected_num_layers
        assert description.d_model == expected_d_model
        assert description.is_embedding_model
        assert description.embedding_readout_type is expected_readout_type
        if expected_readout_type == EmbeddingReadoutType.EOS_TOKEN:
            assert description.eos_token == expected_readout_token
        elif expected_readout_type == EmbeddingReadoutType.EOT_TOKEN:
            assert description.eot_token == expected_readout_type
        # Live tests.
        embedding1 = harness.simple_embed(model_id, text1)
        embedding2 = harness.simple_embed(model_id, text2)
        # Expected shape, normalization, and closeness.
        assert embedding1.shape == (expected_d_model,)
        assert embedding2.shape == (expected_d_model,)
        assert torch.allclose(embedding1.norm(), embedding1.new_tensor(1.0), atol=1e-2)
        assert torch.allclose(embedding2.norm(), embedding2.new_tensor(1.0), atol=1e-2)
        cosine_distance = 1.0 - torch.dot(embedding1, embedding2)
        assert cosine_distance.item() <= expected_max_cosine_distance_delta
