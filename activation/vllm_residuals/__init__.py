"""
Hidden Residuals ("deep AC inputs") on a vLLM 0.28 Qwen3.5 reader, without a vLLM fork.

The HF reference is `harness.hf_utils.install_intervention_hooks`: per request, a delta [M, d] per reader layer is added to the
hidden state entering that layer at the AC rows' prompt positions. vLLM already carries one prompt-embedding row per position
(into `GPUModelRunner.inputs_embeds`, hashed into the prefix-cache block keys), so the deltas ride along side by side:

    AC row position   [row, delta_L1, ..., delta_Lk]      (1 + k) * d wide, layers ascending (`pack_rows`)
    token position    [token embedding, 0, ..., 0]        (the model pads its own token embeddings)

`qwen3_5.Qwen3_5ResForConditionalGeneration` splits the wide rows and adds every delta to the input of its layer; decode
steps carry zero deltas. `engine_overrides` names that architecture and widens the runner's buffer (vLLM sizes it with
`ModelConfig.get_inputs_embeds_size`, which reads `projection_size` from the text config); pass it as the engine's
`hf_overrides`. `register` is the vLLM general plugin (entry point `vllm.general_plugins: ac_residuals`), loaded in every
vLLM process; this module imports nothing heavy so that loading it is cheap.
"""

ARCHITECTURE = "Qwen3_5ResForConditionalGeneration"
LAYERS_KEY = "ac_residual_layers"        # text-config field: the reader layers whose input receives a delta, ascending


def register() -> None:
    """vLLM general plugin: the residual architecture by name (imported lazily by vLLM) and Qwen3.5's config hook for it."""
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, Qwen3_5ForConditionalGenerationConfig
    ModelRegistry.register_model(ARCHITECTURE, f"{__name__}.qwen3_5:Qwen3_5ResForConditionalGeneration")
    MODELS_CONFIG_MAP.setdefault(ARCHITECTURE, Qwen3_5ForConditionalGenerationConfig)   # mamba state dtype from the checkpoint, as upstream


def engine_overrides(hidden_size: int, layers) -> dict:
    """The engine's `hf_overrides` for a Qwen3.5 reader whose AC rows carry deltas for `layers`."""
    layers = sorted(int(layer) for layer in layers)
    if not layers or len(set(layers)) != len(layers):
        raise ValueError(f"residual layers must be distinct and non-empty: {layers}")
    return {"architectures": [ARCHITECTURE], "text_config": {"projection_size": (1 + len(layers)) * int(hidden_size), LAYERS_KEY: layers}}


def pack_rows(rows, layer_inputs: dict):
    """[V, d] rows and {layer: [V, d]} deltas -> [V, (1 + k) * d] in the layer order of `engine_overrides`."""
    import torch
    return torch.cat([rows, *(layer_inputs[layer].to(rows.dtype) for layer in sorted(layer_inputs, key=int))], dim=-1)
