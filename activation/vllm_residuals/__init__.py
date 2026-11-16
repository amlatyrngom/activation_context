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
    allow_short_prompt_embeds()


def allow_short_prompt_embeds() -> None:
    """
    Let a mixed prompt's embedding tensor stop before the prompt's end; every position past it must be a token id (mask True).
    vLLM 0.28 only refuses this in `vllm.utils.length_from_prompt_token_ids_or_embeds` (a length-equality check); its consumers
    already handle a shorter tensor: the runner copies rows only where the tensor has them and embeds every token-id position
    itself (`is_token_ids`), and the prefix-cache hash of a block past the end hashes an empty slice (the same on every turn).
    Why: with packed deltas every position is (1 + k)·d wide, so an agent resending its whole prompt each turn would ship
    gigabytes per request late in a long run; `Agent.prepare_request` cuts a deep model's tensor after its last AC row.
    """
    import sys

    import vllm.utils
    original = vllm.utils.length_from_prompt_token_ids_or_embeds
    if getattr(original, "allows_short_embeds", False):
        return

    def length_from_prompt_token_ids_or_embeds(prompt_token_ids, prompt_embeds) -> int:
        if prompt_token_ids is not None and prompt_embeds is not None and len(prompt_embeds) < len(prompt_token_ids):
            return len(prompt_token_ids)
        return original(prompt_token_ids, prompt_embeds)

    length_from_prompt_token_ids_or_embeds.allows_short_embeds = True
    vllm.utils.length_from_prompt_token_ids_or_embeds = length_from_prompt_token_ids_or_embeds
    for name, module in list(sys.modules.items()):                     # modules that imported the function by name before this ran
        if name.startswith("vllm") and getattr(module, "length_from_prompt_token_ids_or_embeds", None) is original:
            module.length_from_prompt_token_ids_or_embeds = length_from_prompt_token_ids_or_embeds


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
