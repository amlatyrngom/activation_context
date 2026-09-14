"""
HF-facing code for model configuration parsing, message formatting, etc.
Stores overly detailed/specific logic to keep the rest of code cleaner.
"""
from collections import Counter
import torch
import torch.nn.functional as F
from .model_config import (
    LayerType,
    ModelDescription,
    LayerDescription,
    ModelConfig,
    EmbeddingReadoutType,
)
from transformers import (
    AutoConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    PreTrainedConfig,
    AutoTokenizer,
    AutoProcessor,
    ProcessorMixin,
)



TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SOURCE_DEVICE = "cpu"
FREE_DEVICE = "disk"
SUPPORTS_FP4 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)

_FULL_ATTENTION_NAMES = frozenset({"attention", "full_attention", "global_attention"})
_WINDOW_ATTENTION_NAMES = frozenset(
    {"local_attention", "sliding_attention", "window_attention"}
)
_LINEAR_ATTENTION_NAMES = frozenset(
    {"gated_delta_net", "linear_attention", "mamba", "recurrent", "ssm"}
)
_FFN_ONLY_NAMES = frozenset({"mlp", "moe"})



def canonical_layer_type(raw_type: object) -> LayerType:
    normalized = str(raw_type).casefold()
    if normalized in _FULL_ATTENTION_NAMES:
        return LayerType.FULL_ATTENTION
    if normalized in _WINDOW_ATTENTION_NAMES:
        return LayerType.WINDOW_ATTENTION
    if normalized in _LINEAR_ATTENTION_NAMES:
        return LayerType.LINEAR_ATTENTION
    if normalized in _FFN_ONLY_NAMES:
        return LayerType.FFN_ONLY
    raise ValueError(f"Unsupported model layer type: {raw_type!r}")


def canonical_eot_token(tokenizer: PreTrainedTokenizerBase) -> str | None:
    candidates = list(getattr(tokenizer, "additional_special_tokens", None) or ())
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token:
        candidates.append(eos_token)
    for token in candidates:
        normalized = token.casefold()
        if "eot" in normalized or "im_end" in normalized or "end_of_turn" in normalized:
            return token
    return None

def compute_last_token(tokenizer: PreTrainedTokenizerBase) -> str|None:
    # Tokenize empty string and return last special token.
    return tokenizer.convert_ids_to_tokens(
        tokenizer("", add_special_tokens=True)["input_ids"][-1]
    )


def canonical_eos_token(hf_config: PreTrainedConfig, tokenizer: PreTrainedTokenizerBase) -> str | None:
    """Resolve the model config's EOS ID to its tokenizer representation."""
    text_config = hf_config.get_text_config()
    raw_token_ids = getattr(
        text_config,
        "eos_token_id",
        getattr(hf_config, "eos_token_id", None),
    )
    token_ids = (
        [raw_token_ids]
        if isinstance(raw_token_ids, int)
        else list(raw_token_ids or ())
    )
    for token_id in token_ids:
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token is not None:
            return token
    return tokenizer.eos_token


def canonical_embedding_readout_type(model_id: str) -> EmbeddingReadoutType|None:
    normalized_id = model_id.lower()
    if "embed" not in normalized_id:
        return None
    if "qwen" in normalized_id:
        return EmbeddingReadoutType.EOS_TOKEN
    raise ValueError(f"Unknown embedding model: {model_id}")


def model_description_and_tokenizer_from_hf(
    model_id: str,
    dtype: torch.dtype,
) -> tuple['ModelDescription', PreTrainedTokenizerBase, PreTrainedTokenizerBase|ProcessorMixin]:
    # Get basic configs.
    hf_config = AutoConfig.from_pretrained(model_id)
    text_config = hf_config.get_text_config()
    # Load tokenizer and processor based on modality.
    is_multimodal = text_config is not hf_config
    if is_multimodal:
        processor = AutoProcessor.from_pretrained(model_id)
        assert isinstance(processor, ProcessorMixin)
        tokenizer = processor.tokenizer
    else:
        processor = AutoTokenizer.from_pretrained(model_id)
        tokenizer = processor

    # Compute layer descriptions.
    num_layers = int(text_config.num_hidden_layers)
    raw_layer_types = getattr(text_config, "layer_types", None)

    if raw_layer_types is None:
        raw_layer_types = ["full_attention"] * num_layers

    if len(raw_layer_types) != num_layers:
        raise ValueError(
            f"Config reports {num_layers} layers but supplies "
            f"{len(raw_layer_types)} layer types"
        )

    window_size = getattr(text_config, "sliding_window", None)
    layers = []

    for index, raw_type in enumerate(raw_layer_types):
        layer_type = canonical_layer_type(raw_type)
        layers.append(
            LayerDescription(
                layer_type=layer_type,
                layer_idx=index,
                window_attention_size=(
                    window_size
                    if layer_type is LayerType.WINDOW_ATTENTION
                    else None
                ),
            )
        )

    # Get readout type.
    embedding_readout_type = canonical_embedding_readout_type(model_id)
    if embedding_readout_type is EmbeddingReadoutType.EOS_TOKEN:
        eos_token = compute_last_token(tokenizer)
    else:
        eos_token = canonical_eos_token(hf_config, tokenizer)
    # Finalize.
    description = ModelDescription(
        d_model=text_config.hidden_size,
        is_multimodal=is_multimodal,
        dtype=dtype,
        layer_descriptions=layers,
        eos_token=eos_token,
        eot_token=canonical_eot_token(tokenizer),
        is_embedding_model=embedding_readout_type is not None,
        embedding_readout_type=embedding_readout_type,
    )
    return description, tokenizer, processor


def adapt_message_format(model_config: ModelConfig, messages: list[dict]):
    """
    Adapt the message.
    Assumes the input is strongly typed e.g., {type: "text", text: ...}.
    """
    if model_config.model_description.is_multimodal:
        return messages
    for msg in messages:
        if "content" in msg:
            if isinstance(msg["content"], str):
                continue
            if isinstance(msg["content"], list):
                assert len(msg["content"]) == 1
                if msg["content"][0]["type"] == "text":
                    msg["content"] = msg["content"][0]["text"]
                else:
                    raise RuntimeError("Non-multimodal model only supports direct content, texts.")



def readout_embedding(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    readout_type: EmbeddingReadoutType,
) -> torch.Tensor:
    """Readout the embedding from the last hidden state of a model."""
    if readout_type in [EmbeddingReadoutType.EOS_TOKEN, EmbeddingReadoutType.EOT_TOKEN]:
        # Left padded (or no padding) if right-most point is always active.
        # So directly readout.
        left_padded = bool(attention_mask[:, -1].all())
        if left_padded:
            pooled = last_hidden_state[:, -1]
        else:
            # Right-padded: select the last active index (equivalent to sum of 1s up to that point).
            padding_end_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(padding_end_indices.shape[0], device=last_hidden_state.device)
            pooled = last_hidden_state[batch_indices, padding_end_indices]
    else:
        # Avg: just average all active positions.
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return F.normalize(pooled, p=2, dim=1)


def pretty_format_model_description(model_config: ModelConfig) -> str:
    """Return a compact human-readable description of a loaded model."""

    description = model_config.model_description
    if description is None:
        return f"{model_config.model_name}: not loaded"

    counts = Counter(layer.layer_type.value for layer in description.layer_descriptions)
    layer_summary = ", ".join(
        f"{count} {layer_type}" for layer_type, count in sorted(counts.items())
    )
    lines = [
        f"model_name: {model_config.model_name}",
        f"model_id: {model_config.model_id}",
        f"d_model: {description.d_model}",
        f"dtype: {description.dtype}",
        f"layers: {len(description.layer_descriptions)} ({layer_summary})",
        f"multimodal: {description.is_multimodal}",
        f"embedding_model: {description.is_embedding_model}",
        f"eos_token: {description.eos_token}",
        f"eot_token: {description.eot_token}",
    ]
    if description.embedding_readout_type is not None:
        lines.append(f"embedding_readout: {description.embedding_readout_type.value}")
    return "\n".join(lines)
