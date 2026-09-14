"""
Place to store all model configuration related objects.
We need to generally be able to parse:
- Gemma 3 & 4 model family.
- Qwen 3, 3.5 and 3.6.
- Qwen Embedding 3 Family.
- Nemotron 3 Family
- Anything similar to the above, that's not too exotic.

We'll ignore non-attention layers (non-global/windowed), but we need to know they exist.

"""
from dataclasses import dataclass, field
from enum import StrEnum, auto
import typing as t 
import torch


class LayerType(StrEnum):
    """The type of each layer."""
    FULL_ATTENTION = auto()
    WINDOW_ATTENTION = auto()
    LINEAR_ATTENTION = auto()
    FFN_ONLY = auto()

class EmbeddingReadoutType(StrEnum):
    """The readout type of the embedding model."""
    EOS_TOKEN = auto()
    EOT_TOKEN = auto()
    AVG_POOL = auto()


@dataclass
class LayerDescription:
    """Detailed description of a layer."""
    layer_type: LayerType
    layer_idx: int
    window_attention_size: int|None = None
    


@dataclass
class ModelDescription:
    """
    Detailed description of a model.
    Stores d_model, dtype, layer types, eos repr, eot repr, last_layer_index, is_embedding_model, etc.
    eos token repr is of the form "<|eos|>" or something like that, not a raw token id int.
    last_layer_index is the index we need to tap to create adapters and such. Right before lm head I suppose.
    """
    d_model: int
    is_multimodal: bool
    layer_descriptions: list[LayerDescription] = field(default_factory=list)
    dtype: torch.dtype = torch.bfloat16
    eos_token: str | None = None
    eot_token: str | None = None
    is_embedding_model: bool = True
    embedding_readout_type: EmbeddingReadoutType|None = None


@dataclass
class ModelConfig:
    model_name: str
    """Model name: used in interface-level accessors."""

    model_id: str
    """Hugging face model ID."""

    dtype: torch.dtype = torch.bfloat16
    """Model data type."""

    model_description: ModelDescription|None = None
    """Detailed Model Description. Auto-populated."""

    engine_kwargs: dict[str, t.Any]|None = None
    """Additional arguments passed to the engine. Override default ones."""


    def pretty_format_description(self) -> str:
        from .hf_utils import pretty_format_model_description
        return pretty_format_model_description(self)