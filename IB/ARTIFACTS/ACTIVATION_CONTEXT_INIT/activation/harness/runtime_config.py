from dataclasses import dataclass, field
from enum import StrEnum, auto
import typing as t 
import torch
from dotenv import load_dotenv

from .model_config import ModelConfig

@dataclass
class HarnessRuntimeConfig:
    # Map from model_name --> model config.
    model_configs: dict[str, ModelConfig] = field(default_factory=dict)

    # Model configs
    max_lora_rank: int = 128
    max_loras: int = 4

    # Document chunking
    doc_chunk_size_chars: int = 2048
    doc_chunk_overlap_chars: int = 256
    doc_embedding_model_name: str|None = None
    doc_embedding_batch_size: int = 32
    doc_chunk_max_atomic_size: int = 16384 # For long atomic passages.
    doc_embedding_input_limit_chars: int|None = None # Absolute limit on embedding inputs. Used for fast testing.


# Load the right env vars
load_dotenv()