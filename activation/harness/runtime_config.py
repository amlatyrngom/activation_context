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
    max_lora_rank = 128
    max_loras = 4


# Load the right env vars
load_dotenv()