import typing as t
import time
import torch
from torch import nn
import gc
from copy import deepcopy

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
)
import vllm

from .runtime_config import HarnessRuntimeConfig, HarnessStats
from .model_config import ModelConfig
from .hf_utils import (
    model_description_and_tokenizer_from_hf,
)

from .loaded_model import (
    LoadedModel,
)


class HarnessRuntime:
    """
    Centralized object that stores all harness information.
    """
    def __init__(self, harness_config: HarnessRuntimeConfig):
        from ..dataset import DatasetManager
        self.harness_config = harness_config
        self.harness_stats = HarnessStats()
        self.loaded_models: dict[str, LoadedModel] = dict() # Maps from name.
        self._load_models()
        self.dataset_manager = DatasetManager(self)
        pass


    def _load_models(self):
        """
        Load all the models.
        """
        for model_config in self.harness_config.model_configs.values():
            # NOTE: repeated ids result in different loaded models. This is intentional for now.
            self.loaded_models[model_config.model_name] = self._load_model(model_config)
        

    def _load_model(self, model_config: ModelConfig) -> LoadedModel:
        """Load a specific model."""
        initialization_start_time = time.time()
        print(f"{model_config.model_name} - Initializing model.")
        model_config.model_description, tokenizer, processor = model_description_and_tokenizer_from_hf(
            model_id=model_config.model_id,
            dtype=model_config.dtype,
        )
        if model_config.model_description.is_embedding_model:
            model_loader = AutoModel
        elif model_config.model_description.is_multimodal:
            model_loader = AutoModelForMultimodalLM
        else:
            model_loader = AutoModelForCausalLM
        loaded_model = LoadedModel(
            harness=self,
            model_config=model_config,
            model=None, # Not loaded here by default.
            model_loader=model_loader,
            processor=processor,
            tokenizer=tokenizer,
        )
        elapsed = time.time() - initialization_start_time
        print(f"{model_config.model_name} - Initialized model in {elapsed:.2f}s.")
        self.harness_stats.model_initialization_times.append(
            (model_config.model_name, time.time() - initialization_start_time)
        )
        return loaded_model
