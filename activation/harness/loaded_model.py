import typing as t
import torch
from torch import nn
import gc
from copy import deepcopy

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
import vllm

from .runtime_config import HarnessRuntimeConfig
from .model_config import ModelConfig
from .hf_utils import (
    model_description_and_tokenizer_from_hf,
    adapt_message_format,
    readout_embedding,
    SOURCE_DEVICE,
    TARGET_DEVICE,
    FREE_DEVICE,
)
from .vllm_utils import (
    best_effort_shutdown_vllm,
)

if t.TYPE_CHECKING:
    from .runtime import HarnessRuntime

class LoadedModel:
    """
    A loaded model.
    """
    def __init__(
        self,
        harness: "HarnessRuntime",
        model_config: ModelConfig,
        model: PreTrainedModel|None,
        model_loader: type[AutoModel|AutoModelForMultimodalLM|AutoModelForCausalLM],
        processor: ProcessorMixin|PreTrainedTokenizerBase,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.harness = harness
        self.model_config = model_config
        self.model = model
        self.model_loader = model_loader
        self.processor = processor
        self.tokenizer = tokenizer
        self.embedding_layer: nn.Module|None = None
        self.current_model_device: str = str(model.device) if model else FREE_DEVICE
        self.current_embedding_layer_device: str = FREE_DEVICE
        self.vllm_model: vllm.LLM|None = None
        self.current_engine_device: str = FREE_DEVICE


    @staticmethod
    def load(harness: "HarnessRuntime", model_config: ModelConfig) -> "LoadedModel":
        """Load a model."""
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
        return LoadedModel(
            harness=harness,
            model_config=model_config,
            model=None, # Not loaded here by default.
            model_loader=model_loader,
            processor=processor,
            tokenizer=tokenizer,
        )
 
    def model_to_device(self, device: str):
        """
        Move raw model to device.
        """
        if device == self.current_model_device:
            return # Nothing to do.
        
        if device == FREE_DEVICE and self.current_model_device != FREE_DEVICE:
            # Complete free.
            self.model = None
            # GC and cuda frees.
            gc.collect()
            if self.current_model_device.startswith("cuda"):
                torch.cuda.empty_cache()
            self.current_model_device = device
            return

        # Regular load/move.
        self.engine_to_device(FREE_DEVICE) # Avoid dual load.
        if device != FREE_DEVICE and self.current_model_device == FREE_DEVICE:
            # Reload from disk.
            self.model = self.model_loader.from_pretrained(
                self.model_config.model_id, dtype=self.model_config.dtype,
            )
            self.model.eval()
        # Finalize.
        self.model.to(device)
        self.current_model_device = device
        return

    def embedding_layer_to_device(self, device: str):
        """Move the embedding layer to a give device."""
        if self.current_embedding_layer_device == device:
            # No-op: already in the right place.
            return
        if device != FREE_DEVICE and self.current_embedding_layer_device != FREE_DEVICE:
            # Regular device change (cpu -> gpu or gpu -> cpu).
            self.embedding_layer.to(device)
            self.current_embedding_layer_device = device
            return
        if device == FREE_DEVICE and self.current_embedding_layer_device != FREE_DEVICE:
            # Free.
            self.embedding_layer = None
            gc.collect()
            if self.current_embedding_layer_device.startswith("cuda"):
                torch.cuda.empty_cache()
            self.current_embedding_layer_device = device
            return
        # Bring over from scratch. Need to temporarily use the whole model.
        starting_model_device = self.current_model_device
        if starting_model_device == FREE_DEVICE:
            self.model_to_device(SOURCE_DEVICE) # Use cpu for efficiency.
        self.embedding_layer = deepcopy(self.model.get_input_embeddings())
        self.embedding_layer.requires_grad_(False)
        self.embedding_layer.eval().to(device)
        self.current_embedding_layer_device = device
        if starting_model_device == FREE_DEVICE:
            self.model_to_device(FREE_DEVICE) # Restore starting model device.


    def engine_to_device(self, device: str):
        """Move engine model to the given device."""
        assert device != SOURCE_DEVICE, "Our vllm only supports gpu/free, not cpu."
        if device == TARGET_DEVICE:
            if self.vllm_model:
                return
            self.model_to_device(FREE_DEVICE) # Avoid dual load.
            self.vllm_model = vllm.LLM(
                model=self.model_config.model_id,
                max_loras=4,
                max_lora_rank=self.harness.harness_config.max_lora_rank,
                enable_lora=True,
                dtype=self.model_config.dtype or "auto",
            )
            self.current_engine_device = device
            return
        if device == FREE_DEVICE:
            if not self.vllm_model:
                return
            best_effort_shutdown_vllm(self.vllm_model)
            self.vllm_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    def simple_engine_chat(self, messages: list[dict], lora_name: str|None = None) -> str:
        """Simple function to test engine chat."""
        self.engine_to_device(TARGET_DEVICE)
        messages = deepcopy(messages)
        sampling_params = vllm.SamplingParams(
            temperature=1.0,
            max_tokens=128,
        )
        print(f"Prompting VLLM.")
        request_outputs = self.vllm_model.chat(
            messages,
            sampling_params=sampling_params,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        return request_outputs[0].outputs[0].text


    def simple_chat(self, user_msg: str) -> str:
        """Simple function to test the direct chat."""
        self.model_to_device(TARGET_DEVICE)
        messages = [
            {
                "role": "user", "content": [{
                    "type": "text", "text": user_msg,
                }]
            }
        ]
        adapt_message_format(self.model_config, messages)
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(self.model.device)
        print(f"Input IDs Shape: {inputs["input_ids"].shape}")
        prompt_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )
        generated_ids = output_ids[0, prompt_length:]
        print(f"Generated IDs shape: {generated_ids.shape}")
        return self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        )


    def simple_vector_embed(self, text: str) -> torch.Tensor:
        """Simply function to test embeddings"""
        self.model_to_device(TARGET_DEVICE)
        assert self.model_config.model_description.is_embedding_model
        inputs = self.tokenizer(
            text,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
            embedding = readout_embedding(
                last_hidden_state=outputs.last_hidden_state,
                attention_mask=inputs["attention_mask"],
                readout_type=self.model_config.model_description.embedding_readout_type,
            )[0]
        return embedding.detach().to(device=SOURCE_DEVICE)