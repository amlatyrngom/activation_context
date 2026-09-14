import typing as t
import time
import torch
from torch import nn
import gc
from copy import deepcopy
from dataclasses import dataclass

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
from .vllm_wrapper import VLLMWrapper

if t.TYPE_CHECKING:
    from .runtime import HarnessRuntime

@dataclass
class EngineChatOutput:
    text: str
    prompt_token_count: int
    output_token_count: int

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
        self.vllm_model: VLLMWrapper|None = None
        self.current_engine_device: str = FREE_DEVICE


    @staticmethod
    def load(harness: "HarnessRuntime", model_config: ModelConfig) -> "LoadedModel":
        """Load a model."""
        model_config.model_description, tokenizer, processor = model_description_and_tokenizer_from_hf(
            model_id=model_config.model_id,
            dtype=model_config.dtype,
            trust_remote_code=model_config.trust_remote_code,
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
        
        device_change_start_time = time.time()
        harness_stats = self.harness.harness_stats
        if device == FREE_DEVICE and self.current_model_device != FREE_DEVICE:
            assert not self.harness.module_manager.has_loras(self.model_config.model_name), (
                "Free the adapters first (module_manager.free_lora): freeing the base would drop LoRA weights."
            )
            # Complete free.
            self.model = None
            # GC and cuda frees.
            print(f"{self.model_config.model_name} - Freeing model.")
            gc.collect()
            if self.current_model_device.startswith("cuda"):
                torch.cuda.empty_cache()
            self.current_model_device = device
            elapsed = time.time() - device_change_start_time
            print(f"{self.model_config.model_name} - Freed model in {elapsed:.2f}s")
            harness_stats.model_freeing_times.append(
                (self.model_config.model_name, elapsed)
            )
            return

        # Regular load/move.
        print(f"{self.model_config.model_name} - Moving model to target.")
        self.engine_to_device(FREE_DEVICE) # Avoid dual load.
        if device != FREE_DEVICE and self.current_model_device == FREE_DEVICE:
            # Reload from disk.
            self.model = self.model_loader.from_pretrained(
                self.model_config.model_id, dtype=self.model_config.dtype,
                trust_remote_code=self.model_config.trust_remote_code,
            )
            self.model.eval()
        # Finalize.
        self.model.to(device)
        self.current_model_device = device
        elapsed = time.time() - device_change_start_time
        print(f"{self.model_config.model_name} - Moved model in {elapsed:.2f}s")
        harness_stats.model_loading_times.append(
            (self.model_config.model_name, elapsed)
        )
        return

    def embedding_layer_to_device(self, device: str):
        """Move the embedding layer to a give device."""
        if self.current_embedding_layer_device == device:
            # No-op: already in the right place.
            return
        device_change_start_time = time.time()
        harness_stats = self.harness.harness_stats
        if device != FREE_DEVICE and self.current_embedding_layer_device != FREE_DEVICE:
            # Regular device change (cpu -> gpu or gpu -> cpu).
            print(f"{self.model_config.model_name} - Moving embedding layer.")
            self.embedding_layer.to(device)
            self.current_embedding_layer_device = device
            elapsed = time.time() - device_change_start_time
            print(f"{self.model_config.model_name} - Moved embedding layer in {elapsed:.2f}s")
            harness_stats.model_loading_times.append(
                (self.model_config.model_name, elapsed)
            )
            return
        if device == FREE_DEVICE and self.current_embedding_layer_device != FREE_DEVICE:
            # Free.
            print(f"{self.model_config.model_name} - Freeing embedding layer.")
            self.embedding_layer = None
            gc.collect()
            if self.current_embedding_layer_device.startswith("cuda"):
                torch.cuda.empty_cache()
            self.current_embedding_layer_device = device
            elapsed = time.time() - device_change_start_time
            print(f"{self.model_config.model_name} - Freed embedding layer in {elapsed:.2f}s")
            harness_stats.model_freeing_times.append(
                (self.model_config.model_name, elapsed)
            )
            return
        # Bring over from scratch. Need to temporarily use the whole model.
        print(f"{self.model_config.model_name} - Copying embedding layer.")
        starting_model_device = self.current_model_device
        if starting_model_device == FREE_DEVICE:
            self.model_to_device(SOURCE_DEVICE) # Use cpu for efficiency.
        self.embedding_layer = deepcopy(self.model.get_input_embeddings())
        self.embedding_layer.requires_grad_(False)
        self.embedding_layer.eval().to(device)
        self.current_embedding_layer_device = device
        if starting_model_device == FREE_DEVICE:
            self.model_to_device(FREE_DEVICE) # Restore starting model device.
        elapsed = time.time() - device_change_start_time
        print(f"{self.model_config.model_name} - Copied embedding layer in {elapsed:.2f}s")
        harness_stats.model_loading_times.append(
            (self.model_config.model_name, elapsed)
        )
        

    def engine_to_device(self, device: str):
        """Move engine model to the given device."""
        assert device != SOURCE_DEVICE, "Our vllm only supports gpu/free, not cpu."
        device_change_start_time = time.time()
        harness_stats = self.harness.harness_stats
        if device == TARGET_DEVICE:
            if self.vllm_model:
                return
            self.engine_free_other_models()
            self.model_to_device(FREE_DEVICE) # Avoid dual load.
            print(f"{self.model_config.model_name} - Engine moving to target.")
            engine_kwargs = dict(
                model=self.model_config.model_id,
                max_loras=4,
                max_lora_rank=self.harness.harness_config.max_lora_rank,
                enable_lora=True,
                dtype=self.model_config.dtype or "auto",
            )
            if self.model_config.engine_kwargs is None:
                self.model_config.engine_kwargs = VLLMWrapper.recommended_engine_kwargs(self.model_config.model_id)
            engine_kwargs.update(self.model_config.engine_kwargs)
            engine_kwargs.setdefault("trust_remote_code", self.model_config.trust_remote_code)
            self.model_config.engine_kwargs = engine_kwargs
            self.vllm_model = VLLMWrapper(**engine_kwargs)
            self.current_engine_device = device
            elapsed = time.time() - device_change_start_time
            print(f"{self.model_config.model_name} - Engine moved to target in {elapsed:.2f}s")
            harness_stats.model_loading_times.append(
                (self.model_config.model_name, elapsed)
            )
            return
        if device == FREE_DEVICE:
            if not self.vllm_model:
                return
            print(f"{self.model_config.model_name} - Engine freeing.")
            self.vllm_model.shutdown()
            self.vllm_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elapsed = time.time() - device_change_start_time
            print(f"{self.model_config.model_name} - Engine freed in {elapsed:.2f}s")
            harness_stats.model_freeing_times.append(
                (self.model_config.model_name, elapsed)
            )


    def engine_free_other_models(self):
        """Free all other models to make space for this one."""
        for other in self.harness.loaded_models.values():
            if other is not self:
                other.engine_to_device(FREE_DEVICE)


    def ensure_engine_loaded(self):
        """
        Place the engine on the target device now (a no-op when already there), so callers can
        keep the cold load out of their own batch timings.
        """
        self.engine_to_device(TARGET_DEVICE)


    def simple_engine_chat(self, messages: list[dict], lora_name: str|None = None, chat_kwargs: dict|None = None) -> str:
        """Simple function to test engine chat."""
        return self.engine_chat_many(
            conversations=[messages],
            chat_kwargs=chat_kwargs,
        )[0].text


    def engine_chat_many(
        self,
        conversations: list[list[dict]],
        lora_name: str|None = None,
        chat_kwargs: dict|None = None,
    ) -> list[EngineChatOutput]:
        """
        Batched engine chat. Returns texts with token counts for stats.
        """
        self.engine_to_device(TARGET_DEVICE)
        # Harness defaults < recommended chat kwargs for this model < explicit chat kwargs.
        chat_kwargs = dict(chat_kwargs or {})
        recommended_chat_kwargs = VLLMWrapper.recommended_chat_kwargs(self.model_config.model_id)
        default_sampling_params_kwargs = dict(
            temperature=1.0,
            max_tokens=1024,
        )
        recommended_sampling_params_kwargs = recommended_chat_kwargs.pop("sampling_params", None) or dict()
        chat_sampling_params_kwargs = chat_kwargs.pop("sampling_params", None) or dict()
        sampling_params = vllm.SamplingParams(
            **(default_sampling_params_kwargs | recommended_sampling_params_kwargs | chat_sampling_params_kwargs)
        )
        default_extra_kwargs = dict(
            chat_template_kwargs={"enable_thinking": False},
        )
        request_outputs = self.vllm_model.chat(
            conversations,
            sampling_params=sampling_params,
            use_tqdm=False,
            **(default_extra_kwargs | recommended_chat_kwargs | chat_kwargs)
        )
        return [
            EngineChatOutput(
                text=output.outputs[0].text,
                prompt_token_count=len(output.prompt_token_ids or ()),
                output_token_count=len(output.outputs[0].token_ids),
            )
            for output in request_outputs
        ]


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


    def decoder_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor|None = None,
        lora_name: str|None = None,
    ) -> torch.Tensor:
        """
        Last hidden state [B, S, d_model] of the causal decoder without the language-model head.
        With a LoRA name the module manager activates that adapter for this pass (PEFT injects the
        adapters into the base's own linear layers, so self.model is the LoRA'd module tree; the
        wrapper only routes and manages adapters); with None any injected adapters are disabled.
        Gradients flow; the caller sets train/eval.
        """
        assert self.model is not None, f"{self.model_config.model_name} - Model is not loaded."
        model = self.model
        decoder = model.get_decoder() if hasattr(model, "get_decoder") else getattr(model, model.base_model_prefix)
        with self.harness.module_manager.lora_context(self.model_config.model_name, lora_name):
            outputs = decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
        return outputs.last_hidden_state

    def simple_vector_embed_many(self, texts: list[str]) -> torch.Tensor:
        """Returns one normalized embedding per text"""
        self.model_to_device(TARGET_DEVICE)
        assert self.model_config.model_description.is_embedding_model
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
            embedding = readout_embedding(
                last_hidden_state=outputs.last_hidden_state,
                attention_mask=inputs["attention_mask"],
                readout_type=self.model_config.model_description.embedding_readout_type,
            )
        return embedding.detach().to(device=SOURCE_DEVICE)

    def simple_vector_embed(self, text: str) -> torch.Tensor:
        """Simple function to test embeddings."""
        return self.simple_vector_embed_many([text])[0]