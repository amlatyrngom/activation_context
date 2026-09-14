import typing as t
import time
import torch
from torch import nn
import gc
from copy import deepcopy
from dataclasses import dataclass, field

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
    cached_prompt_token_count: int = 0   # prefix-cache hits reported by the engine
    finish_reason: str = ""              # "stop" or "length"
    token_ids: list[int] = field(default_factory=list)   # the sampled tokens (end-of-turn included when the model produced it)
    logprobs: list[float] | None = None                  # log-prob of each sampled token when requested (SamplingParams.logprobs=0)
    lora_name: str | None = None                         # the adapter the request carried (None: base)

    @staticmethod
    def from_request_output(output: "vllm.RequestOutput", lora_name: str | None = None) -> "EngineChatOutput":
        completion = output.outputs[0]
        token_ids = list(completion.token_ids)
        logprobs = None
        if completion.logprobs:
            logprobs = []
            for entry, token_id in zip(completion.logprobs, token_ids):
                item = entry.get(token_id) if hasattr(entry, "get") else None
                logprobs.append(float(item.logprob) if item is not None else float(next(iter(entry.values())).logprob))
        return EngineChatOutput(
            text=completion.text,
            prompt_token_count=len(output.prompt_token_ids or ()),
            output_token_count=len(token_ids),
            cached_prompt_token_count=output.num_cached_tokens or 0,
            finish_reason=completion.finish_reason or "",
            token_ids=token_ids,
            logprobs=logprobs,
            lora_name=lora_name,
        )

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
        print(f"{self.model_config.model_name} - Moving model to {device}.")
        if device == TARGET_DEVICE:
            self._vacate_engine_for_model() # Avoid dual residency on the GPU.
        if device != FREE_DEVICE and self.current_model_device == FREE_DEVICE:
            # Reload from disk.
            self.model = self.model_loader.from_pretrained(
                self.model_config.model_id, dtype=self.model_config.dtype,
                trust_remote_code=self.model_config.trust_remote_code,
            )
            self.model.eval()
        # Finalize.
        previous_device = self.current_model_device
        self.model.to(device)
        self.current_model_device = device
        if previous_device.startswith("cuda") and not device.startswith("cuda"):
            # Hand the memory back to the driver: PyTorch's caching allocator keeps freed blocks
            # reserved, and vLLM's sleep-mode allocator grabs its share straight from CUDA on wake.
            gc.collect()
            torch.cuda.empty_cache()
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
        """
        Place the serving engine: TARGET builds it (or wakes a sleeping one), SOURCE ("cpu") puts it to
        sleep (vLLM sleep level 1: weights offloaded to host RAM, KV cache released, compiled graphs
        kept; the GPU is free for training), FREE shuts it down. Waking takes seconds; building takes
        the full load. A request on a sleeping engine wakes it.
        """
        device_change_start_time = time.time()
        harness_stats = self.harness.harness_stats
        if device == SOURCE_DEVICE and not self.vllm_model:
            return                                                       # nothing to put to sleep (also covers CPU-only hosts, where SOURCE == TARGET)
        if device == TARGET_DEVICE:
            if self.vllm_model:
                if self.current_engine_device == SOURCE_DEVICE:
                    self._vacate_model_for_engine()                      # the wake re-grabs the engine's whole GPU share
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"{self.model_config.model_name} - Engine waking up.")
                    self.vllm_model.wake_up()
                    self.current_engine_device = device
                    elapsed = time.time() - device_change_start_time
                    print(f"{self.model_config.model_name} - Engine woke up in {elapsed:.2f}s")
                    harness_stats.model_loading_times.append((self.model_config.model_name, elapsed))
                return
            self.engine_free_other_models()
            self._vacate_model_for_engine()
            print(f"{self.model_config.model_name} - Engine moving to target.")
            engine_kwargs = dict(
                model=self.model_config.model_id,
                max_loras=self.harness.harness_config.max_loras,
                max_lora_rank=self.harness.harness_config.max_lora_rank,
                enable_lora=True,
                enable_sleep_mode=True,
                dtype=self.model_config.dtype or "auto",
            )
            if self.model_config.engine_kwargs is None:
                self.model_config.engine_kwargs = VLLMWrapper.recommended_engine_kwargs(self.model_config.model_id)
            engine_kwargs.update(self.model_config.engine_kwargs)
            engine_kwargs.setdefault("trust_remote_code", self.model_config.trust_remote_code)
            self.model_config.engine_kwargs = engine_kwargs
            self.vllm_model = VLLMWrapper(tokenizer=self.tokenizer, **engine_kwargs)
            self.current_engine_device = device
            elapsed = time.time() - device_change_start_time
            print(f"{self.model_config.model_name} - Engine moved to target in {elapsed:.2f}s")
            harness_stats.model_loading_times.append(
                (self.model_config.model_name, elapsed)
            )
            return
        if device == SOURCE_DEVICE:
            if not self.vllm_model or self.current_engine_device == SOURCE_DEVICE:
                return
            print(f"{self.model_config.model_name} - Engine going to sleep (weights to host RAM).")
            self.vllm_model.sleep(level=1)
            self.current_engine_device = device
            elapsed = time.time() - device_change_start_time
            print(f"{self.model_config.model_name} - Engine asleep in {elapsed:.2f}s")
            harness_stats.model_freeing_times.append((self.model_config.model_name, elapsed))
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

    def _vacate_model_for_engine(self):
        """The GPU belongs to the engine: a plain model is freed, a model carrying adapters moves to host RAM."""
        if self.current_model_device == FREE_DEVICE:
            return
        if self.harness.module_manager.has_loras(self.model_config.model_name):
            self.model_to_device(SOURCE_DEVICE)
        else:
            self.model_to_device(FREE_DEVICE)

    def _vacate_engine_for_model(self):
        """The GPU belongs to the model: a running engine sleeps (its weights stay in host RAM for a fast wake)."""
        if self.vllm_model and self.current_engine_device == TARGET_DEVICE:
            self.engine_to_device(SOURCE_DEVICE)

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


    def _merged_chat_kwargs(self, chat_kwargs: dict|None, seed: int|None = None) -> tuple["vllm.SamplingParams", dict]:
        """Harness defaults < recommended chat kwargs for this model < explicit chat kwargs; the seed on top."""
        chat_kwargs = dict(chat_kwargs or {})
        recommended_chat_kwargs = VLLMWrapper.recommended_chat_kwargs(self.model_config.model_id)
        default_sampling_params_kwargs = dict(
            temperature=1.0,
            max_tokens=1024,
        )
        recommended_sampling_params_kwargs = recommended_chat_kwargs.pop("sampling_params", None) or dict()
        chat_sampling_params_kwargs = chat_kwargs.pop("sampling_params", None) or dict()
        sampling_kwargs = default_sampling_params_kwargs | recommended_sampling_params_kwargs | chat_sampling_params_kwargs
        if seed is not None:
            sampling_kwargs["seed"] = seed
        sampling_params = vllm.SamplingParams(**sampling_kwargs)
        default_extra_kwargs = dict(
            chat_template_kwargs={"enable_thinking": False},
        )
        return sampling_params, default_extra_kwargs | recommended_chat_kwargs | chat_kwargs

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
        sampling_params, extra_kwargs = self._merged_chat_kwargs(chat_kwargs)
        request_outputs = self.vllm_model.chat(
            conversations,
            sampling_params=sampling_params,
            use_tqdm=False,
            **extra_kwargs,
        )
        return [EngineChatOutput.from_request_output(output) for output in request_outputs]

    def chat_template_kwargs(self, chat_kwargs: dict|None) -> dict:
        """The chat-template kwargs a request would use (harness defaults < recommended < explicit)."""
        _, extra_kwargs = self._merged_chat_kwargs(chat_kwargs)
        return dict(extra_kwargs.get("chat_template_kwargs") or {})

    def engine_submit(
        self,
        messages: list[dict],
        tools: list[dict]|None = None,
        seed: int|None = None,
        agent_id: str = "",
        lora_name: str|None = None,
        chat_kwargs: dict|None = None,
        record_sampling: bool = False,
    ) -> EngineChatOutput:
        """
        One conversation for one agent: templated here, then submitted as tokens (engine_submit_tokens).
        """
        self.engine_to_device(TARGET_DEVICE)
        _, extra_kwargs = self._merged_chat_kwargs(chat_kwargs, seed)
        token_ids = self.vllm_model.template(messages, tools, True, extra_kwargs.get("chat_template_kwargs"))
        return self.engine_submit_tokens(token_ids, seed=seed, agent_id=agent_id, lora_name=lora_name, chat_kwargs=chat_kwargs,
                                         record_sampling=record_sampling)

    def engine_submit_tokens(
        self,
        prompt_token_ids: list[int],
        seed: int|None = None,
        agent_id: str = "",
        lora_name: str|None = None,
        chat_kwargs: dict|None = None,
        record_sampling: bool = False,
    ) -> EngineChatOutput:
        """
        One request whose prompt the caller owns as token ids (the agent loop keeps its own prefix).
        Same kwargs merge as engine_chat_many, a per-request seed, the agent pinned to one replica (by
        agent_id) so its prefix cache serves the next turn, the adapter the module manager currently
        exposes for `lora_name` (None until an exchange), and with `record_sampling` the sampled token
        log-probs. Blocks the calling thread while the engine batches this request with everything in flight.
        """
        self.engine_to_device(TARGET_DEVICE)
        sampling_params, _ = self._merged_chat_kwargs(chat_kwargs, seed)
        if record_sampling and sampling_params.logprobs is None:
            sampling_params.logprobs = 0
        lora_request = self.harness.module_manager.engine_lora_request(lora_name) if lora_name else None
        engine = self.vllm_model
        future = engine.submit(list(prompt_token_ids), sampling_params, replica=VLLMWrapper.replica_for(agent_id, engine.world_size),
                               lora_request=lora_request)
        return EngineChatOutput.from_request_output(future.result(), lora_name=lora_name if lora_request is not None else None)


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
        attention_mask: torch.Tensor | None,
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