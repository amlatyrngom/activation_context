"""
Independent vLLM engine replicas: one `AsyncLLM` per visible GPU, each driven by its own event loop
on a background thread. Requests are submitted one at a time from any thread (`submit`) and the
engine batches whatever is in flight (continuous batching), which is what agent rollouts need: agents
finish their turns at different times and cannot be batched synchronously. `chat` keeps the
`vllm.LLM.chat` signature on top of that: every conversation is templated with the tokenizer,
submitted with strided replica assignment, and the results come back in input order. No scheduler
on our side beyond the replica choice; each replica is data-parallel size one, so vLLM never enters
its lockstep MoE path and single-GPU behavior is exactly one engine.
"""

import asyncio
import concurrent.futures
import os
import threading
import uuid
import zlib

import torch
import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

WORLD_SIZE = max(1, torch.cuda.device_count())
ENGINE_MAX_NUM_SEQS = 128
ENGINE_CONCURRENCY = WORLD_SIZE * ENGINE_MAX_NUM_SEQS
RECOMMENDED_BATCH_SIZE = 4 * ENGINE_CONCURRENCY # Decode-heavy passes (study): backlog amortizes the tail.


class _Replica:
    """One AsyncLLM and the event loop thread that drives it."""

    def __init__(self, engine_kwargs: dict):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="vllm-replica-loop", daemon=True)
        self.thread.start()

        async def construct():
            # Constructed inside the loop so anything AsyncLLM binds to "the running loop" binds to this one.
            return AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs))

        self.engine: AsyncLLM = asyncio.run_coroutine_threadsafe(construct(), self.loop).result()

    async def _collect(self, prompt_token_ids: list[int], sampling_params, request_id: str, lora_request) -> vllm.RequestOutput:
        sampling_params = sampling_params.clone()
        sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
        final = None
        async for output in self.engine.generate(
            vllm.TokensPrompt(prompt_token_ids=prompt_token_ids), sampling_params, request_id, lora_request=lora_request,
        ):
            final = output
        assert final is not None, f"request {request_id} produced no output"
        return final

    def submit(self, prompt_token_ids, sampling_params, request_id, lora_request=None) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(
            self._collect(prompt_token_ids, sampling_params, request_id, lora_request), self.loop,
        )

    def run(self, coroutine):
        """Run one engine coroutine on this replica's loop and wait for it."""
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    def shutdown(self):
        try:
            self.engine.shutdown()
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)


class VLLMWrapper:
    def __init__(self, tokenizer=None, **engine_kwargs):
        """
        `tokenizer` templates conversations for `chat`; without one the engine's own tokenizer is used.
        Every other keyword goes to vLLM's engine arguments.
        """
        engine_kwargs = dict(engine_kwargs)
        if engine_kwargs.setdefault("max_num_seqs", ENGINE_MAX_NUM_SEQS) != ENGINE_MAX_NUM_SEQS:
            print(
                f"Engine max_num_seqs={engine_kwargs['max_num_seqs']} overrides {ENGINE_MAX_NUM_SEQS}; "
                "batch constants assume the latter."
            )
        engine_kwargs.setdefault("disable_log_stats", True)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = visible.split(",") if visible else [str(i) for i in range(WORLD_SIZE)]
        self.replicas: list[_Replica] = []
        try:
            # Sequential: CUDA_VISIBLE_DEVICES is process-global and inherited by each spawned
            # engine core. The first replica pays the kernel JIT; later ones load warm.
            for device in devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = device
                self.replicas.append(_Replica(engine_kwargs))
        finally:
            if visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = visible
        self.tokenizer = tokenizer if tokenizer is not None else self.replicas[0].engine.get_tokenizer()

    @property
    def world_size(self) -> int:
        return len(self.replicas)

    @staticmethod
    def recommended_engine_kwargs(model_id: str, is_for_training: bool = False) -> dict:
        """Provides recommended engine construction kwargs for the given model."""
        assert not is_for_training, "Recommended engine kwargs for training rolouts not yet set."
        base = {
            "max_model_len": 20_000,
            "limit_mm_per_prompt": {"image": 0, "video": 0},
            "kv_cache_dtype": "fp8",
            "gpu_memory_utilization": 0.9,
        }   # LoRA support and sleep mode come from LoadedModel.engine_to_device (agents need both)
        mtp = {"speculative_config": {"method": "mtp", "num_speculative_tokens": 2}}
        known = {
            # Dense generator: MTP helps decode.
            "unsloth/Qwen3.8-27B-NVFP4": base | mtp,
            "Qwen/Qwen3.8-27B-FP8": base | mtp,
            # A3B MoE labeler: MTP taxes prefill, and labeling is prefill-bound.
            "nvidia/Qwen3.6-35B-A3B-NVFP4": base,
            "Qwen/Qwen3.6-35B-A3B-FP8": base,
            # Agent rollouts: 50k of trajectory plus a buffer (the trainer's max_example_tokens matches).
            "Qwen/Qwen3.5-9B": base | {"max_model_len": 52_000},
            "Qwen/Qwen3.5-4B": base | {"max_model_len": 52_000},
        }
        if model_id not in known:
            print(f"{model_id} - No recommended engine kwargs; using the base formula without speculative decoding.")
        return dict(known.get(model_id, base))


    @staticmethod
    def recommended_chat_kwargs(model_id: str, is_for_training: bool = False) -> dict:
        """
        Provides recommended chat kwargs for the given model.
        Should be augmented with prompt-specific parameters.
        """
        assert not is_for_training, "Recommended chat kwargs for training rollouts not yet set."
        # Qwen3.6 / Qwen3.8 publish the same non-thinking sampling; presence_penalty curbs repetition.
        qwen_non_thinking = {
            "sampling_params": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
            },
        }
        known = {
            "unsloth/Qwen3.8-27B-NVFP4": qwen_non_thinking,
            "Qwen/Qwen3.8-27B-FP8": qwen_non_thinking,
            "nvidia/Qwen3.6-35B-A3B-NVFP4": qwen_non_thinking,
            "Qwen/Qwen3.6-35B-A3B-FP8": qwen_non_thinking,
            # Qwen3.5 thinks by default; agents and study prompts want the direct answer.
            "Qwen/Qwen3.5-9B": qwen_non_thinking | {"chat_template_kwargs": {"enable_thinking": False}},
            "Qwen/Qwen3.5-4B": qwen_non_thinking | {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if model_id not in known:
            print(f"{model_id} - No recommended chat kwargs; using engine defaults.")
        return {key: dict(value) for key, value in known.get(model_id, {}).items()}

    # ----------------------------------------------------------------------------- requests
    @staticmethod
    def replica_for(agent_id: str, world_size: int) -> int:
        """Stable agent -> replica assignment, so an agent's prefix cache serves its next turn."""
        return zlib.crc32(agent_id.encode()) % world_size

    def template(self, conversation: list[dict], tools: list[dict] | None = None,
                 add_generation_prompt: bool = True, chat_template_kwargs: dict | None = None) -> list[int]:
        """Token ids of one conversation through the tokenizer's chat template (text-only content)."""
        flattened = []
        for message in conversation:
            message = dict(message)
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            flattened.append(message)
        ids = self.tokenizer.apply_chat_template(
            flattened, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=True,
            **(chat_template_kwargs or {}),
        )
        if hasattr(ids, "keys"):                                                      # BatchEncoding / dict
            ids = ids["input_ids"]
        return list(ids)

    def submit(self, prompt_token_ids: list[int], sampling_params: vllm.SamplingParams, request_id: str | None = None,
               replica: int = 0, lora_request=None) -> concurrent.futures.Future:
        """One request on one replica; the future resolves to the final vllm.RequestOutput."""
        request_id = request_id or uuid.uuid4().hex
        return self.replicas[replica % len(self.replicas)].submit(prompt_token_ids, sampling_params, request_id, lora_request)

    def chat(self, conversations: list, **chat_kwargs) -> list[vllm.RequestOutput]:
        """Same signature as vllm.LLM.chat: template every conversation, submit strided over the replicas, return in order."""
        sampling_params = chat_kwargs.pop("sampling_params", None) or vllm.SamplingParams()
        tools = chat_kwargs.pop("tools", None)
        chat_template_kwargs = chat_kwargs.pop("chat_template_kwargs", None)
        add_generation_prompt = chat_kwargs.pop("add_generation_prompt", True)
        lora_request = chat_kwargs.pop("lora_request", None)
        chat_kwargs.pop("use_tqdm", None)
        if chat_kwargs:
            print(f"VLLMWrapper.chat ignores {sorted(chat_kwargs)}")
        futures = []
        for index, conversation in enumerate(conversations):
            ids = self.template(conversation, tools, add_generation_prompt, chat_template_kwargs)
            futures.append(self.submit(ids, sampling_params, replica=index % len(self.replicas), lora_request=lora_request))
        return [future.result() for future in futures]

    def sleep(self, level: int = 1) -> None:
        """vLLM sleep mode on every replica: level 1 offloads the weights to host RAM and frees the KV cache."""
        for replica in self.replicas:
            replica.run(replica.engine.sleep(level=level))

    def wake_up(self) -> None:
        """Back on the GPU; the prefix cache is reset since the KV cache was released."""
        for replica in self.replicas:
            replica.run(replica.engine.wake_up())
            replica.run(replica.engine.reset_prefix_cache())

    def shutdown(self):
        """Best-effort shutdown of every replica's engine core."""
        for replica in self.replicas:
            replica.shutdown()
        self.replicas = []
