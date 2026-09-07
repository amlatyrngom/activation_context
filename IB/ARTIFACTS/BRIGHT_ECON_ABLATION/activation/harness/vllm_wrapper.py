"""
Independent vLLM engine replicas: one `vllm.LLM` per visible GPU, a batch split by world size,
results reassembled in input order. No scheduler on our side; a shard that finishes early idles
until the others do. Each replica is data-parallel size one, so vLLM never enters its lockstep
MoE path and single-GPU behavior is exactly today's.
"""

import os
import threading

import torch
import vllm

WORLD_SIZE = max(1, torch.cuda.device_count())
ENGINE_MAX_NUM_SEQS = 128
ENGINE_CONCURRENCY = WORLD_SIZE * ENGINE_MAX_NUM_SEQS
RECOMMENDED_BATCH_SIZE = 4 * ENGINE_CONCURRENCY # Decode-heavy passes (study): backlog amortizes the tail.


class VLLMWrapper:
    def __init__(self, **engine_kwargs):
        engine_kwargs = dict(engine_kwargs)
        if engine_kwargs.setdefault("max_num_seqs", ENGINE_MAX_NUM_SEQS) != ENGINE_MAX_NUM_SEQS:
            print(
                f"Engine max_num_seqs={engine_kwargs['max_num_seqs']} overrides {ENGINE_MAX_NUM_SEQS}; "
                "batch constants assume the latter."
            )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = visible.split(",") if visible else [str(i) for i in range(WORLD_SIZE)]
        self.replicas: list[vllm.LLM] = []
        try:
            # Sequential: CUDA_VISIBLE_DEVICES is process-global and inherited by each spawned
            # engine core. The first replica pays the kernel JIT; later ones load warm.
            for device in devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = device
                self.replicas.append(vllm.LLM(**engine_kwargs))
        finally:
            if visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = visible

    @staticmethod
    def recommended_engine_kwargs(model_id: str, is_for_training: bool = False) -> dict:
        """Provides recommended engine construction kwargs for the given model."""
        assert not is_for_training, "Recommended engine kwargs for training rolouts not yet set."
        base = {
            "max_model_len": 20_000,
            "limit_mm_per_prompt": {"image": 0, "video": 0},
            "kv_cache_dtype": "fp8",
            "gpu_memory_utilization": 0.9,
            "enable_lora": False,
        }
        mtp = {"speculative_config": {"method": "mtp", "num_speculative_tokens": 2}}
        known = {
            # Dense generator: MTP helps decode.
            "unsloth/Qwen3.8-27B-NVFP4": base | mtp,
            "Qwen/Qwen3.8-27B-FP8": base | mtp,
            # A3B MoE labeler: MTP taxes prefill, and labeling is prefill-bound.
            "nvidia/Qwen3.6-35B-A3B-NVFP4": base,
            "Qwen/Qwen3.6-35B-A3B-FP8": base,
            # Small dense generator for cheap synthetic passes (descriptions, questions): 256 sequences measured best.
            "RedHatAI/Qwen3.5-4B-FP8-dynamic": base | {"max_num_seqs": 256},
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
            # Qwen3.5 thinks by default; the study prompts want the direct JSON answer.
            "RedHatAI/Qwen3.5-4B-FP8-dynamic": qwen_non_thinking | {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if model_id not in known:
            print(f"{model_id} - No recommended chat kwargs; using engine defaults.")
        return {key: dict(value) for key, value in known.get(model_id, {}).items()}

        

    def chat(self, conversations: list, **chat_kwargs) -> list[vllm.RequestOutput]:
        """Same signature as vllm.LLM.chat; strided shards, one thread per replica, in-order results."""
        num_replicas = len(self.replicas)
        if num_replicas == 1:
            return self.replicas[0].chat(conversations, **chat_kwargs)
        shards = [conversations[rank::num_replicas] for rank in range(num_replicas)]
        results: list = [None] * num_replicas
        errors: list = [None] * num_replicas

        def work(rank: int):
            try:
                results[rank] = self.replicas[rank].chat(shards[rank], **chat_kwargs) if shards[rank] else []
            except BaseException as error:
                errors[rank] = error

        threads = [threading.Thread(target=work, args=(rank,), daemon=True) for rank in range(num_replicas)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for error in errors:
            if error is not None:
                raise error
        return [results[i % num_replicas][i // num_replicas] for i in range(len(conversations))]

    def shutdown(self):
        """Best-effort shutdown of every replica's engine core."""
        for replica in self.replicas:
            try:
                replica.llm_engine.engine_core.shutdown()
            except Exception:
                pass
        self.replicas = []
