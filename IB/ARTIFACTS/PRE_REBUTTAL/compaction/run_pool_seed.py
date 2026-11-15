"""
New-seed reruns of the 9B training-side pools (`pc_pool9_h6_trA/_trB/_trC1/_trC4`, `pc_pool9_h8_D1/_D2`, all seed 0) for the
clean compaction rerun (CLEAN_RECIPES_PLAN §3). Runs the paper kit's `run_pool.py` unchanged, with one engine override:
the KV cache dtype (default `auto` = bf16 on the RTX PRO 6000 spot nodes; the seed-0 pools used fp8 with FlashInfer).

    POOL_KV_CACHE_DTYPE=auto python run_pool_seed.py <run_pool.py arguments>     (run_pool.py must be on PYTHONPATH)
"""
import os

import run_pool
from activation.harness.vllm_wrapper import VLLMWrapper

KV_CACHE_DTYPE = os.environ.get("POOL_KV_CACHE_DTYPE", "auto")
_recommended = VLLMWrapper.recommended_engine_kwargs


def _with_kv_dtype(model_id: str, is_for_training: bool = False) -> dict:
    return _recommended(model_id, is_for_training) | {"kv_cache_dtype": KV_CACHE_DTYPE}


VLLMWrapper.recommended_engine_kwargs = staticmethod(_with_kv_dtype)

if __name__ == "__main__":                                   # vLLM spawns workers: keep the guard
    print(f"run_pool_seed - kv_cache_dtype {KV_CACHE_DTYPE}", flush=True)
    raise SystemExit(run_pool.main())
