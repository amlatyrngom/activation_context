import vllm

def best_effort_shutdown_vllm(vllm_model: vllm.LLM):
    """
    A best-effort shutdown method.
    """
    try:
        vllm_model.llm_engine.engine_core.shutdown()
    except:
        pass
    return