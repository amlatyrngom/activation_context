"""
Harness-path smoke for Hidden Residuals: the T_res_rq9fr_kI_KL pair through HarnessRuntime and an Agent, as a rollout would use it.
LoadedModel.engine_to_device builds the reader engine with activation.vllm_residuals' hf_overrides (enable_lora and sleep mode as in
rollouts, bf16 KV cache); the agent's first prompt holds one activation part (a dev1000 question's k=200 passages at ratio 0.1), the
deep AC model encodes it through the rollout encode queue (rows + deltas), the agent packs them and submits the mixed prompt twice
(the second must hit the prefix cache). Prints the request shape, the engine's architecture and the reply.

  python agent_smoke.py CODE_DIR DATA_DIR AC_DIR READER_DIR
"""
import json
import os
import sys
from pathlib import Path


def main():
    code, data, ac_dir, reader_dir = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    for sub in ("kit", "rq_kit", "bench"):
        sys.path.insert(0, str(code / sub))
    os.environ["SEC45_K"] = "200"
    import sec45_common as S
    S.setup_environment("res")
    import rq_common as C
    from activation.agent import Agent, AgentConfig
    from activation.common.ac_parts import ac_part
    from activation.harness import FREE_DEVICE
    from activation.harness.vllm_wrapper import VLLMWrapper
    from run_qa import AC, NAME, harness

    h = harness(ac_dir, reader_dir, summary_layout=S.LAYOUT)
    ac_model = h.module_manager.get_ac_model(AC)
    loaded = h.loaded_models[NAME]
    loaded.model_config.engine_kwargs = VLLMWrapper.recommended_engine_kwargs("Qwen/Qwen3.5-9B") | {"kv_cache_dtype": "auto"}   # RTX PRO 6000: bf16 KV
    row = S.load_tasks(data, "dev1000")[0]
    passages = C.passages_block(C.load_corpus(data), S.row_passages(row))
    part = ac_part([{"role": "user", "content": passages}], AC, 0.1)
    config = AgentConfig(system_prompt="Answer the question using the passages. End your reply with one line of the form\nFinal answer: <short answer>",
                         user_prompt="Question: " + row["question"], messages_input=[{"role": "user", "content": [part]}],
                         model_name=NAME, ac_model_name=AC, call_kwargs={"sampling_params": {"max_tokens": 96, "temperature": 0.0}}, max_turns=1)
    loaded.ensure_engine_loaded()
    engine = loaded.model_config.engine_kwargs
    replica = loaded.vllm_model.replicas[0].engine
    print("agent_smoke: engine hf_overrides", json.dumps(engine.get("hf_overrides")), "enable_lora", engine.get("enable_lora"),
          "architecture", replica.model_config.architectures, "inputs_embeds_size", replica.model_config.get_inputs_embeds_size(), flush=True)
    agent = Agent(h, config, seed=0)
    agent.step_mode = True
    agent.begin()
    prefix, embeds, mask = agent.prepare_request()
    print(f"agent_smoke: deep {ac_model.is_deep} layers {ac_model.intervention_layers}; prompt {len(prefix)} tokens, rows at {mask.count(False)} "
          f"positions, request tensor {tuple(embeds.shape)} {embeds.dtype} ({embeds.numel() * embeds.element_size() / 2**20:.1f} MiB); "
          f"recorded span rows {tuple(agent.run_results.prompt_ac_spans[0]['rows'].shape)}", flush=True)
    first = agent.submit_probe()
    second = agent.submit_probe()
    print(f"agent_smoke: cached {first.cached_prompt_token_count} -> {second.cached_prompt_token_count}; same reply {first.text == second.text}; "
          f"reply {first.text!r}; gold {row['answer']!r}", flush=True)
    assert embeds.shape[1] == (1 + len(ac_model.intervention_layers)) * loaded.model_config.model_description.d_model
    assert second.cached_prompt_token_count > 0
    print("agent_smoke: ok", flush=True)
    loaded.engine_to_device(FREE_DEVICE)                        # shut the engine core down (os._exit alone orphans it on the GPU)
    os._exit(0)                                              # the AC encode queue thread is a daemon; skip its teardown


if __name__ == "__main__":
    main()
