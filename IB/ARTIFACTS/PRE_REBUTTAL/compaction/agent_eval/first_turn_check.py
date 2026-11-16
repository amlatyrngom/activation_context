"""
First-turn sanity of the AC arm served through the vLLM residual path (one GPU, before the agent smoke). For the first N dev-75
prefixes, the ac_<rows> arm's first prompt is built exactly as run_resume.py builds it (ResumeFromPrefix.install in step mode: no
sandbox, nothing sampled; the rows and deltas come from the agent's own encode), then:

  vLLM (greedy, --max-new tokens, top-20 log-probs, prefix cache reset before every request):
    trim   the request as the agent sends it (tensor cut after the last AC row; vllm_residuals.allow_short_prompt_embeds)
    full   the same request with the tensor zero-padded to full length (vLLM's stock rule)    -> must equal trim
    rows   full length with every delta zeroed (rows only)                                      -> scale: what the deltas change
  HF (after the engine is freed): the reader with the intervention hooks (harness.hf_utils.generate_with_interventions), the
    same token ids, rows and deltas, greedy.                                                   -> should match vLLM within engine noise

Printed and written to --out: first-position max/mean |delta log-prob| over vLLM's top-20 (trim vs full, trim vs rows, trim vs HF),
top-1 agreement and the length of the common greedy prefix.

    python first_turn_check.py --split split.json --prefixes PREFIXES --ac-checkpoint PAIR/ac --reader-checkpoint PAIR/reader \
        --rows 4000 --count 2 --out OUT.json
"""
import argparse
import json
import os
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True); p.add_argument("--prefixes", required=True)
    p.add_argument("--ac-checkpoint", required=True); p.add_argument("--reader-checkpoint", required=True)
    p.add_argument("--rows", type=int, default=4000); p.add_argument("--count", type=int, default=2)
    p.add_argument("--max-new", type=int, default=48); p.add_argument("--max-model-len", type=int, default=100_000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    import vllm
    import run_resume
    import run_resume_res
    from activation.agent.agent import Agent
    from activation.agent_training.agent_training_teacher_study import AgentTeacherKind, AgentTrainingStudy
    from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig
    from activation.harness.hf_utils import generate_with_interventions
    from activation.harness.vllm_wrapper import VLLMWrapper
    from resume_program import ResumeFromPrefix
    from run_pool import NO_COMPACTION, SINGLE_AGENT_SYSTEM_PROMPT
    R = run_resume
    kv = os.environ.get("RESUME_KV_CACHE_DTYPE")
    engine_kwargs = VLLMWrapper.recommended_engine_kwargs(R.READER_MODEL_ID) | {"max_model_len": a.max_model_len} | ({"kv_cache_dtype": kv} if kv else {})
    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={R.ORACLE_MODEL_NAME: ModelConfig(R.ORACLE_MODEL_NAME, R.READER_MODEL_ID, engine_kwargs=engine_kwargs),
                                                                 R.SIDE_BASE_NAME: ModelConfig(R.SIDE_BASE_NAME, "Qwen/Qwen3.5-2B")},
                                                  agent_max_concurrent=None, ac_engine_memory_reservation=0.4))
    run_resume_res.register_pair(harness, a.ac_checkpoint, a.reader_checkpoint, rows_only=False, encode_batch_tokens=32_000)
    ac = harness.module_manager.get_ac_model(R.AC_NAME)
    layers = list(ac.intervention_layers)
    index = json.loads((Path(a.prefixes) / "index.json").read_text())["prefixes"][:a.count]
    benchmarks = sorted({entry["benchmark"] for entry in index})
    study = AgentTrainingStudy(harness, {name: 1.0 for name in benchmarks}, reasoning="medium", max_turns=150, max_duration=6000,
                               compaction_threshold_tokens=NO_COMPACTION, system_prompt=SINGLE_AGENT_SYSTEM_PROMPT)
    study.setup_reusable_artifacts(10 ** 6, seed=0)
    loaded = harness.loaded_models[R.ORACLE_MODEL_NAME]
    d_model = loaded.model_config.model_description.d_model
    prompts = []
    for entry in index:
        task = study.datasets[entry["benchmark"]].scorable_tasks[entry["task_id"]]
        config = study._config_for(entry["benchmark"], task, AgentTeacherKind.BASE, 0)
        config.tools = {**config.tools, "compact": None, "subagent": None}
        config.absolute_trajectory_cap = 90_000
        config.ac_model_name, config.lora_name = R.AC_NAME, R.READER_ADAPTER_NAME
        agent = Agent(harness, config)
        agent.step_mode = True
        note = ResumeFromPrefix(agent, prefix_file=str(Path(a.prefixes) / entry["file"]), variant="ac", keep_tokens=a.rows, budget=50_000).install()
        prefix, trimmed, mask = agent.prepare_request()
        full = torch.zeros(len(prefix), trimmed.shape[1], dtype=trimmed.dtype)
        full[:len(trimmed)] = trimmed
        rows_only = full.clone()
        rows_only[:, d_model:] = 0
        prompts.append({"task_id": entry["task_id"], "note": note, "prefix": prefix, "mask": mask, "trim": trimmed, "full": full, "rows": rows_only,
                        "spans": list(agent.spans)})
        print(f"first_turn_check: {entry['task_id']} prompt {len(prefix)} tokens, rows {mask.count(False)} at {agent.spans}, "
              f"request tensor trim {tuple(trimmed.shape)} ({trimmed.numel() * 2 / 1e6:.1f} MB) vs full {tuple(full.shape)} ({full.numel() * 2 / 1e6:.1f} MB)", flush=True)
    loaded.ensure_engine_loaded()
    run_resume_res.log_engine(loaded)
    wrapper = loaded.vllm_model
    engine = wrapper.replicas[0].engine
    params = vllm.SamplingParams(temperature=0.0, max_tokens=a.max_new, logprobs=20, seed=0)
    for prompt in prompts:
        prompt["vllm"] = {}
        for mode in ("trim", "full", "rows"):
            wrapper.replicas[0].run(engine.reset_prefix_cache())
            started = time.monotonic()
            output = wrapper.submit(prompt["prefix"], params, prompt_embeds=prompt[mode], prompt_is_token_ids=prompt["mask"]).result()
            completion = output.outputs[0]
            first = {int(token): float(item.logprob) for token, item in completion.logprobs[0].items()}
            prompt["vllm"][mode] = {"tokens": list(completion.token_ids), "first": first, "seconds": round(time.monotonic() - started, 2)}
    loaded.engine_to_device(FREE_DEVICE)

    hf_error = None
    try:
        manager = harness.module_manager
        peft_model = manager.ensure_lora(R.READER_ADAPTER_NAME)
        peft_model.eval()
        embedding = loaded.model.get_input_embeddings()
        device = embedding.weight.device
        eos = peft_model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        for prompt in prompts:
            ids = torch.tensor(prompt["prefix"], device=device)
            with torch.no_grad(), manager.lora_context(R.ORACLE_MODEL_NAME, None):
                embeds = embedding(ids).to(torch.bfloat16)
                positions, deltas = [], {layer: [] for layer in layers}
                for start, end in prompt["spans"]:
                    packed = prompt["trim"][start:end].to(device)
                    embeds[start:end] = packed[:, :d_model]
                    positions += list(range(start, end))
                    for slot, layer in enumerate(layers):
                        deltas[layer].append(packed[:, (1 + slot) * d_model:(2 + slot) * d_model])
                payload = {"positions": positions, "layer_inputs": {layer: torch.cat(parts) for layer, parts in deltas.items()}}
                out = generate_with_interventions(peft_model, inputs_embeds=embeds[None], interventions=[payload], max_new_tokens=a.max_new,
                                                  do_sample=False, use_cache=True, eos_token_id=eos, pad_token_id=eos[0],
                                                  return_dict_in_generate=True, output_logits=True)
            logprobs = F.log_softmax(out.logits[0][0].float(), dim=-1).cpu()
            prompt["hf"] = {"tokens": out.sequences[0].tolist(), "first": {token: float(logprobs[token]) for token in prompt["vllm"]["trim"]["first"]},
                            "top1": int(logprobs.argmax())}
    except Exception as error:                                         # the vLLM comparisons stand on their own
        hf_error = repr(error)
        print(f"first_turn_check: HF reference failed: {hf_error}", flush=True)

    def compare(reference: dict, other_first: dict, other_tokens: list[int], other_top1: int | None = None) -> dict:
        diffs = [abs(reference["first"][token] - other_first[token]) for token in reference["first"] if token in other_first]
        common = 0
        for x, y in zip(reference["tokens"], other_tokens):
            if x != y:
                break
            common += 1
        top1 = max(reference["first"], key=reference["first"].get)
        other = other_top1 if other_top1 is not None else max(other_first, key=other_first.get)
        return {"max_abs_dlogprob": round(max(diffs), 4) if diffs else None, "mean_abs_dlogprob": round(sum(diffs) / len(diffs), 4) if diffs else None,
                "compared_tokens": len(diffs), "top1_agree": top1 == other, "common_greedy_prefix": common, "of": len(reference["tokens"])}

    report = {"pair": [a.ac_checkpoint, a.reader_checkpoint], "rows": a.rows, "layers": layers, "kv_cache_dtype": engine_kwargs.get("kv_cache_dtype"),
              "max_new": a.max_new, "hf_error": hf_error, "items": []}
    for prompt in prompts:
        trim = prompt["vllm"]["trim"]
        item = {"task_id": prompt["task_id"], "prompt_tokens": len(prompt["prefix"]), "rows": prompt["mask"].count(False),
                "trim_mb": round(prompt["trim"].numel() * 2 / 1e6, 1), "full_mb": round(prompt["full"].numel() * 2 / 1e6, 1),
                "trim_vs_full": compare(trim, prompt["vllm"]["full"]["first"], prompt["vllm"]["full"]["tokens"]),
                "trim_vs_rows_only": compare(trim, prompt["vllm"]["rows"]["first"], prompt["vllm"]["rows"]["tokens"]),
                "vllm_seconds": {mode: prompt["vllm"][mode]["seconds"] for mode in prompt["vllm"]}}
        if "hf" in prompt:
            hf_tokens = prompt["hf"]["tokens"]
            item["trim_vs_hf"] = compare(trim, prompt["hf"]["first"], hf_tokens, prompt["hf"]["top1"])
        report["items"].append(item)
        print(f"first_turn_check: {json.dumps(item)}", flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("first_turn_check: done", flush=True)
    os._exit(0)                                                        # the AC encode queue thread is a daemon; skip its teardown


if __name__ == "__main__":
    raise SystemExit(main())
