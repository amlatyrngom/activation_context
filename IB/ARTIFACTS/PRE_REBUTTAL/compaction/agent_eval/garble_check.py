"""
Is the AC arm's broken tool-call syntax (e.g. "Let me look ...\n[/shell]\n<parameter=script>" instead of "<tool_call>\n<function=shell>")
the model's or the engine's? For first assistant turns of an ac_<rows> rollout cache that carry no parsed tool call and contain "\n[", rebuild the
first prompt exactly as run_resume.py does (same encoder, same rows and deltas), then teacher-force the turn vLLM sampled through
the HF reader with the intervention hooks and compare per-token log-probs with the ones vLLM recorded for those tokens (raw
log-probs, before temperature / top-p). At the first token of the broken call (the first sampled token whose text starts with
"[" after a newline) it prints both log-probs, HF's top-5 there and HF's log-prob of "<tool_call>".

    python garble_check.py --split split.json --prefixes PREFIXES --ac-checkpoint PAIR/ac --reader-checkpoint PAIR/reader \
        --rollouts CACHE/rollouts.jsonl --rows 2000 --count 6 --out OUT.json
"""
import argparse
import json
import os
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True); p.add_argument("--prefixes", required=True)
    p.add_argument("--ac-checkpoint", required=True); p.add_argument("--reader-checkpoint", required=True)
    p.add_argument("--rollouts", required=True); p.add_argument("--rows", type=int, required=True)
    p.add_argument("--count", type=int, default=6); p.add_argument("--out", required=True)
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    import run_resume as R
    import run_resume_res
    from activation.agent.agent import Agent
    from activation.agent_training.agent_training_teacher_study import AgentTeacherKind, AgentTrainingStudy
    from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
    from resume_program import ResumeFromPrefix
    from run_pool import NO_COMPACTION, SINGLE_AGENT_SYSTEM_PROMPT

    chosen = []
    with open(a.rollouts) as handle:
        for line in handle:
            row = json.loads(line)
            first = next((step for step in row["trajectory"] if step.get("role") == "assistant"), None)
            if first is not None and not first.get("tool_calls") and "\n[" in first.get("content", "") and first.get("logprobs"):
                chosen.append({"task_id": row["config_key"].split("/")[1], "tokens": list(first["token_ids"]), "vllm": list(first["logprobs"]),
                               "prompt": list(row.get("prompt_token_ids") or []), "finish": row["finish_reason"]})
            if len(chosen) >= a.count:
                break
    by_task = {entry["task_id"]: entry for entry in json.loads((Path(a.prefixes) / "index.json").read_text())["prefixes"]}
    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={R.ORACLE_MODEL_NAME: ModelConfig(R.ORACLE_MODEL_NAME, R.READER_MODEL_ID),
                                                                 R.SIDE_BASE_NAME: ModelConfig(R.SIDE_BASE_NAME, "Qwen/Qwen3.5-2B")},
                                                  agent_max_concurrent=None, ac_engine_memory_reservation=0.4))
    run_resume_res.register_pair(harness, a.ac_checkpoint, a.reader_checkpoint, rows_only=False, encode_batch_tokens=32_000)
    ac = harness.module_manager.get_ac_model(R.AC_NAME)
    layers = list(ac.intervention_layers)
    benchmarks = sorted({by_task[c["task_id"]]["benchmark"] for c in chosen})
    study = AgentTrainingStudy(harness, {name: 1.0 for name in benchmarks}, reasoning="medium", max_turns=150, max_duration=6000,
                               compaction_threshold_tokens=NO_COMPACTION, system_prompt=SINGLE_AGENT_SYSTEM_PROMPT)
    study.setup_reusable_artifacts(10 ** 6, seed=0)
    loaded = harness.loaded_models[R.ORACLE_MODEL_NAME]
    d_model = loaded.model_config.model_description.d_model
    tokenizer = loaded.tokenizer
    manager = harness.module_manager
    manager.ensure_lora(R.READER_ADAPTER_NAME).eval()
    from activation.harness.hf_utils import TARGET_DEVICE
    loaded.model_to_device(TARGET_DEVICE)
    model = loaded.model
    tool_call_id = tokenizer.convert_tokens_to_ids("<tool_call>")
    embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
    device = embedding.weight.device
    report = {"rollouts": a.rollouts, "rows": a.rows, "items": []}
    for case in chosen:
        entry = by_task[case["task_id"]]
        task = study.datasets[entry["benchmark"]].scorable_tasks[entry["task_id"]]
        config = study._config_for(entry["benchmark"], task, AgentTeacherKind.BASE, 0)
        config.tools = {**config.tools, "compact": None, "subagent": None}
        config.absolute_trajectory_cap = 90_000
        config.ac_model_name, config.lora_name = R.AC_NAME, R.READER_ADAPTER_NAME
        agent = Agent(harness, config)
        agent.step_mode = True
        ResumeFromPrefix(agent, prefix_file=str(Path(a.prefixes) / entry["file"]), variant="ac", keep_tokens=a.rows, budget=50_000).install()
        prefix, packed_all, mask = agent.prepare_request()
        same_prompt = (not case["prompt"]) or case["prompt"] == list(prefix)
        ids = list(prefix) + case["tokens"]
        with torch.no_grad():
            embeds = embedding(torch.tensor(ids, device=device)).to(torch.bfloat16)
            positions, deltas = [], {layer: [] for layer in layers}
            for start, end in agent.spans:
                packed = packed_all[start:end].to(device)
                embeds[start:end] = packed[:, :d_model]
                positions += list(range(start, end))
                for slot, layer in enumerate(layers):
                    deltas[layer].append(packed[:, (1 + slot) * d_model:(2 + slot) * d_model])
            payload = {"positions": positions, "layer_inputs": {layer: torch.cat(parts) for layer, parts in deltas.items()}}
            hidden = loaded.decoder_forward(embeds[None], None, lora_name=None, interventions=[payload])[0]
            logits = head(hidden[len(prefix) - 1:len(ids) - 1]).float()
            logprobs = F.log_softmax(logits, dim=-1)
        hf = [float(logprobs[i, token]) for i, token in enumerate(case["tokens"])]
        diffs = [abs(x - y) for x, y in zip(hf, case["vllm"])]
        pieces = [tokenizer.decode([token]) for token in case["tokens"]]
        text, starts = "", []
        for piece in pieces:
            starts.append(len(text))
            text += piece
        where = text.find("\n[")
        garble = None if where < 0 else max(i for i, start in enumerate(starts) if start <= where + 1)   # the token holding the "["
        text_so_far = text[:starts[garble]] if garble is not None else text
        item = {"task_id": case["task_id"], "finish": case["finish"], "same_prompt_as_recorded": same_prompt, "turn_tokens": len(case["tokens"]),
                "mean_abs_dlogprob": round(sum(diffs) / len(diffs), 4), "max_abs_dlogprob": round(max(diffs), 4),
                "vllm_turn_logprob": round(sum(case["vllm"]), 2), "hf_turn_logprob": round(sum(hf), 2)}
        if garble is not None:
            top = torch.topk(logprobs[garble], 5)
            item["garble"] = {"index": garble, "token": pieces[garble], "vllm_logprob": round(case["vllm"][garble], 3), "hf_logprob": round(hf[garble], 3),
                              "hf_logprob_tool_call_token": round(float(logprobs[garble, tool_call_id]), 3),
                              "hf_top5": [(tokenizer.decode([int(t)]), round(float(v), 3)) for v, t in zip(top.values, top.indices)],
                              "text_before": text_so_far[-120:]}
        report["items"].append(item)
        print(f"garble_check: {json.dumps(item)}", flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("garble_check: done", flush=True)
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
