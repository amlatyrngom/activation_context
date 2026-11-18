"""
Math synthesis dev evaluation (2026-11-17), ONE code path for every arm (GPU node): the harness's reader engine (vLLM 0.28 with the
Hidden Residuals model class: activation/vllm_residuals) and an Agent per request (step mode, one turn, no tools, no system message),
as the AC arm needs it; the text arms go through the very same engine, Agent and request path with no part in the prompt.

Arms (--arms, comma separated):
  results    the paper's results-only prompt (build_data.py condition "results"), the untuned 9B
  full       the full-runs prompt (the teacher's view), the untuned 9B
  ac_r10     the full-runs layout with each run replaced by its AC part at ratio 1/10 (rows = ceil(encoder tokens x 0.1)), read by the
  ac_r5      pair's frozen reader (adapter all zeros = the untuned 9B); 1/5 the same at 0.2
Text arms are checked token for token against build_data.py's prompt ids (the teacher test's chat text).
The pair (--pair AC READER) is required for every arm: it builds the engine (residual architecture) and, for AC arms, encodes the runs.

Decoding, the same for every arm (the teacher test's): temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5,
max_tokens 64,000 (less only if the prompt leaves less room in max_model_len 160,000), thinking on, SamplingParams(seed = repeat),
repeats --seeds (0,1,2). bf16 KV cache. Scoring is not done here (score_math.py on the host: the teacher-test scorer).
Every finished request is appended to <out>/gen_<arm>.jsonl at once; a restart skips the ids already there.

    python math_eval.py --inputs eval_inputs_dev.jsonl --pair PAIR/ac PAIR/reader --arms results,full --out OUT [--seeds 0,1,2]
"""
import argparse
import json
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
RATIOS = {"ac_r10": 0.1, "ac_r5": 0.2}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--pair", nargs=2, required=True, metavar=("AC", "READER"))
    p.add_argument("--arms", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--max-tokens", type=int, default=64_000)
    p.add_argument("--max-model-len", type=int, default=160_000)
    p.add_argument("--concurrency", type=int, default=96, help="requests in flight (threads)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                   help="engine share before the harness lowers it by its AC side reservation (0.1); 0.9 for the text baselines, an AC arm on a 96 GB RTX PRO 6000 needs less (2026-11-18: 0.8 effective OOMed in the AC encode)")
    p.add_argument("--label", default="")
    a = p.parse_args()
    kit = os.environ.get("SEC45_KIT")
    if kit:
        sys.path.insert(0, kit)
    import sec45_common as S
    S.setup_environment("res")
    from activation.agent import Agent, AgentConfig
    from activation.agent.agent_tools import DEFAULT_TOOLS
    from activation.common.ac_parts import ac_part
    from activation.harness import FREE_DEVICE
    from run_qa import AC, NAME, harness
    a.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    arms = a.arms.split(",")
    assert all(arm in ("results", "full", *RATIOS) for arm in arms), arms
    problems = [json.loads(line) for line in open(a.inputs) if line.strip()]
    problems = problems[:a.limit] if a.limit else problems
    S.check_pair_mode(a.pair[0], a.pair[1], "res")
    h = harness(a.pair[0], a.pair[1], summary_layout=S.LAYOUT)
    loaded = h.loaded_models[NAME]
    loaded.model_config.engine_kwargs = {"max_model_len": a.max_model_len, "limit_mm_per_prompt": {"image": 0, "video": 0}, "kv_cache_dtype": "auto",
                                         "gpu_memory_utilization": a.gpu_memory_utilization}
    clock = time.time()
    loaded.ensure_engine_loaded()
    engine = loaded.model_config.engine_kwargs
    replica = loaded.vllm_model.replicas[0].engine
    meta = {"label": a.label, "pair": a.pair, "host": socket.gethostname(), "sampling": SAMPLING, "max_tokens": a.max_tokens, "max_model_len": a.max_model_len,
            "engine_architecture": list(replica.model_config.architectures), "inputs_embeds_size": replica.model_config.get_inputs_embeds_size(),
            "hf_overrides": engine.get("hf_overrides"), "gpu_memory_utilization": engine.get("gpu_memory_utilization"), "enable_lora": engine.get("enable_lora"), "engine_load_seconds": round(time.time() - clock, 1)}
    try:
        import torch
        meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    print(f"math_eval - engine up: {json.dumps(meta)}", flush=True)
    (a.out / f"meta_{int(time.time())}.json").write_text(json.dumps(meta, indent=1))
    tokenizer = loaded.tokenizer

    def config_for(problem: dict, arm: str, seed: int, max_tokens: int):
        if arm in RATIOS:
            t, runs = problem["texts"], problem["runs"]
            content = [{"type": "text", "text": t[0]}]
            for run, after in zip(runs, t[1:]):
                content += [ac_part([{"role": "user", "content": run}], AC, RATIOS[arm]), {"type": "text", "text": after}]
            messages, ac_name = [{"role": "user", "content": content}], AC
        else:
            messages, ac_name = [{"role": "user", "content": problem["prompts"][arm]["text"]}], None
        config = AgentConfig(system_prompt="", user_prompt="", messages_input=messages, model_name=NAME, ac_model_name=ac_name, lora_name=None,
                             call_kwargs={"sampling_params": {**SAMPLING, "max_tokens": max_tokens}, "chat_template_kwargs": {"enable_thinking": True}},
                             max_turns=1, absolute_trajectory_cap=a.max_model_len)
        config.tools = {key: None for key in set(DEFAULT_TOOLS) | set(config.tools)}
        return config

    lock = threading.Lock()

    def one(problem: dict, arm: str, seed: int, handle) -> str:
        started = time.time()
        room = a.max_model_len - (len(problem["prompts"]["full"]["ids"]) if arm == "full" else 0) - 8
        max_tokens = min(a.max_tokens, room) if arm == "full" else a.max_tokens
        agent = Agent(h, config_for(problem, arm, seed, max_tokens), seed=seed)
        agent.step_mode = True
        agent.begin()
        if arm in ("results", "full") and agent.prefix != problem["prompts"][arm]["ids"]:
            raise ValueError(f"{arm}/{problem['task_id']}: the harness prompt differs from build_data.py's prompt ids")
        rows = sum(end - start for start, end in agent.spans)
        output = agent.submit_probe()
        row = {"id": f"{arm}/{problem['task_id']}/s{seed}", "arm": arm, "task_id": problem["task_id"], "seed": seed, "k": problem["k"],
               "dataset_task": problem["dataset_task"], "solver_results": problem["solver_results"], "prompt_tokens": len(agent.prefix), "rows": rows,
               "parts": len(agent.spans), "max_tokens": max_tokens, "generated_tokens": output.output_token_count, "finish": str(output.finish_reason),
               "text": output.text, "token_ids": list(output.token_ids), "cached_prompt_tokens": output.cached_prompt_token_count,
               "seconds": round(time.time() - started, 1)}
        with lock:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        return row["id"]

    try:
        for arm in arms:
            path = a.out / f"gen_{arm}.jsonl"
            done = set()
            if path.exists():
                good = []
                for line in open(path):
                    try:
                        done.add(json.loads(line)["id"]); good.append(line if line.endswith("\n") else line + "\n")
                    except ValueError:
                        pass
                path.write_text("".join(good))
            todo = [(pr, s) for pr in problems for s in seeds if f"{arm}/{pr['task_id']}/s{s}" not in done]
            print(f"math_eval - {arm}: {len(todo)} to run, {len(done)} done before", flush=True)
            started, finished = time.time(), 0
            with open(path, "a") as handle, ThreadPoolExecutor(a.concurrency) as pool:
                futures = [pool.submit(one, pr, arm, s, handle) for pr, s in todo]
                for future in as_completed(futures):
                    future.result()
                    finished += 1
                    if finished % 50 == 0:
                        print(f"math_eval - {arm}: {finished}/{len(todo)} in {time.time() - started:.0f}s", flush=True)
            with open(a.out / "timing.jsonl", "a") as f:
                f.write(json.dumps({"arm": arm, "label": a.label, "requests": len(todo), "done_before": len(done), "segment_seconds": round(time.time() - started, 1),
                                    "gpu": meta.get("gpu"), "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
            print(f"math_eval - {arm}: {len(todo)} requests in {time.time() - started:.0f}s", flush=True)
    finally:
        loaded.engine_to_device(FREE_DEVICE)
    os._exit(0)                                                 # the AC encode queue thread is a daemon; skip its teardown


if __name__ == "__main__":
    main()
