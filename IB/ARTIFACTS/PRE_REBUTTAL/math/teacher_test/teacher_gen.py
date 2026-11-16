"""
Math teacher test (2026-11-15), step 2 (GPU node): sample the synthesizer replies.

Reads the prompts written by build_prompts.py (chat text already tokenized with the 9B tokenizer) and, for each arm
NAME:CONDITION in --arms, sends every problem once per repeat seed (--seeds; the same seeds for every arm) to one plain vLLM
engine of --model. Arms run one after another, so each arm's wall-clock is its own. Sampling as the paper's synthesis arms
(run_synth1call_v2.SAMPLING: temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5), thinking on (in the
chat text), max_tokens --max-tokens, max_model_len --max-model-len, bf16 KV cache (RTX PRO 6000: fp8 KV needs FlashInfer).
Every finished request is appended to <out>/gen_<arm>.jsonl at once; a restart skips the ids already there (spot resume).
Each engine segment appends its wall-clock to <out>/timing.jsonl. No scoring here (score_teacher.py, on the host).

    python teacher_gen.py --prompts prompts.jsonl --out OUT --model Qwen/Qwen3.5-9B --arms 9b_full:full,9b_results:results
"""
import argparse
import json
import socket
import time
from pathlib import Path

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arms", required=True, help="NAME:CONDITION[,NAME:CONDITION...], run in this order")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-tokens", type=int, default=64_000)
    parser.add_argument("--max-model-len", type=int, default=160_000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None, help="first N tasks only (smoke test)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    prompts = [json.loads(line) for line in open(args.prompts) if line.strip()]
    arms = [spec.split(":", 1) for spec in args.arms.split(",")]
    plan = {}
    for name, condition in arms:
        rows = [p for p in prompts if p["condition"] == condition]
        rows = rows[:args.limit] if args.limit else rows
        path = args.out / f"gen_{name}.jsonl"
        done = {json.loads(line)["id"] for line in open(path) if line.strip()} if path.exists() else set()
        todo = [(row, seed) for row in rows for seed in seeds if f"{name}/{row['task_id']}/s{seed}" not in done]
        plan[name] = (condition, path, todo, len(rows) * len(seeds), len(done))
        print(f"teacher_gen - {name} ({condition}): {len(todo)} to run, {len(done)} done before", flush=True)
    if not any(todo for _, _, todo, _, _ in plan.values()):
        print("teacher_gen - nothing to run", flush=True)
        return 0

    import vllm
    from vllm import LLM, SamplingParams
    clock = time.time()
    llm = LLM(model=args.model, max_model_len=args.max_model_len, kv_cache_dtype="auto", gpu_memory_utilization=args.gpu_memory_utilization,
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)
    load_seconds = time.time() - clock
    engine = llm.llm_engine
    meta = {"model": args.model, "vllm": vllm.__version__, "host": socket.gethostname(), "sampling": SAMPLING, "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len, "kv_cache_dtype": "auto", "seeds": seeds, "engine_load_seconds": round(load_seconds, 1)}
    try:
        import torch
        meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    print(f"teacher_gen - engine up in {load_seconds:.0f}s: {json.dumps(meta)}", flush=True)
    for name, (condition, path, todo, total, done_before) in plan.items():
        if not todo:
            continue
        by_id = {}
        for row, seed in todo:
            request_id = f"{name}/{row['task_id']}/s{seed}"
            by_id[request_id] = (row, seed)
            engine.add_request(request_id, {"prompt_token_ids": row["prompt_token_ids"]},
                               SamplingParams(**SAMPLING, max_tokens=args.max_tokens, seed=seed))
        started, finished = time.time(), 0
        with open(path, "a") as handle:
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if not output.finished:
                        continue
                    row, seed = by_id.pop(output.request_id)
                    completion = output.outputs[0]
                    handle.write(json.dumps({"id": output.request_id, "arm": name, "condition": condition, "model": args.model, "task_id": row["task_id"],
                                             "seed": seed, "k": row["k"], "dataset_task": row["dataset_task"], "solver_results": row["solver_results"],
                                             "prompt_tokens": len(row["prompt_token_ids"]), "generated_tokens": len(completion.token_ids),
                                             "finish": str(completion.finish_reason), "text": completion.text,
                                             "seconds": round(time.time() - started, 1)}, ensure_ascii=False) + "\n")
                    handle.flush()
                    finished += 1
                    if finished % 50 == 0:
                        print(f"teacher_gen - {name}: {finished}/{len(todo)} in {time.time() - started:.0f}s", flush=True)
        seconds = time.time() - started
        with open(args.out / "timing.jsonl", "a") as handle:
            handle.write(json.dumps({"arm": name, "condition": condition, **meta, "requests": len(todo), "done_before": done_before, "total": total,
                                     "segment_seconds": round(seconds, 1), "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        print(f"teacher_gen - {name}: {len(todo)} requests in {seconds:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
