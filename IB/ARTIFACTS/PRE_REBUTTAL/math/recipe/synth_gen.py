"""
Math clean recipe (2026-11-17): sample the TEACHER's syntheses (GPU node, plain vLLM). The teacher is the untuned Qwen/Qwen3.5-9B reading
the three full solver runs (condition "full" of build_data.py's prompts: math/TEACHER_TEST.md's "9B, full runs" arm, same prompt).
These samples are (a) the stage-1 KL targets (teacher-forced, every sample that stopped) and (b) the pool of stage 2 (shortest
correct of the k samples, plus any within 1.3x of it), and on dev (seed 0) the held targets of the teacher-forced dev panel.

Same settings as the teacher test (teacher_test/teacher_gen.py): one plain vLLM engine, bf16 KV cache, max_model_len 160,000,
sampling temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5, max_tokens 64,000, SamplingParams(seed=repeat).
The prompts are the chat text already tokenized with the 9B tokenizer (enable_thinking=True). Additions to teacher_gen.py:
  - the sampled token ids are recorded (`token_ids`; the KL / SFT targets are exactly these tokens), with the stop token when vLLM
    returns it, and `stop_token_appended` False (the packet builder appends <|im_end|> to a stopped reply that lacks it);
  - --shard I/N takes every N-th problem (one process per GPU), --tasks-file restricts to listed task ids;
  - a prompt longer than max_model_len - max_tokens gets max_tokens = max_model_len - prompt (recorded as `max_tokens`; 2 of the
    1,500 training prompts are over 96,000 tokens).
Every finished request is appended to <out>/gen_<arm>.<shard>.jsonl at once; a restart skips ids already there (preemption resume).

    python synth_gen.py --prompts prompts_train.jsonl --out OUT --arm teacher_train --condition full --seeds 0,1,2 --shard 0/2
"""
import argparse
import json
import socket
import time
from pathlib import Path

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
MODEL = "Qwen/Qwen3.5-9B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--condition", default="full")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--tasks-file", type=Path, default=None, help="JSON list of task ids to run (default: all)")
    parser.add_argument("--max-tokens", type=int, default=64_000)
    parser.add_argument("--max-model-len", type=int, default=160_000)
    parser.add_argument("--model", default=MODEL, help="the generator (default the 9B teacher; dev_hard 2026-11-18: unsloth/Qwen3.8-27B-NVFP4, the teacher test's 27B)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    index, count = (int(x) for x in args.shard.split("/"))
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    wanted = set(json.loads(args.tasks_file.read_text())) if args.tasks_file else None
    prompts = [json.loads(line) for line in open(args.prompts) if line.strip()]
    prompts = sorted((p for p in prompts if p["condition"] == args.condition and (wanted is None or p["task_id"] in wanted)), key=lambda p: p["task_id"])
    prompts = prompts[index::count][:args.limit] if args.limit else prompts[index::count]
    path = args.out / f"gen_{args.arm}.{index}of{count}.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try:
                done.add(json.loads(line)["id"])
            except ValueError:
                pass                                            # a line cut by a preemption: rewritten below
        with open(path) as f:
            good = [line for line in f if line.strip() and line.endswith("\n") and _ok(line)]
        path.write_text("".join(good))
    todo = [(row, seed) for row in prompts for seed in seeds if f"{args.arm}/{row['task_id']}/s{seed}" not in done]
    print(f"synth_gen - {args.arm} ({args.condition}) shard {args.shard}: {len(prompts)} prompts x {len(seeds)} seeds, {len(todo)} to run, {len(done)} done before", flush=True)
    if not todo:
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
    print(f"synth_gen - engine up in {load_seconds:.0f}s: {json.dumps(meta)}", flush=True)
    by_id = {}
    for row, seed in todo:
        request_id = f"{args.arm}/{row['task_id']}/s{seed}"
        max_tokens = min(args.max_tokens, args.max_model_len - len(row["prompt_token_ids"]))
        by_id[request_id] = (row, seed, max_tokens)
        engine.add_request(request_id, {"prompt_token_ids": row["prompt_token_ids"]}, SamplingParams(**SAMPLING, max_tokens=max_tokens, seed=seed))
    started, finished = time.time(), 0
    with open(path, "a") as handle:
        while engine.has_unfinished_requests():
            for output in engine.step():
                if not output.finished:
                    continue
                row, seed, max_tokens = by_id.pop(output.request_id)
                completion = output.outputs[0]
                handle.write(json.dumps({"id": output.request_id, "arm": args.arm, "condition": args.condition, "model": args.model, "task_id": row["task_id"],
                                         "seed": seed, "k": row["k"], "dataset_task": row["dataset_task"], "solver_results": row["solver_results"],
                                         "prompt_tokens": len(row["prompt_token_ids"]), "chat_text_sha256": row["chat_text_sha256"], "max_tokens": max_tokens,
                                         "generated_tokens": len(completion.token_ids), "finish": str(completion.finish_reason),
                                         "stop_reason": completion.stop_reason if isinstance(completion.stop_reason, (int, str, type(None))) else str(completion.stop_reason),
                                         "text": completion.text, "token_ids": list(completion.token_ids), "seconds": round(time.time() - started, 1)},
                                        ensure_ascii=False) + "\n")
                handle.flush()
                finished += 1
                if finished % 50 == 0:
                    print(f"synth_gen - {args.arm}: {finished}/{len(todo)} in {time.time() - started:.0f}s", flush=True)
    seconds = time.time() - started
    with open(args.out / "timing.jsonl", "a") as handle:
        handle.write(json.dumps({"arm": args.arm, "shard": args.shard, **meta, "requests": len(todo), "done_before": len(done),
                                 "segment_seconds": round(seconds, 1), "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    print(f"synth_gen - {args.arm}: {len(todo)} requests in {seconds:.0f}s", flush=True)
    return 0


def _ok(line: str) -> bool:
    try:
        json.loads(line)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
