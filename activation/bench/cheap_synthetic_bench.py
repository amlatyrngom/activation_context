"""
Throughput of small generators for a CHEAP_SYNTHETIC phase: the study prompt on real MS-MARCO
chunks, one JSON {question, answer} per chunk, measured per model and engine concurrency. Writes
a live report (watch it with `sky watch`) and `cheap_synthetic_results.json` last.

    uv run sky exec <node> uv run python -m activation.bench.cheap_synthetic_bench \\
        --model-ids inclusionAI/Ling-3.0-tiny-fp8 RedHatAI/Qwen3.5-4B-FP8-dynamic \\
        --max-num-seqs 128 256 512 --num-chunks 2048
    uv run sky watch <node> '~/activation_artifacts/cheap_synthetic/' IB/TMP/CHEAP_SYNTHETIC/run/ \\
        --until-file cheap_synthetic_results.json
"""
import argparse
import gc
import json
import os
import random
import time
import traceback

import torch
from vllm.sampling_params import StructuredOutputsParams

from activation.common.reporting import HtmlReporter
from activation.dataset.dataset_study import STUDY_JSON_SCHEMA, make_study_prompt
from activation.dataset.dataset_utils import safe_truncate_embedding_chunk, shuffle_fill_truncate
from activation.dataset.loaders import MsMarcoDataset
from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig

COLUMNS = ["model", "max_num_seqs", "structured", "thinking", "requests", "prompt tok/s", "output tok/s",
           "requests/s", "QA per GPU-hour", "avg output tokens", "parse failures", "engine load", "run time", "note"]


def load_chunks(harness: HarnessRuntime, num_chunks: int, seed: int) -> list[str]:
    dataset = MsMarcoDataset.load(harness, max_examples=max(256, num_chunks // 8), max_corpus_documents=0)
    index = harness.dataset_manager._get_or_create_index(dataset.dataset_id)
    chunk_ids = shuffle_fill_truncate(list(index.chunk_ids), num_chunks, random.Random(seed))
    return [index.get_chunk_section(index.chunks[chunk_id]) for chunk_id in chunk_ids]


def conversations_for(snippets: list[str], chunk_input_limit: int) -> list[list[dict]]:
    system_prompt, instructions, _ = make_study_prompt(None)
    out = []
    for snippet in snippets:
        snippet = safe_truncate_embedding_chunk(snippet, chunk_input_limit)
        out.append([
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": f"# Corpus Snippet\n```\n{snippet}\n```\n{instructions}"}]},
        ])
    return out


def chat_kwargs_for(structured: bool, thinking: bool, seed: int, max_tokens: int) -> dict:
    sampling = {"max_tokens": max_tokens, "seed": seed, "temperature": 0.7, "top_p": 0.9}
    if structured:
        sampling["structured_outputs"] = StructuredOutputsParams(json=STUDY_JSON_SCHEMA)
    return {"sampling_params": sampling, "chat_template_kwargs": {"enable_thinking": thinking}}


def run_config(model_id: str, max_num_seqs: int, structured: bool, thinking: bool, snippets: list[str], args, seed: int) -> dict:
    engine_kwargs = {"max_num_seqs": max_num_seqs, "kv_cache_dtype": args.kv_cache_dtype}
    trust_remote_code = model_id.startswith("inclusionAI/")                        # Ling ships its own modeling code
    if trust_remote_code:
        engine_kwargs["enable_lora"] = False                                        # vLLM 0.28: BailingMoeV3 has no LoRA support
    config = HarnessRuntimeConfig(model_configs={
        model_id: ModelConfig(model_id, model_id, engine_kwargs=engine_kwargs, trust_remote_code=trust_remote_code),
    })
    harness = HarnessRuntime(config)
    loaded = harness.loaded_models[model_id]
    load_start = time.time()
    loaded.ensure_engine_loaded()
    load_time = time.time() - load_start
    conversations = conversations_for(snippets, args.chunk_input_limit)
    chat_kwargs = chat_kwargs_for(structured, thinking, seed, args.max_tokens)
    # Warm-up: one small batch, untimed.
    loaded.engine_chat_many(conversations[:min(64, len(conversations))], chat_kwargs=chat_kwargs)
    batch_size = 4 * max_num_seqs
    prompt_tokens = output_tokens = failures = 0
    start = time.time()
    for batch_start in range(0, len(conversations), batch_size):
        outputs = loaded.engine_chat_many(conversations[batch_start:batch_start + batch_size], chat_kwargs=chat_kwargs)
        for output in outputs:
            prompt_tokens += output.prompt_token_count
            output_tokens += output.output_token_count
            try:
                qa = json.loads(output.text)
                if not str(qa["question"]).strip() or not str(qa["answer"]).strip():
                    failures += 1
            except (json.JSONDecodeError, KeyError, TypeError):
                failures += 1
    elapsed = time.time() - start
    sample = outputs[0].text[:400] if outputs else ""
    loaded.engine_to_device(FREE_DEVICE)
    del harness
    gc.collect()
    torch.cuda.empty_cache()
    n = len(conversations)
    return {
        "model": model_id, "max_num_seqs": max_num_seqs, "structured": structured, "thinking": thinking, "requests": n,
        "prompt_tokens_per_s": prompt_tokens / elapsed, "output_tokens_per_s": output_tokens / elapsed,
        "requests_per_s": n / elapsed, "qa_per_gpu_hour": 3600 * (n - failures) / elapsed,
        "avg_output_tokens": output_tokens / n, "parse_failures": failures, "parse_failure_fraction": failures / n,
        "engine_load_s": load_time, "run_s": elapsed, "sample_output": sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ids", nargs="+", default=["inclusionAI/Ling-3.0-tiny-fp8", "RedHatAI/Qwen3.5-4B-FP8-dynamic"])
    parser.add_argument("--max-num-seqs", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--structured", choices=["on", "off", "both"], default="on")
    parser.add_argument("--thinking", choices=["on", "off", "both"], default="off")
    parser.add_argument("--num-chunks", type=int, default=2048)
    parser.add_argument("--chunk-input-limit", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-folder", default="~/activation_artifacts/cheap_synthetic")
    args = parser.parse_args()
    args.report_folder = os.path.expanduser(args.report_folder)

    reporter = HtmlReporter(args.report_folder, "Cheap synthetic generators",
                            f"Study-prompt QA generation on {args.num_chunks} MS-MARCO chunks per configuration, one RTX PRO 6000.",
                            eyebrow="CHEAP_SYNTHETIC bench", refresh_seconds=15, min_render_interval_seconds=0)
    reporter.initialize_bar_plot("throughput", "Output tokens per second", "Decode throughput per configuration.", "configuration", "tokens/s", ["output tok/s"])
    reporter.initialize_bar_plot("qa_rate", "QA pairs per GPU-hour", "Parsed pairs only.", "configuration", "pairs/hour", ["QA/hour"])
    reporter.initialize_table("results", "Results", "One row per (model, concurrency, structured output, thinking).", COLUMNS)
    reporter.set_text("plan", "Plan", json.dumps(vars(args), indent=2))
    reporter.set_status(configurations="0", state="loading chunks")
    reporter.render(force=True)

    seed_harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={}))
    snippets = load_chunks(seed_harness, args.num_chunks, args.seed)
    del seed_harness
    structured_options = {"on": [True], "off": [False], "both": [True, False]}[args.structured]
    thinking_options = {"on": [True], "off": [False], "both": [False, True]}[args.thinking]
    results = []
    total = len(args.model_ids) * len(args.max_num_seqs) * len(structured_options) * len(thinking_options)
    done = 0
    for model_id in args.model_ids:
        for max_num_seqs in args.max_num_seqs:
            for structured in structured_options:
                for thinking in thinking_options:
                    label = f"{model_id.split('/')[-1]} · {max_num_seqs} · {'json' if structured else 'free'}{' · think' if thinking else ''}"
                    reporter.set_status(configurations=f"{done} / {total}", state=f"running {label}")
                    reporter.add_data_point("results", {"model": model_id, "max_num_seqs": max_num_seqs, "structured": structured,
                                                        "thinking": thinking, "note": "running", "running": True})
                    reporter.render(force=True)
                    row_start = time.time()
                    try:
                        result = run_config(model_id, max_num_seqs, structured, thinking, snippets, args, args.seed)
                        note = ""
                    except Exception as error:  # a load or kernel failure is a result too
                        traceback.print_exc()
                        result = {"model": model_id, "max_num_seqs": max_num_seqs, "structured": structured, "thinking": thinking,
                                  "error": f"{type(error).__name__}: {str(error)[:300]}", "run_s": time.time() - row_start}
                        note = result["error"]
                        gc.collect(); torch.cuda.empty_cache()
                    results.append(result)
                    done += 1
                    print(json.dumps(result, indent=1), flush=True)
                    if "error" not in result:
                        reporter.add_data_point("throughput", {"label": label, "output tok/s": result["output_tokens_per_s"]})
                        reporter.add_data_point("qa_rate", {"label": label, "QA/hour": result["qa_per_gpu_hour"]})
                    reporter.add_data_point("results", {
                        "model": model_id, "max_num_seqs": max_num_seqs, "structured": structured, "thinking": thinking,
                        "requests": result.get("requests", ""),
                        "prompt tok/s": f"{result['prompt_tokens_per_s']:.0f}" if "prompt_tokens_per_s" in result else "",
                        "output tok/s": f"{result['output_tokens_per_s']:.0f}" if "output_tokens_per_s" in result else "",
                        "requests/s": f"{result['requests_per_s']:.1f}" if "requests_per_s" in result else "",
                        "QA per GPU-hour": f"{result['qa_per_gpu_hour']:,.0f}" if "qa_per_gpu_hour" in result else "",
                        "avg output tokens": f"{result['avg_output_tokens']:.0f}" if "avg_output_tokens" in result else "",
                        "parse failures": f"{result['parse_failure_fraction']:.1%}" if "parse_failure_fraction" in result else "",
                        "engine load": f"{result['engine_load_s']:.0f} s" if "engine_load_s" in result else "",
                        "run time": f"{result['run_s']:.0f} s", "note": note,
                    })
                    reporter.set_status(configurations=f"{done} / {total}", state="between configurations")
                    reporter.render(force=True)
    reporter.set_text("samples", "Sample outputs", "\n\n".join(f"## {r['model']} · {r['max_num_seqs']}\n{r.get('sample_output', r.get('error', ''))}" for r in results))
    reporter.set_status(state="finished")
    reporter.finish()
    with open(os.path.join(args.report_folder, "cheap_synthetic_results.json"), "w") as handle:
        json.dump({"args": vars(args), "results": results}, handle, indent=1)


if __name__ == "__main__":
    main()
