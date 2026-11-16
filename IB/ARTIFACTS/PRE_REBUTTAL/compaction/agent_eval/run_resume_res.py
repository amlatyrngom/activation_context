"""
The compaction agent evaluation (kit `run_resume.py`, resumption benchmark) for Hidden Residuals pairs (`res` write mode:
rows + per-layer residual deltas, 2B encoder -> frozen Qwen/Qwen3.5-9B reader), served through the vLLM residual path
(`activation/vllm_residuals/`). Pre-rebuttal C4 (dev-75), 2026-11-16.

    python run_resume_res.py <run_resume.py arguments>          (kit/ on PYTHONPATH; every run_resume.py argument works as before)

What this changes in run_resume.py (nothing else; the text arms full / none / summary_write / summary run unchanged):
  * register_pair: the AC model is built from the fields saved in ac_modules.pt (run_qa.pair_setup's rule) instead of the kit's
    hard-coded H8a shape; the saved model ids must be [--side-model-id, --model-id]. Without --rows-only a deep checkpoint keeps
    its deltas, so `LoadedModel.engine_to_device` gives the engine the residual `hf_overrides`; both are logged and checked
    (layers in the checkpoint == layers of the loaded model == layers in the engine's hf_overrides; inputs_embeds_size = (1 + k)·d).
    --rows-only keeps the kit's behavior (rows alone, the deltas computed and dropped).
  * A reader adapter whose LoRA B matrices are all zero (a frozen-reader pair) is not handed to the engine: the AC arm is then
    served by the untuned 9B exactly like the text arms (no LoRA kernels). A non-zero adapter is exchanged as in the kit.
  * RESUME_KV_CACHE_DTYPE (optional) overrides the engine's KV cache dtype (default: the recommended fp8, the paper's setting).
  * A monitor line every RESUME_MONITOR_SECONDS (default 60) on stdout and, with RESUME_MONITOR_FILE, as JSON lines: host
    memory, process tree RSS, engine requests, request payload bytes (sent, and what a full-length tensor would have been),
    prompt / cached / output tokens.
"""
import json
import os
import threading
import time
from pathlib import Path

import run_resume
from run_resume import AC_NAME, ORACLE_MODEL_NAME, READER_ADAPTER_NAME, SIDE_ADAPTER_NAME, SIDE_BASE_NAME

# run_qa.pair_setup's fields: what the pair was trained with (the rest of ActivationContextModelConfig keeps its defaults)
PAIR_AC_FIELDS = ("mixer_layers", "mixer_window", "mixer_heads", "min_view_rows", "max_view_rows", "side_lora_rank", "default_compression_ratio",
                  "input_pooling_stride", "input_pooling_window", "recursive_keys", "intervention_frequency", "intervention_layers",
                  "add_last_layer_intervention", "kv_transfer")
SHAPE_FIELDS = {"summary_layout": "appended", "row_head_linear_skip": False, "bin_density_weight": 0.0}
STATE = {"pair": None, "engine_logged": False}


def reader_lora_b_max(reader_checkpoint: str) -> float:
    from safetensors import safe_open
    largest = 0.0
    with safe_open(str(Path(reader_checkpoint) / "adapter_model.safetensors"), "pt") as handle:
        for key in handle.keys():
            if "lora_B" in key:
                largest = max(largest, float(handle.get_tensor(key).abs().max()))
    return largest


def register_pair(harness, ac_checkpoint: str, reader_checkpoint: str, rows_only: bool, encode_batch_tokens: int) -> None:
    import torch
    from activation.ac_model import ActivationContextModelConfig
    saved = torch.load(Path(ac_checkpoint) / "ac_modules.pt", map_location="cpu", weights_only=False, mmap=True)
    config, saved_layers = saved["config"], [int(x) for x in (saved.get("interventions") or {}).get("layers") or []]
    loaded_ids = [harness.loaded_models[SIDE_BASE_NAME].model_config.model_id, harness.loaded_models[ORACLE_MODEL_NAME].model_config.model_id]
    assert list(saved.get("model_ids") or []) == loaded_ids, f"the pair is {saved.get('model_ids')}, the harness loads {loaded_ids}"
    fields = {key: config[key] for key in PAIR_AC_FIELDS if config.get(key) is not None}
    if "intervention_frequency" not in config:                     # older than the field: its recorded layers, given explicitly
        fields.update({"intervention_frequency": 0, **({"intervention_layers": saved_layers} if saved_layers else {})})
    shape = {key: (config.get(key) if config.get(key) is not None else default) for key, default in SHAPE_FIELDS.items()}
    adapter = json.loads((Path(reader_checkpoint) / "adapter_config.json").read_text())
    manager = harness.module_manager
    manager.register_lora(READER_ADAPTER_NAME, ORACLE_MODEL_NAME, rank=int(adapter["r"]), alpha=adapter.get("lora_alpha"), dropout=0,
                          checkpoint_path=str(reader_checkpoint))
    ac_model = manager.register_ac_model(ActivationContextModelConfig(
        AC_NAME, SIDE_BASE_NAME, SIDE_ADAPTER_NAME, ORACLE_MODEL_NAME, READER_ADAPTER_NAME, checkpoint_path=str(ac_checkpoint), **fields, **shape))
    ac_model.config.encode_batch_max_tokens = encode_batch_tokens
    lora_b = reader_lora_b_max(reader_checkpoint)
    if lora_b == 0.0:
        print(f"resume_res - reader adapter LoRA B max |B| = 0.0: not handed to the engine (the AC arm reads with the untuned {loaded_ids[1]})", flush=True)
    else:
        print(f"resume_res - WARNING reader adapter LoRA B max |B| = {lora_b}: a trained reader adapter with deltas is untested in vLLM", flush=True)
        manager.exchange_lora(ORACLE_MODEL_NAME, READER_ADAPTER_NAME, str(reader_checkpoint))
    if rows_only and ac_model.is_deep:                             # the kit's rows-only serving: deltas computed and dropped
        encode_with_layers = ac_model.encode_batch

        def encode_rows(requests, *, with_layers: bool = False):
            outputs = encode_with_layers(requests, with_layers=True)
            return outputs if with_layers else [output.input_embeds for output in outputs]

        ac_model.encode_batch = encode_rows
    else:
        assert list(ac_model.intervention_layers) == saved_layers, f"loaded layers {ac_model.intervention_layers} != checkpoint layers {saved_layers}"
    STATE["pair"] = {"deep": ac_model.is_deep, "rows_only": rows_only, "layers": list(ac_model.intervention_layers), "reader_lora_b_max": lora_b}
    print(f"resume_res - pair registered: AC {ac_checkpoint}; model ids {loaded_ids}; fields {json.dumps(fields, sort_keys=True)}; shape {shape}; "
          f"deep {ac_model.is_deep}, intervention layers {list(ac_model.intervention_layers)} (checkpoint {saved_layers}), rows only {rows_only}; "
          f"reader adapter {reader_checkpoint} (r {adapter['r']}, alpha {adapter.get('lora_alpha')})", flush=True)


def log_engine(loaded) -> None:
    """Once, after the reader engine is built: what it was given and what vLLM made of it."""
    if STATE["engine_logged"] or loaded.vllm_model is None or loaded.model_config.model_name != ORACLE_MODEL_NAME:
        return
    STATE["engine_logged"] = True
    kwargs = loaded.model_config.engine_kwargs or {}
    shown = {key: kwargs.get(key) for key in ("hf_overrides", "enable_prompt_embeds", "kv_cache_dtype", "max_model_len", "gpu_memory_utilization",
                                              "enable_lora", "max_loras", "max_lora_rank")}
    engine = loaded.vllm_model.replicas[0].engine
    width = engine.model_config.get_inputs_embeds_size()
    print(f"resume_res - reader engine kwargs {json.dumps(shown, default=str)}; vLLM architecture {engine.model_config.architectures}, "
          f"inputs_embeds_size {width}", flush=True)
    pair = STATE["pair"]
    if pair and pair["deep"] and not pair["rows_only"]:
        from activation.vllm_residuals import ARCHITECTURE, LAYERS_KEY
        overrides = kwargs.get("hf_overrides") or {}
        d_model = loaded.model_config.model_description.d_model
        assert overrides.get("architectures") == [ARCHITECTURE], f"the residual engine needs hf_overrides with {ARCHITECTURE}: {overrides}"
        assert overrides["text_config"][LAYERS_KEY] == pair["layers"], f"engine layers {overrides['text_config'][LAYERS_KEY]} != AC layers {pair['layers']}"
        assert width == (1 + len(pair["layers"])) * d_model, f"inputs_embeds_size {width} != (1 + {len(pair['layers'])}) x {d_model}"
        print(f"resume_res - residual engine checked: layers {pair['layers']}, row width {width} = (1 + {len(pair['layers'])}) x {d_model}", flush=True)


class Monitor:
    """Counters filled by the wrapped engine calls; a line per period."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.window = self._empty()
        self.total = self._empty()
        self.started = time.time()

    @staticmethod
    def _empty() -> dict:
        return {"requests": 0, "payload_bytes": 0, "payload_max": 0, "full_length_bytes": 0, "full_length_max": 0,
                "prompt_tokens": 0, "cached_tokens": 0, "output_tokens": 0}

    def add(self, **values) -> None:
        with self.lock:
            for counters in (self.window, self.total):
                for key, value in values.items():
                    if key.endswith("_max"):
                        counters[key] = max(counters[key], value)
                    else:
                        counters[key] += value

    def line(self, period: float) -> dict:
        import psutil
        with self.lock:
            window, self.window = self.window, self._empty()
            total = dict(self.total)
        me = psutil.Process()
        tree = [me] + me.children(recursive=True)
        rss = 0
        for process in tree:
            try:
                rss += process.memory_info().rss
            except psutil.Error:
                pass
        memory = psutil.virtual_memory()
        requests = max(window["requests"], 1)
        gib = 2 ** 30
        return {"t": round(time.time() - self.started), "host_used_gib": round((memory.total - memory.available) / gib, 2),
                "host_available_gib": round(memory.available / gib, 2), "tree_rss_gib": round(rss / gib, 2), "load1": round(os.getloadavg()[0], 2),
                "requests": window["requests"], "requests_per_min": round(window["requests"] * 60 / period, 1),
                "payload_mean_mb": round(window["payload_bytes"] / requests / 1e6, 1), "payload_max_mb": round(window["payload_max"] / 1e6, 1),
                "full_length_mean_mb": round(window["full_length_bytes"] / requests / 1e6, 1), "full_length_max_mb": round(window["full_length_max"] / 1e6, 1),
                "output_tok_per_s": round(window["output_tokens"] / period, 1), "prompt_tokens": window["prompt_tokens"],
                "cached_tokens": window["cached_tokens"], "total": total}


def install_monitor() -> None:
    from activation.agent.agent import Agent
    from activation.harness import FREE_DEVICE
    from activation.harness.loaded_model import LoadedModel
    monitor = Monitor()
    prepare = Agent.prepare_request

    def prepare_request(self):
        prefix, embeds, mask = prepare(self)
        if embeds is not None:
            row = embeds.shape[1] * embeds.element_size()
            monitor.add(payload_bytes=embeds.numel() * embeds.element_size(), payload_max=embeds.numel() * embeds.element_size(),
                        full_length_bytes=len(prefix) * row, full_length_max=len(prefix) * row)
        return prefix, embeds, mask

    Agent.prepare_request = prepare_request
    submit = LoadedModel.engine_submit_tokens

    def engine_submit_tokens(self, *args, **kwargs):
        output = submit(self, *args, **kwargs)
        monitor.add(requests=1, prompt_tokens=output.prompt_token_count, cached_tokens=output.cached_prompt_token_count,
                    output_tokens=output.output_token_count)
        return output

    LoadedModel.engine_submit_tokens = engine_submit_tokens
    to_device = LoadedModel.engine_to_device

    def engine_to_device(self, device):
        result = to_device(self, device)
        try:
            log_engine(self)
        except AssertionError as error:                             # first called from an agent thread: stop the whole run, not one rollout
            print(f"resume_res - FATAL engine check failed: {error}", flush=True)
            to_device(self, FREE_DEVICE)
            os._exit(3)
        return result

    LoadedModel.engine_to_device = engine_to_device
    period = float(os.environ.get("RESUME_MONITOR_SECONDS", "60"))
    path = os.environ.get("RESUME_MONITOR_FILE")

    def loop() -> None:
        while True:
            time.sleep(period)
            try:
                line = monitor.line(period)
            except Exception as error:                              # never take the run down
                line = {"monitor_error": repr(error)}
            print(f"resume_res monitor {json.dumps(line)}", flush=True)
            if path:
                with open(path, "a") as handle:
                    handle.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **line}) + "\n")

    threading.Thread(target=loop, name="resume_res_monitor", daemon=True).start()


def main() -> int:
    kv = os.environ.get("RESUME_KV_CACHE_DTYPE")
    if kv:
        from activation.harness.vllm_wrapper import VLLMWrapper
        recommended = VLLMWrapper.recommended_engine_kwargs
        VLLMWrapper.recommended_engine_kwargs = staticmethod(lambda model_id, is_for_training=False: recommended(model_id, is_for_training) | {"kv_cache_dtype": kv})
    print(f"resume_res - kv_cache_dtype {kv or 'recommended'}", flush=True)
    run_resume.register_pair = register_pair
    install_monitor()
    return run_resume.main()


if __name__ == "__main__":                                       # vLLM spawns its engine core: keep the guard
    raise SystemExit(main())
