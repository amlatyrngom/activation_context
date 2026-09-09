"""Metadata-only policy resolution and immutable, sync-shared kernel profiles."""
from __future__ import annotations

from contextlib import nullcontext
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from ..common.data_syncing import resolve_path

SCHEMA = 1
TARGET_SECONDS = 60
MAX_SECONDS = 300
CONFIG_ROOT = Path(__file__).parent / "configs"
MODES = ["dense_backward", "batched_backward", "packed_backward", "prefill_state"]


class AutotuningUnavailable(RuntimeError):
    """No complete, validated configuration could be obtained within the budget."""


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def publish(profile: dict, folder: Path) -> Path:
    """Only complete results belong here; staging files cannot become cache hits."""
    payload = {"schema": SCHEMA, "profile": profile, "sha256": digest(profile)}
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{digest(profile['identity'])}-{payload['sha256']}.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=folder, suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return destination


def read_profile(path: Path, identity: dict | None = None) -> dict | None:
    try:
        data = json.loads(path.read_text())
        profile = data["profile"]
        if data["schema"] != SCHEMA or data["sha256"] != digest(profile):
            return None
        if identity is not None and profile["identity"] != identity:
            return None
        if profile["coverage"] != MODES or not profile["validation"]["passed"] or not profile["records"]:
            return None
        return profile
    except (OSError, ValueError, KeyError, TypeError):
        return None


def lookup(identity: dict) -> tuple[dict, str] | None:
    for label, folder in (("shared", resolve_path("AUTOTUNE/profiles")), ("preset", CONFIG_ROOT)):
        for path in sorted(folder.glob(f"{digest(identity)}-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            profile = read_profile(path, identity)
            if profile is not None:
                from .fla import validate_catalog
                if validate_catalog(profile):
                    return profile, label
    return None


def environment(device) -> dict:
    import torch
    from .fla import inventory, config_values
    props = torch.cuda.get_device_properties(device)
    # Pinned FLA has process-global, import-time architecture decisions. A
    # heterogeneous visible set cannot safely share those decisions.
    if any(torch.cuda.get_device_capability(i) != (props.major, props.minor)
           for i in range(torch.cuda.device_count())):
        raise AutotuningUnavailable("Pinned FLA requires homogeneous visible GPU architectures; select the training GPU with CUDA_VISIBLE_DEVICES")
    versions = {name: metadata.version(name) for name in ("torch", "triton", "fla-core", "transformers")}
    if versions["fla-core"] != "0.5.2" or versions["triton"] != "3.7.1":
        raise AutotuningUnavailable(f"FLA adapter needs review for installed versions: {versions}")
    distribution = metadata.distribution("fla-core")
    sources = {str(p): hashlib.sha256(distribution.locate_file(p).read_bytes()).hexdigest()
               for p in distribution.files or [] if str(p).startswith("fla/") and str(p).endswith(".py")}
    recipe = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    candidates = {name: [config_values(c) for c in tuner.configs] for name, tuner in inventory().items()}
    return {"gpu": props.name, "capability": [props.major, props.minor], "multiprocessors": props.multi_processor_count,
            "memory_bytes": props.total_memory, "cuda": torch.version.cuda, "versions": versions,
            "sources": digest(sources), "recipe": digest(recipe), "candidates": digest(candidates)}


def run_worker(identities: list[dict], device, deadline: float) -> None:
    folder = resolve_path("AUTOTUNE/staging")
    with tempfile.TemporaryDirectory(dir=folder) as staging:
        request = Path(staging) / "request.json"
        request.write_text(json.dumps({"identities": identities, "device": device.index,
                                       "seconds": max(0, deadline - time.monotonic())}))
        log = Path(staging) / "worker.log"
        with log.open("w") as output:
            process = subprocess.Popen([sys.executable, "-m", "activation.autotuners.worker", str(request)],
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            def stop():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                stop()
            except BaseException:
                stop()
                raise
        # Keep diagnostics, but never expose temporary output as a usable config.
        diagnostic = resolve_path("AUTOTUNE/logs") / f"{time.time_ns()}.log"
        diagnostic.write_text(log.read_text())
        if process.returncode:
            print(f"Autotuners: worker stopped ({process.returncode}); completed profiles retained; details {diagnostic}", flush=True)


def profiles_for(geometries: list[dict], device, deadline: float, *, retune=False) -> tuple[dict | None, list[str]]:
    if not geometries:
        return None, ["not_applicable"]
    env = environment(device)
    identities = [{"environment": env, "geometry": geometry} for geometry in geometries]
    found = [lookup(identity) for identity in identities]
    if retune or not all(found):
        lock_path = resolve_path("AUTOTUNE/locks") / f"{digest(env)}.lock"
        with lock_path.open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise AutotuningUnavailable("Timed out waiting for the local autotuning worker")
                    time.sleep(0.1)
            found = [lookup(identity) for identity in identities]
            missing = [identity for identity, value in zip(identities, found) if retune or value is None]
            if missing and time.monotonic() < deadline:
                print(f"Autotuners: {len(missing)} cold geometry profiles; target {TARGET_SECONDS}s, up to {MAX_SECONDS}s for compilation and correctness coverage", flush=True)
                run_worker(missing, device, deadline)
                found = [((result[0], "retuned" if retune else "cold") if (result := lookup(identity)) else value)
                         if retune or value is None else value
                         for identity, value in zip(identities, found)]
    if not all(found):
        raise AutotuningUnavailable("No validated FLA profile within the cold budget; inspect AUTOTUNE/logs and retry to reuse completed work")
    return {"profiles": [value[0] for value in found], "device_index": device.index}, [value[1] for value in found]


def resolve(model_configs, training_config, device, memory_budget_bytes, *, retune=False):
    import torch
    from transformers import AutoConfig
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    threshold = training_config.gradient_checkpointing_min_tokens
    chunk = training_config.logits_chunk_tokens
    if threshold is not None and threshold < 0:
        raise ValueError("gradient_checkpointing_min_tokens must be nonnegative")
    if chunk is not None and chunk <= 0:
        raise ValueError("logits_chunk_tokens must be positive")
    if memory_budget_bytes is not None and memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be positive")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    geometries, vocabularies = [], []
    for model in model_configs:
        # Harness registration already resolves model metadata. Do not hide a
        # network retry loop inside a cached or time-bounded tuning request.
        try:
            config = AutoConfig.from_pretrained(model.model_id, trust_remote_code=model.trust_remote_code, local_files_only=True)
        except OSError as error:
            raise AutotuningUnavailable(f"Model metadata must be cached before autotuning: {model.model_id}") from error
        config = config.get_text_config()
        vocabularies.append(config.vocab_size)
        if device.type != "cuda":
            continue
        linear = "linear_attention" in getattr(config, "layer_types", [])
        if not linear:
            continue
        if config.model_type not in ("qwen3_5_text", "qwen3_5") or model.dtype != torch.bfloat16:
            raise AutotuningUnavailable(f"No validated FLA recipe for {config.model_type}/{model.dtype}")
        geometry = {"heads": config.linear_num_value_heads, "key_dim": config.linear_key_head_dim,
                    "value_dim": config.linear_value_head_dim, "dtype": "bfloat16"}
        if geometry["key_dim"] not in (32, 64, 128, 256) or geometry["value_dim"] not in (32, 64, 128, 256):
            raise AutotuningUnavailable(f"Unsupported gated-delta geometry: {geometry}")
        if geometry not in geometries:
            geometries.append(geometry)
    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        bundle, origins = profiles_for(geometries, device, deadline, retune=retune)
    available = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else (memory_budget_bytes or 2**30)
    if memory_budget_bytes is not None:
        available = min(available, memory_budget_bytes)
    # Allow for FP32 logits, probabilities and backward temporaries. This bounds
    # head scratch only; it is not a claim about whole-model peak memory.
    allowance = min(2**31, available // 16)
    automatic_chunk = max(1, min(2048, allowance // (max(vocabularies) * 16)))
    automatic_chunk = 2 ** (automatic_chunk.bit_length() - 1)
    provenance = {"fla": origins, "seconds": round(time.monotonic() - started, 3),
                  "checkpoint": "explicit" if threshold is not None else "heuristic: always recompute; no matching whole-model memory profile",
                  "logits": "explicit" if chunk is not None else "heuristic: bounded vocabulary scratch",
                  "memory_budget_bytes": available}
    return {"fla_config": bundle, "logits_chunk_tokens": chunk if chunk is not None else automatic_chunk,
            "provenance": provenance}, threshold if threshold is not None else 0
