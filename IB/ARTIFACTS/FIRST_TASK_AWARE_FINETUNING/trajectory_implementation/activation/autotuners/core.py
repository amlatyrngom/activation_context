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


# Whole-model measurements are deliberately separate from the FLA kernel catalog.
EXECUTION_SCHEMA = 2


def execution_batches(records: list[dict], config) -> list[list[int]]:
    """Two real update shapes and a distinct-history padding tail, never a data filter."""
    import random
    count = config.examples_per_update
    if not count or count < 1:
        raise ValueError("Execution tuning requires explicit examples_per_update")
    order = list(range(len(records)))
    if config.shuffle_examples:
        random.Random(config.seed + 1).shuffle(order)
    groups = [order[i:i + count] for i in range(0, len(order), count)]
    groups.sort(key=lambda group: sum(records[i]["encoder_tokens"] for i in group))
    selected = [groups[(len(groups) - 1) // 2], groups[int((len(groups) - 1) * .9)]]
    tail, seen = [], set()
    for index in sorted(order, key=lambda i: (records[i]["encoder_tokens"], records[i]["reader_tokens"]), reverse=True):
        key = tuple(records[index]["request_keys"])
        if key not in seen:
            tail.append(index); seen.add(key)
        if len(tail) == count:
            break
    if len(tail) < min(count, len(order)):
        tail.extend(i for i in reversed(order) if i not in tail)
        tail = tail[:count]
    selected.append(tail)
    return [list(group) for group in dict.fromkeys(tuple(group) for group in selected)]


def execution_candidates(config) -> list[dict]:
    """A small ordered ladder, not a Cartesian search. Every arm needs its own gate."""
    baseline = dict(graph_examples=1, reader_batch=1, encode_batch_tokens=1,
                    reuse_encodings=False, side_checkpoint_tokens=0,
                    target_checkpoint_tokens=0, optimizer_moments="cpu")
    # Reader batching and graph-local reuse leave encoder batch shape unchanged.
    variants = [baseline, {**baseline, "optimizer_moments": "gpu"}]
    variants.append({**baseline, "target_checkpoint_tokens": 4096})
    variants.append({**variants[-1], "side_checkpoint_tokens": 8192})
    for threshold in (16384, 32768):
        variants.append({**variants[2], "side_checkpoint_tokens": threshold})
    variants.append({**baseline, "graph_examples": 4, "reuse_encodings": True})
    for batch in (2, 4):
        variants.append({**baseline, "graph_examples": batch, "reader_batch": batch,
                         "reuse_encodings": True})
    variants.append({**variants[-1], "target_checkpoint_tokens": 4096})
    variants.append({**variants[-1], "side_checkpoint_tokens": 8192})
    variants.append({**variants[-1], "optimizer_moments": "gpu"})
    # Encoder batching is deliberately last: prior shape changes failed parity.
    variants.append({**variants[-1], "encode_batch_tokens": 16384})
    count = config.examples_per_update
    return [policy for policy in variants if policy["graph_examples"] <= count]


def execution_capacity_cases(records: list[dict], config, policy: dict) -> list[dict]:
    """Actual examples just below checkpoint boundaries; separate from speed sampling."""
    update_sizes = [min(config.examples_per_update, len(records))]
    if len(records) > config.examples_per_update and len(records) % config.examples_per_update:
        update_sizes.append(len(records) % config.examples_per_update)
    widths = list(dict.fromkeys(min(policy["graph_examples"], count - start)
                  for count in update_sizes for start in range(0, count, policy["graph_examples"])))
    cases = {}

    def largest(scores, count):
        # Avoid a repeated-history best case when graph-local reuse is enabled.
        ranked = sorted(scores, key=lambda i: (scores[i], records[i]["encoder_tokens"],
                                              records[i]["reader_tokens"], records[i]["penalty_tokens"],
                                              str(records[i]["item"])), reverse=True)
        result, seen = [], set()
        for index in ranked:
            key = tuple(records[index]["request_keys"])
            if key not in seen:
                result.append(index); seen.add(key)
            if len(result) == count:
                return result
        return (result + [i for i in ranked if i not in result])[:count]

    def add(indices, reason, width, penalty_indices=None, offset=0):
        if not indices:
            return
        if len(indices) < width:
            others = {i: records[i]["encoder_tokens"] + records[i]["reader_tokens"]
                      for i in range(len(records)) if i not in indices}
            remaining = largest(others, width - len(indices))
            indices = remaining[:offset] + indices + remaining[offset:]
        flags = [bool(config.penalty_active and records[i]["penalty_tokens"] and
                      (penalty_indices is None or i in penalty_indices)) for i in indices]
        key = (tuple(indices), tuple(flags))
        if key not in cases:
            cases[key] = {"indices": indices, "penalty_flags": flags, "reasons": []}
        cases[key]["reasons"].append(reason)

    for width in widths:
        side = policy["side_checkpoint_tokens"]
        if side:
            scores = {i: max((length for length in record["encoder_sequence_tokens"] if length < side), default=0)
                      for i, record in enumerate(records)}
            add(largest({i: score for i, score in scores.items() if score}, width), f"side below {side}", width)
        target = policy["target_checkpoint_tokens"]
        if not target:
            continue
        # A final short update/graph/reader chunk can enter an uncheckpointed
        # branch even when every full batch of the same examples checkpoints.
        reader_widths = {min(policy["reader_batch"], width): 0}
        if width > policy["reader_batch"] and width % policy["reader_batch"]:
            reader_widths[width % policy["reader_batch"]] = width - width % policy["reader_batch"]
        for reader_width, offset in reader_widths.items():
            scores = {i: record["reader_tokens"] for i, record in enumerate(records)
                      if record["reader_tokens"] * reader_width < target}
            if len(scores) >= reader_width:
                # Put the short reader case in the actual tail, not a full chunk.
                indices = largest(scores, reader_width if offset else width)
                add(indices, f"reader below {target}", width, offset=offset)
            if config.penalty_active:
                # A subset receives penalties, so each possible padded penalty
                # width has its own boundary inside this reader chunk.
                for count in range(1, reader_width + 1):
                    scores = {i: record["penalty_tokens"] for i, record in enumerate(records)
                              if 0 < record["penalty_tokens"] * count < target}
                    if len(scores) >= count:
                        indices = largest(scores, count)
                        add(indices, f"penalty width {count} below {target}", width, set(indices), offset)
    return list(cases.values())


def read_execution_profile(path: Path, identity: dict) -> dict | None:
    try:
        payload = json.loads(path.read_text())
        profile = payload["profile"]
        if (payload["schema"] != SCHEMA or payload["sha256"] != digest(profile)
                or profile["identity"] != identity or profile["kind"] != "ac_execution"
                or profile["execution_schema"] != EXECUTION_SCHEMA
                or not profile["validation"]["passed"]):
            return None
        winner = next(row for row in profile["candidates"] if row["policy"] == profile["policy"])
        if (not winner["passed"] or not winner["cases"] or not all(row["passed"] for row in winner["cases"])
                or len(winner["cases"]) != len(identity["cases"])):
            return None
        required = next(row["cases"] for row in identity["capacity_policies"] if row["policy"] == winner["policy"])
        if ([row["capacity_case"] for row in winner["capacity_cases"]] != required
                or not all(row["passed"] for row in winner["capacity_cases"])):
            return None
        return profile
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None


def _wait_execution_worker(process, progress_path: Path, deadline: float, progress_callback=None):
    """Forward new atomic progress records without resetting the worker deadline."""
    previous = None
    def update():
        nonlocal previous
        if progress_callback is None:
            return
        try:
            current = progress_path.read_text()
            progress = json.loads(current)
        except (OSError, ValueError):
            return
        if current != previous:
            progress_callback(progress)
            previous = current
    while True:
        update()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("AC execution worker", 0)
        try:
            process.wait(timeout=min(5., remaining))
        except subprocess.TimeoutExpired:
            continue
        update()
        return


def tune_ac_execution(harness, ac_name, prepared_examples, training_config, *, seconds=600, progress_callback=None):
    """Trusted local objects only; the child owns all GPU models and trial state."""
    from dataclasses import asdict, replace
    import pickle
    import torch
    from ..ac_model.ac_model_utils import direct_parts, tokenize_with_parts
    from ..common.data_syncing import sync_root
    if training_config.loss_kind != "sft":
        raise ValueError("Whole-execution tuning currently supports SFT only")
    if seconds <= 0 or not prepared_examples:
        raise ValueError("Execution tuning needs examples and a positive budget")
    manager = harness.module_manager
    ac = manager.get_ac_model(ac_name)
    if manager.peft_models or any(model.current_model_device.startswith("cuda") for model in harness.loaded_models.values()):
        raise ValueError("Tune execution before placing/injecting parent models; the child needs exclusive GPU capacity")
    examples = list(prepared_examples)
    counts, sequences = {}, {}
    def request_record(request):
        key = digest(request._asdict())
        if key not in counts:
            children = [ac._child_request(part, request.compression_ratio) for part in direct_parts(request.messages)]
            child_counts = [request_record(child) for child in children]
            child_rows = [ac.num_view_rows(counts[k][0], child.compression_ratio) for k, child in zip(child_counts, children)]
            ids, _ = tokenize_with_parts(ac.side.tokenizer, request.messages, child_rows, ac.pad_id, tools=request.tools)
            length = len(ids) + ac.num_view_rows(len(ids), request.compression_ratio)
            counts[key] = (len(ids), length + sum(counts[k][1] for k in child_counts))
            sequences[key] = [length, *(value for child in child_counts for value in sequences[child])]
        return key
    records = []
    for example in examples:
        keys = [request_record(request) for request in example.part_requests]
        records.append({"item": example.item.item_id, "request_keys": keys,
                        "encoder_tokens": sum(counts[key][1] for key in keys),
                        "encoder_sequence_tokens": [value for key in keys for value in sequences[key]],
                        "reader_tokens": len(example.student_ids), "target_tokens": len(example.target_ids),
                        "penalty_tokens": len(example.penalty.student_ids) if example.penalty else 0,
                        "weight": example.item.weight,
                        "reader_identity": digest([list(example.student_ids), list(example.student_positions), list(example.target_ids)]),
                        "reference_identity": list(example.penalty.teacher_cache_key) if example.penalty else None})
    batches = execution_batches(records, training_config)
    capacity_cases, capacity_policies = [], []
    for policy in execution_candidates(training_config):
        selected_cases = []
        for case in execution_capacity_cases(records, training_config, policy):
            found = next((i for i, prior in enumerate(capacity_cases)
                          if prior["indices"] == case["indices"] and prior["penalty_flags"] == case["penalty_flags"]), None)
            if found is None:
                found = len(capacity_cases); capacity_cases.append(case)
            else:
                capacity_cases[found]["reasons"] = list(dict.fromkeys(capacity_cases[found]["reasons"] + case["reasons"]))
            selected_cases.append(found)
        capacity_policies.append({"policy": policy, "cases": selected_cases})
    indexes = list(dict.fromkeys(i for group in [*batches, *(case["indices"] for case in capacity_cases)] for i in group))
    local = {index: position for position, index in enumerate(indexes)}
    selected = [examples[i] for i in indexes]
    model_names = {ac.config.base_side_model_name, ac.config.target_model_name}
    harness_config = replace(harness.harness_config, model_configs={name: harness.harness_config.model_configs[name] for name in model_names})
    def absolute_checkpoint(value):
        if value is None:
            return None
        path = Path(value)
        return str(path.resolve() if path.is_absolute() else resolve_path(str(path), create=False).resolve())
    ac_config = replace(ac.config, checkpoint_path=absolute_checkpoint(ac.config.checkpoint_path))
    loras = [replace(manager.get_lora_config(name), checkpoint_path=absolute_checkpoint(manager.get_lora_config(name).checkpoint_path))
             for name in dict.fromkeys((ac.config.base_side_model_lora_name, ac.config.target_model_lora_name))]
    device = torch.device("cuda", torch.cuda.current_device())
    env = environment(device)
    # Training knobs that affect graph shape/precision/objective, not elapsed-run state.
    recipe = asdict(training_config)
    for key in ("execution_policy", "max_duration_seconds", "resume_optimizer", "resume_skip_steps", "num_epochs",
                "checkpoint_every_updates", "checkpoint_every_epoch", "reporting_interval", "completion_samples"):
        recipe.pop(key, None)
    def portable(value):
        return json.loads(json.dumps(value, default=str, sort_keys=True))
    checkpoints = {}
    for checkpoint in [ac_config.checkpoint_path, *(lora.checkpoint_path for lora in loras)]:
        if checkpoint:
            checkpoints[checkpoint] = [(str(path.relative_to(checkpoint)), path.stat().st_size, path.stat().st_mtime_ns)
                                       for path in sorted(Path(checkpoint).rglob("*")) if path.is_file()]
    package = Path(__file__).parent.parent
    sources = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in (
        "ac_model/ac_model_training.py", "ac_model/ac_model_utils.py", "ac_model/ac_model.py",
        "harness/hf_utils.py", "harness/loaded_model.py", "harness/module_manager.py", "agent_training/agent_training_utils.py")}
    identity = {"kind": "ac_execution", "environment": env, "execution_schema": EXECUTION_SCHEMA, "execution_sources": sources,
                "models": portable(asdict(harness_config)), "ac": portable(asdict(ac_config)), "checkpoints": portable(checkpoints),
                "loras": portable([asdict(lora) for lora in loras]), "recipe": portable(recipe),
                "workload": digest(records), "cases": [[records[i] for i in group] for group in batches],
                "capacity_cases": [{**case, "records": [records[i] for i in case["indices"]]} for case in capacity_cases],
                "capacity_policies": capacity_policies}
    folder = resolve_path("AUTOTUNE/execution_profiles")
    for path in sorted(folder.glob(f"{digest(identity)}-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        profile = read_execution_profile(path, identity)
        winner = next((row for row in profile["candidates"] if row["policy"] == profile["policy"]), None) if profile else None
        free_bytes = torch.cuda.mem_get_info(device)[0]
        if profile and max(row["peak_memory_bytes"] for row in winner["cases"] + winner["capacity_cases"]) < free_bytes * .96:
            if progress_callback is not None:
                progress_callback({"phase": "reusing measured policy", "completed_cases": len(winner["cases"]),
                                   "total_cases": len(winner["cases"]), "elapsed_seconds": 0.})
            return {"policy": profile["policy"], "receipt": {**profile, "origin": "cached", "path": str(path)}}
    staging = resolve_path("AUTOTUNE/staging")
    with tempfile.TemporaryDirectory(prefix="execution-", dir=staging) as temporary:
        root = Path(temporary)
        request = {"kind": "ac_execution", "identity": identity, "harness_config": harness_config,
                   "loras": loras, "ac_config": ac_config,
                   "ac_state": {key: value.detach().cpu() for key, value in ac.modules.state_dict().items()},
                   "examples": selected, "batches": [[local[i] for i in group] for group in batches],
                   "capacity_cases": [{**case, "indices": [local[i] for i in case["indices"]]} for case in capacity_cases],
                   "capacity_policies": capacity_policies,
                   "training_config": training_config, "seconds": float(seconds), "device": device.index,
                   "autotune_root": str(sync_root().resolve()),
                   "progress": str(root / "progress.json"),
                   "sync_root": str(root / "isolated_run"), "output": str(root / "result.json")}
        request_path = root / "request.pkl"
        with request_path.open("wb") as handle:
            pickle.dump(request, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log_path = resolve_path("AUTOTUNE/logs") / f"execution-{time.time_ns()}.log"
        if progress_callback is not None:
            progress_callback({"phase": "starting execution worker", "completed_cases": 0,
                               "total_cases": len(batches) * (len(capacity_policies) + 2)
                                              + 2 * len(capacity_cases) + sum(len(row["cases"]) for row in capacity_policies),
                               "elapsed_seconds": 0.})
        with log_path.open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "activation.autotuners.worker", "--ac-execution", str(request_path)],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                _wait_execution_worker(process, Path(request["progress"]), time.monotonic() + float(seconds) + 10,
                                       progress_callback)
            except BaseException as error:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                if not isinstance(error, subprocess.TimeoutExpired):
                    raise
        profile = read_execution_profile(Path(request["output"]), identity)
        if profile is None:
            raise AutotuningUnavailable(f"No complete execution candidate passed within the budget; details {log_path}")
        profile["worker_log"] = str(log_path)
        profile["worker_exit_code"] = process.returncode
        path = publish(profile, folder)
        return {"policy": profile["policy"], "receipt": {**profile, "origin": "measured", "path": str(path)}}
