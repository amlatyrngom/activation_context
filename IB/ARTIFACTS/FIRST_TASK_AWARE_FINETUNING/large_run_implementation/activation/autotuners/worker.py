"""Disposable CUDA workers for kernel and actual training-workload profiles."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from .core import MODES, TARGET_SECONDS, environment, publish
from ..common.data_syncing import resolve_path
from .fla import Catalog, catalog_scope, inventory


def apply_runtime_settings(state):
    """Disposable process only: match the recorded mode before any CUDA/model work."""
    import os
    import torch
    from .core import runtime_settings
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != state["cublas_workspace_config"]:
        raise RuntimeError("Worker CUDA workspace differs from its startup environment receipt")
    torch.use_deterministic_algorithms(state["deterministic_algorithms"], warn_only=state["deterministic_warn_only"])
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
    if runtime_settings() != state:
        raise RuntimeError("Worker deterministic kernel settings differ from the profile identity")


def exercise(geometry, catalog, *, length=129, batch=1, packed=False, final_state=False, reference=True):
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
    h, k, v = (geometry[key] for key in ("heads", "key_dim", "value_dim"))
    q = torch.randn(batch, length, h, k, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(q)
    value = torch.randn(batch, length, h, v, device="cuda", dtype=torch.bfloat16)
    gate = -torch.rand(batch, length, h, device="cuda", dtype=torch.float32) * .2
    beta = torch.rand(batch, length, h, device="cuda", dtype=torch.bfloat16)
    # Exercise padded rows without changing the recurrence between real rows.
    if batch > 1:
        q[-1, -7:] = 0; key[-1, -7:] = 0; value[-1, -7:] = 0
    tensors = [x.requires_grad_(not final_state) for x in (q, key, value, gate, beta)]
    boundaries = [0, 65, 68, length] if packed else [0, length]
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int32) if packed else None
    with catalog_scope(catalog):
        out, state = chunk_gated_delta_rule(*tensors[:3], g=gate, beta=beta, cu_seqlens=cu,
                                           output_final_state=final_state, use_qk_l2norm_in_kernel=True)
        cotangent = torch.randn_like(out)
        gradients = torch.autograd.grad(out, tensors, cotangent) if not final_state else []
    assert torch.isfinite(out).all() and all(torch.isfinite(x).all() for x in gradients)
    errors = []
    if reference:
        copies = [x.detach().float().requires_grad_(not final_state) for x in tensors]
        qref, kref, vref, gref, bref = copies
        # Native normalization rounds to BF16 before the delta recurrence.
        qref = (qref * torch.rsqrt(qref.square().sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
        kref = (kref * torch.rsqrt(kref.square().sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
        pieces, states = [], []
        for start, end in zip(boundaries, boundaries[1:]):
            expected, expected_state = naive_recurrent_gated_delta_rule(qref[:, start:end], kref[:, start:end],
                    vref[:, start:end], bref[:, start:end], gref[:, start:end], output_final_state=final_state)
            pieces.append(expected)
            if expected_state is not None:
                states.append(expected_state)
        expected = torch.cat(pieces, dim=1)
        comparisons = [(out, expected)]
        if final_state:
            comparisons.append((state, torch.cat(states)))
        else:
            expected_gradients = torch.autograd.grad(expected, copies, cotangent.float())
            comparisons.extend(zip(gradients, expected_gradients))
        for actual, expected in comparisons:
            relative = float((actual.float() - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8))
            assert relative < .03, (geometry, batch, packed, final_state, relative)
            errors.append(relative)
    torch.cuda.synchronize()
    return max(errors, default=0.0)


def validate(geometry, catalog):
    errors = [exercise(geometry, catalog, length=129),
              exercise(geometry, catalog, length=129, batch=3),
              exercise(geometry, catalog, length=129, batch=7),
              exercise(geometry, catalog, length=129, packed=True),
              exercise(geometry, catalog, length=65, final_state=True)]
    unused = set(inventory()) - catalog.visited
    assert catalog.visited <= set(inventory()), (catalog.visited, set(inventory()))
    if unused:      # FLA's backend dispatch routes some ops to TileLang on Hopper (e.g. chunk_bwd_dqkwg): those Triton tuners never run here
        print(f"Autotuners: Triton kernels not used on this platform: {sorted(unused)}", flush=True)
    catalog.unused_kernels = sorted(unused)
    return max(errors)


def main():
    import torch
    request = json.loads(Path(sys.argv[1]).read_text())
    apply_runtime_settings(request["identities"][0]["environment"]["runtime"])
    torch.cuda.set_device(request["device"])
    torch.manual_seed(314159)
    started = time.monotonic()
    deadline = started + request["seconds"] - 5
    env = environment(torch.device("cuda", request["device"]))
    for identity in request["identities"]:
        assert identity["environment"] == env, "Parent and worker kernel environments differ"
        geometry = identity["geometry"]
        catalog = Catalog(discover=True)
        error = validate(geometry, catalog)
        for length in (2048, 8192, 32768):
            exercise(geometry, catalog, length=length, reference=False)
        def save(phase, selected, error):
            seconds = time.monotonic() - started
            profile = {"identity": identity, "coverage": MODES, "unused_kernels": getattr(selected, "unused_kernels", []),
                       "records": sorted(selected.records.values(), key=lambda r: (r["kernel"], r["key"], r["bucket"])),
                       "validation": {"passed": True, "max_relative_rms_error": error, "batches": [1, 3, 7],
                                      "measured_lengths": [65, 129, 2048, 8192, 32768],
                                      "reuse": "independent rows/batches; performance beyond measured lengths is unmeasured"},
                       "tuning": {"phase": phase, "seconds": round(seconds, 3), "benchmarks": selected.benchmarks,
                                  "extension_reason": "cold compilation and correctness coverage" if seconds > TARGET_SECONDS else None}}
            path = publish(profile, resolve_path("AUTOTUNE/profiles"))
            print(f"Published {phase}: {geometry} in {seconds:.1f}s ({path.name})", flush=True)
        save("validated_baseline", catalog, error)
        # First finish every requested baseline. Improvements cannot consume the
        # budget needed by another geometry's initial usable result.
    for identity in request["identities"]:
        if time.monotonic() >= min(started + TARGET_SECONDS, deadline - 40):
            break
        from .core import lookup
        baseline = lookup(identity)[0]
        geometry = identity["geometry"]
        trial = Catalog(baseline["records"], discover=True, optimize=True, deadline=min(started + TARGET_SECONDS, deadline - 40))
        exercise(geometry, trial, length=2048, reference=False)
        trial.optimize = False
        error = validate(geometry, trial)
        for length in (8192, 32768):
            exercise(geometry, trial, length=length, reference=False)
        save("bounded_search", trial, error)




def _execution_gradient_check(reference, actual):
    """Compare every trainable gradient, including None ownership, by optimizer group."""
    import math
    import torch
    if reference.keys() != actual.keys():
        return {"passed": False, "reason": "gradient ownership changed"}
    sums = {}
    for key, first in reference.items():
        second = actual[key]
        if (first is None) != (second is None):
            return {"passed": False, "reason": f"gradient presence changed: {key}"}
        if first is None:
            continue
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            return {"passed": False, "reason": f"nonfinite gradient: {key}"}
        values = sums.setdefault(key.split(":")[0], [0., 0., 0., 0.])
        first, second = first.float(), second.float()
        values[0] += float(first.square().sum())
        values[1] += float(second.square().sum())
        values[2] += float((first - second).square().sum())
        values[3] += float((first * second).sum())
    groups = {key: {"relative_l2": math.sqrt(delta / max(first, 1e-30)),
                    "cosine": dot / math.sqrt(first * second) if first * second else float(first == second),
                    "reference_norm": math.sqrt(first)}
              for key, (first, second, delta, dot) in sums.items()}
    return {"passed": bool(groups) and all(row["relative_l2"] <= .05 and row["cosine"] >= .995 for row in groups.values()),
            "groups": groups}


def _execution_progress(path: Path, phase: str, completed_cases: int, total_cases: int, started: float, **details):
    payload = {"phase": phase, "completed_cases": completed_cases, "total_cases": total_cases,
               "elapsed_seconds": time.monotonic() - started, **details}
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(payload, allow_nan=False))
    temporary.replace(path)


def ac_execution_main(request_path):
    """Disposable full-model SFT timing; trusted pickle is created by the parent API."""
    import copy
    from dataclasses import replace
    import gc
    import os
    import pickle
    import random
    import traceback
    import torch
    from .core import digest, execution_candidates, SCHEMA, EXECUTION_SCHEMA
    with Path(request_path).open("rb") as handle:
        request = pickle.load(handle)
    apply_runtime_settings(request["identity"]["environment"]["runtime"])
    os.environ["ACTIVATION_SYNC_ROOT"] = request["sync_root"]
    Path(request["sync_root"]).mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(request["device"])
    device = torch.device("cuda", request["device"])
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    from ..harness import HarnessRuntime
    from ..ac_model.ac_model import TRAINING
    from ..ac_model.ac_model_training import ActivationContextTrainer
    from ..ac_model.ac_model_utils import execution_groups, execution_batch_loss, AdamMomentPlacement
    from ..agent_training.agent_training_utils import disable_dropout, optimizer_state_to, set_learning_rates
    from . import get_ac_training_autotuning_variables, configure_fla_runtime
    started = time.monotonic()
    deadline = started + request["seconds"]
    config = replace(request["training_config"], resume_optimizer=False, resume_skip_steps=0,
                     micro_batch_examples=1, execution_policy=None, max_duration_seconds=None)
    candidates = execution_candidates(config)
    completed_cases = 0
    capacity_cases = request["capacity_cases"]
    capacity_policies = request["capacity_policies"]
    if [row["policy"] for row in capacity_policies] != candidates:
        raise ValueError("Execution worker candidate policies differ from the capacity request")
    total_cases = (len(request["batches"]) * (len(candidates) + 2) + 2 * len(capacity_cases)
                   + sum(len(row["cases"]) for row in capacity_policies))
    progress_path = Path(request.get("progress", str(Path(request["output"]).with_name("progress.json"))))
    def progress(phase, **details):
        _execution_progress(progress_path, phase, completed_cases, total_cases, started, **details)
    progress("model registration")
    torch.manual_seed(config.seed)
    harness = HarnessRuntime(request["harness_config"])
    for lora in request["loras"]:
        harness.module_manager.register_lora(lora.lora_name, lora.model_name, rank=lora.rank, alpha=lora.alpha,
                                             dropout=lora.dropout, target_modules=lora.target_modules,
                                             checkpoint_path=lora.checkpoint_path)
    ac = harness.module_manager.register_ac_model(request["ac_config"])
    ac.modules.load_state_dict(request["ac_state"], strict=True)
    trainer = ActivationContextTrainer(harness, config)
    trainer._training_deadline = None
    # Reuse the shared kernel catalog without writing trial checkpoints into a run.
    progress("kernel preparation")
    os.environ["ACTIVATION_SYNC_ROOT"] = request["autotune_root"]
    try:
        variables = get_ac_training_autotuning_variables(ac.side.model_config, ac.target.model_config, config,
                                                        ac_config=ac.config, device=device)
    finally:
        os.environ["ACTIVATION_SYNC_ROOT"] = request["sync_root"]
    trainer._logits_chunk_tokens = variables["logits_chunk_tokens"]
    progress("model placement")
    harness.module_manager.ensure_lora(ac.config.target_model_lora_name)
    ac.prepare()
    ac.set_mode(TRAINING)
    for base in dict.fromkeys((ac.side.model, ac.target.model)):
        base.train(); disable_dropout(base)
    ac._training_gradient_checkpointing = config.gradient_checkpointing
    # Match train()'s FP32 LoRA/gate matmul mode, not the process default.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    examples = request["examples"]
    batches = [[examples[i] for i in group] for group in request["batches"]]
    trainer._check_capacity(ac, examples)
    groups = [{"params": ac.trainable_parameters(), "lr": config.learning_rate_ac},
              {"params": harness.module_manager.lora_parameters(ac.config.target_model_lora_name),
               "lr": config.learning_rate_target_lora}]
    optimizer = torch.optim.AdamW(groups, betas=config.adam_betas, eps=config.adam_eps, weight_decay=config.weight_decay)
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]
    set_learning_rates(optimizer, 0, config.warmup_updates, total_updates=config.schedule_total_updates,
                       schedule=config.lr_schedule, final_fraction=config.lr_final_fraction)
    params = {f"{group}:{index}": parameter for group, values in enumerate(optimizer.param_groups)
              for index, parameter in enumerate(values["params"])}
    placement = None
    rng = random.Random(config.seed)
    flags = [[bool(example.penalty is not None and config.penalty_active
                   and rng.random() < config.no_context_penalty_fraction) for example in batch] for batch in batches]
    records = []
    baseline_gradients, baseline_losses = [], []
    capacity_references = {}
    frozen_weights = frozen_optimizer = None

    def publish_result():
        passing = [row for row in records if row["passed"]]
        if not passing:
            return
        winner = min(passing, key=lambda row: row["seconds_per_question"])
        profile = {"kind": "ac_execution", "execution_schema": EXECUTION_SCHEMA,
                   "identity": request["identity"], "policy": winner["policy"], "candidates": records,
                   "validation": {"passed": True, "loss_absolute_tolerance": .01, "gradient_relative_l2": .05,
                                  "gradient_cosine": .995, "baseline_repeat": True},
                   "sampling": {"batch_indexes": request["batches"], "penalty_flags": flags,
                                "capacity_cases": capacity_cases,
                                "scope": "sampled update shapes and distinct-history padding tail for speed; policy-specific checkpoint boundaries for capacity, with fixed existing penalties; no training-quality or full-resume claim"},
                   "seconds": time.monotonic() - started,
                   "limits": "Same supplied-workload SFT policy; unchanged reference/evaluation execution. Speed includes forward/backward/clipping/Adam/transfers; excludes capacity checks, reference preparation and reporting."}
        payload = {"schema": SCHEMA, "profile": profile, "sha256": digest(profile)}
        path = Path(request["output"])
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(payload, allow_nan=False))
        temporary.replace(path)

    def clear_graphs():
        optimizer.zero_grad(set_to_none=True)
        trainer._penalty_term = None
        trainer._last_hits = None
        ac.invalidate_encoded_rows()
        for value in ac.intervention_diagnostics.values():
            value.clear()
        trainer._row_rms.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def configure(policy, restore=True):
        nonlocal placement
        if placement is not None:
            placement.close(); placement = None
        clear_graphs()
        if restore and frozen_weights is not None:
            with torch.no_grad():
                for key, parameter in params.items():
                    parameter.copy_(frozen_weights[key])
            # load_state_dict casts moments to the parameter device; clone CPU step tensors
            # because Adam updates them in-place and the immutable snapshot must not alias.
            state = {key: {**value, "step": value["step"].clone()} for key, value in frozen_optimizer["state"].items()}
            optimizer.load_state_dict({"state": state, "param_groups": copy.deepcopy(frozen_optimizer["param_groups"])})
        if policy["optimizer_moments"] == "cpu":
            placement = AdamMomentPlacement(optimizer)
            placement.park()
        else:
            optimizer_state_to(optimizer, device)
        ac.config.encode_batch_max_tokens = policy["encode_batch_tokens"]
        ac._side_gradient_checkpointing_min_tokens = policy["side_checkpoint_tokens"]
        trainer._target_gradient_checkpointing_min_tokens = policy["target_checkpoint_tokens"]
        torch.cuda.synchronize()

    class FixedPenaltyDraws:
        def __init__(self, batch, selected):
            self.values = iter(0. if flag else 1. for example, flag in zip(batch, selected)
                               if example.penalty is not None and config.penalty_active)
        def random(self):
            return next(self.values)

    def measure(policy, batch, selected, *, native=False, capture=True):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
        begin = time.monotonic()
        objective_value = 0.
        encodes_before, side_before = ac.stats.encodes, ac.stats.side_tokens
        if native:
            trainer._penalty_rng = FixedPenaltyDraws(batch, selected)
            for example in batch:
                objective, _, _, _ = trainer._example_loss(ac, example, device)
                penalty = trainer._penalty_term
                loss = (objective + (penalty if penalty is not None else 0.)) * (example.item.weight / len(batch))
                loss.backward(); objective_value += float(loss.detach())
                del objective, penalty, loss
        else:
            for group in execution_groups(batch, policy):
                actual = [batch[i] for i in group]
                outcomes = execution_batch_loss(trainer, ac, actual, device, [selected[i] for i in group], policy)
                loss = sum((row["objective"] + (row["penalty"] if row["penalty"] is not None else 0.))
                           * (example.item.weight / len(batch)) for example, row in zip(actual, outcomes))
                loss.backward(); objective_value += float(loss.detach())
                del outcomes, loss, actual
        if config.learning_rate_target_lora == 0:
            for parameter in groups[1]["params"]:
                parameter.grad = None
        torch.cuda.synchronize()
        backward_seconds = time.monotonic() - begin
        gradients = {key: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
                     for key, parameter in params.items()} if capture else None
        # CPU gradient copies/validation are excluded from measured execution time.
        before_step = time.monotonic()
        torch.nn.utils.clip_grad_norm_([p for p in params.values() if p.grad is not None], config.max_grad_norm)
        optimizer.step()
        ac.invalidate_encoded_rows()
        torch.cuda.synchronize()
        seconds = backward_seconds + time.monotonic() - before_step
        result = {"seconds": seconds, "questions": len(batch), "objective": objective_value,
                  "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                  "encodes": ac.stats.encodes - encodes_before, "side_tokens": ac.stats.side_tokens - side_before}
        return result, gradients

    try:
        with configure_fla_runtime(variables["fla_config"]):
            # Singleton, plain-base no-context references stay independent of candidate batching.
            progress("reference preparation")
            trainer._store_teacher_targets(ac, [example.penalty for example in examples if example.penalty is not None],
                                           device, lora=None)
            configure(candidates[0], restore=False)
            progress("optimizer warmup")
            measure(candidates[0], batches[0], flags[0], native=True, capture=False)
            # Every comparison starts at this identical, real post-Adam state.
            frozen_weights = {key: parameter.detach().cpu().clone() for key, parameter in params.items()}
            frozen_optimizer = copy.deepcopy(optimizer.state_dict())
            for state in frozen_optimizer["state"].values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.detach().cpu().clone()
            baseline_record = {"policy": candidates[0], "passed": True, "cases": [], "capacity_cases": [], "baseline": True}
            for index, (batch, selected) in enumerate(zip(batches, flags)):
                progress("native baseline", case=index + 1)
                configure(candidates[0])
                first, gradients = measure(candidates[0], batch, selected, native=True)
                completed_cases += 1
                progress("native repeat", case=index + 1)
                configure(candidates[0])
                repeated, repeated_gradients = measure(candidates[0], batch, selected, native=True)
                comparison = _execution_gradient_check(gradients, repeated_gradients)
                passed = comparison["passed"] and abs(first["objective"] - repeated["objective"]) <= .01
                if not passed:
                    raise RuntimeError(f"Native repeated baseline failed numerical gate: {comparison}")
                completed_cases += 1
                first.update({"case": index, "repeat": repeated, "gradient_check": comparison, "passed": True})
                progress("singleton helper", case=index + 1)
                configure(candidates[0])
                helper, helper_gradients = measure(candidates[0], batch, selected)
                helper_check = _execution_gradient_check(gradients, helper_gradients)
                if not helper_check["passed"] or abs(helper["objective"] - first["objective"]) > .01:
                    raise RuntimeError(f"Execution helper singleton failed native equivalence: {helper_check}")
                completed_cases += 1
                helper.update({"case": index, "native": first, "gradient_check": helper_check, "passed": True})
                helper["passed"] = helper["peak_memory_bytes"] < torch.cuda.get_device_properties(device).total_memory * .96
                baseline_record["passed"] = baseline_record["passed"] and helper["passed"]
                baseline_record["cases"].append(helper)
                baseline_gradients.append(gradients); baseline_losses.append(first["objective"])
                del repeated_gradients, helper_gradients
            baseline_record["seconds_per_question"] = sum(row["seconds"] for row in baseline_record["cases"]) / sum(map(len, batches))
            records.append(baseline_record); publish_result()
            for candidate_index, policy in enumerate(candidates[1:], 1):
                if time.monotonic() >= deadline:
                    break
                record = {"policy": policy, "cases": [], "capacity_cases": [], "passed": True}
                try:
                    for index, (batch, selected) in enumerate(zip(batches, flags)):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Execution search budget reached")
                        progress("candidate case", candidate=candidate_index, policy=policy, case=index + 1)
                        configure(policy)
                        row, gradients = measure(policy, batch, selected)
                        comparison = _execution_gradient_check(baseline_gradients[index], gradients)
                        row.update({"case": index, "gradient_check": comparison,
                                    "loss_difference": abs(row["objective"] - baseline_losses[index])})
                        row["passed"] = comparison["passed"] and row["loss_difference"] <= .01
                        # Avoid choosing an arm with essentially no remaining capacity.
                        row["passed"] = row["passed"] and row["peak_memory_bytes"] < torch.cuda.get_device_properties(device).total_memory * .96
                        record["cases"].append(row)
                        completed_cases += 1
                        del gradients
                        if not row["passed"]:
                            record["passed"] = False
                            break
                    if record["passed"]:
                        for case_index in capacity_policies[candidate_index]["cases"]:
                            if time.monotonic() >= deadline:
                                raise TimeoutError("Execution search budget reached before capacity validation")
                            case = capacity_cases[case_index]
                            batch = [examples[i] for i in case["indices"]]
                            selected = case["penalty_flags"]
                            if case_index not in capacity_references:
                                progress("capacity native baseline", capacity_case=case_index, reasons=case["reasons"])
                                configure(candidates[0])
                                first, gradients = measure(candidates[0], batch, selected, native=True)
                                completed_cases += 1
                                progress("capacity native repeat", capacity_case=case_index)
                                configure(candidates[0])
                                repeated, repeated_gradients = measure(candidates[0], batch, selected, native=True)
                                comparison = _execution_gradient_check(gradients, repeated_gradients)
                                completed_cases += 1
                                if not comparison["passed"] or abs(first["objective"] - repeated["objective"]) > .01:
                                    raise RuntimeError(f"Capacity repeated baseline failed numerical gate: {comparison}")
                                first.update({"repeat": repeated, "gradient_check": comparison})
                                capacity_references[case_index] = (first, gradients)
                                del repeated_gradients, gradients
                            first, native_gradients = capacity_references[case_index]
                            progress("candidate capacity", candidate=candidate_index, policy=policy,
                                     capacity_case=case_index, reasons=case["reasons"])
                            configure(policy)
                            row, gradients = measure(policy, batch, selected)
                            comparison = _execution_gradient_check(native_gradients, gradients)
                            row.update({"capacity_case": case_index, "gradient_check": comparison,
                                        "loss_difference": abs(row["objective"] - first["objective"]), "native": first})
                            row["passed"] = (comparison["passed"] and row["loss_difference"] <= .01
                                and row["peak_memory_bytes"] < torch.cuda.get_device_properties(device).total_memory * .96)
                            record["capacity_cases"].append(row)
                            completed_cases += 1
                            del gradients
                            if not row["passed"]:
                                record["passed"] = False
                                break
                except (torch.cuda.OutOfMemoryError, TimeoutError) as error:
                    record.update({"passed": False, "error": str(error), "error_type": type(error).__name__})
                    error.__traceback__ = None  # release the failed forward's graph before the next restored candidate
                    clear_graphs()
                record["seconds_per_question"] = (sum(row["seconds"] for row in record["cases"]) / sum(map(len, batches))) if record["passed"] else None
                records.append(record); publish_result()
                progress("candidate complete", candidate=candidate_index, policy=policy, passed=record["passed"])
                print(json.dumps({"policy": policy, "passed": record["passed"], "seconds_per_question": record["seconds_per_question"]}), flush=True)
            progress("complete", tested_candidates=len(records), planned_candidates=len(candidates),
                     budget_exhausted=time.monotonic() >= deadline)
    except Exception as error:
        progress("failed", error=str(error))
        traceback.print_exc()
        raise
    finally:
        if placement is not None:
            placement.close()


def _parallel_tensor_hash(tensors):
    import hashlib
    import torch
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _parallel_fixed_weights(harness, ac, config, items, folder, trial, parameters, keys):
    """Exercise native global batches without Adam updates; copies never enter speed results."""
    import torch
    import torch.distributed as dist
    from ..ac_model import ac_model_training as training
    from ..ac_model.ac_model_utils import DistributedTraining, canonical_json
    from .core import digest
    checks, records = [], []
    names = {id(p): 'ac.' + name for name, p in ac.modules.named_parameters()}
    manager = harness.module_manager
    for prefix, lora in [('side.', ac.config.base_side_model_lora_name), ('reader.', ac.config.target_model_lora_name)]:
        model = manager.ensure_lora(lora)
        adapter = manager.get_lora_config(lora).adapter_name
        names.update({id(p): prefix + name for name, p in model.named_parameters() if f'.{adapter}.' in name})
    original_epoch, original_version = ac.epoch_number, ac.version
    initial_hash = _parallel_tensor_hash(parameters)

    class ProbeTrainer(training.ActivationContextTrainer):
        def record(self, example, objective, penalty, flag):
            reference = (self.teacher_cache[example.penalty.teacher_cache_key]
                         if example.penalty is not None else [])
            identity = canonical_json((example.item.item_id, example.item.weight, list(example.student_ids),
                list(example.student_positions), list(example.target_ids), [request._asdict() for request in example.part_requests]))
            records.append({'key': digest(identity), 'penalty_flag': bool(flag),
                'reference': _parallel_tensor_hash(reference), 'objective': float(objective.detach()),
                'penalty': None if penalty is None else float(penalty.detach())})

        def _batch_loss(self, ac_model, batch, device, *args, **kwargs):
            result = super()._batch_loss(ac_model, batch, device, *args, **kwargs)
            flags = kwargs.get('penalty_flags')
            if flags is None:
                raise RuntimeError('Parallel correctness probe requires native preassigned penalties')
            for example, value, flag in zip(batch, result, flags):
                self.record(example, value['objective'], value['penalty'], flag)
            return result

        def _example_loss(self, ac_model, example, device, *args, **kwargs):
            result = super()._example_loss(ac_model, example, device, *args, **kwargs)
            self.record(example, result[0], self._penalty_term, kwargs['penalty_flag'])
            return result

    trainer = ProbeTrainer(harness, config)

    class ProbeParallel(DistributedTraining):
        def initialize(self, values, identity):
            super().initialize(values, identity)
            if [id(p) for p in self.parameters] != [id(p) for p in parameters]:
                raise RuntimeError('Native optimizer parameter order differs from the probe')
            layout = [(names[id(p)], tuple(p.shape), str(p.dtype)) for p in self.parameters]
            if any(value != layout for value in self.gather(layout)):
                raise RuntimeError('Named replica parameter layouts differ')
            self.probe_step = 0
            # Suppress only the optimizer mutation. Native clock, penalty RNG,
            # backward, reduction, clipping and accounting still execute.
            trainer.optimizers[ac.name].step = lambda *args, **kwargs: None
            def initial_check():
                path = folder / 'fixed_initial.json'
                current = {'layout': layout, 'weights': initial_hash}
                current = json.loads(json.dumps(current))
                if trial == 'baseline':
                    path.write_text(json.dumps(current) + '\n')
                    return True
                return json.loads(path.read_text()) == current
            checks.append({'initial_weights_and_named_layout': self.main(initial_check)})

        def reduce_gradients(self):
            # An independent FP64 SUM bounds only dtype rounding; it never replaces
            # native gradients. Rank results must also be bitwise identical.
            present = self.gather([p.grad is not None for p in self.parameters])
            expected_present = [any(row[i] for row in present) for i in range(len(parameters))]
            references = []
            for bucket in self.buckets:
                flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).double()
                                  for p in [self.parameters[i] for i in bucket]])
                magnitude = flat.abs()
                if self.active:
                    dist.reduce(flat, dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(magnitude, dst=0, op=dist.ReduceOp.SUM)
                if self.is_main:
                    references.append((flat.cpu(), magnitude.cpu()))
            super().reduce_gradients()
            ownership = expected_present == [p.grad is not None for p in self.parameters]
            identical, reducer = True, True
            for position, bucket in enumerate(self.buckets):
                flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1)
                                  for p in [self.parameters[i] for i in bucket]])
                replica = flat.clone()
                if self.active:
                    dist.broadcast(replica, src=0)
                identical = identical and torch.equal(flat, replica)
                if self.is_main:
                    expected, magnitude = references[position]
                    precision = torch.finfo(flat.dtype)
                    error = (flat.cpu().double() - expected).abs()
                    bound = self.world_size * (precision.eps * magnitude + precision.tiny)
                    reducer = reducer and bool(torch.all(error <= bound))
            del references
            collective_checks = self.gather((ownership, identical, reducer))
            merged_records = sorted((row for rows in self.gather(records) for row in rows),
                                    key=lambda row: (row['key'], row['penalty_flag']))
            records.clear()
            step = self.probe_step
            def compare():
                gradients = {key: None if p.grad is None else p.grad.detach().cpu().clone()
                             for key, p in zip(keys, self.parameters)}
                path = folder / f'fixed_batch_{step}.pt'
                if trial == 'baseline':
                    torch.save({'gradients': gradients, 'records': merged_records}, path)
                    comparison = {'passed': True, 'baseline': True}
                else:
                    before = torch.load(path, map_location='cpu', weights_only=False)
                    before_records = before['records']
                    signatures = lambda rows: [{k: v for k, v in row.items() if k not in ('objective', 'penalty')} for row in rows]
                    reference_match = signatures(before_records) == signatures(merged_records)
                    presence_match = (len(before_records) == len(merged_records) and all(
                        (x['penalty'] is None) == (y['penalty'] is None) for x, y in zip(before_records, merged_records)))
                    loss_error = max((abs(x['objective'] - y['objective']) for x, y in zip(before_records, merged_records)), default=0.)
                    penalty_error = max((abs((x['penalty'] or 0.) - (y['penalty'] or 0.))
                                         for x, y in zip(before_records, merged_records)), default=0.)
                    gradient = _execution_gradient_check(before['gradients'], gradients)
                    comparison = {'passed': bool(reference_match and presence_match and loss_error <= .01 and penalty_error <= .01 and gradient['passed']),
                        'gradients': gradient, 'exact_items_penalties_references': reference_match and presence_match,
                        'objective_max_abs_difference': loss_error, 'penalty_max_abs_difference': penalty_error}
                comparison.update({'batch': step, 'examples': len(merged_records),
                    'exact_gradient_ownership': all(row[0] for row in collective_checks),
                    'replicas_bitwise_equal': all(row[1] for row in collective_checks),
                    'reducer_matches_fp64_sum_with_dtype_rounding': all(row[2] for row in collective_checks)})
                comparison['passed'] = comparison['passed'] and all(comparison[k] for k in (
                    'exact_gradient_ownership', 'replicas_bitwise_equal', 'reducer_matches_fp64_sum_with_dtype_rounding'))
                return comparison
            checks.append(self.main(compare))
            self.probe_step += 1

    previous_parallel = training.DistributedTraining
    try:
        training.DistributedTraining = ProbeParallel
        stats = trainer.train(ac.name, items)
        unchanged = _parallel_tensor_hash(parameters) == initial_hash
        if not unchanged:
            raise RuntimeError('Fixed-weight probe changed model weights')
        if trainer.optimizers[ac.name].state:
            raise RuntimeError('Fixed-weight probe unexpectedly allocated Adam moments')
        result = {'passed': unchanged and all(row.get('passed', row.get('initial_weights_and_named_layout', False)) for row in checks),
                  'checks': checks, 'updates': stats.steps, 'penalty_counts': stats.penalty_counts,
                  'loss': stats.loss, 'weight_hash_unchanged': unchanged,
                  'seconds': stats.duration_s, 'scope': 'Native forward/backward/global SUM/clipping with Adam.step suppressed; no speed claim.'}
    finally:
        training.DistributedTraining = previous_parallel
        ac.epoch_number, ac.version = original_epoch, original_version
    return result


def ac_parallelism_main(request_path, trial='baseline'):
    """Keep one process group across the two native trainer calls."""
    import pickle
    import torch.distributed as dist
    from ..ac_model.ac_model_utils import DistributedTraining
    with Path(request_path).open('rb') as stream:
        request = pickle.load(stream)
    apply_runtime_settings(request['identity']['environment']['runtime'])
    with DistributedTraining(True) as parallel:
        result = _ac_parallelism_run(request, trial)
        if parallel.active:
            dist.barrier()
        return result


def _ac_parallelism_run(request, trial):
    """Fixed-weight correctness, followed by separate genuine sequential Adam timing."""
    import gc
    import os
    import torch
    from ..harness import HarnessRuntime
    from ..ac_model.ac_model_training import ActivationContextTrainer
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    folder = Path(request['folder'])
    os.environ['ACTIVATION_SYNC_ROOT'] = request['autotune_root']
    config = request['training_config']
    torch.manual_seed(config.seed)
    harness = HarnessRuntime(request['harness_config'])
    for lora in request['loras']:
        harness.module_manager.register_lora(lora.lora_name, lora.model_name, rank=lora.rank,
            alpha=lora.alpha, dropout=lora.dropout, target_modules=lora.target_modules,
            checkpoint_path=lora.checkpoint_path)
    ac = harness.module_manager.register_ac_model(request['ac_config'])
    ac.modules.load_state_dict(request['ac_state'], strict=True)
    harness.module_manager.ensure_lora(ac.config.target_model_lora_name)
    ac.prepare()
    encoder_parameters = ac.trainable_parameters()
    parameters = encoder_parameters + harness.module_manager.lora_parameters(ac.config.target_model_lora_name)
    keys = [f'{int(i >= len(encoder_parameters))}:{i}' for i in range(len(parameters))]
    initial = [p.detach().cpu().clone() for p in parameters] if rank == 0 else None
    print(f'Parallel trial {trial}: fixed-weight correctness phase', flush=True)
    fixed = _parallel_fixed_weights(harness, ac, config, request['training_items'], folder, trial, parameters, keys)
    if rank == 0:
        print(json.dumps({'trial': trial, 'fixed_weight_validation': fixed}), flush=True)
    gc.collect()
    print(f'Parallel trial {trial}: genuine sequential-update timing phase', flush=True)
    trainer = ActivationContextTrainer(harness, config)
    stats = trainer.train(ac.name, request['training_items'])
    if rank != 0:
        return
    deltas = {keys[i]: p.detach().cpu().float() - before.float() for i, (p, before) in enumerate(zip(parameters, initial))}
    optimizer = trainer.optimizers[ac.name]
    states = {keys[i]: {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                      for key, value in optimizer.state[p].items()} for i, p in enumerate(parameters) if p in optimizer.state}
    baseline_path = folder / 'sequential_baseline.pt'
    checks, drift = {}, {}
    if trial == 'baseline':
        torch.save({'deltas': deltas, 'optimizer': states, 'loss': stats.loss,
                    'norms': {name: getattr(stats, name) for name in ('grad_norm', 'grad_norm_ac', 'grad_norm_reader')},
                    'penalty_counts': stats.penalty_counts, 'steps': stats.steps}, baseline_path)
        checks['baseline'] = True
    else:
        baseline = torch.load(baseline_path, map_location='cpu', weights_only=False)
        checks['optimizer_ownership'] = states.keys() == baseline['optimizer'].keys()
        checks['optimizer_clocks'] = checks['optimizer_ownership'] and all(
            torch.equal(state['step'], baseline['optimizer'][key]['step']) for key, state in states.items())
        checks['native_updates_and_penalty_counts'] = stats.steps == baseline['steps'] and stats.penalty_counts == baseline['penalty_counts']
        drift['updates'] = _execution_gradient_check(baseline['deltas'], deltas)
        before = {key + ':' + name: value for key, state in baseline['optimizer'].items()
                  for name, value in state.items() if name != 'step' and isinstance(value, torch.Tensor)}
        after = {key + ':' + name: value for key, state in states.items()
                 for name, value in state.items() if name != 'step' and isinstance(value, torch.Tensor)}
        drift['moments'] = _execution_gradient_check(before, after)
        drift['loss_max_abs_difference'] = max(abs(a - b) for a, b in zip(baseline['loss'], stats.loss))
        drift['preclip_norms'] = {name: {'baseline': reference, 'actual': getattr(stats, name)} for name, reference in baseline['norms'].items()}
    measured = stats.step_seconds[1:]
    result = {'gpus': world, 'trial': trial, 'passed': bool(fixed['passed'] and all(checks.values())),
        'fixed_weight_validation': fixed, 'native_sequential_checks': checks,
        'sequential_drift_diagnostic': drift,
        'seconds_per_update': sum(measured) / len(measured), 'step_seconds': stats.step_seconds,
        'examples': len(request['training_items']), 'global_batch': config.examples_per_update,
        'peak_memory_bytes': stats.peak_memory_bytes, 'total_trainer_seconds': stats.duration_s,
        'placement_seconds': stats.placement_seconds, 'cleanup_seconds': stats.cleanup_seconds,
        'other_seconds': stats.other_seconds, 'loss': stats.loss,
        'populated_adam_parameters': len(states)}
    path = folder / f'result_{trial}.json'; temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, allow_nan=False, indent=2) + '\n'); temporary.replace(path)
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    if sys.argv[1:2] == ["--ac-parallelism"]:
        ac_parallelism_main(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "baseline")
    elif sys.argv[1:2] == ["--ac-execution"]:
        ac_execution_main(sys.argv[2])
    else:
        main()
