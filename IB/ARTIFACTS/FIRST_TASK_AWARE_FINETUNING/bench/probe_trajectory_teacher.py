"""Real 2B→9B compaction gate: two Adam updates, paired save/reload, two more.

Uses real portable material and the campaign item builder. Instrumentation records
real forwards; it never substitutes model outputs. Separate child processes make
checkpoint reload independent of live model/optimizer state.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from run_qa import AC, READER, harness, write
from run_trajectory_tasks import build_items, material_rows, digest


def fingerprint(value):
    import torch
    result = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            result.update(str((tuple(x.shape), str(x.dtype))).encode())
            result.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                add(key); add(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x:
                add(item)
        else:
            result.update(repr(x).encode())
    add(value)
    return result.hexdigest()


def adapter_state(manager, lora_name):
    # lora_parameters() changes requires_grad; inspection must not unfreeze the teacher.
    config = manager.get_lora_config(lora_name)
    model = manager.ensure_lora(lora_name)
    marker = f".{config.adapter_name}."
    parameters = [(name.replace(marker, ".ADAPTER."), p) for name, p in model.named_parameters() if marker in name]
    assert parameters, lora_name
    return fingerprint(dict(parameters)), all(not p.requires_grad for _, p in parameters)


def select_entries(args):
    path = args.output / "selection.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["material"] != str(args.material.resolve()):
            raise ValueError("Probe selection belongs to a different material bank")
        return saved["entries"]
    candidates = []
    for row_index, (shard, ordinal, row) in enumerate(material_rows(args.material)):
        if row_index >= args.scan_rows:
            break
        if row["split"] != "train":
            continue
        for choice in row["compaction_cuts"]:
            candidates.append({"shard": shard, "ordinal": ordinal, "trajectory_id": row["id"],
                "root_id": row["root_id"], "source": row["source"], "row_sha256": digest(row), "choice": choice})
    if not candidates:
        raise ValueError("No published training compaction cuts")
    def heavy_score(entry):
        c = entry["choice"]
        return (c["source_tokens"] >= 28000 and c["continuation_tokens"] >= 6000,
                c["source_tokens"] >= 28000,
                min(c["source_tokens"] / 32768, 1) + min(c["continuation_tokens"] / 8192, 1))
    heavy = max(candidates, key=heavy_score)
    selected, roots = [heavy], {heavy["root_id"]}
    if heavy["choice"]["continuation_tokens"] < 6000:
        alternatives = [entry for entry in candidates if entry["root_id"] not in roots]
        if alternatives:
            suffix_heavy = max(alternatives, key=lambda entry: entry["choice"]["continuation_tokens"])
            selected.append(suffix_heavy); roots.add(suffix_heavy["root_id"])
    for entry in sorted(candidates, key=lambda e: e["choice"]["full_history_tokens"]):
        if entry["root_id"] in roots:
            continue
        selected.append(entry); roots.add(entry["root_id"])
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise ValueError("Need four distinct source roots for the real GPU gate")
    others = sorted(selected[1:], key=lambda e: e["choice"]["full_history_tokens"])
    # Allocate Adam moments on a short example before testing the heavy backward.
    selected = [others[0], heavy, *reversed(others[1:])]
    write(path, {"material": str(args.material.resolve()), "entries": selected,
        "heavy_32k_8k_available": bool(heavy_score(heavy)[0]),
        "note": "Actual lengths are authoritative; no padding or synthetic heavy trajectory."})
    return selected


def run_phase(args, entries):
    import torch
    from activation.ac_model import ActivationContextTrainingConfig, ActivationContextTrainer
    from activation.ac_model.ac_model_study import ActivationContextStudyGenerator
    if not torch.cuda.is_available():
        raise RuntimeError("This gate requires an allocated GPU; it must not run on a login node")
    os.environ["ACTIVATION_SYNC_ROOT"] = str((args.output / "training").resolve())
    config = ActivationContextTrainingConfig(loss_kind="kl", learning_rate_ac=5e-5,
        learning_rate_target_lora=2e-5, examples_per_update=1, micro_batch_examples=1,
        num_epochs=1, warmup_updates=0, schedule_total_updates=4, seed=7,
        teacher_cache_block_examples=2, no_context_penalty_weight=1, no_context_penalty_fraction=.25,
        reporting_interval=.5, completion_samples=0, logits_chunk_tokens=args.logits_chunk,
        gradient_checkpointing_min_tokens=1700, checkpoint_every_updates=None,
        max_duration_seconds=args.seconds, shuffle_examples=False)
    first = args.phase == "first"
    pair = json.loads(args.initial_pair.read_text()) if first else json.loads((args.output / "first.json").read_text())["pair"]
    if not first:
        config = replace(config, resume_optimizer=True, resume_skip_steps=2)
    h = harness(pair["ac_checkpoint"], pair["reader_checkpoint"])
    ac = h.module_manager.get_ac_model(AC)
    ac.epoch_number = 0                    # Independent task, with a fresh epoch numbering origin.
    ac.config.encode_batch_max_tokens = 65536
    trainer = ActivationContextTrainer(h, config)
    trainer._next_epoch_number = lambda model: 1
    study = ActivationContextStudyGenerator(h, AC)
    items = build_items(study, args.material, entries, "compaction", 1 / 16)
    probe = trainer.build_example(ac, items[-1])
    assert probe.penalty is not None
    reference_path = args.output / "initial_probe.pt"
    expected = None if first else torch.load(reference_path, map_location="cpu", weights_only=False)
    saved = None if first else torch.load(Path(pair["ac_checkpoint"]) / "optimizer.pt", map_location="cpu", weights_only=False)
    captures = []
    offloader, transfer_events = None, []
    def callback(progress, summary):
        nonlocal expected, offloader
        if args.offload_moments and progress.epoch == 0:
            # This gate has no reporting/validation set: this is the first forward boundary after
            # the actual optimizer restore, before the probe below or any training backward.
            from probe_adam_moment_offload_v1 import MomentOffload
            assert offloader is None, "Moment offload must enroll exactly once per process"
            offloader = MomentOffload(trainer.optimizers[AC], transfer_events)
            offloader.park()
        if progress.global_step not in (0, 2, 4):
            return
        if args.offload_moments:
            assert offloader is not None and offloader.optimizer is trainer.optimizers[AC]
            assert all(state[key].device.type == "cpu" for state in offloader.optimizer.state.values()
                       for key in ("exp_avg", "exp_avg_sq")), "Moments must be parked before reporting/probe forwards"
        snapshot = trainer._teacher_snapshot
        assert snapshot is not None
        teacher_hash, teacher_frozen = adapter_state(h.module_manager, snapshot["lora_name"])
        reader_hash, _ = adapter_state(h.module_manager, READER)
        assert teacher_frozen, "Teacher adapter became trainable"
        device = next(ac.target.model.parameters()).device
        # Force regeneration through the actual bounded-block preparation path.
        for example in [probe, probe.penalty]:
            trainer.teacher_cache.pop(example.teacher_cache_key, None)
            trainer.teacher_target_logp.pop(example.teacher_cache_key, None)
        routes = []
        original = ac.target.decoder_forward
        def observed(*a, **kw):
            routes.append(kw.get("lora_name"))
            return original(*a, **kw)
        ac.target.decoder_forward = observed
        try:
            trainer._prepare_teacher_block(ac, [probe], device)
        finally:
            ac.target.decoder_forward = original
        assert routes == [snapshot["lora_name"], None], routes
        logp = trainer.teacher_target_logp[probe.teacher_cache_key].clone()
        base_logp = trainer.teacher_target_logp[probe.penalty.teacher_cache_key].clone()
        assert torch.isfinite(logp).all() and torch.isfinite(base_logp).all()
        if expected is None:
            assert teacher_hash == reader_hash, "Initial snapshot differs from initial reader"
            expected = {"teacher_hash": teacher_hash, "initial_reader_hash": reader_hash,
                "teacher_logp": logp, "base_no_context_logp": base_logp,
                "snapshot_identity": snapshot["identity"]}
            torch.save(expected, reference_path)
        assert teacher_hash == expected["teacher_hash"]
        assert snapshot["identity"] == expected["snapshot_identity"]
        teacher_diff = float((logp - expected["teacher_logp"]).abs().max())
        base_diff = float((base_logp - expected["base_no_context_logp"]).abs().max())
        assert teacher_diff <= args.output_tolerance, teacher_diff
        assert base_diff <= args.output_tolerance, base_diff
        if progress.global_step:
            assert reader_hash != expected["initial_reader_hash"], "Actual Adam updates did not change the reader"
        optimizer_state = trainer.optimizers[AC].state_dict()["state"]
        record = {"global_step": progress.global_step, "teacher_hash": teacher_hash,
            "reader_hash": reader_hash, "teacher_frozen": teacher_frozen,
            "teacher_output_max_abs_diff": teacher_diff, "base_output_max_abs_diff": base_diff,
            "teacher_forward_routes": routes, "no_context_reference_is_plain_base": routes[-1] is None,
            "optimizer_moments_hash": fingerprint(optimizer_state),
            "penalty_rng_hash": fingerprint(trainer._penalty_rng.getstate()),
            "offload_moments": args.offload_moments,
            "moment_devices": sorted({str(state[key].device) for state in trainer.optimizers[AC].state.values()
                                      for key in ("exp_avg", "exp_avg_sq")}),
            "peak_memory_bytes": torch.cuda.max_memory_allocated()}
        if not first and not captures:
            assert progress.global_step == saved["updates_done"] == 2
            assert record["optimizer_moments_hash"] == fingerprint(saved["optimizer"]["state"])
            assert record["penalty_rng_hash"] == fingerprint(saved["penalty_rng_state"])
            previous = json.loads((args.output / "first.json").read_text())["captures"][-1]
            assert reader_hash == previous["reader_hash"], "Reloaded reader differs from the paired checkpoint"
            record["optimizer_and_rng_restored"] = True
        captures.append(record)
        write(args.output / f"{args.phase}_progress.json", {"phase": args.phase, "captures": captures,
              "offload_moments": args.offload_moments, "transfer_events": transfer_events})
        print(json.dumps(record), flush=True)
        if first and progress.global_step == 2:
            # Exercise the trainer's real time-limit partial-save path at a known update.
            trainer._training_deadline = 0.0
    try:
        stats = trainer.train(AC, items, reporting_callback=callback)
    finally:
        try:
            if offloader is not None:
                offloader.close()
        finally:
            write(args.output / f"{args.phase}_moment_placement.json", {"offload_moments": args.offload_moments,
                  "enrolled": offloader is not None, "transfer_events": transfer_events})
    assert stats.steps == 2, stats.summarize()
    assert all(math.isfinite(v) for v in stats.loss)
    assert all(math.isfinite(v) and v > 0 for v in stats.grad_norm_ac)
    assert all(math.isfinite(v) and v > 0 for v in stats.grad_norm_reader)
    assert stats.global_steps == ([1, 2] if first else [3, 4])
    if args.offload_moments:
        assert sum(event["action"] == "restore" for event in transfer_events) == stats.steps
        assert any(event["action"] == "park" and event["bytes"] > 0 for event in transfer_events)
    assert stats.stop_reason.startswith("duration limit") if first else stats.stop_reason == "complete"
    state = torch.load(Path(stats.checkpoint_path) / "optimizer.pt", map_location="cpu", weights_only=False)
    assert state["updates_done"] == (2 if first else 4)
    assert state["teacher_snapshot_identity"] == expected["snapshot_identity"]
    assert fingerprint(state["optimizer"]["state"]) == captures[-1]["optimizer_moments_hash"]
    assert fingerprint(state["penalty_rng_state"]) == captures[-1]["penalty_rng_hash"]
    record = {"verdict": "pass", "phase": args.phase, "captures": captures, "stats": stats.summarize(),
        "offload_moments": args.offload_moments, "transfer_events": transfer_events,
        "losses": stats.loss, "encoder_grad_norms": stats.grad_norm_ac, "reader_grad_norms": stats.grad_norm_reader,
        "penalty_counts": stats.penalty_counts, "peak_memory_bytes": stats.peak_memory_bytes,
        "source_tokens": [i.info["source_tokens"] for i in items],
        "continuation_tokens": [i.info["continuation_tokens"] for i in items],
        "pair": {"ac_checkpoint": stats.checkpoint_path, "reader_checkpoint": stats.target_lora_checkpoint_path},
        "resume_optimizer_loaded_from": stats.resumed_from}
    write(args.output / f"{args.phase}.json", record)
    return record


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    entries = select_entries(args)
    if args.phase != "all":
        return run_phase(args, entries)
    if (args.output / "first.json").exists() or (args.output / "training").exists():
        raise FileExistsError("Use a fresh gate output directory; do not accidentally train an earlier gate again")
    for phase in ["first", "resume"]:
        command = [sys.executable, str(Path(__file__).resolve()), "--phase", phase,
            "--material", str(args.material.resolve()), "--initial-pair", str(args.initial_pair.resolve()),
            "--output", str(args.output.resolve()), "--seconds", str(args.seconds),
            "--output-tolerance", str(args.output_tolerance), "--logits-chunk", str(args.logits_chunk)]
        if args.offload_moments:
            command.append("--offload-moments")
        subprocess.run(command, check=True)
    first = json.loads((args.output / "first.json").read_text())
    resumed = json.loads((args.output / "resume.json").read_text())
    assert first["offload_moments"] == resumed["offload_moments"] == args.offload_moments
    result = {"verdict": "pass", "actual_adam_updates": 4, "two_real_processes": True,
        "offload_moments": args.offload_moments,
        "transfer_events": {"first": first["transfer_events"], "resume": resumed["transfer_events"]},
        "immutable_teacher_weights_and_outputs": True, "plain_base_no_context_reference": True,
        "encoder_and_reader_finite_nonzero_gradients": True, "optimizer_moments_rng_and_target_identity_restored": True,
        "peak_memory_bytes": max(first["peak_memory_bytes"], resumed["peak_memory_bytes"]),
        "source_tokens": first["source_tokens"], "continuation_tokens": first["continuation_tokens"],
        "limit": "This verifies real reload state and target semantics. Exact uninterrupted-vs-resumed training equivalence remains the separate tiny CPU test."}
    write(args.output / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--initial-pair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["all", "first", "resume"], default="all")
    parser.add_argument("--scan-rows", type=int, default=3000)
    parser.add_argument("--seconds", type=int, default=4800)
    parser.add_argument("--logits-chunk", type=int, default=256)
    parser.add_argument("--output-tolerance", type=float, default=.002)
    parser.add_argument("--offload-moments", action="store_true", help="Privately park Adam moments between unchanged optimizer steps")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        import traceback
        args.output.mkdir(parents=True, exist_ok=True)
        write(args.output / f"{args.phase}_failure.json", {"verdict": "failed", "error": str(error), "traceback": traceback.format_exc()})
        raise
