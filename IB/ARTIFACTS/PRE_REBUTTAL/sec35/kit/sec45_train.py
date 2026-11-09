"""
Section 4.5 training driver (one process, one GPU): reconstruction (R) or task training (S from a fresh pair, T from an R checkpoint)
in one of the three write modes (sec45_common.MODES). SFT, one pass over exactly --steps x --epu items, cosine schedule over
--steps after --warmup, encoder and reader learning rates given explicitly (the design: 2e-5 / 2e-5).

  R  python sec45_train.py --kind recon --mode MODE --data FINEWEB_DIR --fresh --steps N --seconds S --ckpt-every K --name R_MODE
  S  python sec45_train.py --kind task  --mode MODE --data HOTPOT_DIR  --fresh --steps N --seconds S --ckpt-every N/4 --name S_MODE
  T  python sec45_train.py --kind task  --mode MODE --data HOTPOT_DIR  --pair R/ckpt/final/ac R/ckpt/final/reader --steps N ... --name T_MODE
  P0 the same with --steps 23 --longest-first 3 (the 24 longest items of the pool first, then random ones; no shuffle): the
     stats file holds the peak memory on the longest items and the effective seconds per update (sec45_probe_summary.py)

Same order for S and T (review B1): a loaded pair's epoch counter is reset to 0, so both start at epoch 1 with the same shuffle
seed (seed + 1) and torch seed; run.json holds the predicted order hash (first 32 items), stats.json the trained one.
No-context penalty: weight 1.0 for recon and task alike (as in all earlier reconstruction training, rq_train default); --penalty-weight overrides.
A run that does not end with stop_reason "complete" (e.g. the --seconds backstop) is marked "valid": false in stats.json and
ckpt/final/meta.json.

Outputs under $ACTIVATION_SYNC_ROOT (default ./sync):
  ckpt/step_NNNNN/{ac,reader}   every saved pair (each --ckpt-every updates and at the end), hard links WITHOUT optimizer.pt;
  ckpt/final/{ac,reader}        the last pair (the end of the run, or the stop at the --seconds limit)
  NAME.run.json                 arguments, item counts, planned updates
  NAME.held.jsonl               the held-out panel at every report (recon: the held passages; task: the dev questions)
  NAME.stats.json               per-update loss and seconds, held history, peak memory, stop reason, timing summary
The trainer's own folders (AC_MODELS/, LORAS/) hold the same files; upload ckpt/ and the json files only.

Resume (v4): every --ckpt-every updates the trainer writes ONE recovery slot (AC_MODELS/<ac>/epoch_001_step_NNNNN + LORAS/..., with
optimizer.pt = Adam moments + update count + the no-context-penalty RNG state, and AC_MODELS/<ac>/partial_epoch.json); the previous
slot is deleted only after the new one is published. With --resume-if-possible and a published slot under the sync root, the run
continues from it exactly: the same pool, the same seeded order (the slot's steps are skipped), the LR schedule at the restored
update count, the same penalty draws. The arguments must equal the first attempt's (NAME.run.json), else it refuses.
optimizer.pt lives only in the recovery slot (never in ckpt/), and is deleted when the run completes.
--keep-steps final (R): intermediate pairs are not linked into ckpt/ (only ckpt/final); all (S/T): every K-update pair is linked
into ckpt/step_NNNNN for the job's eval watcher, which deletes it after its eval.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sec45_common as S  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["recon", "task"], required=True); p.add_argument("--mode", choices=sorted(S.MODES), required=True)
    p.add_argument("--data", type=Path, required=True, help="recon: build_fineweb_recon.py's folder; task: build_hotpot_tasks.py's folder")
    start = p.add_mutually_exclusive_group(required=True)
    start.add_argument("--fresh", action="store_true", help="fresh AC modules + zero adapters on the mode's models (run_qa.harness without a pair)")
    start.add_argument("--pair", nargs=2, metavar=("AC_DIR", "READER_DIR"), help="start from this pair (T: the final R checkpoint)")
    p.add_argument("--name", required=True); p.add_argument("--steps", type=int, required=True, help="optimizer updates (items = steps x epu, one pass)")
    p.add_argument("--epu", type=int, default=8, help="examples per update"); p.add_argument("--seconds", type=float, help="time limit of the training call (the trainer saves and stops)")
    p.add_argument("--encoder-lr", type=float, required=True); p.add_argument("--reader-lr", type=float, required=True)
    p.add_argument("--warmup", type=int, default=50); p.add_argument("--schedule", default="cosine", choices=["cosine", "constant"])
    p.add_argument("--ckpt-every", type=int, help="save a pair every K updates (R: about an hour of updates; S/T: steps/4 -> 25/50/75 %%); the end is always saved")
    p.add_argument("--no-save", action="store_true", help="probe: no checkpoint at all")
    p.add_argument("--report-every", type=int, default=250, help="held-out panel every K updates (and before the first update)")
    p.add_argument("--held-panel", type=int, help="held items per report (default: all; recon 500 passages, task the 200 dev questions)")
    p.add_argument("--micro-batch", type=int, default=1, help="examples per padded reader forward")
    p.add_argument("--block-examples", type=int, default=64, help="no-context penalty references prepared per block of this many examples")
    p.add_argument("--penalty-weight", type=float, help="no-context penalty weight (default 1.0 for recon and task; applies while the reader LoRA trains)")
    p.add_argument("--longest-first", type=int, default=0, help="probe: the first K updates take the K x epu longest items of the pool, the rest random; no shuffle")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--offset", type=int, default=0, help="skip the first N items of the pool")
    p.add_argument("--keep-optimizer", action="store_true", help="keep optimizer.pt in the trainer's folders (never in ckpt/)")
    p.add_argument("--resume-if-possible", action="store_true", help="continue from the published recovery slot under the sync root, if any")
    p.add_argument("--keep-steps", choices=["all", "final"], default="all", help="all: link every K-update pair into ckpt/; final: only ckpt/final")
    a = p.parse_args()
    started = time.monotonic()
    if a.penalty_weight is None:
        a.penalty_weight = 1.0
    S.setup_environment(a.mode)
    import torch
    S.require_cuda(torch)
    from activation.ac_model import ActivationContextStudyGenerator, ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.autotuners import configure_fla_runtime
    from run_qa import AC, harness
    root = Path(os.environ.setdefault("ACTIVATION_SYNC_ROOT", str(Path("sync").resolve())))
    root.mkdir(parents=True, exist_ok=True)
    needed = a.steps * a.epu
    if a.kind == "recon":
        pool, held_rows = S.load_recon(a.data)
    else:
        import rq_common as C
        pool, held_rows = S.load_tasks(a.data, "train"), S.load_tasks(a.data, "dev200")
        random.Random(a.seed).shuffle(pool)                # the same seed -> the same questions for S and T
        corpus = C.load_corpus(a.data)
    pool = pool[a.offset:]
    if needed > len(pool):
        raise SystemExit(f"--steps {a.steps} x --epu {a.epu} = {needed} items, the pool holds {len(pool)} (rebuild the data larger)")
    if a.longest_first:                                    # probe: peak memory on the longest items (reader / encoder lengths grow with these)
        size = (lambda r: r["tokens"]) if a.kind == "recon" else (lambda r: len(r["question"]) + sum(len(corpus[i][0]) + len(corpus[i][1]) for i in r["top20"]))
        ranked = sorted(range(len(pool)), key=lambda i: -size(pool[i]))
        head = set(ranked[:a.longest_first * a.epu])
        train_rows = [pool[i] for i in ranked[:a.longest_first * a.epu]] + [r for i, r in enumerate(pool) if i not in head][:needed - len(head)]
        longest = [size(r) for r in train_rows[:len(head)]]
        print(f"sec45_train: longest-first {len(head)} items, size {min(longest)}..{max(longest)} ({'reader tokens' if a.kind == 'recon' else 'characters of question + 20 passages'})", flush=True)
    else:
        train_rows = pool[:needed]
    held_rows = held_rows[:a.held_panel] if a.held_panel else held_rows
    resume = None
    slot_record = root / "AC_MODELS" / AC / "partial_epoch.json"
    if a.resume_if_possible and slot_record.exists():
        resume = json.loads(slot_record.read_text())
        first_run = root / f"{a.name}.run.json"
        if not first_run.exists():
            raise SystemExit(f"recovery slot {slot_record} without {first_run}: refusing to resume")
        mutable = {"seconds", "resume_if_possible"}
        before = {k: v for k, v in json.loads(first_run.read_text())["args"].items() if k not in mutable}
        now = {k: str(v) for k, v in vars(a).items() if k not in mutable}
        if before != now:
            raise SystemExit(f"resume refused: arguments differ from the first attempt: {sorted(k for k in set(before) | set(now) if before.get(k) != now.get(k))}")
        if resume["epoch_number"] != 1 or not 0 < resume["step"] < a.steps:
            raise SystemExit(f"resume refused: unexpected slot {resume}")
        slot_ac, slot_reader = root / resume["ac_checkpoint"], root / resume["target_lora_checkpoint"]
        if not (slot_ac / "optimizer.pt").exists():
            raise SystemExit(f"resume refused: {slot_ac}/optimizer.pt is missing (an exact continuation needs the moments)")
        print(f"sec45_train: RESUME from the recovery slot of update {resume['step']} ({slot_ac})", flush=True)
    if resume:
        S.check_pair_mode(str(slot_ac), str(slot_reader), a.mode)
        h = harness(str(slot_ac), str(slot_reader), summary_layout=S.LAYOUT)
    elif a.pair:
        S.check_pair_mode(a.pair[0], a.pair[1], a.mode)
        h = harness(a.pair[0], a.pair[1], summary_layout=S.LAYOUT)
    else:
        h = harness(None, None, summary_layout=S.LAYOUT, row_head_linear_skip=S.linear_skip())
    study = ActivationContextStudyGenerator(h, AC)
    ac = study.ac_model
    loaded_epoch = ac.epoch_number
    ac.epoch_number = 0                                    # B1: a loaded pair (T) starts at epoch 1 like a fresh one (S): same shuffle and torch seed
    startup_seconds = time.monotonic() - started
    phase = time.monotonic()
    if a.kind == "recon":
        items, held_items = S.recon_items(study, train_rows, a.seed), S.recon_items(study, held_rows, a.seed)
    else:
        items, held_items = S.task_items(study, corpus, train_rows, AC), S.task_items(study, corpus, held_rows, AC)
    assert len(items) == needed, (len(items), needed)
    item_seconds = time.monotonic() - phase
    config = ActivationContextTrainingConfig(
        loss_kind="sft", learning_rate_ac=a.encoder_lr, learning_rate_target_lora=a.reader_lr, warmup_updates=a.warmup, lr_schedule=a.schedule,
        examples_per_update=a.epu, num_epochs=1, schedule_total_updates=a.steps, seed=a.seed, micro_batch_examples=a.micro_batch,
        checkpoint_every_updates=None if a.no_save else a.ckpt_every, checkpoint_every_epoch=not a.no_save, completion_samples=0,
        reporting_interval=min(1.0, a.report_every / a.steps), max_duration_seconds=a.seconds, teacher_cache_top_k=32, max_example_tokens=72000,
        no_context_penalty_weight=a.penalty_weight, teacher_cache_block_examples=a.block_examples, gradient_checkpointing=True, distributed=False,
        shuffle_examples=not a.longest_first, resume_optimizer=bool(resume), resume_skip_steps=resume["step"] if resume else 0)
    trainer = ActivationContextTrainer(h, config)
    start_epoch = trainer._next_epoch_number(ac)
    if start_epoch != 1 or config.cost_bucket_examples or config.part_group_examples:
        raise SystemExit(f"the run would start at epoch {start_epoch} (expected 1: a fresh sync root per run) or with a reordering option on")
    order = list(range(len(items)))
    if config.shuffle_examples:
        random.Random(config.seed + start_epoch).shuffle(order)       # the trainer's own epoch order (ac_model_training: seed + epoch_number)
    predicted = S.order_digest([items[i].item_id for i in order], a.epu)
    skip = resume["step"] if resume else 0
    predicted_now = S.order_digest([items[i].item_id for i in order[skip * a.epu:]], a.epu)    # what this attempt trains first
    phase = time.monotonic()                               # the trainer builds every example inside train(): its cost per item, measured on a sample
    sample = items[:32]
    for item in sample:
        trainer.build_example(ac, item)
    example_seconds_per_item = (time.monotonic() - phase) / max(1, len(sample))
    ckpt = root / "ckpt"
    saved_steps = []

    def keep(function):
        """Each saved pair is hard-linked to ckpt/step_NNNNN without optimizer.pt; optimizer.pt is deleted where the trainer wrote it."""
        def wrapped(ac_model, *args):
            ac_path, reader_path = function(ac_model, *args)
            step = trainer.updates_done.get(ac_model.name, 0)
            recovery = function.__name__ == "_save_partial"       # the recovery slot keeps optimizer.pt (resume); the end-of-run pair does not
            if not a.keep_optimizer and not recovery:
                for folder in (Path(ac_path), Path(reader_path)):
                    (folder / "optimizer.pt").unlink(missing_ok=True)
            link = not (recovery and a.keep_steps == "final")
            if link and S.link_pair(ac_path, reader_path, ckpt / f"step_{step:05d}"):     # False: this update is already kept (a stop right on a K boundary)
                saved_steps.append(step)
                (ckpt / f"step_{step:05d}" / "meta.json").write_text(json.dumps({"name": a.name, "kind": a.kind, "mode": a.mode, "update": step,
                    "planned_updates": a.steps, "seconds": round(time.monotonic() - started), "trainer_folders": [str(ac_path), str(reader_path)]}))
                print(f"sec45_train: pair of update {step} kept in {ckpt / f'step_{step:05d}'}", flush=True)
            return ac_path, reader_path
        return wrapped
    trainer._save_partial = keep(trainer._save_partial)
    trainer._save_epoch = keep(trainer._save_epoch)
    run = {"name": a.name, "args": {k: str(v) for k, v in vars(a).items()}, "train_items": len(items), "held_items": len(held_items), "planned_updates": a.steps,
           "models": {k: os.environ.get(k) for k in ("QA4B_READER", "QA4B_SIDE", "SEC45_ENCODER", "SEC45_LINEAR_SKIP", "QA4B_RANK", "QA4B_LAYERS", "QA4B_KV", "RQ_THINKING")},
           "ac_fields": {k: getattr(ac.config, k) for k in ("kv_transfer", "intervention_frequency", "intervention_layers", "summary_layout", "side_lora_rank", "max_view_rows")},
           "train_target_tokens_mean": round(sum(len(i.assistant_token_ids[0]) for i in items) / len(items), 1),
           "penalty_weight": a.penalty_weight, "penalty_fraction": config.no_context_penalty_fraction,
           "order": {"loaded_epoch_number": loaded_epoch, "start_epoch": start_epoch, "shuffle": config.shuffle_examples,
                     "shuffle_seed": config.seed + start_epoch, "torch_seed": config.seed + start_epoch} | predicted}
    if resume:                                             # the first attempt's run.json stays; each resume is appended here
        with open(root / f"{a.name}.resumes.jsonl", "a") as f:
            f.write(json.dumps({"from_update": skip, "slot": resume, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "job": os.environ.get("SLURM_JOB_ID"), "first_updates_now": predicted_now}, default=str) + "\n")
    else:
        (root / f"{a.name}.run.json").write_text(json.dumps(run, indent=1, default=str))
    print(f"sec45_train: {json.dumps(run, default=str)}; built in {time.monotonic() - started:.0f}s", flush=True)
    held_log = []
    held_file = root / f"{a.name}.held.jsonl"
    held_seen = {json.loads(l)["update"] for l in held_file.read_text().splitlines() if l.strip()} if held_file.exists() else set()
    prefix_tokens = {}
    if a.kind == "task":                          # answer-only numbers of the "Final answer: <answer>" target (rq_train.answer_scores)
        from rq_train import answer_scores
        prefix = study.tokenizer("Final answer:", add_special_tokens=False)["input_ids"]
        prefix_tokens = {i.item_id: len(prefix) if list(i.assistant_token_ids[0][:len(prefix)]) == prefix else 0 for i in held_items}

    resume_baseline = [bool(resume)]                       # a resume's first report is the trainer's baseline panel at the slot's update: not logged

    def report_held(progress, summary):
        if summary is None:
            return
        if resume_baseline[0]:
            resume_baseline[0] = False
            print(f"sec45_train held (resume baseline at update {progress.global_step}, not logged)", flush=True)
            return
        row = {"update": progress.global_step, "held_nll": summary.get("gold_nll"), "held_nll_token": summary.get("gold_nll_token"),
               "token_accuracy": summary.get("token_accuracy"), "items": summary.get("items"), "wall_seconds": round(time.monotonic() - started)}
        if a.kind == "task":
            row |= answer_scores(summary.get("per_item") or [], prefix_tokens)
        held_log.append(row)
        if row["update"] in held_seen:                    # a resume re-reports the panel at the slot's update: logged once only
            print(f"sec45_train held (already logged by the previous attempt): {json.dumps(row)}", flush=True)
            return
        held_seen.add(row["update"])
        with open(held_file, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"sec45_train held: {json.dumps(row)}", flush=True)
    phase = time.monotonic()
    with configure_fla_runtime(trainer._autotuning_variables(ac)["fla_config"]):
        stats = trainer.train(AC, items, reporting_data=held_items, reporting_callback=report_held)
    train_call_seconds = time.monotonic() - phase
    stopped_at = trainer.updates_done.get(AC, 0)
    if stats.stop_reason != "complete" and stopped_at < a.steps and slot_record.exists():
        # e.g. the --seconds backstop: the trainer saved a recovery slot. Not an end: no ckpt/final, no stats.json; the next job resumes.
        record = json.loads(slot_record.read_text())
        print(f"sec45_train: STOPPED ({stats.stop_reason}) at update {stopped_at} of {a.steps}; recovery slot of update {record['step']} kept: "
              f"resubmit to continue", flush=True)
        sys.exit(4)
    valid = stats.stop_reason == "complete" and (not resume or bool(getattr(stats, "resumed_from", None)))
    if resume and not getattr(stats, "resumed_from", None):
        print("sec45_train: INVALID: a resume that did not restore the optimizer state", flush=True)
    done = trainer.updates_done.get(AC, 0)
    if stats.checkpoint_path and not a.no_save and stats.steps:          # the last pair is ckpt/final (renamed from its step folder: no second copy)
        last = ckpt / f"step_{done:05d}"
        on_boundary = bool(a.ckpt_every) and done % a.ckpt_every == 0 and done < a.steps
        if last.exists() and not (ckpt / "final").exists() and not on_boundary:
            last.rename(ckpt / "final")
            saved_steps.remove(done)
        else:                                              # a stop right on a K boundary keeps step_K (the 25/50 % evals read it) AND final (hard links)
            if on_boundary:
                print(f"sec45_train: WARNING the run stopped at update {done}, a --ckpt-every boundary: step_{done:05d} and final are both kept", flush=True)
            S.link_pair(stats.checkpoint_path, stats.target_lora_checkpoint_path, ckpt / "final")
        meta = ckpt / "final" / "meta.json"
        meta.write_text(json.dumps({"name": a.name, "kind": a.kind, "mode": a.mode, "update": done, "planned_updates": a.steps,
                                    "stop_reason": stats.stop_reason, "valid": valid, "seconds": round(time.monotonic() - started)}))
    seconds = list(stats.step_seconds)
    steady = seconds[3:] if len(seconds) > 6 else seconds
    trained = S.order_digest([r["item_id"] for r in stats.example_records], a.epu)
    effective = (train_call_seconds - stats.placement_seconds - stats.baseline_seconds) / stats.steps if stats.steps else None
    final_bytes = sum(f.stat().st_size for f in (ckpt / "final").rglob("*") if f.is_file()) if (ckpt / "final").exists() else None
    timing = {"updates": stats.steps, "seconds_per_update_median": round(statistics.median(steady), 2) if steady else None,
              "effective_seconds_per_update": round(effective, 2) if effective else None,
              "startup_seconds": round(startup_seconds, 1), "item_seconds_per_item": round(item_seconds / max(1, len(items) + len(held_items)), 5),
              "example_seconds_per_item": round(example_seconds_per_item, 5), "train_call_seconds": round(train_call_seconds, 1),
              "longest_first_updates": a.longest_first, "pair_gb": round(final_bytes / 2**30, 3) if final_bytes else None,
              "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else None,
              "seconds_per_update_mean": round(statistics.fmean(steady), 2) if steady else None, "first_updates_seconds": [round(x, 1) for x in seconds[:3]],
              "baseline_panel_seconds": round(stats.baseline_seconds, 1), "placement_seconds": round(stats.placement_seconds, 1),
              "reporting_seconds": round(sum(e.reporting_seconds for e in stats.epochs), 1), "checkpoint_seconds": round(sum(e.checkpoint_seconds for e in stats.epochs), 1),
              "peak_gb": round(stats.peak_memory_bytes / 2**30, 1) if stats.peak_memory_bytes else None, "wall_seconds": round(time.monotonic() - started),
              "held_items": len(held_items), "micro_batch": a.micro_batch, "epu": a.epu,
              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    summary = {"name": a.name, "kind": a.kind, "mode": a.mode, "timing": timing, "held": held_log, "loss": stats.loss, "step_seconds": seconds,
               "grad_norm_ac": stats.grad_norm_ac, "grad_norm_reader": stats.grad_norm_reader, "penalty": stats.penalty, "learning_rates": stats.learning_rates,
               "stop_reason": stats.stop_reason, "valid": valid, "saved_updates": sorted(saved_steps), "final": str(ckpt / "final") if (ckpt / "final").exists() else None,
               "order_trained": trained, "order_predicted": predicted_now, "order_matches": bool(trained["first_updates"]) and trained["first_updates"] == predicted_now["first_updates"][:len(trained["first_updates"])],
               "order_full": predicted, "resumed_from_update": skip, "resumed_optimizer": getattr(stats, "resumed_from", None)}
    (root / f"{a.name}.stats.json").write_text(json.dumps(summary, indent=1, default=str))
    if not valid:
        print(f"sec45_train: INVALID run: stop_reason '{stats.stop_reason}' (planned {a.steps} updates, done {done})", flush=True)
    print(f"sec45_train: done ({stats.stop_reason}) in {time.monotonic() - started:.0f}s; {json.dumps(timing)}", flush=True)


if __name__ == "__main__":
    main()
