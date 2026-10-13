"""Two ordinary AC training recipes over a frozen selection of portable trajectory data."""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import pickle
from pathlib import Path
import random
import shutil
import signal
import time

from run_qa import AC, NAME, READER, harness, answer_positions, resume_checkpoint, write

STOP = False
PREPARATION_CONTEXT = None
CAPACITY_CONTEXT = None


def request_stop(signum, frame):
    global STOP
    STOP = True


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cohort_metrics(summary, items, task):
    groups = {}
    for row, item in zip(summary["per_item"], items):
        length = item.info["source_tokens"]
        size = "≤8k" if length <= 8192 else "8k–16k" if length <= 16384 else "16k–32k"
        tags = ["all", "source: " + item.info["source"], "history: " + size]
        for tag in tags:
            groups.setdefault(tag, []).append((row, item))
    result = {}
    for tag, entries in groups.items():
        positions = sum(row["positions"] for row, _ in entries)
        if task == "compaction":
            nll = sum(row["gold_nll"] * row["positions"] for row, _ in entries) / positions
        else:
            nll = sum(-sum(row["target_logp"][i] for i in item.info["answer_positions"]) / len(item.info["answer_positions"])
                      for row, item in entries) / len(entries)
        result[tag] = {"examples": len(entries), "NLL": nll,
                      "KL": sum(row["kl"] * row["positions"] for row, _ in entries) / positions
                            if all(row.get("kl") is not None for row, _ in entries) else None}
    return result


def material_rows(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    for shard in manifest["shards"]:
        with (folder / shard["path"]).open() as handle:
            for ordinal, line in enumerate(handle):
                yield shard["path"], ordinal, json.loads(line)


def choose_material(args, selection_path):
    """Freeze exact row/window/question IDs once; later bank additions do not change a run."""
    supplied = getattr(args, "selection_file", None)
    if supplied is not None:
        selection = json.loads(Path(supplied).read_text())
        if (args.task != "trajectory_qa" or selection.get("task") != args.task or args.target_tokens
                or len(selection["train"]) != args.items or len(selection["held"]) != args.held
                or selection["seed"] != args.seed
                or Path(selection["material"]).resolve() != args.material.resolve()):
            raise ValueError("Supplied frozen QA selection does not match the requested run")
        if {entry["root_id"] for entry in selection["train"]} & {entry["root_id"] for entry in selection["held"]}:
            raise ValueError("Supplied training roots overlap the fixed held panel")
        if selection_path.exists():
            if json.loads(selection_path.read_text()) != selection:
                raise ValueError("Run already has a different frozen selection")
        else:
            write(selection_path, selection)
        return selection
    request={"task":args.task,"material":str(args.material.resolve()),
             "questions":str(args.questions.resolve()) if args.questions else None,"items":args.items,
             "target_tokens":args.target_tokens,"held":args.held,"seed":args.seed}
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection.get("request") != request:
            raise ValueError("Existing material selection belongs to a different request")
        return selection
    questions = {}
    if args.task == "trajectory_qa":
        bank = json.loads((args.questions / "manifest.json").read_text())
        for shard in bank["shards"]:
            for line in (args.questions / shard["path"]).read_text().splitlines():
                q = json.loads(line)
                questions.setdefault(q["trajectory_id"], []).append(q)
    training, held, pending_held, counts = [], [], [], Counter()
    assistant_tokens = 0
    for shard, ordinal, row in material_rows(args.material):
        choices = row["compaction_cuts"] if args.task == "compaction" else questions.get(row["id"], [])
        if not choices:
            continue
        ordered = list(choices)
        random.Random(args.seed ^ int(digest(row["id"])[:12], 16)).shuffle(ordered)
        # One cut per recorded trajectory first; multiple later questions share a fixed QA window.
        if args.task == "compaction":
            ordered = ordered[:1]
        for choice in ordered:
            entry = {"shard": shard, "ordinal": ordinal, "trajectory_id": row["id"],
                     "root_id": row["root_id"], "source": row["source"], "choice": choice,
                     "row_sha256": digest(row)}
            if args.task == "trajectory_qa" and choice.get("material_row_sha256") != hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest():
                raise ValueError("Question bank and history payload disagree")
            if row["split"] == "held":
                if len(pending_held) < args.held * 8:
                    pending_held.append(entry)
                continue
            if len(training) >= args.items or (args.target_tokens and assistant_tokens >= args.target_tokens):
                continue
            training.append(entry)
            assistant_tokens += int(choice.get("assistant_target_tokens", 0))
            counts[row["source"]] += 1
        enough = len(training) >= args.items or (args.target_tokens and assistant_tokens >= args.target_tokens)
        if enough and len(pending_held) >= args.held * 3:
            break
    # Spread the fixed panel over sources and requested lengths without throwing away scarce groups.
    groups = {}
    for entry in pending_held:
        choice = entry["choice"]
        key = (entry["source"], choice.get("target_tokens", choice.get("window_target_tokens", 0)))
        groups.setdefault(key, []).append(entry)
    while groups and len(held) < args.held:
        for key in list(groups):
            held.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
            if len(held) == args.held:
                break
    if not training or not held:
        raise ValueError("Need published training and root-held material before launching")
    enough = assistant_tokens >= args.target_tokens if args.target_tokens else len(training) >= args.items
    if not enough or len(held) < args.held:
        raise ValueError(f"Material is not ready: {len(training)} training examples, {assistant_tokens} assistant tokens, {len(held)} held examples")
    if set(e["root_id"] for e in training) & set(e["root_id"] for e in held):
        raise ValueError("Root histories overlap between training and held material")
    selection = {"task": args.task, "material": str(args.material.resolve()), "train": training, "held": held,
                 "sources": dict(counts), "assistant_target_tokens": assistant_tokens,
                 "training_roots": len({e["root_id"] for e in training}), "seed": args.seed,"request":request}
    write(selection_path, selection)
    return selection


def resolve_qa_window(row, choice, tokenizer):
    """Old QA rows use material windows; new rows may carry a separately chosen prefix."""
    explicit = choice.get("window")
    if explicit is None:
        return next(w for w in row["qa_windows"] if w["id"] == choice["window_id"])
    window = dict(explicit)
    first, messages = row["initial_end"], row["messages"]
    if any(type(window.get(key)) is not int for key in ("start", "end", "source_tokens", "target_tokens")):
        raise ValueError("Invalid portable QA window bounds or token budget")
    end = window["end"]
    if (window["start"] != first or not first < end <= len(messages)
            or (end < len(messages) and messages[end].get("role") != "assistant")
            or window["id"] != f"{row['id']}:qa:{first}:{end}"
            or window["id"] != choice["window_id"]
            or window["source_tokens"] != choice["source_tokens"]
            or window["target_tokens"] != choice["window_target_tokens"]):
        raise ValueError("Portable QA window changed or breaks a complete original-history prefix")
    from activation.ac_model.ac_model_utils import tokenize_with_parts
    frame = [{"role": "system", "content": row["system_prompt"]}] + messages[:first]
    ids, _ = tokenize_with_parts(tokenizer, frame + messages[first:end], [], tokenizer.pad_token_id or 0,
                                tools=row["tools"], chat_template_kwargs={"enable_thinking": True})
    if len(ids) != window["source_tokens"] or not 1024 <= len(ids) <= window["target_tokens"]:
        raise ValueError("Portable QA window disagrees with the native encoder-family token budget")
    return window


def build_items(study, folder, entries, task, ratio, progress=None, *, encoder_sees_question=False):
    from activation.ac_model import ActivationContextTrainingItem
    from activation.ac_model.ac_model_utils import assistant_target_ids, prepare_training_history
    from activation.common.ac_parts import ac_part, COMPACTION_INSTRUCTIONS
    from activation.dataset.dataset import DatasetDocument, DataModality
    needed = {}
    for entry in entries:
        needed.setdefault(entry["shard"], set()).add(entry["ordinal"])
    rows = {}
    for shard_number, (shard, ordinals) in enumerate(needed.items(), 1):
        with (folder / shard).open() as handle:
            for ordinal, line in enumerate(handle):
                if ordinal in ordinals:
                    row = json.loads(line)
                    rows[row["id"]] = row
        if progress:
            progress("reading selected histories", shard_number, len(needed))
    items, qa_windows = [], {}
    for entry in entries:
        row, choice = rows[entry["trajectory_id"]], entry["choice"]
        if digest(row) != entry["row_sha256"]:
            raise ValueError("A selected trajectory payload changed")
        if task == "trajectory_qa":
            key = (row["id"], choice["window_id"], digest(choice.get("window")),
                   choice.get("source_tokens"), choice.get("window_target_tokens"))
            if key not in qa_windows:
                qa_windows[key] = resolve_qa_window(row, choice, study.tokenizer)
            window = qa_windows[key]
            messages = row["messages"][:row["initial_end"]] + row["messages"][window["start"]:window["end"]]
            document = DatasetDocument(row["id"], row["source"], trajectory=row["messages"],
                trajectory_kwargs={"system_prompt": row["system_prompt"], "tools": row["tools"]}, modality=DataModality.TRAJECTORY)
            item = study.trajectory_qa_item(document, messages, choice["question"], choice["answer"],
                item_id=choice["id"], compression_ratio=ratio, encoder_sees_question=encoder_sees_question)
            item.info.update(answer_positions=answer_positions(study.tokenizer, item.assistant_token_ids[0], item.info["answer"]),
                             window_id=window["id"], source_tokens=window["source_tokens"],
                             chat_template_kwargs={"enable_thinking": False}, evidence_indices=choice["evidence_indices"])
        else:
            messages = row.get("compaction_messages", row["messages"])
            tools = row.get("compaction_tools", row["tools"])
            first = row.get("compaction_initial_end", row["initial_end"])
            initial = deepcopy(messages[:first])
            system = [{"role": "system", "content": row["system_prompt"]}] if row["system_prompt"] else []
            prefix = system + initial + deepcopy(messages[choice["start"]:choice["end"]])
            continuation = deepcopy(messages[choice["end"]:choice["continuation_end"]])
            kwargs = {"enable_thinking": True}
            targets = [assistant_target_ids(study.tokenizer, message, tools, kwargs)
                       for message in continuation if message["role"] == "assistant"]
            part = ac_part(deepcopy(prefix), study.ac_name, ratio, kind="compaction")
            if tools:
                part["tools"] = deepcopy(tools)
            student = system + initial + [{"role": "user", "content": [part,
                {"type": "text", "text": "\n\n" + COMPACTION_INSTRUCTIONS}]}]
            ids, _, positions = prepare_training_history(study.tokenizer, prefix + continuation, len(prefix), targets, [],
                                                        tools=tools, chat_template_kwargs=kwargs)
            if len(ids) > 72000 or len(ids) - positions[0] - 1 > 8192:
                raise ValueError("Frozen compaction cut exceeds actual rendered budgets")
            if sum(map(len, targets)) != choice["assistant_target_tokens"]:
                raise ValueError("Compaction assistant targets changed after material preparation")
            item = ActivationContextTrainingItem(item_id=choice["window_id"], kind="compaction",
                in_context_messages=prefix + continuation, ac_messages=student + deepcopy(continuation),
                in_context_start=len(prefix), ac_start=len(student), assistant_token_ids=targets,
                tools=tools, dataset_id=row["source"], doc_ids=[row["id"]],
                info={"chat_template_kwargs": kwargs, "source_tokens": choice["source_tokens"],
                      "continuation_tokens": choice["continuation_tokens"], "assistant_turns": len(targets),
                      "synthetic_continuation_turns": row.get("compaction_synthetic_continuation_turns", 0), "ratio": ratio})
        item.info.update(root_id=row["root_id"], source=row["source"])
        items.append(item)
        if progress and (len(items) % 64 == 0 or len(items) == len(entries)):
            progress("constructing training items", len(items), len(entries))
    return items


def compact_example(example):
    # Token vectors remain ordinary integer sequences, without one Python object per token.
    # Keep the cache key computed from the original lists and all item/request identities.
    for name in ("teacher_ids", "student_ids", "teacher_positions", "student_positions", "target_ids"):
        setattr(example, name, array("i", getattr(example, name)))
    if example.penalty is not None:
        compact_example(example.penalty)
    return example


def prepare_chunk(entries):
    study, folder, task, ratio, builder, ac, encoder_sees_question = PREPARATION_CONTEXT
    items = build_items(study, folder, entries, task, ratio, encoder_sees_question=encoder_sees_question)
    return items, [compact_example(builder.build_example(ac, item)) for item in items]


def check_capacity_chunk(examples):
    from activation.ac_model.ac_model_utils import validate_part_capacity
    for example in examples:
        if max(len(example.teacher_ids), len(example.student_ids)) > 72000:
            raise ValueError(f"{example.item.item_id}: prepared reader history exceeds 72000 tokens")
        for request in example.part_requests:
            validate_part_capacity(CAPACITY_CONTEXT, request, 65536)
    return len(examples)


def main(args):
    import torch
    from activation.ac_model import ActivationContextTrainingConfig, ActivationContextTrainer
    from activation.ac_model.ac_model_study import ActivationContextStudyGenerator
    from activation.common.reporting import HtmlReporter
    from activation.ac_model.ac_model_reporter import ActivationContextTrainingReporter
    class CampaignTrainer(ActivationContextTrainer):
        build_progress = None
        prepared_examples = None
        prepared_capacity = None
        def build_example(self, ac_model, item, reference=False):
            if not reference and self.prepared_examples is not None and id(item) in self.prepared_examples:
                return self.prepared_examples[id(item)]
            example = super().build_example(ac_model, item, reference)
            if not reference and self.build_progress is not None:
                self.build_progress[0] += 1
                done, total = self.build_progress
                if done % 64 == 0 or done == total:
                    preparation_progress("tokenizing teacher and student histories", done, total)
                if done == total:
                    self.build_progress = None
            return example
        def _check_capacity(self, ac_model, examples):
            examples = list(examples)
            if self.prepared_capacity and all(self.prepared_examples.get(id(e.item)) is e for e in examples):
                limits = []
                for model in (ac_model.target.model, ac_model.side.model):
                    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
                    limits.append(getattr(config, "max_position_embeddings", None))
                if all(actual and actual >= checked for actual, checked in
                       zip(limits, (self.prepared_capacity["reader_tokens"], self.prepared_capacity["encoder_tokens"]))):
                    return
            if len(examples) < 512:
                return super()._check_capacity(ac_model, examples)
            preparation_progress("checking context capacities", 0, len(examples))
            for start in range(0, len(examples), 128):
                super()._check_capacity(ac_model, examples[start:start + 128])
                preparation_progress("checking context capacities", min(start + 128, len(examples)), len(examples))
        def _check_training_deadline(self):
            if STOP:self._training_deadline=0.0
            super()._check_training_deadline()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["ACTIVATION_SYNC_ROOT"] = str(args.output.resolve())
    report = args.output / "report"
    result_path = report / "result.json"
    if result_path.exists() and json.loads(result_path.read_text()).get("stop_reason") == "complete":
        print("Run already complete", flush=True)
        return 0
    r = HtmlReporter(str(report), "Trajectory QA" if args.task == "trajectory_qa" else "Compaction continuation",
        "2B encoder → 9B reader · original history only in the encoder · frozen initial-reader controls",
        page_rewrite_seconds=math.inf)
    r.set_status(phase="selecting published material", task=args.task)
    r.set_text("recipe", "Training recipe", "Preparing the frozen material selection and paired checkpoint.")
    r.initialize_line_plot("quality", "Held-out quality", "Fixed source-root held-out panel; references stay frozen.",
                           "examples presented", "nats per target token", [])
    r.initialize_line_plot("learning", "Learning loss", "A fixed training sample; answer tokens only for QA, all retained assistant targets for compaction.",
                           "examples presented", "nats per target token", [])
    if args.task == "compaction":
        r.initialize_line_plot("divergence", "Held-out reconstruction of the teacher distribution",
                               "Top-32 + remainder KL against the immutable initial reader with full history; a lower bound on full-vocabulary KL.",
                               "examples presented", "nats per target token", [])
        r.initialize_line_plot("turns", "Immediate and later continuation",
                               "First assistant turn versus later teacher-forced turns; token-weighted gold NLL.",
                               "examples presented", "nats per target token", [])
        r.initialize_line_plot("immediate_kl", "Information loss immediately after the cut",
                               "First up to 128 tokens of the first assistant turn, versus remaining supervised tokens. Later tokens receive the correct continuation as context.",
                               "examples presented", "nats per target token", [])
    r.initialize_table("panels", "Completed quality panels", "Full/no-context baselines use the initial reader.",
                       ["examples presented", "supervised tokens presented", "epoch", "held NLL", "learning NLL", "held KL", "teacher NLL", "seconds"])
    r.initialize_table("cohorts", "Latest panel by source and history length",
                       "The same fixed held-out items, grouped by actual input length. Source and length rows overlap.",
                       ["cohort", "examples", "AC NLL", "frozen full NLL", "frozen no-context NLL", "AC KL"])
    r.set_trajectories("examples","Decoded examples","A fixed small sample at baseline, halfway and completion. Compaction shows the immediate next assistant turn; generation can reach its stated token cap.",[])
    r.set_text("pending", "Measurement status", "Quality measurements are pending model placement and baseline evaluation.")
    r.render(force=True)
    selection = choose_material(args, args.output / "selection.json")
    steps = math.ceil(len(selection["train"]) / args.batch)
    def presentations(global_step):
        epoch,step=divmod(global_step,steps)
        return min(epoch*len(selection["train"])+min(step*args.batch,len(selection["train"])),len(selection["train"])*args.epochs)
    execution_policy = None
    if getattr(args, "execution_policy", None):
        if args.task != "trajectory_qa" or args.offload_moments:
            raise ValueError("Execution policy requires SFT QA and owns moment placement; omit --offload-moments")
        measured = json.loads(args.execution_policy.read_text())
        execution_policy = measured["policy"]
        if not measured.get("receipt", {}).get("validation", {}).get("passed"):
            raise ValueError("Use a completed, validated execution-tuner result")
        write(args.output / "execution_measurement.json", measured)
    config = ActivationContextTrainingConfig(loss_kind="kl" if args.task == "compaction" else "sft",
        learning_rate_ac=args.encoder_lr, learning_rate_target_lora=args.reader_lr, examples_per_update=args.batch,
        micro_batch_examples=1 if args.task == "compaction" or execution_policy is not None else args.micro_batch,
        execution_policy=execution_policy, num_epochs=args.epochs,
        warmup_updates=0 if args.smoke else 50, schedule_total_updates=steps * args.epochs, seed=args.seed,
        teacher_cache_block_examples=args.teacher_block, no_context_penalty_weight=1, no_context_penalty_fraction=.25,
        reporting_interval=.1, completion_samples=0, completion_max_new_tokens=1024 if args.task=="compaction" else 256, logits_chunk_tokens=args.logits_chunk,
        gradient_checkpointing_min_tokens=args.checkpoint_tokens, checkpoint_every_updates=min(50,max(1,steps//10)),
        max_duration_seconds=args.seconds, shuffle_examples=True)
    identity = {"selection": digest(selection), "initial_pair": digest(json.loads(args.initial_pair.read_text())), "ratio": args.ratio,
                "learning_examples":args.learning,"qualitative_examples":args.qualitative,"encode_tokens":args.encode_tokens,
                "offload_moments":args.offload_moments,
                "config": {k:v for k,v in asdict(config).items() if k not in ("max_duration_seconds", "resume_optimizer", "resume_skip_steps")}}
    if args.task == "trajectory_qa" and args.encoder_sees_question:
        identity["encoder_sees_question"] = True
    identity=json.loads(json.dumps(identity))
    identity_path = args.output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Frozen material, initialization or training recipe changed")
    pair = json.loads(args.initial_pair.read_text())
    resume = None
    markers = args.output / "AC_MODELS" / AC
    if (markers / "partial_epoch.json").exists() or (markers / "completed_epoch.json").exists():
        resume = resume_checkpoint(args.output, steps, args.epochs)
        if not resume["remaining_epochs"]:
            print("Run already complete", flush=True)
            return 0
        pair = {"ac_checkpoint": resume["ac_path"], "reader_checkpoint": resume["reader_path"]}
        config = replace(config, resume_optimizer=True, resume_skip_steps=resume["skip_steps"], num_epochs=resume["remaining_epochs"])
        attempt = args.output / "attempts" / f"resume_{time.time_ns()}"
        attempt.mkdir(parents=True)
        for name in ("panels.json", "report_data.json", "result.json", "selected_checkpoint.json"):
            if (report / name).exists():shutil.copy2(report / name, attempt / name)
        if (report / "panels.json").exists():
            prior = json.loads((report / "panels.json").read_text())
            write(report / "panels.json", [p for p in prior if p["examples presented"] <= presentations(resume["updates"])])
        for path in report.glob("panel_*.json"):
            if int(path.stem.split("_")[-1]) > presentations(resume["updates"]):
                path.rename(attempt / path.name)
        for path in report.glob("qualitative_*.json"):
            value=json.loads(path.read_text())
            if value.get("examples_presented",0)>presentations(resume["updates"]):path.rename(attempt/path.name)
        selected=report/"selected_checkpoint.json"
        if selected.exists():
            value=json.loads(selected.read_text())
            if value["examples_presented"]>presentations(resume["updates"]):
                value["retained_from_prior_attempt"]=str(attempt.relative_to(args.output));write(selected,value)
        for parent in (args.output / "AC_MODELS" / AC, args.output / "LORAS" / READER):
            for path in parent.glob("epoch_*"):
                if path.name[6:].isdigit() and int(path.name[6:]) >= resume["first_epoch"]:
                    path.rename(path.with_name(f"incomplete_{path.name}_{attempt.name}"))
    elif (report / "panels.json").exists() and any(p["examples presented"] for p in json.loads((report / "panels.json").read_text())):
        raise ValueError("Interrupted run has trained panels but no published paired checkpoint")
    write(identity_path, identity)
    write(args.output / "training_config.json", asdict(config))
    r.set_status(phase="loading paired models");r.render(force=True)
    h = harness(pair["ac_checkpoint"], pair["reader_checkpoint"])
    ac = h.module_manager.get_ac_model(AC)
    ac.config.encode_batch_max_tokens = args.encode_tokens
    if not resume:ac.epoch_number = 0
    study = ActivationContextStudyGenerator(h, AC)
    def tokenizer_identity(tokenizer):
        return {"vocabulary": digest(tokenizer.get_vocab()), "chat_template": digest(tokenizer.chat_template),
                "special_tokens": digest(tokenizer.special_tokens_map)}
    tokenizer_record = tokenizer_identity(study.tokenizer)
    tokenizer_path = args.output / "tokenizer_identity.json"
    if tokenizer_path.exists() and json.loads(tokenizer_path.read_text()) != tokenizer_record:
        raise ValueError("Training tokenizer or chat template changed")
    write(tokenizer_path, tokenizer_record)
    def preparation_progress(phase, completed, total):
        phase_changed = r.status.get("phase") != phase
        r.status.clear()
        r.set_status(phase=phase, completed=f"{completed:,}/{total:,}",
                     elapsed_minutes=f"{(time.monotonic()-started)/60:.1f}")
        r.render(force=phase_changed)
    preparation_progress("reading selected histories", 0, len(selection["train"]))
    build_started = time.monotonic()
    import activation
    source_root = Path(activation.__file__).parent
    preparation_identity = {"run":identity,"reader_tokenizer":tokenizer_record,
        "encoder_tokenizer":tokenizer_identity(ac.side.tokenizer),
        "ac_config":{k:v for k,v in asdict(ac.config).items() if k!="checkpoint_path"},
        "source":digest({str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(source_root.rglob("*.py"))}),
        "runner":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    preparation_identity=json.loads(json.dumps(preparation_identity))
    prepared_path=args.output/"prepared_examples.pkl"
    prepared_identity_path=args.output/"prepared_identity.json"
    prepared=None
    if prepared_path.exists() and prepared_identity_path.exists():
        if json.loads(prepared_identity_path.read_text())!=preparation_identity:
            raise ValueError("Prepared examples belong to different material, tokenizers or source")
        preparation_progress("loading prepared histories",0,len(selection["train"])+len(selection["held"]))
        # This is a private, locally generated cache, never an imported dataset artifact.
        with prepared_path.open("rb") as handle:prepared=pickle.load(handle)
        if prepared["identity"]!=preparation_identity:
            raise ValueError("Prepared example cache identity disagrees with its manifest")
        train,held=prepared["train"],prepared["held"]
    else:
        if args.require_prepared:
            raise ValueError("CPU preparation must finish before this GPU run starts")
        if args.prepare_workers > 1:
            if not args.prepare_only or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
                raise ValueError("Parallel preparation requires a CPU-only allocation with CUDA_VISIBLE_DEVICES='' before imports")
            torch.set_num_threads(1)
            global PREPARATION_CONTEXT
            PREPARATION_CONTEXT=(study,args.material,args.task,args.ratio,ActivationContextTrainer(h,config),ac,args.encoder_sees_question)
            entries=[*selection["train"],*selection["held"]]
            chunks=iter(entries[i:i+64] for i in range(0,len(entries),64))
            items,examples=[],[]
            with ProcessPoolExecutor(max_workers=args.prepare_workers,mp_context=multiprocessing.get_context("fork")) as pool:
                pending=deque(pool.submit(prepare_chunk,chunk) for chunk in [next(chunks,None) for _ in range(args.prepare_workers)] if chunk)
                while pending:
                    chunk_items,chunk_examples=pending.popleft().result()
                    items.extend(chunk_items);examples.extend(chunk_examples)
                    preparation_progress("preparing teacher and student histories on CPU",len(items),len(entries))
                    chunk=next(chunks,None)
                    if chunk:pending.append(pool.submit(prepare_chunk,chunk))
            PREPARATION_CONTEXT=None
            train,held=items[:len(selection["train"])],items[len(selection["train"]):]
            prepared={"identity":preparation_identity,"train":train,"held":held,"examples":examples}
        else:
            train = build_items(study,args.material,selection["train"],args.task,args.ratio,preparation_progress,encoder_sees_question=args.encoder_sees_question)
            held = build_items(study,args.material,selection["held"],args.task,args.ratio,preparation_progress,encoder_sees_question=args.encoder_sees_question)
            preparation_progress("tokenizing teacher and student histories",0,len(train)+len(held))
            builder=CampaignTrainer(h,config)
            builder.build_progress=[0,len(train)+len(held)]
            prepared={"identity":preparation_identity,"train":train,"held":held,
                      "examples":[compact_example(builder.build_example(ac,item)) for item in [*train,*held]]}
        preparation_progress("saving prepared histories",len(train)+len(held),len(train)+len(held))
        temporary=prepared_path.with_name(prepared_path.name+".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(prepared,handle,protocol=5)
            handle.flush();os.fsync(handle.fileno())
        temporary.replace(prepared_path)
        write(prepared_identity_path,preparation_identity)
    write(report / "preparation_timing.json", {"preparation_or_load_seconds":time.monotonic()-build_started,
        "training_items":len(train),"held_items":len(held),"cache_bytes":prepared_path.stat().st_size})
    capacity_path = args.output / "prepared_capacity.json"
    capacity_record = json.loads(capacity_path.read_text()) if capacity_path.exists() else None
    if capacity_record and (capacity_record["prepared_identity"] != preparation_identity or
                            capacity_record["checked_examples"] != len(prepared["examples"])):
        raise ValueError("Prepared capacity check belongs to different examples or source")
    if args.prepare_only:
        if capacity_record is None:
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
                raise ValueError("Capacity preparation requires a CPU-only allocation")
            torch.set_num_threads(1)
            global CAPACITY_CONTEXT
            CAPACITY_CONTEXT = ac
            capacity_started = time.monotonic()
            examples = prepared["examples"]
            chunks = iter(examples[i:i+64] for i in range(0, len(examples), 64))
            checked = 0
            with ProcessPoolExecutor(max_workers=args.prepare_workers, mp_context=multiprocessing.get_context("fork")) as pool:
                pending = deque(pool.submit(check_capacity_chunk, chunk)
                                for chunk in [next(chunks, None) for _ in range(args.prepare_workers)] if chunk)
                while pending:
                    checked += pending.popleft().result()
                    preparation_progress("checking context capacities on CPU", checked, len(examples))
                    chunk = next(chunks, None)
                    if chunk:
                        pending.append(pool.submit(check_capacity_chunk, chunk))
            CAPACITY_CONTEXT = None
            capacity_record = {"prepared_identity": preparation_identity, "checked_examples": checked,
                               "reader_tokens": 72000, "encoder_tokens": 65536,
                               "seconds": time.monotonic() - capacity_started}
            write(capacity_path, capacity_record)
        r.set_status(phase="CPU preparation complete; waiting for training",training_examples=f"{len(train):,}")
        r.render(force=True)
        return 0
    learning = train[:min(args.learning,len(train))]
    token_prefixes=[]
    for epoch in range(1,args.epochs+1):
        order=list(range(len(train)));random.Random(args.seed+epoch).shuffle(order)
        prefix=[0]
        for index in order:prefix.append(prefix[-1]+sum(map(len,train[index].assistant_token_ids)))
        token_prefixes.append(prefix)
    def counts(global_step):
        epochs_done,step=divmod(global_step,steps)
        examples=presentations(global_step)
        tokens=sum(p[-1] for p in token_prefixes[:epochs_done])
        if epochs_done<len(token_prefixes):tokens+=token_prefixes[epochs_done][min(step*args.batch,len(train))]
        return examples,tokens
    trainer = CampaignTrainer(h, config)
    trainer.prepared_examples={id(example.item):example for example in prepared["examples"]}
    trainer.prepared_capacity=capacity_record
    if resume:
        ac.epoch_number = resume["saved_epoch"]
        # The trainer uses existing epoch directories only to allocate a new publication name.
        trainer._next_epoch_number = lambda model: resume["first_epoch"]
    recipe = {"task":args.task,"training_examples":len(train),"distinct_training_roots":selection["training_roots"],
              "epochs":args.epochs,"supervised_assistant_tokens":sum(sum(map(len,x.assistant_token_ids)) for x in train),
              "held_examples":len(held),"ratio":args.ratio,"source_mix":selection["sources"],"config":asdict(config),
              "optimizer_moments":"CPU between optimizer steps" if args.offload_moments or
                  (execution_policy is not None and execution_policy["optimizer_moments"] == "cpu") else "GPU during training"}
    def distribution(values):
        values=sorted(values)
        return {"min":values[0],"median":values[len(values)//2],"p90":values[min(len(values)-1,int(.9*len(values)))],"max":values[-1]}
    recipe["source_tokens"]=distribution([x.info["source_tokens"] for x in train])
    if args.task=="compaction":recipe["continuation_tokens"]=distribution([x.info["continuation_tokens"] for x in train])
    write(report / "recipe.json",recipe);r.set_text("recipe","Training recipe",json.dumps(recipe,indent=2))
    panels_path=report / "panels.json"
    panels=json.loads(panels_path.read_text()) if panels_path.exists() else []
    fixed_path=report / "fixed_references.json"
    sft=CampaignTrainer(h,replace(config,loss_kind="sft",no_context_penalty_weight=0,micro_batch_examples=1))
    offloader = None
    transfer_events = []
    def nll(summary,items):
        if args.task=="compaction":return summary["gold_nll_token"]
        return sum(-sum(row["target_logp"][i] for i in item.info["answer_positions"])/len(item.info["answer_positions"])
                   for row,item in zip(summary["per_item"],items))/len(items)
    def turn_nll(summary,items):
        values={"first":[],"later":[]}
        for row,item in zip(summary["per_item"],items):
            cut=len(item.assistant_token_ids[0])
            values["first"].extend(row["target_logp"][:cut]);values["later"].extend(row["target_logp"][cut:])
        return {key:-sum(tokens)/len(tokens) if tokens else None for key,tokens in values.items()}
    def immediate_kl(summary,items):
        values={"first 128":[],"remaining":[]}
        for row,item in zip(summary["per_item"],items):
            cut=min(128,len(item.assistant_token_ids[0]))
            values["first 128"].extend(row["target_kl"][:cut]);values["remaining"].extend(row["target_kl"][cut:])
        return {key:sum(tokens)/len(tokens) if tokens else None for key,tokens in values.items()}
    def panel(progress,summary):
        nonlocal offloader
        if args.offload_moments and progress.epoch == 0:
            from probe_adam_moment_offload_v1 import MomentOffload
            assert offloader is None
            offloader = MomentOffload(trainer.optimizers[AC], transfer_events)
            offloader.park()
        now=time.monotonic();presented,target_tokens=counts(progress.global_step)
        r.status.clear();r.set_status(phase="evaluating fixed panels",examples_presented=str(presented),
            dataset=f"{len(train):,} {'questions' if args.task=='trajectory_qa' else 'compaction examples'}",epochs=f"{presented/len(train):.3f}")
        r.render(force=True)
        if fixed_path.exists():
            fixed=json.loads(fixed_path.read_text())
            if fixed["selection"]!=identity["selection"]:raise ValueError("Frozen reference selection changed")
        else:
            if resume:raise ValueError("Resumed reader cannot replace missing frozen initial references")
            fixed={"selection":identity["selection"],"initial_pair":identity["initial_pair"],"views":{},"turns":{},"kl":{},"immediate_kl":{}}
            names=["full_context","no_context"]+(["recent_text"] if args.task=="compaction" else [])
            for name in names:
                views=study.transform_for_eval_reference(held,name)
                view_summary=(trainer.eval(AC,views,release=False,teacher="stored") if args.task=="compaction"
                              else sft.eval(AC,views,release=False))
                fixed["views"][name]=nll(view_summary,views)
                if args.task == "compaction":
                    fixed["turns"][name]=turn_nll(view_summary,views)
                    fixed["kl"][name]=sum(row["kl"]*row["positions"] for row in view_summary["per_item"])/sum(row["positions"] for row in view_summary["per_item"])
                    fixed["immediate_kl"][name]=immediate_kl(view_summary,views)
                write(report/f"frozen_{name}.json",view_summary)
            write(fixed_path,fixed)
        learning_summary=trainer.eval(AC,learning,release=False,teacher="stored")
        held_nll=nll(summary,held);learning_nll=nll(learning_summary,learning)
        if args.task == "compaction":
            rows=summary["per_item"];total=sum(row["positions"] for row in rows)
            held_kl=sum(row["kl"]*row["positions"] for row in rows)/total
            learning_kl=sum(row["kl"]*row["positions"] for row in learning_summary["per_item"])/sum(row["positions"] for row in learning_summary["per_item"])
            teacher_nll=-sum(sum(row["teacher_target_logp"]) for row in rows)/total
            if presented==0 and abs(teacher_nll-fixed["views"]["full_context"])>.001:
                raise ValueError("Stored teacher and initial full-context reference disagree")
        else:held_kl=learning_kl=teacher_nll=None
        record={"examples presented":presented,"epoch":round(presented/len(train),4),"held NLL":held_nll,
                "learning NLL":learning_nll,"held KL":held_kl,"learning KL":learning_kl,"teacher NLL":teacher_nll,
                "supervised tokens presented":target_tokens,"seconds":round(time.monotonic()-now,2)}
        if args.task=="compaction":
            record["turn NLL"]=turn_nll(summary,held);record["immediate KL"]=immediate_kl(summary,held)
        write(report/f"panel_{presented:07d}.json",{"summary":summary,"learning":learning_summary,"progress":asdict(progress)})
        if not panels or panels[-1]["examples presented"]!=presented:panels.append(record)
        write(panels_path,panels)
        cohorts = cohort_metrics(summary, held, args.task)
        controls = {name: cohort_metrics(json.loads((report / f"frozen_{name}.json").read_text()), held, args.task)
                    for name in ("full_context", "no_context")}
        r.widgets["cohorts"]["rows"] = [{"cohort": tag, "examples": values["examples"], "AC NLL": values["NLL"],
            "frozen full NLL": controls["full_context"][tag]["NLL"],
            "frozen no-context NLL": controls["no_context"][tag]["NLL"], "AC KL": values["KL"]}
            for tag, values in cohorts.items()]
        r.widgets["quality"]["series"]=([{"name":"AC held","x":[p["examples presented"] for p in panels],"y":[p["held NLL"] for p in panels]}]+
            [{"name":"frozen "+name,"x":[p["examples presented"] for p in panels],"y":[value]*len(panels)} for name,value in fixed["views"].items()])
        r.widgets["learning"]["series"]=[{"name":"AC learning","x":[p["examples presented"] for p in panels],"y":[p["learning NLL"] for p in panels]}]
        if args.task == "compaction":
            r.widgets["divergence"]["series"]=[{"name":name,"x":[p["examples presented"] for p in panels],"y":[p[name] for p in panels]} for name in ("held KL","learning KL")]
            r.widgets["divergence"]["series"] += [{"name":"frozen "+name,"x":[p["examples presented"] for p in panels],"y":[value]*len(panels)} for name,value in fixed["kl"].items()]
            r.widgets["immediate_kl"]["series"]=[{"name":name,"x":[p["examples presented"] for p in panels],"y":[p["immediate KL"][name] for p in panels]} for name in ("first 128","remaining")]
            r.widgets["immediate_kl"]["series"] += [{"name":"frozen "+name+" first 128","x":[p["examples presented"] for p in panels],"y":[values["first 128"]]*len(panels)} for name,values in fixed["immediate_kl"].items()]
            r.widgets["turns"]["series"]=([{"name":"AC "+key,"x":[p["examples presented"] for p in panels],"y":[p["turn NLL"][key] for p in panels]} for key in ("first","later")]+
                [{"name":"frozen full "+key,"x":[p["examples presented"] for p in panels],"y":[fixed["turns"]["full_context"][key]]*len(panels)} for key in ("first","later")])
        r.widgets["panels"]["rows"]=panels
        r.set_text("pending","Measurement status","Panels use fixed items and immutable initial-reader references. Offline continuation measurements do not establish live task success.")
        selected_path=report/"selected_checkpoint.json"
        previous=json.loads(selected_path.read_text()) if selected_path.exists() else {}
        selection_loss=held_kl if args.task=="compaction" else held_nll
        if presented and selection_loss<previous.get("selection_loss",math.inf):
            destination=args.output/"selected"/f"examples_{presented:07d}"
            if destination.exists():destination=destination.with_name(destination.name+f"_retry_{time.time_ns()}")
            destination.mkdir(parents=True)
            ac_path=ac.save(str(destination/"ac"),epoch_number=progress.epoch_number)
            reader_path=h.module_manager.save_lora(NAME,READER,str(destination/"reader"))
            if trainer._teacher_snapshot:
                teacher_folder=args.output/"selected"/"frozen_teacher"
                if not teacher_folder.exists():
                    temporary=teacher_folder.with_name(f".frozen_teacher_{time.time_ns()}")
                    shutil.copytree(trainer._teacher_snapshot["folder"],temporary)
                    temporary.rename(teacher_folder)
                if trainer._teacher_identity(ac,teacher_folder)!=trainer._teacher_snapshot["identity"]:
                    raise ValueError("Selected checkpoint's immutable teacher copy is incomplete or changed")
                trainer._save_teacher_snapshot(destination/"ac")
                teacher_manifest=json.loads((destination/"ac"/"teacher_snapshot.json").read_text())
                teacher_manifest["adapter_path"]=os.path.relpath(teacher_folder,destination/"ac")
                write(destination/"ac"/"teacher_snapshot.json",teacher_manifest)
                (teacher_folder/"PRESERVE").write_text("Immutable teacher shared by selected paired checkpoints.\n")
            (destination/"PRESERVE").write_text("Promising development checkpoint; preserve pair. Training recovery uses the run's published optimizer checkpoint.\n")
            write(selected_path,{"examples_presented":presented,"selection_loss":selection_loss,
                "selection_metric":"held token KL" if args.task=="compaction" else "held answer NLL",
                "ac_checkpoint":ac_path,"reader_checkpoint":reader_path,"optimizer_saved":False})
        qualitative_stage="baseline" if presented==0 else "complete" if presented>=len(train)*args.epochs else "halfway" if presented>=.5*len(train)*args.epochs else None
        if qualitative_stage and args.qualitative:
            qual_path=report/f"qualitative_{qualitative_stage}.json"
            if not qual_path.exists():
                r.set_status(phase="decoding fixed examples",sample_stage=qualitative_stage);r.render(force=True)
                sft._training_deadline=trainer._training_deadline
                sample=held[:args.qualitative];references_path=report/"qualitative_references.json"
                if not references_path.exists() and not resume:
                    references={}
                    for name in ("full_context","no_context"):
                        views=study.transform_for_eval_reference(sample,name)
                        references[name]=sft._sample_completions(ac,views,progress)
                    write(references_path,references)
                references=json.loads(references_path.read_text()) if references_path.exists() else {}
                samples=sft._sample_completions(ac,sample,progress)
                write(qual_path,{"examples_presented":presented,"max_new_tokens":config.completion_max_new_tokens,
                    "samples":samples,"frozen_references_available":bool(references)})
                r.set_trajectories("examples","Decoded examples — "+qualitative_stage,
                    f"Greedy next-turn generations; cap {config.completion_max_new_tokens} tokens. Frozen controls were decoded before training.",
                    [{"title":item.item_id,"state":"finished","stats":{"source":item.info["source"],"history tokens":item.info["source_tokens"]},
                      "prompt":item.info.get("question","Continue the original task after compressed history."),"file":qual_path.name,
                      "steps":[{"role":"Gold next turn","turn":0,"content":sample_row["reference"],"calls":[],"results":[]}]+
                        [{"role":"Frozen "+name,"turn":0,"content":values[index]["student"],"calls":[],"results":[]} for name,values in references.items()]+
                        [{"role":"AC","turn":0,"content":sample_row["student"],"calls":[],"results":[]}]}
                     for index,(item,sample_row) in enumerate(zip(sample,samples))])
        r.set_status(phase="quality panel complete",held_nll=f"{held_nll:.4f}",learning_nll=f"{learning_nll:.4f}")
        r.render(force=True)
    class ProgressReporter(ActivationContextTrainingReporter):
        def set_status(self, **fields):
            super().set_status(**fields)
            if "phase" in fields:
                phase_changed = r.status.get("phase") != fields["phase"]
                r.status.clear()
                r.set_status(phase=fields["phase"], elapsed_minutes=f"{(time.monotonic()-started)/60:.1f}")
                r.render(force=phase_changed)
        def report_phase(self,progress,phase):
            super().report_phase(progress,phase)
            presented,tokens=counts(progress.global_step)
            r.status.clear();r.set_status(phase=phase,examples_presented=str(presented),
                training=f"{len(train):,} unique {'questions' if args.task=='trajectory_qa' else 'examples'} × {presented/len(train):.3f} epochs",
                supervised_tokens_presented=f"{tokens:,}",distinct_source_roots=selection["training_roots"])
            r.render(force=True)
        def report_step(self,progress,stats):
            super().report_step(progress,stats)
            presented,tokens=counts(progress.global_step)
            r.status.clear();r.set_status(phase="training",examples_presented=str(presented),
                training=f"{len(train):,} unique {'questions' if args.task=='trajectory_qa' else 'examples'} × {presented/len(train):.3f} epochs",
                supervised_tokens_presented=f"{tokens:,}",distinct_source_roots=selection["training_roots"],
                latest_batch_loss=f"{stats.loss[-1]:.4f}",elapsed_minutes=f"{(time.monotonic()-started)/60:.1f}",
                peak_memory_gb=f"{torch.cuda.max_memory_allocated()/2**30:.2f}")
            r.render(force=True)
        def report_evaluation_progress(self,name,completed,total):
            super().report_evaluation_progress(name,completed,total)
            r.set_status(phase=name,items=f"{completed}/{total}");r.render(force=True)
    diagnostics=ProgressReporter(str(report/"diagnostics"),"Training diagnostics",args.task)
    r.set_status(phase="building training examples");r.render(force=True)
    try:
        stats=trainer.train(AC,train,reporting_data=held,validation_data=learning,reporter=diagnostics,reporting_callback=panel)
    finally:
        if offloader is not None:
            offloader.close()
            write(report/"moment_placement.json",{"offload_moments":True,"transfer_events":transfer_events})
    write(report/"result.json",stats.summarize())
    r.status.clear();r.set_status(phase="complete" if stats.stop_reason=="complete" else "paused for continuation",
        reason=stats.stop_reason,examples_presented=str(presentations(trainer.updates_done.get(AC,0))))
    r.finish() if stats.stop_reason=="complete" else r.render(force=True)
    if args.smoke:
        assert stats.steps>=2 and max(stats.grad_norm_ac)>0 and max(stats.grad_norm_reader)>0
        assert all(math.isfinite(v) for v in stats.loss)
        write(report/"gpu_proof.json",{"finite_losses":True,"encoder_and_reader_gradients":True,"steps":stats.steps,
            "peak_memory_bytes":stats.peak_memory_bytes,"source_tokens":[i.info["source_tokens"] for i in train]})
    return 0 if stats.stop_reason=="complete" else 75


if __name__=="__main__":
    signal.signal(signal.SIGUSR1,request_stop);signal.signal(signal.SIGTERM,request_stop)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",choices=["trajectory_qa","compaction"],required=True)
    parser.add_argument("--material",type=Path,required=True)
    parser.add_argument("--questions",type=Path)
    parser.add_argument("--initial-pair",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--items",type=int)
    parser.add_argument("--selection-file",type=Path,help="Explicit frozen QA train/held selection; preserves the supplied held panel")
    parser.add_argument("--target-tokens",type=int,default=0)
    parser.add_argument("--held",type=int,default=100)
    parser.add_argument("--learning",type=int,default=24)
    parser.add_argument("--qualitative",type=int,default=6)
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--ratio",type=float,default=1/16)
    parser.add_argument("--encoder-sees-question",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--encoder-lr",type=float,default=5e-5)
    parser.add_argument("--reader-lr",type=float,default=2e-5)
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--micro-batch",type=int,default=2)
    parser.add_argument("--execution-policy",type=Path,help="Validated tune_ac_execution result; training policy with fixed singleton evaluation")
    parser.add_argument("--teacher-block",type=int,default=32)
    parser.add_argument("--logits-chunk",type=int,default=256)
    parser.add_argument("--checkpoint-tokens",type=int,default=1700)
    parser.add_argument("--encode-tokens",type=int,default=65536)
    parser.add_argument("--seconds",type=int,default=21000)
    parser.add_argument("--seed",type=int,default=20260924)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--offload-moments",action="store_true",help="Privately park Adam moments between unchanged optimizer steps")
    parser.add_argument("--prepare-only",action="store_true",help="Prepare reusable tokenized examples on CPU, without training")
    parser.add_argument("--require-prepared",action="store_true",help="Refuse to spend a GPU allocation on initial CPU preparation")
    parser.add_argument("--prepare-workers",type=int,default=1,help="Tokenization workers for an explicitly CPU-only preparation job")
    args=parser.parse_args()
    if args.items is None:args.items=100000 if args.target_tokens else 16000
    if args.smoke:args.items=8;args.batch=2;args.held=3;args.learning=1;args.qualitative=1;args.target_tokens=0
    args.output.mkdir(parents=True,exist_ok=True)
    writer_lock=(args.output/"writer.lock").open("a")
    # A duplicate launch must not overwrite the active run's report with a failure.
    fcntl.flock(writer_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        raise SystemExit(main(args))
    except Exception as error:
        import traceback
        write(args.output/"failure.json",{"error":str(error),"traceback":traceback.format_exc()})
        report_path=args.output/"report"/"report_data.json"
        if report_path.exists():
            document=json.loads(report_path.read_text());document["status"]={"phase":"failed","error":str(error)}
            document["updated_at"]=time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime());write(report_path,document)
        raise
