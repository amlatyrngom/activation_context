"""
What the three BRIGHT economics fitting benches share: arguments, the harness with every model,
the whole economics corpus, the training model, stage timing, the evaluation on the full test set
(one batch, next to the frozen base and the trained reference embedder), the report folders under
the synced area, and the timing file with the budget arithmetic that a probe prints.

A bench runs, in order, on one node: generation (engine), labeling (engine, one or two labelers),
engines freed, the base loaded, eval before, training, eval after. `--probe` shrinks every stage
to a few minutes and writes `probe_timing.json` with the rates and the hours for 25k–200k samples.
"""
import argparse
import datetime as dt
import json
import os
import uuid
import time
import typing as t
from contextlib import contextmanager

import torch

from activation.common.data_syncing import resolve_path
from activation.dataset.dataset import DataOrigin, LabeledRetrievalQAExample, LoadedDataset
from activation.dataset.loaders import BrightDataset
from activation.harness import FREE_DEVICE, SUPPORTS_FP4, HarnessRuntime, HarnessRuntimeConfig, ModelConfig
from activation.retrieval import (
    RetrievalModel,
    RetrievalReporter,
    RetrievalTrainer,
    RetrievalTrainingConfig,
    StandardRetrievalACModel,
)

DOMAIN = "economics"
SYNC_FOLDER = "BRIGHT_ECON_ABLATION"
DEFAULT_LABELER = "nvidia/Qwen3.6-35B-A3B-NVFP4" if SUPPORTS_FP4 else "Qwen/Qwen3.6-35B-A3B-FP8"
BUDGET_HOURS = 5.5                    # 6 h round minus 30 min of safety
EXTRAPOLATION_SAMPLES = (25_000, 50_000, 100_000, 200_000)

MODE_ORIGINS = {
    "programmatic": DataOrigin.PROGRAMMATIC,
    "description": DataOrigin.SYNTHETIC_DESCRIPTION,
    "question": DataOrigin.SYNTHETIC_QA,
}
PROBE_SAMPLES = {"programmatic": 2048, "description": 512, "question": 512}
FULL_SAMPLES = {"programmatic": 200_000, "description": 50_000, "question": 50_000}


def parse_args(mode: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"BRIGHT {DOMAIN}: {mode} fitting", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", action="store_true", help="short run of every stage; writes probe_timing.json with the budget arithmetic")
    parser.add_argument("--num-samples", type=int, default=None, help=f"chunks to study (default: {PROBE_SAMPLES[mode]} probe, {FULL_SAMPLES[mode]} full)")
    parser.add_argument("--epochs", type=int, default=None, help="default: 2 probe, 1 full")
    parser.add_argument("--pretrain-programmatic", type=int, default=0, help="Round 2: this many programmatic examples trained first, in the same process")
    parser.add_argument("--study-only", action="store_true", help="generate and label into the cache, then stop (no model, eval or training)")
    parser.add_argument("--base-model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--baseline-model-id", default="Qwen/Qwen3-Embedding-4B", help="trained reference embedder; 'none' to skip")
    parser.add_argument("--generator-model-id", default="RedHatAI/Qwen3.5-4B-FP8-dynamic")
    parser.add_argument("--label-model-ids", nargs="+", default=None,
                        help=f"labelers; the first one's labels are kept, the others are compared (default: {DEFAULT_LABELER}, plus the generator in a probe)")
    parser.add_argument("--caching-id", default="econ", help="cache key for generation and labels under the synced CACHE folder")
    parser.add_argument("--batch-size", type=int, default=None, help="examples per step; default: sized for the GPU")
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--num-prefix-tokens", type=int, default=16)
    parser.add_argument("--num-view-tokens", type=int, default=8)
    parser.add_argument("--d-ac-model", type=int, default=1024)
    parser.add_argument("--num-ac-layers", type=int, default=8)
    parser.add_argument("--chunk-size-chars", type=int, default=4096)
    parser.add_argument("--val-ratio", type=float, default=0.02, help="share of the study examples held out as the reporting pool")
    parser.add_argument("--reporting-size", type=int, default=50)
    parser.add_argument("--reporting-fraction", type=float, default=0.1, help="a reporting point every this fraction of an epoch")
    parser.add_argument("--report-on-test", action="store_true",
                        help="report on the test queries (one batch, the eval's pool) instead of the study hold-out: a learning curve on the eval metric")
    parser.add_argument("--report-root", default=None, help=f"default: the synced folder {SYNC_FOLDER}/")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.mode = mode
    args.num_samples = args.num_samples or (PROBE_SAMPLES[mode] if args.probe else FULL_SAMPLES[mode])
    args.epochs = args.epochs or (2 if args.probe else 1)
    if args.label_model_ids is None:
        args.label_model_ids = [DEFAULT_LABELER] + ([args.generator_model_id] if args.probe else [])
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    root = args.report_root or str(resolve_path(SYNC_FOLDER))
    # Sample count and a random suffix keep parallel runs of one mode (same minute) in separate folders.
    args.report_folder = os.path.join(root, "PROBE_REPORTS" if args.probe else "REPORTS", f"{mode}_n{args.num_samples}_{stamp}_{uuid.uuid4().hex[:8]}")
    os.makedirs(args.report_folder, exist_ok=True)
    return args


def model_name_of(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def build_harness(args: argparse.Namespace) -> HarnessRuntime:
    """Base, baseline, generator and labelers registered; the study cache under the synced CACHE folder."""
    ids = [args.base_model_id, args.generator_model_id, *args.label_model_ids]
    if args.baseline_model_id.lower() != "none":
        ids.append(args.baseline_model_id)
    model_configs = {model_name_of(model_id): ModelConfig(model_name_of(model_id), model_id) for model_id in dict.fromkeys(ids)}
    return HarnessRuntime(HarnessRuntimeConfig(
        model_configs=model_configs,
        doc_chunk_size_chars=args.chunk_size_chars,
        doc_embedding_input_limit_chars=args.chunk_size_chars,
        dataset_study_qa_model_name=model_name_of(args.generator_model_id),
        dataset_study_label_model_name=model_name_of(args.label_model_ids[0]),
        dataset_study_seed=args.seed,
        cache_storage_dir=str(resolve_path(f"{SYNC_FOLDER}/CACHE")),
    ))


def load_economics(harness: HarnessRuntime) -> LoadedDataset:
    """The whole domain: every labeled query and every document, BM25 built."""
    dataset = BrightDataset.load(harness, max_examples=None, domain=DOMAIN, max_corpus_documents=None)
    harness.dataset_manager.build_bm25_indexes()
    return dataset


def build_model(harness: HarnessRuntime, args: argparse.Namespace) -> RetrievalModel:
    base = model_name_of(args.base_model_id)
    lora_name, ac_name = f"{base}-econ-lora", f"{base}-econ-ac"
    harness.module_manager.register_lora(lora_name, base, rank=args.lora_rank)
    torch.manual_seed(args.seed)
    ac_model = StandardRetrievalACModel(
        harness, base, d_ac_model=args.d_ac_model, num_prefix_tokens=args.num_prefix_tokens,
        num_view_tokens=args.num_view_tokens, num_ac_layers=args.num_ac_layers,
    )
    harness.module_manager.register_retrieval_ac(ac_name, base, ac_model)
    return RetrievalModel(harness, base, d_embedding_result=None, ac_name=ac_name, lora_name=lora_name)


def training_config(args: argparse.Namespace, harness: HarnessRuntime, epochs: int) -> RetrievalTrainingConfig:
    baseline = None if args.baseline_model_id.lower() == "none" else model_name_of(args.baseline_model_id)
    return RetrievalTrainingConfig(
        epochs=epochs, batch_size=args.batch_size, seed=args.seed, reporting_size=args.reporting_size,
        reporting_fraction=args.reporting_fraction, baseline_model_name=baseline, reference_frozen_base=True,
    )


class StageTimer:
    """
    Wall time and unit count per stage, in order; a stage name may repeat (labelers). Model and
    engine loads that happen inside a stage are measured apart (`load_seconds`, from the harness
    stats), so `net_seconds` is the per-unit work and the loads are a run's fixed cost.
    """

    def __init__(self, harness: HarnessRuntime):
        self.harness = harness
        self.stages: list[dict] = []

    def _loads(self) -> float:
        return sum(seconds for _, seconds in self.harness.harness_stats.model_loading_times)

    @contextmanager
    def stage(self, name: str, units: int = 0):
        start, loads_before = time.time(), self._loads()
        record = {"stage": name, "units": units, "seconds": 0.0, "load_seconds": 0.0, "net_seconds": 0.0}
        self.stages.append(record)
        print(f"=== {name} ({units} units) ...", flush=True)
        try:
            yield record
        finally:
            record["seconds"] = time.time() - start
            record["load_seconds"] = self._loads() - loads_before
            record["net_seconds"] = record["seconds"] - record["load_seconds"]
            rate = record["units"] / record["net_seconds"] if record["net_seconds"] > 0 and record["units"] else 0.0
            print(f"=== {name}: {record['seconds']:.1f} s ({record['load_seconds']:.1f} s of model loads)"
                  + (f", {rate:.2f} units/s net" if rate else ""), flush=True)

    def net_seconds(self, prefix: str) -> float:
        return sum(record["net_seconds"] for record in self.stages if record["stage"].startswith(prefix))


def free_engines(harness: HarnessRuntime) -> None:
    for loaded in harness.loaded_models.values():
        loaded.engine_to_device(FREE_DEVICE)


def label_examples(
    harness: HarnessRuntime, dataset: LoadedDataset, examples: list[LabeledRetrievalQAExample], origin: DataOrigin,
    args: argparse.Namespace, timer: StageTimer,
) -> dict:
    """
    Label the examples with every labeler in turn (each pass timed separately, each cached under
    its own key); the first labeler's labels are the ones kept. Between labelers the examples are
    reset to their source-chunk positive so each labeler judges the same pool. Returns per-labeler
    counts and, for the extra labelers, the agreement with the first (mean Jaccard of the positive
    and of the negative sets over the examples both labeled).
    """
    ids = {example.example_id for example in examples}
    snapshots: dict[str, dict[str, tuple[set, set]]] = {}
    summary: dict = {"labelers": {}}
    for model_id in args.label_model_ids:
        for example in examples:                                                         # same starting point for every labeler
            example.positive_chunk_ids = list(example.positive_chunk_ids[:1])
            example.hard_negative_chunk_ids = None
            example.oracle_labeled = False
        name = model_name_of(model_id)
        with timer.stage(f"labeling:{name}", units=len(examples)) as record:
            labeled = harness.dataset_manager.label_study_examples(
                dataset.dataset_id, len(examples), origins=[origin], caching_id=args.caching_id, label_model_name=name,
            )
            record["units"] = len(labeled)
        snapshots[name] = {e.example_id: (set(e.positive_chunk_ids or []), set(e.hard_negative_chunk_ids or [])) for e in labeled if e.example_id in ids}
        summary["labelers"][name] = {
            "labeled": len(labeled),
            "positives_per_example": sum(len(p) for p, _ in snapshots[name].values()) / max(1, len(snapshots[name])),
            "negatives_per_example": sum(len(n) for _, n in snapshots[name].values()) / max(1, len(snapshots[name])),
        }
    first = model_name_of(args.label_model_ids[0])
    for name, snapshot in snapshots.items():
        if name == first:
            continue
        shared = [example_id for example_id in snapshot if example_id in snapshots[first]]
        def jaccard(a: set, b: set) -> float:
            return 1.0 if not a and not b else len(a & b) / len(a | b)
        summary["labelers"][name]["agreement_with_first"] = {
            "examples": len(shared),
            "positives_jaccard": sum(jaccard(snapshot[i][0], snapshots[first][i][0]) for i in shared) / max(1, len(shared)),
            "negatives_jaccard": sum(jaccard(snapshot[i][1], snapshots[first][i][1]) for i in shared) / max(1, len(shared)),
        }
    for example in examples:                                                             # keep the first labeler's verdicts
        if example.example_id in snapshots.get(first, {}):
            positives, negatives = snapshots[first][example.example_id]
            example.positive_chunk_ids = [example.positive_chunk_ids[0]] + sorted(positives - {example.positive_chunk_ids[0]})
            example.hard_negative_chunk_ids = sorted(negatives) or None
            example.oracle_labeled = True
    return summary


def evaluate(
    trainer: RetrievalTrainer, model: RetrievalModel, reporter: RetrievalReporter, test_data: list[LabeledRetrievalQAExample],
    config: RetrievalTrainingConfig, label: str, progress: float, timer: StageTimer,
) -> dict:
    """The full test set as one batch (103 queries against every query's gold chunks), the model next to the references."""
    with timer.stage(f"eval:{label}", units=len(test_data)):
        results = trainer.eval(model, reporter, test_data, config, label=label, progress=progress)
    return {name: {"loss": loss, **metrics} for name, (loss, metrics) in results.items()}


def load_seconds(harness: HarnessRuntime) -> dict[str, float]:
    """Model and engine load time per model name, from the harness stats (fixed costs of a run)."""
    totals: dict[str, float] = {}
    for name, seconds in harness.harness_stats.model_loading_times:
        totals[name] = totals.get(name, 0.0) + seconds
    return totals


def extrapolate(mode: str, timer: StageTimer, loads: dict[str, float], training_summary: dict, probe_epochs: int, num_samples: int) -> dict:
    """
    Hours for n study chunks from the probe's net rates: fixed costs (every model and engine load,
    the corpus, both evals) plus n x (generation + first-labeler labeling seconds per chunk) plus
    epochs x (training examples per chunk) x n x seconds per training example. Descriptions yield
    fewer examples than chunks (needs_abstraction_bridge), which the examples-per-chunk ratio
    carries. Given for one epoch (Round 1) and for the probe's epochs.
    """
    generation = next((r for r in timer.stages if r["stage"] == "generation"), None)
    labeling = next((r for r in timer.stages if r["stage"].startswith("labeling:")), None)
    study = next((r for r in timer.stages if r["stage"] == "study data"), None)
    per_chunk = 0.0
    if generation and generation["units"]:
        per_chunk += generation["net_seconds"] / generation["units"]
    if labeling and labeling["units"]:
        per_chunk += labeling["net_seconds"] / labeling["units"] * (labeling["units"] / max(1, num_samples))
    examples_trained = training_summary["num_examples"]
    train_seconds_per_example_epoch = training_summary["total_train_time"] / max(1, examples_trained * probe_epochs)
    examples_per_chunk = (study["units_out"] if study and study.get("units_out") else num_samples) / max(1, num_samples)
    fixed = sum(loads.values()) + timer.net_seconds("eval:") + timer.net_seconds("corpus")

    def table(epochs: int) -> dict:
        per_sample = per_chunk + epochs * examples_per_chunk * train_seconds_per_example_epoch
        hours = {str(n): round((fixed + n * per_sample) / 3600, 2) for n in EXTRAPOLATION_SAMPLES}
        largest = int(max(0.0, BUDGET_HOURS * 3600 - fixed) / per_sample) if per_sample else None
        return {"epochs": epochs, "seconds_per_chunk": per_sample, "hours_for_chunks": hours, "largest_chunks_under_budget": largest}

    return {
        "mode": mode, "probe_epochs": probe_epochs, "probe_chunks": num_samples,
        "fixed_seconds": fixed, "model_load_seconds": loads,
        "generation_seconds_per_chunk": generation["net_seconds"] / generation["units"] if generation and generation["units"] else None,
        "labeling_seconds_per_example": labeling["net_seconds"] / labeling["units"] if labeling and labeling["units"] else None,
        "examples_per_chunk": examples_per_chunk,
        "training_seconds_per_example_epoch": train_seconds_per_example_epoch,
        "training_tokens_per_example": training_summary["avg_tokens_per_example"],
        "training_tokens_per_s": training_summary["tokens_per_s"],
        "budget_hours": BUDGET_HOURS,
        "round_1": table(1), "probe_shape": table(probe_epochs),
    }


def run(mode: str, make_training_data: t.Callable[[HarnessRuntime, LoadedDataset, argparse.Namespace, StageTimer], tuple[list[LabeledRetrievalQAExample], dict]]) -> None:
    """The bench: corpus, study data, engines freed, base, eval, training, eval, timing file."""
    args = parse_args(mode)
    print(f"{mode} fitting -> {args.report_folder}\n{json.dumps(vars(args), indent=1, default=str)}", flush=True)
    harness = build_harness(args)
    timer = StageTimer(harness)
    with timer.stage("corpus"):
        dataset = load_economics(harness)
    test_data = harness.dataset_manager.select_testing_data(dataset.dataset_id, seed=args.seed)
    study_summary: dict = {}
    with timer.stage("study data", units=args.num_samples) as study_record:
        study_examples, study_summary = make_training_data(harness, dataset, args, timer)
        study_record["units_out"] = len(study_examples)
    if args.study_only:
        output = {"args": vars(args), "stages": timer.stages, "model_load_seconds": load_seconds(harness), "study": study_summary,
                  "dataset_stats": dataset.stats.summarize(), "harness": harness.harness_stats.summarize()}
        with open(os.path.join(args.report_folder, "study_summary.json"), "w") as handle:
            json.dump(output, handle, indent=1, default=str)
        print(json.dumps({"stages": timer.stages, "study": study_summary}, indent=1, default=str), flush=True)
        return
    pretrain_examples = []
    if args.pretrain_programmatic:
        pretrain_examples = harness.dataset_manager.synthesize_programmatic_examples(dataset.dataset_id, args.pretrain_programmatic)
    free_engines(harness)

    model = build_model(harness, args)
    trainer = RetrievalTrainer(harness)
    reporter = RetrievalReporter(
        args.report_folder, title=f"BRIGHT {DOMAIN}: {mode} fitting" + (" (probe)" if args.probe else ""),
        description=f"{args.base_model_id} frozen + LoRA r{args.lora_rank} + AC (P={args.num_prefix_tokens}, V={args.num_view_tokens}, "
                    f"d {args.d_ac_model}, {args.num_ac_layers} layers); {len(study_examples):,} {mode} examples, {args.epochs} epoch(s); "
                    f"eval on the {len(test_data)} native queries as one batch.",
    )
    config = training_config(args, harness, args.epochs)
    evaluations = {"before": evaluate(trainer, model, reporter, test_data, config, "before training", 0.0, timer)}
    summaries = {}
    if pretrain_examples:
        train, _, report = harness.dataset_manager.select_training_data(
            dataset.dataset_id, len(pretrain_examples), origins=[DataOrigin.PROGRAMMATIC], oracle_labeled_only=False,
            val_ratio=args.val_ratio, max_reporting_size=args.reporting_size, seed=args.seed,
        )
        if args.report_on_test:
            report = test_data
        with timer.stage("training:programmatic", units=len(train)):
            summaries["programmatic"] = trainer.train(training_config(args, harness, 1), model, reporter, train, report).summarize()
        evaluations["after programmatic"] = evaluate(trainer, model, reporter, test_data, config, "after programmatic", 1.0, timer)
    train, _, report = harness.dataset_manager.select_training_data(
        dataset.dataset_id, len(study_examples), origins=[MODE_ORIGINS[mode]], oracle_labeled_only=(mode != "programmatic"),
        val_ratio=args.val_ratio, max_reporting_size=args.reporting_size, seed=args.seed,
    )
    if args.report_on_test:
        report = test_data                                                 # the eval batch as the learning curve
    with timer.stage(f"training:{mode}", units=len(train)):
        summaries[mode] = trainer.train(config, model, reporter, train, report).summarize()
    evaluations["after"] = evaluate(trainer, model, reporter, test_data, config, "after training", float(args.epochs), timer)

    loads = load_seconds(harness)
    timing = extrapolate(mode, timer, loads, summaries[mode], args.epochs, args.num_samples)
    output = {
        "args": vars(args), "stages": timer.stages, "model_load_seconds": loads, "study": study_summary,
        "training": summaries, "evaluations": evaluations, "extrapolation": timing,
        "dataset_stats": dataset.stats.summarize(), "harness": harness.harness_stats.summarize(),
    }
    name = "probe_timing.json" if args.probe else "run_summary.json"
    with open(os.path.join(args.report_folder, name), "w") as handle:
        json.dump(output, handle, indent=1, default=str)
    reporter.set_text("timing", "Stages and budget", json.dumps({"stages": timer.stages, "extrapolation": timing, "study": study_summary}, indent=2, default=str))
    reporter.render(force=True)
    print(json.dumps({"evaluations": evaluations, "extrapolation": timing, "study": study_summary}, indent=1, default=str), flush=True)
