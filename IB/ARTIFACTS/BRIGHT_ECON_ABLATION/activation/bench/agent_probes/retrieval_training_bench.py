"""
Slice-1 acceptance run: train the LoRA + activation-context retrieval model on MS-MARCO and write
the live report. The 1000-query / 50-reporting run on one GPU is the acceptance shape; the same
script with 50k queries and one epoch is the production shape.

    uv run python -m activation.bench.retrieval_training_bench --num-queries 1000 --epochs 2 \
        --report-folder ~/activation_artifacts/retrieval_slice1
"""
import argparse
import json
import os
import time

import torch

from activation.dataset.loaders import MsMarcoDataset
from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
from activation.retrieval import (
    RetrievalModel,
    RetrievalReporter,
    RetrievalTrainer,
    RetrievalTrainingConfig,
    StandardRetrievalACModel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--baseline-model-id", default="Qwen/Qwen3-Embedding-0.6B",
                        help="well-trained embedder scored on the validation set as a reference; 'none' to skip")
    parser.add_argument("--num-queries", type=int, default=1000, help="queries selected before the validation split")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--reporting-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None, help="examples per step; default: sized for the GPU")
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--num-view-tokens", type=int, default=8)
    parser.add_argument("--d-embedding-result", type=int, default=None)
    parser.add_argument("--chunk-size-chars", type=int, default=4096)
    parser.add_argument("--report-folder", default=os.path.expanduser("~/activation_artifacts/retrieval_slice1"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.report_folder = os.path.expanduser(args.report_folder)                       # a quoted "~" reaches us unexpanded

    base_model_name = args.base_model_id.split("/")[-1].lower()
    lora_name, ac_name = f"{base_model_name}-retrieval-lora", f"{base_model_name}-retrieval-ac"
    model_configs = {base_model_name: ModelConfig(base_model_name, args.base_model_id)}
    baseline_model_name = None
    if args.baseline_model_id.lower() != "none":
        baseline_model_name = args.baseline_model_id.split("/")[-1].lower()
        model_configs[baseline_model_name] = ModelConfig(baseline_model_name, args.baseline_model_id)
    harness = HarnessRuntime(HarnessRuntimeConfig(
        model_configs=model_configs,
        doc_chunk_size_chars=args.chunk_size_chars,
        doc_embedding_input_limit_chars=args.chunk_size_chars,
    ))
    # Rows without a selected passage yield no example: load extra rows so num_queries are labeled.
    load_start = time.time()
    dataset = MsMarcoDataset.load(harness, max_examples=int(args.num_queries * 1.25) + 8, max_corpus_documents=0)
    harness.dataset_manager.build_bm25_indexes()
    training_data, validation_data, reporting_data = harness.dataset_manager.select_training_data(
        dataset.dataset_id, num_samples=args.num_queries, oracle_labeled_only=False,
        val_ratio=args.val_ratio, max_reporting_size=args.reporting_size, seed=args.seed,
    )
    print(f"Data ready in {time.time() - load_start:.1f}s: {json.dumps(dataset.stats.summarize(), indent=1)}")

    harness.module_manager.register_lora(lora_name, base_model_name, rank=args.lora_rank)
    torch.manual_seed(args.seed)                                                       # seeded AC and head init
    ac_model = StandardRetrievalACModel(harness, base_model_name, num_view_tokens=args.num_view_tokens)
    harness.module_manager.register_retrieval_ac(ac_name, base_model_name, ac_model)
    retrieval_model = RetrievalModel(
        harness, base_model_name, d_embedding_result=args.d_embedding_result, ac_name=ac_name, lora_name=lora_name,
    )
    config = RetrievalTrainingConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, baseline_model_name=baseline_model_name)
    reporter = RetrievalReporter(
        args.report_folder,
        title=f"Retrieval training — MS-MARCO train, {len(training_data):,} queries",
        description=f"{args.base_model_id} frozen + LoRA r{args.lora_rank} + AC (V={args.num_view_tokens}), "
                    f"{args.epochs} epochs, {len(validation_data)} validation / {len(reporting_data)} reporting queries.",
    )
    trainer = RetrievalTrainer(harness)
    evaluations = {"before": trainer.eval(retrieval_model, reporter, validation_data, config, label="validation before", progress=0.0)}
    stats = trainer.train(config, retrieval_model, reporter, training_data, reporting_data)
    evaluations["after"] = trainer.eval(retrieval_model, reporter, validation_data, config, label="validation after", progress=float(args.epochs))
    summary = stats.summarize() | {"evaluations": evaluations}
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.report_folder, "training_stats.json"), "w") as handle:
        json.dump({"summary": summary, "harness": harness.harness_stats.summarize()}, handle, indent=1)


if __name__ == "__main__":
    main()
