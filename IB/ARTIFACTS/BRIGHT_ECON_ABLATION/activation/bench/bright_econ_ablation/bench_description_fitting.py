"""
Like programmatic, but a generated search-friendly description of each chunk is the query and the
labeler adds positives and hard negatives from the BM25 top-10. Generation and labels are cached
under the synced folder IB/TMP/SYNC/BRIGHT_ECON_ABLATION/CACHE/ by `--caching-id`, so Round 2
(`--pretrain-programmatic N`) pays training only.

    uv run sky exec --sync <node> -- uv run python -m activation.bench.bright_econ_ablation.bench_description_fitting --probe

`--probe`: 512 chunks through generation, both labelers (the configured one and the generator,
agreement reported), 2 training epochs, the eval on the 103 native queries before and after.
"""
from activation.bench.bright_econ_ablation._common import MODE_ORIGINS, label_examples, run


def make_training_data(harness, dataset, args, timer):
    with timer.stage("generation", units=args.num_samples) as record:
        examples = harness.dataset_manager.synthesize_study_examples_description(dataset.dataset_id, args.num_samples, caching_id=args.caching_id)
        record["units"] = args.num_samples
    summary = {"chunks": args.num_samples, "descriptions": len(examples), "samples": [(e.query, e.positive_chunk_ids[0]) for e in examples[:4]]}
    summary |= label_examples(harness, dataset, examples, MODE_ORIGINS["description"], args, timer)
    return examples, summary


if __name__ == "__main__":
    run("description", make_training_data)
