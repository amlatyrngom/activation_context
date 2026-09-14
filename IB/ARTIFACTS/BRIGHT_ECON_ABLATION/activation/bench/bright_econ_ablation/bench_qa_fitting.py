"""
Like description but with question generation: a hard question with its answer per chunk (the
existing study pipeline), labeled against the BM25 top-10. Same flags, cache and probe shape as
the description bench.

    uv run sky exec --sync <node> -- uv run python -m activation.bench.bright_econ_ablation.bench_qa_fitting --probe
"""
from activation.bench.bright_econ_ablation._common import MODE_ORIGINS, label_examples, run


def make_training_data(harness, dataset, args, timer):
    with timer.stage("generation", units=args.num_samples) as record:
        examples = harness.dataset_manager.synthesize_study_examples_qa(dataset.dataset_id, args.num_samples, caching_id=args.caching_id)
        record["units"] = args.num_samples
    summary = {"chunks": args.num_samples, "questions": len(examples), "samples": [(e.query, e.gold_answers[0], e.positive_chunk_ids[0]) for e in examples[:4]]}
    summary |= label_examples(harness, dataset, examples, MODE_ORIGINS["question"], args, timer)
    return examples, summary


if __name__ == "__main__":
    run("question", make_training_data)
