"""
How far dumb programmatic fitting takes BRIGHT economics: a random span of a chunk is the query,
the chunk its positive, negatives from the batch only. No oracle stages; the cost is training.
The eval is the 103 native queries in one batch, before and after, next to the frozen base and
the trained reference embedder.

    uv run sky exec --sync <node> -- uv run python -m activation.bench.bright_econ_ablation.bench_programmatic_fitting --probe

`--probe` trains 2,048 examples for 2 epochs and writes PROBE_REPORTS/programmatic_<stamp>/probe_timing.json
with the hours per sample count; without it the run is the Round 1 shape (200k examples, 1 epoch;
`--num-samples` overrides). `--pretrain-programmatic` is meaningless here. Reports go to the synced
folder IB/TMP/SYNC/BRIGHT_ECON_ABLATION/(PROBE_)REPORTS/.
"""
from activation.bench.bright_econ_ablation._common import run


def make_training_data(harness, dataset, args, timer):
    with timer.stage("programmatic examples", units=args.num_samples):
        examples = harness.dataset_manager.synthesize_programmatic_examples(dataset.dataset_id, args.num_samples)
    return examples, {"examples": len(examples)}


if __name__ == "__main__":
    run("programmatic", make_training_data)
