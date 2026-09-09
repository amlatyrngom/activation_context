# Slice 3a1 implementation

These product additions, replacements and deletions are exact deltas relative to source HEAD 302be2f plus the user’s uncommitted prototype, as verified by source_manifest.json. The user source remained read-only throughout implementation. Apply changes.patch from the user repository root after checking the source hashes, or apply the equivalent exact per-file diffs in ../SLICE3A1_ROUND.tressoir.md. Preserve unrelated work, including the two dataset-study sketches.

The two activation/bench/agent_probes files are IB-only benchmark/profile tools. Keep them in IB; they are deliberately excluded from the product patch and its diff cards. To run one in an isolated checkout, mirror the staged probe under that checkout’s activation/bench/agent_probes, as done for validation.

New AC checkpoints use architecture version 2 and cannot reinterpret old learned-marker checkpoints. load_completed_epoch() recovers both committed numbered adapters; optimizer state is not resumed across processes. Source application status is tracked in IB/STATE.md.

The autotuner extension is included. Agent and AC trainers request execution settings through activation.autotuners. Compatible shared profiles precede the two validated RTX PRO 6000 presets; incompatible environments perform bounded cold tuning or fail clearly. The ten legacy files are seeds requiring validation. Checkpoint/logits auto settings are conservative heuristics; explicit values win. Inspect, retune or promote with python -m activation.autotuners. See autotuner_validation_summary.json for tested revisions, coverage, timings and limits; the original validation_summary.json retains the earlier AC quality results.
