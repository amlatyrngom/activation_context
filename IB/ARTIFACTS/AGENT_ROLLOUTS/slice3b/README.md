# Slice 3b implementation

Activation context in the agent loop. These product additions and replacements are exact deltas relative to the current `/source` (which already carries the applied slice 3a1), as verified by `source_manifest.json`. The user source remained read-only throughout implementation. Apply `changes.patch` from the repository root with `git apply` (or `patch -p1`) after checking the source hashes, or apply the equivalent exact per-file diffs in `../SLICE3B_ROUND.tressoir.md`. Preserve unrelated work.

`probes/ac_payload_probe.py` is the IB-only M0 payload probe. It is deliberately excluded from the product patch; to run it, mirror it under `activation/bench/agent_probes/` in a checkout, as done for validation.

Serialized run records now carry `compactions`, per-step `messages` and `ac_spans`; records written before this slice deserialize with empty segments. `AgentConfig.deserialize` drops the legacy `ac_inputs` key. Training on AC-bearing runs raises until the fixed-row trainer integration (deferred to the training-recipe slices); text-only runs train as before, with compacted segments included in selection.

The full 3b tree is `IB/TMP/SLICE3B/implementation/work`; CPU checks are `IB/TMP/SLICE3B/cpu_checks/`; node logs and results are `IB/TMP/SLICE3B/node/`.
