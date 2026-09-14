# Slice 4a, first green light: implementation

M0 prototype changes, M1 benchmark suite and environments, M2 rollout probe, plus the M4 reorganization to your data model and the rollout throughput work (tuning subplan v0.3). These product additions and replacements are exact deltas relative to the current `/source` (which carries slice 3b and your prototype diff), as verified by `source_manifest.json`. The user source remained read-only throughout. Apply `changes.patch` from the repository root with `git apply` (or `patch -p1`) after checking the source hashes, or apply the equivalent exact per-file diffs in `../SLICE4A_ROUND.tressoir.md`. Preserve unrelated work.

The 4a work tree is `IB/TMP/AGENT_AC_PRETRAINING/work`; node logs are `IB/TMP/AGENT_AC_PRETRAINING/*.log`; the pulled probe folder is `IB/TMP/AGENT_AC_PRETRAINING/node/`.
