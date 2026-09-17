# AC and agent co-design handoff

This folder contains the final product files, `changes.patch` against the current `/source`, and `source_manifest.json` with source/staged hashes. Private test sources are excluded. The two complete mainline examples and the approved existing-test compatibility edit are included.

Review `../CODESIGN_ROUND.tressoir.md` for behavior, validation and complete per-file diffs. Apply only to a checkout matching the manifest source hashes: run `git apply --check changes.patch` before `git apply changes.patch`. This workspace already contains the changes.
