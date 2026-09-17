# RAG POC reporting additions

These four current product files add preparation/evaluation progress and named evaluation curves. The experiment, private tests and monitoring remain outside the main implementation.

`incremental.patch` applies after the accepted co-design handoff, to files matching `before_poc_sha256` in `source_manifest.json`. Run `git apply --check incremental.patch` before applying. This workspace already contains it.

`source_reference.patch` contains the complete four-file delta against the current `/source` baseline for review. It also includes earlier co-design changes in overlapping files, so it is not an additional patch to apply after co-design. Other co-design files remain required; refer to `../CODESIGN_ROUND.tressoir.md`.

Validation and experiment outcomes will be linked when the monitored GPU runs finish. Source was verified before launch; the final handoff review is pending.
