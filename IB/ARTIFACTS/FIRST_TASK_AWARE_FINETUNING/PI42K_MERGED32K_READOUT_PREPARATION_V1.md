# Readout after960 fresh questions at merged32k

This separate candidate accepts only `pi42k_merged32k_fresh960_v1` after960 fresh questions once:42,960 unique questions, own120-update epoch001 AC/reader publication, and615 actual Adam states at5370. It checks the frozen training configuration, source receipt, paired initialization, subset/order, final panel and published marker. Selected or partial checkpoints are rejected. Actual optimizer validation happens on allocated compute; no optimizer is used for evaluation.

One45-minute CSAIL H200 allocation runs three fresh processes in order:

1. Final32k: the same original100, fresh AC/full/no-context NLL, and100 greedy AC answers. Every target-token score replays the own saved final100 panel under the existing1e-3 gate. Full sources, actual rows, single merged encoder request and exact training-input fingerprints are checked. Answers retain actual IDs, chat-EOS termination and160-token caps.
2. Parent42k at native2k: fresh NLL for all three views on the same100.
3. Final42,960 at native2k: identical inputs and fresh NLL. A retention summary requires matching input/row fingerprints, model provenance and hardware/precision policy across the two2k cells.

There is no RTX score reuse. Parent42k full-context32k answers are supplied by its separate, explicitly labeled RTX readout. No extra full-context answers are generated here. The32k cell has1850seconds; each2k cell300seconds. Expect roughly15–25minutes overall, with model load and repeated publication checks included; this is not a measured runtime. Partial output is retained and stops the launcher so an explicit repeat resumes the same cell. No confirmation data is read.

Candidate: `bench/evaluate_pi42k_merged32k_v1.sbatch <frozen training design>`. Default host paths are CSAIL; `QA_ROOT`, `QA_PYTHON`, `QA_HF_HOME`, `QA_READOUT_AUTOTUNE` and scheduler flags can be overridden together. Own checkpoint paths still must match the actual frozen training publication. The separate `PI42K_MERGED32K_READOUT_MANIFEST_V1.json` pins readout helpers without changing the training design after launch.

Twenty-one focused CPU checks passed: changed config/order/parent, incomplete final, wrong marker, foreign pair and wrong optimizer clocks reject. Future publication tests use small synthetic metadata; no future checkpoint is claimed to exist. The runner's harness, material builder and answer masks are structurally identical to the parent builder just checked on all100 real tokenized merged32k inputs. GPU-time rebuilding and own final replay remain required. No source edits, staging or GPU jobs were performed.
