# Shared continuation: 32,000 questions → 10,000 fresh questions

ORCD CPU job **23498249** completed preparation in 95 seconds. Both continuation
arms share one immutable dataset and one derived copy of the preselected final
32,000-question pair. Original P2-initialized 2B encoder / 9B reader weights and
the original optimizer remain untouched.

The paused 53,000-question 27B bank has sufficient eligible data after excluding
all parent training IDs and normalized questions, plus held/reserved questions
and their whole synthesis requests. Existing question-only reservation lists
were read; confirmation-answer files were not opened. Questions from other
siblings of earlier training requests remain eligible under the existing split
policy. No model answers or losses guided selection.

| Source family | Selected | Eligible |
|---|---:|---:|
| Code | 3,000 | 6,120 |
| Math/arXiv | 1,200 | 2,100 |
| Math/textbook | 800 | 1,572 |
| Financial | 500 | 1,003 |
| Web | 2,000 | 4,080 |
| Technical web | 500 | 1,020 |
| Documents | 800 | 1,579 |
| Legal | 600 | 1,174 |
| Office | 600 | 1,215 |

The original 400 development questions, cohort assignments and retrieval pool
are unchanged. Required new source evidence comes from each fresh example's own
units. The learning panel contains 32 fixed questions from the fresh training
set. Family quotas, source/answer-length strata and fixed ordering reuse the
existing selection helpers. Both arms must use this same material and seed.

The derived pair contains independent full AC and reader directory copies.
All weight bytes are unchanged. All **615 Adam parameter states**, including
moments and per-parameter step **4000**, are byte-exact; parameter groups are
unchanged. Only the optimizer file's trainer counters `updates_done` and
`epoch_number` are reset to zero for the new stage. This is preserved Adam,
not a fresh optimizer. The AC modules retain their saved epoch metadata; the
runner must set the loaded `ac.epoch_number=0`, validate exact optimizer layout,
and enable optimizer restoration. Stage learning rates are applied at runtime.

The copied `parent_panel.json` and complete parent readout fingerprints support
stage-zero checks. Identical inputs and weights are required for both arms;
H200-versus-RTX numerical differences remain a diagnostic measurement.

[ORCD preparation receipt](CONTINUATION_10K_PREPARATION.json) records all source,
copy and optimizer hashes plus helper versions. Material SHA256:
`4b67c74e365dc46093efa2d0e89bb2770b530e572d88e29035b07702b51d2585`.

CSAIL destinations:

- `/data/scratch/ngom/activation/taskaware/continuation_10k_material/material_27b_10000.json`
- `/data/scratch/ngom/activation/taskaware/continuation_32k_initial/pair.json`

The 7,686,696,886-byte relay completed. `pair.orcd.json` and the original
preparation receipt remain intact; the relocated pair changes only its three
runtime checkpoint/panel paths and `provenance.material`. Its SHA256 is
`cd0ed98de2c22e7b20865971dff7f4da9f587b8b13e88a880685de3a511c92ec`.
CSAIL CPU verification **2143414** completed successfully in **62 seconds**; all
19 received files matched SHA256 and size, and the pair changes were verified as
paths only. The [relocation receipt](CONTINUATION_10K_CSAIL_RELOCATION.json) is
also present at `continuation_32k_initial/CSAIL_RELOCATION.json`. Both shared
inputs are ready for GPU consumers.

Focused CPU fixtures passed for late reserved siblings, normalized question/ID
exclusion, exact quotas/reproducible ordering, tensor-byte comparison and a full
small derived-pair round trip retaining unequal Adam parameter steps. No GPU
jobs, synthesis, pruning, or changes to existing material/reservations were made
by this preparation task.
