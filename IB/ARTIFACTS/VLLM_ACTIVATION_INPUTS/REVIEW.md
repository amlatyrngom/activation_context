# Deep intervention plan — reviewed handoff

Both independent final verdicts are **pass**, with no unresolved findings. These are reviews of the implementation plan and its source evidence; they do not certify a feature implementation.

| Reviewer | Scope | Final report |
|---|---|---|
| correctness_review | Exact vLLM source seams, request lifetime/cache identity, HF semantics, deployment and local integration | [Correctness: pass](CORRECTNESS_REVIEW.md) |
| training_review | Differentiability, checkpoint/adapter binding, packed inputs, frozen replay, heads/scales and ongoing POC evidence | [Training friendliness: pass](TRAINING_REVIEW.md) |

The reviewers independently inspected the paired plan and the available source. The initial round requested corrections; both checked the revision. The solver independently accepted and classified all five findings below. No reviewer requested an architecture expansion.

| Finding | Solver classification | Incorporated correction |
|---|---|---|
| C1 — Mixed token/embed mask omitted from cache identity | genuine | Hash the mask for every mixed block, including blocks before the first delta and mixed requests without interventions; test changed-mask misses |
| C2 — Offline helper propagation insufficiently specific | cheap-nit | Name the actual inherited submission helpers, rendered ordering and sampling-child association |
| T1 — Logical examples confused with physical HF batch rows | genuine | Collation emits one payload per physical B row, applying offsets once; explicit E=2/B=1 mapping and gradient/causality gate |
| T2 — Fresh-model calibration lifecycle unstated | cheap-nit | Explicit training-only calibration after reader loading and before first deep encoding/optimizer creation; save provenance and never recalibrate on load |
| T3 — Shuffled controls could have incompatible row counts | cheap-nit | Match row count, width and layer-set buckets, retain recipient positions and report coverage; otherwise explicitly rerender |

The final plan also makes installation concrete: immutable fork Git dependency matching the gitlink, pinned compatible native-wheel reuse or exact-source build, and a clean environment capability/SHA check while preserving the IB upload exclusion.

## Completed preparation and validation

- Created GitHub fork `amlatyrngom/vllm` with the existing authenticated account and pushed `ac-interventions-v0.28.0`. Remote branch and local submodule both resolve to upstream v0.28.0 commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`.
- Committed the preexisting local product/dependency work as `0a77891` (53 paths); registered the submodule as `13f19d9`. No root-repository push or upstream PR was made.
- `git apply --check IB/ARTIFACTS/AGENT_AC_PRETRAINING/codesign/changes.patch` passed against that product snapshot. The patch remains unapplied; M0 reconciles it before new carrier changes.
- Checked the two projections against source, retained independent final reports, validated local document links and ran the Tressoir Markdown checker. The editor's visual rendering was not inspected in this environment.
- [Source manifest](SOURCE_MANIFEST.json) records SHA-256 hashes of the local/pinned source referenced during handoff. [Training source findings](TRAINING_SOURCE_FINDINGS.md) distinguishes available co-design code from absent experimental source tags. POC interpretation is frozen at the cutoff stated in PLAN.md, not represented as final results of ongoing runs.

No intervention product code, build, GPU parity test, gradient test or quality experiment ran as part of this planning turn. Existing snapshot checks and earlier probe/POC results do not certify this future implementation. M6 lists the required acceptance gates; long campaigns require a separate budget. Active POC jobs and user TASK content were untouched.

Next: approve or revise the [human plan](PLAN.tressoir.md). The [agent plan](PLAN.md) remains the detailed implementation authority.
