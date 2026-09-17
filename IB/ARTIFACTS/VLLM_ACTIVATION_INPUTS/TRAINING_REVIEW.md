# Training-friendliness review — plan v2

Overall verdict: **pass**.

Reviewed the revised `PLAN.md` and `PLAN.tressoir.md` together, including the physical-row HF contract, calibration lifecycle, shape-compatible controls and the other review's cache/install clarifications. This is approval of the plan's training design and scope; no intervention implementation, GPU derivative check or quality experiment has run during this review.

## Findings resolved

- **T1 resolved:** pre-collation payloads belong to logical examples; HF collation applies offsets exactly once and emits one prepared payload per physical tensor row. `decoder_forward` validates list length B and row bounds and performs no second offset conversion. Packed E-example/B=1 batches merge into one payload while existing GDN/full-attention boundaries retain isolation. Both projections now express the same rule. A two-example mapping such as offsets [0,8], positions [2] and [1] → one physical payload [2,9] is a suitable concrete acceptance case.
- **T2 resolved:** `calibrate_interventions(training_prompts)` runs after intended reader-adapter loading and before first deep encode/optimizer construction, with a deterministic training-only sample. Uncalibrated rollout fails clearly; input-only behavior needs no calibration; deep checkpoints restore their scales/provenance. This makes sample ownership and resume behavior explicit rather than deriving initialization silently from an inference request.
- **T3 resolved:** shuffled controls permute whole bundles only within matching row-count/width/layer-set buckets, preserve recipient offsets and report coverage. Different-length controls require an explicit rerender/alignment and are not silently mixed into the matched subset.

## Final assessment

The plan preserves differentiable AC training, exact frozen AgentTrainer replay, scale/head optimizer state, checkpoint-bound adapters and payloads, sample-generation semantics, stable serialization and the existing recursive input route. New FFN heads have a concrete modest bottleneck and FP32 normalized/scaled output; the 1% calibration is honestly a starting hypothesis. Source reconciliation is an explicit prerequisite, and absent POC product snapshots are not confused with available code.

The experimental evidence is used proportionately: useful input-head preservation and content-use controls are required; skip/gold, gate LR and warmup/decay effects remain qualified. Long recipe campaigns are separate, while every supported feature still requires the focused replay/gradient/collation/serving gates in M6. No remaining training-friendliness finding or requested architecture change.
