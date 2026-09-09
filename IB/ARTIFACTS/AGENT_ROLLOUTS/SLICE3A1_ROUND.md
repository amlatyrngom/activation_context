# Activation context, slice 3a1: implementation handoff

Slice 3a1 and its approved autotuner extension are implemented, reviewed, validated and staged below. Agent and AC callers now obtain shared hardware configs through task-specific getters. The source remains unchanged; earlier mixed small-test quality results are preserved.

## What changed

- Stride 0 retains content and adds V separately pooled summaries. Calibrated inputs preserve source signal through the mixer.
- Continuations preserve exact text/tokens and post-cut observations. Trajectory windows use original spans, keep source tasks and exclude future QA questions. Terminal normalization uses recorded answers only.
- The trainer owns complete epochs, persistent optimizer/global steps, exact periodic evaluation, bounded inspection samples and matched AC/target checkpoints. Reports retain history and start each new report at epoch 0 while checkpoint IDs remain unique.
- Checkpoints and reports sync live every 60 seconds with resumable transfers and visible failure handling. The study model is 9B; cache reuse validates settings and payload.

The approved autotuner extension centralizes Agent and AC execution settings, validates shared/committed configs, and bounds cold synthetic tuning. Both trainers apply scoped native FLA configs; explicit checkpoint/logits settings win. [Detailed autotuner plan and completion](AUTOTUNERS_PLAN.tressoir.md).

Runtime agent compaction, genuine subagent exchanges, broader agent-training repairs, wider LoRA families and unseen-corpus QA evaluation remain outside 3a1.

## Apply to your source

The implementation is staged under [slice3a1](slice3a1/README.md). The [product patch](slice3a1/changes.patch) and cards below are exact deltas against `/source`, including the user’s prototype edits. The source itself has not been modified. Check [source hashes](slice3a1/source_manifest.json) before applying; preserve unrelated changes. The benchmark/profile scripts stay IB-only and are not part of the product patch.

## Validation

- Autotuners: independent plan/code reviews and CPU cache, native-bridge, concurrency and watchdog checks pass. Actual CPU Agent and AC lifecycles pass. On RTX PRO 6000, both kernel geometries pass numerical/gradient, padded/batched/packed and prefill-state checks; final validated profiles are staged for version control.
- Cache and cold behavior: initial fresh-GPU compilation/tuning for both geometries took 60.8 seconds on the preceding recipe. The final revision adds another long check and separately passes rebuilt profiles and public trainers. Fresh-process shared and committed-preset getters took 0.551s / 0.557s, with zero worker starts or native runtime benchmarks; covered execution includes 65,539-token backward. These are lookup timings, not trainer speedups.
- Public GPU callers: Qwen4B Agent dense/packed training and side0.8B/target4B AC eval-before-train, recursive/batched optimization, generated samples and eval-after all pass. Automatic checkpoint/logits values remain conservative heuristics, with explicit settings preserved. [Autotuner validation record](slice3a1/autotuner_validation_summary.json) contains revisions, coverage, evidence and limits. The temporary GPU was torn down after scoped synchronization.
- CPU: focused model/data/sync fixtures passed; real tiny-model integration passed two epochs plus another call (22 total steps), 10 teacher-cache hits, persistent optimizer and fresh paired reload. Two existing dataset tests passed. Report-origin/reference/annotation and revised smoke-gate probes passed.
- GPU: exact 8,192/16,384/32,768-token parts and full depth-2 backward/optimizer at 71,999 teacher tokens / 512 targets passed. Two-epoch synthetic smoke completed 20 steps, all ten reporting boundaries per epoch and four bounded teacher/student sample pairs at baseline and each epoch end. Initial reload failed due to fresh PEFT dropout; corrected GPU reload is bitwise exact, including target-adapter weights.
- Real data: all three kinds completed two epochs / ten updates, matching eight held items, zero dropped training/evaluation items, finite losses, four sample pairs at each boundary and delivered epoch checkpoint pairs. Compaction trained on 33 items, trajectory-QA on 40, RAG-QA on 34. Fresh QA studies used the 9B model. Source-faithful generation reported its compaction candidate shortfall (42/48 items generated).
- Quality and actual pytest outcomes: compaction and trajectory-QA failed the preceding strict matched-adapter KL assertion; RAG-QA passed. All are small smoke tests, not the larger 2,000/200 experiments. The staged test now persists quality comparisons and gates finite/count-matched mechanics. That test/report-only change passed a focused fixture and independent review; the full three-test GPU suite was not repeated afterward. Original failures remain recorded.
- Sync: live epoch weights arrived locally while training continued. Final reports, JUnit and every expected AC/side/target checkpoint file were verified locally; final sync completed. ac-fp4-probe is paused with disk retained.
- Limits: small and synthetic checks do not establish general AC quality. QA evaluates disjoint questions on familiar source contexts. Custom-editor visual inspection is unavailable. Independent exact patch/card/source/projection checks and both Markdown checkers pass; source application remains pending.

| Real-data smoke | Train / held | Initial AC KL | Final AC KL | Final empty-context KL | AC advantage (positive is better) |
| --- | ---: | ---: | ---: | ---: | ---: |
| compaction | 33 / 8 | 0.4303 | 0.3594 | 0.3553 | -0.0041 |
| traj_qa | 40 / 8 | 0.6171 | 0.4496 | 0.4454 | -0.0042 |
| rag_qa | 34 / 8 | 0.5350 | 0.4493 | 0.4547 | +0.0055 |

Final comparisons use the same trained target adapter. These are small held sets, not a quality guarantee. [Validation record](slice3a1/validation_summary.json) includes mechanics checks, checkpoint delivery and actual pytest outcomes.


| Synthetic GPU case | Teacher tokens | Actual decoder lengths, children included | Target tokens | Peak allocated GiB | Whole call seconds |
| --- | ---: | --- | ---: | ---: | ---: |
| part_8192 | 8,208 | 8,704 | 6 | 13.5 | 60.2 |
| part_16384 | 16,400 | 17,408 | 6 | 16.3 | 8.9 |
| part_32768 | 32,784 | 34,816 | 6 | 19.8 | 10.6 |
| retained_72k_depth2 | 71,999 | 34,816, 36,196 | 512 | 22.6 | 17.8 |

These repeated-text fixtures prove finite backward/optimizer behavior and memory at the measured settings (side 0.8B, target 4B, stride 0, checkpointing forced on). They do not establish learning quality. The two-epoch synthetic report begins at cumulative epoch 4 because it followed four capacity updates; that recorded one-off report remains unchanged. A later general report-origin correction makes new reports start at 0 while preserving checkpoint identity.

The actual real-data checks used `uv run pytest activation/tests/test_basic_agent_ac_training.py --gpu --slow -s`, first with `-x` after the corrected GPU reload, then with `-k 'not test_ac_compaction'` to run both QA kinds after the compaction quality failure. The later smoke-gate change received a focused test/report check; it was not presented as a fresh all-three pytest pass.

Evidence: [CPU lifecycle log](../../TMP/SLICE3A1/implementation/cpu_validation_v3.log), [GPU capacity and initial reload](../../TMP/SLICE3A1/implementation/gpu_validation.log), [corrected reload and compaction](../../TMP/SLICE3A1/implementation/gpu_final_validation.log), [QA run](../../TMP/SLICE3A1/implementation/gpu_qa_tests.log), [smoke-gate check](../../TMP/SLICE3A1/implementation/review/smoke_diagnostic_probe.json), and [independent quality review](../../TMP/SLICE3A1/implementation/compaction_quality_review.md).

## Review and differences from the plan

The model/data and trainer/sync reviews passed after genuine fixes to suffix/template limits, source-task deduplication, observed-tool-call filtering, incomplete checkpoint rejection, adapter dropout after reload, repeated-call reports and sync failure handling. Caller review found answer scoring must inspect the submitted argument; that was corrected. Small follow-ups added cache payload integrity, explicit approximate-FLOP labeling, complete reference reporting through eval(), and typed AC/test signatures. The final exact-diff review is recorded below.

Independent autotuner plan and implementation reviews pass after genuine fixes to callback locking, bounded alternate-candidate recovery and local-only model metadata. Final GPU/profile review and exact handoff checks are recorded in [final validation review](../../TMP/AGENT_ROLLOUTS/autotuners/FINAL_VALIDATION_REVIEW.md) and [final handoff review](../../TMP/SLICE3A1/implementation/autotuner_final_handoff_review.md). The two committed profiles match the final code recipe. The old fuzzy helper was removed and its ten JSON files moved unchanged to unvalidated legacy seeds.

Model/data, trainer/sync, callers, follow-up report/API and revised smoke-diagnostic reviews pass. Independent quality review found no indicated implementation defect behind the small compaction gap. Preserve all measured comparisons and original assertion failures; do not substitute initial-adapter references or tune to force a pass. The exact handoff audit passed scratch patch application, all 47 complete cards, unchanged source hashes and paired milestone agreement; two prose-spacing nits were corrected. Final mechanical checks are repeated after status regeneration.

## Patch

The human projection contains every exact per-file diff. The same product delta is slice3a1/changes.patch; source_manifest.json verifies both sides.
