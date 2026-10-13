# Compaction teacher review and bounded preparation handoff

September 24, 2026. Scope: the authorized 36-hour trajectory-QA/compaction campaign. Read-only review expanded, at the root agent's request, to a focused trainer implementation. Product ownership in this slice: `activation/ac_model/ac_model_training.py` only. No remote calls or new mainline test files.

**Initial verdict: incorrect-or-missing.** The existing model, paired-history representation and KL loss are sufficient, but the eager trainer cannot safely preserve one teacher across interrupted large compaction runs. The implementation below addresses the trainer findings. Root review and one representative GPU proof remain before a large launch.

## Findings and dispositions

1. **Genuine — resumed KL silently changes its teacher.** `train()` clears cached targets and previously recomputed them from the resumed, updated reader. Loading `teacher_targets.pt` before `train()` does not survive that clear; partial checkpoints did not contain targets. The new optional mode persists the original reader adapter and rejects missing, changed or incompatible teacher provenance on resume.
2. **Genuine — eager target storage is too large and postpones every update.** With top-k 32, each target position uses 264 bytes: 32 int32 IDs, 32 FP32 log probabilities, remainder mass and gold-token log probability. At 50M supervised tokens that is approximately 13.2 GB per reference, or 26.4 GB for full-context teacher plus base no-context reference, before examples, Python objects, models and optimizer. Complete checkpoints and latest aliases additionally copied the cache. The new mode holds one bounded block plus fixed reporting targets and persists a small adapter reference instead of all training targets.
3. **Genuine — requested history lengths are not enforced by the existing public builder.** `_compaction_item` clamps its estimated threshold to available history; complete turns may overshoot it. Repeated document cycling can yield identical cuts with different item IDs. A count of 100k outputs is not evidence of 100k distinct useful examples. Parent owns source inventory and builder adjustments: split whole source-qualified roots, retain actual cut IDs/token counts, prefer distinct histories, and report real length distributions. Existing `item_origin` correctly groups public compaction by source/document and recorded runs by root; trajectory-QA currently requires its separate whole-root split.
4. **Genuine — strict resume also needs the penalty random state.** The no-context penalty's sampler formerly restarted at the seed after a requeue. Optimizer metadata now stores/restores this state. Bounded mode rejects missing optimizer state or incompatible optimizer groups rather than silently restarting moments.
5. **Measured opportunity, deferred — KL microbatching.** The configuration explicitly forbids `micro_batch_examples > 1` for KL, although teacher-only batching code exists. Keep microbatch 1 for the first launch. Add the shared padded KL path only after profiling and independent unequal-length loss/gradient checks.
6. **Measured opportunity, deferred — duplicate output-head work.** `_store_teacher_targets` computes full-vocabulary logits/log-softmax separately for top-k and gold log probability. Fusing those computations is a small potential optimization; benchmark its contribution first. Bounded base references are still prepared for every example in each block even though the penalty is sampled on 25%; lazily preparing only selected references is another later optimization.

The complete-turn target masks are already structurally correct: one authoritative token sequence for every retained assistant output, identical target IDs at teacher/student predicting positions, no loss on tool replies, and maximal complete suffixes within the 8192 rendered-token and total guards. This does not establish that offline teacher-forced continuation predicts live task completion.

## Implemented API and behavior

```python
config = ActivationContextTrainingConfig(
    loss_kind="kl",
    teacher_cache_block_examples=128,
    micro_batch_examples=1,
    examples_per_update=8,
    # Existing Golden learning rates/schedule and other run settings follow.
)
trainer.train(ac_name, training_items, reporting_data=held_items, ...)
```

`teacher_cache_block_examples=None` preserves eager behavior. A positive value enables bounded preparation for KL and/or SFT's base no-context penalty. Blocks are rounded down to whole optimizer updates, with at least one update prepared. They are internal preparation units within **one train call, epoch, optimizer and learning-rate schedule**; they must not be reported as separate data passes.

For KL, the trainer saves the reader adapter before any update, registers it as a distinct adapter on the same frozen base, and excludes it from gradients/optimizer ownership. Fixed reporting/validation targets are prepared first. Each upcoming training block then gets its full-context targets from that exact snapshot and its no-context references from the base with adapters disabled. Prior training-block targets are evicted; reporting targets stay pinned. No second 9B base is loaded.

Teacher adapter/config bytes, base model/revision, tokenizer identity/size and top-k are checked on restoration. Teacher completions after resume use this same adapter, not the updated reader. The resumed optimizer also checks the teacher identity. Missing state raises before an update.

Each complete or partial checkpoint gets `teacher_snapshot.json`, pointing relatively to `AC_MODELS/<name>/teacher_snapshots/...`. Fixed-panel targets alone are saved as `teacher_targets.pt`; training caches and base-reference caches are not copied into every checkpoint. **Preserve/export the referenced teacher snapshot directory with any resumable checkpoint.** The manifest intentionally fails if it has been lost; copying only the AC/reader pair is insufficient for this optional mode.

An arbitrary later evaluation uses:

```python
trainer.eval(ac_name, items, teacher="snapshot")
```

This regenerates bounded targets against the immutable snapshot. Existing `teacher="stored"` still requires cached targets; fixed-panel files remain compatible with `load_teacher_targets()`. `teacher="current"` retains its explicit existing meaning. No moving-reader fallback was added.

## Validation and limits

Private checks: `IB/TMP/FIRST_TASK_AWARE_FINETUNING/compaction_teacher_review/test_bounded_teacher.py`. Six real tiny-decoder checks passed in 21 seconds:

- Eager and bounded targets and final updates match exactly, while teacher parameters stay frozen and outside the optimizer.
- Cache size is bounded; only fixed-panel targets persist; recomputation after student updates still matches the original teacher.
- Interrupted resume reproduces learning rates, penalty RNG state and final weights of uninterrupted training.
- Arbitrary snapshot readouts agree after reload; corrupted/missing teacher metadata and missing optimizer state refuse.
- Resumed teacher completions route through the frozen adapter.
- SFT with bounded base-reference preparation matches eager SFT.

Five existing focused resume, checkpoint-publication and frozen-reader checks also passed (37 seconds). Syntax compilation passed. The six-check suite should be rerun after root review changes. Private pre-change file: `IB/TMP/FIRST_TASK_AWARE_FINETUNING/compaction_teacher_review/ac_model_training.before.py`.

**Remaining scale limit:** all paired examples and prepared teacher/student/penalty token lists still reside on CPU. At 25k examples with 20k teacher tokens each, the teacher IDs alone comprise 500M Python integers plus list storage, roughly 18 GB on typical 64-bit CPython; other views, positions, raw messages and model/optimizer state add substantially. Requesting 128 GB host RAM and measuring real RSS on the first material is reasonable. Do not claim constant total memory, and do not add a new streaming framework before that measurement warrants it.

**GPU gate:** run the actual 2B→9B pair through at least two updates, then a paired resume after Adam moments exist, including a representative 32k-history/8k-suffix example. Verify finite losses, identical frozen-teacher targets before/after updates and resume, teacher/reader/encoder parameter ownership, actual target masks, and peak GPU/host memory. Keep an unequal-length padded KL proof separate if that optimization is subsequently added.

Actual eligible corpus size and current length-bin yields were not measured in this review; parent owns the lightweight inventory. Historical small compaction checks establish mechanics, not the promised 50M-token supply or convergence. All 8/12/16-hour results should report distinct roots, examples, supervised assistant tokens and real history lengths.
