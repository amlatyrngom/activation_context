# Gentle context adaptation from a completed42k checkpoint

Prepared candidate only. Parent choice remains open between the completed constant-floor and rewarm10k continuations. No checkpoint/material transformation, staging, submission or training has run for this candidate.

Train on **4,000 familiar questions once** at8k evidence, starting from the chosen42k paired final and its actual Adam moments. This adds4k presentations, reaching **46k presentations while remaining42k unique questions**. It tests whether gentle adaptation improves use of the longer, independent-passage context. It is neither fresh-data continuation nor an isolated layout comparison.

- Select exactly half of every source-family quota in the existing8k long-study pool. Preserve all original32 learning questions; fill the remaining quota in the existing interleave order. Retain the exact400 development questions,32398-row retrieval corpus, fixed100 selection and evidence construction. Check selectedIDs against the parent's already-saved trainingIDs to substantiate “familiar”; no corpus audit.
- Preserve615 populated Adam states at lifetime step5250 and all group/moment bytes. Reset only optimizer outer `updates_done` and `epoch_number` to0, and set loaded AC epoch to0. Train500 stage updates at constant encoder5e-6/reader2e-6, no warmup. Fresh Adam is explicitly rejected.
- Preserve 8k evidence/up to16 passages, independent-passage layout, equal total1/16 rows,65536 encoder batching budget, original precision/masks, checkpoint threshold0, microbatch2/logical8, joint clip1 and the original bare-base no-context KL on25% of questions.
- Measure heldAC400+learning32 every400 stage questions. Frozen controls remain the actual initial42k reader; moving-reader full/no-context controls occur only at0/50/100%. Final checkpoint only. Own partial pairs resume with the original stage order,500-update schedule and lifetime Adam clock5250+savedstageupdates.
- Require the selected parent's **complete matched8k400/100 readout** before preparing real inputs. At startup compare full400 rendered-input/target/row fingerprints and fixed100 evidence/source/mask metadata to that readout. The exact referencefile hash is part of the design and resume receipt. Before the first update, reuse the trainer’s existing initial baseline to replay all400 answerNLL values and the selected100 gold-token scores against that parent; keep the existing1e-3 limit and retain any failure. Own resumes require the original passed replay receipt.

The private source is unchanged `train_ws_night_v1`. The candidate runner is a sibling of `run_qa_strong_long_v1.py`; its harness, material builder, row allocator, answer masks, published-pair resume parser, resume history trimming and initial-reference validator are unchanged. No existing run or frozen comparison changes.

## Files and validation

- `bench/prepare_gentle42k_long_v1.py`: allocated-CPU-only material selection and independent pair copy. Copies with `copy2`, verifies distinct files, hashes originals/copies, reloads the derived optimizer and checks every preserved tensor/group byte. Produces material, initial pair receipt and a **prepared**, not launchable, design.
- `bench/run_qa_gentle42k_long_v1.py`: inherited-Adam stage with strict actual615-state/5250 guard and own-resume clock guard.
- `bench/evaluate_gentle42k_long_v1.py` and `.sbatch`: own final500-update/actualAdam5750 guard, all400 final-panel NLL replay, then the same100 actual AC answers at8k. Uses the existing greedy160/no-thinking/EOS-union helper, retaining actual token IDs and caps. Its output/identity is separate and partial readouts can resume; no optimizer or checkpoint is modified. One RTX,64GiB host,45-minute ceiling.
- `bench/train_gentle42k_long_v1.sbatch`: one RTX,64GiB host,5h wall; max training duration4h45. Requires a separately reviewed/frozen design and checks source, helpers, all derived payload hashes and parent-reference hash before loading. Optional `--resume` uses the existing paired protocol.

Thirty focused synthetic CPU checks passed against the exact frozen trainer `38ca9885…`: family quotas/learning retention; input immutability; incorrect shape/group/clock/missing-state rejection; tensor-exact serialization; identical next Adam update before/after outer-clock reset; different next update with fresh Adam; constant learning rates; full actual constructor/config equality; and unchanged builder/resume helper ASTs. Twenty-one additional focused protocol checks cover initial numerical replay, wrong gold/order, actual finalAdam5750 and rejection of incomplete/foreign/wrong-config/missing-replay/wrong-publication finals. Root’s review removed an unnecessary cross-arm design comparison: alternate-parent readouts can coexist, while each keeps its own resume identity and exact parent-input binding. Syntax checks passed. The fixtures use synthetic small parameters and explicitly assigned lifetime clocks; they do **not** claim that the real parent optimizer has already been inspected. Full receipts/hashes: `GENTLE42K_LONG_CANDIDATE_V1.json`.

## Preparation after parent selection

Run the following command **inside an allocated CPU compute job**, with frozen source on `PYTHONPATH`; it refuses execution without `SLURM_JOB_ID`. Parent can instead be `continue32k_10k_rewarm` once its own matched8k readout is complete. Use a fresh output directory. The preparer does not submit training.

```bash
QA_ROOT="$HOME/orcd/pool/activation/taskaware"
QA_PARENT=continue32k_10k_final_lr
"$HOME/activation_venv/bin/python" "$QA_ROOT/bench/prepare_gentle42k_long_v1.py" \
  --parent-arm "$QA_PARENT" \
  --material "$QA_ROOT/strong_long_adaptation_8k_material/material.json" \
  --continuation-design "$QA_ROOT/bench/CONTINUATION_10K_DESIGN_ORCD.json" \
  --continuation-initial-pair "$QA_ROOT/continuation_32k_initial/pair.json" \
  --reports "$QA_ROOT/reports" \
  --parent-readout "$QA_ROOT/reports/continuation_long_cohort100/${QA_PARENT}__8k/result.json" \
  --output "$QA_ROOT/gentle42k_${QA_PARENT}_familiar4k_preparation"
```

After inspecting the compute receipt and choosing the parent, root can freeze the generated design and pass its path to the launcher. Expected compute preparation: copy/hash about7GB paired weights+optimizer, requiring roughly24–32GiB host RAM; no GPU. Runtime planning is approximately3.1h pure training using the measured22.6s/8-question original-budget path, plus reduced reporting/model preparation/save time. Five-hour allocation is a margin, not a throughput guarantee. The current stronger8k adaptation continues independently.

Own final8k400NLL plus the same100 actual AC answers is prepared in the sibling evaluator. Pass the same frozen design path to its launcher after successful training.32k NLL remains optional after reviewing8k quality; no32k evaluator or job is added here. Parent selection and scheduling remain root decisions after the current actual-answer comparisons.
