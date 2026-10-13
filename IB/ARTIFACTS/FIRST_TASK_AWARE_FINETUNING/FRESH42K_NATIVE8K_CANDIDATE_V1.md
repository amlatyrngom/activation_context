# Fresh 8k native continuation candidate

Prepared for review only. The proposed parent is the **completed constant-floor 42k pair** (`continue32k_10k_final_lr`), chosen in `FRESH8K_NATIVE_PARENT_DECISION_V1.json`. Its supported-reference answers were 71/87 correct versus 62/87 for rewarm; this is development evidence, not an independent confirmation. The separate gentle long-context candidate's parent remains open.

Train **8,000 new questions once: 42k → 50k unique questions**. Preserve the parent's actual Adam moments at step 5250, reset only the stage clock, and finish at stage 1000 / actual Adam 6250. Keep constant encoder LR 5e-6 and reader LR 2e-6, native 2k evidence/up to four merged passages, ratio 1/16, microbatch 2/logical batch 8, original 65536 encoder budget, checkpoint threshold 1700, precision, masks, joint clipping and bare-base no-context KL. No product or existing frozen file changes.

The read-only availability pass found enough accepted data without synthesis. Exact quotas are code 2400, arXiv 960, textbook 640, financial 400, web 1600, technical web 400, documents 640, legal 480, office 480. The selector reads immutable committed arXiv surplus to cover the published pool's deficit; it never republishes or modifies the bank. It excludes all previous 42k IDs and normalized question text, plus existing held/reserved synthesis requests. Raw sources may recur. The actual selection and independent optimizer copy have **not** run.

Five private files in `bench/` provide the preparer, training runner/launcher, and final evaluator/launcher (`*fresh42k_native8k_v1*`). They reuse existing helpers, paired resumption, and the frozen training source. The material retains the exact original 400 development questions, retrieval pool and fixed 100 selection; it gets a new balanced learning sample of 32 fresh questions. The initial AC400/token-score replay must match the chosen native parent within the existing 1e-3 gate before training. Held AC400 and learning32 run every 800 questions; full/no-context controls stay frozen at initialization, with current-reader controls measured at 0%, 50%, and 100%.

The final evaluator requires the **own completed 1000-update pair**, full configuration/input/source identity, published marker, actual Adam6250, and 50k cumulative count. It replays all 400 final scores in the original microbatch shape, then saves the same 100 actual AC answers with greedy/no-thinking generation, cap160, inherited EOS plus chat end, actual IDs and stop information. Partial/best checkpoints are rejected. No reserved confirmation evaluation is included.

## Review and preparation sequence

1. Review these private candidates and the focused CPU receipt. All 32 checks passed against the exact frozen trainer, covering synthetic surplus selection, prior-question/reservation exclusion, native input/replay binding, configuration, inherited clocks and negative final-pair cases. Python syntax and both shell launchers pass. Actual full material/tokenization, inherited optimizer payload verification and GPU replay remain pending.
2. Run the preparer on **allocated CPU compute** with `--parent-arm continue32k_10k_final_lr`; it refuses login/local execution without a Slurm allocation. Review the resulting selection/independent-copy receipts, then freeze its generated design. The preparer never overwrites an existing output.
3. Train with `train_fresh42k_native8k_v1.sbatch <frozen-design-path>`; if preempted, use the same command plus `--resume`. The stage keeps its original 1000-update schedule/order and restores paired optimizer state. The launcher never automatically requeues.
4. After successful completion, run `evaluate_fresh42k_native8k_v1.sbatch <same-frozen-design-path>`. It resumes its own partial output only with unchanged identity.

Example preparer arguments, to execute only on allocated compute after review:

```bash
python bench/prepare_fresh42k_native8k_v1.py \
  --material "$QA_ROOT/model_comparison_32k_material/material_27b_32000.json" \
  --bank "$QA_ROOT/synthesis_bank_27b" \
  --continuation-design "$QA_ROOT/bench/CONTINUATION_10K_DESIGN_ORCD.json" \
  --continuation-initial-pair "$QA_ROOT/continuation_32k_initial/pair.json" \
  --reports "$QA_ROOT/reports" \
  --parent-readout "$QA_ROOT/reports/continuation_cohort100/continue32k_10k_final_lr.json" \
  --reservations "$QA_ROOT/model_comparison_32k_material/reserved_questions.json" "$QA_ROOT/model_comparison_material/reserved_questions.json" \
  --parent-arm continue32k_10k_final_lr \
  --output "$QA_ROOT/fresh42k_native8k_final_lr_material"
```

Allocation candidate: one RTX, 64GiB host, **2h training wall time** with a 6600s trainer bound, plus a separate 20min readout allocation. Prior native throughput predicts about 62min pure updates and roughly 90–110min end to end, but new bare-base reference preparation is still required. A time-limited partial is resumable and is not reported as a completed 50k run. No staging or job submission has occurred.
