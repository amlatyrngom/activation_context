# Fixed 100-question development readout

Prepared for source-grounded review of the 16,000-question experiments. This is development analysis; this worker reads no confirmation questions or scores.

The frozen selection contains 50 questions from each generator cohort, chosen by a salted SHA-256 of the immutable item ID within source-family quotas. Each half has 15 code, 6 arXiv math, 4 textbook math, 10 web, 4 documents, 3 legal and 3 office questions. The remaining five are 2 financial/3 technical-web for the 9B cohort and 3 financial/2 technical-web for the 27B cohort. No losses or answers influence membership. Selected IDs retain their original saved 400-question reporting order.

Each invocation loads one fixed paired checkpoint in a fresh process. Available conditions are:

- `frozen_p2`: original P2 reader, full context and no context.
- `frozen_p4`: original P4 reader, full context and no context.
- `oracle_9b_16k_r16`, `oracle_9b_16k_r8`, `oracle_27b_16k_r16`, `oracle_27b_16k_r8`: that arm’s final AC and reader pair.
- `encoder4b_oracle27b_16k_r16`: the 4B-encoder arm’s final AC and reader pair.

Only the requested trained arm must have completed 16,000 questions × 1 epoch and published its matching pair. Frozen-reader conditions do not require a completed training run. P4 and P2 have different pretrained reader adapters as well as different encoders; this does not isolate encoder size.

The worker scores all 400 original development questions with microbatch 2 and the arm’s actual compression ratio before selecting the 100 scores. It preserves the original length-sorted batch partners. When the corresponding initial/final panel exists, it records per-question NLL recheck differences rather than claiming numerical agreement in advance. Full-context evidence, targets and answer masks must match across conditions; AC fingerprints may differ with ratio and encoder.

The output retains exact required sources, complete evidence, per-token gold losses, complete decoded responses and text after the first answer wrapper. Greedy generation uses thinking disabled and a 160-new-token cap. The private generation wrapper explicitly adds the existing chat-end token to inherited EOS IDs without mutating shared configuration. Actual returned token IDs, special-token text, EOS termination and token-budget flags are recorded. Visible retokenized length remains separately labeled. Gold-prefix NLL and independently generated answers remain distinct measurements.

The prepared ORCD launcher runs one condition on one RTX GPU, bounded to 45 minutes:

```bash
sbatch --output="$HOME/orcd/pool/activation/taskaware/logs/model_cohort_%j.out" \
  "$HOME/orcd/pool/activation/taskaware/bench/model_comparison_cohort.sbatch" frozen_p2
```

The conditions are now submitted in dependent, slot-reusing chains; see [queue receipt](EVALUATION_QUEUE.json). The command above is the standalone invocation pattern. Replace the final argument with another listed condition. The launcher uses the immutable `train_ws_night_v1` product snapshot, the pinned `run_qa_confirmation_v3.py` input builder and the original `run_qa_encoder4b_v1.py` harness. Required sibling helpers are `evaluate_model_confirmation.py`, `evaluate_long_cross.py` and `evaluate_qualitative_cohort.py`; import of the first helper calls no confirmation evaluation. Ship `MODEL_COMPARISON_DESIGN.json` and `MODEL_COMPARISON_COHORT100_SELECTION.json` into the remote bench folder as well.

Outputs are `reports/model_comparison_cohort100/<condition>.json` and `<condition>_status.json`. A repeated identical invocation retains completed scores/responses and fills only missing work; changed checkpoint/input/worker identities reject. One writer per condition is required. Checkpoint SHA-256s are computed on the allocated node. The evaluator performs no training and uses no native benchmark questions. It adds no product files.

CPU validation receipt: `IB/TMP/FIRST_TASK_AWARE_FINETUNING/model_comparison_cohort_checks/cpu_check.json`. It checks real material selection and held order, equality of the original and fixed builders on all 400 inputs, common P2/P4 evidence and targets, both row budgets, and the requested-final-only guard. No model weights are loaded by this check. Runtime NLL agreement and greedy outputs were subsequently measured for every condition, as recorded below.

Validation passed in 279 seconds. All conditions share input fingerprint `200626d81e8720151c112f5a86d175275895928ecf0d44d19cae2ce2b6df4dcc`. The 400 items contain 811,299 encoder tokens: P2/P4 at 1/16 allocate 50,898 rows; P2 at 1/8 allocates 101,590. Python compilation and shell syntax checks also passed.

Deployed SHA-256s:

| File | SHA-256 |
|---|---|
| `bench/evaluate_model_comparison_cohort.py` | `167aa74528eeb642f3b0318dc43194ebd2b2ad30d30cb6a07cacab192b003f90` |
| `bench/model_comparison_cohort.sbatch` | `2a55bfa1d4eb8cb9e7cf1e84d1b95ce2f210f3d313e68340c2d900e649ffeaae` |
| `MODEL_COMPARISON_COHORT100_SELECTION.json` | `469320c02db9347d4d93b99c94003f9eb40d738d03ee272ff83a0129fda92051` |

All seven conditions were submitted. The six conditions share one serial slot; the4B final readout follows its own training. These reuse existing GPU slots and keep four active autonomous GPUs maximum. The reserved confirmation is evaluated by its separate fixed-five-arm worker.

Sep22 05:33UTC: the chat-end correction is deployed for all seven pending conditions, after focused real tiny-model checks/review and a successful real9B argument A/B. The unchanged pinned trainer still computes NLL. Frozen selection and all input identities are unchanged. No readout had begun before deployment. See [decoder evidence](EVALUATION_CHAT_STOPPING.md).

The serial chain was reordered at05:42UTC to expose semantic evidence earlier: after27B/r16 finishes, decode frozenP2,27B/r16,then27B/r8 (also waiting for its final); next run9B/r16 training, followed by both9B readouts and frozenP4. The4B and long-cross chains retain their slots. The dependency graph and final guards were reviewed before editing held pending jobs.

Sep22 06:30UTC: frozenP2 and both27B-supervised final readouts are complete. Every original400 per-question answer-NLL replay matches its saved panel exactly. All400 responses across these four views have source-grounded annotations in [the comparison report](MODEL_COMPARISON_COHORT100.tressoir.html), retaining all100 questions. On the same87 supported references: full78, no-context15, 1/16AC66, 1/8AC51 fully supported responses. Format completion is reported separately. Remaining9B/P4/4B readouts follow training.

Sep22 09:35UTC: both9B final readouts and frozenP4 are now complete, again with exactly matching original400 NLL. All800 available responses are source-reviewed. Same87-reference correct counts: P2full78/noctx15,27B-r16AC66/r8AC51,9B-r16AC52/r8AC55,P4full80/noctx18. The4B final remains pending. A prose-only cohort count was corrected: review halves contain both generators, so 9B-generated supported counts are29/42 versus36/42 for9B-r16 versus27B-r16, not the earlier first-review-half27/43 versus34/43. Overall counts and generated scores were unaffected.


Sep22 10:35UTC: all nine views are complete and all900responses source-reviewed. The4B final readout completed0:0 in4m46s, with per-question NLL replay difference exactly0 on the original400. Same87supported references:4BAC60correct/7partial/20incorrect versus2B27B/r16AC66correct/5partial/16incorrect.4B gains6andloses12 fullycorrectanswers against2B. FrozenP4fullgets80correct. All100retainNLLandvisibleannotations;4Bhas100parsedanswercalls,99EOSterminations,oneoutputcap. No new source-reference concerns.

The4B final's400-question mean NLL(.58350) is lower than2B27B/r16(.60326), but the exploratory paired-question bootstrap interval for their difference crosseszero:−.04793to+.00774. One seed, different pretrained pairs and these development questions cannot establish an encoder-capacity advantage. The requested32krun retains2B. The HTML's prior standalone interaction/browser checks are reused because this is a content-only completion; source/selection/common-input/runtime-score assertions pass on the new results.
