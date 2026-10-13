# Completed 42k checkpoint on matched 8k evidence

Prepared one development readout of `continue32k_10k_final_lr`, the completed 32k → fresh 10k constant-LR continuation. This asks whether that checkpoint is already a stronger long-context starting point. It is separate from the running 32k → fresh 8k adaptation and performs no training.

- Use the existing frozen strong-long builder/material and original fixed 100 IDs. Score the original 400 in saved order, microbatch 2; then generate the same 100 AC answers, 50 per generator.
- Use 8,192 evidence tokens, up to 16 whole passage parts, equal merged row budget at 1/16, and original encoder batch budget 65,536. Preserve original precision settings.
- Before scoring, match all 400 shared/AC/row fingerprints and actual token/row/span records to completed `initial32k__8k`; match every selected question, gold mask, required source and full evidence record. Preserve required sources whole.
- Reuse the existing completed-continuation guard: own published final pair, 1,250 stage updates, 10,000 questions once, original 32k parent, full config/source/order checks and inherited-Adam provenance. Bind the exact paired payloads to the already-completed native-2k readout. Selected/partial/foreign pairs fail.
- Report AC, full-context and no-context NLL from the **42k paired reader**, including per-item values and both generator cohorts. Decode AC greedily with thinking off, 160-token cap, inherited EOS plus chat EOT; retain actual token IDs and termination details.
- Partial decoding can resume only with the same identity, payloads, inputs and generation settings. Completed scores are reused. There is no same-weight 8k training panel to replay; input matching to the 32k reference must not be described as NLL reproduction.

Worker: `bench/evaluate_continuation_long_8k_v1.py`; launcher: same basename `.sbatch`, one RTX PRO6000, 64 GiB host memory, eight CPU cores, 45-minute wall bound and 35-minute inference budget. Expected useful runtime is around 15 minutes; unmeasured for this exact checkpoint.

Output: `reports/continuation_long_cohort100/continue32k_10k_final_lr__8k/result.json`, synced locally under `IB/TMP/SYNC/TASK_AWARE_QA/continuation_long_cohort100/`.

Focused checks passed against actual completed receipts: valid input/identity acceptance plus 16 rejection cases for incomplete/foreign checkpoints, wrong contexts/layout/budget/source/order, changed rows/evidence/gold/masks. Python and shell syntax passed. Runtime still must verify actual payloads, model configuration and rebuilt inputs before GPU inference. Receipt and hashes: `CONTINUATION_LONG_8K_PREPARATION_V1.json`. No confirmation data, new 32k cell, staging or submission was performed.
