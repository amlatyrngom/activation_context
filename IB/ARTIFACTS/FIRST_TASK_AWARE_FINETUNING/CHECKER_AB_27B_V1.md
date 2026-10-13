# Brief-check 27B comparison — completed

**Retain production A.** ORCD job **23421224** completed the fixed 23-case comparison. Asking for a brief check before the verdict made this result worse: it introduced unsupported justifications as well as two truncated outputs. Increasing the token limit alone would not fix the semantic regressions. No production worker or bank data was changed.

| Condition | Correct / 20 scored | False accepts | False rejects | Invalid outputs | Measured call time | Output tokens |
|---|---:|---:|---:|---:|---:|---:|
| A: current boolean checker | 19 | 1 | 0 | 0 | 3.99 s | 203 |
| B: brief check, then verdict | 15 | 3 | 0 | 2 | 9.13 s | 1,728 |

The three ambiguous cases remain unscored. Invalid outputs are separate from false rejects. Call timing excludes warm-ups and 217.3 seconds of model startup; total job execution recorded by the worker was 233.7 seconds. B took about 2.3× the measured call time on this small set. These are diagnostic counts and timings, not population accuracy or production throughput estimates.

## Which judgments changed

All changed decisions and their raw outputs were inspected. ID prefixes resolve to full cases in the saved JSON files.

| Case | A → B | Finding |
|---|---|---|
| DynaResource financing `890b026dc1e6` | accept → invalid | B computes $2.5M + $1.52M correctly and emits `supported: true`, but reaches 128 tokens before the closing brace. This is truncation, not a wrong judgment. |
| Cell-map compatibility `ee5bf06b11ac` | accept → invalid | B spends the budget quoting escaped LaTeX and stops at 128 tokens mid-string, before a verdict. |
| Cultural trust `9eee3d85aad3` | reject → accept | B says the text links private meetings to hierarchical cultures. The source instead distinguishes “some cultures” and “other cultures”; B invents the missing relationship. Normal stop at 66 tokens. |
| Missing template definition `…591:97d62bae…` | reject → accept | B asserts that the first two template arguments mean key/value capacity, although their definition is absent. It repeats the unsupported interpretation. Normal stop at 74 tokens. |
| Julia/Python comparison `9f44c605aa6d` | reject → accept | B relies on the article's broad Python claim. This case was already marked a source-quality/qualification concern and remains unscored. |

The persistent algebra failure did **not** change: **both A and B accept** `fb62521293af…`. The source denominator is `sqrt(n·Δ³) = n^(1/2)·Δ^(3/2)`, but B explicitly writes that it equals `(n·Δ)^(3/2)`. This completed normally in 83 tokens. Its check field repeats the error instead of independently testing it.

Both conditions correctly reject the earlier wrong revenue-direction and wrong JavaScript-branch examples. Both accept the unscored Read the Docs proposal-status and PEP-framing examples; no new conclusion is assigned to those ambiguous cases.

## What explains the failures

**The 128-token allowance explains exactly the two malformed outputs.** Both have `finish_reason=length`; neither failure is a field-order problem. A 40-word request does not reliably bound tokenizer length, especially for quoted LaTeX. The longest measured prompt was only 3,661 tokens against an 8,192-token context, so evidence/context truncation is not implicated. Even if both incomplete answers were optimistically rescued, B would reach only **17/20**, still below A's 19/20. No repair was applied to the measured results.

**Completed checks show a semantic problem beyond formatting.** The three false accepts all stop normally below the cap. Their text rationalizes the proposed answer: wrong exponent expansion, an invented cross-group link, and an assumed dependency definition. The available output does not establish why the prompt caused those regressions, but it rules out the output limit as their immediate cause.

**A reproduces one of the two initial 27B failures, not both.** The algebra error remains accepted; the office relationship now gets rejected despite having entered the original bank. The original bank's checker responses were not retained, so that earlier acceptance cannot be reconstructed more precisely from its receipt.

There is also direct replay instability within this experiment: A rejects the supported serotonin case `7e325519091a…` during its singleton warm-up, then accepts it in the measured 11-case batch. It has the same 308 prompt tokens, model, question/evidence, temperature zero and seed; caching is disabled and all measured cache-hit counts are zero. B accepts it in both calls. This supports caution about single replay decisions. Batch shape, scheduling or numerical execution effects are plausible; the run does not isolate the cause. It does not justify attributing all of B's semantic failures to batching.

## Recommendation

Keep the deployed boolean checker and preserve the current bank. Do not adopt B, add another checking stage, repair measured outputs after the fact, or increase B's cap as the presumed semantic fix.

**Prioritize the paired training and layout results.** A low-reasoning comparison with a matched A replay is an **optional next diagnostic, not a queued or prepared job**. The persistent algebra error gives it a concrete purpose if the coordinator later chooses to run it.

That optional diagnostic can reuse these same 23 cases and final boolean schema, with a small explicit reasoning budget and recorded tokens, finish reasons and time. Keep the three ambiguous cases unscored. The useful test is whether it rejects the algebra error while preserving A's supported decisions and avoiding new false accepts or truncation. Otherwise retain A and record the limitation. No new data collection or wider validation campaign is needed.

## Exact setup and evidence

The 23 frozen pairs comprise the initial mixed 20-pair 27B sample plus three earlier diagnostic cases: **15 supported, five unsupported, three unscored**. The missing-definition case receives only the original header, with no supplemental Breakpad definition. Manual expectations/reasons are never passed to the model.

A copies the deployed prompt verbatim, with a 48-token output limit. B changes only the final output instruction to request a brief independent calculation/exact evidence field before `supported`, with a 128-token limit. One pinned `Qwen/Qwen3.5-27B` dynamic-FP8 engine serves both: revision `fc05daec18b0a78c049392ed2e771dde82bdf654`, thinking off, temperature/presence penalty zero, 64 maximum sequences. Two halves use A→B then B→A; each format warms up first on the same shortest existing case. Prefix caching is disabled equally.

- Raw outputs, recipes, warm-ups and call timing: `IB/TMP/SYNC/TASK_AWARE_QA/checker_ab_27b_v1/results.json`, SHA-256 `ccfaaca69b3410462e604462587e9de3ee57602f02a593be82c0b74cc4620e00`.
- Frozen inputs: `IB/TMP/FIRST_TASK_AWARE_FINETUNING/checker_ab_27b_v1/cases_23.jsonl`, SHA-256 `fc7b7dcfea8bff0aa12a56c102f1064938316e37883fddaecb95e0c043e5bcf4`.
- Worker: `bench/run_checker_ab_27b_v1.py`, SHA-256 `9b4d1aaf3eea80b236fa2e485f1e56e381210649b26cc03c4a519a3a4262eab9`; allocation: `bench/checker_ab_27b_v1.sbatch` (one RTX PRO, 20-minute limit).
- CPU checks of prompt parity, input/label isolation, parsing and publication remain in the input folder's `cpu_checks.json`.
