# Native 50k continuation: fixed-100 source review

The completed **50,000 unique questions × 1** pair gives **70/87 fully correct**, versus **71/87 at 42k** and **63/87 at 32k**. It has 7 partial and 10 incorrect answers; 42k has 5 partial and 11 incorrect. This is no clear semantic improvement over 42k on the inspected development panel, despite five concrete repairs.

The run adds 8,000 truly fresh questions once to the 32k + 10k constant-rate history, preserving Adam state and the final learning rates. Only its completed final encoder/reader pair was evaluated; no interim checkpoint was selected.

| Supported cohort | 32k | 42k constant | 50k constant | 50k gains / losses vs 42k |
|---|---:|---:|---:|---:|
| 9B-generated, 42 questions | 34 | 37 | 36 | 2 / 3 |
| 27B-generated, 45 questions | 29 | 34 | 34 | 3 / 3 |

Against 42k, 65 are correct in both, 5 only at 50k, 6 only at 42k, and 11 in neither. Gains are indices 8, 18, 72, 74 and 94: the former-employee exception, batch-size divisor, mesh list, speed units and exact tokenizer output. Losses are 20, 22, 24, 35, 87 and 99: a date, country intersection, repeated-window condition, omitted advantages attribute, unsupported egg-hatching rationale and wrong mineral. Three losses become partial and three incorrect. Against 32k, 50k has 11 gains and 4 losses, with 59 correct in both.

Full-400 gold NLL improves **0.532686→0.527131**; selected-100 NLL worsens **0.554420→0.570376**. Keep these populations separate from the 87 supported-reference semantic denominator. Individual examples also disagree: index 24’s NLL improves 1.390→1.220 while the answer loses the repeated-final-window rule; index 74’s nearly unchanged NLL accompanies a repaired time unit.

All 100 totals are 74 correct, 9 partial, 15 incorrect and 2 ambiguous. All 100 reach EOS, none hits the 160-token cap, and all have detected answer fields. The two ambiguous cases (89 and 95) retain pre-existing source concerns; no new ambiguity or denominator changes were introduced. Minor spelling/paraphrase differences are judged on substance. Required source and relevant full evidence were checked, including default ListBox items in a retrieved passage.

The completed final pair and all 100 input IDs, questions, prompts, gold/masks and supplied evidence match the previous evaluation. The original 400 fingerprint/order is unchanged. Maximum final-panel NLL and gold-token replay differences are both exactly zero. The evaluator verifies inherited Adam at lifetime step 6,250. All 500 earlier answer records and original 87-supported/13-concern source labels remain unchanged.

These results do not establish convergence or predict longer-context quality. The 50k reader’s own full/no-context loss views are distinct from the frozen P2 answer baselines and are not substituted into them. No confirmation data, external judge, GPU jobs or product-source changes were used for this review. All indices above are zero-based.

The updated report is [CONTINUATION_COHORT100.tressoir.html](./CONTINUATION_COHORT100.tressoir.html), with [complete evidence](./CONTINUATION_COHORT100.evidence.json). It contains 100 questions and 600 answer cards. Focused browser checks passed for metrics, new filters/cards, prior-record equality and desktop/mobile layout; actual VSCode integration remains untested.

Frozen input SHA-256: `0e3e08bdef2e2012cdb77ef70d79cd4c26d3d763d7790720361aab61b0611cbe`. Per-answer judgments, source hashes, final-pair payload hashes and replay receipt are retained in `IB/TMP/FIRST_TASK_AWARE_FINETUNING/fresh50k_native_review` and the report evidence.
