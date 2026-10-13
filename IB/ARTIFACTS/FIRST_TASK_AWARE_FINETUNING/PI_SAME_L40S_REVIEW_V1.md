The matched L40S comparison is complete: **66/87 correct for monolithic AC, 59/87 for passagewise AC, and 81/87 for the same 42k reader given full text**. [The report](PI_SAME_L40S_COHORT100.tressoir.html) contains all 300 actual answers, source-based judgments, gold-token losses and complete supplied evidence.

| View, identical 100 questions | Gold NLL, all 100 | Correct, supported 87 | Partial | Incorrect | Ambiguous |
|---|---:|---:|---:|---:|---:|
| Monolithic AC, 32k | 0.631402 | 66 | 4 | 17 | 0 |
| Passagewise AC, 32k | 0.679233 | 59 | 10 | 17 | 1 |
| Same reader, full 32k text | 0.574369 | 81 | 1 | 4 | 1 |

Both AC layouts use the same constant-floor 42k paired checkpoint, exact questions, gold, required source, complete evidence and total compressed rows. All three views ran on L40S with the same recorded precision policy and microbatch-1 scoring, in distinct allocations. There was no additional 32k training or optimizer update. The model had seen 42,000 unique questions once at native 2k.

Monolithic versus passagewise has 14 monolithic-only correct answers and 7 passagewise-only correct answers. Full text versus monolithic has 16 full-only correct answers and one AC-only correct answer. These paired differences describe this fixed development cohort; they do not establish a population-level or training-seed result.

The final passagewise snapshot preserves the first 88 complete rows exactly. Eighty complete replies match the earlier RTX passagewise replies byte for byte under identical source/evidence, allowing their prior judgments to be carried. Every one of the 20 changed replies was inspected directly. Relative to that earlier RTX review, passagewise gains indices 8, 64 and 94 and loses 69 and 73. Indices here are zero-based; the HTML displays items 1–100. The 87 supported and 13 reference-concern labels are unchanged, and all 100 remain visible.

Concrete differences include the missing unrepresented-former-employees qualification (8), default extras (64), en dash versus plus (69), `get_input_ids` versus `get_inputs` (73), and the exact TweetTokenizer output (94). Item 31 remains contextually ambiguous: the actual evidence contains pytest announcements for 2.8.4, 3.9.0 and 7.2.0. It remains in the original supported-reference denominator; no labels or historical judgments were changed to improve scores.

Fresh L40S native-2k scores are AC 0.554841, full text 0.633243 and no context 1.842807. The 32k execution's own no-context score is 1.841309. These are score-only controls; this report does not borrow older native-2k or no-context decoded replies. Slight no-context execution differences are recorded without claiming a moving reader or a proven numerical cause.

Monolithic AC scored 100 questions in 277.90 seconds at 28.27 GiB peak allocated GPU memory; passagewise took 425.87 seconds at 31.88 GiB. Separate allocations and one scoring pass per layout make these observations, not a serving benchmark. Monolithic source counters exclude appended summaries: up to 33,177 formatted source tokens plus 2,074 summary positions gives 35,251 positions in the single causal encoder pass.

The [evidence JSON](PI_SAME_L40S_COHORT100.evidence.json) binds all snapshots, paired checkpoint hashes, input identities, raw generated IDs and review notes. Local review and browser receipts live in `IB/TMP/FIRST_TASK_AWARE_FINETUNING/same_l40s32k_review/`. Chromium checks cover 100 questions, 300 answer cards and token-loss sections, filters, disclosures, render-event state, mobile layout and simulated dark theme. Actual editor integration and feedback persistence were not exercised. No reserved confirmation data, active source, GPU jobs or previous reports were changed.
