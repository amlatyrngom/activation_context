# Final long-context adaptation: 40,000 questions

The final checkpoint gives **65/87 correct at 8k** and **60/87 at 32k** on the fixed development cohort. All 200 new answers were reviewed against supplied sources. The original 87 supported / 13 reference-concern labels and 700 earlier answer records remain unchanged.

| View | Correct / 87 | Partial | Incorrect | Ambiguous | Gold NLL, selected 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 32,000 questions, 8k | 58 | 11 | 17 | 1 | 0.735250 |
| Constant-floor 42,000 questions, 8k | 61 | 13 | 12 | 1 | 0.656747 |
| Final long-adapted 40,000 questions, 8k | **65** | 13 | 9 | 0 | **0.634750** |
| Constant-floor 42,000 questions, 32k | 58 | 10 | 18 | 1 | 0.678535 |
| Final long-adapted 40,000 questions, 32k | **60** | 12 | 14 | 1 | **0.649727** |

At 8k, the final gains 11 fully correct answers and loses 4 versus its original 32,000-question parent; versus constant floor it gains 9 and loses 5. At 32k, it gains 9 and loses 7 versus constant floor. Comparing the final checkpoint with itself across contexts gives three gains and eight losses at 32k. These are small descriptive differences, not proof of general superiority or that longer context generally hurts.

The generator cohorts give 35/42 and 30/45 correct at 8k, then 31/42 and 29/45 at 32k, for 9B and 27B respectively. Across all 100 references, 8k gives 69 correct / 15 partial / 14 incorrect / 2 ambiguous; 32k gives 64 / 14 / 19 / 3. All 100 questions remain in NLL and examples.

Concrete differences: human Item 056 removes an invented redaction API at 32k; Item 084 restores coupon-income values. But Item 035 confuses royalties paid after expiry with those accruing after expiry, Item 074 names the module instead of the frontend instance, and Item 065 reverses the default-extras condition. Item 032 changes from intended-source pytest 3.9.0 to an actual competing 7.2.0 announcement in 32k Passage 26; it remains ambiguous under the existing rule. Both contexts misapply a 20% reactivation-fee rate to the application-fee deduction at Item 057.

Training identity is `strong32k_piecewise_8k_8k`, preselected final `epoch_001`: original 32,000-question pair plus **8,000 fresh questions ×1 pass**, with fresh Adam and independent 8k passages. This is **40,000 lifetime unique**, not 42,000 or 50,000. The 8,000 questions are the first 8,000 of the same presentation pool as native 10,000-question continuations, by design. Comparisons with 42,000-question continuations change data amount, context, optimizer and schedule together.

All five final model/config hashes agree across both readouts. The 8k readout exactly replays all three original 400-question NLL panels, then selects 100 (AC mean 0.596817 over 400). The 32k decoder binds to the completed selected-100 NLL output, verifies actual inputs and rehashes the same pair; it does not rescore NLL. Its source is `strong_long_32k_answers100/result.json`, not the NLL-only 32k result.

The actual 32k inputs match the prior constant-floor 32k inputs in evidence, messages, answer positions, assistant IDs and row budget: 31,736–32,767 evidence tokens, 10–64 passages and 1,995–2,074 rows, ratio 1/16. Required sources remain intact; extra retrieval mainly adds distractors. The HTML shows a separate 32k panel, and the evidence JSON explicitly binds both 32k views to that verified input record.

Generation is greedy, thinking disabled, no gold prefix, cap 160, with EOS IDs 248044 and 248046. All 200 new replies reach EOS; none caps, trails after its wrapper or visibly loops (4,353 generated tokens at 8k; 4,338 at 32k). Raw replies and token IDs are retained.

Frozen P2 full/no-context answers remain **8k with different reader weights**. Own-reader full/no-context are NLL-only: selected-100 0.593878 / 1.833538 at 8k and 0.566293 / 1.833124 at 32k. There is no decoded full-32k ceiling here and no 400-question 32k score. This repeatedly examined development cohort is not independent confirmation. Source review did not audit every distractor.

[Interactive report](./STRONG_LONG_COHORT100.tressoir.html) · [All 900 answers and evidence](./STRONG_LONG_COHORT100.evidence.json) · [Exact result and review receipts](./STRONG_FINAL_LONG_QUALITATIVE_RESULT_V1.json)
