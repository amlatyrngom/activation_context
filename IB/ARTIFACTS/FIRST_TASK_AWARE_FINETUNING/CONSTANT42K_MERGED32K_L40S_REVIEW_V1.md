# Constant-floor 42k checkpoint: monolithic 32k answers

The 100-answer L40S readout gives **66/87 correct** on the unchanged source-supported subset: 4 partial and 17 incorrect. Across all 100, including the original 13 reference concerns, the judgments are 70 correct, 6 partial, 22 incorrect and 2 ambiguous. Every response ended at EOS; none reached the 160-token cap or added text after the answer wrapper.

| Same checkpoint, fixed questions | Correct / supported 87 | Paired changes versus monolithic 32k |
|---|---:|---|
| Native 2k, prior RTX readout | 71/87 | 1 monolithic-only gain; 6 losses |
| Passagewise 32k, prior RTX readout | 58/87 | 12 monolithic-only gains; 4 losses |
| Monolithic 32k, L40S | **66/87** | This readout |

The same 100 IDs, gold answers and paired weights were verified. The two 32k layouts retain exactly the same full evidence and total compressed-row budgets. **GPU hardware/runtime also changes (L40S versus RTX PRO); these results do not isolate a layout effect.** They show that useful answers survive a single approximately 32k encoder sequence, with remaining degradation relative to native 2k.

The monolithic gains over passagewise include the correct Sharding-JDBC component(#5), Yakima / 1860(#20), BioEssence net income 223,752(#47), AfterDocumentMerge API names(#81), forfeited-coupon decrease 22,959−2,217=20,742(#83), and ROLE_ANONYMOUS(#91). All 12 gained indices are 5, 20, 21, 35, 46, 47, 78, 81, 83, 86, 91, 96.

The four losses relative to passagewise are 7, 28, 55, 72. For example, the inverse-trigonometric branch formula gives F₂(1)=2π+π/2=5π/2, but this answer gives 1+π/2(#7). It selects para-Bannai-Ito rather than generalized Gegenbauer polynomials(#28), and invents a PDU redact method where the code calls prune_event(pdu)(#55, partial). Against native 2k, losses are 7, 22, 28, 31, 55, 62; the sole gain is the correct en-dash code point(#69). The liabilities calculation still fails:27,468,403−24,407,145=3,061,258, not 3,260,888(#62).

Two adjudications merit explicit context. The gall-fly answer(#68) correctly gives the specific stated physical limitation—difficulty righting themselves—and need not repeat the poor-fliers adjective. The pytest answer(#31) gives 3.7.0. No supplied announcement supports that version: actual announcements include 2.8.4, 3.9.0 and 7.2.0, while the 3.7 mentions concern Python. The earlier passagewise 2.8.4 ambiguity is retained, and none of the original 87/13 source labels changed.

By actual generator cohort, correct supported answers are 34/42 for 9B-generated questions and 32/45 for 27B-generated questions. These are descriptive counts on the existing development selection, not population accuracy estimates.

The unchanged all 100 teacher-forced answer NLL is**0.6314022821**. Scoring took 277.90s and peaked at 28.27GiB allocated GPU memory; these are scoring-only measurements. NLL measures the supplied gold strings, including the 13 known reference concerns, and is separate from this answer-content review. Optional matched full-context answers are outside this review.

All 100 notes, exact input hashes, runtime metadata, paired changes and supporting excerpts are in [the evidence JSON](CONSTANT42K_MERGED32K_L40S_REVIEW_V1.json). Frozen raw inputs and the review files are in `IB/TMP/FIRST_TASK_AWARE_FINETUNING/constant42k_merged32k_l40s_review/`. The AC 100 snapshot SHA256 is `7050fef460a077298cb8c1c2ce906a53a7648b876ccfaf1f42edbfa8deca6d39`.
