# 27B whole-parent replay: bounded semantic review

**The 27B replay does not establish reliable multi-section supervision.** Whole-parent generation avoids the earlier temporary-label problem and finds usable questions, but 11 of the fixed 15 accepted examples can be answered from one cited unit. Only two examples are both fully grounded and clearly require combining naturally related information. Do not expand this unchanged recipe while describing its output as questions requiring multiple sections.

## Yield, separately from quality

The same 75 parents produced **73 accepted questions from 207 candidates**, versus the earlier 9B attempt's 69 from 221. Accepted questions came from 49 parents, versus 38 for 9B. The 27B worker finished at 2026-09-22 00:16:46 UTC; its report records 14.5 minutes elapsed. Source-file SHA256 is unchanged: `7e3941fde695ddb3ba75151b1d9fc8cdfae4f789d99e57358b4467d6228565f6`. Prompts and decoding settings were preserved, with thinking off. These are generation results, not measurements of answer quality or population error rates.

| Domain | ~8k accepted | ~16k accepted | ~32k accepted | Total |
|---|---:|---:|---:|---:|
| Code | 5 | 2 | 9 | 16 |
| Manuals | 3 | 4 | 2 | 9 |
| Legal | 5 | 7 | 8 | 20 |
| Math/scientific papers | 6 | 7 | 7 | 20 |
| Office | 2 | 4 | 2 | 8 |

Each cell had five parent inputs. Natural lengths vary; parents were not padded. Office inputs remain explicitly partial, coherent reference dossiers. The matched review below contains no Office examples or 32k Manuals examples, so those newly productive cells have **not** received a manual semantic check.

Thirteen candidates failed evidence-index checks: 12 repeated an index (six Manuals, five Office, one Code), and one Manual had another invalid index. Of 194 checker calls, 73 accepted and 121 rejected. Among rejections, 109 failed only `needs_multiple_sections`, three failed both that flag and grounding, five failed only grounding, and four failed grounding plus self-containment. These flags are the model's decisions, not independently verified rejection labels. All checker calls ended normally. Six generator outputs reached the 1,024-token cap and were malformed, compared with one in the 9B attempt. There were zero temporary-reference rejections; a bounded pattern scan of the 73 accepted pairs also found none.

## Fixed 15-example review

The selection plan was saved before inspecting new answers. It first reserved accepted questions from the same 15 parents used in the earlier 9B review, selecting the smallest seeded question-ID hash per parent. Twelve parents matched. The Keras, Swin and Byte Buddy parents had no accepted 27B questions; the predeclared fallback selected three unused Code parents in their matching length cells. No choice was based on answer quality. Exact selection, complete portable rows and evidence receipts are preserved in `IB/TMP/FIRST_TASK_AWARE_FINETUNING/long_qa_whole_parent_27b_v1_review/`.

All 15 selected support spans match their recorded offsets in the complete parent. The manual grounding counts are **11 supported, one incorrect, one incomplete chosen-support set, and two needing qualifications**. Evidence-necessity counts are **11 single-unit answers with redundant citations, three natural relationships, and one distributed lookup**. One natural-relationship answer needs qualifications, leaving two clear, fully grounded combinations. No sampled pair leaks temporary labels. These are descriptive counts for this fixed sample, not an estimated bank-wide error rate.

| ID suffix | Domain/band | Grounding and evidence necessity |
|---|---|---|
| `aa68394c92843dab139d49c0` | Code/8k | ZAP secure URLs are correct. Unit1 alone contains both proxying cases. |
| `573fb61ad082863c37be5f4e` | Code/16k | Certificate-tab disabling is correct. Unit7 alone contains the complete checkbox handler. |
| `a1e75c15ab5af27f8411be23` | Code/32k | TSpan's elapsed-presence bit check is correct. Unit4 alone contains the method and mechanism. |
| `2566355a11c8076cf341607f` | Manuals/8k | Bugsnag listener implementation and installation are correct. Both already appear in unit2. |
| `d8d983728a13e98105c52b57` | Manuals/16k | **Incomplete support:** the answer includes Vector defaults, absent from units0/1/8. Full-parent section7.1 confirms them. A distributed lookup across client descriptions, not a demonstrated reasoning dependency. |
| `963e3eab884a26bf1b06eec7` | Legal/8k | Grounded relationship between the business-judgment rationale, company-specific due-diligence facts and the court's disposition. |
| `1429648a9387bbf885726024` | Legal/16k | No-return limitations rule is supported. Unit1 alone gives both the rule and when the clock starts. |
| `479224fab7b0f90de92e05cf` | Legal/32k | Federal uniformity/intervention explanation is correct. Unit14 alone contains the full rationale. |
| `3ea2b8c5f9bfbbe1d8dadf9a` | Math/8k | **Incorrect equation and corrupted notation.** Source has `beta0(alpha)=O(beta(alpha)^2/alpha)`; answer replaces alpha with beta. Unit3 alone already contains the comparison. Details below. |
| `cbb0047bf3b706ac8c28ef77` | Math/16k | Concurrence comparison is supported for the study's measurement setting. Unit7 alone supplies both regimes and their values. |
| `d844a95c69267bf31c4de305` | Math/32k | Complexity exponents are supported, but positive Elkies-prime proportion and field-size conditions are omitted. Naturally related baseline/result passages, with a needed qualification. |
| `c677dc1c0bd5c79cff7afd31` | Code/16k | Grounded caller/helper dependency: draw calls `switchTexture`; its separate definition calls `flush` before switching. The buffer-overflow branch also flushes. |
| `0cb0a5e8e62e494a4c5cf983` | Math/8k | Sphere/fiber coupling-peak diameter comparison is supported. Unit4 alone contains both; fiber length can still affect the wider distribution. |
| `6cc7ccb84778407dd77b63b7` | Legal/32k | Jurisdiction and cost-allocation reasoning is supported, but “explicitly excludes” overstates the statutory wording: exclusion follows from scope and judicial construction. Unit2 alone contains the explanation. |
| `8d8b1ab39e357461eebb218f` | Legal/32k | Merger-damages explanation is supported entirely by unit12. “Too speculative and speculative” is a style defect only. |

Full IDs, the matched 9B IDs, exact decisions and source notes are in [LONG_QA_WHOLE_PARENT_27B_REVIEW.json](LONG_QA_WHOLE_PARENT_27B_REVIEW.json). New questions were generated on the matched parents; this is not a same-question correctness comparison. In particular, the missing Swin acceptance cannot be presented as a correction of 9B's Swin error. The Lovelock parent still produced an incorrect accepted answer, with a different error.

## Exact mathematical/serialization defect

The source states `beta0(alpha) = O(beta(alpha)^2 / alpha)` and assumes `beta(alpha) << alpha`. The raw generator instead wrote `beta0(beta) -> beta(beta)^2 / beta`. The original gold answer is therefore incorrect even before serialization; it has not been repaired for this review.

There is also a distinct formatting defect in the same item. The raw JSON uses a **single** backslash before LaTeX `beta` and `to`. Standard JSON parsing interprets these as backspace and tab escapes. The portable question/answer consequently contains eight backspace characters, yet the checker returned all three flags true. The raw output, parsed candidate, exact source equation and checker response are saved in `lovelock_raw_parsed_receipt.json` under the review TMP folder. A bounded scan found this nonprinting-control defect in one of the 73 accepted rows. This is not evidence about either ordinary production bank; root is assessing those separately.

## Recommendation

Keep whole-parent access as the useful design choice: it lets the generator choose coherent evidence and resolved the earlier label-dependent pairs from forced distant spans. The accepted citations, however, are often redundant. A 32k parent with an answer in one short passage is a long-context retrieval example, not proof of multi-section reasoning; replacing 9B with 27B alone did not fix that distinction here.

Retain these outputs as a pilot and prioritize the current training results before another synthesis experiment. Do not expand sources or add more checking stages on this evidence. If this route later scales, report it as **long-parent QA** unless the multi-section requirement is shown to hold; keep exact source spans and parent splits. Before reusing the mathematical output path, the narrow serialization remedy is to request plain-text/Unicode math or valid doubled JSON backslashes and reject nonprinting controls. That fixes transport corruption, not the independently wrong equation or unnecessary citations. No production worker, prompt, accepted reference or GPU job was changed by this review.
