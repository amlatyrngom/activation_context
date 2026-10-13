The r8 half-AC-LR candidate does **not** show a semantic improvement sufficient to overturn its worse final NLL. This check does not justify expanding to all100.

Both candidate and original r8 have **8 correct, 5 partial and 5 incorrect** responses. Original r16 has **11 correct, 2 partial and 5 incorrect** on the same selection. These are18 preselected, supported-reference questions—one per source family in each generator cohort—not representative accuracy estimates.

The candidate removes a false success claim in item17 and recovers one of four MixR values in item50. It regresses on item8’s former-employee rule and item34’s royalty-accrual distinction. Item24 also loses useful detail within the partial category.

All18 candidate and matched-control responses parse, end at EOS and remain below the token cap. Neither exhibits a wrapper or repetition advantage over the other. Original r16’s item94 reaches the cap while repeating punctuation; original r8 already fixed that termination problem without fixing its tokenization answer.

| Index | Cohort / family | Original r8 | Half-LR r8 | Source-grounded assessment |
|---:|---|---|---|---|
| 94 | 27b / code | incorrect | incorrect | Still merges the colon with This and writes waaay rather than waaayyy; the doctest requires separate colon/This tokens and three a and y characters. |
| 24 | 27b / documents | partial | partial | Identifies window block counts but omits successive list entries and repetition of the final entry for remaining windows. |
| 62 | 27b / financial | incorrect | incorrect | Gives $1,008,776; the current-liabilities row requires $27,468,403 − $24,407,145 = $3,061,258. |
| 8 | 27b / legal | partial | incorrect | Invents former-client relationships and excludes managers; the opinion permits contact with unrepresented former organizational employees, including managers. |
| 42 | 27b / math_arxiv | incorrect | incorrect | Unequal bounds 7/12 and (28+19√19)/243 differ by 0.1273, not 0.0000. |
| 30 | 27b / math_textbook | correct | correct | The open interval (−4/3, ∞) follows from 3x+4>0 and includes the requested boundary. |
| 75 | 27b / office | correct | correct | The Remarks explicitly say EndConnectionSite generates an error when the connector end is unattached. |
| 22 | 27b / technical_web | incorrect | incorrect | East Timor is an ASEAN observer in the source, but is absent from its APEC and current Commonwealth-realm lists; Papua New Guinea satisfies all three. |
| 37 | 27b / web | partial | partial | Names the East India Company correctly but gives $15; Ireland615 minus India550 is $65. $15 instead matches the gap to China600. |
| 17 | 9b / code | partial | correct | Correctly gives LOGGER.error logging without throwing; the code still returns exit code1. |
| 79 | 9b / documents | partial | partial | Correctly identifies overriding the sysconfig data-module name but omits the specified _PYTHON_SYSCONFIGDATA_NAME environment variable. |
| 52 | 9b / financial | correct | correct | The 2024 supplemental noncash row gives $1,365,000 for stock issued for intangible assets. |
| 34 | 9b / legal | correct | partial | Says payment after expiration is illegal; the source restricts illegality to royalties accruing after expiration and preserves earlier accruals. |
| 50 | 9b / math_arxiv | incorrect | partial | Recovers p_cat=8, but q_fun=1, q_cat=1 and p_fun=2 are wrong; the sparse MixR values are2,2,8,8. |
| 10 | 9b / math_textbook | correct | correct | The supplied fourth-moment equation explicitly subtracts3. |
| 14 | 9b / office | correct | correct | The Job Redesign paragraph explicitly identifies increased anxiety and fear of failure. |
| 46 | 9b / technical_web | correct | correct | The sfyl.ifas.ufl.edu navigation link identifies Extension–Solutions for Your Life. |
| 80 | 9b / web | correct | correct | Accurately reports the passage’s central stated implication: thrombosis has not occurred. This is source support, not independent medical validation. |

[Detailed judgments and raw responses](LR_R8_QUALITATIVE_CHECK.json). [Fixed selection](LR_QUALITATIVE_CHECK_SELECTION.json). [Sources and unchanged original judgments](MODEL_COMPARISON_COHORT100.evidence.json).

Only these18 candidate responses were reviewed. Selection and prior annotations remain unchanged; no new reference concerns were found. Medical, legal and financial statements above describe support within the supplied sources.
