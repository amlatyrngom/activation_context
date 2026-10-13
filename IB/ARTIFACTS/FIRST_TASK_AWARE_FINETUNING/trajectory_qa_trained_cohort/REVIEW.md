# Trained trajectory-QA: paired 100-item review

**Answer quality is effectively flat in this small cohort:** initial AC 40/100, after 400 questions 40/100, after 2,000 questions 41/100 semantically correct. Training changes many individual answers in both directions. The small gold-NLL reduction is not a clear qualitative gain.

All 100 new AC answers at both checkpoints and all 100 400-question no-history answers were manually read. Original gold/full-history/no-history/AC judgments were retained, with the same item IDs and reference concerns. All 100 final no-history answers were also reviewed.

| View | Correct /100 | Partial | Wrong | Abstain/unanswered | Unknown | Correct + complete submit /100 | Gold NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen full history | 96 | 1 | 1 | 1 | 1 | 95 | 0.299322 |
| frozen no history | 26 | 10 | 59 | 4 | 1 | 25 | 1.772159 |
| initial AC | 40 | 14 | 45 | 0 | 1 | 40 | 0.944860 |
| AC after 400 questions | 40 | 11 | 48 | 0 | 1 | 40 | 0.909373 |
| Reader after 400 questions, no history | 22 | 13 | 61 | 3 | 1 | 21 | 1.781572 |
| AC after 2000 questions | 41 | 13 | 45 | 0 | 1 | 41 | 0.908110 |
| Reader after 2000 questions, no history | 22 | 13 | 55 | 9 | 1 | 20 | 1.758753 |

| Comparison | Gained correct | Lost correct | Stayed correct | Neither correct |
|---|---:|---:|---:|---:|
| initial AC → AC after 400 questions | 9 | 9 | 31 | 51 |
| initial AC → AC after 2000 questions | 9 | 8 | 32 | 51 |
| AC after 400 questions → AC after 2000 questions | 9 | 8 | 32 | 51 |
| frozen no history → Reader after 400 questions, no history | 1 | 5 | 21 | 73 |
| Reader after 400 questions, no history → AC after 400 questions | 24 | 6 | 16 | 54 |
| frozen no history → Reader after 2000 questions, no history | 1 | 5 | 21 | 73 |
| Reader after 400 questions, no history → Reader after 2000 questions, no history | 2 | 2 | 20 | 76 |
| Reader after 2000 questions, no history → AC after 2000 questions | 25 | 6 | 16 | 53 |

At 2,000 questions, the nine gains over initial AC are items **3, 9, 17, 21, 38, 43, 76, 77, 92**; the eight losses are **5, 36, 40, 41, 44, 55, 61, 66**. These counts retain the unknown within the denominator of 100.

- **Real recoveries:** Barisan Nasional, Malaysia, the FISA subchapter, Playwright, UNESCO database name, correct birth/death rates, repeats ≥1, exact SpikedSwordAndShield, and the correct unexpanded exponential.
- **Lost exact details:** api.sandbox.orcid.org becomes sandbox.orcid.org; 151 tests becomes 158 then 100; _set_optbools becomes _set_opt_bools then _set_optional; alphabetical boolean ordering becomes end-of-list placement; the plate gains a spurious digit. These are substantive failures, not paraphrase penalties.
- **No stable broad progression:** several 400-question gains disappear at 2,000, including the actual import paths, Infernal Arms, Davy Jones, and the linear approximation. Conversely the final checkpoint repairs other facts. Some answers move nearer to gold while staying wrong: hermes_json has wrong numeric offsets, the file path omits services/gcs, and timeout 30 is restored while the ordering inequality remains reversed.
- **Formatting is largely stable:** AC completes submit_answer on 99/100 items at every checkpoint. The recurring zero-byte prologue loop remains unfinished. This is not primarily a tool-formatting failure.

By source, initial → 400 → 2,000 AC correctness is coding **14/42 → 10/42 → 10/42**, research **18/43 → 22/43 → 22/43**, and math **8/15 → 8/15 → 9/15**. For the longest histories (16k–32k), it is **9/28 → 10/28 → 8/28**. These small correlated groups do not support significance or capacity claims, but the aggregate hides losses in exact code details.

At 400 questions, no-history semantic correctness falls from 26 to 22. Its gold NLL is 1.781572 versus frozen 1.772159. AC’s 400-question gains therefore are not explained by a general improvement in answering without history. This does not isolate encoder learning: the trained reader can change how it interprets AC, and individual priors change in both directions.

At 2,000 questions, no-history remains **22/100** correct, with gold NLL **1.758753**. Relative to frozen no-history it gains only item 16 and loses items 29, 45, 51, 65 and 83. Completed submit calls decline **96 → 95 → 87**; final no-history has 7 unanswered items and 2 explicit prose abstentions, mostly with unfinished reasoning. Its 22 semantically correct answers include two unfinished tool calls, so only **20/100** are both correct and submitted. This is visible reader-behavior drift despite slightly better gold likelihood, under the same 256-token decode cap. AC remains 99/100 complete and adds **25 correct answers** while losing **6 no-history successes** versus this final reader ablation. The channel remains useful; trajectory-QA fine-tuning has not yet produced a broad gain over the starting channel.

The final AC NLL is 0.908110 versus initial 0.944860, a reduction of about 3.9%. Concrete answers do not show a corresponding broad gain. For example, the exact protocol name has improved NLL but remains invented; a low-NLL wrong frame-class or integer still fails the question. Use this cohort’s own NLLs; it is a fresh hardware/kernel evaluation, not a bitwise replay of training panels.

The original caveats remain: item 24 has conflicting ages and remains unknown in every view; item 4 names the wrong method; item 98 asks for eight digits but the supported LCCN has ten. No item was dropped. Some questions can be solved from the question or prior knowledge. The 100 questions come from 30 roots, so this is not 100 independent trajectories. Equivalent expressions and semantic error-message paraphrases are judged under the prior documented policy; borderline choices remain visible.

This diagnostic does not establish convergence or an architecture-capacity limit. It supports retaining the initial pair as a control and testing an explicit next hypothesis rather than claiming a quality improvement from the held NLL alone.

Evidence: [item-keyed paired responses, verdicts, identities and transitions](evidence.json). Initial selective grounding is included from the prior review. No remote calls or new external factual audit.

| # | Initial AC → 400 AC → 2000 AC | 400 no-history | 2000 no-history | Review note |
|---:|---|---|---|---|
| 1 | C → C → C | E | E | AC retains 4; the 400 no-history response states 4 in prose but remains unfinished. Final no-history: States GCD 4 correctly in prose, but again does not finish the submit call. |
| 2 | E → E → E | W | W | Both trained AC answers retain physical searches; no-history still substitutes electronic surveillance. |
| 3 | W → E → C | W | E | AC fixes Pakatan Harapan to Barisan Nasional at both checkpoints; 400 no-history moves from initial correct to wrong. Final no-history: Recovers Barisan Nasional, matching frozen no-history and final AC. |
| 4 | W → W → W | W | W | Protocol identifiers remain invented despite improving gold NLL. Original wrong-method question caveat retained. |
| 5 | C → W → W | W | W | Training loses the correct sandbox URL by dropping api. at both checkpoints. |
| 6 | E → P → E | W | P | At 400 AC omits qcurve; at 2000 it recovers the full correct tuple. Final no-history: Now includes all three correct segment types, but adds unsupported ccurve. |
| 7 | C → C → C | C | C | All retain 1934; no-history also succeeds. |
| 8 | C → E → E | W | I | Both trained AC answers retain the particular solution (4,-18). Final no-history: Spends the output limit on a derivation without returning the requested particular solution. |
| 9 | W → W → C | W | W | Only 2000 AC fixes Subchapter II to Subchapter I. |
| 10 | C → C → E | W | W | Both AC answers retain 113; the final answer omits only the unit. |
| 11 | W → W → W | W | W | Both trained AC outputs repeat zero bytes without NoiseAPIInit and never close the tool call. |
| 12 | W → W → W | W | W | The identifier changes from orcidProxy to orcid_api to local; none is the recorded orcid. Final no-history: Changes the invented identifier to ORCIDProxy. |
| 13 | P → E → P | W | W | 400 AC recovers start of an open path; 2000 keeps start but loses the essential open-path distinction. |
| 14 | W → W → P | W | W | 2000 AC recovers hermes_json: Extra data, but column/character 1 are still wrong versus 47/46. |
| 15 | C → C → C | W | A | Both trained AC answers retain coefficient -24. Final no-history: Explicitly recognizes absent trajectory/context and says the coefficient cannot be determined; unfinished prose abstention. |
| 16 | C → C → C | W | C | Both trained AC answers retain Subchapter II. Final no-history: Recovers Subchapter II without history; AC already answered this correctly initially. |
| 17 | W → C → C | W | W | AC fixes Turkey to Malaysia at both checkpoints; 400 no-history remains Turkey. |
| 18 | P → P → P | P | P | All trained views give the mask but omit the explicitly requested zero-bit shift, as in the baseline. |
| 19 | W → W → W | W | W | All continue omitting OAUTH from both exact configuration identifiers. |
| 20 | E → E → E | W | W | Both trained AC answers retain move. |
| 21 | W → W → E | W | W | 2000 AC recovers Playwright; 400 AC and no-history substitute WeasyPrint. |
| 22 | W → W → W | W | A | Coefficients and right-hand sides are rearranged incorrectly at both checkpoints. Final no-history: Explicitly recognizes missing trajectory, then discusses a hypothetical example; no actual answer. |
| 23 | C → E → E | W | W | Both trained AC answers retain -7.3; missing percent sign is harmless here. |
| 24 | U → U → U | U | U | The 92/93 source conflict remains unknown for every view. Trained AC now matches gold 93; this is not counted as a factual gain. Final no-history: Same conflicting 92/93 reference retained as unknown; supplies 92. |
| 25 | W → W → W | W | W | RequiresEncryptionError and InvalidPreambleError are not RequiresEncryptionAPIError. Final no-history: Now invents InvalidPreambleError, also used by final AC. |
| 26 | C → C → C | C | C | All retain the production authorization URL, also known without history. |
| 27 | W → W → W | W | W | 400 AC invents a point-count implementation; 2000 confuses the function with an unsupported-curve exception. |
| 28 | C → C → C | P | P | AC retains the complete date; 400 no-history gives only 1934. Final no-history: Still gives only the year, omitting the requested month and day. |
| 29 | E → E → E | I | I | Both AC answers retain equal service rates. 400 no-history derivation ends before a substantive completed answer. Final no-history: Unfinished derivation does not supply the equal-service-rates answer. |
| 30 | W → W → W | W | W | Both AC answers remain wrong; -2.1 at 2000 has the right magnitude but the wrong sign. |
| 31 | P → P → P | P | P | The topic survives, but every trained answer omits the required site restriction/exact query. |
| 32 | C → E → C | W | W | Both AC answers retain line 291. |
| 33 | W → W → W | W | W | All trained views remain generic missing-field messages without the recorded node-qualified template. |
| 34 | W → W → W | W | W | 2000 moves closer to collection.py but still omits services/gcs from the required path. |
| 35 | P → W → P | W | W | 400 corrupts both numbers; 2000 restores $2.9 billion but gives 21408 instead of 21489 complaints. |
| 36 | E → W → W | I | I | 400 inverts the normalizer; 2000 introduces an extra denominator with p0. Initial correct expression is lost. Final no-history: Unfinished derivation does not supply the requested normalized formula. |
| 37 | W → W → W | W | W | The recorded aggregate 291.2 is not recovered. |
| 38 | P → W → E | W | W | 2000 recovers UNESCO Database of National Cultural Heritage Laws; 400 substitutes ICH Platform. |
| 39 | E → W → E | W | W | 400 replaces masking with array conversion; 2000 restores selecting the non-masked elements. |
| 40 | E → W → W | E | E | Both trained AC answers lose initial_system_message; 400 no-history retains it. |
| 41 | E → W → W | W | W | Both trained AC answers replace alphabetical sorting with end-of-list placement, losing an initial success. |
| 42 | W → W → W | W | W | Both AC answers remain wrong years. Outside-world accuracy of the source claim remains unaudited. |
| 43 | W → W → E | W | I | 2000 fixes the birth/death rates. 400 AC keeps ratios; 400 no-history reverses the rates. Final no-history: Unfinished generic queueing explanation never supplies the correct two rates. |
| 44 | E → C → W | W | W | 400 retains Indonesia/Pakistan; 2000 loses them to Japan/South Korea. |
| 45 | P → P → P | P | P | AC still invents an empty-list return; no-history supplies an unsupported document-content explanation for the failed extraction. Final no-history: Retains failure outcome but adds unsupported language/format speculation. |
| 46 | P → P → P | P | P | Unregistration purpose survives, but both trained AC outputs invent its implementation details. |
| 47 | W → E → W | E | E | 400 AC fixes the already-initialized warning semantically; 2000 returns to a different generic warning. |
| 48 | E → E → E | E | E | All retain the unsupported keyword failure; final AC is brief but semantically sufficient. |
| 49 | W → W → W | W | W | AC changes 42% to 3125% to 112%; none recovers 442%. |
| 50 | E → E → E | E | E | Equivalent-normalizer convention retained; final AC additionally restores the reference summation form, valid at rho=1. Final no-history: Correct normalization p0 times geometric sum equals 1 is stated in prose, including the rho=1 sum, but the final formula/tool call is unfinished. |
| 51 | E → E → E | P | P | AC retains wind/15%; 400 no-history identifies wind but gives 13%. |
| 52 | W → W → W | W | W | All retain the wrong-jurisdiction 1987 answer instead of the source's 2003. |
| 53 | P → P → P | P | P | 400 AC restores Meta but misses UNKNOWN; 2000 omits Ctrl/Meta/UNKNOWN and adds AltGr. |
| 54 | W → W → W | W | W | Both AC outputs invent near-matching frame-class names. |
| 55 | E → W → W | W | W | Initial correct _set_optbools is lost to _set_opt_bools then _set_optional. |
| 56 | E → E → E | E | E | All retain the lowered-barriers claim, which also succeeds without history. |
| 57 | E → E → E | I | I | AC retains both dimension pairs; 400 no-history again spends the cap on unfinished derivation. Final no-history: Derivation is unfinished before either dimension pair is given. |
| 58 | C → C → C | W | W | Both AC answers retain 25%. |
| 59 | E → E → E | C | C | Both AC answers retain Latin Bridge and replace the earlier irrelevant plate addition with its local name. |
| 60 | W → W → W | P | P | Both AC outputs continue inventing method names/operation sequences; no-history supplies only broad creation and validation. |
| 61 | E → W → W | W | W | Initial correct 151 tests becomes 158 then 100. |
| 62 | W → W → W | P | P | AC still omits the failure prerequisite; no-history has failure-like behavior but adds an unsupported maximum-count condition. |
| 63 | P → W → W | W | W | Trained AC invents Carpentry Store then Sea of Fortune rather than Atlantica Casino. Final no-history: Rephrases the sea-theme clue as a fabricated district, without naming Atlantica Casino. |
| 64 | W → W → W | W | I | Both AC answers retain wrong coefficients, including a different orientation's equation at 2000. Final no-history: Unfinished general fencing derivation does not return the recorded numerical quadratic. |
| 65 | C → C → C | C | W | All retain 10%. Final no-history: Loses the initial/400 correct 10% answer to 1.5. |
| 66 | E → E → W | W | W | 400 retains the plate; 2000 inserts an extra 1 into AIII 1118. |
| 67 | W → W → W | E | W | AC remains at 400; the 400 no-history reader now correctly supplies 422. Final no-history: Reverts the 400 correct 422 answer to initial wrong 400. |
| 68 | P → E → P | E | E | 400 AC identifies the actual /workspace versus /testbed copies; 2000 replaces this with an invented branch/commit/traceback explanation. Final no-history: Preserves the broad old-copy explanation, with an unverified cached-version alternative. |
| 69 | E → E → E | E | E | All correctly decrement the counter. The question asks about the counter, so omitting requeue prose is accepted. |
| 70 | W → C → W | W | W | 400 AC fixes Infernal Arms; 2000 loses it to Infernal Sword. |
| 71 | W → W → W | W | I | Both AC answers give wrong dimension pairs, each retaining only one of the required values. Final no-history: Unfinished generic fencing reasoning does not give either requested numerical length. |
| 72 | W → W → W | E | E | AC titles remain wrong; 400 no-history retains the correct Japanese title. |
| 73 | E → E → E | P | W | AC retains the armistice-date interpretation. No-history's August claim conflicts with its own November prose and is unfinished. Final no-history: Now gives an explicitly wrong August-date interpretation with invented offensive/German-owner details; submit call unfinished. |
| 74 | W → W → W | E | E | AC continues inventing method/receiver names; 400 no-history retains register_blueprint. |
| 75 | P → P → P | P | P | All retain a broad invalid-state error; trained AC also conflates source-state and destination-state wording. Final no-history: Adds ValueError and source-state wording, but still omits the requested parameterized message. |
| 76 | W → E → E | E | E | Both trained AC answers fix the minimum-one validation rule; no-history already knew it. |
| 77 | W → W → C | W | W | 2000 AC recovers the exact SpikedSwordAndShield file identifier. |
| 78 | W → E → W | P | P | 400 AC fixes the linear approximation; 2000 substitutes the unexpanded exponential and leaves a unsubstituted. |
| 79 | W → W → W | W | W | Neither AC checkpoint recovers Princess Connect! Re:Dive. |
| 80 | E → E → E | E | E | All retain chauffer/to heat. |
| 81 | E → E → E | E | E | All retain ValueError, which is all the question requires. |
| 82 | P → P → P | P | P | Error meaning survives, but requested literal templates remain changed; 2000 includes more of the required placeholders. Final no-history: Adds ValueError but still omits the requested state/symbol/alphabet template. |
| 83 | W → W → W | W | W | AC remains at wrong default 1; 400 no-history regresses from initial 0 to 1 as well. |
| 84 | C → C → C | W | W | Both AC answers retain Holy Arms. |
| 85 | W → W → W | W | W | Neither trained AC answer restores the negative quadratic coefficient. |
| 86 | P → W → W | W | W | Both trained AC answers invent initiative names instead of WREN; this is weaker than identifying the actual questionnaire. |
| 87 | P → P → P | W | W | AC retains lying on grass but invents gaze/closed-eye details and omits the stream/close-to-earth particulars. |
| 88 | W → W → W | W | W | Wrong compared objects/conditions persist; 2000 invents an rstrip comparison. |
| 89 | W → W → W | W | W | 2000 copies a JSON-serialization error instead of the recorded msgpack/NAType error. |
| 90 | W → W → W | W | W | AC remains at 0x00000002 instead of 0x00000100. |
| 91 | W → C → W | W | W | 400 AC recovers Davy Jones; 2000 loses it to Captain Deadbeard despite the strong clue. Final no-history: Changes the invented captain name to Captain Gumm. |
| 92 | W → W → C | W | W | 2000 AC fixes the exact unexpanded exponential expression. Final no-history: Explicitly endorses the wrong product expression in unfinished prose. |
| 93 | E → E → E | E | E | All retain no mention of Offwind; source-summary limitation remains. |
| 94 | W → P → W | E | E | 400 AC is only a generic seasons/emotion paraphrase; 2000 supplies an unrelated quote. No-history retains the leaf-loss metaphor. Final no-history: Preserves the autumn/withered-tree metaphor under the baseline semantic policy. |
| 95 | W → P → P | W | W | Training restores timeout 30, but both AC outputs still reverse the required timeout ordering; 2000 also recovers WinRMError. |
| 96 | W → W → W | W | W | None recovers literal <NA>; final AC changes None to nan. |
| 97 | C → W → E | C | C | 400 AC regresses from false to -1; 2000 restores false. No-history already succeeds. |
| 98 | C → C → C | W | W | Both AC answers retain exact LCCN 2006901713; original eight-versus-ten-digit question defect retained. Final no-history: Returns to the initial invented LCCN, not the recorded identifier. |
| 99 | E → C → C | E | E | Both trained AC answers make the formerly verbal Taylor terms explicit; no-history also succeeds. |
| 100 | W → W → W | W | W | Both trained AC URLs are wrong; 2000 restores the correct host but invents the path. Final no-history: Still invents an NREL report URL. |

C/E=correct direct/equivalent; P=partial; W=incorrect; A=abstention; I=unanswered; U=unknown conflicting reference. Tool-call completeness is separate.
