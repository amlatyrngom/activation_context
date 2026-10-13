# Initial trajectory-QA cohort: manual review

All 100 questions / 300 responses were read. This is the **initial 42k RAG pair**, before trajectory-aware training. The questions span 30 roots. Original histories were checked selectively on 25 items listed in evidence.json; no external or exhaustive factual audit was performed. No items were removed.

| View | Semantically correct /100 | Correct + complete submit call /100 | Partial | Incorrect | Abstain / unanswered | Unknown | Mean gold NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen full history | 96 | 95 | 1 | 1 | 1 | 1 | 0.299322 |
| frozen no history | 26 | 25 | 10 | 59 | 4 | 1 | 1.772159 |
| initial AC | 40 | 40 | 14 | 45 | 0 | 1 | 0.944860 |

AC answers 23 items correctly where no-history does not; no-history answers 9 correctly where AC does not. Full history answers 56 correctly where AC does not. These are paired content judgments, not an estimate of independent-sample significance.

The channel carries concrete information: item 5 retrieves the sandbox ORCID URL from 29k tokens; item 61 retrieves 151 passed tests from 25k tokens; item 98 retrieves the exact 10-digit LCCN from 32k tokens. It is nevertheless substantially weaker than full history here. Errors include nearby-entity substitutions (Pakatan Harapan/Barisan Nasional, Holy/Infernal Arms), invented code identifiers, dropped digits and incorrect mathematical signs or offsets.

Gold NLL is useful in aggregate but not a literal answer-quality scale. Correct AC alphabetical-order wording scores 1.859 (item 41); correct full-context chauffer/to heat scores 1.208 (item 80). Incorrect AC 42% versus 442% scores .146 (item 49). Full history scores .030 on the song title while emitting a visit call instead of answering (item 72). Full item 43 states the correct rates in prose but never completes submit_answer. Tool behavior and the 256-token cap must be separated from factual recall; cap exhaustion is inferred from unfinished output, since stop reasons were not persisted.

The gold wording often exceeds what the question requires: full/no-history answers can correctly give only ValueError, a year, a method name, or an algebraically equivalent expression. Literal matching and exact-gold likelihood can penalize these equivalents. Conversely a single wrong identifier token can invalidate an otherwise nearly predictable answer.

Reference caveats: item 24 has conflicting 92/93 ages in the supplied history and remains unknown for all views. Item 4 incorrectly locates protocol initialization in write_packet rather than _setup_proto, although its target string is supported. Item 98 says eight-digit while its supported target has 10 digits. Item 30 is a verified full-reader error, not a bad gold: the tool explicitly reports 2.1, but full answers 3.11. Item 52 is a jurisdiction mix-up: supplied search results put the Cook Islands act in 2003, while AC/no-history supply 1987. Literary dates and broad claims repeated by tool summaries were not independently verified.

Some questions are solvable or strongly cued without the history: GCD(96,20), the fully specified fencing problem, the first Taylor terms, the squid-faced captain with a detached heart, and invalid header magic. No-history successes elsewhere could also reflect prior knowledge or luck. Their presence weakens this cohort as a pure channel-isolation test; they remain included.

The current evidence supports an effective but imperfect latent communication channel. It does not show that the channel has saturated, identify an architecture-capacity limit, or establish trained trajectory-QA quality. The trained 100-item paired cohort is the next direct comparison.

These are manual semantic judgments, with borderline equivalents documented rather than hidden: for example, item 50 omits the rho=1 limit, item 47 paraphrases an exact warning, and item 99 names the Taylor terms without their coefficients. They should not be treated as a calibrated automatic benchmark. Unknowns remain within the denominator of 100.

Evidence: [item-keyed judgments and selective excerpts](evidence.json). Source: `/workspace/IB/TMP/SYNC/TRAJECTORY_COMPACTION/csail/qa_cohort_initial/report/`. Counts are manual, with partial/unknown categories retained; semantic versus exact-message boundaries are documented per item.

| # | Source | Full / no-history / AC | Note |
|---:|---|---|---|
| 1 | math | C / E / C | GCD is explicitly computable from the question; no-history states 4 but overruns before completing its tool call. |
| 2 | research | E / W / E | Physical search answers the requested type; gold adds a long statutory description. |
| 3 | research | E / E / W | Full/no-history name Barisan Nasional; AC substitutes its rival Pakatan Harapan. |
| 4 | coding | C / W / W | Exact protocol string is present. Question names write_packet, but the initialization actually occurs in _setup_proto. |
| 5 | coding | C / W / C | AC preserves the exact sandbox URL despite the 29k-token history. |
| 6 | coding | C / W / E | Quote-style differences are immaterial; no-history substitutes ccurve for move. |
| 7 | research | C / C / C | All three give 1934; no-history success makes this weak evidence of context use. |
| 8 | math | C / W / C | AC retrieves the trajectory's particular solution; no-history pair does not satisfy the stated equation. |
| 9 | research | C / W / W | AC/no-history confuse statutory subchapters. |
| 10 | research | E / W / C | Full answer 113 omits the unit; AC supplies 113 seats. |
| 11 | coding | E / W / W | Full byte literal is equivalent. AC repeats zero bytes and truncates instead of retaining NoiseAPIInit. |
| 12 | coding | C / W / W | Exact variable is orcid; ORCID and orcidProxy are different identifiers. |
| 13 | coding | E / W / P | AC retains on-curve information but misses the requested OPEN-path/first-segment meaning. |
| 14 | research | C / W / W | Exact hermes_json error is in the transcript; AC substitutes unrelated nearby defunct-1934 content. |
| 15 | math | C / W / C | AC retains the negative coefficient -24. |
| 16 | research | C / W / C | AC retrieves Subchapter II while no-history selects III. |
| 17 | research | C / W / W | AC and no-history both substitute Turkey for the recorded Malaysia. |
| 18 | coding | E / P / P | Masking with 0xff is behaviorally equivalent to zero-shift-plus-mask, but does not identify the explicitly requested shift operation. |
| 19 | coding | C / W / W | Both failed views omit OAUTH from exact configuration variable names. |
| 20 | coding | E / W / E | Full and AC identify move; added wording/quotation differences are equivalent. |
| 21 | research | E / W / W | Playwright with Chromium is sufficient; AC invents a WhiteLine library from nearby White Star material. |
| 22 | math | C / W / W | AC changes variables, signs and right-hand constant in the recorded equation. |
| 23 | research | C / W / C | AC recovers India's recorded -7.3 percent value. |
| 24 | research | U / U / U | Transcript conflicts: assistant reasoning says age 92, while a retrieved extract says 93. Full matches the chosen gold; AC/no-history match the other passage. No external factual adjudication. |
| 25 | coding | C / W / W | RequiresEncryptionAPIError is recorded; AC invents APINoisyFrameError. |
| 26 | coding | C / C / C | All three know the production authorization URL; this item alone does not establish channel use. |
| 27 | coding | P / W / W | Full gets the move/line grouping but incorrectly adds segmentType=1 from a nearby implementation. Actual _drawPoints passes segment_type to pen.addPoint. AC conflates the functions more extensively. |
| 28 | research | C / W / C | AC retrieves May 10, 1934; no-history supplies a different date. |
| 29 | math | E / E / E | Equal service rates answer the asked condition; the gold additionally gives the uniform mass. |
| 30 | research | W / W / W | Tool output explicitly lists 2020 average 2.1. Full's 3.11 and AC's -4.3 are unsupported. |
| 31 | research | C / P / P | No-history and AC give related searches but omit/reorder the exact recorded query and site restriction. |
| 32 | coding | E / W / C | Full omits the word Line; AC retains Line 291. |
| 33 | coding | E / W / W | Full's X is an example substitution for node_id in the same error template; other views change the message. |
| 34 | coding | C / W / W | AC invents a different file path. |
| 35 | research | E / W / P | AC gets the loss magnitude but substitutes 59000 for 21489 complaints. |
| 36 | math | E / W / E | AC's reciprocal-normalizer expression is algebraically identical for the stated unequal-rate case. |
| 37 | research | C / A / W | No-history appropriately lacks data; AC changes 291.2 to 291.0. |
| 38 | research | E / W / P | AC retains the distinctive database-name words but mislabels it ICH; not a fully clean identification. |
| 39 | coding | E / W / E | AC correctly describes filtering with the unmasked boolean mask. |
| 40 | coding | E / E / E | Omitting self. and using prose does not change the requested attribute name. |
| 41 | coding | E / W / E | Full and AC correctly state alphabetical ordering, despite their high gold NLL. |
| 42 | research | E / W / W | Recorded retrieved article explicitly says 2030. AC substitutes 2025; outside-world accuracy of that article was not audited. |
| 43 | math | E / I / W | Full states the correct rates in prose but truncates before a completed submit call. No-history never reaches a substantive answer; AC replaces rates by ratios. |
| 44 | research | C / W / E | AC supplies the correct two countries with punctuation added. |
| 45 | research | E / E / P | No-history's short failure statement answers the outcome. AC adds an unsupported empty-list claim instead of the returned header. |
| 46 | coding | E / P / P | AC gets unregistration's purpose but invents a dictionary/key mechanism; no-history is only generic cleanup. |
| 47 | coding | E / E / W | No-history paraphrases the already-initialized warning; AC replaces it with a generic usage instruction. |
| 48 | coding | C / E / E | Shortened TypeError and the AC invalid-keyword paraphrase express the same failure; literal error-string matching would miss these. |
| 49 | research | E / W / W | AC outputs 42 instead of 442, despite low .146 gold NLL. |
| 50 | math | E / E / E | Geometric-series closed forms match the reference for rho != 1; all omit the removable rho=1 case. Treat as equivalent with this domain caveat. |
| 51 | research | E / E / E | All views return wind/15 percent. No-history success does not reveal whether this was prior knowledge or a lucky answer. |
| 52 | research | C / W / W | Search results explicitly identify the Cook Islands Te Reo Maori Act as 2003. The 1987 answer confuses it with a different jurisdiction's act. |
| 53 | coding | E / P / P | Both limited views omit members of the exact modifier tuple; AC also loses Meta. |
| 54 | coding | C / W / W | AC replaces the concrete frame type with a generic task_messages description. |
| 55 | coding | E / W / E | AC retrieves _set_optbools; the additional prose is harmless. |
| 56 | research | E / E / E | All state that GenAI lowers barriers. This is plausibly answerable from general knowledge without this history. |
| 57 | math | E / I / E | Full and AC give both dimension pairs. The question is self-contained; no-history spends its output budget deriving the equations without finishing. |
| 58 | research | C / W / C | AC retrieves the recorded solar increase of 25 percent. |
| 59 | research | C / C / E | AC names the right bridge but unnecessarily appends a plate-like identifier from nearby context. |
| 60 | coding | E / P / W | AC invents KeyBindingRule.validate/register and misses the actual binding/validation/wrapping sequence. |
| 61 | coding | E / W / E | AC retains the exact observed 151-test result despite 25k history. |
| 62 | coding | E / P / W | Full's added previously-rerun guard is supported by code. AC omits the failure requirement and invents an attribute-based condition. |
| 63 | research | C / W / P | AC repeats the sea-themed-location clue but never names Atlantica Casino. |
| 64 | math | E / W / W | AC uses coefficient 24 rather than the equivalent doubled correct coefficient 22. |
| 65 | research | C / E / C | No-history omits the percent sign; numerical answer is correct. |
| 66 | research | C / W / E | Removing spaces from the recorded plate is a harmless rendering difference. |
| 67 | coding | E / W / W | Code explicitly defaults to 422; AC/no-history supply 400. |
| 68 | coding | E / E / P | No-history gives the broad wrong-copy explanation but omits the import-path mechanism. AC's main-branch/local-copy story is less accurate. |
| 69 | coding | E / E / E | All correctly decrement the counter; full and AC also explain requeueing. |
| 70 | research | C / W / W | AC substitutes Holy Arms for the requested alternate Infernal Arms, a nearby entity in the same history. |
| 71 | math | E / I / W | No-history never completes; AC carries over dimensions from a different orientation case. |
| 72 | research | I / E / W | Full emits a visit call instead of answering despite .03 NLL. No-history Japanese title matches the gold romanization; AC changes Prologue to a different Japanese title. |
| 73 | research | E / P / E | AC preserves the armistice-date interpretation; no-history incorrectly expands A as All. |
| 74 | coding | E / E / W | No-history method name suffices; AC invents a different method and receiver. |
| 75 | coding | E / P / P | Full repr placeholders refine the reference template; limited views retain only the generic error and omit exact fields. |
| 76 | coding | E / E / W | Positive integer is equivalent to >=1; AC gives a different non-negative-float rule. |
| 77 | research | C / W / W | SpikedSword omits AndShield and is not the requested exact file identifier. |
| 78 | math | C / P / W | AC substitutes f for x in the linear term; no-history gives only the zeroth-order value. |
| 79 | research | C / W / W | AC confuses Princess Connect with Granblue Fantasy: Relink. |
| 80 | research | E / E / E | All correctly explain chauffer/to heat; full's 1.208 NLL is not a substantive failure. |
| 81 | coding | E / E / E | Question asks exception type; all answer ValueError. Gold unnecessarily includes the whole message. |
| 82 | coding | E / P / P | AC retains the general invalid-symbol meaning but changes/omits parts of the requested message template. |
| 83 | coding | E / E / W | Default is 0; AC changes it to 1. |
| 84 | research | C / W / C | AC correctly retrieves Holy Arms; contrasts with its wrong alternative-weapon answer on item70. |
| 85 | math | E / W / W | Full substitutes a=b+cd correctly. AC loses the negative sign of the quadratic coefficient. |
| 86 | research | E / W / P | AC names the metadata questionnaire rather than the WREN partnership initiative. |
| 87 | research | E / W / P | AC gets lying on grass but invents closed eyes/listening and misses the stream/close-to-earth details. Record's literary attribution was not externally checked. |
| 88 | coding | E / W / W | AC changes the compared objects and omits required truthiness checks. |
| 89 | coding | E / W / W | AC substitutes None for NAType in the serialization failure. |
| 90 | coding | C / W / W | Header explicitly assigns 0x00000100; AC returns a different bit value. |
| 91 | research | C / W / W | The squid-faced captain/heart clues strongly identify Davy Jones even without the specific record; both limited views nevertheless fail. |
| 92 | math | E / W / W | Full is algebraically equivalent; AC loses the additive a and fails the x=0 boundary. |
| 93 | research | E / E / E | All answer no. This negative claim is supported by the tool's summary of the retrieved PDF, not an independently audited whole PDF. |
| 94 | research | E / E / W | No-history preserves the autumn/leaf metaphor; AC invents a spring/summer living-organ metaphor. Supplied tool summary backs the gold wording, but the literary date was not externally checked. |
| 95 | coding | E / W / W | Actual code has default 30 and 1<=operation<read; AC changes both the number and inequality direction. |
| 96 | coding | E / W / W | Recorded conversion produces the literal string <NA>, not None. |
| 97 | coding | C / C / C | The valid-header/mismatched-magic question strongly cues false; no-history agrees. |
| 98 | research | C / W / C | Both full and AC retrieve 2006901713. Question erroneously calls this ten-digit value eight-digit; gold identifier itself is supported. |
| 99 | math | E / E / E | First two Taylor terms are computable from the question; AC's verbal answer identifies them without coefficients. |
| 100 | research | C / W / W | AC inserts www into the exact visited URL. Host/redirect equivalence is unknown and was not assumed. |

C/E=correct direct/equivalent; P=partial; W=incorrect; A=abstention; I=unanswered; U=unknown because the reference conflicts. Tool-call validity is separate.
