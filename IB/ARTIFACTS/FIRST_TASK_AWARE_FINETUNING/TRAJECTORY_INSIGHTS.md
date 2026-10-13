# Trajectory QA and compaction: campaign findings

The campaign started September 24 at 01:22 UTC. These are development measurements from fixed source-root held-out panels. They do not establish live task success.

## Starting compaction checkpoint — September 24, approximately 05:46 UTC

The strong 42k-question RAG pair has received **zero compaction examples** at this measurement. The panel contains 100 held-out cuts with complete retained assistant suffixes. NLL and KL below are weighted by supervised tokens, in nats/token; KL uses the immutable initial reader's top-32-plus-remainder distribution.

| Reader input | Whole-suffix gold NLL | KL on the first up to 128 assistant tokens |
|---|---:|---:|
| Frozen full history | 0.6891 | 0.00010 |
| AC, starting RAG pair | 0.8440 | 0.6804 |
| Frozen no history | 0.8645 | 0.8692 |
| Frozen recent text, same row budget | 0.8112 | 0.4631 |

The starting latent channel provides a small benefit over no history, but recent text is currently better. This is a starting point, not a compaction-trained result. AC's KL is 0.6804 on the first up to 128 tokens and 0.1473 on remaining targets. Later teacher-forced tokens receive the correct preceding continuation, which can reduce their need for retained history. The immediate-token panel is therefore essential to interpreting learning; a low whole-suffix KL alone is insufficient.

Source: `IB/TMP/SYNC/TRAJECTORY_COMPACTION/orcd/compaction_50m_v8/report/` (`panels.json`, `report_data.json`, frozen reference files). The tiny nonzero full-context KL is consistent with numerical differences between repeated forwards; the independently measured full-context NLL and stored-teacher NLL agree within 0.00003.

Next evidence: trained panels on this same cohort; QA's first trained panels; later decoded continuations. At a promising checkpoint, measure the trained reader without compressed history as an explicitly separate ablation. This can distinguish improvements due to the latent channel from improvements due to the reader adapter. The frozen initial-reader controls must remain unchanged.

## Starting trajectory QA checkpoint — September 24, approximately 05:54 UTC

On 100 questions from 30 held-out roots, the starting RAG pair scores **0.9458 answer NLL with AC**, versus **1.7759 without history** and **0.2991 with full history**. These are the same frozen initial reader and question/answer tokens; each question was generated after its encoder history was fixed. This supports useful question-independent information transfer before trajectory tuning. Questions from the same root are correlated; this is not 100 independent histories.

The first QA attempt completed its baseline and decoded samples, then exhausted L40S memory during its first eight-example accumulation batch, before any optimizer update. The traceback confirms a reader forward without activation checkpointing: many QA reader sequences fall below the existing 1,700-token threshold despite long encoder histories. The retry enables checkpointing at all training lengths, preserving the same data, targets and schedule. The earlier compaction memory gate used a different accumulation shape; it was not a complete QA memory proof.

A quick read of the six fixed baseline QA decodes shows both successes (an API sandbox URL and specific point-pen segment types) and clear misses (an exact protocol identifier and a coalition name), relative to their recorded answers. One early-bank arithmetic question supplies both operands and is answerable without history. It remains in the fixed panel; do not silently remove it from aggregate NLL. The producer's later prompt already rejects such standalone questions. Larger qualitative readouts should label these cases and distinguish history-dependent gains from formatting or prior-knowledge effects.


## First trajectory QA learning panel — September 24, approximately 06:38 UTC

After **200 unique questions (2,000 × 0.1 epoch)**, held answer NLL is **0.9233**, compared with the starting0.9458. Learning-panel NLL moves0.9172→0.7858. Frozen full/no-history baselines remain0.2991/1.7759. This is a small early gain halfway through the50-update warmup, not convergence evidence.

| Actual history length | Held questions | Starting AC NLL | After200 questions |
|---|---:|---:|---:|
| ≤8k |44|0.8422|0.8278|
| 8k–16k |28|1.0688|1.1049|
| 16k–32k |28|0.9855|0.8917|

Questions share30 held source roots, so length groups are small and correlated. Do not infer a general length advantage from this first panel. Source: trajectory_qa/panel_0000200.json and panels.json. The2k run is diagnostic;16k unique questions remains the main QA target.


At400 unique questions, held NLL improved to0.9028 and learning NLL to0.7254. At600, these returned to0.9481 and0.9183. References stayed fixed. The early trajectory is non-monotonic; the400-question gain does not yet establish sustained learning. Continue the diagnostic and inspect gradients before changing the main recipe.

## September 24, 07:44 UTC — QA at 800 questions

The fixed held answer NLL is 0.939470 (initial 0.945786; best so far 0.902769 at400). Learning NLL is 0.678676. No sustained held improvement yet. A bounded independent review reconstructed all124 answer-token indices with the cached tokenizer and reproduced all panel NLLs. IDs/order/target lengths are unchanged; adapter restoration, penalty routing and moment parking show no concrete bug. The learning24 contains six roots, and only11 of its questions have been presented by800. One three-token answer accounts for0.0808 of the400→600 learning increase; it has not yet been trained. These are real changing probabilities on fixed examples. Continue the complete2k pass before interpreting convergence or changing LR.

The private V11 CPU capacity prevalidation passed review and a focused identity/capacity/fallback check. It preserves recursive side-forward validation and reader limits, binds to the complete prepared identity, and reuses results only for identical prepared objects with sufficient runtime capacities. Actual full-data CPU validation is pending; no training arithmetic changes.

## September 24, 08:11 UTC — first half of QA diagnostic

After1,000 questions, held answer NLL0.926702 versus0.945786 initially; fixed learning NLL0.592890. The400-question held panel remains best at0.902769. Six fixed decodes do not establish improvement: the ORCID sandbox URL changes from the correct `https://api.sandbox.orcid.org` to `https://sandbox.api.orcid.org`; the previously correct three on-curve point types gain an incorrect fourth (`cubic`). The exact Noise protocol and Malaysian coalition remain wrong; the FISA answer now incorrectly adds electronic surveillance. The standalone GCD answer stays correct but does not require history. Full-context controls supply the key details in these cases. Continue the complete2k pass and100-item qualitative comparison; no convergence or robust-quality claim.

The diagnostic checkpointed/requeued at1,000 questions, restoring optimizer/schedule. Maincompaction resumed on H100 at08:05, and the RTX combined32k+8k restart/memory gate passed; its32-example throughput result remains pending.

## September 24, 08:22 UTC — first trained compaction panel

After912 distinct histories and2,608,122 supervised assistant tokens (5.2% of50M), fixed100-held token-weighted NLL improves0.843984→0.803785; frozen full-context0.689061, no-history0.864523, equal-budget recent-text0.811162. Coarsened teacher KL improves0.170550→0.129331 (recent-text0.136686). First128-token KL improves0.680367→0.429199, now below recent-text0.463070; later-token KL0.147264→0.115635. Learning KL0.122467 and NLL0.867121. This is early offline teacher-forced evidence, not live continuation success.

All three source groups and all three length bins improve. The longest bin (36items,16k–32k) still trails recent text: NLL0.91004→0.87199 versus recent-text0.85670 and frozenfull0.72791; KL0.19171→0.15141 versus recent-text0.14119. Math also still trails recent text slightly. Shorter histories and research account for more of the control-beating aggregate. Maintain the50M run before inferring a final long-history limit.

RTX PRO6000 completed the four-update/two-process combined32k+8k gate and the full32-example runner, exit0. The long gate retained immutable teacher/base outputs and restored optimizer/RNG, with finite nonzero gradients on both trainable sides. This clears hardware execution, not quality. A separate actual vLLM FP8/micro16 synthesis pilot with a reviewed conditional handoff is queued as23636639. CurrentL40Sproduction remains active until that pilot succeeds.

## September24,09:05UTC — initial100-question qualitative review

All100questions/300responses were reviewed manually, with25original histories checked selectively. Semantic correct: frozenfull96/100, initialAC40/100, frozennohistory26/100. AC gains23answers and loses9 relative to nohistory. The cohort has30roots; no items were removed. It demonstrates specific information transfer before trajectory tuning, with a substantial gap on names, identifiers, digits and mathematical signs. One conflicting reference staysunknown for everyview; two question-wording defects are documented. Some questions are independentlysolvable. These are manual judgments, not a calibrated automatic score or trained-checkpoint result.

The report is [100 annotated comparisons](TRAJECTORY_QA_INITIAL_COHORT.tressoir.html), with complete question/gold/responses and per-item explanations. FreshCSAILNLL is .299322full/.944860AC/1.772159nohistory; slight differences from ORCD baselines are retained rather than called exact replay. Source hashes are verified. Static HTML/ID/case-count checks passed; an actual extension-webview check was unavailable in this session.

At1400unique training questions, heldNLL.954150 and learningNLL.552099: still no sustained heldgain. Keep the maincompaction50Mrun unchanged. P2reconstruction-initializer comparison and differentiable shared-history encoding gate both started on separateH200allocations onCSAIL at~09:02. The latter has a bounded correctnessreview pass and makes no resume/currentmicro1-speedupclaim.

## September24,09:38UTC — QA initializer and optimization gates

The complete originalP2 reconstruction pair scored answerNLL3.48422 withAC, .41113 fullhistory and1.74772 nohistory on the same100QA items. The strongRAG initializer remains muchbetter withAC (.94486 onCSAIL). P2 predates rowreadouts and loads their documented identity migration, so this is evidence about this checkpoint as loaded, not a general rejection of reconstruction initialization. Keep the strongRAG pair for the mainQA run.

At1800questions, diagnostic heldanswerNLL.918263 andlearning.487121 (initial.945786/.917222, bestheld400Q.902769). The full2000 pass finished updates: heldNLL.912029, learning.526122,87,865 supervised formatting+answer tokens; finalsave is pending. Selecteddevelopmentpair remains400Q/.902769.

Sharedhistory encoding failed its fixedweight gradient gate: long4Q encoder cosine.56148/relativeL2.87378, readercosine.96907/L2.25199, while duplicate-repeat is near1. The firstcomparison changed encoderbatchshape and maskbranch (singletonNone versus batchedallones). Do not promote or dismiss as harmless noise. A bounded followup forces every encoderforward to singleton with the existing tokenbudget knob, leaving readerbatch and tolerances unchanged. JobCSAIL2203333.

Legacy27B synthesis failed with filesystem Errno36 (overlong cachefilename), not a modelgeneration failure. Recovery2203330 uses a shorter symlink whose basename preserves the same modelrevision, with unchanged producer/source/cursors. A small general cachefilename fix and real filesystem roundtrip check are included in the now13-file sourcehandoff.

A fixed ten-item spot check of the P2 cohort (items1,2,3,7,14,25,40,60,80,100) shows frequent tool-schema/prompt reproduction rather than answering the future question. This is consistent with task-format transfer failure, not evidence that no information was encoded. The complete100 outputs are retained under `IB/TMP/SYNC/TRAJECTORY_COMPACTION/csail/qa_cohort_p2/report/`; identical held IDs/order verified against the initialRAG cohort. No semantic accuracy count is inferred from this spot check.


## September 24, 10:18 UTC — compaction continues improving; QA gains do not survive semantic review

Compaction at **1,848 distinct cuts / 5,210,764 supervised assistant tokens**: fixed held NLL **0.798607** (initial0.843984;912cuts0.803785), teacherKL **0.123373** (0.170550→0.129331), first128KL **0.394914** (0.680367→0.429199), laterKL0.110970. FrozenfullNLL0.689061, nohistory0.864523, recenttext0.811162. The longest36items improve toNLL0.864805/KL0.144050, but still trail recenttextNLL0.85670/KL~0.14119. This is continued offline teacher-forced progress at10.4% of the50M budget, not live success or proof that all gains arise in the encoder.

The complete [trained100-question comparison](TRAJECTORY_QA_TRAINED_COHORT.tressoir.html) has700 judgments across7views. InitialAC40correct→400Q40→2000Q41; final gains9 and loses8 versusinitial. Coding14/42→10/42, research18/43→22/43, math8/15→9/15; longest9/28→8/28. Frozenfull96/100. Finalreaderwithout history22correct versusfrozen26; complete submit calls96→87 despite slightly better NLL. FinalAC still supplies25correct answers absentwithout history and loses6 nohistory successes. Keep the channel-usefulness claim separate from a training-quality claim.

Freshcohort ACNLL initial0.944860,400Q0.909373,2000Q0.908110; finalnohistory1.758753. These fresh measurements remain distinct from trainingpanel0.912029. IDs, questions, gold and history lengths match; different hardware/kernel evaluations are not bitwise replays. Same reference caveats retained, noitemsremoved. HTML static100case/700response/unique-ID checks passed; actualextensionwebview unavailable.

A bounded same-weight1/8row-budget evaluation is queued, retaining all1/16controls. Main16kQA waits for enough acceptedtrainingquestions and this recipe evidence. A cross-task compaction-pair QA evaluation can test whether broader question-independent supervision helps more than sparse short answers, with source-root separation checked before interpreting it.


## September 24, 10:24 UTC — a valid speed opportunity and a concrete live failure

The follow-up encoder-sharing gate **passes all four short/long × 2/4-question cases** when both sides use singleton encoder forwards. On H200, four questions over32,754source tokens take13.892s with separate encodings versus6.064s with one shared differentiable encoding; allocated peak72.88→45.83GiB. The loss is identical0.346734762; shared encodergradient cosine0.999841/relativeL2.018759 and reader0.999959/.009059, within the unchanged gate. This isolates the earlier encoderbatch-shape confound; it does not validate the old mismatched-batch result.

Before promoting it, a reviewed followup **CSAIL2203529** compares the actual singleton-reader production loss path with immediate backwards, combined separate-encoding backwards, and combined shared-encoding backwards. It has real allocated Adam moments parked onCPU, repeated controls, exact request keys and wholeACOutput sharing. Grouping questions by history would be an explicit training-order change; no claim yet about logical8 capacity, restart or full-run speed.

The initial privateHF live-tool probe completed all18conditions. Frozenfullhistory succeeds **6/6**; nohistory **0/6**; initialAC **0/6**. AllACconditions carried inputrows and all8deepinterventionlayers, with no runtime errors. AC repeatedly attempts to reread one-time packets; the8kcases then submit fabricated release details/receipts without a successfulcommit. Fullhistory makes the exactcommit then submits its issuedreceipt in2turns in everycase. This is a useful executed-tool failure, not evidence that the channel works for live continuation. The six fictional releasepacket fixtures deliberately stress exact fields; they are not a natural benchmark or productionvLLM validation. A selectedcompaction1848pair is being staged for the identical trained comparison.

Cross-task QA from the samecompaction1848pair is queued **ORCD23640218**, after the1/8row-budget evaluation releases itsGPU. It preserves the same100futurequestions and checks trainedreaderwith/without history. All30QAheldroots are disjoint from18,408compactiontrainingroots; trainingbudgets differ and are labeled.


## September24,11:05UTC — first real compaction continuation success

The5.21M-token/1848-history compaction pair succeeds on **1/6** fixedlive releasepacket cases, versusinitialAC0/6,frozenfull6/6,andbothnohistoryviews0/6. In the16k/20%-position case it recovers everyfield, executescommit_release correctly and submits the actualreceipt. No fixturetraining or factualsummary was used. Three of six firstactions nowattemptcommit (initialACalwaysreread); two of those commits have wrongsettings. The remainingthree submitanswers withoutcommitting. This is a small but concrete livegain, with mostcases stillfailing and exactidentifier/digiterrors remaining. All12trainedconditions completedwithout runtimeerrors in5m39s onH200. Rawtraces: `IB/TMP/SYNC/TRAJECTORY_COMPACTION/csail/live_compaction_1848_v1/report/`.

Sharedencoding also passes the **actualproduction singleton-reader gradient gate** in allfour2/4Qshort/longcases, with1230allocatedAdammomenttensors parkedonCPU. Long4Qproduction22.21s/37.99GiB versusshared14.44s/39.18GiB; about35%less time, modestlymorepeakmemory than immediatebackwards. Combinedseparateencodings use66.81GiB, illustrating why capacitymustcover unsharedhistories. Sameweightobjectiveisidentical0.355430; sharedencodergradientcos.999901/relL2.014123,reader.999957/.009326, withinunchangedbounds. This updates the earlier2.3× comparison, which usedbatchedreaders.

A final boundedreader-batch comparison2203823 keepsencoderB1fixed and compares theexistingbatchedreader loss toproduction andsharedsingletonreaders. It can expose a larger safespeedup orreject it. Actual32-question logical8/micro4worker CPUpreparationcompleted2203708(1m37s),includingfull32768token singletons andfour-questiongroups; GPUrestartgate waits for thechosen readerpath. No mainrunpromotionyet.

Legacy27B RAG synthesis recovery is producing data again:57,000 published/58,033 accepted at10:58UTC, up4,000 published during the first recovered segment. The sameproducer automatically requeued and is awaiting capacity. This is separate from the trajectory-question bank.


## September 24, 12:52 UTC — compaction gain depends on the latent history

On the unchanged100-history cohort, the912-history compaction pair scores token NLL **0.803791 with AC** and **0.864302 with the same trained reader but no history**. Frozen no-history NLL is0.864523; initialAC0.843984. Thus the trained reader's history-free performance is essentially unchanged while AC improves by about0.0402nats/token. This supports improved use of the compressed history; it does not isolate encoder adaptation from reader adaptation to AC. All100 teacher-forced measurements are present; trained no-history greedy decoding is still underway. Evidence: `IB/TMP/SYNC/TRAJECTORY_COMPACTION/orcd/compaction_cohort_912/report/result.json`.

Suffix analysis of the existing1848-history full-supervision run: first512 assistant targets contain17% of held tokens but about51% of the measured NLL improvement. Targets after2048 still account for24% of improvement, and frozen full context continues to beat no-history on46/50 histories in this late range. Shorter supervision is therefore a useful experiment, not an established lossless optimization. The prepared private912-history comparison preserves every encoder input and the full held100; only training supervision changes.

## Sep24 13:30UTC — continuation control reaches 10.24M targets

3,688 distinct histories in the first pass (0.2003 epochs), 10,243,959 supervised assistant tokens: held NLL **0.787066**, KL **0.111015**, first128 KL **0.340314**, later KL **0.100541**. Previous1,848-history held NLL0.798607; initial0.843984. The continuation control is still improving; this is not yet a live-task success claim. New summary-SFT study will freeze controls under its own current resume cue, so do not splice its losses into this curve.

Private RTX tests: output-head alternatives passed objective/gradient gates but gained<0.7%; keep native. Shortened512/2048 reader views failed gradient equivalence to fullforward masked to the same targets; no prefix912 training launched. Compiled normalization retry preserves eager precision casts and retains the same gates.

## Sep24 14:04UTC — summary-training execution proof

The private new worker completed32 real histories with short handwritten summary fixtures, including a32,767-token history. It stopped after two logical8 updates and resumed for two more, restoring615 Adam states and penalty RNG exactly; fixed controls, selectedtargets and actual example order remained unchanged. Both sides had finite nonzero gradients; peak42.62GiB on the temporaryRTX. This is a plumbing test with disposable weights, not oracle-summary quality evidence. Receipt COMPACTION_SUMMARY_GPU_GATE.json.

Continuation control paused after3,888 histories at its normal allocation checkpoint; the last scored panel remains3,688/10.24Mtargets. This preserves compute priority for8k QA and the official summary objective. The initial no-reasoning27B summaries contained unsupported confirmations and invented restrictions in a quickspotcheck. A short prompt clarification did not fully resolve them; bounded-reasoning oracle generation is being tested before a quality training run.


### September24,14:18UTC — Capacity control and reusable targets

The same100 trajectory-QA items show no aggregate gain from doubling AC rows at fixed weights. For the2k-trained pair, answer NLL changes0.90811→0.91400; for the initial pair,0.94486→0.95469. This tests row count at existing weights, not training a new ratio. The8k×1 run retains1/16.

The independent continuation-teacher cache GPU gate passed on8 stress cuts: exact stored payloads, no target forwards on cache hits, missing-only fallback and unchanged RNG/snapshot. Cached-gradient differences matched ordinary repeat noise. A single vocabulary projection halved head calls but only reduced teacher time11.49→11.22s. Neither is promoted into main training; the official summary-SFT objective reduces the urgency of optimizing the old continuation teacher.

The temporary EC2 node was terminated after the real summary training/resume proof and optimization gates. Reusable trajectory QA generation now runs microbatch16 on ORCD RTX after its handoff pilot passed. The summary bank remains paused for fidelity: the four-case bounded-thinking probe is queued before any targets are consumed by training.
