# AC training performance

Approved September24 at15:24UTC. Make hardware/workload-specific execution tuning reusable, and reduce8k-question insight turnaround toward3–4h. Two reviewers inspect throughput and numerical integration; no additional approval is needed for these authorized changes.

## Executive Summary

Current L40S executes logical8 as eight singleton examples, with both models checkpointed and Adam moments onCPU. Measured40.35s/logical8=714questions/h, or11.2traininghours/8k beforepanels. Actual frozen8k means:14,080source tokens,14,960encoder positions includingrows,1,332reader tokens,451no-history tokens,43target tokens. The existing autotuner tunes FLA kernels and heuristic scratch limits; it does not measure complete-workload batching.

Use existing autotune/core/worker and AC utils for workload profiling, measured policy selection, shared encoding and moment placement. Keep trainer integration small and explicit. Prioritize faster execution on H200/H100/RTX PRO while preserving the original history distribution, including 32k cases. Shorter histories are an optional fallback only if measured end-to-end turnaround still requires them; a 4k median is not a goal. Test the short fixture as a performance comparison, not a commitment to shorten training data. Never truncate evidence beneath existing answers.

| Change | Purpose | Boundary |
|---|---|---|
| Reader microbatch1/2/4/8, encoderbudget independent | BetterGPU utilization | Same logicalupdate, itemweights/targets/penaltychoices |
| Measured checkpointing + Adamplacement | Trade memory for less recomputation/transfers | RealAdamallocated and worst distinct-history graph fits |
| FullAC reuse within backwardgroup | Avoid repeatedhistoryencoding | Complete requestidentity; no reuseacross updates |
| Optional history-grouped question order | Make reuse frequent | Explicit experimental recipe, not hidden autotuner change |
| Optional shorter-history bulk | Reduce arithmetic only if optimization + larger GPUs remain insufficient | Choose the least shortening needed; synthesize fresh questions after window selection |
| 16k/32k coverage | Preserve/test long-context skill throughout | Original held100 kept; length panels guide any later adaptation |

The throughput target is2,000–2,667questions/h end-to-end: roughly10–13s/logical8 after reserving reporting overhead. RawGPUpeak is not a promised achievable training rate. Results must distinguish measuredspeed from projections, data changes from execution changes, startup/queue from steadytraining.

## Requested Decisions

Accepted inchat: invest inreasonable performance changes, includingcomplexones whencontained inexistingautotune/utils; use two reviewers; testL40S,RTXPROspotEC2(max2–3h),H200orH100ORCD/CSAIL. One newEC2spotallocation is authorized; terminatewithin3hfromlaunch withindependentexpiry. PreviousEC2exceptionwasalreadyterminated. Keep reusable synthesis productive. PreservecurrentQAcheckpoint before GPUrepurposing. Existingfrozenruns are not edited in place.

User clarification (16:22 UTC): optimization plus larger GPUs should remove the need to over-shorten. Keep the original useful workload when it meets the turnaround target. The queued short-profile producer is held while that comparison finishes; existing synthesis continues.

## Milestones

<details class="card" data-tressoir-markdown>
<summary><span class="card-title">P1 — Measured execution profiles</span><span class="card-oneliner">Choose the fastest validated policy for real shapes.</span><span class="card-badge">Implementing</span></summary>

#### Planning Overview

Extend existingautotuner/core/worker with a bounded disposablewhole-model probe and cached receipts. Include representative source/reader/outputlengths, repeat-sharingfrequency, distinct-historytail, model/LoRA/deeparchitecture, dtype, loss/penalty recipe, GPU/software. Reuse exact nativeoptimizer/math; allocate momentstate beforepeakmeasurement. Prefer simplecandidate ladders and retainfittingfallback. Explicitcaller settingswin. Automaticselection istraining-only; reference/report arithmeticstaysfixed. Probe freshworkerweights/RNG, not liveoptimizerstate.

#### Planned Changes

`activation/autotuners/core.py`, `activation/autotuners/worker.py`, `activation/autotuners/__init__.py`

```diff
+ summarize a small representative prepared workload
+ benchmark full forward/backward/Adam candidates with memory margin
+ select measured questions/second; publish validated execution receipt
+ cache by hardware, model/recipe and workload shape coverage
```

Focusedbehavioralhunks; exactintegration follows thetwo reviews. A fittingbatch alone isnot a throughputresult. Unsupported/OOMcandidates failindependently and donot become cachehits.
</details>

<details class="card" data-tressoir-markdown>
<summary><span class="card-title">P2 — Contained execution changes</span><span class="card-oneliner">Batch and share without changing the learning objective.</span><span class="card-badge">Implementing</span></summary>

#### Planning Overview

Keep objectiveconstruction andupdateclock native. Drawpenalties once in canonicalorder before executionregrouping. Keep logical8 and per-itemnormalization, includingpartialbatches. Share completeACOutput tensors onlyinside onebackwardgraph; separate mutable readerpayloads. Start withprovenencoderB1 while testingreaderbatching. Treat encoderbatchshape as itsown numericalcandidate. Adaptivepolicy mayfall backbefore anoptimizerstep, restoringRNG/gradients; neverretry apartiallyfailedAdamstep.

#### Planned Changes

`activation/ac_model/ac_model_utils.py`, `activation/ac_model/ac_model_training.py`

```diff
+ utility-owned batch grouping, request reuse and optimizer placement
+ small trainer policy/application hooks with canonical penalty assignment
+ separate training batch execution from fixed evaluation/reference path
- compute every intervention gradient norm at every update
+ compute intervention norms only at actual diagnostic points
```

Checkpoint thresholds must includerelatedencodergraphs/gradients, notjustshortreaderlength. Existing1700thresholdOOMedL40S onlonghistories; do notblindlydisablecheckpointing. Groupingsiblingsacrossupdates isexplicitrecipeoption. No newutilsfile, no pooling/targetshortening or architecturechange.
</details>

<details class="card" data-tressoir-markdown>
<summary><span class="card-title">P3 — GPU comparisons and restart</span><span class="card-oneliner">Prove correctness and report attainable speed.</span><span class="card-badge">Planning</span></summary>

#### Planning Overview

Run the sameboundedworkload onL40S,RTXPRO96GB,andH200/H100. Include shortunequallengths, sharedhistories, distinct8ktail, pluslong32kstress; keepa separatecompaction-shaped sample. Compare baseline/repeat/candidates forloss and bothgradientgroups againstexisting tolerances. Do notrelaxgradtolerancesforspeed. Finalwinner getsreal2+2save/resume gate andthroughput includingAdam/report/checkpointcost. No needexhaustivecartesiansearchorlongqualitytraining toverifyexecution.

#### Planned Changes

`IB/ARTIFACTS/FIRST_TASK_AWARE_FINETUNING/bench/`, existing GPU launchers

```diff
+ reproducible representative workloads and whole-step measurements
+ hardware comparison receipts, numeric gates, selectedwinner restartproof
+ spotexpiry/teardown receipt and stableHTML/liveJSON reporting
```

OneRTXspot, hardexpiry<=3h fromlaunch; launchwhen runnableinputsready. ORCD/CSAILtestsuseexistingguardedaccess. PausecurrentQAonlyafterpairedsave; keep synthesisrunning. Reportqueueshonestly.
</details>

<details class="card" data-tressoir-markdown>
<summary><span class="card-title">P4 — Optional shorter-history curriculum</span><span class="card-oneliner">Use only if faster execution and larger GPUs still miss the turnaround target.</span><span class="card-badge">Planning</span></summary>

#### Planning Overview

First measure the optimized original workload on larger GPUs, including median and distinct 32k histories. If that delivers useful 8k-question turnaround, retain the original lengths and existing bank; no short-data requirement or synthesis wait blocks training. If it remains too slow, choose the least restrictive history profile supported by measured timing and quality. A near4k/max8k curriculum remains an available fallback, not the selected default.

The short-window producer, portable metadata and balanced selector are implemented and CPU-checked. The replacement producer is held before allocation; the existing long-history producer continues. If activated later, choose complete-turn windows before question synthesis, retain original frame/tools and all old portable rows, and preserve the 50/30/20 source mix. Keep the original held100 and use length panels to decide adaptation.

#### Planned Changes

Existing `bench/generate_trajectory_questions.py` and `bench/run_trajectory_tasks.py` data builders

```diff
+ explicit short-history generation profile and portable chosen-window metadata
+ backwards-compatible old/newwindow loading and balanced shortselection
```

No clipping existinglongQAs. Synthesislatency andqueueingareseparatefromthe3–4htrainingtarget. Saveallportabledataforlaterlarger/longerruns.
</details>

## Validation and review record

Initialreviews identifygenuineintegrationrisks: microbatch currentlychangespenaltyassignmentandreference/evalbatching; sharingrequiresonegraph; L40Slonggraphmemorycannotbeestimatedfromreaderlengthalone. Allacceptedforimplementation. Exactsource/patch/reviewreceiptswill beupdatedbeforeGPUtests; /source remainsuntouched.

## Reporting

Currentcampaign overview: `/workspace/IB/TMP/SYNC/TRAJECTORY_COMPACTION/report.tressoir.html`. Performanceevidence and launchreceipts live inthissameartifactfolder; existingHTML remainsstable withJSONupdates.
