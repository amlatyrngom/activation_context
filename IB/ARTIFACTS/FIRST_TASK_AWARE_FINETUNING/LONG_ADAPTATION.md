# Adaptation to additional retrieved context

This comparison asks whether brief training with much longer evidence improves AC robustness. It reuses questions already trained in the completed learning-rate wave. It provides no fresh-data result; the original 100 development questions have already been used for tuning.

Both arms start from the same frozen control checkpoint at update 900. The sample contains 640 of that wave’s 4,000 training questions: 192 code, 160 numeric, 160 web and 128 documents. Both retain the exact original 100 development records, the same 4,100-record retrieval corpus and the same 32 training questions for learning panels.

| Setting | Short-context control | Long-context adaptation |
|---|---:|---:|
| Report/run name | `long_adapt_2k` | `long_adapt_32k` |
| Evidence token budget | 2,048 | 32,768 |
| Maximum passages | 4 | 64 |
| Training questions × epochs | 640 × 1 | 640 × 1 |
| Optimizer updates | 80 | 80 |

Shared settings: 2B encoder → 9B reader, 1/16 compressed rows per encoder source token, microbatch 1, eight examples per update, encoder/reader learning rates 5e-5/2e-5, 50-update warmup and cosine decay to 10%. Gold SFT and the original-base no-context KL penalty remain unchanged. This is a new optimizer/schedule initialized from the paired weights; it does not resume the old optimization history.

The explicit common retrieval corpus uses the current builder’s deterministic question-local passage shuffle. Consequently, the new short-context initial panel can differ from the historical control-900 score despite identical weights and questions. Compare against each new arm’s measured initial panel.

CPU checks covered all 640 training and 100 development items with the real tokenizers. Gold assistant targets were identical across arms; full-context evidence inclusion and exact compressed-row counts passed.

| Actual encoder inputs | Training median / p95 / maximum | Development median / p95 / maximum |
|---|---:|---:|
| 2k evidence arm | 1,958.5 / 2,933 / 6,021 | 1,761 / 2,061 / 2,096 |
| 32k evidence arm | 32,926 / 33,137 / 33,272 | 32,942 / 33,181 / 33,270 |

The 32k arm uses 1,910–2,080 compressed rows in training and 1,996–2,080 in development. The minimum actual encoder length is 30,549 training / 31,927 development: a few questions reach the 64-passage limit before filling the evidence budget. Encoder lengths include question/template overhead; required gold sources are retained whole, explaining the short arm’s occasional inputs above 2k. No pooling is introduced.

Live panels appear at initialization and every eight updates (10%). Each arm’s full-context reader receives exactly the evidence supplied to that arm’s compressor, including its distractors. Full/no-context controls use the current reader adapter and can move during training.

**Required final comparison:** evaluate both final paired checkpoints on both fixed 2k and 32k development inputs. For all four combinations, report AC plus matched full-context and no-context views. Include the original control-900 values measured at each arm’s initial panel. Improvements on the same 32k inputs establish adaptation to extra context; the 2k comparison checks short-context retention. Comparing the two live curves alone would compare different evidence inputs. Keep literal answer NLL distinct from generated-answer correctness.

Material and receipts are under `IB/TMP/FIRST_TASK_AWARE_FINETUNING/long_adaptation_material/`: `material.json`, `initial_pair.json`, `input_lengths.json` and `summary.json`. The latter records exact input lengths and hashes. The two SBATCH files beside that folder target `bench/run_qa_night_v2.py` with `train_ws_night_v1`; no GPU jobs were submitted during preparation. Explicit paired `--resume` is available after interruption; automatic training requeue is disabled.

At the measured 44.9 seconds per eight-example 32k update, optimization takes about 60 minutes. Eleven 100-development/32-learning panels at roughly 4–5 minutes each add 44–55 minutes. Allow about 1h50–2h10 including loading and saving; the scheduler limit is three hours and the trainer ceiling is 9,600 seconds. The short-context control should finish substantially sooner. Actual rates and input lengths determine the final estimate.

Recovery saves occur every8updates (each10% panel) in both arms, limiting long-update preemption loss; training schedule/data are unchanged.

The prepared final evaluator is `bench/evaluate_long_cross.py`, launched by `bench/long_cross_eval.sbatch` with the pinned `bench/run_qa_long_cross_v2.py` builder. It rescores control900 at both evidence budgets because historical panels lack a saved complete-input fingerprint, then evaluates both final adaptation pairs at both budgets: six fresh processes, microbatch1, no optimizer. Outputs under `reports/long_cross_eval/<checkpoint>__<2k|32k>/` contain answer-only NLL for all100 questions and all three views, actual lengths/rows, timing/memory, and up to12 fixed decodes per view when the time budget permits. `summary.json` reports each final pair versus the rescored initial pair and long versus short adaptation on identical inputs. The three-hour allocation can be rerun for individual cell names passed as positional SBATCH arguments; matching completed cell outputs are skipped unchanged, and interrupted cells retain completed views. It rejects incomplete/mismatched final pairs before loading models. Focused CPU checkpoint, mask, paired-delta and restart checks passed; The two2k cells completed; remaining32k/control and long-final cells await a released GPU slot.

## Resource check — 2026-09-21 23:33 UTC

Keep paired resume 23431235 queued on ORCD RTX, excluding node5101; cross-evaluation 23431251 remains dependent on it. The latest saved pair is update 24/80, with optimizer and no-context reference intact. The scheduler reason changed from Resources to Priority during this check; no start estimate was available. The next 16k fresh-data runs retain priority.

ORCD `mit_normal_gpu` is non-preemptable and would require only partition/GPU request changes, retaining the same paths, environment and resume identity. The account has room for one more normal-tier GPU alongside the 9B generator (limit 2 GPUs / 32 CPUs / 515 GB). However, all 104 H200s in that partition were allocated or reserved, with 21 single-H200 jobs waiting on priority; there was no immediate capacity advantage. If this option is used later, replace the pending continuation and its cross dependency under root control so only one continuation can run and the four-autonomous-GPU cap remains enforced.

CSAIL access is live and no jobs of ours were running. The shared H200 tier showed four unallocated GPUs on `khattab-h200-1`, but our account still has only shared, preemptable access; no quanta account was granted. The existing Python 3.14 / torch 2.13 / Transformers 5.15 / FLA 0.5.2 environment, H200 autotuning cache and 2B/9B model caches are present. Thus a fresh future run needs source/material shipping, not environment bootstrap, and avoids most continuation migration work.

Moving this continuation would require roughly 6.8 GiB of AC/reader weights and optimizer combined, 53 MB frozen material, the 7.5 MB no-context reference, current training source/runner and report/identity files. The old CSAIL trainer differs from the current ORCD snapshot. In addition, v3 checks that the original initialization's ORCD absolute paths exist even during `--resume`; moving it needs a narrow private resume-path adjustment while preserving the original receipt hash. Final checkpoint/result paths and the ORCD cross-job dependency also need a deliberate return handoff. That effort is not justified solely by speculative queue improvement while CSAIL remains preemptable. No jobs, queues or remote files were changed during this investigation.


### Prepared 32k passage-layout screen

The short-context frozen screen did not help when passages were encoded separately. At32k, separate encoding also keeps each encoder call near its trained input length, so its result may differ. Prepare a bounded no-training comparison on the same100 held questions and the earlier4,000-question×1.8-pass pair: merged versus separate question-conditioned passages,1/16ratio,identical total rows per question,32k evidence/up to63distractors,microbatch1. Keep exact evidence,gold,and readerweights fixed; report repeated-question framing overhead. Recompute merged controls in the same execution conditions before comparing. Optional12decodes;40minuteGPUbound. CPUinput validation job23433036 checks actual materials,identical full-context inputs/gold,and exact row equality. GPU submission waits for those checks and available campaign budget;the next16k×1 comparisons have priority. The private screen worker only gained explicit context/passages/microbatch arguments; the original deployed2k worker remains unchanged.

## Avoid repeated resumed baselines — Sep22, private v4

`bench/run_qa_night_v4.py` adds an explicit `--resume-baseline-receipt` option. For the currently published update24 pair, pass `bench/long32k_resume24_baseline_receipt.json`. The receipt refers to the completed actual baseline from `attempts/resume_002/resume_baseline.json`, measured on RTX during job23431235. It checks the frozen material/order/config, paired resume paths/update, checkpoint file sizes/timestamps, source-summary hash, unchanged input builder/trainer source, and all four summaries' item IDs, target lengths, answer masks and scores. The existing immutable checkpoint files all predate that completed baseline; this legacy adoption uses file metadata, not retrospective weight-content hashes.

On a match, a one-use private instance wrapper supplies only the trainer's first held-AC summary and immediately restores its normal evaluator. The existing callback then reuses the other three baseline summaries. Reports and `baseline_reuse.json` explicitly label prior actual measurements, including provenance; they are not new measurements on the resumed GPU. Later scheduled panels evaluate normally. A different checkpoint or any failed receipt check falls back to the original baseline path. V3, product files, optimizer restoration, seeded shuffle/skip24, penalty RNG advancement, and the original80-update schedule are unchanged. No automatic job retry is added.

Focused CPU checks rebuilt the actual100 held and32 learning inputs at32k, validated every saved summary's IDs/targets/masks/scores, proved one-call reuse followed by normal evaluation, and checked that optimizer/update/RNG state is untouched. Changed pair/update metadata is rejected. A separate callback fixture proved the reused branch performs no evaluations and writes honest provenance. Evidence is under `IB/TMP/FIRST_TASK_AWARE_FINETUNING/v4_resume_cpu_checks.json` and `check_v4_callback.py`. GPU execution of v4 remains pending; no job was submitted during preparation.

A normal-tier L40S continuation with the same microbatch1 is plausible but unproven. The41.07GiB CUDA allocated peak came from the successful32k smoke's whole training/evaluation window with8 training and4 held examples; it does not certify the largest item in the full640/100 run. Physical L40S capacity leaves only a few GiB for non-PyTorch allocations and variation. The preempted full run did not publish a final peak-memory statistic. Preserve the update24 pair if a controlled L40S attempt runs out of memory; do not silently change the training recipe. Reused baseline values remain explicitly RTX measurements if subsequent training runs on L40S.


## Current status — Sep22 00:55 UTC

The L40S continuation failed before any new update; do not retry that device for this exact workload. The saved 640-question × 0.3-pass pair remains intact. Private v4 successfully reused the verified prior baseline. CSAIL migration is fully staged, but its job 2110090 was canceled while still pending. ORCD RTX job 23438900 now resumes the same optimizer, item order and schedule, with cached actual baseline measurements. No simultaneous CSAIL continuation runs. On completion, the original ORCD cross-evaluation path can be used directly.

The frozen 32k layout screen completed: merged NLL 0.8758 versus separate 0.8019, with equal total rows. The uncertainty interval crosses zero, gains concentrate in numeric questions, and the 12 decodes include regressions. See [the detailed result](PASSAGE_LAYOUT_32K.md). Job 23439119 expands the decoded comparison to all 100 held questions under a 55-minute bound while the larger fresh-data material fills. This is evaluation only; separate-passage training is not queued.

## Completed adaptation — Sep22 02:49 UTC

Both arms finished640 familiar questions×1additionalpass. The long run retained optimizer state through its recorded preemptions; final paired publication is verified.

| View | Short2k initial | Short2k final | Long32k initial | Long32k final |
|---|---:|---:|---:|---:|
| Held AC answer NLL |0.59163|0.66096|0.87578|0.82605|
| Held full-context NLL |0.41853|0.43099|0.37666|0.38643|
| Held no-context NLL |1.87817|1.87042|1.87817|1.86920|
| Fixed learning NLL |0.48830|0.33602|0.71257|0.45118|

Long adaptation improves its own32k panel by0.04972nats/token (5.7%); the gap to matched full context remains0.43962. This is development data from a familiar-question continuation, not evidence of fresh-question sample efficiency. Short adaptation regresses despite its learning improvement. A matched L40S replay confirms short NLL0.59047→0.65930; hardware/kernel variation relative to theRTX panels is about0.001. Cross-evaluation of the long-final pair on2k remains necessary before claiming short-context retention.

Final long pair: `taskaware/long_adapt_32k/AC_MODELS/rag_poc/epoch_001` and `LORAS/rag_reader/epoch_001`; published epoch marker verified, job23438900 complete. Final result timing/steps describe only its last resumed attempt; completed exposure is640×1, not640×0.7. No separate-passage training is queued.


## Final cross-evaluation — Sep22 10:00 UTC

All six cells completed; final job23452012 exited0:0. Same original100 development questions and frozen inputs within each evidence budget, microbatch1, no optimizer. This older study is separate from the newer16k/32k-question campaign.

| Pair | AC with2k evidence | AC with32k evidence | Full with2k | Full with32k |
|---|---:|---:|---:|---:|
|Original4,000-question×1.8-pass pair|0.59047|0.87578|0.41839|0.37666|
|Then640 familiar questions×1 at2k|0.65930|1.00484|0.43144|0.38690|
|Then640 familiar questions×1 at32k|0.64053|0.82605|0.43362|0.38643|

Long-context adaptation improves32k AC NLL by0.04972 but worsens2k AC NLL by0.05006. Short-context repetition worsensboth2k and32k results. The long continuation therefore trades some short-context quality for distractor robustness; it does not close the large full-context gap. Neither arm establishes fresh-data sample efficiency or generated-answer improvement.

The earlier short cells ran on L40S and remaining cells on RTX; their tiny no-context replay differences are execution differences, not changing checkpoints. The newer context-transfer screen runs both lengths on the same RTX with identical no-context vectors. Existing old decodes keep their original stopping protocol; NLL is the primary result here.

Raw scores and per-question changes: `IB/TMP/SYNC/TASK_AWARE_QA/long_cross_eval/summary.json`.
