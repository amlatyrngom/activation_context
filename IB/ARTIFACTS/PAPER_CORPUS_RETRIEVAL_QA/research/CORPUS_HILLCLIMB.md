# Corpus Q/A hill-climb: key/value transfer (Oct 19; written by the hill-climb fork)

The user's go-ahead, verbatim (Oct 19, ≈ 01:50Z): "1, 2, 3 all make sense. For mix, just do all ten. No big deal." Their earlier condition: hill-climb as I see fit, "as long as it stays interpretable of course". LoRA rank stays 128.

## Goal and comparison rule

- **Target:** K = key/value transfer at 1/15, with Llama-3.1-8B as both encoder and reader.
- **Metric:** accuracy on the 200 LongHealth questions, all ten records in view, direct letter.
- **Comparison:** each run is read against K15's curve at matched Q/A pairs, and differs from its reference by one named lever.
- **References** (accuracy with all ten records, unless noted):

  | Reference | Accuracy | At |
  |---|---|---|
  | K15 (key/value) | .290 / .325 | update 100 (560 pairs) / update 200 (1,360 pairs) |
  | R15b (rows) | .435 / .425 | updates 100 / 200 |
  | D15b (rows + residuals) | .480 | update 100 |
  | Base reader, all ten records in context | .715 | — |

## The ladder

| Run | Lever (versus its reference) | Reference | Data, order, start | Where |
|---|---|---|---|---|
| H1a_lr1e4 | encoder-side rate (AC modules + side LoRA) 1e-4 instead of 2e-5; reader stays 2e-5 | K15 | K15's packet exactly, derived with `kit_hill/bench/derive_lr_packet.py`: same examples, same order, same task panel; start = a copy of K15's fresh pair | CSAIL, 1 H200 |
| H1b_lr2e4 | the same lever at 2e-4 | K15 | as H1a | ORCD mit_preemptable, 1 H200 |
| H2_unikeys | uniform multiple-choice keys | the better of H1a / H1b | K15's rows and order. Each multiple-choice row gets a uniform key with permuted options, and the teacher target is regenerated exactly (`permute_mc.py --done 0`). A row whose regenerated teacher letter disagrees with the new key reverts to its original form (`kit_hill/unikeys_revert.py`), so the row set and order stay K15's | queued after H1 |
| H3_all10 | all ten records in the encoder view on every example | H2 | as H2 | queued after H1 |

Mechanics:
- **Same as K15:** 8 examples per update; 1 GPU × micro-batch 4 with shared record encodings; encode cap 16k; checkpoint every 25 updates; snapshot every 100 updates; 6,160-pair cap; reconstruction 240.
- **Warm-up and schedule:** K15's own (50 updates of warm-up, then constant).
- **Evaluations:** the existing `eval_snapshot.py` / `eval_watch.sh`, with the all / right / none views.

## Log (UTC)

- 01:54Z: H1a_lr1e4 is CSAIL job 2707472 (1 × H200, quanta-h200-1, `--time 05:40`, ends before the 08:00Z window).
  - The packet was derived from K15's: its check passed, and only `learning_rate_ac` changed (2e-5 → 1e-4). Recipe sha `dc9a612e…` is the source's.
  - Training started 01:55:48Z, with at most 19,145 s.
- ≈02:00Z: H1b_lr2e4 is ORCD job 26319533 (mit_preemptable H200). It is pending on priority; D15b is also pending on ORCD.
  - Its inputs are K15's packet files and fresh pair, copied CSAIL → container → ORCD (`hill_src/`). No `optimizer.pt` was copied.
  - The items sha matches K15's (3248b88a33dafbba).
- ≈02:00Z: two more CSAIL jobs.
  - Uniform-key permutation of all study multiple-choice rows: CSAIL GH200 job 2707526 → `synth_full_permuted_all/items.jsonl`.
  - Eval watcher for the H runs: CSAIL GH200 job 2707527, after the permutation, `--time 05:00`.

- 02:23Z:
  - H1b_lr2e4 started on ORCD (node4002) at ≈02:18Z.
  - H1a measures 37–58 s / update; step 23 has sft loss 3.05.
  - The CSAIL permutation is still waiting for a GH200, so the same command also went to ORCD as RTX job 26321396 (`kit_hill/orcd_rtx_py.sbatch`). Whichever starts first is kept; the other is cancelled.
  - ORCD eval watcher for the H runs: RTX job 26321409.
- 02:49Z:
  - H1a is at update 51 (≈55 s / update), so update 100 lands ≈03:35Z and update 200 ≈05:10Z. H1b is at update 23 (47 s / update).
  - ORCD has hit its per-user GPU cap (QOSMaxGRESPerUser, 4 GPUs: R15b, D15b, H1b and the R15b/D15b eval watcher). The permutation (26321396) and the H eval watcher (26321409) wait for a free slot.
  - On CSAIL I cancelled my GH200 permutation 2707526 (it was waiting on Resources; ORCD runs the same command) and the watcher 2707527 that depended on it. I resubmitted the watcher with no dependency as 2707784 (`--time 05:00`).
- 04:25Z:
  - H1a is at update 150 (≈57 s / update); H1b is at update 118 (≈46 s / update on ORCD).
  - Neither eval watcher has started: CSAIL GH200s are busy (Resources), and ORCD is at the per-user GPU cap.
  - I copied H1a's update-100 snapshot (no optimizer) to ORCD `runs/H1a_lr1e4/packet/` and resubmitted the ORCD watcher with H1a included (26331682, replacing 26321409).
  - /data/scratch has 470 G free.
- **Panel at update 100 (800 examples presented), training-panel NLL (lower is better):**

  | Run | mc (24 Q, mixed view) | mc_norows | study_held | spans (held reconstruction) | held headline | learning |
  |---|---|---|---|---|---|---|
  | K15 (2e-5) | 0.859 | 4.064 | 0.802 | 2.934 | 2.034 | 1.838 |
  | H1a (1e-4) | 0.971 | 5.236 | 0.755 | 2.720 | 2.369 | 1.676 |
  | H1b (2e-4) | **0.842** | 2.534 | **0.746** | **2.680** | **1.533** | **1.648** |

  - **What changes:** the higher encoder rates lower the study and span NLLs a little (study_held −6–7 %, spans −7–9 %). On the 24-question mc NLL, H1a is worse and H1b about even.
  - **The held headline is not usable here.** It averages in mc_norows, the benchmark questions with no rows at all, where a lower value means the reader alone puts more mass on the gold letter. That value swings with the reader's prior.
- 04:42Z, schedule correction from main: Slurm printed EDT, so the windows are later than assumed.
  - **CSAIL maintenance:** 12:00Z Oct 19 → 06:00Z Oct 20.
  - **ORCD maintenance:** 04:00Z Oct 20 → 01:00Z Oct 21.
  - **EC2** 2× RTX PRO 6000, 12:00Z Oct 19 → 06:00Z Oct 20.
  - **ORCD watcher resubmitted:** with a 1-day limit, 26331682 could not start before the ORCD window. The replacement is 26332553 (`--time 12:00`).
- 05:19Z: H1a is at update 202. Its panel at update 200 (1,600 examples) next to K15's at the same point:

  | Run | mc | study_held | spans |
  |---|---|---|---|
  | H1a | 0.843 | 0.704 | 2.699 |
  | K15 | 0.864 | 0.697 | 2.840 |

  The two are about equal. H1b is at update 174.
- 05:22Z: neither GPU watcher had started, so there were still no 200-question numbers. I moved the evaluations onto H1a's own H200 slot (still 1 H200), in this order:
  1. USR1 to H1a's batch script: it paused at an update boundary (checkpoint `epoch_001_step_00206`) and requeued. I held it with `scontrol hold`.
  2. Eval job 2708748 on 1 H200 (`kit_hill/csail_h200_eval.sbatch`, `--time 0:45`), for H1a updates 100 and 200 and for H1b update 100. H1b's snapshot was copied ORCD → container → CSAIL (`runs/H1b_lr2e4/packet/`), without optimizer.
  3. The uniform-key permutation on the same H200 slot: job 2708750 (`kit_hill/csail_h200_py.sbatch`, x86 venv with vLLM), afterany the evals.
  4. H1a resumes from step 206 afterwards. Its limit was lowered to 5:10 (afterany 2708750), so it ends before CSAIL's 12:00Z window.
- 05:23–05:38Z: the evals ran on the H200 slot (results table below). The eval job ended at 05:48Z, and the permutation ran 05:50–06:00Z.
  - The 64 original prompts reproduced 62 stored targets.
  - The 1,566 multiple-choice study rows got uniform keys (313–314 per letter). The regenerated teacher agrees with the new key on 1,411.
- 06:01Z: `unikeys_revert.py` wrote `synth_full_unikeys/items.jsonl` (sha 4826752f9b9abcca).
  - It reverted 155 rows; 114 of them are in the cap.
  - Train order and held rows are unchanged (checked).
  - Inside the 6,160-pair cap, 1,056 of the 1,170 multiple-choice items are permuted.
  - Keys in the cap: before A208 B403 C329 D115 E115; after A219 B235 C237 D241 E238.
- 06:00Z: H1a resumed on CSAIL (2707472, aia-h200-10) from step 206 and ends ≈11:10Z.
- 05:58Z: H1b was stopped (`scancel` 26319533) after its update-200 snapshot (05:49:52Z). The H1 question is settled; that snapshot gets evaluated by the ORCD watcher 26332553.
- 06:03Z: H2_unikeys and H3_all10 submitted on ORCD (26337356 / 26337357, mit_preemptable H200, `--time 12:00`), both pending on the per-user GPU cap.
  - Settings: build mode, lr 1e-4 / 2e-5, `synth_full_unikeys`, start = copy of K15's fresh pair; views mixed / all.
- ≈06:15Z, main's answers:
  - lr_ac 1e-4 and stopping H1b: confirmed.
  - **EC2 at 12:00Z gets H3_all10.** The downtime fork launches it with my env. The ORCD copy of H3 (26337357) is cancelled only once main says H3 is training on the node.
  - Comparisons stay single-lever: H2 vs H1a (letters only), H3 vs H2 (view only), with balanced accuracy at matched pairs.
  - **If /data/scratch nears the 300 G floor,** I tell main before changing anything.
- 06:15Z: backed up to `s3://general-bucket-aln/activation-context-data/corpus_hillclimb_20261019/` (via boto3, 16 files): the H1a/H1b eval JSONs, `synth_full_unikeys/` and `kit_hill/`.
- 06:52Z:
  - **H2_unikeys** started on ORCD node5000 (26337356, the slot R15b freed). Its packet was built at 06:37:35Z: 800 updates, training from the start, step 9 at 44 s.
    - Recipe against K15's: the same counts (240 reconstruction + 6,160 study), the same reconstruction target tokens (66,580), and identical panel and held sha.
    - Study target tokens are 386,425 against 386,302: the regenerated multiple-choice targets.
    - Only `learning_rate_ac` (1e-4) and the study file differ.
  - **H3_all10** (26337357) is still pending on the per-user GPU cap. It is planned for EC2 at 12:00Z.
  - **H1a** is at step 250, ≈59 s / update. /data/scratch has 452 G free.
  - **K15c** is running (CSAIL 2706531).
  - **H1b at update 200** (ORCD RTX eval): all .395 (bal .398), right .460 (bal .463), none .025.
    - Its no-context leak from update 100 is gone (.295 → .025), and it is about level with H1a at update 200 (.425, bal .405).
    - So 1e-4 and 2e-4 are close by update 200, and 1e-4 stays the pick.
  - **Eval reproducibility across hardware:** H1b's update-100 snapshot scores all .420 on the ORCD RTX eval, against .410 on the CSAIL H200. A ±1-point difference comes from the eval hardware alone.
- 07:43Z:
  - H1a is at update 300 (≈60 s / update), and H2 at update 58 (≈49 s / update).
  - The CSAIL GH200 watcher 2707784 could not start before the 12:00Z window ("Reserved for maintenance"), so I cancelled it.
  - H1a's later snapshots are mirrored to ORCD for the ORCD watcher: `hill_xfer/mirror_h1a.sh`, one rsync per cluster per 55-minute cycle, only complete snapshots, until 12:05Z.
  - /data/scratch has 405 G free.
- 07:54Z: the mirror pulled H1a's update-200 and update-300 snapshots and pushed them to ORCD.
- **08:18Z: the ORCD scratch quota was exceeded** (Errno 122, "Disk quota exceeded"). The filesystem itself shows 247 T free, so this is the per-user quota.
  - **Failed:** D15b (26297746, not mine), H2_unikeys (26337356; it has a resumable checkpoint at step 75) and H3_all10 (26337357, in the packet build).
  - **What I did:** told main, stopped the mirror loop, and stopped all writes to ORCD. Nothing deleted.
  - **Cleanups for the user to approve:**
    - H1b_lr2e4's training checkpoints (H1b is stopped);
    - the H1a mirror snapshots, once evaluated;
    - the per-run `init_pair` copies (identical to `hill_src/K15_fresh_pair`).
- **H1a at update 300** (2,160 pairs; ORCD eval): all .305 (bal .326), right .330 (bal .345), none .010. All-ten letters: A111 B29 C34 D16 E10.
  - This is the same letter flip that R15b (rows) showed at update 300 (B108 → A91, .425 → .345). Both arms collapse onto A at the same point in the shared data order.
  - **The training data has no A excess there.** I checked the original multiple-choice keys per 800-pair window of the train order:
    - pairs 800–1600: A22 B62 C30 D11 E9;
    - pairs 1600–2400: A28 B51 C47 D9 E13;
    - every window is B-heavy and D/E-light, and none leans to A.
  - So the A flip is not a local data skew. It is the reader's letter prior swinging between snapshots, on top of a persistent B pull in the keys.
  - Uniform keys (lever 2) remove the B pull. Whether they also damp the swing is for H2 to show. Balanced accuracy stays the main read.
  - H1a's update-100 re-eval on the ORCD RTX gives .405 (bal .372), against .410 on the CSAIL H200.
- ≈08:45Z, main's plan:
  - No ORCD writes until the user approves a cleanup.
  - **H2 restarts from scratch on CSAIL's H200 pair after K15c.** The ORCD step-75 checkpoint holds optimizer state and is not copied.
  - **Evals** run on CSAIL GH200s until 12:00Z.
  - **H3** stays on EC2 at 12:00Z.
  - **CSAIL jobs placed:**
    - H2_unikeys: job 2709057 (`kit_hill/csail_h200x2_hill.sbatch`, 2 × H200, micro 4 shared, `afterany:2706531`, `--time 02:25`).
    - GH200 eval watcher for H1a and H2: job 2709058 (`--time 03:05`).
- ≈08:50Z: ORCD sizes, measured read-only (du in CPU srun steps). The `paper_corpus_qa_v1` tree is ≈268 G; my part is ≈79 G.

  | Path | Size |
  |---|---|
  | hill_src | 6.9 G |
  | runs/H1a_lr1e4 (mirror) | 18 G |
  | runs/H1b_lr2e4 | 30 G: init_pair 5.9, AC_MODELS 9.6, LORAS 1.3, SNAPSHOTS 12 |
  | runs/H2_unikeys | 18 G |
  | runs/H3_all10 | 5.9 G |

  - The deletion-candidate list (A–F, ≈59 G without anything unique) went to main for the user's approval.
- ≈09:00Z: H1b's update-200 snapshot (6,309,321,452 bytes, no optimizer) was read from ORCD and uploaded to `s3://…/corpus_hillclimb_20261019/snapshots/H1b_lr2e4/epoch_001_025pct/`. Every object's size matches. A copy stays in the container (`hill_xfer`).
- H2's stop before the window is scheduled: `hill_xfer/h2_window_stop.sh` at 11:48Z. USR1 pauses the run at an update boundary with a checkpoint; then the requeued job is cancelled. The checkpoint stays on /data/scratch.
- 09:56Z:
  - **H1a at update 400 (2,960 pairs):** all .490 (bal .453) against K15's .415 (.384), so +7.5 / +6.9 points at matched pairs.
  - **H2_unikeys** is running on CSAIL 2709057 (quanta-h200-1, 2 GPUs). Its packet was built 09:35–09:39Z (3.6 min); training started 09:39Z with up to 7,237 s.
  - **GH200 eval watcher** 2709058: running, limit cut to 1:30.
  - /data/scratch: 360 G (383 G at 09:11Z).
- 10:37Z:
  - H2 is at update 96, at 25–32 s / update on 2 × H200.
  - H1a is at update 475. Its USR1@900 pause comes at ≈10:55Z, around update 490, so it will not reach update 500: **update 400 is H1a's last snapshot** (already evaluated).
  - A second GH200 watcher, 2709161 (`--time 0:50`, afterany 2709058), is queued for H2's update 200.
  - /data/scratch: 345 G.
  - At 11:40Z, `hill_xfer/pull_before_window.sh` pulls every unevaluated H1a/H2 snapshot (weights only) to the container and S3. It also cancels H1a's requeued pending job once the requeue line shows.
  - At 11:46Z, `h2_window_stop2.sh` stops H2.
- 11:29Z: `orcd_ssh.sh` returned exit 97 (auth needed), per main. No ORCD calls since.
- 11:40Z:
  - **H2** stopped at its duration limit at **update 196**, 4 updates short of the update-200 snapshot. Its checkpoint `epoch_001_step_00196` and `partial_epoch.json` are on /data/scratch. The job requeued, and at 11:46:09Z `h2_window_stop2.sh` cancelled the pending requeue.
  - **H1a**'s pending requeue (2707472, held back by "Reserved for maintenance") was cancelled after its second requeue line. Its checkpoint `epoch_001_step_00492` and its update-400 snapshot stay on /data/scratch.
  - **Nothing to pull:** every snapshot of H1a (updates 100–400) and H2 (update 100) is already evaluated.
- **Letter prior (main's request):** the 200-question accuracy partly tracks the reader's dominant letter, which swings between snapshots. R15b went from B 108 at update 200 to A 91 at update 300 and dropped from .425 to .345.
  - So every row below also has **balanced accuracy**: the mean recall over the five gold letters, where a pure letter prior scores .20.
  - It also has the training panel's losses. `mc/longhealth` is the gold-letter NLL of 24 benchmark questions in the mixed view; `study_held` is 24 held self-study items. Both come from `kit_hill/hill_table.py`.
  - With 200 questions, about ±3 points of accuracy is noise.

## Results

Accuracy and balanced accuracy (bal) on the 200 questions; NLL from the training panel at the same update:

| Run | Update | Pairs | All ten (bal) | Right record (bal) | None | All-ten letters | Panel mc NLL | Panel study_held NLL | s / update |
|---|---|---|---|---|---|---|---|---|---|
| K15 | 100 | 560 | .290 (.239) | .375 (.346) | .125 | A1 B169 C22 D5 E2 | 0.859 | 0.802 | 25–36 (2 × H200) |
| K15 | 200 | 1,360 | .325 (.299) | .400 (.393) | .070 | A37 B134 C18 D6 E5 | 0.864 | 0.697 | |
| K15 | 400 | 2,960 | .415 (.384) | .455 (.412) | .005 | A36 B78 C45 D27 E14 | 0.967 | 0.688 | |
| H1a (1e-4) | 100 | 560 | **.410 (.376)** | .425 (.399) | .005 | A5 B118 C43 D22 E12 | 0.971 | 0.755 | ≈57 (1 × H200) |
| H1a (1e-4) | 200 | 1,360 | **.425 (.405)** | .425 (.426) | .005 | A59 B92 C31 D12 E6 | 0.843 | 0.704 | |
| H1a (1e-4) | 300 | 2,160 | .295 (.317) (ORCD .305 / .326) | .335 (.349) | .010 | A113 B29 C33 D15 E10 | 0.940 | 0.742 | |
| H1a (1e-4) | 400 | 2,960 | **.490 (.453)** | .500 (.464) | .005 | A34 B85 C43 D23 E15 | 0.929 | 0.694 | |
| H2 (1e-4 + uniform keys) | 100 | 560 | .360 (**.361**) | .395 (.416) | .055 | A30 B60 C30 D37 E43 | 1.065 | 0.765 | 25–32 (2 × H200) |
| H1b (2e-4) | 200 | 1,360 | .395 (.398) | .460 (.463) | .025 | A62 B72 C30 D21 E15 | | | |
| H1b (2e-4) | 100 | 560 | .410 (.359) | .400 (.375) | **.285** (parse .68) | A14 B121 C35 D22 E8 | 0.842 | 0.746 | ≈46–61 (ORCD H200) |

**H1 reading (05:50Z).** The encoder-side rate is a real lever.
- **H1a (1e-4) against K15 at matched pairs,** all ten records:
  - accuracy +12 points at update 100 and +10 at update 200; balanced accuracy +14 and +11. That is well beyond ±3.
  - H1a at update 200 already matches K15 at update 400 (.415), and the rows-only arm R15b (.435 / .425 at updates 100 / 200).
- **The panel NLLs barely move.** mc is 0.843 against 0.864, study_held 0.704 against 0.697. The gain shows in the decoded 200 questions, not in the 24-question gold-letter NLL.
- **H1b (2e-4)** matches H1a's all-ten accuracy at update 100 but has lower balanced accuracy (.359 against .376). Its reader, with no rows at all, answers .285 (parse .68), where K15 and H1a answer .005.
  - So at 2e-4 the reader starts answering from memory without the encoder. That confounds the channel, and it is the less interpretable choice.
- **Pick: lr_ac = 1e-4** for H2 and H3 (reader stays 2e-5).

**H2 reading (one point, update 100; reference H1a at update 100):**
- Accuracy: .360 against .410 (−5).
- Balanced accuracy: .361 against .376 (−1.5, within noise).
- The all-ten letters spread out: A30 B60 C30 D37 E43, against H1a's A5 B118 C43 D22 E12.
- Panel: mc 1.065 against 0.971, study_held 0.765 against 0.755.

So uniform keys remove the B pull, as intended. Much of the accuracy drop comes from losing a B prior that partly matches the benchmark's gold letters (B 49 / 200). Ability with the letter prior removed is about the same. One snapshot cannot say whether the letter swing (the update-300 flip) is damped. H2 resumes after the window from step 196.

## Recommendation for the long key/value run (as of 12:00Z Oct 19)

1. **Encoder-side rate 1e-4** (reader 2e-5). It is the one lever with a clear, repeated gain at matched pairs:

   | Update | H1a (all, bal) | K15 (all, bal) | Gain (all / bal) |
   |---|---|---|---|
   | 100 | .410 (.376) | .290 (.239) | +12 / +14 |
   | 200 | .425 (.405) | .325 (.299) | +10 / +11 |
   | 400 | .490 (.453) | .415 (.384) | +7.5 / +7 |

   2e-4 is not better: its balanced accuracy is lower at update 100, and at update 100 the reader alone answered with no context.
2. **Uniform keys: adopt them for the final long run, on interpretability grounds.** They remove a data-made letter bias. The cost on raw accuracy is a prior, not ability: balanced accuracy is flat at update 100.
   - Report balanced accuracy next to raw accuracy for every arm, since raw accuracy here partly tracks the reader's letter prior. Example: H1a and R15b both flipped to A at update 300 and lost 12–13 points, then recovered.
   - If the paper needs comparability with K15 / R15b / D15b, those arms use the original keys. Then either keep the original keys for the KV long run, or re-run the channel arms with uniform keys. Uniform keys everywhere is my pick.
3. **All ten records in view (H3):** the run is on EC2 (owned by the downtime fork). There is no result yet. If H3's balanced accuracy at matched pairs beats H2's update-100 .361 by more than noise, add it to the long-run config.

Long-run config pending H3: lr_ac 1e-4 + uniform keys (+ all ten if H3 wins). Same 6,160-pair cap and order, 8 examples per update, shared encodings on H200s, snapshots every 100 updates.

K15's panel at update 300 (2,400 examples presented): mc 0.935, study_held 0.809. Its fixed reference, the gold-letter NLL with the right record in context, is 0.57.
