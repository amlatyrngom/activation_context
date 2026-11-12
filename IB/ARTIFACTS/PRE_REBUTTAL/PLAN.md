# Pre-rebuttal plan (green-lit by the user 2026-11-09)

Goal: by the MLSys rebuttal (reviews 2027-01-18, responses 2027-01-21) and for the camera-ready, replace the
paper's ad hoc training with one clean, explainable, repeatable recipe per experiment, add fair trained text
baselines, and get AC at least level with them (ideally above).

## Rules for everything that goes into the updated paper (user)

- **One recipe per arm, trained once.** One config, one launch, data fixed up front with a fixed order. A
  preemption resumes automatically from the last checkpoint with the same optimizer, schedule and data
  position; no hand-built "continue from checkpoint X" chains.
- **Hill-climb freely** (user 2026-11-09: "Allow yourself to add more data, more hyper parameters and things like
  that to hill climb ... as long as the final recipe is something clean and explainable"). Probes and sweeps
  are fine; only the final recipe goes in the paper.
- **Text + reader-LoRA gets at least the same task data as AC** (same items, same amount). It need not get AC's
  learning-rate tricks. The FineWeb reconstruction stage stays AC-only (AC-specific pretraining).
- **One scorer per workload** for every arm: same answer parser, same generation cap, same decoding.
- **Selection hygiene.** Probes and checkpoint choices use a dev split; the reported set (≈1,000 held-out
  questions for §3.5) is never used to choose anything; report the final checkpoint.
- **Seeds.** Final §3.5 and retrieval numbers: 3 seeds per arm.

## Work, in priority order

1. **§3.5 ablation (Qwen3-4B → Qwen3-4B, HotpotQA).**
   a. Rescore existing checkpoints (`T_kv_v4`, `S_kv_v4fr`, `T_text_lora`, …) under one scorer (no training).
   b. All-Layer KVs vs text + LoRA: probes (no reconstruction stage, frozen vs trained reader, LR, data amount,
      overfitting), then one frozen recipe, both arms on the same larger HotpotQA set (train has 90,447
      questions; v4 used 32,000 items).
   c. Input Embeddings and Hidden Residuals with the least-squares vocabulary init + linear-skip head (the trick
      used for the 0.6B/1.7B encoders), under the same recipe.
2. **Retrieval (§3.1, Qwen3.5-2B → 9B).** Train a 2B→9B text + reader-LoRA baseline on the AC task data; replace
   Arm G's four stacked stages with one recipe; find the data amount and recipe that put AC at least level.
3. **Compaction (§3.3): rerun the summary baseline without the word cap** (prompt keeps "concise" and the five
   sections, drops "Keep the entire handoff under 500 words"; its own prompt copy so other results are
   unchanged). Same eval-200, seeds 2–4, 50k budget, untuned 9B. Runs on any free GPU; depends on nothing.
4. **Multi-agent math (§3.4), last:** more data; a fresh reader LoRA trained without AC as the baseline (the
   current "Results + LoRA" is the AC-trained reader run without ACs).

Not in scope: corpus compression (§3.2) stays as submitted (user); minor caption/wording issues.

## GPUs (user)

CSAIL `quanta-h200` / `quanta-gh200` (priority; 4-day limit) → `csail-shared-h200` (1 day) → ORCD →
EC2 spot RTX PRO 6000 only when none of these work or more parallelism is needed, at most 2 nodes × 2 GPUs or
4 nodes × 1 GPU. One job per GPU. Back up checkpoints to S3 as they appear.

State on 2026-11-09 ≈21:30Z: CSAIL priority and shared H200s all allocated, `/data/scratch-fast` at 100 %
(59 GB free); ORCD only preemptible GPUs free (≈3 RTX PRO 6000, ≈4 H200), our ORCD scratch 93 % and pool 97 %
full; EC2 spot quota 64 vCPUs for G instances.

## Reporting

Result tables in chat and `IB/STATE.md` at each milestone; ask the user only for decisions that change the
paper's story or spending beyond the EC2 limit above.

## Log

### 2026-11-09 evening: §3.5 setup and first probes (CSAIL `quanta-h200`, preempting shared jobs)

- **Eval sets** (`sec35/build_eval_sets.py` → `IB/TMP/PRE_REBUTTAL/sec35_eval_v1/`, staged as `data/hotpot_tasks_v2` = v1 + these):
  HotpotQA validation minus eval500/dev200 (6,705), seed 20261109: `dev1000` (all probes and choices) and `test1000` (reported
  numbers only). Same construction as dev200 (question-only BM25 top 20 over the pooled corpus, golds injected); the builder
  reproduces dev200's stored ranking 200/200. No overlap with train or dev200.
- **Single scorer** for both arms: HF greedy decode, cap 256, strict last "Final answer:" line, `rq_common.score`
  (AC: `kit/sec45_eval.py --max-new 256`; text: new `text_lora/text_lora_eval_hf.py`, base 4B + merged PEFT adapter).
- **Kit** `IB/ARTIFACTS/PRE_REBUTTAL/sec35/` (copied from `IB/TMP/PAPER_SEC45`, paths fixed, new knobs: encoder LR, penalty,
  seed, eval split, steps/ckpt overrides; Slurm limit 96 h). Bundle `sec35_code_v1.tar.gz` on S3 `pre_rebuttal_20261109/sec35/code/`.
  `text_lora_train.py` now resumes exactly from `OUT/resume` (smoke on H200: killed after update 4, resumed, finished).
- **Running** (work root `/data/scratch-fast/ngom/sec35`):
  - `S_kv_fr90k` — All-Layer KVs, fresh pair (no reconstruction), frozen reader, encoder LR 2e-5, 11,200 × 8 = 89,600 items
    (the whole 90,447 train pool but 847), evals on dev1000 every 800 updates. ~5–6 s/update on H200 → ~17 h + evals.
  - `X_text_90k` — text + reader-LoRA, LR 2e-5, the same 89,600 items in the same order, adapters + dev1000 evals every 800.
  - `rescore` — the paper's checkpoints (T_{in,res,kv}_v4, S_kv_v4, S_kv_v4fr, T_kv_v4fr, text u1000–u4000, untuned 4B) on dev1000
    under the single scorer.
- **Why these first:** the paper's best AC run was `S_kv_v4fr` (no reconstruction, frozen reader: 71.5 on dev200, still rising at
  the end: 68.5, 70.5, 71.5 over the last three evals), while text + LoRA peaked at u1000–u2000 (72.5) and fell to 70.5. Both
  arms on the full train pool tests whether AC's later, longer climb overtakes text with more data. A frozen reader also needs no
  no-context penalty, which keeps the recipe simple.

### Retrieval (§3.1) recipe v1, as set up 2026-11-09 (to be hill-climbed; nothing here is final)

Same driver, data rules and scorer as §3.5, so both workloads share one explainable recipe:
- **Task data** `rq_tasks_v1` (`retrieval/build_rq_tasks.py`): the 90,447 HotpotQA train questions (the §3.5 pool), each with its
  question-only BM25 top 200 over the pooled corpus, **no gold injection** (the realistic setting of Fig. 3). Choices on `dev200`,
  reported numbers on the paper's `eval500`.
- **Items:** the reader sees top-k with k drawn per question from {20, 100, 200} (fixed by the question id, so every arm and seed
  sees the same k); AC reads them as k/20 question-first encoder calls of 20 at ratio 0.1; text reads them as text.
  Target "Final answer: <answer>", SFT, thinking off (the §3.5 target). This replaces Arm G's six stages (reconstruction ×2, SFT,
  KL to same-context references ×2, KL to the gold-view teacher) with reconstruction pretraining (AC only) + one task stage.
- **AC:** Qwen3.5-2B encoder → Qwen3.5-9B reader, Input + Residuals at every full-attention layer (Arm G's architecture), fresh
  pair with the least-squares row head, reconstruction on FineWeb (`fineweb_recon_v1`, as §3.5's R), then task training.
- **Text + reader-LoRA:** Qwen3.5-9B + LoRA r128 on the language model's q/k/v/o/gate/up/down (Arm G's reader targets), the same
  items in the same order; logits only over the target (24k-token prompts).
- **Eval:** greedy, cap 256, strict parse, at k = 20/100/200 on eval500 for every arm (AC, text + LoRA, untuned 9B, gold and
  no-context lines). This also removes the presence-penalty and sampling differences of the submitted Fig. 3.
- **Open:** whether answer-only SFT matches Arm G's distillation from the gold-view teacher (54.07 at k=200, sampled, with
  reasoning). If not, the hill-climb adds the gold-view KL target to **both** arms.

### 2026-11-10 ~00:00Z: EC2 spot fleet (4 × RTX PRO 6000) and retrieval target probe
- Spot nodes run their work as SkyPilot jobs (`python -m activation.cloud.sky exec NODE -- ...`), never as ssh-started processes: the 30-minute idle autodown counts only sky jobs, and then tears the node down once its job ends. The first compaction node (cmpsum) was probably lost to that autodown rather than a spot reclaim.
- boto3 is not in the project lock: the node scripts use a side venv `/root/s3venv` (boto3 only) for s3x.
- cmpsum2: `compaction/cmp_node.sh` (uncapped-summary baseline seeds 2-4; rollout cache restored from and mirrored every 10 min to S3 `compaction/ROLLOUTS/`, so a reclaim resumes).
- Retrieval target probe (`sec35/jobs/rq_probe_node.sh`): rqb `refs` (base 9B gold-view references, greedy, first 32,000 of train's seed-0 shuffle + dev200), rqc `ans` (9B text + reader-LoRA, answer-only target), rqa `gold` (same, gold-view reference target; waits for the refs). Both arms: 1,000 updates × 8 items (41 s/update on the RTX PRO 6000: 2,000 would take ~23 h), LR 2e-5, k ∈ {20,100,200}, eval dev200 at k=100, cap 1024, every 250 updates. Untuned 9B (u0): dev200 k=100 acc 53.0 (10 length stops).
- Gold-view refs: train 32,000 → 26,836 kept (83.9%), median 56 tokens (p90 81); dev200 138/200 kept. P0_res_rq9 (9B AC throughput) runs on rqb (ORCD copy still queued).
- §3.5 rescore under the single scorer (dev1000), first rows: paper text+LoRA T_text_lora u1000 74.1, u2000 72.6.

### 2026-11-10 ~02:00Z: §3.5 rescore done; retrieval AC launched
- Rescore (ORCD 28529330, dev1000, single scorer), acc / EM / F1: T_text_lora u1000 74.1 / 67.1 / 80.22, u2000 72.6, u3000 72.8, u4000 72.3;
  S_kv_v4fr 73.5 / 68.9 / 81.60; T_in_v4 73.4; S_kv_v4 72.6; T_kv_v4fr 71.7; T_kv_v4 71.5; T_res_v4 71.5; untuned 4B 29.7 (56.4% answered).
  On 1,000 questions and one scorer the gap is ≤ 0.6 at the text peak and reversed at the final checkpoint.
- P0_res_rq9 on rqb (RTX PRO 6000): R 6.98 s/update median at micro-batch 2 (9.54 effective), task 17.6 s/update at micro-batch 1 on the
  longest items (the 9B text arm: 32.5 s/update). ORCD copy cancelled (redundant).
- Retrieval AC (`jobs/rq_ac_node.sh`, scripts `jobs/generated_rq9_rtx/`, values `jobs/values_rq9_rtx.json`) on rqb: R_res_rq9 (2,000
  updates, as §3.5's R) then T_res_rq9 (1,000 updates × 8 like the text probe; encoder 5e-5, reader LoRA 2e-5, penalty 1.0; dev200 k=100, cap 1024).
- Compaction uncapped summary: s2 140/200 solved (0.70), s3 136/200 (0.68); the capped summary's 3-seed mean is 68.67 (PAPER_CANON).
- **Compaction uncapped summary done** (cmpsum2, 02:00Z; logs `s3://…/pre_rebuttal_20261109/compaction/logs/`, rollouts `…/compaction/ROLLOUTS/pc_r9d50u_s{2,3,4}_{summary,summary_write}`):
  solved 140 / 136 / 139 of 200 (mean 69.17 % vs capped 68.67 %), mean continuation tokens 22.01k / 21.21k / 20.44k (mean 21.22k vs 21.17k),
  summary write 435 / 577 / 552 tokens. Removing the 500-word cap changes the text arm by +0.5 point; AC 5× 71.33 and 10× 71.0 stay above.
- 4th spot slot: rqd runs S_res_rq9 (retrieval AC from a fresh pair, no reconstruction; same task settings as T_res_rq9).
- §3.5 probes at ≈02:50Z (dev1000 acc): X_text_90k u800 71.3, u1600 70.3 (untuned 29.6); S_kv_fr90k u800 64.4, u1600 68.8, u2400 70.8 (still
  climbing); S_kv_fr_e1e4 u400 58.5, u800 60.3; S_kv_pen0 u400 55.8, u800 58.9. Text peaks early, frozen-reader KV climbs; encoder 1e-4 trails 2e-5 so far.
- Retrieval target probe u250 (dev200, k=100, n=200): answer-only 58.0 / EM 55.5 / F1 66.94 (6.7 reply tokens); gold-view ref 60.5 / 57.0 /
  68.07 (53.9 tokens); untuned 53.0. +2.5 for gold (5 questions; noise level). R_res_rq9 at update ~700/2000 (8.5 s); S_res_rq9 started 02:54Z on rqd.
- 2026-11-10 ~07:00Z: R_res_rq9 done (valid, 2,000 updates; held-out reconstruction token accuracy 0.918 @200 → 0.990 @1000 → 0.995 @1600;
  final pair sha 8e26e806…, on S3). S_res_rq9 (fresh pair, no reconstruction) stopped at update 641: dev200 21.5 @200, 22.5 @400, 23.0 @600
  — about the no-passages level, so a fresh cross-model res pair needs the reconstruction stage. Both rows use the least-squares head (fit cosine 0.361).
- Now: T_res_rq9 (reader LoRA 2e-5, penalty 1.0) on rqb and T_res_rq9fr (frozen reader; `jobs/generated_rq9fr_rtx`, R's pair fetched from S3)
  on rqd, both from R_res_rq9's final pair. Text probe u500: answer-only 57.5 (u250 58.0).
- ~09:00Z dev200 k=100: text answer-only u750 59.5; text gold-ref u500 58.5, u750 62.5 (27.4 reply tokens). AC T_res_rq9 u0 12.5 → u200 36.5;
  T_res_rq9fr (frozen) u200 27.5. Paper (eval500, old scorer): AC Arm G 52.87, untuned text 49.2, no context 23.07. AC is far from level at u200;
  next levers if the slope stays flat: gold-view reference targets for AC (Arm G's key stage was KL to the gold-view teacher), longer T.
- ~10:00Z: T_res_rq9 u400 44.0 (u200 36.5), T_res_rq9fr u400 37.0 (27.5): ~+7.5 per 200 updates, reader LoRA ahead. Queued behind the text
  probes (start when they finish): T_res_rq9Lg on rqa (4,000 updates = the 32,000 questions the gold refs cover, gold-view reference
  targets via RQ_TARGETS=1) and T_res_rq9L on rqc (same, answer-only), both reader LoRA 2e-5, from R_res_rq9's pair, evals every 400.
  sec45_train's pool order (raw train.jsonl, Random(seed 0).shuffle, first N) = gold_refs.py's, so every training row has a ref.
- ~11:00Z **rqc spot-reclaimed** (us-west-2b) at text update ~950, its queued T_res_rq9L lost with it. **Bug found:** `s3x putdir` skipped
  files whose S3 copy had the same size, so fixed-size slot files never refreshed: the text slot on S3 was still update 250
  (`resume/state.json`, adapter, optimizer.pt; local 750), and AC runs' `partial_epoch.json` stayed at the first slot's.
  Fixed in `sec35/kit/s3x.py` (skip only when same size AND the other side is at least as new; slot pointers state.json /
  partial_epoch.json uploaded after the slot files; DONE last); pushed to rqa/rqb/rqd's `/root/sec35/s3x.py` (their 15-min mirror
  picks it up) and to S3 `jobs/s3x.py`; `ec2_prep.sh` now always refreshes it. Generated job scripts embed an older s3x but run with S3 off.
- X_rq9_text_ans not resumed (the S3 slot was update 250: ~7 h for one u1000 point; its u250/u500/u750 already answer the target question,
  and the final text arm is rerun with 3 seeds under the final recipe). rqc2 (us-east-1d) runs T_res_rq9L instead.
- T_res_rq9fr (frozen reader) u600 44.0 (u400 37.0).
- ~11:40Z: **rqb spot-reclaimed** too (us-west-2b, ~10:50Z, T_res_rq9 at update ~640; S3 holds slots 200/400 and evals u0–u400). Not resumed:
  the 4,000-update T_res_rq9L/Lg supersede it. Fixed s3x verified on rqd: one manual putdir refreshed T_res_rq9fr's pointer to step 600.
- **Target probe result (9B text + LoRA, dev200 k=100):** gold-view ref u250/500/750/1000 = 60.5 / 58.5 / 62.5 / **61.0** (final, valid);
  answer-only u250/500/750 = 58.0 / 57.5 / 59.5 (u1000 lost to the reclaim). Gold ahead at every shared checkpoint (+2.5 / +1.0 / +3.0),
  each within one-seed n=200 noise; replies 27–61 tokens vs 7. Working choice: **gold-view reference targets**, confirmed on AC with
  T_res_rq9Lg vs T_res_rq9L (4,000 updates each).
- rqe: X_rq9_text_gold4k (text + LoRA, gold targets, 4,000 updates = T_res_rq9Lg's 32,000 questions; ~37 h) — the data-matched text baseline.
- ~12:30Z §3.5 (dev1000): X_text_90k u5600 76.0 (peak) / u7200 74.3; S_kv_fr90k u6400 74.7 (peak) / u7200 74.4; S_kv_fr_e1e4 final 72.8;
  S_kv_pen0 final 72.1 (both 4,000 updates). 90k data: +~2 for both arms; at u7200 level; text peak 1.3 above AC peak.
- **§3.5 in/res "least-squares init" (plan 1c):** for the same-model 4B→4B pair the side→reader embedding map is the identity, and the row head
  already starts as the identity skip at equal widths (ac_model.py: row_head_linear_skip is only for unequal widths), so the paper's
  T_in_v4 / T_res_v4 already had it. 1c is therefore "in/res under the same recipe": T_in_fr90k and T_res_fr90k (frozen reader, encoder 2e-5,
  11,200 × 8, dev1000 every 800 = S_kv_fr90k's recipe, from the paper's frozen-reader R pairs R_{in,res}_v4fr_final, relayed CSAIL → host →
  ORCD, pair hashes checked against DONE) on ORCD 28583999 / 28584000 (`jobs/generated_p3_rtx`). CSAIL's H200 node was full of our own
  group's quanta-main jobs (not preemptible by us), so ORCD.
- ~14:00Z: T_res_rq9fr (frozen, 1,000) u800 46.5 (u600 44.0). 4,000-update arms at u167–210; X_rq9_text_gold4k at u50 (33.8 s/update);
  ORCD T_in_fr90k / T_res_fr90k running. Queued on rqd after T_res_rq9fr: T_res_rq9Lge (= T_res_rq9Lg with encoder LR 1e-4; the fresh
  cross-model head starts from a weak fit, cosine 0.36, and may need a faster rate than §3.5's identity-skip pair).
- ~15:00Z: **T_res_rq9fr done** (valid; frozen reader, 1,000 updates, 19.4 s/update): u200/400/600/800/final = 27.5 / 37.0 / 44.0 / 46.5 / **48.0**
  (dev200 k=100), still rising as the LR decayed. T_res_rq9Lge (encoder 1e-4) started on rqd.
- ~14:30Z: **§3.5 S_kv_fr90k u8000 75.6, u8800 77.0** (new peak; X_text_90k u8000 75.8, u8800 75.2, u9600 74.7): KV AC now level with or above text at
  the same update. **ORCD T_res_fr90k failed** at its first slot (u800, 13:48Z): `torch.save` of optimizer.pt raised
  `basic_ios::clear: iostream error` (T_in_fr90k wrote the same-size slot in the same minute fine; scratch 18% used → transient write
  error). Broken slot moved to `runs/T_res_fr90k/broken_slot_u800_iostream`, resubmitted as 28594453 (fresh start; u0 = 0.0, 987/1000 length
  stops). T_in_fr90k u800 19.0 (u0 8.9) — far behind S_kv_fr90k's u800 64.4.
- **rqd spot-reclaimed** (us-west-2b, third time) with T_res_rq9Lge before its first slot (S3 had only u0) → relaunched fresh on rqd2.
  Retrieval 4,000-update arms at u400 (dev200 k=100): T_res_rq9Lg (gold) **36.5**, T_res_rq9L (answer-only) **38.5** (frozen T_res_rq9fr at
  u400: 37.0); both still at full LR. X_rq9_text_gold4k at u250 (35 s/update).
- ~15:40Z: **X_text_90k done** (valid, 3 attempts): final u11200 **75.7** (u10400 76.3; dev1000). S_kv_fr90k u9600 75.4 (u8800 77.0).
  Text recipe fixed (90k rows, 11,200 × 8, LR 2e-5, final checkpoint), so its replicates start now on ORCD: X_text_90k_s1 / _s2
  (28599473 / 28599474, pinned to bundle v1 like seed 0; v3 differs only in retrieval support) and the test1000 score of seed 0's final
  adapter + the untuned 4B (28599416, `rescore_job.sh` with the new `RESCORE_DIR=$W/final_test`, so the paper checkpoints staged in
  `rescore/` are not scored on test). Retrieval u400 unchanged; Lg u686, L u740, Lge u130, text gold4k u375.
- ~17:40Z: **ORCD scratch quota full (≈1.0 TB).** T_res_fr90k (twice: `iostream error` at the u800 save, then `Errno 122 Disk quota exceeded`),
  T_in_fr90k (truncated optimizer.pt at the u1600 save) and X_text_90k_s2 (exit 0:53 in 4 s) all failed on it; df shows 18 % (the
  filesystem, not the quota). Inventory by `du` in `srun -p mit_normal`: `~/orcd/scratch/hf` 241 G (not ours to prune: other models),
  `activation/` 758 G (KEEP 170, paper_corpus_qa_v1 163, hf 145, sec35 75, trajectory_compaction 43, rag_arms_v1 35, …). **Pruned (≈110 GB):**
  `paper_corpus_qa_v1/runs/{R15b,D15b}/packet/SNAPSHOTS` (42 + 40 GB, intermediate %-snapshots; their evals, final pairs and D15b's
  u400/u650 pairs are on S3 `corpus_qa_20261019/`; recovery slots and fresh pairs kept); `sec35/rescore/{ac/*,text/T_text_lora_u01000}`
  (≈22 GB, staged copies of the paper pairs, byte sizes equal to S3 `sec45_20261026/runs/*/sync/ckpt/final`; rescore outputs kept);
  the two truncated slots (T_in u1600, T_res u800; 5.6 GB). Resubmitted: T_in_fr90k 28610377 (resumes from u800), T_res_fr90k 28610379
  (fresh), X_text_90k_s2 28610380. Canon "Quotas you cannot see" records the symptoms and the ≥100 GB headroom rule.
- **First test1000 numbers (single scorer):** X_text_90k final (u11200) **76.1** (dev1000 75.7); untuned 4B **26.2** (dev1000 29.7).
  §3.5 dev1000: S_kv_fr90k u10400 75.3 (u8800 77.0, u9600 75.4); T_in_fr90k u800 19.0.
- ~18:40Z: **S_kv_fr90k done** (valid, 1 attempt, final sha 1309adaa…): final **75.2** (dev1000; text 90k final 75.7). KV recipe fixed (fresh pair,
  frozen reader, encoder 2e-5, 90k × 1 pass, final checkpoint) → CSAIL: test1000 score of its final (3021416, `RESCORE_DIR=final_test`),
  seeds S_kv_fr90k_s1 / _s2 (3021417 / 3021418, `jobs/generated_p1s`, make_jobs `--seed`, bundle v1 like seed 0); all pending on Priority.
  ORCD: X_text_90k_s1 u800 70.1 (seed 0: 71.3); T_in_fr90k resumed, T_res_fr90k and X_text_90k_s2 running.
- Retrieval: the eval watcher's u800 evals **OOMed** on T_res_rq9Lg and T_res_rq9L (training ≈54 GB + 9B eval at k=100, cap 1024, ≈41 GB on
  one 96 GB card); u400 had fit. The pairs are kept and the watcher's final pass retries each failed eval once on the free GPU.
  T_res_rq9Lge u400 **36.5** (encoder 1e-4; Lg at u400 36.5, L 38.5). X_rq9_text_gold4k u500 **62.5** (u0 53.0; the 1,000-update probe
  had 58.5 at u500).
- 2026-11-11 ~00:30Z: **Retrieval k-diagnostic** (user: "diagnose first"; `jobs/rq_eval_job.sh` on ORCD 28631062, the runs' exact eval env; pairs
  from S3; a passage set of k is k/20 encoder calls of 20, question first, each its own compressed part). dev200 acc:

  | | k=20 (1 call) | k=100 (5) | k=200 (10) |
  |---|---|---|---|
  | untuned 9B, text | 48.5 | 53.0 | 55.0 |
  | text + LoRA, gold targets, u1000 (X_rq9_text_gold) | 60.0 | 61.0 (= in-run) | 66.0 |
  | AC T_res_rq9fr (frozen reader), final u1000 | 43.5 | 48.0 (= in-run) | 51.5 |
  | AC T_res_rq9Lg (gold), u1200 of 4000 | 44.0 | 46.0 | 47.5 |
  | AC T_res_rq9L (answer-only), u1200 of 4000 | 42.5 | 45.0 | 47.0 |

  The AC − text+LoRA gap is ~15 points at every k (16.5 / 13 / 14.5 for the best AC), and AC at k=20 is below the untuned 9B reading
  text. So the loss is in compressing ONE 20-passage call (2B encoder → 9B reader, ratio 0.1, LS-skip head), not in reading 5–10
  parts: a 20→100→200 curriculum would not reach the gap's cause. k=100 trajectory: L 38.5 / 39.5 / 45.0 (u400/800/1200),
  Lg 36.5 / 44.5 / 46.0. Text seed 2 paused at its u3200 slot for the eval (dependency resubmit 28638532, now running).
- 2026-11-11 ~15:40Z: §3.5 status (dev1000 acc). Text seeds done: X_text_90k_s1 final **76.2**, _s2 final **75.3** (seed 0 75.7). ORCD cancelled
  T_in_fr90k and T_res_fr90k together at 13:12Z (not by us; not requeued). T_res_fr90k (frozen reader, Input + Residuals) u800–u5600 =
  58.8 / 64.0 / 70.9 / 68.1 / 71.8 / 69.7 / **73.2**, resubmitted as 28694217 (resumes from the u5600 slot). T_in_fr90k (frozen reader, Input only)
  is flat at 19–20 from u800 to u6400: Input rows need a trained reader (the paper's T_in_v4, reader LoRA, 73.4); not resubmitted.
  User (retrieval): first get 2B→9B AC level on **20I** (top 20 with both golds injected at seeded positions = the §3.5 data) before BM25 k mixes.

### 2026-11-11 ~16:00Z: retrieval 20I phase (user green light, 12–24 h+ autonomy; do NOT block on the user)

Aligned with the user: get 2B→9B AC level on **20I** first (the §3.5 data `hotpot_tasks_v2`: BM25 top 20, both golds injected at
seeded positions; choices on dev1000, reported on test1000), then plain BM25 k = 20 → 100 → 200. The 90,447 train questions (fixed
seed-0 order) are split so no question is trained twice: **first 44,800 (5,600 × 8) for 20I**, the remaining **45,647 for the BM25 stage**.
Text + LoRA always gets the same blocks in the same order. Target: answer-only "Final answer: X" (as §3.5). Single scorer (greedy, cap 1024).

Runs:
- A. Eval only (ORCD, `rq_eval_job.sh` on 20I dev1000): untuned 9B text; current AC pairs T_res_rq9fr final and T_res_rq9Lg u1200; the
  paper's Arm G if the old kit's pair loads (skip it if not; note why).
- B1. **R_res_rq9fr**: R_res_rq9 with the reader frozen (2B→9B Input + Residuals, LS head, 2,000 × 8 FineWeb, encoder 5e-5, reader LR 0).
  Check: held-out reconstruction token accuracy vs R_res_rq9's 0.995 (4B: frozen R loses some, res still worked).
- B2. **T_res_rq9fr_20I**: from R_res_rq9fr, reader frozen, encoder 5e-5, 5,600 × 8, dev1000 every 800.
- B3. **T_res_rq9L_20I**: from R_res_rq9, reader LoRA 2e-5, encoder 5e-5, penalty 1.0, 5,600 × 8, dev1000 every 800.
- B4. **X_rq9_text_20I**: 9B text + LoRA r128, LR 2e-5, the same 44,800 items in the same order, dev1000 every 800.
- Stopped to free spot GPUs (pairs/adapters stay on S3): T_res_rq9L, T_res_rq9Lg, T_res_rq9Lge, X_rq9_text_gold4k (BM25-mix recipe, superseded).

Decision rules (act on them without asking; log each decision here):
- **Gate** (dev1000, final u5600): best AC ≥ text + LoRA − 1 → level. Then the BM25 stage from each arm's 20I final on the 45,647 questions,
  three equal blocks at k = 20 → 100 → 200, text + LoRA the same; evals at k = 20/100/200 on dev200 BM25.
- AC behind by > 3 at u2400 (or flat over two evals): probes of 1,600 updates on dev1000, one lever at a time, keep the winner:
  encoder LR (2e-5 as 4B, 1e-4); KL to the gold-view 9B teacher (top-32 log-probs; wire the trainer's teacher cache); ratio 0.2 (env knob
  for `RATIO`). The main arms keep running.
- R_res_rq9fr reconstruction poor: still run B2 (4B's frozen res recovered at task time); add a fresh-pair frozen arm only if B2 is flat.
- §3.5 alongside: T_res_fr90k to final (28694217) → test1000; S_kv_fr90k test1000 + seeds 1–2 (CSAIL); text seeds 1–2 test1000;
  T_in_L90k (Input only, reader LoRA 2e-5, penalty 1.0, from R_in_v4, 90k recipe) when a GPU is free (frozen Input stays at 20).
- Infrastructure: spot reclaim → relaunch and resume from S3; ORCD quota ≥ 100 GB headroom; CSAIL priority first; exit 97 (Duo) →
  move the work to EC2 or the other cluster, never wait. Up to 4 EC2 spot GPUs.
- Report: result tables in chat and this log at each milestone; commit and push. Math multi-agent stays last (not in this window).
- 2026-11-11 ~16:00Z, on the green light: spot nodes rqa / rqc2 / rqd2 had been **reclaimed overnight** (T_res_rq9Lg, T_res_rq9L, T_res_rq9Lge
  stopped with them; superseded, not relaunched). rqe keeps X_rq9_text_gold4k to its end (u2850/4000; dev200 k=100 u500–u2500 = 62.5 / 65.0 /
  62.5 / 59.5 / 64.0), then becomes the probe node. **Launched** (`jobs/rq20i_node.sh`, scripts `generated_rq9fr20I_rtx`, `generated_rq9L20I_rtx`;
  micro-batch 4 for 20I task items, 2 for R): rqf `frozen` (R_res_rq9fr → T_res_rq9fr_20I), rqg `lora` (T_res_rq9L_20I from R_res_rq9),
  rqh `text` (X_rq9_text_20I). ORCD: 20I eval-only baselines 28695819 (`rq_eval_job.sh` with RQ_EVAL_DATA=hotpot_tasks_v2, dev1000: untuned 9B,
  text gold u1000, AC T_res_rq9fr final, T_res_rq9Lg u1200); test1000 of text seeds 1–2 finals 28696121; T_in_L90k (`generated_p4_rtx`: Input only,
  reader LoRA 2e-5, penalty 1.0, from R_in_v4, 90k recipe). CSAIL: S_kv_fr90k_s1 / _s2 at u4800 (71.8 / 74.7); the KV test1000 rescore 3021416
  had been cancelled at start (no log), resubmitted as 3028881.
- ~16:00Z: **§3.5 text + LoRA final numbers (test1000, 3 seeds, final u11200):** 76.1 / 77.4 / 76.0 (mean **76.5**; dev1000 75.7 / 76.2 / 75.3).
  T_in_L90k submitted on ORCD (28697294). R_res_rq9fr started on rqf (15:57Z).
- ~16:20Z: **20I baselines (2B→9B retrieval setup, dev1000 20I, greedy cap 1024; ORCD 28695819):** untuned 9B text **68.8**; text + LoRA trained on
  BM25 mixes (X_rq9_text_gold u1000) **73.9**; AC trained on BM25 mixes: T_res_rq9fr final 59.5, T_res_rq9Lg u1200 61.4. Speeds: R_res_rq9fr ≈4 s/update,
  X_rq9_text_20I 5.3 s/update (≈8 h for 5,600).
- User (16:30Z): the 9B text + LoRA must get the same data as AC, so it can go in the figure. Status: X_rq9_text_20I already trains on the same 44,800
  20I questions (5,600 × 8, same order) as the AC 20I arms. Stage 2 (BM25 k = 20 → 100 → 200 on the other 45,647) starts for BOTH arms only after
  the 20I gate passes, each continuing from its own 20I final on identical blocks. X_rq9_text_gold4k (BM25 mix, 4,000 on the first 32k) is a
  reference only, not the final text arm.
- User (16:40Z): **stage 2 waits for the user's feedback** once the 20I gate passes; if no reply by then, run **both** variants (plain BM25 and golds
  injected at k = 20/100/200), each for AC and text + LoRA. The user is wary of targets that need passages the input lacks (gold-view references,
  or KL to the gold-view teacher, on questions whose golds are not retrieved): on 20I the golds are always in the input, so the 20I arms and the
  20I hill-climb levers are unaffected; for plain-BM25 stage 2 use answer-only targets and no gold-view teacher.
- ~17:45Z **20I u800 (dev1000):** AC T_res_rq9L_20I **58.7** (u0 19.6); text + LoRA X_rq9_text_20I **73.7** (u0 68.4). Gap 15 (4B at u800: 12.5, closed to ~2).
  Probes started early on ORCD's two free GPUs (cheap; the decision rule's first levers), same schedule as the main arm (5,600 planned, compared at
  u800 / u1600, then stopped unless they lead): T_res_rq9L_20I_r02 (**ratio 0.2**, from R_res_rq9 trained at 0.1) and T_res_rq9L_20I_e2 (**encoder LR 2e-5**,
  as 4B). Code bundle v4 = v3 + `SEC45_RATIO` knob (unset = 0.1) + the s3x fix; scripts `jobs/generated_rq9_20Iprobe_rtx`.
- ~19:15Z **20I u1600 (dev1000):** AC T_res_rq9L_20I **65.7** (u800 58.7); text X_rq9_text_20I **76.3** (73.7). Gap 15.0 → 10.6, the 4B §3.5 shape
  (12.5 → ~9 → ~2 at u5600). §3.5 T_in_L90k (Input only, reader LoRA) u800 **60.4** (frozen Input: 19.0) → Input needs the trained reader.
  Probes r02 / e2 running on ORCD (u0: 16.1 / 19.6).
- ~21:10Z **20I (dev1000):** main AC T_res_rq9L_20I u2400 / u3200 = **67.8 / 67.8** (flat; untuned 9B 68.8); text u2400 **76.3** (u1600 76.3). Probes at
  u800 / u1600: encoder 2e-5 (e2) 62.9 / **64.9** (main 58.7 / 65.7: faster start, no gain → stopped, ORCD 28707393 cancelled); ratio 0.2 (r02)
  **61.8** at u800. R_res_rq9fr done (valid; held-out token accuracy 0.991 at u1600 vs 0.995 with a trained reader); T_res_rq9fr_20I u0 9.2.
  **KL probe** T_res_rq9L_20I_kl (ORCD 28723308; bundle v5 = v4 + `sec45_train --loss kl` + a TARGETS setting): main arm's settings with
  KL to the gold-view teacher (the reader at the start reading only the gold passages, top-32) along the gold-view reference replies
  (26,836 of the first 32,000 questions; the rest answer-only). On 20I the golds are in the student's input too (user's concern does not apply).
  §3.5 T_in_L90k u1600 65.9; T_res_fr90k u7200 72.4.
- ~21:45Z: ratio 0.2 probe u1600 **65.2** (main 65.7) → no gain, stopped (28707392). Frozen 20I T_res_rq9fr_20I u800 **58.9** (reader LoRA 58.7).
  Both rate and ratio levers land on the main arm by u1600, so the next question is encoder capacity: **Qwen3.5-4B → 9B** (R_res_rq9e4 → T_res_rq9e4,
  same recipe as the main arm, LS head, `jobs/generated_rq9e4_rtx`; ORCD 28727321 → 28727322 afterok). A bigger encoder changes AC's encode cost
  (paper TTFT used the 2B encoder): a story decision for the user if it wins. KL probe 28723308 training.
- ~23:05Z **20I (dev1000)**: text u3200 / u4000 = 75.7 / **76.2**; AC reader LoRA u4000 / u4800 = **70.6 / 70.8** (picked up as the LR decays);
  AC frozen u1600 / u2400 = 61.3 / **70.1**; KL probe u800 **61.9** (main 58.7; the other probes also led at u800 and not at u1600). 4B-encoder R running
  (~9 s/update). §3.5: T_res_fr90k u8000 73.6; T_in_L90k u2400 70.3. BM25 reference X_rq9_text_gold4k u3500 65.0 (dev200 k=100).
- ~23:55Z: **T_res_rq9L_20I done** (valid; reader LoRA, encoder 5e-5, 5,600 × 8 on 20I): u800–u4800 = 58.7 / 65.7 / 67.8 / 67.8 / 70.6 / 70.8, **final 72.3**
  (dev1000). Text u4000 76.2 → gap ≈ 4: gate not passed. Frozen 20I u3200 68.5 (u2400 70.1). rqg now runs **T_res_rq9fr_20I_kl** (frozen reader + KL to
  the gold-view teacher = the untuned 9B reading the gold passages; from R_res_rq9fr; `rq20i_node.sh frozen_kl`, bundle v5), kept only if the KL probe
  (reader LoRA) still leads at u1600. §3.5 T_res_fr90k u8800 **76.5** (text 90k final 75.7).
- ~00:20Z (11-12): KL probe (reader LoRA) u1600 **65.5** (main 65.7) → no gain; stopped (ORCD 28723308) and the frozen + KL arm on rqg cancelled (sky job 3).
  **All three levers** (encoder 2e-5, ratio 0.2, KL to the gold-view teacher) lead at u800 and land on the main arm by u1600. What still moves AC is
  training length (main arm 67.8 flat at u2400–u3200, then 70.6 / 70.8 / 72.3 as the LR decayed; 4B res reached text only with 90k). Diagnostic on rqg:
  **T_res_rq9L_20I_90k** = the main arm on all 90,447 20I questions (11,200 × 8, the §3.5 amount; bundle v3 like the main arm). It uses the stage-2
  questions, so whether stage 1 takes more data is the user's call; this run only answers whether AC is data-limited on 20I. 4B-encoder probe R running.
- ~01:30Z (11-12): **X_rq9_text_20I final (u5600, 44,800 questions) dev1000 77.1** (u4800 76.0). Gate = 76.1. Best AC final T_res_rq9L_20I 72.3 → gap 4.8.
  Frozen 20I u4800 69.9 (still running to u5600). rqh now free → **T_res_rq9fr_20I_90k** (frozen arm, all 90,447 questions, 11,200 × 8; the 4B frozen
  res reached 76.5 only with 90k) as the second data-amount diagnostic alongside T_res_rq9L_20I_90k on rqg. Same caveat: uses the stage-2 questions; diagnostic only.
- ~02:15Z (11-12): **T_res_rq9fr_20I final (frozen reader, 44,800 questions) dev1000 71.9.** Gate at 44.8k **not passed** by either AC arm:
  text+LoRA 77.1 vs AC reader LoRA 72.3 / AC frozen 71.9 (gate 76.1). Stage 2 stays parked. Both 90k diagnostics running (rqg reader-LoRA u800 58.9,
  rqh frozen started). rqf left idle to autodown. Next lever, only if 90k helps: 2 passes over the same 44,800 (11,200 updates), to tell
  "more updates" apart from "more unique questions" (it needs a duplicated-pool data folder; the trainer runs one epoch only), so that stage 1 need not use the stage-2 questions.
- ~03:10Z (11-12): BM25 reference X_rq9_text_gold4k (rqe, dev200 k=100) **final u4000 63.5** (u3000 63.5, u3500 65.0); done, rqe left to autodown.
  90k diagnostics: rqg reader-LoRA u1600 60.9 (44.8k arm 65.7 at the same point; its LR is still near the peak on the 11,200 schedule); rqh frozen u800 59.7 (44.8k 58.9).
- ~03:50Z (11-12): **CSAIL shared ssh connection down (exit 97)**: needs one interactive login by the user (`ssh csail echo ok`); no retry. Only CSAIL item: KV test1000 rescore 3028881 (was PENDING Priority).
  ORCD: R_res_rq9e4 at 1827/2000 (T_res_rq9e4 afterok); T_in_L90k u4000 67.4 (u2400 70.3, u3200 68.7; let it run to final, LR still high).
- ~04:30Z (11-12): rqh reclaimed at u1600 of T_res_rq9fr_20I_90k → relaunched on **rqi**, attempt 2 RESUMED from the u1600 recovery slot (no work lost beyond the last slot).
  R_res_rq9e4 DONE valid (2 attempts) → T_res_rq9e4 queued on ORCD. T_res_fr90k u10400 73.8. rqg 90k reader-LoRA u2400 65.6 (44.8k arm 67.8); rqi frozen u1600 60.8 (44.8k 61.3).
- ~06:40Z (11-12): **T_res_fr90k (§3.5 frozen Input + Residuals, 90k × 1) final u11200 dev1000 73.8** (u8800 76.5, u9600 74.1, u10400 73.8). test1000 of the final
  pair: ORCD 28775667 (`final_test/ac/T_res_fr90k_final`, rescore_job.sh, test1000; text_base already scored there). rqg 90k reader-LoRA u4000 67.0.
- ~06:25Z (11-12): **§3.5 AC frozen Input + Residuals, final pair, test1000: 75.4** (seed 0; dev1000 73.8). Text + LoRA test1000 76.1 / 77.4 / 76.0
  (mean 76.5): AC is −0.7 vs text seed 0 and −1.1 vs the text mean. Seeds 1–2 of the same recipe submitted on ORCD
  (`generated_p3s/T_res_fr90k_s{1,2}.sh`, SEED 1/2, same R_res_fr90k init; 28776489 / 28776490, mit_preemptable) → 3-seed AC test1000 mean.
- ~09:20Z (11-12): **T_res_rq9e4 (4B encoder, enc 5e-5) diverged ~u1100**: held NLL 0.33 (u1000) → 0.97 (u1200), stuck at 0.92 by u1600; dev1000 u800 60.2
  → u1600 25.1 (answered 100, no length stops: the pair is wrong, not the format). Cancelled (28727322). Rerun **T_res_rq9e4_e2** from the same
  R_res_rq9e4 pair with encoder LR 2e-5 (stable on the 2B encoder; everything else equal), ORCD 28787956. Also: T_in_L90k u5600 72.7;
  rqg 90k reader-LoRA u5600 69.9; rqi 90k frozen u4800 65.3; §3.5 seeds 1–2 training.
- ~10:45Z (11-12): **§3.5 seed 1 (T_res_fr90k_s1) stuck in a bad basin**: same R_res_fr90k pair (sha 0b9ed4…) as seeds 0/2, but held NLL
  1.135 (u200) → 1.059 (u800) vs seed 0 0.709 → 0.489 and seed 2 0.651 → 0.594 (u400); dev1000 u800 **20.1** (seed 0 58.8). With a frozen reader the
  seed only changes the data order (no penalty), so the recipe is order-sensitive early: it looks like the T_in_fr90k "flat at 20" mode
  (answers without using the memory). Kept running to u2400 to see whether it escapes (nothing is queued behind it); **no seed swap**: all seeds
  get reported. If it stays stuck, the candidate fix is a longer warmup (50 → 500) or a lower encoder LR in the first updates, rerun on all 3 seeds.
- ~09:50Z (11-12): seed 1 escaping: held NLL 1.059 (u800) → 1.024 (u1000) → **0.798 (u1200)**. Seed 2 u600 0.535 (on track). T_res_rq9e4_e2 u400 held
  0.333 (diverged run 0.412 at u400). rqg 90k reader-LoRA u6400 70.3; rqi 90k frozen u6400 70.0. All 4 ORCD jobs ~9 s/update.
- ~10:00Z (11-12): **ORCD scratch write error** (`basic_ios::clear: iostream error` in torch.save, both at 09:52Z during checkpoint writes; disk 18% used):
  T_in_L90k failed at the u6400 save (recovery slot u5600 kept) and T_res_fr90k_s2 at its first slot (none published). Resubmitted the same scripts:
  T_in_L90k 28797183 (resumes from u5600), T_res_fr90k_s2 28797184 (fresh start; loses ~2 h).
- ~10:40Z (11-12): **The 09:52Z ORCD failures were the scratch user quota** ("Disk quota exceeded"; the 18% was the filesystem, not the quota). It
  killed T_in_L90k (u5600 slot kept), T_res_fr90k_s2 (twice), T_res_fr90k_s1 (u800 slot kept), T_res_rq9e4_e2 (no slot). Freed space by deleting
  only the **model weights** (sync/AC_MODELS, LORAS, ckpt, AUTOTUNE) of dead runs: T_res_rq9e4 (diverged), T_res_rq9L_20I_{e2,r02,kl} (stopped
  probes), T_in_fr90k (abandoned, flat at 20), plus the sync/ of the 8 `.prev_*` dirs of earlier attempts; logs, evals, run.json kept; the
  reported T_res_fr90k final pair untouched; no other project dir touched. A 2 GB test write then succeeded. Resubmitted: T_in_L90k 28797183
  (running, resumes u5600), T_res_fr90k_s1 28799112 (resumes u800), T_res_fr90k_s2 28799113 (fresh), T_res_rq9e4_e2 28799114 (fresh).
  rqg 90k reader-LoRA u7200 **72.1** (44.8k final 72.3).
  Usage after cleanup (sbatch du): sec35 ≈ 115 GB (runs 73, init 23, rq_eval 13); `quota` reads only /home (75 G of 195 G), the scratch quota is not
  readable (no lfs). Extra margin: deleted the 13 intermediate text adapters per seed (u800–u10400, all dev-scored) of X_text_90k{,_s1,_s2}
  (~39 GB); the step_11200 finals (used by final_test) kept.
- ~13:30Z (11-12): **90k diagnostics say AC on 20I is data-limited**: T_res_rq9L_20I_90k u8800 **75.6** (u7200 72.1, u8000 71.8; 44.8k final 72.3);
  T_res_rq9fr_20I_90k u8000 **72.8** (u7200 70.0; 44.8k final 71.9). Gate (text 77.1 at 44.8k − 1) = 76.1. If a 90k AC number is used, the text arm
  needs the same 90k (data parity, user rule): X_rq9_text_20I_90k (11,200 × 8) to be launched on a spot node when the AC finals land — stage 2 would
  then have no fresh questions left, which is the user's call (stage 1 = 90k vs 2 passes over 44.8k).
- ~12:45Z (11-12): rqi reclaimed at the u8800 eval → T_res_rq9fr_20I_90k resumed on **rqk** (attempt 3, from the u8000 slot). **X_rq9_text_20I_90k**
  (text + LoRA 9B, 11,200 × 8, same order/LR as the 44.8k text arm) started on **rqj** for data parity with the 90k AC runs.
  T_in_L90k u6400 72.0 (resumed fine). §3.5 seed 1 u1600 **49.8** (u800 20.1; seed 0 u1600 64.0): escaping.
- ~14:35Z (11-12): **User: MVP = AC at 200I beating text + LoRA at 20I (and maybe untuned text at 200I); defer 20I hill-climbing.** Built
  `hotpot_tasks_kI_v1` (`sec35/build_kI_sets.py`): the 20I dev1000/test1000 questions with BM25 top 100 / 200 and both golds injected
  (rebuilt 20I rows identical to hotpot_tasks_v2 1000/1000 on both; both golds in BM25 top 100/200 before injection: dev 71.3/75.7 %,
  test 73.4/79.4 %), and the 90,447 train questions with k ∈ {100, 200} per row by the SEC45_K=100,200 crc32 rule (45,160 / 45,287 rows).
  **Zero-training read** (`jobs/rq_kI_eval_node.sh`): text (untuned 9B + X_rq9_text_20I u5600) on rql now; AC (T_res_rq9L_20I and
  T_res_rq9fr_20I u5600 pairs) on rqg after its 90k run. Stage-2 base (44.8k vs 90k 20I finals) asked of the user.
- ~15:30Z (11-12): **User correction of the MVP:** TRUE BM25 (no injection) AC at k = 100 / 200 should match or beat text at k = 20 —
  definitely untuned, preferably tuned. Built `hotpot_tasks_bm25e_v1` (`sec35/build_bm25_eval.py`): the 20I dev1000/test1000 questions with
  plain BM25 top 200 (same ranking as 20I: check 1000/1000). Both golds in top 20 / 100 / 200: dev **55.5 / 71.3 / 75.7 %**, test 57.4 / 73.4 / 79.4 %
  (the MVP's headroom: k = 200 sees both golds for +20 points of questions). The read now runs plain BM25 first (text k = 20, 100, then 100I,
  200, 200I; AC k = 100, 200, 20, then 100I, 200I). First text attempt on rql failed (rq_eval_job.sh missing on S3 jobs/; uploaded), relaunched.
- **User: "KL to gold — is that not the main thing?"** Yes: Arm G's decisive stage was KL to the gold-view teacher; recipe v1 replaced it
  with answer-only SFT and the 20I hill-climb treated KL as a lever, stopped at u1600 (12,800 of 44,800 questions) on 65.5 vs 65.7 — before the
  LR decay where every other effect appeared. Not a real test. The gold-view teacher (untuned 9B reading only the 2 golds) is correct on
  26,836 / 32,000 = 83.9 % of the train refs, above text + LoRA at 20I (77.1).
- ~15:45Z (11-12): §3.5 seed 1 u2400 **61.2** (u800 20.1, u1600 49.8; seed 0 u2400 70.9): escaped, still 10 behind → kept to the final, no recipe
  change; all 3 seeds reported. Text kI read relaunched on rql (2nd failure: rq_eval_job.sh's EC2 python path; node script now sets SEC45_PY).
- ~15:40Z (11-12): **T_res_rq9L_20I_90k final (u11200) dev1000 20I 75.0** (44.8k final 72.3: +2.7 from doubling the data; text 44.8k 77.1).
  Frozen 90k u10400 74.5. **Plain BM25 k = 20, text + LoRA 20I final: 56.3** (the MVP comparator; both golds in BM25 top 20 for 55.5 %).
  AC kI / plain-BM25 read started on rqg. User (≈15:50Z): hold new launches (KL arm) until a planning session.
- ~16:15Z (11-12): **User decisions (planning session):** 90k is dropped (stage 1 stays at 44.8k; the 45.6k rest is for stage 2); KL to the
  gold-view teacher is the main configuration (not a lever); KL arm on 20I 44.8k with **both** readers (LoRA and frozen); general-QA tuning
  phase on **2WikiMultiHopQA + MuSiQue** (one continuous run, one config, HotpotQA 20I dev1000 checked periodically); X_rq9_text_20I_90k
  **stopped** (rqj sky job 2 cancelled at ≈u2400). Stage-2 data (plain BM25 vs golds injected) still under discussion.
- ~16:30Z (11-12): **User go for the batch; proceed autonomously; hill-climb only on noticeable differences (≈2–3 points, judged by significance).**
  Launched / chained:
  - rqj: gold-view replies for train questions 32,000–44,800 (`rq20i_node.sh refs_all`, REFS_UPTO 44800; S3 runs/rq9_gold_refs_all) →
    **T_res_rq9L_20I_KL** (KL to the gold-view teacher, reader LoRA 2e-5, encoder 5e-5, from R_res_rq9, 44,800 = 5,600 × 8, bundle v5).
  - rqk (after its 90k run + the 44.8k refs): **T_res_rq9fr_20I_KL** (same, frozen reader, from R_res_rq9fr).
  - rqg (after its AC read): gold-view replies for qa_tune_v1 (100,000) → **T_res_rq9L_QA_KL** (general-QA phase: KL, reader LoRA, from
    R_res_rq9, 12,500 × 8 one pass, HotpotQA 20I dev1000 every 800).
  - Data `qa_tune_v1` (`sec35/build_qa_tune.py`): 19,938 MuSiQue-Ans + 80,062 2Wiki train questions, each with its own 10–20 paragraphs as
    ONE encoder call (order shuffled per id), golds = supporting paragraphs; corpus = HotpotQA's (same ids) + 514,777 new paragraphs;
    dev1000 = HotpotQA 20I dev1000; qa_dev1000 = 500 + 500 in-domain.
  - **Stage 2 decided: train on injected 100I/200I** (the held-back 45.6k, k ∈ {100, 200} per row = hotpot_tasks_kI_v1), evaluate on plain BM25
    k = 20/100/200 (the figure) + 100I/200I. SFT from SFT 20I finals, KL from KL 20I finals; text + LoRA parity continues on the same 45.6k at 20I.
  **Zero-training read so far (dev1000, plain BM25):** untuned 9B text k=20 **49.6**; text + LoRA (20I) k=20 **56.3**, k=100 **61.3**;
  AC 20I finals at k=100: reader-LoRA **51.9**, frozen **52.0** (AC > untuned text-20 by 2.3 with no multi-call training).
  4B→9B encoder (T_res_rq9e4_e2, SFT, enc 2e-5): u800/1600/2400 = 66.0 / 70.7 / **74.6** vs 2B 58.7 / 65.7 / 67.8 — a story decision for the user.
- ~16:20Z (11-12): **Stage 2 set up and queued** (all start from the 20I final of the same arm; 5,600 × 8 of the 45,647 held-back questions,
  `--offset 44800` on the seed-0 order; fresh warm-up + cosine; in-run eval = PLAIN BM25 dev1000).
  - Data: `hotpot_tasks_kI_v2` (`sec35/build_kI_v2.py`: kI_v1 train/dev200 + bm25e dev1000/test1000 + the 100I/200I sets);
    `hotpot_tasks_t2_v1` (`--text`: v2 20I train/dev200 + bm25e dev1000/test1000) for the text arm.
  - Scripts `jobs/generated_rq9kI_rtx/` (from the v5 KL 20I scripts; TASK_K 100,200, EVAL_K 100, OFFSET 44800): T_res_rq9L_kI (SFT, LoRA),
    T_res_rq9fr_kI (SFT, frozen), T_res_rq9L_kI_KL, T_res_rq9fr_kI_KL. job_template.sh / make_jobs.py gained OFFSET and `--init`.
  - Text parity X_rq9_text_t2: text_lora_train.py `--offset` / `--init-adapter` (bundle v6 = v5 + that file only), from X_rq9_text_20I u5600,
    20I train rows, eval plain BM25 k=20.
  - Queues: rql — after the untuned-text k=100 pass (remaining text 100I/200I/k=200 passes deferred), kI_probe (48 updates on the longest
    k=200 items, micro 4 → 2) → **T_res_rq9L_kI** → **X_rq9_text_t2**. rqj — after T_res_rq9L_20I_KL: refs to all 90,447 → **T_res_rq9L_kI_KL**.
    rqk — after T_res_rq9fr_20I_KL and the full refs: **T_res_rq9fr_kI_KL**. T_res_rq9fr_kI (SFT frozen): first free GPU.
  - T_res_rq9fr_20I_90k final 73.2 (u10400 74.5; 90k deprecated).
- ~18:10Z (11-12): rql reclaimed during the stage-2 probe → **rqm**. Probe (48 updates, longest k=200 items first, `kI_probe`): micro 4 OOM;
  **micro 2 OK, peak 64 GB, 26.5 s/update median** (20I: ~4–5). 5,600 updates would be ~42 h per arm → **stage 2 = 2,400 × 8 = 19,200
  held-back questions, eval every 400** (all four AC arms and the text arm alike; ~18–20 h each). T finals as INIT: checked OK (DONE valid).
  Untuned 9B text plain BM25 **k=100 54.8**. AC zero-training read (20I finals): LoRA k=20 **52.0**, k=200 **49.8**, 100I **61.5**, 200I **56.1**;
  frozen k=20 **52.7**, 100I **63.7**. First KL point: **T_res_rq9fr_20I_KL u800 63.0** (frozen SFT u800 58.9).
