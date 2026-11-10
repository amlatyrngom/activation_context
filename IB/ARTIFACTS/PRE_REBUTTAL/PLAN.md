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
