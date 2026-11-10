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
