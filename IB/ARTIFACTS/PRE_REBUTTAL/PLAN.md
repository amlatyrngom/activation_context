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
