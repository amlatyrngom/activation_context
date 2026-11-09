# Corpus Q/A MVP — probe report (Oct 18, written by the corpus fork)

Spec: PLAN.md, "Corpus Q/A MVP — settled in chat (Oct 18 ≈19:00–19:50Z) and the probe". Everything below was measured
today unless it says **unverified**. Kit: `IB/TMP/PAPER_CORPUS_QA/kit/` (shipped to CSAIL and ORCD as
`paper_corpus_qa_v1/kit` and `kit2`). Code tree on the clusters: `ws_corpus` (the local `activation/` plus the one fix
below).

## P1 — harness port (Llama-3.1-8B-Instruct as encoder and reader)

- Model: `unsloth/Llama-3.1-8B-Instruct`, revision `4699cc75b550f9c6f3173fb80f4703b62d946aa5`, downloaded without a token into
  the shared caches (CSAIL job 2705189 → `/data/scratch/ngom/activation/hf`, ORCD job 26285774 → `~/orcd/scratch/activation/hf`;
  4 safetensors shards, 16.06 GB). `refs/main` written so offline loads resolve.
- No tool calls anywhere in this section: the Llama-3.1 template renders a tool call as raw JSON. Targets are plain text
  (a letter for MC items, a prose answer otherwise). The template's fixed system header (knowledge/today dates) is the
  same for student, teacher and baselines.
- **Harness fix (one change, `activation/harness/kv_transfer.py`)**: key/value transfer passed the reader a
  `{layer_type: mask}` dict; Qwen accepts it, Llama's `create_causal_mask` does not (`'dict' object has no attribute 'ndim'`).
  `bind_kv_prefix` now returns the plain 4D mask when the decoder config has no `layer_types` (transformers then uses it
  as-is), and `generate_with_kv_prefix` builds its step mask the same way. Qwen path unchanged.
- CPU tests: `kit/tests/test_llama_harness.py` re-runs the harness KV and intervention tests on a 4-layer tiny Llama with
  the real Llama tokenizer (KV: transfer equals a masked full run, generation equals a full recompute, untrained pipeline
  equals the plain model with markers, gradients with and without checkpointing, training moves the encoder, checkpoints
  round-trip): **6 passed, 17 skipped** (skips: fixture-parametrised intervention tests, and three whose expectations are tied
  to the tiny Qwen3.5 config). The original `activation/tests/test_kv_transfer.py` + `test_interventions.py` on Qwen:
  **24 passed** after the fix.
- GPU fwd/bwd per channel with the real 8B: covered by the P5 probes (all three channels trained real updates).
- Channels: K = key/value transfer (side = second view of the same Llama base with its own LoRA, `interleaved` layout);
  R = input rows only; D = input rows + residual deltas at reader layers 3,7,11,15,19,23,27,31 (every 4th; given explicitly,
  since `resolve_intervention_layers(32, 4, ...)` would mean 4 interior points, not "every 4th"). Ratio 1/15 everywhere.
  Reader LoRA rank 128; one shared base with two adapters in every arm.

## P2 — corpus

`kit/build_corpus.py` → `kit/corpus/{documents,questions,token_counts}.json`. LongHealth v5 (AM clone), patients 01–10, one
document per patient in the AM/Cartridges note format (`<text_k>…</text_k>`). Questions carry the AM/Cartridges patient
prefix ("Please answer the question below about the following patient: ID …, Name …, Birthday …, Diagnosis …").

| | Llama tokens |
|---|---|
| patients 01..10 | 11,735 · 12,105 · 12,176 · 11,017 · 9,488 · 10,694 · 10,580 · 9,708 · 9,855 · 10,647 |
| corpus | **108,005** (not ~120k) |
| question | 50–84 (mean 65) |

Gold letters are unbalanced: A 24, B 49, C 61, D 45, E 21 (always-C = 0.305).

## P3 — vLLM baselines (Llama-8B, greedy, GH200, `kit/baselines.py`)

Layout = `corpus_prompts.py` (system line, "Passages: / Passage k:", instruction, question). Direct = `mc_eval.question_text`
("Answer with the single letter…"), 8 tokens; forced = most likely of A–E at the first token. CoT = "think step by step …
end with Answer: X", 1024 tokens.

| context | direct acc | parsed | bare letter | forced acc | CoT acc | CoT parsed |
|---|---|---|---|---|---|---|
| none | 0.030 | 0.05 | 0.05 | 0.330 | 0.375 | 0.94 |
| right record (~12k) | **0.790** | 0.99 | 0.98 | 0.790 | 0.830 | 0.96 |
| all ten records (108k) | **0.715** | 1.00 | 0.93 | 0.715 | 0.695 | 0.925 |

- Sanity vs "Cartridges' Llama-8B ICL ≈ 0.54": **not reproduced — ours is higher (0.715 direct / 0.695 CoT)**. AM's cached
  Llama-3.1-8B full-context generations for the same 10-patient group (`reference_answers_cache/…patients_01-10…`, CoT, 2048
  tokens) score 0.59 with AM's parser (15 unparsed); their 5-patient groups score 0.75 / 0.78. Differences are prompt and
  parsing (direct letter vs option-text CoT), not a bug we found. Reported as is.
- Direct letter ≈ CoT with context (0.79 vs 0.83; 0.715 vs 0.695), so the **eval is direct letter** (8 tokens, `mc_eval.choice`),
  which is also what the trained reader is asked. Without context the base reader refuses to pick a letter (5% parsed); the
  forced choice there (0.33) is the letter prior, close to always-C.

## P4 — self-study data (`kit/synth_corpus.py`)

Writer and answerer: Llama-8B (vLLM). Diversity machinery reused from FIRST_TASK_AWARE_FINETUNING:
covering units of 64–256 tokens per record with round-shifted cuts (686 units per round over the ten records; every unit gets
exactly one item per round), eight clinical question kinds with fixed quotas (detail .20, timeline .14, treatment .14,
finding .12, numeric .10, reasoning .10, negation .05, cross-record .15 = 2–3 units of different notes, require_multiple_sources)
— the bank's nine *source* families do not apply to one clinical corpus, so the quotas are over question kinds —, an
already-asked list per overlapping unit, a content-word-Jaccard dedupe (≥ .6 within a record) and provenance (`study_source`:
record, round, unit ids and character ranges; the writer's evidence quote, checked verbatim). The writer always sees the full
record; the units only steer the question. 25 % of items are written as LongHealth-format 5-option MC (letter target).
Target = Llama-8B's own greedy reply with the full right record + question in the training layout (exactly the teacher
view); an MC item is kept only if that reply's letter is the writer's key. Leakage: content-word Jaccard ≥ .5 to any of the
200 benchmark questions → dropped; max over kept items .476 (164 of 4,573 above .3, half A).

**Probe (~200 pairs, round 0, 20 units per record) vs a plain seed-prompt baseline (Cartridges' five seed prompts, full
record, same answerer):**

| | covering (ours) | seed prompts |
|---|---|---|
| prompts → kept | 200 → 186 | 200 → 78 (120 answers truncated at 320 tokens) |
| mean pairwise content-word Jaccard (same record, all 200 generated) | **0.025** | 0.103 |
| near-duplicate rate (J ≥ .5) / duplicates (J ≥ .8) | 0.21 % / 0.11 % | 0.47 % / 0 % |
| distinct 2-word openings (kept) | 29 / 186 | 17 / 78 |
| top opening share | 0.31 ("what was") | 0.41 ("what is") |
| question words (mean) | 13.5 | 36 |
| target tokens (mean) | 50 | 165 |
| provenance (unit ranges, evidence quote verbatim) | yes, 75 % verbatim | none |

(A first seed run reused one sampling seed per round, which made identical prompts give identical questions, 65 % near-
duplicates; rerun with per-request seeds, figures above.)

Kinds kept (probe): detail 39, cross 28, treatment 28, timeline 27, finding 19, numeric 19, reasoning 18, negation 8; MC 33.

Sample items (probe, verbatim, truncated):
- detail · "How many millimeters is a nodule in the right upper lobe?" writer: "4mm"; target: "According to the passage, the
  nodule … is approximately 1.5 cm … 15 mm" (the teacher's own reply is what trains: correctness of the teacher is not
  filtered for free items).
- cross · "What was the result of the pathology report after analyzing the fluid from the pleural effusion and how did it
  affect the patient's treatment?" target: "…atypical cells consistent with pancreatic cancer cells. This finding confirmed…"
- finding · MC "What does the ultrasound imaging of the right arm reveal?" target "C" (the writer's key).

**Full generation** (48 rounds, records split over two GH200s; per-record already-asked lists make the split exact): the
covering assignment saturates — after ~10 rounds most new questions are near-duplicates of earlier ones for the same record
(round 45: 83 kept of 301). Half A (patients 1–5): 14,486 prompts → 4,573 kept (near-duplicate 8,153, no JSON 1,070, MC key
disagreement 406); every record character covered by ≥ 10 kept items for 98 % of the text; mean pairwise Jaccard 0.035,
near-duplicate rate 0.09 %, 121 distinct openings, top share 0.32. Half B: see below.

## P5 — training probes

All probes: Llama-8B encoder and reader, 1/15, learning rates 2e-5 / 2e-5, 8 examples per update, the exact full-vocabulary KL to the
base reader with the right record + question (`full_context_match_weight` 1, no text loss), no-context penalty on 25 % of examples,
reconstruction first, then study pairs. Encoder views: `one` = the target record alone; `mixed` = target + 3 other records (all ten on
1 example in 8); `all` = all ten. Encode cap `CQA_ENCODE_BATCH_TOKENS=16384` (≈ one record per padded encoder batch) where marked.

| probe | GPU | view | s / update | peak memory | outcome |
|---|---|---|---|---|---|
| kv_one | 1 GH200 | one | 27 | 41 GB | 27 updates; study KL 6.0 → 1.6–2.0, MC gold-letter NLL 8.13 → 1.63, held study 5.47 → 2.56, recon 8.5 → 5.5; no NaN |
| kv_all | 1 GH200 | all | — | OOM | default encode batch (65,536 tokens) |
| kv_all_b | 1 GH200 | all, cap | 205 | 81.5 GB | runs |
| kv_mixed2 | 2 H200 | mixed | ≈ 100 | 112 GB | no cap; stopped at step 6 with a resumable checkpoint; fixed references: MC gold NLL 0.57 with the right record, 5.41 without |
| rows_mixed | 1 RTX PRO 6000 | mixed | 379 (step 1) | OOM (96 GB) | |
| deltas_mixed_b | 1 RTX PRO 6000 | mixed, cap | — | OOM at step 1 | |
| rows_mixed_g | 1 GH200 | mixed, cap | 83–110 (mean ≈ 95) | 93.7 / 97.9 GB | loss 4.38 → 1.14 in 7 updates; too tight for an 8 h run |
| deltas_mixed_g | 1 GH200 | mixed, cap | — | OOM (allocator failed at 97.3 / 97.9 GB, then hung) | |

The kv_one step-27 checkpoint on the 200 questions (direct letter): right view 0.20 (mostly "A", 98.5 % bare letters), all view 0.03
(garbage: a model trained on one-record views does not read ten), none view 0.135 — the eval script works (all view 223 s, right
view 33 s on a GH200) and the views must match training: hence the mixed view.

### Speed: where an update's time goes, and two fixes

**Profile** (`kit3/bench/profile_train.py`: device-synchronized timers around run_channel_mixture's own training; 1 GH200, KV, mixed
view, 8 examples, encode cap 16k; update 2): 113 s = encoder forward 30 s + backward 78 s (gradient-checkpoint recompute + backward)
+ the reader and the rest ≈ 5 s. The full-context teacher states are prepared per block, outside the update. The encoder read
**475k side tokens** in that update, at ≈ 250 TFLOP/s in its forward. So the update is compute-bound, and its cost is the token count.

**Fix 1: the encode cap (padding/masking).** kv_mixed2 had no cap, so each example's 4 records went through one padded encoder batch
with a padding mask (≈ 100 s / update on 2 × H200). With `CQA_ENCODE_BATCH_TOKENS=16384`, each record is its own unpadded batch
(SDPA causal, no mask). K15 launched with the cap and runs **38–49 s / update** on 2 × H200 (updates 1–31) — ≈ 2× from the cap alone.

**Fix 2: per-record re-encoding → shared part encodings.** Each example encodes its own 4 (or 10) records, but an update holds at most
the 10 distinct records (≈ 108k tokens): ≈ 4.4× redundant on one GPU with 8 examples, ≈ 2.4× per GPU with 4.
- The trainer's `share_part_encodings` (SFT micro-batches): identical records among the examples of one micro-batch are encoded once,
  every example reads that encoding, and the encoder gets the sum of their gradients.
- It is switched on without touching the packet (`kit3/bench/train_shared.py --micro-batch-examples N`, via
  `run_corpus_shared.sh` with `CQA_MICRO`). Each update keeps the same examples in the same order, and the no-context penalty draws
  are taken in the global order as before, so a run can switch at a resume boundary and its curve stays one run.
- Correctness (tiny Llama): `test_kv[test_shared_part_encodings_and_part_grouped_order]` (key/value) and
  `kit/tests/test_llama_shared_rows.py` (rows, deltas) pass — same losses and gradients as separate encodings, with fewer encodes.
- Real model: kv_mixed2's packet inputs, micro-batch 4, 1 H200. Update losses 1–6 match the micro-batch-1 run to bf16 noise
  (8.4650 / 8.4632, 8.6522 / 8.6525, 8.3627 / 8.3628, 8.3304 / 8.3306, 8.2020 / 8.2010, 7.8865 / 7.8819).

| test (8 examples / update, 1 GPU, mixed view, cap 16k) | s / update | peak |
|---|---|---|
| KV, no sharing (GH200 profile) | 97–113 | 76 GiB |
| KV, micro 4 shared, GH200 | 58 (update 1), then OOM | > 96 GB |
| KV, micro 8 shared, GH200 | OOM at update 1 | > 96 GB |
| **KV, micro 4 shared, H200** | **43–63 (mean ≈ 54)** | 102 GB |
| rows, no sharing (GH200) | 83–110 (mean ≈ 95) | 94 GB |
| **rows, micro 8 shared, H200 (R15's first job, updates 1–38)** | **28–36** | 127 GB |
| deltas, micro 8 shared, H200 | OOM at 139.8 GiB | |

| **KV, K15 itself (2 × H200, micro 4 shared per GPU), updates 33–74** | **22–50 (mean 32.1, median 33.8)** vs 44.9 before (updates 1–32) | 102 GB / GPU |
| **deltas, micro 4 shared, H200 (D15b, updates 1–15)** | **47–55** (no unshared baseline: deltas OOM on GH200 and RTX) | 137 GB |

Sharing needs the H200's memory: up to 10 records sit in one autograd graph. Measured speedup on one GPU: ≈ 1.9× (KV, micro 4) and
≈ 2.9× (rows, micro 8). For K15 (2 GPUs × one shared group of 4) I expected ≈ 1.6–1.8×; measured **1.40×** on the mean. Its ceiling is the 10 distinct
records, about 9 of which each GPU still encodes per update. Going further needs records sharded across GPUs, or company-grouped
updates; the second changes the data order.

### Decisions

- **Examples per update: 8** global (spec). K15: 2 GPUs × 4 examples (one shared micro-batch of 4 per GPU from update 33). R15:
  1 GPU × 8 (micro 8, shared). D15: 1 GPU × 8 (micro 4, shared: micro 8 does not fit).
- **Encoder view: mixed** (target + 3 others, all ten on 1 in 8). One-record training does not transfer to the ten-record eval
  (kv_one: all view 0.03), and all ten every time costs ≈ 2.5× and needs the cap.
- **Reconstruction: 240 examples (30 updates)** before the study pairs. In the probes the recon loss falls fastest in the first ~30
  updates; longer recon is time not spent on the measured objective.
- **Study pairs**: 8,354 generated (P4), 8,330 after the 24 held items. Every arm has the same cap of **6,160**, so every arm
  trains on the same prefix of the same order. That is < 1 pass: no pair is seen twice. The learning rate is constant after warm-up,
  so the cap is only an upper bound. The x-axis is pairs trained.
- **MC share**: the natural 19 % of the study pairs (1,568 MC of 8,354), with no reweighting.
- **Checkpoints and snapshots**:
  - Resumable checkpoint every 25 updates.
  - Snapshots are taken at the panel reports, every 100 updates. K15's rule is 5 % up to 25 %, then multiples of 10 %; with reports
    every 100 of 800 updates this gives snapshots at updates 100, 200, 400, 500 and 800 (300, 600 and 700 fall on 35 / 75 / 85 % and
    are skipped), so the gaps are ≈ 55–110 min at 32 s / update. A snapshot is skipped, with a line in
    `snapshot_skipped.jsonl`, when /data/scratch has < 300 G free.
  - R15b / D15b: snapshots every 10 %, i.e. at every report (100 updates ≈ 55 min for R15b, ≈ 85 min for D15b).
  - Each snapshot gets the direct-letter 200-question accuracy and parse rate on the all / right / none views.

### K15

| | |
|---|---|
| examples per update | 4 per GPU, 8 global (2 × H200) |
| reconstruction | 240 examples = 30 updates |
| study pairs available / per pass (cap) | 8,330 / 6,160 |
| expected passes | < 1: one partial pass. At 32 s / update the cap's 800 updates take ≈ 7.8 h of training, so the run finishes ≈ 05:20Z Oct 19, before the window, with ≈ 8 h of the 16 h budget left (what to do with it is open) |
| x-axis | Q/A pairs trained (the reconstruction examples are reported separately) |

### Runs and the CSAIL maintenance window (Oct 19 08:00Z → Oct 20 02:00Z)

- **K15**:
  - Attempts: 2705643 landed on aia-h200-12, whose NVLink fabric manager was down (CUDA error 802), so it was resubmitted. 2705694
    trained updates 1–32 without sharing (38–49 s / update).
  - It was paused at update 32 (partial checkpoint) and resubmitted as 2705909 in the same folder with shared encodings (micro 4)
    and the new snapshot policy, with a plain `--time` that ends before the window.
  - It resumes on CSAIL after the window, from `partial_epoch.json`. Moving it to ORCD would need its `optimizer.pt`: the resume
    refuses a checkpoint without Adam moments (`run_qa_9b.resume_checkpoint`), and copying that file is the user's decision.
  - Lesson: with `--time-min`, `SLURM_JOB_END_TIME` at the start reflects the minimum, so run_corpus gave the first job only 9,241 s.
    Use a plain `--time` with run_corpus.
- **R15b, D15b** (= the R15 / D15 arms with the 6,160 cap):
  - ORCD mit_preemptable, 1 H200 each, from scratch. The jobs are preemptible, requeue themselves, resume in place, and are not
    affected by the CSAIL window.
  - mit_normal_gpu estimated a start on Oct 22, so neither runs there.
  - The first R15 (cap 2,064, 38 updates) was stopped for the larger cap; the first D15 (micro 8) ran out of memory.
- **Evals**: K15 on a CSAIL GH200 watcher. R15b / D15b on an ORCD RTX PRO 6000 watcher.
- **Status at 00:30Z**: K15 at update 215 (≈ 25–36 s / update). R15b at update 190 (≈ 35 s). D15b at update 84 (≈ 52 s; preempted
  once, resumed in place at 23:23Z).
- **K15 storage**: snapshots live on /data/scratch (symlinked from the run's packet), because scratch-fast has ≈ 34 G left under the
  200 G cap; checkpoints stay on scratch-fast.

Fresh-pair (update 0) evals of K15, direct letter, accuracy / parse rate: all view 0.025 / 0.225, right 0.065 / 0.25, none 0.085 / 0.19.

### K15 snapshots (direct letter, 200 questions; accuracy / parse rate; answer letters)

| update | pairs trained | all ten | right record | none |
|---|---|---|---|---|
| 0 (fresh pair) | 0 | 0.025 / 0.225 | 0.065 / 0.25 | 0.085 / 0.19 |
| 100 | 560 | 0.290 / 0.995 (B 169, C 22, D 5, E 2, A 1) | 0.375 / 1.00 (B 101, D 31, C 30, E 25, A 13) | 0.125 / 0.215 (157 unparsed) |
| 200 | 1,360 | 0.325 / 1.00 (A 37, B 134, C 18, D 6, E 5) | 0.400 / 1.00 (A 37, B 105, C 25, D 19, E 14) | 0.070 / 0.13 (174 unparsed) |

R15b (rows): u0 0.000 / 0.06 / 0.08 (all / right / none; all-view parse 0). u100 (560 pairs): all 0.435 / 1.00 (A 19, B 100, C 36,
D 28, E 17), right 0.455 / 1.00 (A 34, B 84, C 34, D 31, E 17), none 0.025 / 0.025. Gold letters: A 24, B 49, C 61, D 45, E 21.

References: base reader in context 0.715 (all ten) / 0.79 (right); no context 0.03; always-B = 0.245.

### Letter balance of the synthetic MC pairs

K15's 6,160 pairs include 1,170 MC. Their keys are A .18, B .34, C .28, D .10, E .10. The teacher's reply letters are identical to the keys, 1,170 / 1,170, because an MC item was kept only when the teacher's letter equalled the writer's key. LongHealth gold is A .12, B .245, C .305, D .225, E .105.
- B is over-represented by ≈ 1.4× and D under-represented by ≈ 2.2×.
- B is the most common key in 22 of 32 rounds; its share per round ranges from .19 to .52, with no trend.
- The 396 MC rows of the continuation are similar: A .20, B .31, C .29, D .09, E .11.

The u100 all-view bias (B 169 / 200) is much stronger than this skew. So the skew may add to it but does not explain it alone; early-training collapse onto a letter prior is the likelier main cause (to be checked on later snapshots).

**Fix, for the continuation and future packets only** (`kit/permute_mc.py`; K15's running data is unchanged):
- Keys are dealt uniformly over A–E, in seeded rounds of five letters.
- Each MC row's options are permuted so that the correct option sits at its new key; the other options take the remaining letters in a seeded order.
- The target is regenerated: greedy Llama-8B with the full right record and the permuted question, which is exactly the teacher view the KL reads. Distillation therefore stays exact, with no remapping of letter tokens.
- Provenance is kept: the old options, key and target, plus `teacher_agrees`.
- A sanity pass first regenerates 64 rows in their original order and must reproduce the stored targets.

### K15 continuation (decision (a): after the cap, the 2,170 unused pairs; stops at the maintenance pause)

- `kit/bench/build_continuation_packet.py` rebuilds build_corpus_packet's row order and held draw. It checks that the source packet trained exactly the first 6,160 rows of that order, with the same ids and the same encoder companies. It takes the remaining rows, keeping the rounds in order and shuffling within each round under a new seed, and cuts them to a multiple of 8: 2,168 pairs = 271 updates. There is no reconstruction phase, and K15's task panel is copied.
- Initial pair: K15's `packet/AC_MODELS/rag_poc/epoch_001` plus `LORAS/rag_reader/epoch_001`, used in place; its `optimizer.pt` is never copied.
- With `resume_optimizer`, the first attempt loads K15's Adam moments and its 800-update clock. `recipe.updates_offset = 800`, so later resumes check the clock against 800 + step and snapshots number their updates on K15's axis.
- Runner: `kit/run_continue.sh` with `csail_h200_continue.sbatch`. It refuses to start unless the source optimizer's clock equals `CQA_OFFSET`. Its folder, K15c, lives under /data/scratch, because scratch-fast is near the 200 G cap.
- **Test (`csail_h200_continue_test.sbatch`, 1 H200): passed.**
  - Setup: an 8-update source run to completion, then a continuation attempt from its `epoch_001`, then a resume of that
    continuation.
  - The first attempt restored "optimizer moments and 8 updates" from the source's `epoch_001/optimizer.pt`. Its first update was
    global update 9.
  - The resume restored "12 updates" from the continuation's own `epoch_001_step_00004`.
  - Both attempts exited 0 at their duration limits.
  - A first run of the test refused both attempts, because the test gave each attempt < 600 s. That was a test-setting error.
- **K15c as placed**:
  - Packet `build_continuation_packet2.py`, built ahead of the run: 2,128 pairs = 266 updates.
  - Pairs come from `permute_mc.py`'s uniform-key rows. Its check passes: the trained 6,160 rows are identical in both files.
  - Of the 396 MC rows: keys 79–80 per letter; the regenerated teacher agrees with the key on 358. The 38 that disagree are dropped.
  - Sanity pass: 63 / 64 original prompts reproduced their stored target. The one miss is vLLM batch nondeterminism; the rebuilt
    questions are identical.
  - Job: CSAIL `csail_h200_continue2.sbatch` on 2 × H200, micro 4 shared, `--time=02:00`, `--dependency=afterany:` K15's job. Its
    eval watcher (GH200) has the same dependency.
  - It continues after the maintenance window for whatever is left.

### Maintenance hand-off (Oct 19)

**The window is 12:00Z Oct 19 → 06:00Z Oct 20, not 08:00Z → 02:00Z.** CSAIL Slurm prints local time (EDT, UTC−4). This was checked on the login node at 04:41Z:
- `date` printed 00:41 EDT; `date -u` printed 04:41 UTC.
- K15 job 2705909 shows StartTime 2026-10-18T18:20:14; its console logs the same start as 22:20:15Z.
- Reservation `maintenance_october_19` shows 2026-10-19T08:00:00 → 2026-10-20T02:00:00 local, which is 12:00Z → 06:00Z.

The earlier "08:00Z" in this report read those local times as UTC.

ORCD has its own window: `monthly_maint`, 2026-10-20 00:00 → 21:00 local (EDT), which is **04:00Z Oct 20 → 01:00Z Oct 21**. An ORCD job whose time limit runs past 04:00Z Oct 20 will not start ("ReqNodeNotAvail, Reserved for maintenance").

**Consequence: K15c runs entirely on CSAIL before the window.**
- At 04:40Z K15 was at update 647/800, at ≈36.5 s/update, so it ends ≈06:15Z.
- K15c (2706531) and its eval watcher (2706532) start after K15 (afterany). Their time limits were 1:20 for a while, so they could start before the misread 08:00Z cutoff. They are back at 4:30, which fits before 12:00Z.
- Projection: 266 updates at ≈33–37 s/update gives an end ≈09:15Z, and the evals finish before 12:00Z.
- No optimizer.pt copy was needed for K15c.
