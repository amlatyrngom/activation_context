# Retrieval Q/A plan (agent-facing source)

Projection: `RETRIEVAL_QA_PLAN.tressoir.md`. First written Oct 20 ≈10:30Z. Reshaped ≈10:50Z to the top-20 / 20× setting
(user: "instead of top 10, we placed the top 20. Then expands 20x instead of 10x. The two baselines are accuracy@20 and
accuracy@400. This is the kind of setting I am looking for."). Research basis: `research/RETRIEVAL_RESEARCH.md` (Oct 18), plus the
measurements below (`IB/TMP/PAPER_RETRIEVAL_QA/fit_n/`: fit_n.py, recall_k.py, fit_20x20.py, run Oct 20 in the container).

## User intent (Oct 20)

- Benchmark: HotpotQA with the pooled corpus (509,315 paragraphs), as the research recommends.
- Baseline 1: the 9B expands the question into a query, BM25 returns the top 20, the 9B answers (accuracy@20).
- Baseline 2: the same with the top 400 as text (accuracy@400).
- Ours: the 9B expands, BM25 returns the top 400, and the 9B answers from contextual 2B activation-context (AC) calls.
  - Compaction ratios 1/10, 1/20, 1/40. The user suspects 1/50 might do a decent job.
- Training: KL toward the 9B reading the golden context (the gold passages), not toward the 9B reading the long context.
- History: the first plan used top 10 × 10 calls (100 passages). The user then asked for recall at 20 and 400 and chose top 20
  expanded 20× (400 passages).

## Accepted (Oct 20, ≈10:45–10:55Z; user in chat + plan form)

- **Retrieval:** BM25 with the question only, no expansion. Form: rqa.query feedback "Let's actually go with question-only. Simple.";
  chat: "question-only bm25. No more expansion."
- **Encoder mode:** contextual. "AC encoder sees both passages and question, so we are in contextual mode here."
- **Ratios:** 1/10, 1/20 (MVP; "this should work well"), 1/40.
- **Channel:** Input + Residuals, "starting from some good reconstruction checkpoint". Proposed: CSAIL
  `recon_100m_v1/initial_pair` = reconstruction_warmup_110m_v1 final (golden recipe, 2B → 9B, 110M tokens, ratios 1/30–1/10,
  held-out reconstruction gold NLL 0.175; reader LoRA r128 on Qwen/Qwen3.5-9B). Its intervention setting is to be confirmed from
  ac_modules.pt before launch (canon golden recipe = rows + residual deltas at every full-attention layer).
- **Tuning:** "Hillclimb on the LR, training speed, etc." Hill-climb at 1/20 on the development set, then carry over.
- **Gold-missing training questions:** kept (form: rqa.gold_missing.keep).

## Resolved (user, Oct 20 ≈11:00Z: "All the defaults and recommendation LGTM. Try to make the (accuracy, speed) hillclimbing more 'fast-turnaround'")

- Q2 calls: 20 × 20 passages. Q3 teacher targets: correct replies only (lenient accuracy), sample 0, finish = stop.
- Q1 start checkpoint, **corrected (11:15Z)**: `recon_100m_v1/initial_pair` is reconstruction_warmup_110m_v1 at epoch 1 step 6,459 of
  21,982 (≈32M target tokens; stage record "complete": false), not the final. Candidates (both 2B → 9B, reader LoRA r128,
  intervention_frequency −1 = rows + deltas at layers 3, 7, …, 31, max_view_rows 4096; read from ac_modules.pt on CSAIL):
  `PAIR6459` = initial_pair/{ac,reader}; `PAIR75M` = recon_100m_v1/run_75m_cont/{AC_MODELS/rag_poc,LORAS/rag_reader}/epoch_001
  (the finished 75M pair; MIXED_RUN_PLAN.md notes reconstruction-only training raised rag_qa/contextual 1.099 → 1.317). Both go
  into the hill-climb.

## Kit (Oct 20 ≈11:20Z; `IB/TMP/PAPER_RETRIEVAL_QA/kit/`, CSAIL `/data/scratch-fast/ngom/rq_v1/{kit,code_rq1,env.sh,jobs,out}`)

- rq_common.py: READ prompt (the probe's), student = [user: READ head, 20 parts (kind "search"; encoder messages =
  qa_encoder_messages([passages 20i+1..20i+20 with global numbers], question) → passages then "Question: q"), "Question: q"] +
  teacher reply (reasoning_content + content, enable_thinking True); teacher = [user: READ with the gold passages] + reply.
- rq_teacher.py: base-9B (lora None) top-32 targets for every kept reply + the penalty's no-context references (one file per
  reader start; independent of ratio and calls). rq_train.py: KL, teacher_targets_file, held KL panel (64 of the last 200 train12k).
- rq_eval.py: trainer._sample_completions_batched, passes adjacent (row cache 2 → one encode per question), batch 96, bf16 reader
  adapter, --shard I/N, append per batch (resumable), sampling T 1.0 / top_p .95 / top_k 20 (no presence penalty in HF).
- rq_text_eval.py: vLLM prefix caching, arms none / gold / topK; dev200 BM25 built on first use (DATA/dev200_top1000_q.npy).
- code_rq1 = the workspace activation package at 11:20Z. DATA = /data/scratch-fast/ngom/am_ec2_out/rq_m1 (rq_data2 output dir).

## Fast-turnaround hill-climb (M4)

- Short runs, one per GPU in parallel: 800 items (100 updates); grid start pair {6459, 75M} × encoder LR {5e-5, 1e-4, 2e-4},
  reader 2e-5. Fast signal: held KL every 25–50 updates; confirming signal: dev200 accuracy, 1 pass, max_new 4096, at u100.
- Speed knobs: gradient checkpointing, CQA_ENCODE_BATCH_TOKENS, dropping replies > 4k tokens.

## Measurements (Oct 20)

Setup: the probe's 500 validation questions, the probe's 9B queries, and the repo BM25 (bm25s defaults, English stop words).
Each cell is mean recall / all-gold recall (both gold passages found).

| Search text | @10 | @20 | @50 | @100 | @200 | @400 | @1000 |
|---|---|---|---|---|---|---|---|
| **Raw question (chosen)** | 70.9 / 47.4 | 75.4 / 54.8 | 81.7 / 65.0 | 84.6 / 69.8 | 87.9 / 76.4 | 90.8 / 82.0 | 92.7 / 85.6 |
| 9B query | 70.0 / 47.0 | 75.5 / 54.0 | 81.9 / 65.2 | 84.9 / 71.0 | 88.7 / 77.8 | 91.0 / 82.4 | 93.3 / 86.8 |
| Question + query | 71.7 / 48.8 | 77.4 / 57.6 | 82.7 / 67.0 | 87.0 / 74.4 | 90.3 / 80.8 | 91.9 / 84.0 | 94.1 / 88.4 |

Qwen3.5-9B tokens (passages as "title\ntext", over 100 of the 500 questions):

| | 9B query | question + query |
|---|---|---|
| Top 20: mean / p90 / max | 2,087 / 2,560 / 3,094 | 2,087 / 2,608 / 3,379 |
| Top 400: mean / p90 / max | 44,664 / 51,948 / 56,394 | 45,754 / 52,906 / 56,685 |
| One 20-passage call: mean / p99 / max | 2,233 / 3,176 / 3,720 | 2,288 / 3,230 / 3,588 |

A HotpotQA passage is one Wikipedia introductory paragraph: median 450 characters (≈100 tokens), p90 841, p99 1,371.

**Fit.**
- Qwen3.5-9B has 262,144 positions. Text top 400 is at most 57k tokens, plus 3–6k of thinking.
- Reader rows for 400 passages (≈45k tokens): ≈4.5k at 1/10, ≈2.3k at 1/20, ≈1.1k at 1/40, ≈0.9k at 1/50.
- At 1/20 the reader sees about as many rows as baseline 1 sees tokens (≈2.1k), for 20× the passages.
- Per 20-passage call: ≈223 rows at 1/10 and ≈56 at 1/40 (≈2.8 per passage).
- A training example is ≈46k encoder tokens through the 2B (20 calls). M3 measures training speed.

## Arms

| Arm | Reader input | Role |
|---|---|---|
| No retrieval | question | floor |
| Text top 20 | ≈2.1k tokens | baseline 1 (accuracy@20) |
| Text top 400 | ≈45k tokens | baseline 2 (accuracy@400) |
| Text top 100 | ≈11k tokens | optional middle point |
| Gold context | the 2 gold passages | ceiling, the teacher's view |
| AC contextual 1/10, 1/20, 1/40 (1/50 optional) | 20 calls × (question + 20 passages) → rows | ours |

The reader thinks in every arm (research finding 1).

Query: none. BM25 searches with the question alone (accepted). Token sizes above were measured with the 9B-query retrieval;
question-only retrieval gives nearly the same lists' lengths (same corpus, same k).

## Training

- **Starting pair:** PAIR6459 or PAIR75M (hill-climb decides; Input + Residuals).
- **Data:** 12,000 of the 90,447 HotpotQA training questions (2 gold passages each, seed 1). Per question:
  - BM25 top 1,000 with the question only (stored; the top 400 are used);
  - the teacher: the 9B (thinking on) reading the question + gold passages.
- **Teacher targets:** correct teacher replies (lenient accuracy) are kept. Top-k teacher logits on the reply tokens are
  precomputed, as for AM's am_teacher targets.
- **Student:** the 9B reading the question + 20 contextual AC calls (call i = question + passages of ranks 20i..20i+19), trained
  by KL toward the teacher over the teacher's reply tokens.
- **Gold coverage:** questions without both gold passages in the top 400 (18% with the question only) are kept (accepted).
- **Hyperparameters:** start at canon's recipe (encoder 5e-5, reader 2e-5), 8 examples per update, cosine, checkpoints every
  250 updates; hill-climb the encoder LR (≈5e-5 to 2e-4) and training speed at 1/20 on the development set.
- **Runs:** 1/20 first (MVP), then 1/10 and 1/40 with the winning settings, each from the Q1 pair.

## Evaluation

- **Questions:** 500 held-out validation questions (`m1/eval500.questions.jsonl`, disjoint from the probe's 500). The 200 probe
  questions that were read are the development set.
- **Metrics:** lenient accuracy, exact match and F1, over 3 sampled passes.
- **Tests:** paired against text top 20 and text top 400.
- **Main figure:** accuracy vs compaction ratio, with text top 20, text top 400, no retrieval and gold context as horizontal lines.

## Compute

- **Where:** CSAIL quanta-h200 / csail-shared-h200. EC2 was torn down Oct 20 10:20Z; ORCD is in maintenance until 01:00Z Oct 21.
- **Data + teacher:** ≈2.5 h on one H200 (research estimate).
- **Text baselines:** top 400 is ≈45k prompt tokens per read, so ≈1 h per pass of 500 on one H200 (estimate).
- **Training:** per-arm time not yet measured (M3).

## Milestones

- **M1 Data and teacher targets** (`IB/TMP/PAPER_RETRIEVAL_QA/m1/`):
  - build_questions.py done (≈10:34Z). train12k = 9,681 bridge + 2,319 comparison; eval500 = 424 bridge + 76 comparison;
    pooled corpus verified identical to the probe's.
  - rq_data.py (job 2716525, with the 9B query step) was cancelled while pending; rq_data2.py = question-only BM25 top 1,000 +
    gold-context teacher reads (`paper_retrieval_qa_v1/m1_20261020/`).
- **M2 Baselines:** no retrieval, text top 20, text top 400, gold context (+ optional top 100).
- **M3 Training setup and speed check:** contextual 20-call examples; a few updates at 1/20 and 1/40 (memory, seconds per update).
- **M4 Train and hill-climb:** 1/20 first (LR and speed sweep), then 1/10 and 1/40.
- **M5 Evaluate:** paired tests and the figure.
