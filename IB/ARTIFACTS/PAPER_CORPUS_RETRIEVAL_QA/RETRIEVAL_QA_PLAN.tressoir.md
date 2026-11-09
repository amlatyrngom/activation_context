# Retrieval Q/A on HotpotQA

The plan for the retrieval workstream. BM25 searches with the question alone. The 9B then answers in one of three ways:
- from the top 20 passages as text (baseline 1);
- from the top 400 as text (baseline 2);
- from the top 400 compacted by 20 contextual 2B calls of 20 passages each (ours).

Ours is trained toward the 9B that sees only the gold passages. All defaults were accepted on Oct 20, and the kit is written. The first jobs are queued on CSAIL.

## Executive Summary

### Goal

At 1/20 (the main result), the 9B reading **400 compacted passages** should beat both:
- the usual setup (text top 20);
- the 9B reading all 400 passages as text.

1/10 and 1/40 complete the ratio curve.

### Pipeline

```text
question ─► BM25 (question only) over 509k pooled HotpotQA paragraphs ─► top 400 passages (ranks 1–400)
   │
   ├─ baseline 1: 9B reads ranks 1–20 as text (≈2.1k tokens)
   ├─ baseline 2: 9B reads ranks 1–400 as text (≈45k tokens)
   └─ ours: 20 calls, call i = ranks 20i+1 … 20i+20, then "Question: q" ─► 2B encoder (contextual)
            ─► input rows + residual deltas ─► 9B reader (thinking on)
training target: the 9B (thinking on) reading question + the 2 gold passages; KL on its reply tokens
```

### Arms

| Arm | What the 9B reads | Size | Role |
|---|---|---|---|
| No retrieval | the question | — | floor |
| **Text top 20** | 20 passages | ≈2.1k tokens | baseline 1 (accuracy@20) |
| **Text top 400** | 400 passages | ≈45k tokens | baseline 2 (accuracy@400) |
| Gold context | the 2 gold passages | ≈0.2k tokens | ceiling (the teacher's view) |
| **AC 1/20** | 20 contextual calls | ≈2.3k rows | **main result** |
| AC 1/10 | 20 contextual calls | ≈4.5k rows | curve point |
| AC 1/40 | 20 contextual calls | ≈1.1k rows | curve point |

At 1/20 the reader holds about as many rows as baseline 1 holds tokens, but covers 20× the passages. The reader thinks in every arm.

### Why this setting (measured Oct 20)

Setup: BM25 with the question only, on 500 validation questions. Token counts use the Qwen3.5-9B tokenizer.

| | Top 20 | Top 400 |
|---|---|---|
| Questions with both gold passages retrieved | 54.8% | **82.0%** |
| Gold passages retrieved (share) | 75.4% | 90.8% |
| Text tokens: mean / max | 2.1k / 3.1k | 45k / 57k |

- **Room to gain.** Retrieval alone puts both gold passages in reach for 27 points more of the questions.
- **It fits.** The 9B has 262,144 positions, so text top 400 plus thinking fits.
- **Thinking is long.** The probe's thinking replies average 3.5k tokens (p90 6.5k). This is the main cost of every HF evaluation.

### Accepted (Oct 20)

| Topic | Decision |
|---|---|
| Retrieval | BM25 with the **question only**; no query expansion |
| Baselines | accuracy@20 and accuracy@400 (text), with the reader thinking |
| Encoder mode | **contextual**: each call sees its passages, then the question |
| Calls | **20 calls × 20 passages** |
| Ratios | 1/10, **1/20 (main result)**, 1/40 |
| Channel | **Input + Residuals**, from a reconstruction checkpoint |
| Training target | KL toward the base 9B reading the gold passages; **correct teacher replies only** |
| Questions whose gold passages weren't retrieved | **kept** in training |
| Tuning | fast-turnaround hill-climb of the learning rates and training speed |

### Correction: the starting checkpoint

The pair I proposed, `recon_100m_v1/initial_pair`, is **not** the reconstruction run's final checkpoint. It is that run at step 6,459 of 21,982 (about 32M target tokens). Two candidates are checked:

| Pair | What it is | Channel (read from ac_modules.pt) |
|---|---|---|
| 6459 pair | reconstruction warmup, step 6,459 | rows + deltas at layers 3, 7, …, 31 |
| 75M pair | the finished 75M reconstruction pair (`run_75m_cont`) | the same |

Both are 2B → 9B with a rank-128 reader LoRA. The mixed-run plan notes that reconstruction-only training **raised** the contextual RAG loss, so neither is obviously better for this task. The hill-climb tries both.

## Milestones

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M1 — Data and teacher targets</span>
    <span class="card-oneliner">Question-only BM25 top 1,000 and gold-context teacher replies for 12,000 training and 500 evaluation questions.</span>
    <span class="card-badge">Implementing</span>
  </summary>

#### Planning Overview

1. **Questions (done).**
   - Training: 12,000 HotpotQA training questions (9,681 bridge, 2,319 comparison).
   - Evaluation: 500 validation questions, disjoint from the probe.
   - Development: the probe's first 200 validation questions (160 bridge, 40 comparison), disjoint from evaluation.
2. **Retrieval and teacher replies.** CSAIL job 2716597 (pending).
   - The first submission died at once when CSAIL's /data/scratch hit its disk quota at 11:06Z.
   - The job is resubmitted with all writes on /data/scratch-fast.
3. **Teacher targets.** `rq_teacher.py` stores the base 9B's top-32 targets on every kept reply, plus the no-context references of the trainer's penalty. One file serves every run of a ratio sweep.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M2 — Baselines</span>
    <span class="card-oneliner">No retrieval, gold context, text top 20 and text top 400, with vLLM.</span>
    <span class="card-badge">Implementing</span>
  </summary>

#### Planning Overview

- **Script:** `rq_text_eval.py`. vLLM with prefix caching. The passes of a prompt are adjacent, so they share one prefill.
- **Development run first.** Job 2716613: 200 dev questions, 1 pass. It also builds the dev BM25 lists. These are the reference lines for the hill-climb.
- **Evaluation run:** 500 questions × 3 passes, after M1's retrieval.
- **Scoring:** the probe's rule (final-answer line after the thinking; lenient accuracy, exact match, F1).

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M3 — Training setup and speed check</span>
    <span class="card-oneliner">The kit is written; a smoke run measures the untrained pairs' decode speed and sanity on 32 dev questions.</span>
    <span class="card-badge">Implementing</span>
  </summary>

#### Planning Overview

The kit lives in `IB/TMP/PAPER_RETRIEVAL_QA/kit/` (CSAIL copy: `/data/scratch-fast/ngom/rq_v1/`).

| File | Role |
|---|---|
| `rq_common.py` | prompts, the 20-call student view, the gold-passage teacher view, scoring |
| `rq_teacher.py` | the KL teacher targets, computed once |
| `rq_train.py` | KL training of the pair (canon recipe by default), with held KL every 50 updates |
| `rq_eval.py` | batched HF evaluation with the residual deltas |
| `rq_text_eval.py` | the vLLM text baselines |

**Evaluation speed.** vLLM cannot carry the residual deltas, so evaluation of our arm uses HF generation. The speedups:
- the passes of a question are adjacent, so their 20 parts are encoded once (row cache);
- large prefill batches (default 96 rows). The hybrid 9B keeps a small cache: only 8 of its layers use full attention;
- the reader adapter runs in bf16;
- `--shard I/N` splits the questions over GPUs, and outputs are appended per batch, so a requeued job resumes;
- a lower thinking cap for development runs (4–6k tokens; a length stop scores wrong) than for the final run.

**Smoke run (job 2716614).** Both start pairs, untrained, at 1/20 on 32 dev questions. It measures memory, seconds per batch, and whether the decode returns `</think>` and a final answer.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M4 — Fast-turnaround hill-climb</span>
    <span class="card-oneliner">Short parallel runs at 1/20, each judged within about an hour, then the full runs.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

1. **Short runs, one per GPU, in parallel.**
   - 800 training items each (100 updates of 8).
   - Grid: start pair (6459, 75M) × encoder LR (5e-5, 1e-4, 2e-4), with reader LR 2e-5.
2. **Two signals per run:**
   - **fast:** held KL on 64 held training items every 25–50 updates, printed by the trainer as it trains;
   - **confirming:** dev accuracy on 200 questions, 1 pass, thinking capped at 4k tokens, from the 100-update checkpoint.
3. **Speed knobs**, timed against seconds per update:
   - gradient checkpointing on or off;
   - encoder batch size (tokens per padded encode);
   - dropping teacher replies over 4k tokens.
4. **Full 1/20 run** from the winner. Then 1/10 and 1/40 in parallel. Learning curves at matched training pairs.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M5 — Evaluation and figure</span>
    <span class="card-oneliner">Paired tests against text top 20 and text top 400; accuracy against compaction ratio.</span>
    <span class="card-badge">TBD</span>
  </summary>

#### Planning Overview

- **Main figure:** accuracy on the 500 evaluation questions (3 passes) at 1/10, 1/20 and 1/40.
- **Reference lines:** text top 20, text top 400, no retrieval and gold context.
- **Tests:** paired per question against both text baselines.
- **Matched protocol:** the same thinking cap in every arm. HF generation has no presence penalty (vLLM uses 1.5), so the final comparison also runs the text top-20 baseline through HF.

</details>

## Compute and timing

- **Where:** CSAIL H200s (quanta-h200 and csail-shared-h200). EC2 was torn down at 10:20Z. ORCD is in maintenance until 01:00Z Oct 21.
- **Disk:** CSAIL /data/scratch is over its quota, so every new job writes to /data/scratch-fast.
- **Queue:** the corpus-compression evals and the R_deepS rerun go first. The retrieval jobs follow as GPUs free.
