# Corpus Q/A and Retrieval Q/A — plan

The next two evaluation sections of the paper. Round 1 (intent) is settled; your answers are recorded below. Round 2 (research) is running, with one agent per section. Its results come back here for your green light, and the runs should finish 24–48 h after it.

## 1. Executive Summary

| | **Corpus Q/A** (ablation-heavy) | **Retrieval Q/A** |
|---|---|---|
| Claim | Activation contexts store one long, fixed corpus. Key/value transfer is expected to be the best channel. | Activation contexts let a reader use many more retrieved passages at the same reader cost. |
| Encoder input | The corpus only, never a question | The retrieved passages: contextual (with the question) and uncontextual arms, trained separately |
| Shape | One reader call | Programmatic: the 9B writes one query, BM25 retrieves, one read. Two rounds only if the benchmark truly needs them. |
| Model | Llama-3.1-8B-Instruct as encoder and reader for every arm (public ungated copy `unsloth/Llama-3.1-8B-Instruct`) | Qwen3.5-2B encoder → Qwen3.5-9B reader |
| Benchmark | LongHealth, at large (Cartridges-scale) context | A wide survey; qualifies only if expanded-BM25 recall@100 − recall@10 ≥ 20 points and text top-50 beats top-10 by ≥ 5 points |
| GPUs (CSAIL) | 3 GH200 + the H200 pair + 1 H200 in all; the extra GH200 and H200 preferably for short jobs such as evals | At most 2–3 of those |

**Corpus grids are semi-independent (your C6):**

| Comparison | Arms | Ratios |
|---|---|---|
| Channel ablation | AC input rows only · input + residual every 4th layer · key/value transfer | 1/20 only |
| Against other approaches | The best channel (expected: key/value transfer) · full corpus in context · no context · the external baseline (open) | Best channel at 1/10, 1/20 and 1/40 |

That's 3 + 2 = 5 AC training runs per corpus, with no "reader LoRA without AC" arm here. Training on several distinct corpora (like 2602.16284's 4 inputs) is allowed, to give us an edge.

**Retrieval arms:** no retrieval · text top-K, up to what fits · reader LoRA with text top-10 · AC contextual and AC uncontextual, each at 1/10 and 1/20. The encoding unit (per passage or a block of passages) is whichever is faster. Targets are the 9B's own answers when reading the full text top-K, kept when correct.

## Corpus Q/A MVP (settled in chat, Oct 18 ≈19:00–19:50Z)

- **Baseline route:** a Cartridges-compatible "own baselines". There is one corpus of about 120k tokens (LongHealth patients 1–10, split at document boundaries) with our own self-study data and our own numbers, set up so Cartridges' Figure 5 (Llama-8B) can be overlaid later. Figure 4 (Llama-3B) is dropped.
- **Training:** a short reconstruction phase on the 10 documents (the same data for every arm), then self-study. Llama-8B writes the question/answer pairs with the full patient document in context and is also the distillation teacher. Rates are 2e-5 / 2e-5, constant after warm-up.
- **x-axis:** question/answer pairs trained on. Budgets: about 8 h for an ablation run and about 16 h for the key/value run.
- **Runs:**

  | Run | Channel | Ratio | GPUs |
  |---|---|---|---|
  | K15 | key/value transfer | 1/15 | CSAIL H200 pair |
  | R15 | input rows only | 1/15 | ORCD RTX PRO 6000 (preemptable) |
  | D15 | input rows + residual every 4th layer | 1/15 | ORCD RTX PRO 6000 (preemptable) |

  1/30 comes later, only if needed or asked for. Then 24 h of hill-climbing. Corpus has priority over retrieval until then.
- **Probe before the runs.** Your instruction: measure, don't assume numbers.
  - **Harness:** Llama in the harness, all three channels.
  - **Corpus:** the corpus and the 200 questions.
  - **Baselines with answer-format checks:** full corpus, no context, and the right document in context.
  - **Self-study synthesis quality and format:** question quality, answer format, and whether some synthetic multiple-choice items are needed.
  - **Short training probes per channel:** seconds per update and memory when the encoder reads 1, 4 and 10 patients, on an H200 and an RTX; loss falling; the answer format kept; checkpoint save and resume.

## 2. Round 2: the open question

### The corpus baseline (the blocker)

We need a published number to stand next to, and the teacher must hold up at long context. The research agent is evaluating four routes:

| Route | What it gives | Main risk |
|---|---|---|
| **A. Switch to Llama-3-8B and reuse Cartridges' single-corpus LongHealth numbers** | Direct comparison with the Cartridge and its baselines | Our harness is built around Qwen. An 8B encoder plus an 8B reader at ~100k context has to fit in training. |
| **B. Qwen3-4B-Instruct-2507 in 2602.16284's setting** (several cartridges, 4 distinct inputs, shortened context) | Reuse their 4B numbers; native long context | Their setting must be matched exactly, and it's multi-cartridge with a shorter context |
| **C. Run the open-source Cartridges code ourselves** on our model and corpus | A like-for-like baseline in our setting | Cost: the self-study data plus training |
| **D. Our own baselines only** (full context, no context), citing Cartridges | Cheapest | No external method in the figure |

The same agent also measures whether the 4B reading the whole corpus is a usable teacher, or whether the 27B is needed (your C4).

### Retrieval

A wider survey than our first list, ranked by your two-part rule, with a preference for benchmarks where one expanded search is mostly enough. The agent measures BM25 and expanded-BM25 recall on CPU, then the untuned 9B at text top-10 vs top-50 on one GPU.

## Accepted decisions (Round 1, read Oct 18 18:42Z)

| # | Decision |
|---|---|
| C1 | LongHealth |
| C2 | The same model as encoder and reader for every arm. Which model depends on the baseline route: Qwen3-4B-Instruct-2507 for its longer native context, or Llama-3-8B. |
| C3 | Large context, Cartridges-scale |
| C4 | The teacher is the 4B reading the full corpus if it holds up at that length, otherwise the 27B. Research compares the two. |
| C5 | Core arms: full context, no context, and AC with input rows only, input + residual every 4th layer, and key/value transfer. No reader-LoRA-without-AC arm. |
| C6 | Semi-independent grids: channels at 1/20; the best channel at 1/10, 1/20 and 1/40 against other approaches |
| R1 | Recall gap ≥ 20 points and accuracy gap ≥ 5 points. Prefer benchmarks where one expanded search is largely enough; expand-search-expand-search only if one is truly needed. Survey a wider set. |
| R2 | One query |
| R3 | Whichever encoding is fastest |
| R4 | 1/10 and 1/20 for both arms |
| R5 | The 9B's own answer with full text top-K, kept when correct |
| X1 | No EC2 (warning email). CSAIL budget plus one GH200 and one H200, preferably for short jobs. |
| X2 | Default, with the size depending on the benchmark |
| X3 | The frontier co-design paragraph is out of scope |

## 3. Milestones

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M1 — Round 1: intent</span>
    <span class="card-oneliner">Shape agreed; 14 decisions recorded above.</span>
    <span class="card-badge">Completed</span>
  </summary>

Your answers came in through the decision form on Oct 18 at 18:42Z. The verbatim text is in `PLAN.md`.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M2 — Round 2: research (two agents)</span>
    <span class="card-oneliner">The baseline route, teacher check, retrieval benchmark pick and run lists with GPU-hours.</span>
    <span class="card-badge">Implementing</span>
  </summary>

- **Corpus:**
  - Cartridges and 2602.16284: setups, models, corpus sizes, question sets and published numbers;
  - routes A–D with feasibility in our harness (Llama support, key/value transfer at long context, memory);
  - 4B vs 27B as the full-corpus teacher on LongHealth;
  - the run list.
- **Retrieval:**
  - a wide benchmark survey;
  - recall measurements;
  - the 9B accuracy check;
  - the pick, training split and targets;
  - the run list.

Each returns a short report. Whichever finishes first is shown to you first, since you may green-light early.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M3 — Runs (24–48 h after the green light)</span>
    <span class="card-oneliner">Training and evaluation, then tables and figure mocks.</span>
    <span class="card-badge">TBD</span>
  </summary>

The run list comes from M2.

</details>
