# Corpus Q/A and Retrieval Q/A — paper evaluation sections (agent-facing source)

Round 1 (intent), written Oct 18 ≈18:30Z on the user's request. The human projection is `PLAN.tressoir.md`; it carries every
decision below. Round 2 (research: datasets, training data, setups; two research subagents) starts once the Round 1 decisions are
answered. After Round 2 the user green-lights, and the whole run should finish ≈24–48 h later. Budget: retrieval at most 2–3 GPUs,
everything else to corpus.

## User intent (Oct 18, verbatim where it matters)

- **Corpus Q/A, the ablation-heavy section.** "take something like what Cartridges did with LongHealth (combining documents into
  something that barely fits context). Then study that thing and only that thing. Like, the encoder only ever receives the corpus as
  input I think, so all it's representation power goes to this. This must stay uncontextual. This is also not agentic. Ablation: KV
  transfer, input-only, Input+Residuals every 4th, etc. Might have to use the Qwen 3 4B model. Note that this differs from our
  previous experiments, which varied the encoder input."
- **Retrieval Q/A.** "find a benchmark whose recall@10 is **noticeably** worse than its ... recall@20, recall@50 or recall@100 on
  bm25 or, better, expanded bm25 (9B formulates query from task, retrieve, answer). Then, show how ACs can consume all that context and
  place the answer in condensed form. Here, I'd have an explicit contextual encoder arm, as uncontextual, when the input varies
  per-question, is a hard problem. With contextual, I am pretty sure even 1/20th should be competetitive lol, but keep an uncontextual
  arm. (should be trained independently). Note that won't be the agentic style I previously drafted, but the more structured
  programmatic expand-retrieve-answer style. Should use the 2B/9B pair."
- Process: "Round 1: intent. Write the document, resolve all ambiguities, etc. Round 2: research. Find specific datasets, training
  data, and setups, likely use 2 subagents here. The present them to me." "Retrieval research can take several hours".

## Fixed by the user (not open)

| Item | Corpus Q/A | Retrieval Q/A |
|---|---|---|
| Encoder input | the corpus only, never a question (uncontextual) | the retrieved passages; contextual AND uncontextual arms, trained independently |
| Shape | one reader call, not agentic | programmatic: 9B writes a query → BM25 → one reader call; not the ReAct agent of the draft |
| Model | probably Qwen3-4B (reason below) | Qwen3.5-2B encoder → Qwen3.5-9B reader |
| Main variable | the channel (KV transfer, input rows only, input + residual deltas every 4th layer, …) | how many retrieved passages the reader can use at a fixed reader budget |
| Benchmark | Cartridges-style concatenated corpus (LongHealth-like) | one where expanded-BM25 recall@10 is clearly below recall@50/100 |

## Facts from the code and canon that shape the design

- **KV transfer needs a full-attention reader and an encoder of the same architecture.** `activation/harness/kv_transfer.py`: the
  encoder's keys/values at marker positions become attention entries at every reader layer; it "needs a full-attention decoder";
  residual deltas and KV transfer are not combined in one payload. The earlier KV arms (`IB/TMP/G5_CURRICULUM/kv1/ac_recon/arms.json`,
  KVa) used Qwen3-4B as both encoder (second copy, own rank-128 adapter, one marker per bin) and reader. Qwen3.5 is hybrid (linear
  attention), so only the "overwrite" KV mode works there. Hence Qwen3-4B for the corpus section.
- **The other channels already run on a Qwen3-4B reader.** G4 in `IB/TMP/G5_CURRICULUM/*/ac_recon/run_recon.py`: Qwen3.5-2B encoder →
  Qwen3-4B reader (LoRA 128), input rows + deltas at layers 3, 7, …, 35 (every 4th). `intervention_frequency` 0 / N / −1 is the
  canon's whole option set (`ACTIVATION_CONTEXT_CANON.md:28`).
- **Earlier evidence on KV:** canon `:28` says the K/V-slot and native K/V channels were stronger (by 0.02 at 1/32 with a bare reader,
  0.10 with a LoRA'd reader at the long budget) and were dropped only for simplicity. The corpus section is where that question gets
  a clean answer.
- **QA conditioning is explicit** (`encoder_sees_question`, canon `:62`); questions are held out, the source corpus need not be
  (`:58`), which is what Cartridges does (it trains on the corpus itself through synthetic questions).
- **BM25 exists** in `activation/dataset/dataset_index.py` / `dataset_study.py` (RETRIEVAL_TRAINING folder); a search part encodes
  "Passage N:" blocks (canon `:63`).
- **Lesson from the synthesis section:** a tuned reader without AC ("Results + LoRA") can take most of a gain. Both sections need the
  matching "reader LoRA, no AC" arm trained on the same data.
- **Compute right now:** EC2 nodes torn down (user: near limits); CSAIL rule ≤2 GH200 + the H200 pair; ORCD reservation 23493337 off
  limits. Results + Summary currently holds 2 single H200 jobs on the CSAIL shared tier (until ≈22:00Z).

## Proposed shape — Corpus Q/A

- **Corpus:** one fixed corpus that barely fits the reader's context (LongHealth: several patients' records concatenated; Cartridges'
  other options were QASPER papers concatenated and the MTOB grammar book). Questions on that corpus are the evaluation; training
  questions are synthetic questions about the same corpus ("self-study"), never the evaluation questions.
- **Encoder:** sees only the corpus. The corpus's rows are computed once and reused for every question. Since the encoder only ever
  sees this corpus, the comparison is about how well each channel stores one fixed document, not about generalization.
- **Training signal (proposal):** context distillation: the target is the same reader answering with the whole corpus in its context
  (KL over the answer tokens), as Cartridges did; our trainer supports KL (canon `:42`).
- **Arms:**

  | Arm | What the reader gets | Purpose |
  |---|---|---|
  | Full corpus in context | the text | ceiling-ish reference |
  | No context | question only | floor |
  | Reader LoRA, no AC | question only; LoRA trained on the same synthetic data | "is it just parametric memory?" |
  | AC input rows only | rows at the input | channel ablation |
  | AC input + residual every 4th layer | rows + deltas at every 4th layer | channel ablation (current default family) |
  | AC input + residual every layer (optional) | rows + deltas at every full-attention layer | channel ablation |
  | AC KV transfer | encoder keys/values in every reader layer | channel ablation; closest to a Cartridge |
  | BM25 top-k text (optional) | retrieved chunks of the corpus | standard alternative |
  | Directly trained KV cache = Cartridge (optional) | trained keys/values, no encoder | the published method |

  Each AC arm at 2 compression ratios (proposal 1/10 and 1/40; Cartridges' sizes are ≈1/50 of a 100k corpus).

## Proposed shape — Retrieval Q/A

- **Pipeline:** question → 9B writes a search query (expansion) → BM25 top-K → one reader call with the passages (text or AC) →
  answer. No agent loop.
- **Benchmark criteria (proposal):** with expanded BM25, gold-evidence recall@100 − recall@10 ≥ 20 points, AND the untuned 9B with text
  top-50 beats text top-10 by ≥ 5 points (otherwise more recall doesn't buy accuracy). Candidates for research: MuSiQue, 2WikiMultihopQA,
  HotpotQA, FRAMES, BrowseComp-Plus, Natural Questions / TriviaQA (open), BRIGHT-style reasoning retrieval.
- **Main comparison at a fixed reader budget:** text top-10 vs AC top-100 at 1/10 cost about the same reader tokens. x = passages
  retrieved (10, 20, 50, 100), y = accuracy.
- **Arms:** no retrieval; text top-K (up to what fits); tuned reader LoRA with text top-10 (no AC); AC uncontextual at 1/10 and 1/20;
  AC contextual at 1/10 and 1/20 (+1/40 optional). Contextual and uncontextual trained independently.
- **Training data (proposal):** the benchmark's training split, same retrieval pipeline; target = the 9B's own answer when reading the
  full text top-K (kept if correct), so the AC learns to match full reading; gold-answer SFT as fallback.

## Open decisions (projected)

C1 corpus benchmark · C2 corpus models · C3 corpus size · C4 corpus training signal · C5 corpus arms/baselines · C6 corpus ratios ·
R1 retrieval selection rule · R2 query expansion · R3 encoding unit · R4 retrieval ratios · R5 retrieval targets · X1 compute pool ·
X2 eval size/seeds · X3 draft's frontier-model co-design paragraph.

## Round 2 (after the answers)

Two research subagents, one per section, each returning a concrete setup (dataset files, sizes, token counts, baseline numbers we
can reproduce, training data plan, run list with GPU-hours):

- Corpus: Cartridges paper/code details (corpus construction, question format, self-study recipe, sizes), LongHealth availability
  and licence, Qwen3-4B context and KV-transfer readiness in our harness (KVa status), training cost per arm.
- Retrieval: measure BM25 / expanded-BM25 recall@{10,20,50,100} on candidate benchmarks on CPU in the container, then the untuned 9B's
  accuracy with text top-10 vs top-50 on a small sample (needs a GPU); pick the benchmark; training split and targets. May take hours.

## Round 1 answers (read Oct 18 18:42Z from PLAN.interactions.json; copy kept at IB/TMP/round1_interactions_20261018.json)

Verbatim free text:
- C3: "Let's shoot for large context. See my free-form feedback."
- C4: "The problem is that at a sufficiently large context size, the 4B reference is likely bad. So the research should include a
  comparison of the two. If 4B degrades too much, I'd use 27B."
- C5: "Core without \"reader lora without AC\" here."
- C6: "Let's think about this more. We likely do not need so many grids if we make the comparisons semi-independent. As in, when
  comparing the AC variants, we only use 1/20th. But when comparing to other approaches, we pick the best AC Variant (almost certainly
  KV transfer), and vary it from (1/10th, 1/20th, 1/40th)?"
- R1: "If the benchmarks strongly requires multiple rounds of search, I'd make an expand-search-expand-search pipeline. But prefers
  benchmarks where single expanded search is largely enough. Ask researcher to look for a wider set of benchmarks than these."
- R3: "Whichever is fastest."
- X1: "I just got an email warning for EC2, so a no go right now. Add one GH200 and one H200 to our budget for now, but, as much as you
  can (not strict) truly try to use them for the shorter jobs like evals, etc."
- X2: "Even 300 will depend on the specific benchmark we choose."
- Free form: "The key problem is going to be how to the get a baseline number for the corpus q/a. They have single-corpus numbers for
  Llama-3-8B AFAIK. Can we reuse those, and switch to LLama-3-8B instead? This paper has a 4B baseline, but it's multi-cartridge, and
  it shorts the context. https://arxiv.org/html/2602.16284v1#S4 I am actually also fine with training on 4 distinct inputs like they
  do, we need to give ourselves some kind of edge there. Also see this: For long-context benchmarks (LongHealth), we use the
  Qwen3-4B-Instruct-2507 variant instead of Qwen3-4B because it has a longer native sequence length. (note: if one research finishes
  early, I may be back early to give a green light for the runs)."

Accepted:
- C1 LongHealth. C2 same model as encoder and reader for every arm (which model is now open: Qwen3-4B-Instruct-2507 or Llama-3-8B,
  see the baseline question). C3 large context (Cartridges-scale). C4 teacher = whichever of the 4B (full corpus in context) and the
  27B holds up at that length; research compares. C5 core arms: full context, no context, AC input rows only, AC input + residual every
  4th layer, AC key/value transfer (no reader-LoRA-without-AC arm here). C6 semi-independent grids: channel ablation at 1/20 only; the
  best channel (expected key/value) at 1/10, 1/20, 1/40 against other approaches. Training on several distinct corpora (as 2602.16284
  does with 4 inputs) is allowed.
- R1 recall gap ≥ 20 and accuracy gap ≥ 5; prefer benchmarks where one expanded search is largely enough; if multi-round is truly
  needed, expand-search-expand-search; survey a wider benchmark set. R2 one query. R3 whichever encoding is fastest. R4 1/10 and 1/20,
  both arms. R5 the 9B's own answers with full text top-K, kept if correct.
- X1 no EC2. CSAIL budget raised by one GH200 and one H200 (now: 3 GH200 + the H200 pair + 1 H200), extra ones preferably for short
  jobs such as evals. X2 default, size per benchmark. X3 out of scope.

## Round 2 — research (started Oct 18 ≈18:50Z)

The blocker is the corpus baseline: a published number we can stand next to. Candidate routes the corpus agent must evaluate:
1. Switch the corpus section to Llama-3(.1)-8B and reuse Cartridges' single-corpus LongHealth numbers (ICL, Cartridge, and its other
   baselines). Needs our harness (AC encoder, key/value transfer, residuals, vLLM rows) to run a Llama-8B pair at ~100k context.
2. Qwen3-4B-Instruct-2507 in the setting of arXiv 2602.16284 (multi-cartridge, 4 distinct inputs, shortened context) and reuse its 4B
   numbers, matching their setup exactly.
3. Run the open-source Cartridges code ourselves on our model and corpus to produce the baseline (cost?).
4. Our own baselines only (full context, no context) and cite Cartridges for context.

## Corpus Q/A MVP — settled in chat (Oct 18 ≈19:00–19:50Z) and the probe

User decisions (verbatim where it matters):
- Model: Llama-3.1-8B(-Instruct) as encoder and reader for every arm ("llama 3 8B seems to have a context length of 8k tokens,
  where 3.1 has 128k"). Cartridges-compatible route D: "a single 120k corpus (chunked at document boundaries) ... we run our own
  self-study ... The MVP is to just have our own numbers, but to make it so that we can just plug their numbers as we see fit".
  Figure 4 (Llama-3B) dropped: "8B. Let's not worry about Figure 4 then."
- x-axis: "#question/answer pairs trained on", not Cartridges' step units. Budgets "8h for ablations, and 16h for the best".
- Teacher/self-study: "Llama-3.1-8B, but with the full right document in context, since it's not that long."
- "Short reconstruction first. Same data for all. Then study."
- "Fully bet on key/value. Don't even expect something else." Only KV at 1/15 for the MVP, "use 2 gpus to speed up time to answer.
  Once the MVP is in and looks good, we add a 1/30th tune if needed, or if someone asks for it." Channel ablation at 1/15 re-uses the
  KV 1/15 run ("not a separate 1/20th"). ORCD's RTX and others "can deal with the lower-priority ablations".
- "On top of the initial training, I'd add an extra 24h just for hillclimbing. Until then, corpus q/a has priority over retrieval."
- Rates: the standing 2e-5 / 2e-5 (encoder side + AC modules / reader LoRA), both phases; constant after warm-up (my default).
- "rather than specific numbers, I'd dedicate some time to a probe. (don't forget to also probe things like whether the answer
  format is being respected well, etc. ... Both for synthesis and training)."

Weights: `unsloth/Llama-3.1-8B-Instruct` (public, ungated copy; the official meta-llama repo is gated and no HF token may leave the
container), downloaded directly on CSAIL and ORCD into the shared HF caches. Record the revision hash.

Runs after the probe:
| Run | Channel | Ratio | GPUs | Priority |
|---|---|---|---|---|
| K15 | key/value transfer | 1/15 | CSAIL H200 pair (2) | first |
| R15 | input rows only | 1/15 | ORCD RTX PRO 6000 (preemptable) | ablation |
| D15 | input rows + residual deltas every 4th layer | 1/15 | ORCD RTX PRO 6000 (preemptable) | ablation |
Evaluations of checkpoints on the extra CSAIL GPUs (1 GH200 + 1 H200) or ORCD mit_normal_gpu. Then +24 h hill-climbing.

Probe (all items must be reported with numbers and examples):
- P1 Harness: Llama-3.1-8B as side and reader (model description, tokenizer, chat template, no thinking), the three channels; CPU unit
  tests with a tiny Llama config; one GPU forward/backward per channel.
- P2 Corpus: LongHealth patients 1–10 from the public data (Cartridges / adamzweiger/compaction copies), per-patient documents at
  record boundaries, Llama token counts, the 200 multiple-choice questions, `mc_eval.py` scoring on Llama outputs.
- P3 Baselines (vLLM): full 10-patient corpus in context, no context, the right patient's document in context; accuracy AND answer
  format (letter parse rate, no-answer count, sample outputs). Sanity: Cartridges' Llama-8B ICL read off Fig. 5 ≈ 0.54.
- P4 Self-study synthesis: ~200 pairs from Llama-8B with the full patient document in context; question quality (about the record,
  answerable, diverse, duplicates, leakage vs the 200 eval questions), answer format and length; whether a share of synthetic
  multiple-choice items in LongHealth's format is needed for the reader to answer eval questions with a letter. Teacher targets =
  Llama-8B base with the full right document + question in context, top-k per answer token (trainer's stored-target path or
  generation log-probs, whichever is exact).
- P5 Training: per channel, short runs: s/update and peak memory with the encoder reading 1, 4 and 10 patients, on H200 and RTX PRO
  6000 (and the 2-GPU pair for K15); loss falls in reconstruction and in self-study; panel answers keep the letter format; checkpoint
  save/load and preempt-resume; no NaNs.
- Outcome decisions (mine, reported): examples per update (8 default), encoder view (target patient + 3 others with ~1 in 8 updates
  showing all 10, vs all 10 always), number of self-study pairs, reconstruction-phase length, multiple-choice share, checkpoint cadence.
