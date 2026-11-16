# Math teacher test (§3.4 clean rerun, DeepMath): result (2026-11-16)

Question: how well does a synthesizer do when it reads the three 9B solver runs **in full, as text**? We compare the
untuned `Qwen/Qwen3.5-9B` and `unsloth/Qwen3.8-27B-NVFP4` with the 9B reading **results only** (the paper's text arm). The
answer picks the teacher for AC training (CLEAN_RECIPES_PLAN §4 M1, §7).

Every number below comes from `math/teacher_test/result.json`, which `teacher_test/score_teacher.py` writes from the
generation files.

## Result

The set is 200 training-pool problems at difficulty ≥ 8.5. Each arm ran 3 sampled repeats per problem (seeds 0, 1, 2, the
same in every arm), so each arm has 600 replies. Accuracy is the mean over problem × repeat, in %. The CI is a paired
bootstrap 95 % CI of the difference to 9B results-only (10,000 resamples of problems). W / L counts problem × repeat pairs
(repeat i against the reference's repeat i), with an exact sign test.

| Arm | Accuracy | Per repeat (s0 / s1 / s2) | Δ vs 9B results-only [95 % CI] | W / L vs 9B results-only (p) | Gap closed | Length stops | Generated tokens: median / p90 / max | Wall-clock |
|---|---|---|---|---|---|---|---|---|
| **9B, results only** (reference; paper prompt) | **77.67** | 77.0 / 76.5 / 79.5 | – | – | 63.4 % | 13 | 19,560 / 43,391 / 64,000 | 10,851 s (181 min) |
| **9B, full runs** | **77.50** | 77.0 / 77.0 / 78.5 | −0.17 [−3.00, +2.50] | 20 / 21 (p 1.0) | 62.0 % | 1 | 3,542 / 8,464 / 64,000 | 2,934 s (49 min) |
| **27B, full runs** | **78.83** | 80.0 / 78.5 / 78.0 | +1.17 [−1.83, +4.17] | 30 / 23 (p 0.41) | 73.2 % | 0 | 2,106 / 12,361 / 46,603 | 3,879 s (65 min) |
| 27B, results only (lowest priority) | 78.00 | 77.0 / 78.5 / 78.5 | +0.33 [−2.33, +3.17] | 30 / 28 (p 0.90) | 66.2 % | 4 | 2,995 / 18,421 / 64,000 | 3,182 s (53 min) |

"Gap closed" = (arm − single solver) / (any of 3 − single solver).

Solver references on the same 200 problems (from the solver runs, no generation):

| Reference | All 200 | Mixed problems (k = 1 or 2, n = 53) |
|---|---|---|
| Single solver (mean of the three 9B runs) | 70.17 | 55.35 |
| Majority vote (random tie-break, expected value) | 74.33 | 71.07 |
| Any of 3 | 82.00 | 100.00 |

Strict parsing gives the same accuracy as lenient for three arms. 27B results-only is the exception: 77.67 strict vs 78.00
lenient. Every arm beats the majority vote in the paired count:
- 9B results-only: +44 / −25 (p 0.030)
- 9B full: +36 / −18 (p 0.020)
- 27B full: +50 / −24 (p 0.003)
- 27B results-only: +52 / −31 (p 0.028)

### Breakdown by k (number of the three 9B solvers that were right)

Accuracy in %. W / L is against 9B results-only (problem × repeat pairs).

| k (problems) | 9B results-only | 9B full | W / L | 27B full | W / L | 27B results-only | W / L |
|---|---|---|---|---|---|---|---|
| 0 (36) | 11.1 | 9.3 | 4 / 6 | 17.6 | 7 / 0 | 19.4 | 9 / 0 |
| **1 (18)** | **68.5** | **57.4** | **2 / 8** | 64.8 | 6 / 8 | 63.0 | 7 / 10 |
| **2 (35)** | **87.6** | **86.7** | 6 / 7 | 85.7 | 10 / 12 | 87.6 | 6 / 6 |
| 3 (111) | 97.6 | 100.0 | 8 / 0 | 98.8 | 7 / 3 | 96.4 | 8 / 12 |
| **Mixed, k = 1–2 (53)** | **81.13** | **76.73** | Δ −4.40 [−10.69, +1.89] | **78.62** | Δ −2.52 [−11.32, +6.29] | 79.25 | Δ −1.89 [−9.43, +5.66] |

Length stops by k:
- 9B results-only (13): k=0: 5, k=2: 3, k=3: 5. These account for its k = 3 losses.
- 9B full (1): k=0: 1.
- 27B results-only (4): k=0: 2, k=1: 1, k=3: 1.
- 27B full: none.

## What it says

1. **Reading the full runs does not make the 9B a better synthesizer.** Overall it is 77.50 vs 77.67, CI [−3.0, +2.5].
   - On the mixed problems, where the signal is, it is 4.4 points *below* results-only (CI [−10.7, +1.9]).
   - At k = 1 it is 57.4 vs 68.5, with 2 wins / 8 losses.
   - With the runs in front of it, the 9B thinks about 5× less: median 3.5k vs 19.6k generated tokens.
   - On the 54 k = 1 replies, its wrong answers are the (wrong) majority answer 11 times and another answer 12 times.
     Results-only: 8 and 9 (computed ad hoc from the generation files).
   - Its only net gain over results-only is at k = 3 (8 / 0). Five of those 8 are length stops of results-only.
2. **The 27B with full runs is the best arm, but not separably.** It scores 78.83, +1.17 over 9B results-only
   (CI [−1.8, +4.2]; 30 / 23, p 0.41).
   - Its net edge comes from k = 0, problems that no solver got right (17.6 vs 11.1; 7 / 0). Some also comes from k = 3
     (7 / 3).
   - On the mixed problems it is 2.5 points below 9B results-only (CI [−11.3, +6.3]).
   - 27B full is not worse than 27B results-only: 78.83 vs 78.00 overall, 78.62 vs 79.25 on mixed problems. Its replies are
     the shortest (median 2.1k tokens) and it never hit the length cap.
3. **No arm is clearly above 9B results-only.** Every arm closes 62–73 % of the single → any-of-3 gap. Every difference to
   the reference has a CI that straddles 0.
   - Under the plan's M1 rule ("if [the 9B full-run reader] is clearly above results-only, use KL to it; if not, distill the
     27B syntheses kept when correct"), the 9B full-run teacher fails.
   - The 27B is not clearly better either; the decision is the user's.
4. **The old 27B targets agree.** The earlier 27B completions (old target prompt) on a subset of these problems scored
   66.7 / 83.9 / 98.3 % at k = 1 / 2 / 3 (DATA_INVENTORY §6). Here, 27B full scores 64.8 / 85.7 / 98.8.
5. **Cost.** 9B results-only is by far the most expensive arm: mean 23.1k generated tokens and 181 min on one GPU. The 9B
   full arm took 49 min, the 27B full arm 65 min and the 27B results-only arm 53 min (each excluding engine load).

Noise caveat: only 53 of the 200 problems are mixed. The mixed-problem CIs are about ±8–9 points wide, and the per-k cells
(18 and 35 problems × 3 repeats) are small.

## Exact settings

**Problems.** `IB/TMP/PRE_REBUTTAL/inventory_math/teacher_set_m1v2_hard200.json`:
- 200 problems drawn with `random.Random(20261115).sample` from the 1,000 M1 v2 training problems.
- Difficulty: 8.5: 150, 9: 48, 9.5: 2.
- k = 0 / 1 / 2 / 3: 36 / 18 / 35 / 111.
- No overlap with dmext or dmhard (DATA_INVENTORY §1).

**Runs.** From `IB/TMP/PAPER_TRAJ_QA/m1/v2/slims/tqa_m1_v2_9b_{a,b}/rollouts.jsonl`, seeds 0, 1, 2 as Agents 1, 2, 3.
- Rendered with `tqa_sft_prep_runs.render_run` at cap 40,000.
- Each run's result follows the 40k rule (`solver_result`). The results equal the inventory's for all 200 problems (asserted
  in `build_prompts.py`).

**Prompts.** One user message, no system message, thinking on.
- **Full-run arms** (DATA_INVENTORY §6), in this order:
  - the task text minus its submit_answer sentence (`run_synth1call.TASK_EDITS`);
  - the instruction "Three independent agents attempted this task. Their runs (their reasoning, tool calls and tool results)
    and their final answers are below. They may be wrong or disagree. No tools are available to you: reason about the task,
    their runs and their answers, then give the single best answer.";
  - per agent `Agent i\nIts run:\n<render_run>\n\nFinal answer: <result>`;
  - `FINAL_RULE["deepmath"]`.
- **Results-only arms:** `run_synth1call_v2.build_prompt(..., "synth_result", ...)`, the paper's prompt unchanged.
- **Prompt lengths** (with the chat template):
  - full runs: median 8,906 tokens, max 92,776;
  - results only: max 748 tokens.
- **Prompt file:** `IB/TMP/PRE_REBUTTAL/math_teacher_test/prompts.jsonl` (sha256 `150e0d44f2e12ed5…`; also on S3 `data/`).

**Chat template and tokenizer.**
- The chat text comes from the 9B template with `enable_thinking=True`.
- The 27B template with `enable_thinking=True, reasoning_effort="medium"` renders the identical string for all 400 prompts
  (checked; the "medium" setting adds no system sentence, unlike its default "xhigh").
- Both models receive the same prompt token ids, tokenized with the 9B tokenizer. The 27B decodes with its own tokenizer;
  the vocabulary is identical.

**Engine.** vLLM 0.28.0 on one NVIDIA RTX PRO 6000 Blackwell Server Edition (EC2 spot g7e.2xlarge, SkyPilot cluster
`mteach`, ap-south-1b).
- `LLM(model, max_model_len=160_000, kv_cache_dtype="auto"` (bf16), `gpu_memory_utilization=0.9,
  limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)`. No speculative decoding.
- KV cache: 2,058,322 tokens (9B) and 1,833,394 tokens (27B).
- All 600 requests of an arm were queued at once; the arms ran one after another (9B full → 9B results; 27B full → 27B
  results).

**Sampling.** As the paper's synthesis arms (`run_synth1call_v2.SAMPLING`): temperature 1.0, top-p 0.95, top-k 20, min-p 0,
presence penalty 1.5. Also `max_tokens` 64,000 and `SamplingParams(seed=repeat)` with repeats 0, 1, 2. The largest
prompt + generation was 127,400 tokens, which fits the 160k context.

**Scoring.** As the paper table (`tqa_unified_table_sumres.py`):
- The answer is the lenient reading of `score_synth_lenient.readings("deepmath", text)`.
- A length stop counts unsolved.
- The answer is right when `score_synth1call.score` (normalised match, then math-verify, against the gold answer and its
  aliases) is ≥ 1.
- The majority vote is `score_synth_lenient.majority_answer` with a uniformly random tie-break. Its accuracy is the expected
  value; its paired tests use one draw seeded by task id.
- No row is missing in any arm. Unparsed replies: 27B full 1, 27B results-only 2.

**Spot / resume.** Every reply was appended to `gen_<arm>.jsonl` on the node and mirrored to S3 every 5 minutes. No reclaim
happened.

One failure: the first 27B start died before generating. Its NVFP4 JIT compile ran 9 parallel `cicc` processes (~5.4 GB
each) and filled the 62 GB host RAM, so Ray killed the job. `node_job.sh` now sets `MAX_JOBS=3 FLASHINFER_NVCC_THREADS=1`,
and the rerun went through. The 27B engine load took 448 s (9B: 313 s).

## Files

- Scripts (`IB/ARTIFACTS/PRE_REBUTTAL/math/teacher_test/`):
  - `build_prompts.py`: CPU; builds the prompts.
  - `teacher_gen.py`: node; generation.
  - `node_job.sh`: node wrapper with the S3 restore / mirror.
  - `tt_s3.py`: boto3 mirror.
  - `launch.sh`: `sky exec` bootstrap; IB/ is not in the SkyPilot mirror, so the job files travel through S3 `jobs/`.
  - `score_teacher.py`: CPU; scoring.
  - `result.json`: all numbers above.
- Generations: `s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/math/teacher_test/out/`
  (`gen_{9b_full,9b_results,27b_full,27b_results}.jsonl`, `timing.jsonl`). A local copy is in
  `IB/TMP/PRE_REBUTTAL/math_teacher_test/out/`.
- Logs: `IB/TMP/PRE_REBUTTAL/math_teacher_test/exec_{9b,27b,27b_oom1}.log`.
- Node `mteach` was torn down on 2026-11-16 at about 04:35Z.
