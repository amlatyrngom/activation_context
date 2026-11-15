# Math synthesis (§3.4) rerun: data inventory (2026-11-15)

Purpose: find the data for the clean DeepMath-only rerun of the multi-agent synthesis experiment, and set up the
**teacher test**. In that test, the untuned 9B and the 27B each read the three 9B solver runs in full, as text, and
answer. Read-only survey; nothing was run on a GPU and nothing on CSAIL or ORCD was touched.

Conventions:
- `T/` = `IB/TMP/PAPER_TRAJ_QA/`
- `S3/` = `s3://general-bucket-aln/activation-context-data/`
- `INV/` = `IB/TMP/PRE_REBUTTAL/inventory_math/` (new: small downloads and the outputs of this survey)
- The survey scripts are in `IB/TMP/PRE_REBUTTAL/inventory_math_scripts/`. Their working copies are in the session scratchpad.
- "k" = how many of a problem's three 9B runs are right.
- "Right" is scored the way the paper table scores solvers (`tqa_unified_table_sumres.py`):
  - The answer comes from `solver_result` under the 40k rule: if a run's prompt plus its steps reach 40,000 tokens, it counts as "no answer".
  - That answer is checked with `score_synth1call.score`: math-verify against the gold answer and its aliases.
- Check: on the M1 v2 pool this scoring reproduces the logged `plan_summary.json` k histogram exactly (151 / 82 / 179 / 588).
- The `score` field stored in the rollouts disagrees with this rescoring on 45 of the 3,000 M1 v2 runs, so only the rescored values are used here.

---

## 1. Training-pool DeepMath problems that already have three 9B runs

There are two training pools. They do not overlap, and neither overlaps any evaluation set.

| Pool (sample id) | Local file (text of every run) | Full rollouts (token ids, logprobs) | Problems | Difficulty | k = 0 / 1 / 2 / 3 | Used for |
|---|---|---|---|---|---|---|
| **`tqa_m1_train_v2`**<br>= `tqa_m1_v2_9b_a` + `tqa_m1_v2_9b_b` | `T/m1/v2/slims/tqa_m1_v2_9b_{a,b}/rollouts.jsonl`<br>(37.0 + 34.6 MB, 1,500 runs each, seeds 0–2) | `S3/node_final_20261018/pc-a/SYNC/ROLLOUTS/tqa_m1_v2_9b_a/rollouts.jsonl` (279.6 MB)<br>`…/pc-b/SYNC/ROLLOUTS/tqa_m1_v2_9b_b/rollouts.jsonl` (252.8 MB) | **1,000** | 8.5: 741<br>9: 248<br>9.5: 10<br>10: 1 | 151 / 82 / 179 / 588 | M1 targets (27B syntheses) |
| `tqa_train_solve_v1` (round 1) | `T/node_sync/slim_train_v1/ROLLOUTS/tqa_train_solve_v1/rollouts.jsonl`<br>(156.2 MB; 1,500 MuSiQue + 1,500 DeepMath runs) | Not found on S3: only the slim copy exists | **500** | 6: 164<br>6.5: 55<br>7: 71<br>7.5: 62<br>8: 85<br>8.5: 50<br>9: 13 | 62 / 24 / 58 / 356<br>(≥ 8.5 only: 11 / 3 / 11 / 38) | `synth_sft_v1` (432 DeepMath problems) and `r2_bu` (445), all from this pool |

M1 v2 pool by band (the hard band is where the teacher test should run):

| Band | Problems | k = 0 / 1 / 2 / 3 | Single solver | Any of 3 | Gap |
|---|---|---|---|---|---|
| 8.5 | 741 | 107 / 57 / 127 / 450 | 74.7 % | 85.6 % | 10.9 |
| ≥ 9 | 259 | 44 / 25 / 52 / 138 | 69.9 % | 83.0 % | 13.1 |
| All | 1,000 | 151 / 82 / 179 / 588 | 73.5 % | 84.9 % | 11.4 |

In total, **1,500 training problems** have three 9B runs, and **1,063** of them are at difficulty ≥ 8.5 (1,000 + 63).

**Format.** Each slim row holds:
- `agent_config.{dataset_task, metadata, user_prompt}`. `dataset_task` has `task_id`, `gold_answer`, `gold_answer_aliases` (non-empty in 330 rows across these files) and `reference_metrics_kind: math_verify`.
- `seed`, `score`, `finish_reason`, `answer` (the submitted answer), `prompt_messages`, `prompt_tokens`.
- `trajectory[]`: one entry per step, with `role`, `messages` (full text) and `tokens` (the 9B token count of that step).

Assistant messages carry the reasoning text without the `<think>` tags, plus `tool_calls` (shell / python / submit_answer). Tool messages carry the tool output. The slim format comes from `T/kit/slim_tqa_runs.py`. The full S3 rows also have `token_ids`, `logprobs`, `think_end` and `call_kwargs` per step.

Difficulty is the `task_id` prefix (for example `d8.5-965`). Every pool comes from `deepmath___min_difficulty_6__n_1000000__seed_0`.

**Other sources that are not usable here, or are not training data:**
- `tqa_m1_v2_27b_{a,b}` (`T/m1/v2/slims/`): **27B** solver runs (seeds 3–4) on the 233 M1 v2 problems with k ≤ 1. That is 466 runs, 153 of them right (`plan_summary.json`).
- The round-2 solver samples have no DeepMath:
  - `tqa_train2_solve_v1`: 1,909 problems.
  - `tqa_train1_mus_s34`.
  - The `slim_r2a` slims.
- `slim_p2` and `slim_partial1` are partial MuSiQue copies of `tqa_train_solve_v1`.
- `tqa_m1_train` v1 (1,500 problems at ≥ 8) was drawn but never solved: its queues were killed (`T/m1/QUEUES_V2.md`). 467 of its problems were carried into v2.
- `csail_train_v1` and `tqa_own_collection_v1` are question/answer collections derived from the same runs. They contain no new trajectories (not inspected in detail).
- Not training data:
  - `tqa_reserve_solve_v1m`: 100 DeepMath problems, a reserve evaluation set.
  - `tqa_dev_solve_v1`: 200 DeepMath problems, round-2 dev. Its difficulty mix is 6: 69, 6.5: 16, 7: 27, 7.5: 31, 8: 37, 8.5: 16, 9: 4. It has 3 seeds and exists only on S3: `S3/node_final_20261018/pc-a/SYNC/ROLLOUTS/tqa_dev_solve_v1/rollouts.jsonl` (743.5 MB).

**Overlap checks** (`IB/TMP/PRE_REBUTTAL/inventory_math_scripts/overlap.py`, `qtext.py`):

| Pair | Same task id | Same normalised question text |
|---|---|---|
| M1 v2 vs dmext / dmhard | 0 / 0 | 0 / 0 |
| `train_solve_v1` vs dmext / dmhard | 0 / 0 | 0 / 0 |
| M1 v2 vs `train_solve_v1` | 0 | 0 |
| dmext vs dmhard | 0 | 0 |
| M1 v2 or `train_solve_v1` vs dev-200 / reserve-100 | 0 | not checked |

This agrees with `T/m1/v2/draws_check.json`, which reports 0 overlap for every sample. All 300 dmext, 300 dmhard and 200 dev ids are also in `T/m1/v2/excluded_tasks_m1.json` (1,800 ids).

## 2. Solver runs for the test sets

| Set | Where | Rows | Problems | Difficulty | k = 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|
| dmhard `tqa_dmhard` | `T/node_sync/dmhard/rollouts.jsonl` (151.7 MB, full format; also `half_a`/`half_b`) and `S3/node_final_20261018/pc-a/SYNC/ROLLOUTS/tqa_dmhard/rollouts.jsonl` | 900 (seeds 0–2) | 300 | 8.5: 226<br>9: 72<br>9.5: 2 | 35 / 29 / 67 / 169 |
| dmext `tqa_dmext_solve_v1` | **On S3, not missing**: `S3/node_final_20261018/pc-a/SYNC/ROLLOUTS/tqa_dmext_solve_v1/rollouts.jsonl` (200.5 MB, full format). Report files sit next to it under `…/PAPER_TRAJ_QA/tqa_dmext_solve_v1/`. | 900 (seeds 0–2) | 300 | 6: 103<br>6.5: 34<br>7: 45<br>7.5: 36<br>8: 54<br>8.5: 20<br>9: 7<br>9.5: 1 | not computed |

- The dmext file was streamed and checked, not downloaded. Its task ids equal `dmext/sample.json`, and its finishes are: submitted 865, no_tool_call 23, max_tool_errors 7, trajectory_cap 5.
- The dmhard solver scores from this rescoring reproduce the paper table: single solver 74.4 % and any of 3 88.3 %, which matches the 226/74 weighting of 75.1 / 72.5 and 87.6 / 90.5.

**The synthesis reads are also on S3.** The appendix README says the results-only files are missing; they are on S3:
- **Results-only:**
  - dmhard: `S3/node_final_20261018/pc-b/SYNC/PAPER_TRAJ_QA/tqa_r2dmh_res_s{0..4}/{gen_0.jsonl,synth1call.json}`
  - dmext: `…/pc-a|pc-b/…/tqa_r2dmx_res_s{0..2}/`
- **Results + Summary:**
  - `S3/container_final_20261109/home_claude/jobs/6f1c1373/tmp/sumres/pulled/tree/SYNC/PAPER_TRAJ_QA/tqa_r2dm{h,x}_sumres_s*`
  - Summaries: `…/ROLLOUTS/tqa_sum_dm{h,x}_summaries/`
- **AC and no-rows (Results + LoRA):** `S3/node_final_20261018/pc-b/SYNC/ROLLOUTS/tqa_r2dm{h,x}_{ac,norows}_m1_s*`
- The two `synth1call.json` files are copied to `INV/s3_small/`.

## 3. Token lengths of the three full runs per problem

Measured with the 9B `tokenizer.json`:
- **Per-run text:** each run rendered exactly as the 27B prompts rendered it (`tqa_sft_prep_runs.render_run`): reasoning, `[call …]` lines and `[result]` blocks, with submit_answer withheld and the steps cut at the 40k rule.
- **Full teacher prompt:** the task, the instruction, then for each agent "Agent i / Its run: … / Final answer: …", then the final-answer rule. This is the message content, not counting the few chat-template tokens.
- The 27B (unsloth) tokenizer gives the same counts to within 4 tokens.

| Set | Problems | Three runs combined: median / p90 / p99 / max | Full teacher prompt: median / p90 / p99 / max | Prompts > 64k / > 72k |
|---|---|---|---|---|
| M1 v2 training (≥ 8.5) | 1,000 | 9,040 / 35,490 / 69,022 / 92,399 | 9,312 / 35,730 / 69,294 / **92,756** | 13 / 7 |
| `train_solve_v1` (6–9) | 500 | 15,236 / 43,071 / 77,137 / 101,993 | 15,484 / 43,310 / 77,346 / 102,313 | 12 / 9 |
| dmhard (test) | 300 | 9,308 / 31,952 / 53,780 / 71,086 | 9,575 / 32,216 / 54,040 / 71,365 | 2 / 0 |
| **Proposed teacher set** | 200 | 8,576 / 34,716 / 63,007 / 92,399 | 8,821 / 35,063 / 63,401 / **92,756** | 2 / 1 |

- **One M1 v2 run:** median 2,615 tokens, p90 12,238, max 39,257 (rendered text).
- The raw step-token sums (chat-format tokens of the steps under the 40k rule) are within about 3 % of the rendered counts (M1 v2: median 9,294 vs 9,040).
- Every run is under 40k on its own, so the prompt can never exceed about 3 × 40k.
- Per-problem values are in `INV/per_task_lengths_and_scores.json`; the summary is in `INV/lengths_summary.json`.

## 4. Models, vLLM settings, tokenizer and vocabulary

| Role | Model id | Settings (source) |
|---|---|---|
| Solvers | `Qwen/Qwen3.5-9B` | Agent harness: thinking on, `reasoning_effort` medium, temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5. 8,192 tokens per turn, 100 turns, trajectory cap 40,000, vLLM `max_model_len` 56,000, 40 agents per GPU (`run_tqa_solve_v2.py`, `solver_dmhard.agent_config.json`). |
| Results-only / Summary | `Qwen/Qwen3.5-9B` | Plain vLLM: one user message, no system prompt, thinking on, the same sampling, `max_tokens` 64,000, `max_model_len` 72,000, `kv_cache_dtype` fp8, seed = repeat index (`run_synth1call_v2.py`; `INV/s3_small/*synth1call.json`). |
| AC and no-rows synthesis | `Qwen/Qwen3.5-9B` + M1 reader adapter; encoder `Qwen/Qwen3.5-2B` | Agent harness, `max_turns` 1. The program overrides the per-turn 8,192 with `max_tokens` 64,000; trajectory cap and `max_model_len` 80,000. **The effective cap is 64,000**: a no-rows read reached exactly 64,000 output tokens. AC prompt median 1,481 tokens on dmhard s1. |
| Targets / oracle | **`unsloth/Qwen3.8-27B-NVFP4`** | `tqa_r2_generate.py` (via `m1v2_gen.sh`): vLLM with `max_model_len` 131,072, `max_new_tokens` 32,768 (capped by the room left), fp8 KV cache, prefix caching, `gpu_memory_utilization` 0.9, seed 0. Thinking on with `reasoning_effort` medium and the same sampling. 4 samples per mixed triple and 3 per unanimous triple. `Qwen/Qwen3.8-27B-FP8` is the named alternative on H200/GH200 nodes (`write_tqa_questions.py`); the harness default for both is 64,000 + MTP speculative decoding (`vllm_wrapper.py`). |

**Shared vocabulary: yes.** Checked from the public Hugging Face files, saved in `INV/hf_configs/`:
- **Same vocabulary and merges:**
  - `vocab.json` is byte-identical across Qwen3.5-9B, Qwen3.8-27B, Qwen3.8-27B-FP8 and unsloth NVFP4.
  - `merges.txt` is identical across the three Qwen repos; unsloth ships no `merges.txt`.
  - In `tokenizer.json`: 248,044 base tokens, identical; merges identical, including unsloth's after normalising its format.
- **Same output size:** all four configs say `Qwen3_5ForConditionalGeneration` with `vocab_size` 248,320, so the logit index space is the same.
- **Differences:**
  - The 27B has 7 extra special tokens (ids 248070–248076: audio / TTS markers) that the 9B tokenizer does not define. They never occur in text.
  - The unsloth NVFP4 `tokenizer.json` uses an older pre-tokenizer regex (`\p{L}+` instead of `[\p{L}\p{M}]+`). On these prompts it changes token counts by at most 4 tokens, but tokenisation is not guaranteed identical. **Tokenise with the 9B tokenizer for both models.**
  - **The chat templates differ.** The 27B template injects a reasoning-effort sentence and **defaults to `xhigh`** when no effort is passed. The 9B template has no effort option. Pass `reasoning_effort="medium"` to the 27B, as the old targets did.

**So KL from the 27B is possible in principle**: both models produce distributions over the same 248,320 token ids. Practical caveats:
- Mask or renormalise the 7 extra ids and the padding ids.
- A full-distribution KL needs the 27B's logits on the student's sequence. That means an HF forward pass (or a top-k approximation from vLLM logprobs), not plain vLLM sampling.
- The 9B and 27B do not share hidden sizes (4,096 vs 5,120) or depths (32 vs 64 layers). This rules out any hidden-state teacher, only logits.

## 5. Prompts, parsers and scoring for the arms

All arms use one user message, with no system message and thinking on:
1. The task text, minus its submit_answer sentence (`run_synth1call.TASK_EDITS`).
2. An instruction.
3. One block per agent.
4. The final-answer rule (`FINAL_RULE`): "End your reply with one last line of the form `Final answer: <answer>`…".

| Arm | Script | Instruction (abridged) | Per-agent block |
|---|---|---|---|
| Results-only | `T/kit/run_synth1call_v2.py` (condition `synth_result`) | "Three independent agents attempted this task. Their final answers are below. … No tools are available to you: reason about the task and their answers, then give the single best answer." | `Agent i final answer: <result>` |
| Results + Summary | same (`synth_summary_result`); summaries from `T/kit/gen_summaries_vllm.py` (untuned 9B, one 8,192-token turn, one resample at seed + 1000) | "…Their summaries and final answers are below…" | `Agent i\nSummary of its run:\n<summary>\n\nFinal answer: <result>` |
| Results + AC (M1) | `T/kit/run_synth_ac.py` | "…Their runs (in compressed form) and their final answers are below…" | `Agent i\nIts run:\n<AC part: rows at ratio 0.1, encoder sees the run + "Summarize the useful facts…">\n\nFinal answer: <result>` |
| Results + LoRA (M1 no rows) | `T/kit/run_synth_ac_norows.py` | as AC | the same, with the part removed |
| Old 27B targets | `T/kit/tqa_m1_plan.py` → `tqa_r2_generate.py` → `tqa_r2_keep.py` | "…Each agent's run (its reasoning, searches and search results) and its final answer are below. … the majority can be wrong … check each final answer against the evidence … name the agent and quote the fact briefly…" | `Agent i\nIts run:\n<render_run text>\n\nFinal answer: <result>` |

`<result>` follows the 40k rule: the submitted answer, or "no answer (the run reached its token budget)" or "no answer (the run ended without submitting one)".

**Parsers** (`T/kit/score_synth_lenient.py`):
- **Strict** (`run_synth1call.parse_answer`): the last `Final answer:` line after `</think>`, or else the last `\boxed{}`.
- **Lenient:** strict, else the last-positioned of the `Answer:` / `Final answer:` labels, `<answer>` tags, `\boxed{}`, "the answer is …", or `submit_answer(…)`.
- **Lenient + last line:** adds the last non-empty line as a final fallback.
- The paper table uses **lenient**. A length stop or a missing row counts as unsolved.

**Scoring** (`score_synth1call.score`): `activation.dataset.scoring.math_verify_match` against the gold answer and its aliases. It first tries a normalised string match, then math-verify.

The table producer is `IB/ARTIFACTS/PAPER_TRAJ_QA/scripts/tqa_unified_table_sumres.py`:
- Each problem's score is the mean over its synthesis repeats.
- The majority vote breaks ties at random.
- Repeats: **3 on dmext (seeds 0–2) and 5 on dmhard (seeds 0–4)**. The table file's "separate" layout and `tqa_unified_sumres.json` both say so.

**Existing 27B evidence on the training pool.** The M1 v2 target run is already a partial 27B teacher test. Its outputs are `T/m1/v2/gen/{a,b}/gen_*.jsonl` over `T/m1/v2/plan/triples.jsonl`, rescored here with the lenient parser (`INV/existing_27b_m1v2_completions_scored.json`). Restricted to triples made of the three 9B runs:
- **k = 1 problems (82):** 66.2 % per-problem mean (656 samples).
- **k = 2 problems (179):** 87.0 % (1,432 samples).
- **k = 3 problems (412 of 588):** 98.3 %.

Caveats:
- No k = 0 problem was run.
- The prompt is the old target prompt, with the quoting demand.
- Generation was capped at 32,768 tokens; 2 samples hit the cap.
- Generation length: median 2,751 tokens.

## 6. Proposed teacher test

**Problem set:** `IB/TMP/PRE_REBUTTAL/inventory_math/teacher_set_m1v2_hard200.json`, **200 problems**.
- **How they were picked:** a uniform random draw from the 1,000 M1 v2 training problems (all ≥ 8.5): `random.Random(20261115).sample(sorted(task_ids), 200)`.
- **What is in it:**
  - Difficulty: 8.5: 150, 9: 48, 9.5: 2.
  - k = 0 / 1 / 2 / 3: 36 / 18 / 35 / 111.
- **CPU baselines on these 200:**
  - Single solver: 70.2 %.
  - Any of 3: 82.0 %.
  - The gap is 11.8 points.
- **File contents:** per problem, the gold answer, aliases, the three results, which runs are right, k and prompt length.
- **Run texts:** rebuild them from `T/m1/v2/slims/tqa_m1_v2_9b_{a,b}/rollouts.jsonl` with `render_run` (cap 40,000), seeds 0, 1, 2 in that order.
- **Existing 27B completions** (old prompt) cover 18 of the k = 1 problems, 35 of the k = 2 and 79 of the k = 3. Their lenient scores are 66.7 / 83.9 / 98.3 %.
- **A caveat:** a random draw has only 53 mixed problems (k = 1 or 2), and those carry most of the signal. If the comparison turns out too noisy, the alternative is to take all 261 mixed problems of the pool and reweight to the pool mix. This is not done here.

**Arms** (same 200 problems and the same repeats for each):
- (a) The untuned 9B reading the full runs.
- (b) The 27B reading the full runs.
- (c) Results-only with the untuned 9B, as the reference. No results-only run exists on training problems.
- The majority vote, single solver and any-of-3 come from the solver rows on CPU.
- Use **3 repeats** per problem and arm (one count for every arm, per CLEAN_RECIPES_PLAN §7), at seeds 0–2.

**Prompt to use, for both teachers.** This is the AC-arm layout with the runs as text, so the teacher's view and the AC student's view differ only in how the runs are represented:

```
<task text minus the submit_answer sentence>

Three independent agents attempted this task. Their runs (their reasoning, tool calls and tool results) and their final answers are below. They may be wrong or disagree. No tools are available to you: reason about the task, their runs and their answers, then give the single best answer.

Agent 1
Its run:
<render_run(run of seed 0, cap 40,000)>

Final answer: <solver_result of seed 0>

Agent 2
…

Agent 3
…

End your reply with one last line of the form `Final answer: <answer>`, with the final answer only, in the form the problem asks for: a number, a LaTeX expression (for example \frac{3}{4}, \sqrt{2}, x^2+1, an interval or a set), or Yes/No.
```

- It is one user message with no system prompt and thinking on, sent through plain vLLM as in `run_synth1call_v2.py`.
- Sampling: temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5. `max_tokens` 64,000, the same as the synthesis arms.
- The 27B also needs `reasoning_effort="medium"`.
- Score with the lenient parser and math-verify; report strict alongside.
- The lengths in §3 were measured with a near-identical instruction ("Their runs and their final answers…", about 10 tokens shorter).
- The old target prompt (§5, with the quoting demand) is the alternative if the goal is comparability with the old 27B targets. Its quoting demand only served the old keep filter.

**Context the test needs:**
- **Longest prompt:** 92,756 tokens (p99 63,401; median 8,821), plus the chat template.
- **Generation:** up to 64,000 tokens.
- **Recommended:** `max_model_len` **160,000** for both models. Their configs allow 262,144 positions. This fits every problem with the full 64k generation.
- **Smaller option:** 131,072 with `max_tokens = min(64,000, room)`, as `tqa_r2_generate.py` did. That leaves the longest problem about 38k tokens to generate in.
- The old synthesis setting (72,000) would not fit the 1 problem in this set whose prompt is over 72k, or the 2 over 64k with a full generation.

**27B KL: possible on vocabulary grounds** (§4). Both models share the token ids and the 248,320-wide output layer, and the 27B adds only 7 unused special tokens. A full-distribution KL needs the 27B's logits from an HF forward pass, or a top-k approximation.
