# Paper — trajectory question answering (first look)

Source plan. Projection: `PLAN.tressoir.md`. Written Oct 14 00:12Z while the user is away (their note: "I'd also start looking into traj Q/A now, as we'll need it later if there are usable gpu slots. I am logging off for likely 8-12h, so don't block on me."). Intent level; nothing here is agreed yet.

## Origin and reading

- Paper target: `IB/evaluation_section_snapshot.tex:58-64` (Multi-Agent Collaboration): a synthesizer sees each subagent's trajectory through vectors; variant 1 IMO 2024/25 with DeepMath-103k training, variant 2 Musique; todo at :62 wants all@3 / any@3 / mean@3 on math and search.
- My reading of "traj Q/A": the existing task where the reader answers a question about a trajectory from AC rows of it (item kind `traj_qa`). It is the natural training task for a synthesizer that queries subagent trajectories. Alternative readings kept open as a decision (`tqa.role.*`).

## Evidence (read-only research pass by an Opus agent, Oct 14 00:05Z; not re-verified by me unless marked V)

FT = `IB/ARTIFACTS/FIRST_TASK_AWARE_FINETUNING`.

- Canon: `encoder_sees_question` for RAG and trajectory QA; reader always gets the question, encoder never the answer (`IB/CANON/ACTIVATION_CONTEXT_CANON.md:62`); defaults RAG true, trajectory false (`FT/CHANNEL_OPERATIONS_20260924.md:73`); held out by question not by source (`:58, :66`); QA NLL is not accuracy, decoded-answer panels required (`:48`); reconstruction-only pair is never a QA comparison (`:89`); subagent exchange ratio 1/20 (`:74`); `TRAJECTORY_QA_CANONICAL_PROPOSAL.md` unaccepted.
- Format: system "You answer questions about agent trajectories…", user `Task: {question}`, `Trajectory N:` parts, instructions + question; target a `submit_answer` call (`activation/ac_model/ac_model_study.py:26-27, 642-684`).
- Code: `trajectory_qa_item` (`ac_model_study.py:627`), `transform_for_eval_reference` (`:1171`), `ITEM_KINDS` and `full_context_match_kinds` default `("rag_qa",)` (`ac_model_training.py:54, :120`); `FT/bench/prepare_trajectory_material.py` (public traces -> question-independent 8k/16k/32k windows), `generate_trajectory_questions.py` (Qwen3.5-27B writes and checks questions), `run_trajectory_tasks.py --task trajectory_qa|compaction`, `evaluate_trajectory_cohort.py`, `run_channel_mixture.py` (`QA_FAMILIES=('rag_qa','trajectory_qa')`). Launch examples: `IB/TMP/FIRST_TASK_AWARE_FINETUNING/trajectory_qa_16k_question_aware_launch_v1/{prepare,train_h200}.sbatch`.
- Data: question bank `IB/TMP/SYNC/REUSABLE_DATA/trajectory_qa/` (6.2 GB, 112,007 rows: train 100,564, held 11,443; coding 65,017, research 20,523, math 26,467); trajectories are public traces (Open-SWE minisweagent, S1-DeepResearch, Nemotron-Math-v4), questions by Qwen3.5-27B; no Musique, DeepMath or IMO trajectories. Copies: S3 snapshot `snapshot_20260926T125357Z`, CSAIL `$R/data/trajectory_qa` ($R probably `recon_100m_v1`), ORCD `trajectory_compaction/material_v3`. Our own agents' trajectories: `IB/TMP/SYNC/REUSABLE_DATA/agent_teacher_regular/` (27B; v2 export 11,313 segments, 791 subagent segments; musique, deepmath, dapo_math and others); older campaign 7,446 subagent segments.
- Saved pairs: `IB/TMP/FIRST_TASK_AWARE_FINETUNING/important_pairs/trajectory_qa_{blind_7200,aware_8000,aware_16000}_20260925/`.
- Earlier results (2B -> 9B, 1/16, SFT, 100 held questions from 30 roots; held answer NLL | judged correct): frozen full history 0.299 | 96; frozen no history 1.772 | 26; starting RAG pair with AC 0.945 | 40; after 2k blind questions 0.908 | 41; after 1,848 compaction histories 0.880 | 48; blind 8k best at 7,200: 0.684 | not judged; question-aware 16k: 0.308 | not judged. Doubling rows to 1/8 at fixed weights: no gain. No KL run on trajectory QA.
- Multi-agent pieces: `Agent.run_subagent`, `to_activation_context`, `SubagentTool` returning a `subagent_return` part (`activation/agent/agent.py:531, :232`; `agent_tools.py:456`), ratio `ac_subagent_ratio` 1/20; only existing program `SolveThenVerifyProgram` (a test); `ParallelSolveProgram` dropped; `perform_grouped_rollouts(group_count)` runs a config under several seeds (`rollout_manager.py:73`).
- Not found: any all@k / any@k / mean@k measurement; a three-solver + synthesizer program; an IMO loader (only `aime.py`); DeepMath–IMO decontamination; synthesizer training data; any pair trained on `subagent_return` parts.

## Decision keys published in the projection (never reuse)

`tqa.role.*`, `tqa.question.*`, `tqa.source.*`, `tqa.away.*`.

## Milestones (intent)

- T0 — today's state: does the trajectory-QA pipeline run in the cluster tree with a rows-only start; all@3 / any@3 / mean@3 for the untuned 9B on Musique and math (three seeds), when a 9B window is free on an EC2 node.
- T1 — first rows-only trajectory-QA pair on the existing bank, blind, sizes 1/20 to 1/5, decoded-answer cohort.
- T2 — questions over our own agents' trajectories (math, search).
- T3 — three solvers + synthesizer program and its evaluation.

Compaction keeps priority on every GPU slot.

## T1 first run (Oct 14 00:46Z; built by a background fork, logged 00:58Z)

`/data/scratch/ngom/activation/paper_traj_qa_v1/`: wrappers `build_traj_packet.py`, `check_traj_packet.py` (the tree is not edited); packet `packets/tqa_blind_ro_v1` (derived from clean `tqa_blind_src_v1`, 8,400 items): 8,000 `traj_qa` items, all `trajectory_qa/uncontextual`, sizes log-uniform 1/20–1/5 (min 0.05, median 0.1002, max 0.2; drawn by the wrapper, the tree samples uniform 1/20–1/10), 1,000 updates, cosine, 50 warm-up, panel every 10 %, `deltas_off_training=true`, 2e-5 | 2e-5, snapshots 50 %. Held 64 questions (the mixed run's panel) in groups 1/16, r20, r10, r5. Job 2653843 on `quanta-gh200-3`. Start readings: no context 0.6653; full 0.2782; r20 0.5255; 1/16 0.5230; r10 0.4597; r5 0.4507. Build job 2653794 (CPU, 4 min 49 s).


## T1 result (Oct 14 07:52Z) — status Review

- Run `packets/tqa_blind_ro_v1` (job 2653843) done; held loss 0.526 | 0.523 | 0.460 | 0.451 → 0.253 | 0.226 | 0.186 | 0.162 (1/20 | 1/16 | 1/10 | 1/5). Pair `pairs/tqa_blind_ro_final`.
- Judged correct of 64 (blind Sonnet, 5 judges, 368 distinct answers; `IB/TMP/PAPER_TRAJ_QA/judge/keys/t1/scores_t1.json`): untuned full text 51 (9 not submitted); nothing 16; starting pair 18 | 18 | 19 | 17; trained pair 37 | 40 | 49 | 51 (1/20 | 1/16 | 1/10 | 1/5).
- On the 52 held questions whose trajectory no training question used: full text 41, nothing 11, trained 1/5 43, 1/10 40, 1/20 29 (`checks/held_overlap.py`, job 2660594: 12 of 64 share a trajectory with training).
- McNemar (64 paired): trained 1/5 vs full text 8 | 8 (p = 1.0); 1/10 vs full 7 | 9 (p = 0.80); 1/20 vs full 6 | 20 (p = 0.009); trained 1/5 vs start 1/5 35 | 1 (p ≈ 1e-9).

## 40k cap (Oct 14 09:17Z)

- User (verbatim): "In case I haven't said it: limit traj q/a runs and agent collaboration runs to 40k without compaction. Otherwise, compaction and agent collab are interfering experiment-wise. Unless there is already a reasonable limit?"
- Existing limit: the @3 baselines (`run_at3.py`) use a 62k cap, compaction off. Re-scored exactly at 40k (`score_at3_cap.py`): the harness cap ends the run once the prefix reaches it, without telling the agent. Seed 0: 21 of 100 Musique runs reach 40k.
- User (≈09:20Z, verbatim): "For traj q/a before selecting the benchmark, wait for me to plan a few more things." The @3 baselines already running finish as measurements; nothing else starts.

## Untuned 9B @3 baselines (Oct 14 09:43Z, measurements only; benchmark not selected)

Untuned Qwen 3.5 9B, single agent, thinking on, compaction off, 3 seeds, 100 Musique validation (answerable) + 100 DeepMath (difficulty ≥ 6) tasks, sample seed 20261014; runs made under a 62k cap, re-scored exactly at 40k (`score_at3_cap.py`). Solved = score ≥ threshold (Musique: token F1; DeepMath: math-verify 0/1).

| Benchmark | Cap | Solved at | all@3 | any@3 | mean@3 | Runs reaching the cap |
|---|---|---|---|---|---|---|
| Musique | 40k | F1 = 1.0 | 0.26 | 0.43 | 0.34 | 63 of 300 |
| Musique | 40k | F1 ≥ 0.5 | 0.34 | 0.54 | 0.43 | 63 of 300 |
| Musique | 62k | F1 = 1.0 | 0.29 | 0.48 | 0.37 | 1 of 300 |
| DeepMath | 40k | 1 | 0.56 | 0.87 | 0.73 | 3 of 300 |
| DeepMath | 62k | 1 | 0.57 | 0.87 | 0.73 | 0 |

DeepMath: 15 of 300 runs ended without a tool call (no answer). Its 100 tasks were not checked against earlier campaigns or the question bank.

## Synthesis run (Oct 14, requested ≈10:00Z)

- User (verbatim): "Ok we need one more run on both deepmath and musique. One more run to make: synthesis numbers. Give each agent result-only, and summary+result of the 3 runs and allow them 64k tokens to synthesize an answer (we might truncate down to 40k; it's fine, but synthesis may need to look at a few different things). Record both the continuation tokens vs original 3, and the solve rates."
- Design as built: solvers = the three untuned-9B @3 runs per task (100 Musique + 100 DeepMath), cut by the 40k rule (a run reaching 40k before answering gives "no answer"); summaries written by the 9B over its own run (harness compaction demand, trajectory up to 40k); synthesizer = untuned 9B agent with the solvers' tools, 64k cap, compaction off, one run per task per condition (results only; summaries + results). Recorded: solve rates (also re-scored at 40k), synthesizer tokens against the three solvers' tokens, summary-writer tokens, and the solvers' majority vote / any@3 / mean@3 for reference. Runs on `pc-b` (free again by 12:00Z for the compaction hill-climb).

### Synthesis result (Oct 14, ended 10:45Z; built and run by a background fork on `pc-b`)

All 200 tasks ran; every condition paired over the same 100 tasks per benchmark; solvers scored at 40k; one synthesizer run per task per condition.

| | Musique F1 = 1 | Musique F1 ≥ 0.5 | DeepMath |
|---|---|---|---|
| Solvers all@3 / any@3 / mean@3 | 0.26 / 0.43 / 0.34 | 0.34 / 0.54 / 0.43 | 0.56 / 0.87 / 0.73 |
| Solvers majority@3 | 0.33 | 0.39 | 0.73 |
| Synthesizer, results only (64k / re-scored 40k) | 0.40 / 0.39 | 0.49 / 0.48 | 0.79 / 0.79 |
| Synthesizer, summaries + results (64k / 40k) | 0.37 / 0.37 | 0.47 / 0.47 | 0.82 / 0.82 |

- Against majority vote (tasks flipped wrong→right vs right→wrong): results only Musique 7–0 (sign test p ≈ 0.016), DeepMath 8–2 (p ≈ 0.11); summaries + results Musique 5–1 (p ≈ 0.22), DeepMath 10–1 (p ≈ 0.012). Summaries + results vs results only: Musique 0–3, DeepMath 7–4 (noise).
- The synthesizer almost never solves a task no solver solved: 1 of 57 such Musique tasks, 1 of 13 DeepMath.

Tokens (mean per task; solver runs counted up to 40k):

| | Musique | DeepMath |
|---|---|---|
| Three solver runs, steps (prompt + steps) | 62,940 (65,520) | 23,303 (25,743) |
| Synthesizer results only: steps / prompt / median steps | 15,066 / 930 / 10,563 | 5,651 / 880 / 4,021 |
| Synthesizer summaries + results: steps / prompt / median steps | 10,475 / 1,834 / 5,907 | 3,699 / 1,858 / 1,390 |
| Writing the 3 summaries (steps) | 1,483 | 2,357 |
| Summary length | 293 tokens | 320 tokens |

Results only costs 24 % of the three runs' steps on both benchmarks; summaries + results (with summary writing) ≈ 19 % on Musique, ≈ 26 % on DeepMath. Synthesizer runs reaching 40k: Musique 9 (results only) and 5 (summaries + results); DeepMath none.

Caveats: 3 of 600 summaries not written (plain text without the compact tool; shown as "(no summary was written)"); the summary writer's system prompt is ≈290 tokens longer (compact tool listed); Musique majority uses normalized string match, so differently worded correct answers don't agree; synthesizer failures (1 at the 64k cap, 1 tool errors, 3 no tool call) count unsolved. Files: `IB/TMP/PAPER_TRAJ_QA/kit/{run_synth.py, score_synth.py, synth_node.sh, synth_v1_score.json}`; pc-b caches `ROLLOUTS/tqa_synth_v1_{summaries,result,summary_result}`, notes in `SYNC/PAPER_TRAJ_QA/tqa_synth_v1/` (prompts in `synth.json`).

### Single-call synthesis (requested Oct 14 ≈11:35Z)

- User (verbatim): "Ok since we have time until I sit back for planning, try the synthesis run again, but non-agentic/single-call only. That might or might not be the baseline we end up picking for clean comparison."
- Design: same 200 tasks, same three solver runs cut at 40k, same two conditions; the summaries of the agentic run are reused; the synthesizer is one chat completion by the untuned 9B (thinking on, no tools, 64k generated tokens, also re-scored at 40k), answer in a fixed final line. Same scorers. Tag `tqa_synth1call_v1` on `pc-b` (free by 12:20Z for the compaction dev reading).
- Result (ended ≈12:06Z; 400 single calls, none cut). Same tasks and scorers; one call per task per condition.

| | Musique F1 = 1 | Musique F1 ≥ 0.5 | DeepMath |
|---|---|---|---|
| Solvers any@3 / mean@3 / majority@3 | 0.43 / 0.34 / 0.33 | 0.54 / 0.43 / 0.39 | 0.87 / 0.73 / 0.73 |
| Single call, results only (64k / 40k) | 0.38 / 0.38 | 0.47 / 0.47 | 0.81 / 0.80 |
| — flips vs majority (wins : losses) | 7 : 2 (p 0.18) | 10 : 2 (p 0.039) | 10 : 2 (p 0.039) |
| Single call, summaries + results (64k / 40k) | 0.40 / 0.40 | 0.51 / 0.51 | 0.83 / 0.83 |
| — flips vs majority | 7 : 0 (p 0.016) | 12 : 0 (p < 0.001) | 11 : 1 (p 0.006) |
| Agentic synthesizer, for reference (results / summaries + results) | 0.40 / 0.37 | 0.49 / 0.47 | 0.79 / 0.82 |

  Single call vs agentic: no paired difference (all p ≥ 0.125). Rescues of tasks no solver solved: Musique 0 and 1 of 57; DeepMath 2 and 2 of 13.

| Tokens (mean per task) | Musique | DeepMath |
|---|---|---|
| Three solver runs' steps | 62,940 | 23,303 |
| Single call, results only: generated (median) / prompt; share | 5,360 (5,268) / 161; 8.5 % | 17,957 (16,391) / 239; 77 % |
| Single call, summaries + results: generated (median) / prompt; share (with summary writing) | 2,224 (1,579) / 1,068; 3.5 % (5.9 %) | 7,187 (5,831) / 1,220; 31 % (41 %) |
| Agentic share, results / summaries + results | 24 % / 19 % | 24 % / 26 % |

  On DeepMath without summaries the single call mostly re-solves the problem in its thinking (≈ one more solver run). Parsing: 399 of 400 ended with `Final answer:`; 1 DeepMath reply hit 64k (unsolved); 4 DeepMath results-only replies pass 40k. Caveats: on Musique the single call cannot search the corpus; no system prompt or tool list (≈750 fewer prompt tokens); one seed; no separate GPU smoke (CPU dry run plus the first 12 outputs inspected). Files: `IB/TMP/PAPER_TRAJ_QA/kit/{run_synth1call.py, score_synth1call.py, synth1call_v1_score.json}`; pc-b `SYNC/PAPER_TRAJ_QA/tqa_synth1call_v1/` (`gen_*.jsonl` full replies, `synth1call.json` prompts).

## Your answers (read Oct 14 12:50Z from PLAN.interactions.json; keys retired)

- Role: synthesizer. "I thinking of considering it a starting checkpoint/first try. If it's inferior to summaries, then we  have a dedicated training pass does SFT on some valid 27B summaries + single-call completions."
- Question: blind. "Here it should be question-blind, since we just want general summaries. Make it question-aware only if we see a good reason for this (e.g., RAG later)."
- Source: bank (first).
- Away: "What items?" → the two items were the T1 rows-only run on the bank and the untuned-9B @3 measurement; both are done (T1 judged, @3 tables above). Nothing for trajectory Q/A runs now.
- Free form (verbatim): "A few more things:
  - we'll do single-call training.
  - Traj Q/A training should try to include multi-traj questions where the oracle sees the golden traj.
  - There might be a special training pass just for synthesis using some dataset-specific 27B summaries + completions (summaries because I doubt 3 full contexts would fit on gpu).
  - Let's train both musique and deepmath (most of the training should be largely agnostic anyway, aside from the potential last-mile-step above; make sure the training set has non-eval entries from both, but keep an extra bank of potential eval questions in case we need 200 samples instead of 100)."

## Defaults taken from the answers (no question needed)

- Synthesizer = one call, no tools, blind AC rows of each solver trajectory; baselines = the single-call results-only and summaries + results runs above (same 100 + 100 tasks).
- Evaluation tasks: the current 100 Musique validation + 100 DeepMath tasks, plus a reserve of 100 more per benchmark drawn now with the same rules and never trained on. Training tasks: Musique training split; DeepMath problems disjoint from both evaluation sets (exact and normalized text match).
- Solver runs for training follow the 40k rule (compaction off).
- Order: (1) own-trajectory Q/A training (single- and multi-trajectory questions, Musique + DeepMath, mixed with the bank) from the T1 pair; (2) synthesis reading against summaries; (3) the synthesis SFT pass only if (2) is below summaries.

## Open before the green light (asked in the projection, keys `tqb.*`)

1. Multi-trajectory questions: what "the oracle sees the golden trajectory" means.
2. What the AC synthesizer sees besides the rows (final answers or not).
3. Rows per trajectory at evaluation.
4. Who writes the training solver runs (9B like evaluation, or 27B).
5. The synthesis SFT target's form (thinking kept or not; filter to correct answers).
6. Training volume per benchmark.

## Your answers, second round (read Oct 14 13:06Z; keys tqb.* retired)

- Oracle (verbatim): "I was thinking it terms of KL. My bad. The reader should just the question + AC rows, and maybe final answer as below." → the "oracle" is the KL teacher: base 9B reading the full trajectory text(s); the student reads question + AC rows (+ the solvers' final answers).
- AC synthesizer input: rows + final answers.
- Size: "1/10th until its hillclimbing is "good enough", then eval all three variants since that's what the paper needs."
- SFT target: correct completions, thinking kept.
- Solvers (verbatim): "(this is for the AC inputs of the SFT pass run? Do 9B at first, since that's what eval will see, but train completions on 27B I think.)" → yes: solver runs (encoded as AC inputs, also carrying the Q/A questions) by the untuned 9B; completions (targets) by the 27B.
- Volume: 500 per benchmark first. Their question: "were 500 enough there?" → T1 trained on 8,400 bank questions over 7,477 distinct trajectories (overlap check, job 2660594), about one question per trajectory. 500 tasks × 3 seeds = 1,500 trajectories per benchmark, so fewer trajectories than T1 but several questions each; the bank stays in the mix for trajectory variety.
- Also (chat, 13:06Z): compaction results "just good enough for our MVP"; hill-climbing continues on the side. On green light, if GPUs allow, prepare the last-mile synthesis SFT in parallel.

## Defaults for the green light (no question left open)

- Q/A loss: KL from the base 9B reading the full text (the H2 path that now runs), for single- and multi-trajectory questions; the bank keeps its existing answers. Multi-trajectory questions over 2–3 runs of the same task whose full texts together stay under the teacher limit tried in H2 (views up to ≈53k tokens ran).
- Questions over own runs written by the 27B with the bank's question-writing pipeline.
- SFT pass material: 27B summaries of each 9B solver run, then a 27B single call over the three summaries + final answers, kept when correct; two 27B samples per task to offset the filter (Musique solves ≈0.4).
- Evaluation: the current 100 + 100 tasks at 1/10 rows per run; reserve 100 per benchmark drawn now; all three sizes once hill-climbing is good enough.
- Nodes: trajectory Q/A takes the EC2 nodes first; compaction keeps one CSAIL GPU on the side.
- Correction (Oct 14 ≈13:15Z, after the user asked): the default is **SFT**, not KL. T1 trained with `loss_kind: sft` (recipe checked) on 27B-written answers, as the September runs did. The "oracle" is the 27B question/answer writer that reads the full trajectory text(s); in multi-trajectory questions it reads all of them. KL (base 9B teacher on full text) stays a possible later variant, only if compaction's H2 shows a gain. The "KL" line in the defaults above is superseded.

## First try: AC synthesizer with the T1 pair (Oct 14 13:37–14:02Z, pc-b GPU 0; built and run by the eval fork)

Single call; each of the three solver runs (40k rule) encoded question-blind at 1/10 (rows only, T1's blind view), plus each solver's final answer as text. Same tasks, scorers, sampling as the text single-call runs.

| | Musique F1 = 1 | Musique F1 ≥ 0.5 | DeepMath |
|---|---|---|---|
| Solvers any@3 / mean@3 / majority@3 | 0.43 / 0.34 / 0.33 | 0.54 / 0.43 / 0.39 | 0.87 / 0.73 / 0.73 |
| **AC rows + answers, single call (T1 pair)** | **0.36** | **0.45** | **0.80** |
| — strict parsing only (`Final answer:` / boxed) | 0.31 | 0.36 | 0.51 |
| vs majority (wins : losses) | 4 : 1 (p 0.38) | 8 : 2 (p 0.11) | 13 : 6 (p 0.17) |
| vs single-call summaries + results (0.40 / 0.51 / 0.83) | 0 : 4 (p 0.125) | 2 : 8 (p 0.11) | 3 : 6 (p 0.51) |
| vs single-call results only (0.38 / 0.47 / 0.81) | 3 : 5 | 5 : 7 | 6 : 7 |

Tokens: rows in the prompt 6,540 (Musique) / 2,575 (DeepMath) from 65,390 / 25,732 encoder tokens; generated 1,823 / 6,959 mean (2.9 % / 30 % of the three solver runs' steps).

- Untrained for synthesis, the T1 pair sits slightly below summaries + results (no difference significant; Musique consistently behind: 0 : 4 at F1 = 1).
- Format: T1's reader adapter learned bare answers without thinking; 17 Musique and 2 DeepMath replies never closed their thinking, and 26 / 40 needed a last-line fallback (spot check: right in 10 of 10). Strict-parse rates are much lower, so part of the gap is format. Own-run Q/A training with thinking + `Final answer:` targets should fix this.
- Run settings that fit one GPU: `--agents-per-gpu 40 --ac-memory-share 0.5 --encode-batch-tokens 32000`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (a first try at 0.4 / 48k ran out of memory, discarded).
- Files: `IB/TMP/PAPER_TRAJ_QA/kit/{run_synth_ac.py, score_synth_ac.py, synth_ac_node.sh, synth_ac_t1_v2_score.json}`; pc-b `SYNC/PAPER_TRAJ_QA/tqa_synth_ac_t1_v2/`. Re-run on any pair: `bash kit/synth_ac_node.sh <gpu> <tag> <pair dir> --agents-per-gpu 40 --ac-memory-share 0.5 --encode-batch-tokens 32000`.

## Green-light run: data done, training running (Oct 14 17:43Z)
- **9B solves** (40k cap, 3 seeds; Musique F1 = 1 / F1 ≥ 0.5, DeepMath math-verify): train Musique any@3 .556 / .666, all@3 .398 / .496, mean .481 / .582 (229 of 1,500 runs at the cap); train DeepMath any .856, all .656, mean .767; reserve Musique any .46 / .57, all .28 / .38, mean .363 / .47 (65 at the cap); reserve DeepMath any .94, all .74, mean .85.
- **Questions (27B on CSAIL):** train single 8,657 (Musique 4,937, DeepMath 3,720), train multi 1,180 (708, 472); reserve single 1,706 (994, 712), reserve multi 239 (144, 95). **Multi-trajectory rows are not in this packet**: the builder reads single-trajectory rows only; using them needs a builder change (next packet).
- **SFT material (27B on pc-a):** summaries 2,994 of 3,000 (844 recovered from a plain reply); completions kept 1,364 of 2,000 (Musique 511 over 263 tasks, DeepMath 853 over 432 tasks; correct, finished, thinking kept); pc-a `SYNC/PAPER_TRAJ_QA/tqa_sft27_v1/sft_kept.jsonl`.
- **Packet** `paper_traj_qa_v1/tqa_own_v1/packets/tqa_own_ro_v1`: from the T1 pair `tqa_blind_ro_final`, blind encoder, sizes log-uniform 1/20–1/5, rows only, 2e-5 / 2e-5; 8,000 items = 1,000 updates (Musique 4,556, DeepMath 3,444). Held panel `task_panel_own_v1`: 64 reserve questions × 4 sizes (256 items); held scores are not comparable to T1's panel. Reserve questions outside the panel are excluded from training.
- **Training:** job 2665289, 2 H200 data-parallel (torchrun; single node only, so GH200 nodes cannot pair). Verified running 17:43Z (models loading). Estimate ≈20:45–21:30Z (T1 took 5 h 32 min on one GH200).
- **Next:** AC synthesizer eval with the new pair (`synth_ac_node.sh`, one GPU) against summaries + results (.40 / .51 / .83) and single call; last-mile SFT on the kept completions if still below; multi-trajectory rows in the next packet.

## Result: the new pair on the reserve set (Oct 14 21:08Z)
- Training job 2665289 completed (2 h 48 min on 2 H200, 1,000 updates). Pair `tqa_own_ro_final`.
- Task set: the 200 reserve tasks (100 Musique, 100 DeepMath; 3 solver runs each). The earlier baseline task set was lost with a node, so results-only was rerun here for a like-for-like baseline; summaries + results, the agentic synthesizer and the T1 pair were not rerun on this set.

| Condition (reserve set) | Musique F1 = 1 | Musique F1 ≥ .5 | DeepMath |
|---|---|---|---|
| Solvers mean@3 | .363 | .467 | .85 |
| Majority of 3 | .35 | .45 | .87 |
| Single call, results only | .40 | .51 | .90 |
| **AC rows + answers, new pair** | **.31** | **.47** | **.83** |

- Paired against results-only: Musique F1 = 1 3 wins / 12 losses (p 0.035); DeepMath 5/12 (p 0.14). Below the text baseline, as T1 was.
- Format: the reader rarely writes the `Final answer:` line (strict parse .05 / .07 / .13); scores above use the last-line fallback.
- By the agreed plan (round 1: "If it's inferior to summaries, then we have a dedicated training pass does SFT on some valid 27B summaries + single-call completions"), the last-mile SFT pass is next.

## Deep residual interventions for the single-call synthesizer (user, Oct 14 ≈22:05Z; lower priority)
- User, verbatim: "One potential thing to experiment with at lower priority (I am saying it now because I may have to log off for ~12h, so you'll have space when it comes). The synthesis only task is single-call, meaning that it likely does not benefit too strongly from using vllm. That means deep residual interventions are technically back in play. I'd experiment with them to boost the numbers."
- Reading: every TQA pair so far is rows only (the vLLM path takes input rows through `prompt_embeds` and nothing else). The synthesizer is one generation per task with no tools or sandbox, so it can run in plain PyTorch, where the AC model's per-layer additions (deep residual interventions) can be applied. Training is plain PyTorch already.
- Plan (runs after the last-mile SFT readout and the compaction 200×3, on CSAIL where possible; no EC2 GPU needed for a single-call eval):
  1. Find what supports additions today: an additions-capable starting pair (the reconstruction pair before the rows-only restriction; any non-`_ro` TQA pair), the trainer's switch, and a PyTorch generation path that applies additions (the side_base path used for compaction encoding is the starting point).
  2. Eval path check: the PyTorch synthesizer with additions turned off must match the vLLM rows-only scores on the reserve set within noise before any additions number is trusted.
  3. Arms, each on the reserve set against results-only (.40 / .51 / .90) and the rows-only pairs: (a) the last-mile SFT data trained with additions on, from an additions-capable start; (b) if cheap, the same pair evaluated rows-only vs with additions.
  4. Rates stay 2e-5 / 2e-5; GH200 at most 2, H200 grouped at most 2.

## Overnight hill-climb (user away ≈12–16 h from ≈22:10Z, Oct 14)
- Proposer findings that change the plan:
  - The existing pairs carry stale addition heads, from the older lc3i training run. Switching additions on for them means nothing, so an additions arm resets the heads and trains them.
  - Most of the Musique gap to results-only is answer extraction, not the rows.
- Schedule:
  1. **Extraction fix (CPU).** A wider scorer applied identically to every condition; strict numbers are always reported alongside.
  2. **SFT readout** (job 2667005, ≈00:15Z). Arms on the reserve set:
     - the final pair;
     - a rows-removed control (the same reader given only the answers);
     - if time allows, the 50 % snapshot and the T1 pair.
     These run on EC2 GPUs handed over by the main thread.
  3. **PyTorch eval path with additions.** Gate before trusting it:
     - teacher-forced logprobs match with additions off;
     - re-encoded rows match the saved rows;
     - a 200-task run of `tqa_own_ro_final` lands within paired noise of .31 / .47 / .83.
  4. **SFT with additions**: the same packet, `deltas_off_training=false`, heads reset. Evaluated through the gated path.
  5. If 2 helps, run more Musique completions (27B or self-samples) and multi-trajectory questions.
- Rates 2e-5 / 2e-5. GH200 at most 2, shared with compaction. No training on reserve tasks.
- Extraction fix done (`kit/score_synth_lenient.py`; same parser for every condition).
  - AC own pair, lenient + last line: .37 / .47 / .89. Strict: .05 / .07 / .13.
  - Results-only: .40 / .51 / .90.
  - The paired gap to results-only is no longer significant: Musique +3/−6, p .51; it was 3/12, p .035.
- **Last-mile SFT readout (u245 = update 245 of 326; the final save failed on the CSAIL disk quota).** Reserve 200, every reply in the requested format (strict = lenient).
  - Musique F1=1: .42 vs results-only .40 (+6/−4) and majority .35 (+7/−0, p .016).
  - DeepMath: .94 vs .90 (+5/−1) and .88 (+6/−0, p .031).
  - AC rows + answers is now at or above the single-call text baseline; the gap to results-only is not yet significant.
  - Pending: the rows-removed control, u82, and a second seed.
- **Rows-removed control (same SFT reader, answers only):** Musique .41, DeepMath .90. So the SFT gain is mostly the reader learning the task and format.
  - Rows add ≈+1 point on Musique (u245 vs rows-removed +5/−4) and +4 points on DeepMath (+5/−1, p .22).
  - u82 (.41 / .92) ≈ u245, so the SFT saturates early.
  - A second seed of u245 and the control is queued. The next arms aim at making the rows matter: the additions SFT (running) and harder, multi-trajectory questions.
- **Two seeds (u245):** AC .405 / .49 / .92; rows-removed .405 / .51 / .895; results-only .40 / .51 / .90.
  - Rows vs rows-removed per task: Musique 6/6, DeepMath 6/1 (p .125).
  - So AC matches the text baseline and beats majority vote. The rows add a small, non-significant DeepMath edge, and nothing on Musique. The seed-0 DeepMath .94 was partly luck.
- **Rows only, answers removed (u245, seed 0):** Musique .38, DeepMath .92, against results-only .40 / .90 and majority .35 / .88.
  - The rows alone carry nearly all the signal the answer text carries. The two are redundant, not additive.
- **Deep residual interventions:** additions on vs off on the same trained pair gives .39 / .51 / .90 vs .40 / .50 / .91 (all p = 1.0). No measurable effect.
- **Rebalanced SFT (1a):** .40 / .49 / .92, the same as u245. On Musique tasks where no solver was right (54), every condition scores 0–1.
- **Last arm:** 1b, 27B targets written from the trajectory text without the agents' answers (Musique), to open that pool.
- **1b closed:** the 27B, reading the full runs with the agents' answers withheld, recovered the gold on ≈2 % of samples for Musique tasks no agent solved (6 of 264). The k=0 pool is not reachable this way. Synthesis hill-climb ends with u245 as the best arm.

## Round 2: disagreement-targeted SFT (Oct 15 ≈12:20Z, user green light ≈12:00Z)
- User: more, better-selected data; Musique first ("0.5 for musique seems small and there is room for improvement"); DeepMath "still look for a push; ... don't dedicate too much of your brain to it".
- Proposer finding (from local data): on the 100 reserve Musique tasks only ≈11–13 can move (k=3 28, majority-right 7, minority-right 11, k=0 54). The most any arm can gain over results-only is ≈6 tasks, so 100 tasks cannot show a significant gain. Last-mile SFT targets were mostly copying (396 of 511 Musique from k=3 tasks; 33 from minority-right).
- Decisions (mine, not the user's):
  - Pre-register an extended reserve of ≥300 more MuSiQue validation tasks before any new training. The 200 reserve stays as is.
  - Dev set (≈300 Musique + 200 DeepMath) for the gate.
  - Material: ≈5 solver runs per train task, mixed triples (minority-right and majority-right ≈1 to 1, k=3 at 25 %, no k=0), 27B question-aware digests, 4 completions per mixed triple, grounded keep rule.
  - Generation on CSAIL; EC2 stays with compaction.
- Gate: on dev, AC beats its rows-removed control by ≥4 Musique mixed tasks, unanimous ≥95 %. Then read on reserve + extended reserve, 2 seeds, by task group.
- Run by a fork; numbers land here when read.

## Presentation for the writing session (Oct 16 ≈11:00Z)

The user (10:40Z) asked for simpler presentation. DeepMath may become the main synthesis dataset, because it does not confound with retrieval, the next experiment. All numbers are to come from 3 seeds.

**Proposal:**
- **One held-out set per dataset.** Musique: the extended reserve (300, frozen, never used for a decision). DeepMath: a new frozen set of 300, being built.
- **One metric per dataset.** Musique exact match (F1 = 1); DeepMath accuracy. Answers are read the same way for every condition.
- **One table per dataset:** mean ± sd over 3 seeds.
  - Columns: majority vote over the 3 runs, the text baseline (the 3 runs' results in one call), and AC rows + answers.
  - Reference rows: the single-run mean and the any-of-3 ceiling.
- **Left out of the main text:**
  - The agreement groups (k = 0–3).
  - Rows-removed and rows-only controls; these go to one ablation table.
  - F1 ≥ .5.
  - The dev and reserve sets; dev and reserve are development history.
- **What the current numbers support on the Musique extended reserve** (two seeds so far):
  - AC .350 / .357 against majority vote .310, a single run .322, the text baseline .343 / .327, and the ceiling .420.
  - So AC beats majority vote and a single run, and is level with the text baseline.
- **Pending:** the DeepMath extended reserve, a third seed everywhere, and arm B.
- **DeepMath headroom (Oct 16 13:40Z).** On the new frozen set of 300:
  - A single run solves .850, majority vote .867, and any of 3 runs .920. That leaves 16 tasks between majority and the ceiling.
  - By difficulty:

    | Difficulty | Tasks | Single run | Majority | Any of 3 |
    |---|---|---|---|---|
    | 6 | 103 | .90 | .92 | .96 |
    | 8 | 54 | .81 | .81 | .87 |
    | 8.5 and up | 28 | .61 | .61 | .71 |

  - A harder draw mainly adds tasks no run solves. The majority-to-ceiling gap only grows from 5% to 7% at difficulty 8 and up.
  - The lever for room is more runs per task (k = 6), which raises the any-of-k ceiling and the share of tasks where the runs disagree. Difficulty alone does not do it.
  - Untouched pool: 15,246 tasks at difficulty 8 and up, 4,829 at 8.5 and up.
- **DeepMath, 300-task frozen set, 3 seeds (Oct 16 16:20Z).**

  | Condition | Solved of 300 | At difficulty 8 and up (82) |
  |---|---|---|
  | One solver run | .850 | .740 |
  | Majority vote | .867 | .744 |
  | Results only (text) | .898 | .785 |
  | AC (u245) | .889 | .785 |
  | AC (M1, math round) | .894 | .780 |
  | Any of 3 (ceiling) | .920 | .817 |

  - u245 beats majority vote (34 tasks won, 14 lost; p .006) but not results-only (13 won, 21 lost; p .23).
  - The math-targeted round M1 trains at difficulty 8.5 and up. It is read on a new frozen set of 300 at that difficulty.
  - On this set M1 ties results-only (15 tasks won, 18 lost). Below difficulty 8.5 there is little room: at 7.5–8 results-only already matches the ceiling.
  - **Difficulty figure (Oct 17 11:30Z):** `IB/TMP/PAPER_TRAJ_QA/figs/fig_tqa_difficulty_m1.png`, built by `kit/make_fig_tqa_difficulty.py`. Buckets 6 to 8 come from this set and 8.5 and ≥9 from the hard set. M1 minus results-only by bucket: 0, −0.4, −2.2, +1.0, +5.4 points. The gain appears only at the top.
- **Musique, extended reserve of 300, 3 seeds (Oct 16 17:10Z).** Exact match (F1 = 1).

  | Condition | Solved |
  |---|---|
  | One solver run | .322 |
  | Majority vote | .310 |
  | Results only (text) | .330 |
  | AC (u245) | .351 |
  | Any of 3 (ceiling) | .420 |

  - u245 beats results-only: 44 tasks won, 25 lost (p .029).
  - The edge comes from tasks where all three runs agree. There u245 wins 15 and loses 0, because results-only sometimes changes a unanimous answer.
  - Where the runs disagree, the two tie (24 won, 24 lost).
- **Musique extended reserve, round 2 model B with and without rows, 3 seeds (Oct 17 08:50Z).** Exact match, of 300.

  | Condition | Solved | Disagreement tasks (57) |
  |---|---|---|
  | Majority vote | 93 (.310) | |
  | Results only (text) | 99.0 (.330) | 34.7 |
  | AC (u245) | 105.3 (.351) | 34.7 |
  | AC (B) | 108.7 (.362) | 39.0 |
  | B, rows removed | 101.7 (.339) | 35.7 |
  | AC (M1, math round) | 109.0 (.363) | 38.7 |
  | Any of 3 (ceiling) | 126 (.420) | |

  - The rows help here. B beats its rows-removed control: 38 tasks won, 17 lost (p .006).
  - B beats results-only overall (+43/−14) and on the disagreement tasks (+25/−12, p .047).
  - On DeepMath the rows add nothing so far (dmhard: B vs rows removed +28/−38).
  - Math targeting cost nothing here: M1 ties B (+20/−19) and beats results-only (+48/−18; disagreement tasks +29/−17).
- **DeepMath hard set (300 at difficulty 8.5 and up, frozen), 3 seeds (Oct 17 01:35Z).**

  | Condition | Solved of 300 | At difficulty 9 and up (74) |
  |---|---|---|
  | One solver run | .744 | .725 |
  | Majority vote | .790 | .757 |
  | Results only (text) | .812 | .820 |
  | AC (u245) | .804 | .797 |
  | AC (B) | .812 | .829 |
  | AC (M1, math round) | .833 | .874 |
  | M1, rows removed | .820 | .838 |
  | Any of 3 (ceiling) | .883 | .905 |

  - M1 (Oct 17 08:40Z) is the best arm: 250.0 ± 1.0 of 300. Against results-only it wins 41 tasks and loses 22 (p .023); at difficulty 9 and up, 15 and 3 (p .008). Against B, 37 and 18 (p .014).
  - B ties results-only (38 tasks won, 38 lost) and beats majority vote (46 won, 26 lost; p .024).
  - No model yet closes the gap of 21 tasks to the ceiling. Most of that gap is at difficulty 8.5.
  - B with its rows removed (B-RR) solves .823, the best so far. With rows against without, B wins 28 tasks and loses 38 (p .27). The current set shows the same (.899 without rows, .889 with).
  - So on DeepMath the trajectory rows add nothing yet. B's fine-tune reads the three results slightly better than the untuned model (33 won, 23 lost; not significant).
  - With the rows removed, M1 drops to 246.0 ± 2.0 and is no longer clearly above results-only (+29/−22, p .40). M1 with rows against without: 30 tasks won, 18 lost (p .11); at difficulty 9 and up, 12 and 4 (p .08).
  - On the extended set the rows hurt M1 instead: M1 without rows solves 272.3 (.908), above M1 268.3 (19 won, 7 lost; p .03) and above results-only (13 won, 4 lost; p .049). The losses sit at difficulty 7.5–8 (+1/−12 for the rows).
  - Pooled over both sets, M1's rows win 37 and lose 37. On DeepMath the math fine-tune is what helps; the rows' effect is unresolved. On Musique the rows clearly help (B vs rows removed, p .006).
  - **Hard set with 5 seeds (Oct 17 14:30Z; `node_sync/dmhard/tables_M1_5seed.txt`).** Seeds 3 and 4 came in lower for M1, so the 3-seed numbers above were on the lucky side.

    | Condition | Solved of 300 | 8.5 (226) | 9 and up (74) |
    |---|---|---|---|
    | Majority vote | .790 | .801 | .757 |
    | Results only (text) | .815 | .812 | .824 |
    | M1 | .827 | .818 | .857 |
    | M1, rows removed | .821 | .818 | .832 |
    | Any of 3 (ceiling) | .883 | .876 | .905 |

  - M1 against results-only: 64 won, 46 lost overall (p .11, no longer significant); at difficulty 9 and up, 22 won, 10 lost (p .050).
  - M1 against majority vote: 86 won, 30 lost (p < .001).
  - Rows (M1 against M1 without rows): 47 won, 38 lost (p .39); at 9 and up, 18 won, 9 lost (p .12).
  - Figure with 5 seeds on the hard set: `IB/TMP/PAPER_TRAJ_QA/figs/fig_tqa_difficulty_m1_5seed.png`. Gain over results-only by bucket: 0, −0.4, −2.2, +0.5, +3.2 points.

## Paper figure: DeepMath by difficulty (Oct 18)

- **Canonical MVP figure (user, Oct 18; Results + Summary added Oct 19 at the user's request):** `figs/fig_tqa_mvp.{pdf,png}`, numbers in `figs/fig_tqa_mvp.json`. The version without Summary is `figs/fig_tqa_mvp_nosummary.*`. Bars show % solved for Single Solver, Majority Vote, Results-Only, Results + Summary, Results + LoRA and Results + AC (ours), with Any of 3 as a dashed ceiling. The y-axis runs from 70 to 95.
- Columns, kept separate (user's choice):
  - 6–8 from the wide set: 272 tasks, 3 seeds.
  - 8.5 and ≥ 9 from the hard set: 226 and 74 tasks, 5 seeds.
  - The wide set's 28 tasks at 8.5 and up are left out.
- Framing (user): the selection leans toward hard problems because that is where differences are expected. On easier problems the answers alone are enough.
- Majority Vote uses a random tie-break: with no shared answer, it takes the expected score of a random run's answer. The paper still calls it "Majority Vote".
- Results + LoRA is M1's reader adapter with the AC parts removed.
- Compression: 1/10 per solver run (realized 0.100).

  | % solved | 6–8 (272) | 8.5 (226) | ≥ 9 (74) |
  |---|---|---|---|
  | Single Solver | 87.5 | 75.1 | 72.5 |
  | Majority Vote | 90.1 | 81.0 | 76.1 |
  | Results-Only | 92.4 | 81.2 | 82.4 |
  | Results + LoRA | 93.3 | 81.8 | 83.2 |
  | Results + AC (M1) | 91.5 | 81.8 | 85.7 |
  | Any of 3 | 94.1 | 87.6 | 90.5 |

- Paired results, task × seed (wins/losses, sign test):

  | Comparison | 6–8 | 8.5 | ≥ 9 |
  |---|---|---|---|
  | Results + AC vs Results-Only | +9/−16 | +42/−36 | +22/−10 (p .050) |
  | Results + AC vs Majority Vote | +25/−13 (p .073) | +37/−33 | +41/−4 (p < .001) |
  | Results + AC vs Results + LoRA | +4/−18 (p .004) | +29/−29 | +18/−9 (p .12) |

- Absolute rates are not monotone in difficulty. Any of 3 is 90.5 at ≥ 9 and 87.6 at 8.5, which is noise at n = 74. Two-thirds of the hard set's gold answers are yes/no. The ceiling matters more than the label.
- **Results + Summary (added Oct 19 to the table and, at the user's request, to the figure).**
  - Setup: the untuned 9B writes one summary per solver run; the reader then answers from the results plus the summaries.
  - Seeds and coverage: the same seeds as the other arms (5 for the hard set, 3 for the wide set). Every row is present: 300 per seed, with 0 bad and 0 cut off at the length limit.
  - Files: table at `tables/tables_unified_tb_sumres.txt`, numbers at `tables/tqa_unified_sumres.json`, script at `scripts/tqa_unified_table_sumres.py`.

  | % solved | 6–8 (272) | 8.5 (226) | ≥ 9 (74) | all (572) |
  |---|---|---|---|---|
  | Results-Only | 92.4 | 81.2 | 82.4 | 86.7 |
  | Results + Summary | 92.6 | 79.6 | 83.5 | 86.3 |
  | Results + LoRA | 93.3 | 81.8 | 83.2 | 87.4 |
  | Results + AC (M1) | 91.5 | 81.8 | 85.7 | 86.9 |

  | Paired | 6–8 | 8.5 | ≥ 9 | all |
  |---|---|---|---|---|
  | Summary vs Results-Only | +9/−7 | +24/−43 (p .027) | +13/−9 | +46/−59 |
  | AC vs Summary | +8/−17 (p .11) | +50/−25 (p .005) | +18/−10 (p .19) | +76/−52 (p .042) |

  - Reading: the summaries do not help, and at 8.5 they hurt. Results + AC beats Results + Summary overall.
- Scripts are copied to `scripts/` and run from `IB/TMP/PAPER_TRAJ_QA/kit`:
  - `tqa_unified_table.py --sync-root <SYNC mirror> --json ...`
  - `make_fig_tqa_unified.py --style solved --ylim 70 95`
- Table text: `IB/TMP/PAPER_TRAJ_QA/node_sync/unified/tables_unified_tb.txt`.
