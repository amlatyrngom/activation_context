# Paper MVP — agent trajectory compaction (agent-facing source)

Source of truth for `PLAN.tressoir.md` in this folder. Version 3, Oct 13 2026 (17:46Z; version 2 was 17:20Z), still intent level by the user's wish ("I think the next version will also be intent"): no milestone is approved, no product code may be written for it yet. Later version sections win over earlier text. The human projection carries every decision and caveat; this file adds the evidence, paths and commands behind them.

## Origin

User message, Oct 13 ≈16:45Z (new thread of work): downgrade every running experiment to lowest priority and push for the paper; relentless focus on an MVP; use 2×H200/GH200 as much as possible; give a gentle warning when asked to micro-optimize (the paper matters, not 1–2 %). Requested beats for this plan: train/test dataset selection and task limits; trajectory syntheses (27B? 9B?); training targets (from the aspirational writing); plots (main text, maybe appendix) as the terminal goal.

Paper text: `IB/evaluation_section_snapshot.tex`, subsection "Agent Trajectory Compaction" (lines 8–31). Its fixed points: Long Code Arena bug localization, Java only; compaction limit 32k; 27B agent at medium reasoning builds the pool; runs under 32k and over 50k are excluded; 200 evaluation traces from dev or test; successful training-set traces for training, plus unrelated tasks (math, search); comparisons = text summary and truncation at ratios (no context, full context, 1/4, 1/8, 1/16, 1/32); encoder Qwen 3.5 2B with adapter modules, reader Qwen 3.5 9B; open TODO "27B or untuned 9B shared prefix; likely 27B"; single-agent trajectories; token consumption instead of raw speed, except the compaction-time comparison on an H200/GH200.

## Evidence gathered for this version

### Benchmark sizes (dataset card, read Oct 13)

`JetBrains-Research/lca-bug-localization`: java dev 2,522 / train 2,472 / test 50; py 4,339 / 4,289 / 50; kt 618 / 568 / 50. `train` + `test` = `dev`.

Java `train` rows: 286 repositories. Under the loader's default filters (`activation/dataset/loaders/lca_bug_localization.py · LcaBugLocalizationDataset.load`: at most 3 changed files, at most 3,000 repository files, a changed code file, at most 20 tasks per repository): 1,046 tasks in 253 repositories, 100 repositories with a single task. Java `test`: 45 repositories, 29 tasks after the same filters. The 12-hour campaign's pool id is `lca_bug_localization___config_java__split_dev__n_1925__seed_0` (looser settings; confirm which before M1).

### Window yield (12-hour 27B campaign)

Cache: `IB/TMP/AGENT_AC_PRETRAINING/node_camp/ROLLOUTS/teacher_campaign_12h/rollouts.jsonl` (4,708 rollouts, `unsloth/Qwen3.8-27B-NVFP4`, reasoning medium, `max_turns` 100, `max_duration` 1,800 s, `compaction_threshold_tokens` 32,768, `absolute_trajectory_cap` 50,000). Teacher mix: 20 % base (single agent), 80 % multi-agent. Script used: job-temporary `lca_window.py` (length = prompt tokens + every step's tokens, summed over compaction segments of the main agent; re-create from `IB/TMP/AGENT_AC_PRETRAINING/token_stats.py`).

Bug localization, 1,090 rollouts over 925 tasks (py 693, java 337, kt 60); finish reasons: submitted 864, max_duration 186, max_tool_errors 23, max_turns 15, trajectory_cap 1, no_tool_call 1; compactions per run: none 950, one 117, two 15, three 5, four 3.

Runs without subagents (250): under 8k 86 runs / 82 solved; 8k–16k 47 / 43; 16k–32k 38 / 24; 32k–50k 32 / 7; 50k–80k 37 / 3; over 80k 10 / 0.

By language, runs without subagents: py 164 runs, 107 solved, window 20 (4 solved; finish: submitted 7, max_duration 9, max_turns 2, max_tool_errors 2), over 50k 27; java 79 runs, 48 solved, window 12 (3 solved; finish: max_turns 5, submitted 4, max_duration 2, max_tool_errors 1), over 50k 17 (2 solved); kt 7 runs, 4 solved, window 0, over 50k 3.

All teacher kinds, main agent's own trajectory in the window: py 52 of 693 (14 solved), java 34 of 337 (14 solved), kt 3 of 60 (2 solved).

Caveats: compaction was on in those runs (the totals include the re-prompt after a compaction), and time/turn limits ended many long runs. A pool run without compaction and with a generous time limit will shift the bands; M0 measures the real yield.

### Earlier compaction work (research pass over `IB/ARTIFACTS/FIRST_TASK_AWARE_FINETUNING/`, called FT below; claims not re-verified by me are marked R)

- R: Regular agent teacher segments are the only bank with bug-localization traces: `campaign_12h_20260911`, 4,279 runs / 12,103 segments, local `IB/TMP/SYNC/REUSABLE_DATA/agent_teacher_regular/`; v2 export 9,391 roots / 11,313 segments on `s3://general-bucket-aln/activation-context-data/datasets/agent_teacher_regular` and CSAIL `/data/scratch/ngom/activation/recon_100m_v1/data/agent_teacher_regular` (`FT/AGENT_REGULAR_V2_EXPORT.json`). Row format: one JSONL row per segment (`main`, `compaction_i`, `subagent_j`) with messages, `target_start`, tools, gold answer, score, finish reason; 4 % held by root task. Producers: `activation/bench/canonical_training/generate_agent_teacher_trajectories.py`, `export_agent_teacher_segments.py`, `check_exported_segments.py`.
- R: Compaction bank: 65,924 rows of pre-cut history plus a Qwen3.8-27B-FP8 five-section handoff, from public coding/research/math traces (not bug localization): `IB/TMP/SYNC/REUSABLE_DATA/compaction/`.
- R: Continuation selection: 18,408 training cuts + 100 held, histories of 8k/16k/32k, suffix at most 8,192 tokens (`FT/TRAJECTORY_COMPACTION_SCALE_REVIEW.md`, `FT/COMPACTION_TO_QA_1848.json`).
- R: Offline continuation loss (NLL per token) on the held 100 cuts, 2B encoder → 9B reader at 1/16: frozen full history 0.689; no history 0.8645; recent text at the same row budget 0.811; oracle-summary text 0.811; AC untrained start 0.844; AC after 1,848 histories of continuation KL 0.799; after 3,688 histories 0.787. First-128-token KL 0.680 → 0.340; the 16k–32k bin still trailed recent text at 1,848 (`FT/TRAJECTORY_INSIGHTS.md`). Pair on ORCD `~/orcd/pool/activation/trajectory_compaction/compaction_50m_v8/selected/examples_0001848/`.
- R: Oracle-summary SFT on 8k histories: summary NLL 0.941 → 0.598, continuation NLL 0.798 → 0.799, first-128 worse (1.156 → 1.232).
- R: Live resume probe `FT/bench/probe_live_compaction.py` (six synthetic fixtures, serial HF agent): full history 6/6, no history 0/6, initial AC 0/6, continuation-KL pair 1/6. No script resumes a bug-localization task from a replaced prefix and scores it.
- R: Deep-intervention checkpoints cannot run through the ordinary `Agent`/vLLM path, which accepts input rows only (`FT/TRAJECTORY_IMPLEMENTATION_HANDOFF.md`).
- Canon (`IB/CANON/ACTIVATION_CONTEXT_CANON.md`, "Decision — Compaction learns an oracle handoff"): official compaction target = SFT on the 27B oracle summary; dedicated continuation training deprioritized; continuation evaluated separately with full-history, no-history and oracle-summary controls and live tool-use checks. This plan asks to reverse the target for the paper path (decision `pcomp.target.*`); update the canon only after the user's answer.
- Agent runtime: `activation/agent/agent.py · Agent.compact`, `Agent.resume_from_run_result`, `Agent.flattened_view`, `to_activation_context("compaction")`; config `activation/agent/agent_config.py` (`compaction_threshold_tokens` 32,768, `absolute_trajectory_cap` 50,000, `ac_compaction_ratio` 1/10); summary prompt `COMPACTION_SUMMARY_PROMPT` in `activation/common/ac_parts.py`.

### This thread's 4B question-answering results that bear on the choices

Frozen reader beat trained reader in every pair (59.7 vs 56.4 at 1/20 on the full set); text loss beat KL-only by 8–10 points on the 4B reader, while KL had won on the 9B reader earlier (67.4 vs 61.7); encoder rate 1e-4 best of {2e-5, 1e-4, 3e-4}; 5e-5 is never used; reconstruction starts did not help question tuning; key/value transfer exists for same-model encoder/reader only (`activation/harness/kv_transfer.py`), with an unexplained batched-versus-single decode difference in the 9B overwrite mode.

## Intended design (pending decisions)

1. **Split.** Repository-level split of the Java `dev` rows; evaluation side about 60 % of repositories and all 50 official test tasks; training side the rest. Python/Kotlin wholly training if chosen. Freeze the split in a file under this folder before the pool run.
2. **Pool run.** 27B (`Qwen3.8-27B-FP8` writer by standing decision; the campaign used the NVFP4 build — pick one and record it), medium reasoning, base teacher only (no subagent tool), no compaction (threshold above the cap), `absolute_trajectory_cap` 50,000, generous `max_duration`, new caching id.
3. **Window and cut.** Qualifies: passed 32,768 tokens, ended under 50,000 (any finish reason unless the window decision says solved only). Cut at the first turn boundary past 32,768, counted with the reader's tokenizer. Store per prefix: messages up to the cut, sandbox restore recipe (repository tarball + replay or snapshot — to be designed in M0), the 27B's own continuation, gold answer.
4. **Resume benchmark (the missing piece).** One program taking (prefix, variant) → resumed 9B run with the same tools → solved flag, continuation tokens (generated + tool output tokens after the cut), turns. Variants: full, none, recent text at each grid ratio, text summary by 9B / by 27B, AC rows at each grid ratio (frozen reader), AC rows + 1k recent text, AC rows with tuned reader. Same continuation budget for all (cap − prefix ≈ 17,000 tokens); unanswered = unsolved at the full budget. Drive the reader through the HF path the trainer uses so per-layer AC inputs work; check on the pilot that its full-context accuracy agrees with the serving engine's.
5. **Training set.** Successful runs from training repositories, several cuts each between 8k and 40k, plus existing math/search segments from the regular bank; continuation tokens = the 27B's own; ratio drawn per example from the grid range. Evaluation repositories and the retrieval subsection's MuSiQue test questions excluded.
6. **Training arms.** A: KL to the frozen 9B reading the raw prefix (top-32 targets, existing trainer option). B: text loss on the continuation. Reader frozen in both; encoder rate from the existing recipe that worked on this pair (not 5e-5). Selection on the 50-prefix development panel by live solved rate, never on the 200.
7. **Figure.** One script reading one results JSON; panels (a) solved vs size, (b) continuation tokens vs size, (c) seconds to compact; appendix by-products from the same runs.

## Open technical points for M0

- Where sandboxed (Podman) agent runs can execute beside the two GPUs: CSAIL GPU nodes are unverified for Podman; earlier campaigns ran on rented machines. No Sky cluster exists now.
- Restoring the sandbox at the cut: bug localization is read-only exploration of a checkout, so re-creating the checkout is probably enough; verify that prefixes contain no state-changing shell commands that the continuation depends on.
- Whether an existing 2B → 9B pair is a usable start (ORCD `compaction_50m_v8/selected/examples_0001848`, or the mixed run's stop checkpoint) or training starts fresh.
- Throughput of one resume run on the HF path; decides whether all variants × 200 fit comfortably.
- The 27B build (FP8 vs NVFP4) and which GPU serves it while the other trains.

## Decision keys published in the projection (never reuse)

`pcomp.evalset.{repo_split_seeds,hundred,add_python,feedback}`, `pcomp.traces.{all_then_gate,teacher_solved,lower_limit,feedback}`, `pcomp.prefix.{teacher27b,reader9b,both_small,feedback}`, `pcomp.target.{continuation,keep_summary_arm,reconstruction_first,feedback}`, `pcomp.form.{canon_2b_9b,port_kv,same_model,feedback}`, `pcomp.grid.{decimal,binary,feedback}`, `pcomp.trainmix.{py_kt_unrelated,java_unrelated,feedback}`. Answers arrive in `PLAN.interactions.json` beside the projection, or in chat. All `pcomp.*` keys are retired (answered Oct 13 17:14Z, removed from the interactions file); live keys are `pcomp2.split.{own_split,official_only,feedback}`, `pcomp2.window.{w20k,w16k,pilot_decides,feedback}`, `pcomp2.trainmix.{with_unrelated,buglocal_only,feedback}`, `pcomp2.engine.{parallel,trainer_path_dev,first,feedback}`.

## Milestones (all TBD)

- M0 — figure first on baselines: ≈350 Java evaluation-repository tasks → ≈50 prefixes; resume benchmark; untrained variants; summary timing; figure script. Gate: full-context minus no-context solved rate on the 9B reader (come back to the user if under about 10 points), real window yield, time per resume run.
- M1 — pool: fix the split, run the rest, freeze the 200 prefixes, export training cuts.
- M2 — training: build the continuation set, two arms on two GPUs, pick on the development panel; tuned reader only afterwards.
- M3 — final: all series on the 200 once; figure and table; rewrite the Results paragraph from the figure.
- M4 — appendix extras: training mix, target, prefix writer; only after M3.

## Version 2 (Oct 13, after the user's answers)

### Answers received (`PLAN.interactions.json`, 17:14Z, verbatim)

- `pcomp.evalset.feedback`: "Let's actually just use all splits. I wrote java in a rush." (no box ticked)
- `pcomp.traces.feedback`: "I am thinking of lowering to 16k-50k range, or 20k-50k since 20k looks better on paper (>19k lol)." (no box ticked)
- `pcomp.prefix.teacher27b`: true; feedback: "However, the higher priority experiment is the 9B summary. 9B reader means untuned 9B BTW I hope."
- `pcomp.target.continuation`: true
- `pcomp.form.canon_2b_9b`: true
- `pcomp.grid.feedback`: "1/5, 1/10, and 1/20th only for now. More on request." (no box ticked)
- `pcomp.trainmix.feedback`: "Revise this question in light of above."
- `PLAN-free_form_feedback`: "There is possibly a big missing preliminary step: pushing interventions into vllm. Fork it (if not already done) and push a copy to my github. This will be one the claims in the earlier parts of the paper and is needed for rollouts."
- Chat: "See my interactions. I think the next version will also be intent."

### How each was integrated

- Languages: read "all splits" as all three language subsets (my reading; asked back as `pcomp2.split.*`). Own repository split still needed: official test = 150 tasks.
- Window: lower bound open between 20k and 16k (`pcomp2.window.*`), cap stays 50k. "All qualifying runs count" kept from the snapshot; the pilot gate stays.
- Prefix by 27B accepted. Text summary by the untuned 9B is the first baseline built; 27B summary demoted to "on request". "9B reader" = untuned `Qwen3.5-9B` everywhere; pure AC series trains the encoder only against the plain base (no reader adapter; existing 2B → 9B pairs that carry a trained reader adapter are not "untuned reader" pairs — start the encoder fresh or load only its side).
- Target: continuation matching accepted. Canon section retitled "Decision — Compaction learns to continue (2026-10-13; replaces the oracle-handoff target)" with the old target recorded as superseded and evaluation by resumption.
- AC form: existing 2B → 9B (input rows + per-layer residual additions) accepted.
- Grid: 1/5, 1/10, 1/20 only. "AC rows + 1k recent text" = rows at 1/20 + 1,000 tokens (equal in size to 1/10 at a 20k prefix).
- Training mix: re-asked as `pcomp2.trainmix.*`.
- Engine: new milestone E and a section in the projection; ordering asked as `pcomp2.engine.*`.

### Evidence added

Windows, 12-hour campaign, runs without subagents (250; py 164, java 79, kt 7), lower bound → runs inside / solved / submitted: 16k 70 / 31 / 44 (py 49/23, java 20/7, kt 1/1); 20k 61 / 24 / 35 (py 43/17, java 17/6, kt 1/1); 24k 50 / 16 / 25; 32k 32 / 7 / 11. Over 50k: 47. Every teacher kind (1,090 runs, main agent's own trajectory): 16k 305 (155 solved); 20k 239 (112); 32k 89 (30). Overall solved, single agent: 159 of 250.

Pool under the loader's default filters (`train` rows): py 2,337 tasks / 499 repositories; java 1,046 / 253; kt 214 / 39 (raw rows 4,289 / 2,472 / 568 in 516 / 286 / 43 repositories). Official `test` under the same filters: py 38 of 50, java 29, kt 35.

Arithmetic used in the projection: 200 prefixes at 24 % yield ≈ 830 runs (16k: ≈710; 32k: ≈1,560); evaluation side ≈1,000 tasks, training side ≈2,700, of which ≈64 % solved ≈1,700; pool ≈3,700 runs. Interval for 35 prefixes at 35 % ≈ ±16 points; for 200 ≈ ±6.6.

### Engine state (checked Oct 13 17:15Z)

- `https://github.com/amlatyrngom/vllm` exists, public, fork of `vllm-project/vllm`; branch `ac-interventions-v0.28.0` = `2cf0a6915ce544dc493a0990f2ea38d81601128a` (the upstream v0.28.0 tag; no further commits). `amlatyrngom/activation_context` is public as well.
- `IB/REPOS/vllm`: working tree identical to that branch (`diff -rq` against a fresh shallow clone: no differences); its `.git` file points at `.git/modules/IB/REPOS/vllm`, which no longer exists, and the root has no `.gitmodules`. Engine work must start from a fresh clone of the fork (not from this tree).
- Installed engine: vllm 0.28.0 from the upstream wheel (`.venv`). It carries input rows only (mixed token/embedding prompts); canon: "Engine mapping: a per-request scatter-add on the hidden state entering the named layers at the rows' prompt positions."
- HF side: `activation/harness/hf_utils.py · Interventions, BoundInterventions, install_intervention_hooks`; `activation/harness/loaded_model.py · decoder_forward(interventions=...)`.
- Specification to implement: `IB/ARTIFACTS/VLLM_ACTIVATION_INPUTS/PLAN.md`, sections V1 (public API to engine IPC: `entrypoints/llm.py`, `v1/engine/async_llm.py`, `input_processor.py`, `EngineCoreRequest`, serial utils), V2 (request state, preemption/resume, per-block cache hashing in `v1/core/kv_cache_utils.py`), V3 (`gpu_model_runner._prepare_inputs` → `layer_input_deltas` → `qwen3_5.py` wrappers → `qwen3_next.py · Qwen3NextModel.forward` `index_add` before the layer; CUDA graphs and compile off for intervention engines), V4 (pin the fork commit through `[tool.uv.sources]`, precompiled-wheel reuse). Re-verify every anchor against the fork before detailing milestone E.
- Nothing was pushed to the fork in this pass: there is nothing to push until the implementation is approved.

### Milestones, version 2 (all TBD)

E (engine, beside M0) → M0 (≈250 evaluation-side tasks, resume benchmark, variants in order: full, none, 9B summary, recent text at three sizes; figure script) → M1 (pool ≈3,700 runs, frozen 200) → M2 (two arms, reader untouched) → M3 (final, all AC rollouts through the fork) → M4 (extras on request).

## Version 3 (Oct 13, after the second round of answers)

### Answers received (`PLAN.interactions.json`, 17:40Z, verbatim; found at the timed check-in, no chat message)

- `pcomp2.split.own_split`: true; feedback: "Doesn't the benchmark have a test set, a dev set and a train set?"
- `pcomp2.window.w20k`: true
- `pcomp2.trainmix.with_unrelated`: true; feedback: "Not necessary if tuning from our best reconstruction works (see data from ~1 week ago, something like/adjacent to lc3i or something was the checkpoint). My thinking was simply that ~2k tasks may not be enough for training from scratch."
- `pcomp2.engine.parallel`: true; feedback: "I'd rather we don't turn off the prefix cache. Introduces an extreme inefficiency that instantly makes the code unusable. I'd try an input-vector only training probe (some easy metric like nll)."

### Integration

- Split question answered with a check of task ids (streamed `text_id` per split): java dev 2,522 / train 2,472 / test 50, kt 618 / 568 / 50, py 4,339 / 4,289 / 50; in each, train ∩ test = ∅ and train ∪ test = dev. Evaluation side = official test (150; 102 pass the loader's default filters → ≈25 prefixes at 24 %) + held-out `train` repositories up to ≈1,000 tasks.
- Window fixed: 20k–50k; prefix 20,000; sizes 4,000 / 2,000 / 1,000; continuation budget 30,000 (20k prefix + 30k continuation = the 50k bound). The user asked for 32k and withdrew it within minutes on Oct 13: "Oh my bad: 30k is actually the nicer story since we are doing the 20k-50k bound."
- Training: tune from the best reconstruction pair; bug-localization cuts only at first; math/search traces are the fallback (the user ticked the mix but said it is not necessary if the reconstruction start works — I read the text as the binding part).
- Engine: no cache-off stage anywhere. New milestone P (probe) placed first; E's place in the order depends on P.
- All `pcomp2.*` keys retired and removed from the interactions file. Live keys: `pcomp3.reader.{keep_adapter,drop_adapter,both,feedback}`, `pcomp3.start.{probe,pilot,engine_design,nothing,feedback}`.

### The reconstruction pair (checked on ORCD, ≈17:44Z)

`~/orcd/pool/activation/long_recon_v1/lc3i/`: `AC_MODELS/rag_poc/epoch_001/{ac_modules.pt (1.2 GB), optimizer.pt (4.7 GB, never copy), side_lora/}` + `LORAS/rag_reader/epoch_001`; also `recipe.json`, `panels.json` (11 panels), `items.pkl`, `prepared*.json`, `fixed_references.json`, `DIAG/`. Recipe: ratio 0.1 (single ratio), `summary_layout` interleaved, layout `bundle`, loss SFT, `lr_ac` 2e-5, cosine to 10 %, 50 warm-up updates, 8,300 updates (66,400 examples). Final panel: held by kind `passage` 0.0319 gold NLL/token (48 items), `lookup` 0.0579 (48), `reconstruct` 0.0430 (504), `rag_qa` 0.610 (128). Earlier notes: final mid .188 (stop-inclusive), passage_1-4k .067, bundle .014; "most advanced checkpoint on everything Q/A uses" (archived state, Oct 4–5). Whether its intervention settings are frequency −1 is to be read from the checkpoint config before the probe (the recipe file does not list them under an "intervention" key). Adjacent candidates named in the archive: the dense-reconstruction continuations from it (`dense_pow1_tgt48_256_lc3i`, the control `ctrl_pow0_tgt48_256_run`); pick by held passage/lookup loss when detailing P.

Compatibility risk: `lc3i` was written Oct 2 by the `long_recon_v1/source` tree on ORCD; the current workspace model code has since gained key/value transfer and trainer options (additive). Run the probe from the ORCD source tree that wrote it, or verify the pair loads in the current tree before relying on it.

### Prior evidence on rows against additions

- Pathway diagnostic on the `qa_sft` final pair (2B → 9B, trained with both; archived state): `rag_unexposed` loss full .292, rows_only .426, deltas_only .332, none .507; answer tokens deltas_only .79 / rows_only 1.02 / rows_shuffled .95; reading then: "residual deltas carry nearly everything (for reconstruction almost all of it), the row embeddings add little on their own, and row ORDER matters".
- 2B → 4B from-scratch ablation `G4_in` (rows at the input only, reader adapter on): same panel loss as the deltas arm at update 500 (1.990 against 1.976), stopped at ≈1,073 during the cold-start plateau; its records are no longer on CSAIL. Not informative.
- Kit arms that exist for a from-scratch variant if wanted: `run_recon.py` `G4_in`, `G4_in_frozen`, `G4_frozen` (2B → 4B; `reader_model` can be set to the 9B).

### Probe P as intended

Readings on `lc3i`'s held panel (loss per token; passage, lookup, reconstruct): R0 as trained; R1 additions off at read time (the DIAG tool's `rows_only` pathway). Then two continuations of ≈1,000 updates on its own packet, same recipe and seed: T0 as trained; T1 with additions removed (input rows only). Decision rule to propose when detailing: rows-only "enough" if T1's passage and lookup losses end within about 1.5× of T0's; otherwise the engine step is on the critical path. Site: ORCD (pair, packet and source tree are there) unless the pair is first copied to CSAIL.

### Milestones, version 3 (all TBD until `pcomp3.start.*`)

P (probe) and M0 (pilot, resume benchmark, figure script) first and side by side → E (engine; order set by P; cache identity from the first cut) → M1 (pool ≈3,700 runs, frozen 200) → M2 (tune from `lc3i`, two arms) → M3 (final) → M4 (extras on request).

## Go-ahead (Oct 13 17:51Z)

Answers (`PLAN.interactions.json`, verbatim): `pcomp3.reader.keep_adapter`: true; `pcomp3.reader.feedback`: "untuned 9B is what does the compaction.\nTasks with AC always train the reader."; `pcomp3.start.probe`: true; `pcomp3.start.pilot`: true; `pcomp3.start.engine_design`: false. Chat: "See my interactions again. You can start now. Add this note: whenever you find yourself waiting, do these two things in order of priority: (1) cancel the old lower priority jobs, and (2) allocate max 2 ec2 spot instances with 2 rtx gpus each."

Integration: every AC variant trains the reader adapter with the encoder (no untouched-reader series); the untuned 9B writes the text summary and runs every text baseline; reference point added = adapter-carrying reader with the full text. Milestones P and M0 are `Implementing`; E's detailed plan is not started. All `pcomp3.*` keys retired and removed.

## Changes after the go-ahead (Oct 13)

- Evaluation set: **100 prefixes for now** (user, by 19:12Z: "let's go for 100 for now, not 200. Use the reamining for training if needed."). Taken in `evaluation_order` from the qualifying runs; pool stage = evaluation tasks 0–599 at the pilot's 17 % yield (41 prefixes from the first 236 runs). Left-over evaluation-side runs may serve training only when their repository holds none of the 100 (repository split stays clean). Interval for 100 prefixes ≈ ±9.5 points at 35 %.
- Continuation budget stays 30,000 (a 32k request was withdrawn the same hour: 20k + 30k = the 50k bound).
- Pool runs after the first 240 use `--max-duration 2700` (the first 240 all finished inside 40 minutes of wall time under the 5,400 s limit).
- Continuation text for training: **the untuned 9B's own full-context continuation of the 27B prefix**, not the 27B's (user, by 19:16Z: "If we are training on 27B text, then we end with more verbose text. Meaning the synthesis might actually need to be done with 9B."). Measured on the 41 pilot prefixes (`kit/continuation_stats.py`, steps after the cut): 27B own continuation 28 solved, 6.9 turns, 5,696 tokens written (4,839 reasoning), 2,848 tool output, 8,544 in all, 831 written per turn; 9B with the whole prefix 26 solved, 13.0 turns, 4,480 written (3,340 reasoning), 3,981 tool output, 8,461 in all, 346 per turn. The prefix writer stays the 27B (settled). Synthesis = the resume benchmark's `full` variant on training-side prefixes (minutes per hundred prefixes); the 27B continuation is kept from the pool for free.

## Code map for the training step (read-only search, Oct 13 19:36Z)

`B/` = `IB/ARTIFACTS/FIRST_TASK_AWARE_FINETUNING/bench/`, `A/` = `activation/`.

- Trainer of the `lc3i` pair: `B/run_channel_mixture.py` (`train` at :856; phases prepare | panel | derive | train), launched by `B/launch_channel_mixture.sbatch`. Packet folder = `recipe.json` (`training_config` = kwargs of `ActivationContextTrainingConfig`, `initial_pair`, `held`), `items.pkl` (`{'train': [ActivationContextTrainingItem], 'held': []}`), `prepared_examples.pkl`, `prepared.json`, `task_panel/` (items need `info['panel_group']`). Packet writer to copy: `B/prepare_long_recon.py` (:330–343).
- Item (`A/ac_model/ac_model_training.py:58`): `in_context_messages` (raw-text view, the KL teacher's input), `ac_messages` (student view with activation parts inline), `in_context_start` / `ac_start`, `assistant_token_ids` (one list per assistant message after start = the loss targets), `tools`, `info` (`family`, `ratio`, `chat_template_kwargs`). Everything outside a part goes through the reader's chat template with tools, so turns, tool calls and tool results are supported; tool and user messages between targets are context.
- Loss: `training_config.loss_kind` = `'sft'` or `'kl'` (top-32 KL to a frozen snapshot of the starting reader on `in_context_messages`; forces one example per micro-batch; multi-GPU KL unverified). Alternative: SFT plus `full_context_match_weight` with `full_context_match_kinds: ["compaction"]`.
- Closest existing builder: `B/run_trajectory_tasks.py --task compaction`, `build_items` :221–249 (prefix part = `ac_part(prefix, ac, ratio, kind="compaction")` + tools; student = system + task + [user: part + `COMPACTION_INSTRUCTIONS`]; the run behind "0.787 against 0.811 for recent text", `TRAJECTORY_INSIGHTS.md:128`). It read public traces; **no script builds mid-run continuation items from a rollout cache**. Helpers: `ActivationContextStudyGenerator.items_from_run_results` (`A/ac_model/ac_model_study.py:314`), `_run_tools` (:435), `A/bench/canonical_training/export_agent_teacher_segments.py` (`ToolDefinitions` :81).
- To write: `prepare_agent_continuation.py` (from the resume `full` caches: head, prefix as the part, the 9B's recorded continuation tokens as targets, ratio drawn in [1/20, 1/5], held panel by repository with a recent-text reference, two recipes). `max_view_rows` is 4,096, so a 20,000-token prefix at 1/5 sits at the clamp.
- Rollout with the pair: `B/run_qa.py:26 harness(ac_path, reader_path, …)` registers the reader adapter (rank 128) and the AC model (`min_view_rows=2`, `intervention_frequency=-1`, `mixer_layers=2`); the engine serves the adapter only after `module_manager.exchange_lora(NAME, READER)`; then `AgentConfig(lora_name=…, ac_model_name=…, ac_compaction_ratio=…)`. **The vLLM path carries input rows only**: a model with per-layer additions raises in `encode_batch` (`A/ac_model/ac_model.py:468`), so `lc3i` as trained cannot run in the stock agent; the only rollout path with additions is the HF one (`DeepHFAgent`, `B/probe_live_compaction.py:32`). This is why probe P decides the engine step.
- Teacher-view detail: the trainer re-renders the prefix's assistant turns through the template with reasoning as text, while the benchmark's `full` variant feeds the 27B's tokens verbatim.

## M0 result (Oct 13 20:00Z) — status Review

41 prefixes (`prefixes_eval_v1` on `pc-b` = first cut of `pc_pool_eval_v1`, 236 runs), untuned 9B, thinking on, budget 30,000, one sample per prefix, caches `pc_resume_v1_<variant>` (local copy `IB/TMP/PAPER_COMPACTION/node_sync/pc-b_SYNC/`). Size | solved | answered | mean continuation tokens (unanswered = 30,000) | mean seconds:
full 19,759 | 26 | 39 | 8,743 | 97; summary 413 | 23 | 32 | 15,702 | 118; recent_4000 3,996 | 23 | 27 | 19,327 | 167; recent_2000 1,991 | 15 | 19 | 23,904 | 227; recent_1000 992 | 11 | 14 | 24,642 | 203; none 32 | 10 | 14 | 27,455 | 199. summary_write: 41 of 41 written, compact turn 15.0 s and 528 output tokens mean (41 concurrent runs), 757 continuation tokens mean.
Pool yield: 236 runs → 41 prefixes; 181 under 20k (170 solved); 12 at the cap; 2 with nothing after the cut. M1 status Implementing (pool chain on `pc-a`).

## M2 detail (Oct 13 20:20Z) — status Planning, data preparation running under the go-ahead

- Example: student view = head + [user: AC part (head + prefix, tool definitions) + compaction instructions] + continuation; targets = the untuned 9B's sampled tokens for every continuation assistant turn (9B, thinking on, full prefix in view, budget 30,000). Verified by the builder agent: the `ac` variant's live prompt equals the student view token for token.
- Training runs: training side only (2,676 tasks, 615 repositories). Cuts: window rule as the benchmark, plus the short rule (`cut_prefixes.py --short-lower 6000 --short-fraction 0.6`: runs ending in [6k, 20k) cut at the first non-assistant boundary at or past 60 % of the run). Reason: at 545 evaluation-side runs, 416 ended under 20k and 102 were window prefixes. Evaluation prefixes are window-only. Evaluation-side left-overs are not used for training (keeps the 100 -> 200 extension open: the first 600 evaluation tasks already touch most of the 189 evaluation repositories).
- Continuations kept: solved only (default of `export_continuation_material.py`).
- Sizes: `--nominal-prefix-tokens 0` (rows = own prefix tokens x r), r log-uniform in [1/20, 1/5], `--ratios-per-prefix 4`.
- First-wave arms on the two group H200s, both `deltas_off_training=true` (rows only: the only form the stock engine serves), encoder and reader trained at 2e-5: A `sft`, B `kl`. Start: the probe's `lc3i_rowsonly_1000` final pair if it ends clearly below the untrained rows-only reading, else `lc3i`. `checkpoint_every_updates=null` (scratch is tight).
- Tree: packets are built and trained with `probe_p_v1/source_p` (has every name the builder imports, checked by grep and by a CPU test build, job 2651134: KL packet all checks passed; the checker itself needed `getattr` for the match-loss field the older tree lacks, fixed in `kit_v2`, job 2651149). `sft_match` needs the workspace tree's `full_context_match_weight`; not in the first wave.
- Cluster layout: `/data/scratch/ngom/activation/paper_compaction_v1/{kit,kit_v2,material,packets,jobs,logs}`; a packet folder is the run folder (`RUN=packets/<name>/<loss>` for `jobs/train_continuation.sbatch`).
- Selection: live resumed runs (`ac_1000`, `ac_2000`, `ac_4000`) on held training-repository prefixes; never the 100.

## M1 progress (Oct 13 20:32Z)

- Evaluation pool complete: 600 runs, 488 solved, 425 under 20k, 38 at the cap, 130 window prefixes (79 where the 27B solved). Frozen list `prefixes_eval_100` (first 100 in evaluation order; index sha256 3d5e70afd63f4a69...; 46 py, 36 java, 18 kt; 61 repositories; 11 official-test tasks). 30 left over (12 from repositories not among the kept), unused.
- Training pool: `split_train_shuffled.json` (seed 20261013) because `split.json` lists training tasks by language; `pc-a` tasks 0-1337, `pc-b` from 1338 in chunks between 9B windows.
- Kotlin: 78 training tasks of 2,676 against 18 evaluation prefixes.

## Text series on the frozen 100 (Oct 13 21:23Z)

Tag `pc_resume_v1` on `pc-b`; caches pulled to `IB/TMP/PAPER_COMPACTION/node_sync/pc-b_SYNC`; figure `figs/frozen100_text_fig.{png,pdf,json}` (`make_fig_compaction.py --index prefixes_eval_100/index.json`). Size | solved | answered | mean continuation tokens: full 19,688 | 55 | 91 | 11,269; 9B summary 426 | 45 | 67 | 18,776; recent 4,000: 3,992 | 38 | 55 | 20,948; recent 2,000: 1,975 | 30 | 44 | 23,885; recent 1,000: 978 | 21 | 32 | 24,669; nothing 32 | 16 | 24 | 28,259. Summary write 100 of 100, 17.8 s and 518 output tokens per compact turn under load.

## User call (Oct 13 22:36Z): rows only is the first attempt

Verbatim: "Let's tentatively consider that input-only, when it's done training, is good enough as a first attempt. Unless even 1/5th looks very bad (let alone 1/10th and 1/20th), gently push back if I start asking for minor improvements." Integration: P settled for the MVP (probe at 7,000 examples: rows only 0.0983 | 0.1074 | 0.0931 against 0.0369 | 0.0499 | 0.0565 as trained); E after the MVP; first wave arms: A `sft` and B `kl` from the probe's rows-only pair (H200s), C `sft` from `lc3i` (GH200), all `deltas_off_training=true`; no arm with additions.

## M2 first wave (Oct 13 23:18Z) — status Implementing

Material `train_w1.jsonl`: 314 solved 9B continuations of 468 training prefixes (169 window, 299 short). Packets `w1_from_rowsonly/{sft,kl}`, `w1_from_lc3i`: 2,400 training items (300 runs × 8 sizes), 42 held (14 runs × 3 sizes, 12 held repositories), 300 updates, one pass, rows only. Jobs: A 2652901 (H200), C 2652902 (GH200), B after the probe's as-trained arm ends. Drift from the M2 card: 8 sizes over one pass instead of 4 sizes per run (storage: one 7 GB save per pass).


## First wave result, arm choice, first full figure (Oct 14 03:41Z) — M2 Implementing, M3 Implementing

- **First wave** (300 training runs × 8 sizes, 300 updates, rows only, rates 2e-5 | 2e-5). Held loss per token, 1/20 | 1/10 | 1/5 (no context 0.972, full context 0.790): arm A (cross-entropy from the probe's rows-only pair, job 2652901) 0.994 | 1.008 | 1.025 → 0.855 | 0.834 | 0.822; arm C (cross-entropy from `lc3i`, job 2652902) 0.969 | 0.968 | 0.965 → 0.862 | 0.841 | 0.828. Arm B (KL, job 2653093) failed at its first step ("no stored teacher targets", cluster tree) and was dropped.
- **Development prefixes** (`prefixes_dev_w1`, 50 window prefixes, 45 training-side repositories wave 1 never trained on; 29 py, 21 java). Text references, solved | answered | mean continuation tokens: full 31 | 45 | 12,453; 9B summary 20 | 31 | 19,168; nothing 8 | 12 | 27,853. AC at 4,000 | 2,000 | 1,000 rows: arm A at 25 % 24 | 44 | 9,883, 17 | 40 | 12,957, 17 | 32 | 17,889; arm C at 25 % 19 | 44 | 10,616, 13 | 38 | 13,181, 7 | 24 | 22,963; arm A at 50 % 27 | 40 | 14,577, 18 | 35 | 17,178, 18 | 26 | 21,435. **Arm A chosen.**
- **Frozen 100, arm A's final pair** (`pairs/w1A_final`; tag `pc_resume_v1`; run 03:06Z–03:38Z on `pc-b`): 4,000 rows 44 | 86 | 11,984; 2,000 rows 34 | 78 | 12,632; 1,000 rows 30 | 69 | 16,837. Text series as before (full 55 | 91 | 11,269; summary 45 | 67 | 18,776; recent 38 | 55 | 20,948, 30 | 44 | 23,885, 21 | 32 | 24,669; nothing 16 | 24 | 28,259). Figure `IB/TMP/PAPER_COMPACTION/figs/frozen100_w1A_fig.{png,pdf,json}`; panel (c) there is from many-at-once runs and is to be replaced by the one-at-a-time pass (tag `pc_timing_v1`).
- **Reading**: AC rows beat truncation at equal size on every metric (solved intervals overlap at single sizes; answered and token gaps do not); tie the 9B summary on solved only at 1/5 (the summary is 426 tokens) while finishing in far fewer tokens; lose 11 solved to full context at 1/5.
- **Second wave**: material `train_w2.jsonl` (1,066 training prefixes → 712 solved); `train_w2_new.jsonl` = the 398 not in wave 1, 353 usable after excluding evaluation and development repositories (`excluded_eval100_dev_w1_index.json`). Packet `packets/w2new_from_w1A` (338 training runs, 188 repositories, 2,704 items, 338 updates; 15 held runs, 12 held repositories), starting pair `pairs/w1A_final`, job 2657579 on an H200 since 03:14Z.
- **Third wave (data)**: `pc-a` pool complete (all 2,676 training tasks pooled over both nodes); 512 prefixes not in `prefixes_train_w2` → `prefixes_train_w3`; 9B continuations running → `material/train_w3.jsonl`.
- Kit additions: `eval100_ac_node.sh`, `slim_rollouts.py`, `window4_a.sh`, `merge_prefix_folders.py --exclude-index`.
- **Paired differences on the frozen 100** (same prefixes; 95 % bootstrap interval over prefixes; solved and answered in points, tokens in continuation tokens):

  | A − B | solved by A only \| B only | solved | answered | tokens |
  |---|---|---|---|---|
  | AC 4,000 − recent 4,000 | 20 \| 14 | +6 [−5, +18] | +31 [+21, +41] | −8,964 [−11,322, −6,750] |
  | AC 2,000 − recent 2,000 | 20 \| 16 | +4 [−8, +16] | +34 [+23, +45] | −11,252 [−13,518, −8,994] |
  | AC 1,000 − recent 1,000 | 18 \| 9 | +9 [−1, +19] | +37 [+26, +48] | −7,832 [−10,048, −5,654] |
  | AC 4,000 − summary | 14 \| 15 | −1 [−12, +10] | +19 [+9, +29] | −6,792 [−9,292, −4,309] |
  | AC 2,000 − summary | 12 \| 23 | −11 [−22, +1] | +11 [+0, +22] | −6,144 [−8,556, −3,762] |
  | AC 1,000 − summary | 8 \| 23 | −15 [−26, −5] | +2 [−9, +13] | −1,939 [−4,401, +560] |
  | AC 4,000 − full | 7 \| 18 | −11 [−20, −1] | −5 [−13, +2] | +716 [−1,314, +2,725] |
  | summary − full | 5 \| 15 | −10 [−18, −2] | −24 [−33, −14] | +7,508 [+5,543, +9,476] |

  So with 100 prefixes: the solved gain over truncation is not established at any single size; the answered and token gains are. AC at 1/5 and the summary are both about 10 solved under full context; AC at 1/5 answers as often as full context and costs the same tokens, the summary does not. Solved by language (py 46 | java 36 | kt 18): full 25 | 20 | 10; summary 21 | 18 | 6; AC 4,000 21 | 16 | 7; recent 4,000 15 | 14 | 9.
- **Panel (c), one-at-a-time pass** (tag `pc_timing_v1`, 30 frozen prefixes, one run per GPU, 04:05Z): summary write 7.4 s per compact turn (510 output tokens); encoder pass 1.34 s mean over 90 encodes (4,000 rows: mean 1.89, median 1.42, max 6.25 — first encode; 2,000: 1.06 | 0.94; 1,000: 1.07 | 0.93). Figure redrawn with `--timing-tag pc_timing_v1`.
- **Combined waves** (04:21Z): packet `packets/w23new_from_w1A` (608 training runs, 271 repositories, 4,864 items, 608 updates; 37 held runs), job 2658052 on a GH200 from `pairs/w1A_final`, beside the second wave alone (job 2657579, H200).
- **Second-seed 27B pool** for more trajectories: `pc_pool_train_s1`, seed 1, `pc-a` tasks 0– in shuffled order (03:56Z), `pc-b` tasks 2007–2675 (04:26Z); its prefixes need a task-id suffix before training (same task ids as the first pool).
- **Results draft** (04:27Z): `compaction_results_draft.tex` (this folder), from the first-wave figure; lists the snapshot sentences the figure refutes; snapshot untouched. Figure copied to `figs/` here.

## More trajectories on the development prefixes (Oct 14 08:47Z)

- Development prefixes (50), solved | answered | mean continuation tokens at 4,000 | 2,000 | 1,000 rows: arm A final (300 runs) 30 | 44 | 11,053, 20 | 42 | 13,509, 17 | 37 | 17,021; second wave final (+338) 22 | 37 | 13,232, 28 | 40 | 14,496, 21 | 36 | 15,730; combined waves at 50 % (update 304 of 608) 30 | 45 | 10,786, 26 | 38 | 14,239, 20 | 39 | 14,314.
- Paired against arm A (solved by one only): combined 50 % 3 | 3, 10 | 4, 8 | 5; second wave 0 | 8, 14 | 6, 10 | 6. Discordance this large means 50 prefixes cannot resolve a few-task gain.
- Rule fixed now, before any further frozen-100 run: the figure uses the combined waves' final pair unless it reads worse than arm A on the development prefixes; one pair only goes onto the frozen 100; never chosen on frozen-100 numbers.

## MVP accepted; hill-climb opened (Oct 14 09:06Z) — M3 Completed, M5 Implementing

- User (≈09:05Z, verbatim): "Ok mark the current results as enough for MVP: 1/5 has ~same resolution rate at lower continuations." and "From now, consider anything hillclimbing I ask on this specific task "moderately high" priority. As in, try to keep it in progress on at least 1 GPU, but only use more gpus if no other task is pending. I'd like to push the at least 1/10th pair nearer summary."
- Facts read for the answer: rates 2e-5 (encoder) | 2e-5 (reader LoRA), cosine to 10 %, warm-up 50, 8 examples per update (recipes of `w1_from_rowsonly/sft` and `w23new_from_w1A`); starting pair `rowsonly_1000` = `lc3i_rowsonly_1000` (1,000 updates of long-context reconstruction at 1/10, rows only, 2e-5, from `lc3i`). Training prefixes (all waves, excluding evaluation/development repositories): 956 runs, median 8.7k prefix tokens, 283 at ≥ 19k.
- H1 started: material `train_window_all.jsonl` (320 runs with prefix ≥ 19k tokens from waves 1–3, before repository exclusion), packet `win_from_rowsonly` from `pairs/rowsonly_1000`, build job 2661209; trains on the H200.
- Measurement from now: 2,000 rows only, two seeds, 50 development prefixes, arm A's final as baseline.
- CSAIL scratch: deleted further mid-run snapshots (50–70 %) of the nine finished `paper_v1/hc_*` runs (80–90 % and finals kept) → 74 GB free.
- User (≈09:20Z, verbatim): "(for compaction hillclimbing, keep training at the different ratios, but eval at 1/10th is fine until we reach a good point. One other candidate should be using the 27B text loss btw, if the previous ones don't pan out)". Added H4 (27B continuation text as the target) after H0–H2; training sizes stay 1/20–1/5; candidates read at 1/10 only.
- H2 (≈09:33Z): root cause of the KL failure — `run_channel_mixture.py train` evaluates the frozen references before `train()` stores teacher targets, and `eval()` defaults to the stored teacher. Fix without touching the tree: wrapper `paper_compaction_v1/kl_v1/kl_channel_mixture.py` (pre-training evals use the current teacher; teacher = base 9B with no adapter, `PC_KL_TEACHER=base`; refuses block-wise target storage), jobs `launch_continuation_kl.sbatch`, `train_continuation_kl{,_gh200}_v2.sbatch`, check `check_kl_v1.sbatch`; packets need `--train teacher_cache_block_examples=null` (top-32 targets stored once, ≈0.5 GB). CPU check passed on `packets/kl_test_v3`. Notes: `IB/TMP/PAPER_COMPACTION/cluster/H2_KL_NOTES.md`. Arm: `packets/w1_kl_from_rowsonly` = arm A's build with `--loss kl` (build 2661580, check 2661581), trains on the free GH200 slot.

### M5 readings (Oct 14 12:40Z)
- Baseline arm A (`w1A_final`), development, 2,000 rows: seed 0 20 | 42 | 13,509; seed 1 20 | 35 | 15,485 (solved | answered | tokens).
- H0 (`w23A_final`): seed 0 22 | 38 | 14,300; seed 1 28 | 40 | 14,881. 50 vs 40 of 100; per prefix better 15, worse 8 (sign p 0.21). Arm A disagrees with itself on 18 of 50 prefixes, H0 on 10.
- H1 trained (job 2661443, 263 updates); last held nll/token 0.989 | 0.963 | 0.947 (1/20 | 1/10 | 1/5) on its own 20 held runs (not comparable to other packets' panels). Dev reading started ≈12:50Z.
- H3 redefined: second-seed material (990 solved continuations: 739 pc-a, 251 pc-b; median prefix ≈11.5k, same cuts as before, so not "evaluation length"), 780 drawn for the 12 h limit, trained on top of H0 (`packets/s1_from_w23A`, build job 2663087).
- H1 dev (13:03Z): 22 | 41 | 12,811 and 21 | 40 | 13,368 → 43 of 100; vs arm A 11/9 (p 0.82), vs H0 9/13 (p 0.52).
- H0 frozen-100 (14:01Z): AC 1/5 | 1/10 | 1/20 = 47 | 47 | 39 solved (arm A 44 | 34 | 30); now the figure's pair (`figs/frozen100_h0_fig.*`, timing panel linear since 14:30Z).
- H2 dev (14:20Z): 21 + 21 = 42 of 100. H3: training job 2663534 failed at start on the storage preflight (32 GiB needed, 21 free); parked.
- 9B-written prefixes (user ≈13:58Z, medium-high): `kit/chain9v2_b.sh` on pc-b GPU 1 (first try failed: 9B engine length 52k < 50k cap + 8k turn).
- **Rule change (user, Oct 14 ≈16:30Z):** "Oh the training only used new repositories? I think that's unnecessary harmful to us to begin with: with are pitching domain specialization afterall. Drop the rule stating this." From now, training excludes only the exact evaluation and development tasks (task level), not their repositories; evaluation-side tasks outside the frozen sets may train. Kit: `prepare_agent_continuation.py --exclude-tasks <index.json ...> --sides training evaluation` (new option; goes to CSAIL as a new kit folder when first used). Root canon updated. All packets so far (arm A, H0–H3) used the repository rule.
- Intervals (asked ≈16:20Z): 95 % half-width on solved near 47 %, from arm A's dev seed agreement (between-prefix variance ≈ 0.069, within-prefix ≈ 0.18): 100×1 ±10; 100×3 ±7; 200×1 ±7; 200×3 ±5.0; 200×5 ±4.5; floor at 200 prefixes ±3.6. Seeds are averaged per prefix (mean@k).

## 9B-written prefixes at a 50k limit; the MVP argument (Oct 14 ≈17:40Z)
- User (≈17:15Z, verbatim): "One thing to do in the 200 larger run (for paper justifications). Run with a limit 50000 (which might require redoing some tasks), but at first, report as if the limit was 30k. I want to see how much is caused by our limit, which is a fair limit but I still want to know."
- Done: 200 prefixes × 3 seeds at a 50,000-token continuation limit (`kit/run9L_job.sh`, tags `pc_r9b50_{f,n}_s{0,1,2}`); the 30k reading is a replay of the same runs (`kit/view_budget.py`, which reproduces the 30k runs' summaries exactly), so 30k and 50k differ only by the limit.
- **The MVP argument (user, ≈17:40Z, adopting my wording verbatim):** "What 50k won't erase is the cost. A variant that catches up does it by spending more continuation tokens. So if solved converges, the paper's argument shifts to tokens: AC and full reach the answer early, the others pay to rebuild the context. Only a solved-rate gap that persists at 50k would mean the context itself is missing." The user adds: "(Also more defensible than explaining failures by a 30k limit)."
- Consequence for the write-up: the headline reports solved and continuation tokens at 50k side by side; the 30k reading is the secondary view. Failures are not explained by the limit.
- First 30k reading (seed 0, first 100 prefixes, 30k runs `pc_resume9_v2`): full 60 | 79 | 16,396; AC 1/5 61 | 77 | 17,384; AC 1/10 57 | 72 | 18,113; AC 1/20 53 | 65 | 19,449; recent 1/5 55 | 67 | 20,556; none 38 | 44 | 25,681 (solved | answered | tokens).

## Overnight hill-climb (user away ≈12–16 h from ≈22:10Z, Oct 14)
- User, verbatim: "any general hillclimbing you+some proposers can think of in general for either compaction or synthesis. If I am off 12h (possibly up to 16h even), I hope we have hillclimbed numbers by then, so I can write that in. (use many gpus, more/less data, maybe faster turnarounds etc as you see fit; up to you)."
- Fraction scan result (seed 0, 200 tasks): the summary loses most at the **65 % cut** (30k: full − summary solved +17, 27 vs 10, p 0.01; 50k: +3, tokens −3,025). AC (H0) on the 65 % and 80 % cuts is queued on pc-a GPU 1.
- Schedule (proposer ranking, merged):
  1. EC2 (main thread): 200×3 at 50k (`pc_r9d50_s{0,1,2}`), then AC on the 65/80 % cuts. If AC at 65 % beats the summary, the remaining seeds move to the 65 % cut.
  2. H3 as built (`packets/s1_from_w23A`) on the CSAIL H200 pair once the TQA SFT ends (≈00:15Z).
  3. 9B dev set: the 75 usable left-over pool-v3 prefixes (`prefixes_dev9_75`). References: full, summary, AC 1/10 and 1/5, 2 seeds.
  4. Pool→material exporter. **H5** = H0 + the 155 solved 9B pool-v3 runs outside the eval, dev and frozen sets.
  5. **H6**: a new 9B training pool on EC2 after the 200×3 (≈03:00Z), then H0 + ≈700 runs.
- Decision rule on the dev set: tokens are the main metric. Promote an arm at tokens −1.0k with solved within 4, or at solved +10.
- Rates 2e-5 / 2e-5. GH200 at most 2, shared with synthesis. H200 grouped at most 2.

## Final figure spec and the H7 round (Oct 15 ≈12:15Z)
- User (verbatim): "from now on, when making the final compaction plot: one token count plot should show continuation quantiles, and one solving plot should be a top difficulty one. TBD whether top 50% or top 25% or something".
- Difficulty subsets: tasks ranked by the summary−full continuation-token gap. Clean version: rank on separate selection seeds (full + summary only), report every arm on seeds 0–4. Ranking and reading on the same seeds overstates the gap (regression to the mean), so it is not used for the figure.
- Green light (≈12:00Z): (a) hard-subset readout with clean selection seeds; (b) H7 = more, better-selected continuation targets (k samples per prefix, shortest correct), after a 30–45 min parallelism probe and 1–2 proposers refining selection and count; (c) H0/H5 at 1/20 on eval-200. Synthesis gets the same treatment (more, better-selected targets; Musique first).

## Per-ratio specialists (user request, Oct 16 ≈10:10Z)

The request, verbatim: "Small addition to the experiments: a per-desired ratio fine-tuning. So 1/10th is specialized at producing 1/10th, etc. You can pick whichever shared checkpoint you want."

- **Shared checkpoint:** h5_final, H7S's own init. Starting from H7S would have given the specialists a second epoch on the same lines, so a gain could not be credited to specialization.
- **Specialists:** R5, R10 and R20, each one epoch from h5_final on all 2,843 H7 material lines at exactly one ratio. Same targets and rates (2e-5/2e-5).
  - H7S itself trained on a continuous log-uniform r in [1/20, 1/5], so at any one ratio it saw only about a third of the material nearby.
  - The comparison is specialist against H7S at the same ratio: equal material, equal epochs, single ratio against mixed.
- **Compute:** R10 and R20 run first, as two GH200 jobs; R5 runs after.
- **Evaluation:** each specialist at its own ratio, seeds 2–4, read in the held-out quantile table against H7S at the same ratio.
