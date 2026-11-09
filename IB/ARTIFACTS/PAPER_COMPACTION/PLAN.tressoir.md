# Paper MVP — agent trajectory compaction

Plan for the paper's "Agent Trajectory Compaction" subsection (`IB/evaluation_section_snapshot.tex`), version 3 with your go-ahead of Oct 13 17:51Z folded in. All shape decisions are settled. Two pieces of work are running: probe P (are input rows alone enough?) and pilot M0 (the resume benchmark and the first figure on untrained variants). Later milestones stay at intent level until those two report.

## Executive Summary

### Goal

Fill the figure `fig:evalCompaction` with real numbers as fast as possible, on two GPUs (2×H200 or 2×GH200). The figure has to be able to confirm or refute the five sentences the snapshot asserts:

| # | Sentence in the snapshot | What the figure must show |
|---|---|---|
| 1 | AC rows preserve more information than text | AC line above the recent-text line and the text-summary point at equal size |
| 2 | The agent finishes at higher quality and speed, matching or exceeding an untuned reader with full context | AC at the mild ratio inside the full-context interval, with no more continuation tokens |
| 3 | At 1/10 compression it is as good as seeing everything | Same panel, the 1/10 point |
| 4 | AC rows are an order of magnitude faster to compute than a text summary | A timing panel: summary decode against encoder pass |
| 5 | AC rows combine well with truncation | A series "AC rows + 1k recent text" above plain AC rows at equal size |

### What changed since version 2

| Topic | Version 2 | Now |
|---|---|---|
| Window | 20k or 16k | **20k–50k**, settled |
| Split | open | all three languages, our own repository split, settled |
| Training start | from scratch, with math and search traces to make up the volume | **tune from the best reconstruction pair** (`lc3i`); math and search traces only if that is not enough |
| Engine shortcut | first cut with prefix-cache reuse off | **dropped**: the prefix cache stays correct and on from the first cut |
| First experiment | the pilot | **a probe first: are input rows alone enough?** If yes, the paper's rollouts run on today's engine and the engine step stops blocking |

### The experiment in one picture

Data flow behind the figure in `IB/evaluation_section_snapshot.tex`:

```text
 27B agent, medium reasoning, single agent, bug localization (Java, Python, Kotlin)
        │  run every task once, no compaction, stop at 50k tokens
        ▼
 keep runs that passed 20k and ended under 50k  ──►  cut each at the 20k turn boundary (the shared prefix)
        │
        ├── evaluation repositories: 100 prefixes ──►  untuned 9B reader resumes each prefix once per variant,
        │                                              same tools, 30k continuation budget
        │                                              → solved?  → continuation tokens used
        └── training repositories:                  ──►  tune the reconstruction-trained 2B encoder so the reader,
            successful runs, cut at many points          given AC rows instead of the prefix, continues the same way
```

| Series in the figure | What replaces the 20k prefix | Sizes | Training |
|---|---|---|---|
| Full context | nothing; the reader sees the raw prefix | 20,000 tokens | none |
| No context | the task statement only | 0 | none |
| Recent text (truncation) | the most recent part of the prefix, as text | 4,000 / 2,000 / 1,000 tokens | none |
| Text summary by the untuned 9B | the harness's five-section summary | its own measured length | none |
| AC rows | rows from the 2B encoder, read by the 9B with its AC adapter | 4,000 / 2,000 / 1,000 rows | encoder and reader adapter, tuned from the reconstruction pair |
| AC rows + 1k recent text | 1,000 rows plus the last 1,000 tokens | one point, equal in size to 2,000 | same model |

Baselines, including the text summary, always run on the untuned Qwen 3.5 9B: the untuned 9B is what does the text compaction. Every AC variant trains the reader's adapter together with the encoder; there is no AC series with an untouched reader.

### Facts behind this version

- **Benchmark splits.** The benchmark names three sets, but I checked the task ids: in every language `dev` is exactly `train` plus `test`, with no overlap (Java 2,522 = 2,472 + 50; Python 4,339 = 4,289 + 50; Kotlin 618 = 568 + 50). So there are two real sets: 7,329 training tasks and 150 test tasks.
- **Yield at 20k.** In the existing single-agent 27B runs, 24 % end inside 20k–50k and the 27B solves 39 % of those. The 150 test tasks therefore give roughly 25 prefixes; the other 175 come from repositories we hold out of `train`.
- **The reconstruction pair.** `lc3i` is on ORCD: 2B encoder → 9B reader, trained at 1/10 for 8,300 updates with a reader adapter; held whole-passage loss 0.032 per token, lookup 0.058. It was trained at one ratio only, so 1/5 and 1/20 are new to it.
- **Input rows against per-layer additions.** No clean comparison exists. One diagnostic on a pair trained with both found the additions carrying most of the content (question loss 0.292 with both, 0.426 with rows only, 0.332 with additions only, 0.507 with nothing). That says what such a pair relies on, not what a pair trained on rows alone can reach. The probe answers that.

### What exists and what is missing

| Piece | Status |
|---|---|
| 27B rollout generator, segment export, token checks | exists |
| Reconstruction pair `lc3i` (2B → 9B) and its training kit | exists on ORCD |
| vLLM serving input rows with prefix caching | exists (the engine in use today); one cache-identity detail for mixed token-and-row prompts is to be re-checked |
| vLLM fork on your GitHub | exists: `amlatyrngom/vllm`, branch `ac-interventions-v0.28.0`, still identical to upstream |
| Per-layer AC inputs inside vLLM | missing; needed for the paper only if the probe says rows alone fall short, and for the systems claim in any case |
| **A benchmark that resumes a real task from a replaced prefix and scores it** | **missing** |
| Single-agent pool at scale, repository split, 9B baselines on it | missing |
| The figure script | missing; written first, before any training |

### Boundaries

- Single agent only; 2B encoder and 9B reader; token counts, not wall-clock, except the timing panel.
- No sweeps for one or two points.
- Everything that was running before this thread is lowest priority and gives way.

## Requested Decisions

**Accepted so far**

- **Prefix writer:** the 27B. **Text summary:** the untuned 9B's, built first; the 27B's summary on request.
- **Training target:** continuation matching (canon updated).
- **AC form:** the existing 2B → 9B model.
- **Size grid:** 1/5, 1/10, 1/20 only for now.
- **Split:** all three languages; evaluation = the official test tasks plus repositories we hold out of `train`; no repository on both sides.
- **Window:** 20k–50k. Prefix 20,000 tokens, continuation budget 30,000.
- **Training data:** bug-localization training repositories, tuned from the best reconstruction pair. Math and search traces are the fallback if that start proves too thin.
- **Engine:** work in parallel; the prefix cache is never switched off as a shortcut.

- **Reader with AC rows (17:51Z):** the AC series starts from the reconstruction pair as it is, adapter included, and tasks with AC always train the reader. The figure gains one reference point: the same adapter-carrying reader with the full text.
- **Started (17:51Z):** probe P and pilot M0. The detailed engine plan is not started; it waits for the probe.

No decision is open. The next ones will come with the probe's result (does the engine step block the figure?) and the pilot's gate (is there a gap between full context and no context?).

## Milestones

M0 is in review; P and M1 are being carried out; M2 has its plan and its data preparation has started; the others are intent only.

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">P — Probe: are input rows alone enough?</span>
    <span class="card-oneliner">One cheap loss comparison that decides whether the engine step blocks the paper.</span>
    <span class="card-badge">Implementing</span>
  </summary>

#### Planning Overview

Start from the `lc3i` pair and its own held panel, so nothing new has to be built. Three readings: the pair as trained; the same pair with the per-layer additions switched off at read time; then two short continuations of about 1,000 updates, one as trained and one with the additions removed for good. The question is whether the rows-only arm returns to the level of the arm that keeps the additions.

- **If it does** (within a small margin on whole passages and lookups): the paper's compaction model uses input rows only and rolls out on today's engine, prefix cache included. The engine step moves out of the critical path and stays as the systems contribution.
- **If it does not:** the engine step is on the critical path, and it is built with cache identity from the first cut.

Loss per token is the only metric here, as you suggested; no question answering, no rollouts.

**Outcome so far (Oct 13 22:36Z).** After 7,000 examples of the 8,000, rows only reads 0.098 on passages, 0.107 on lookups and 0.093 on reconstruction, against 0.037, 0.050 and 0.057 as trained: about twice the loss, and levelling off. Your call of 22:36Z: rows only, once trained on continuation, is good enough as a first attempt. So the compaction model of the MVP serves input rows on today's engine, the engine step leaves the critical path, and the first training wave has no arm with additions.

#### Planned Changes

No product code. The probe runs from the source tree that wrote the pair, on ORCD, in new run folders beside it:

`~/orcd/pool/activation/long_recon_v1/` · new runs `lc3i_keep_1000` and `lc3i_rowsonly_1000`

```text
read 1   lc3i as trained, held panel                      (zero training)
read 2   lc3i with per-layer additions off, same panel    (zero training)
run  T0  lc3i continued about 1,000 updates as trained
run  T1  lc3i continued about 1,000 updates, additions removed for good
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M0 — Figure first, on untrained variants</span>
    <span class="card-oneliner">A pilot, the resume benchmark, and the real figure script.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report (Oct 13, 20:00Z)

**What landed.** The benchmark exists and the gate is passed. On 41 real prefixes (16 Python, 16 Java, 9 Kotlin; cut from the first 236 pool runs), the untuned 9B resumed each one under seven variants with a 30,000-token continuation budget:

| What replaces the 20k prefix | Size (tokens) | Solved | Answered | Mean continuation tokens |
|---|---|---|---|---|
| Full context | 19,759 | 26 of 41 (63 %) | 39 | 8,743 |
| 9B text summary | 413 | 23 of 41 (56 %) | 32 | 15,702 |
| Recent text, 1/5 | 3,996 | 23 of 41 (56 %) | 27 | 19,327 |
| Recent text, 1/10 | 1,991 | 15 of 41 (37 %) | 19 | 23,904 |
| Recent text, 1/20 | 992 | 11 of 41 (27 %) | 14 | 24,642 |
| Nothing | 32 | 10 of 41 (24 %) | 14 | 27,455 |

- **Gate.** Full context beats nothing by 39 points (the bar was about 10) and by 18,700 continuation tokens.
- **The summary is the baseline to beat.** At 413 tokens it matches recent text ten times its size, and its solved rate is within the interval of full context. Its weakness is cost: 1.8 times the continuation tokens of full context, and 15 seconds to write (528 tokens per summary, measured under load).
- **So the AC series has to win in panels (b) and (c)**: finish in fewer continuation tokens than the summary, and compact in a fraction of 15 seconds. A solved rate above the summary's would need about 100 prefixes to show (the interval at 41 is about ±15 points).
- **Yield.** 17 % of pool runs give a prefix, lower than the 24 % of the old campaign, because 77 % of runs end under 20k. 100 prefixes need about 600 evaluation-side runs.
- **Speed.** One variant over 41 prefixes takes 5 to 9 minutes on two RTX PRO 6000 cards; the engine start adds about 4 minutes once.
- **27B against 9B after the cut.** Same total (8,544 against 8,461 tokens) and nearly the same solved count (28 against 26), but the 27B writes 831 tokens per turn against 346. This is why the training continuation now comes from the 9B (Beat 3).

**Drifts, challenges, and unplanned steps.**

- The runner is `run_resume.py` with the program in `resume_program.py`, not one `resume_bench.py`; variants run in sequence on one engine.
- Resuming is at token level: the 27B's turns enter as its own tokens (the vocabularies are identical) and the text between turns is rendered by the reader. On every checked prefix the reader's rendering equals the 27B's record.
- The recency baseline cuts the oldest kept turn at its head (the call stays, its output and reasoning keep their ends); whole turns alone kept nothing at 1,000 tokens.
- Not done: the sandbox is not restored to its state at the cut. A resumed run starts from a fresh checkout; 5 of 41 prefixes contain a command that may have changed files.
- Sandboxed runs need a container runtime, which CSAIL lacks, so pool and resume runs live on the two EC2 nodes; CSAIL takes training only.
- Three launch faults cost about 50 minutes in all (a build tool missing from the path, a lost execute bit, a waiting shell that matched itself).
- The summary's time is read from runs made 41 at a time; a one-at-a-time timing pass for panel (c) is still owed.

**Focused actual diffs.** New files only, all in `IB/TMP/PAPER_COMPACTION/kit/`; contents omitted here.

```text
build_split.py, run_pool.py, pool_node.sh, pool_chain.sh   split and pool runs
cut_prefixes.py                                            window filter and cut
resume_program.py, run_resume.py, resume_node.sh           the benchmark (variants full, none, recent_N, summary_write, summary, ac_N)
continuation_stats.py                                      turns and tokens after the cut, per run
make_fig_compaction.py                                     the three-panel figure and its numbers file
```

**Validation.** Seven caches `pc_resume_v1_*` (41 runs each, no errors) pulled to `IB/TMP/PAPER_COMPACTION/node_sync/pc-b_SYNC/`; figure and numbers in `IB/TMP/PAPER_COMPACTION/figs/pilot41_fig.{png,pdf,json}`, drawn by:

```text
uv run --no-project --with matplotlib python IB/TMP/PAPER_COMPACTION/kit/make_fig_compaction.py \
    --sync-root IB/TMP/PAPER_COMPACTION/node_sync/pc-b_SYNC --tag pc_resume_v1 --out IB/TMP/PAPER_COMPACTION/figs/pilot41_fig
```

#### Planning Overview

Run about 250 evaluation-side tasks with the single-agent 27B, cut the window runs at 20k, and build the benchmark that resumes each prefix with the untuned 9B under a named variant and records solved or not plus continuation tokens. Variants in this order: full context, no context, the untuned 9B's text summary, recent text at the three sizes. Time the summaries. Draw the figure with them.

This is the gate before any training: does full context beat no context by a clear margin, what the real yield is, how long one resume run takes, and where sandboxed agent runs can execute beside the two GPUs. None of it needs an engine change.

#### Planned Changes

New files only, under a kit folder that survives sessions; the product tree is not edited unless the resume path needs a fix, which would come back here first.

`IB/TMP/PAPER_COMPACTION/` · pilot kit

```text
split.json            repository split, frozen before the pool runs
pool_pilot/           about 250 single-agent 27B runs on evaluation-side tasks (rollout cache)
cut_prefixes.py       window filter (20k to 50k) and the cut at the 20k turn boundary
resume_bench.py       (prefix, variant) -> resumed untuned-9B run -> solved, continuation tokens
make_fig_compaction.py  one results file -> the three-panel figure
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">E — Per-layer AC inputs in the vLLM fork</span>
    <span class="card-oneliner">The engine serves what the trainer trains, with the prefix cache correct from the first cut.</span>
    <span class="card-badge">TBD</span>
  </summary>

#### Planning Overview

Implement the Sept 17 source-level specification on the fork branch, narrowed to what the paper needs: the Python interface the harness uses, the Qwen 3.5 reader, additions to the hidden state entering named layers at the row positions. Requests that carry additions get their own cache identity block by block, so an agent's earlier turns stay cached when a tool result is appended; there is no stage in which caching is off. Accept it on agreement with the trainer's forward pass, a multi-turn rollout that reuses the cache, and a measured throughput cost. After the MVP: the first attempt serves input rows only (your call of Oct 13), so this step no longer blocks the figure.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M1 — The pool</span>
    <span class="card-oneliner">Single-agent 27B runs over both sides of the split.</span>
    <span class="card-badge">Implementing</span>
  </summary>

#### Planning Overview

Fix the repository split, run the rest of the pool, freeze the list of 100 evaluation prefixes as a file that never changes afterwards, and export training cuts from successful runs in training repositories.

**Evaluation side done (Oct 13 20:29Z).** 600 runs gave 130 window prefixes; the first 100 in the split's evaluation order are frozen as `prefixes_eval_100` (46 Python, 36 Java, 18 Kotlin; 61 repositories; the 27B went on to solve 55 of them).

**Training side running** on both EC2 nodes since 20:27Z, tasks in a shuffled order because the split lists Python first. Training continuations are written by the untuned 9B with the whole prefix in view (the benchmark's full-context variant on training-side prefixes).

One imbalance to state in the text: the training side has 78 Kotlin tasks of 2,676, while 18 of the 100 evaluation prefixes are Kotlin.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M2 — Training</span>
    <span class="card-oneliner">Tune the reconstruction pair on continuation, two arms on two GPUs.</span>
    <span class="card-badge">Implementing</span>
  </summary>

#### Planning Overview

**State at Oct 14 03:41Z: first wave done, second wave training, third wave's data being written.**

First wave (300 training runs from 186 repositories at 8 sizes each, one pass of 300 updates, 3 h 43 min on one H200). Held loss per token on 14 runs of 12 held repositories; references on the same runs: no context 0.972, full context 0.790.

| Arm | Start | 1/20 | 1/10 | 1/5 |
|---|---|---|---|---|
| A, at the start | Probe's rows-only pair | 0.994 | 1.008 | 1.025 |
| A, final | | **0.855** | **0.834** | **0.822** |
| C, at the start | `lc3i` directly | 0.969 | 0.968 | 0.965 |
| C, final | | 0.862 | 0.841 | 0.828 |

Arm B (KL) failed at its first step in the cluster's tree (no stored teacher targets for continuation packets) and was dropped: cross-entropy is the first attempt.

**Arm chosen on 50 development prefixes** (window prefixes from 45 training-side repositories the wave never trained on; never the frozen 100). Solved | answered | mean continuation tokens, both arms at a quarter of training:

| Rows | Arm A | Arm C |
|---|---|---|
| 4,000 | 24 \| 44 \| 9,883 | 19 \| 44 \| 10,616 |
| 2,000 | 17 \| 40 \| 12,957 | 13 \| 38 \| 13,181 |
| 1,000 | 17 \| 32 \| 17,889 | 7 \| 24 \| 22,963 |

References on the same 50: full 31 | 45 | 12,453; 9B summary 20 | 31 | 19,168; nothing 8 | 12 | 27,853. Arm A leads at every size and on held loss, so its final pair went onto the frozen 100 (M3). Arm A at half of training read 27 | 40 | 14,577, 18 | 35 | 17,178 and 18 | 26 | 21,435: no clear change against a quarter, inside the noise of 50 prefixes. Held loss moved by 0.001 to 0.005 per panel over the second half. Both point to your lever, more trajectories, not more updates.

**Second wave (training since 03:14Z, ends near 07:15Z).** Starts from arm A's final pair and sees only new runs: 338 training runs from 188 repositories at 8 sizes each, 338 updates, same loss and rates; evaluation and development repositories excluded. The 9B solved 712 of 1,066 training prefixes; 353 of those are new and usable.

**More trajectories on the development prefixes (Oct 14 08:47Z).** Solved | answered | mean continuation tokens on the same 50 prefixes:

| Pair | Training runs | 4,000 rows | 2,000 rows | 1,000 rows |
|---|---|---|---|---|
| Arm A final | 300 | 30 \| 44 \| 11,053 | 20 \| 42 \| 13,509 | 17 \| 37 \| 17,021 |
| Second wave final | 300 + 338 | 22 \| 37 \| 13,232 | 28 \| 40 \| 14,496 | 21 \| 36 \| 15,730 |
| Second + third waves, half way | 300 + 304 of 608 | 30 \| 45 \| 10,786 | 26 \| 38 \| 14,239 | 20 \| 39 \| 14,314 |

Paired against arm A, 13 to 20 of the 50 prefixes flip between solved and unsolved at each size, in both directions (combined run, half way: 3 \| 3, 10 \| 4, 8 \| 5). Run-to-run variance dominates: 50 development prefixes cannot resolve a gain of a few solved tasks, so more data shows no clear gain yet.

**Rule fixed before any further frozen-100 run.** The paper's figure uses the pair trained on all trajectories (second and third waves, ends near 12:00Z) unless it reads worse than arm A on the development prefixes; otherwise it keeps arm A's. Only that one pair goes onto the frozen 100; the choice is never made on frozen-100 numbers.

**Third wave (data).** All 2,676 training tasks are now pooled. The last 512 prefixes are getting their 9B continuations; the wave trains from the second wave's pair when both exist. After that, more data means a second 27B pass over the training tasks with a new seed.

Written Oct 13 20:20Z, after the pilot. Data preparation runs now under your go-ahead; the first training arms start when the two H200s come free (the probe arms end near 21:40Z) and enough training runs exist. Tell me if any default below is wrong; none of them blocks.

**What one example teaches.** The encoder turns a 27B-written prefix into AC rows. The reader sees the task head and the rows, and must write what the untuned 9B wrote when it had the whole prefix in view. Targets are the 9B's own sampled tokens, every assistant turn of its continuation; tool results between turns are context.

**Where the examples come from.** Training-side pool runs only (2,676 tasks, 615 repositories). Defaults I took:

| Choice | Default | Why |
|---|---|---|
| Which runs are cut | Window runs (20k–50k, cut at 20k, as the benchmark) **plus** runs that ended between 6k and 20k, cut at 60 % of their length | Three quarters of pool runs end under 20k. Window runs alone give about 450 prefixes, arriving over 5 to 10 hours; the short ones triple the examples and let the first arms start hours earlier. Evaluation prefixes stay window-only. |
| Continuation writer | Untuned 9B, thinking on, whole prefix in view, budget 30,000 | Your point about 27B verbosity; the pilot measured 346 tokens per turn for the 9B against 831 for the 27B. |
| Which continuations | Only those where the 9B went on to solve the task | A run that wanders to the budget is a poor target. |
| Sizes | One example per (run, size); size r drawn log-uniformly in [1/20, 1/5] of the prefix's own length, 8 sizes per run in one pass (first wave; one pass saves one 7 GB pair) | The three evaluated sizes are equally spaced in log. |
| Evaluation-side left-overs | Not used for training | Keeps going from 100 to 200 evaluation prefixes possible later without leaking a repository. |

**Arms, first wave (two H200s).** Both train encoder and reader, both serve rows only, because rows are all the current engine can carry in a live run:

| Arm | Loss | Start |
|---|---|---|
| A | Cross-entropy on the 9B's continuation tokens | The probe's rows-only pair if it ends clearly below the untrained rows-only reading, else `lc3i` |
| B | KL to the starting reader reading the whole prefix | Same |

The held panel is the runs of a few held training repositories at 1/20, 1/10 and 1/5. The arm is chosen by live resumed runs on held training-repository prefixes, never on the 100. A third arm on a GH200 uses cross-entropy from `lc3i` directly, to see whether the reconstruction adaptation matters. No arm with per-layer additions: rows only is the first attempt.

#### Planned Changes

No product code changes. Kit files (written, tested on the pilot's 41 runs) and cluster job files:

`IB/TMP/PAPER_COMPACTION/kit/` · training data

```text
cut_prefixes.py                  --short-lower 6000 --short-fraction 0.6 adds the short runs (training side only)
run_resume.py --variants full    the 9B continuation on training prefixes (tag pc_train_cont_v1)
export_continuation_material.py  full-context runs + notes + prefixes -> material.jsonl (solved only)
prepare_agent_continuation.py    material.jsonl -> one trainer packet per loss (sft, kl)
check_continuation_packet.py     CPU check: every item rebuilds, live prompt equals the student view
```

`IB/TMP/PAPER_COMPACTION/cluster/` · CSAIL jobs, folder `/data/scratch/ngom/activation/paper_compaction_v1/`

```text
build_packets_v2.sbatch          CPU job: packets built with the tree the trainer runs from
train_continuation.sbatch        one H200 per arm, the existing trainer unchanged
```

Checked so far: the builder runs under the cluster's tree (test packet from the pilot's runs, all checks passed for the KL packet). That tree has cross-entropy and KL; the combined "cross-entropy plus full-context match" loss exists only in the workspace tree and is left out of the first wave.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M3 — Final numbers and the main figure</span>
    <span class="card-oneliner">Accepted as the MVP figure: at 1/5, about the summary's solve rate at far fewer continuation tokens.</span>
    <span class="card-badge">Completed</span>
  </summary>

#### Completion report (Oct 14, 09:06Z)

**What landed.** You accepted this figure as enough for the MVP (Oct 14 ≈09:05Z): at 1/5, AC rows solve about as many tasks as the 9B's own summary (44 against 45 of 100) and finish in far fewer continuation tokens (12.0k against 18.8k, about full context's 11.3k). Figure `IB/ARTIFACTS/PAPER_COMPACTION/figs/frozen100_w1A_fig.{pdf,png,json}` (arm A, first wave, 300 training runs); text draft `IB/ARTIFACTS/PAPER_COMPACTION/compaction_results_draft.tex`.

**Drifts.** The AC + recent-text series was not run. Panel (c) comes from a one-at-a-time pass on 30 prefixes, not on all 100. The snapshot's Results paragraph is drafted beside it, not merged.

**Focused actual diffs.** Kit only: `eval100_ac_node.sh`, `slim_rollouts.py`, `make_fig_compaction.py --index --timing-tag`, `window4_a.sh`, `window5_b.sh`, `train_cont_tag_node.sh`, `merge_prefix_folders.py --exclude-index`.

**Validation.** Every series on the same 100 frozen prefixes (paired); 95 % bootstrap intervals; paired differences in `PLAN.md`; the pair was chosen on development prefixes, never on the 100.

#### Planning Overview

**First full figure (Oct 14 03:41Z)**: every series on the frozen 100, AC rows from the first wave's pair (arm A, 300 updates). Untuned 9B reader for the text series, continuation budget 30,000. Figure: `IB/TMP/PAPER_COMPACTION/figs/frozen100_w1A_fig.png` (numbers in the `.json` next to it).

| Series | Size (tokens or rows) | Solved | Answered | Mean continuation tokens |
|---|---|---|---|---|
| Full context | 19,688 | 55 | 91 | 11,269 |
| 9B summary | 426 | 45 | 67 | 18,776 |
| **AC rows, 1/5** | 4,000 | **44** | **86** | **11,984** |
| **AC rows, 1/10** | 2,000 | **34** | **78** | **12,632** |
| **AC rows, 1/20** | 1,000 | **30** | **69** | **16,837** |
| Recent text, 1/5 | 3,992 | 38 | 55 | 20,948 |
| Recent text, 1/10 | 1,975 | 30 | 44 | 23,885 |
| Recent text, 1/20 | 978 | 21 | 32 | 24,669 |
| Nothing | 32 | 16 | 24 | 28,259 |

What the figure shows, in plain words:

- **Against truncation at the same size**, AC rows solve more at every size (44 | 34 | 30 against 38 | 30 | 21) and answer far more often (86 | 78 | 69 against 55 | 44 | 32). Paired over the same prefixes, the solved gain is not established at any single size (+6 [−5, +18], +4 [−8, +16], +9 [−1, +19] points, 95 % intervals); the answered gain (+31 to +37 points) and the token saving (7,800 to 11,300 fewer continuation tokens) are.
- **Against the text summary**, AC rows at 1/5 tie on solved (44 against 45) with nine times the summary's size, and are well ahead on answered (86 against 67) and on cost to finish (11,984 against 18,776 tokens, close to full context's 11,269). At 1/10 and 1/20 the summary solves more (45 against 34 and 30). On solved alone, the summary is the stronger compaction per token.
- **Against full context**, 1/5 loses 11 solved of 55 (−11 [−20, −1] paired), the same loss as the summary (−10 [−18, −2]); but it answers as often as full context (−5 [−13, +2]) at the same token cost (+716 [−1,314, +2,725]), where the summary answers 24 points less and costs 7,500 tokens more.

So the sentence the figure supports today is about cost and speed (the run goes on as if it had the prefix, without a long write), not about solving more than a summary. The second and third waves test whether more trajectories move panel (a).

**Panel (c)** is from the one-at-a-time pass (one run per GPU, 30 frozen prefixes): the 9B writes its summary in **7.4 s** (510 output tokens); the encoder pass takes **1.3 s** (mean over 90 encodes; median 1.42 s at 4,000 rows, 0.94 s at 2,000, 0.93 s at 1,000; the slowest, 6.3 s, is the first encode after start). With a hundred runs at once the same steps took 17.8 s and 89.9 s, mostly queueing; those numbers are not drawn.

#### Planned Changes

No product code changes. Kit additions and what remains:

```text
kit/eval100_ac_node.sh           one pair on the frozen 100 at three sizes, then the one-at-a-time timing pass
kit/slim_rollouts.py             slim copies of AC caches for the figure script (an AC cache stores its rows, GBs per size)
kit/make_fig_compaction.py       --index (frozen list), --timing-tag (panel (c) from the one-at-a-time pass)
```

**Results draft (first wave):** `IB/ARTIFACTS/PAPER_COMPACTION/compaction_results_draft.tex` — Datasets, Comparisons and Results paragraphs rewritten from this figure, with the snapshot sentences it refutes listed at the top (full-context match, 1/10 as good as full, beats summarization on solved, an order of magnitude faster, AC plus truncation). The figure is copied to `IB/ARTIFACTS/PAPER_COMPACTION/figs/`. The snapshot itself is untouched.

Remaining: rerun the three AC sizes with the second and third waves' pairs (chosen on the development prefixes first); then the Results paragraph, written from the final figure, as a draft next to the snapshot for you to accept.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M5 — Hill-climb: 1/10 toward the summary</span>
    <span class="card-oneliner">Moderately high priority: always at least one GPU on it; more only when nothing else waits.</span>
    <span class="card-badge">Implementing</span>
  </summary>

#### Planning Overview

Your target (Oct 14 ≈09:05Z): bring AC rows at 1/10 (34 solved of 100 today) nearer the 9B summary (45). The frozen 100 is not used to choose; candidates are read on development prefixes at 2,000 rows.

**Why a few hundred runs already work.** The starting pair is not fresh. `lc3i` was trained on long-context reconstruction at 1/10. The probe's rows-only arm (`lc3i_rowsonly_1000`) then trained it for 1,000 more updates (8,000 reconstruction examples, rows only). The continuation step adapts a pair that can already compress text. Rates are **2e-5 for both encoder and reader**, cosine to 10 % with 50 warm-up updates, 8 examples per update, not 1e-4. The standing rule says 5e-5 should not be used, so I will not raise the rate without your word.

**How training runs are chosen today.**

1. Training-side pool: the 27B on 2,676 training tasks, split by repository.
2. Cut prefixes: runs ending in 20k–50k are cut at the first tool result past 20k (as the benchmark); runs ending in 6k–20k are cut at 60 % of their length.
3. The untuned 9B resumes each prefix with the whole prefix in view (budget 30k); only continuations that solve the task are kept.
4. Evaluation and development repositories are left out; 12 repositories form the held panel.
5. Each run gives 8 items, sizes drawn log-uniformly between 1/20 and 1/5 of its own prefix length.

The resulting mismatch: the training prefixes' median is 8.7k tokens (two thirds under 10k), while every evaluation prefix is about 20k. At evaluation 1/10 means 2,000 rows from a 20k prefix; in training it is mostly 500 to 1,000 rows from a short one.

**Evaluation sizes.** One model for all sizes; read at 4,000, 2,000 and 1,000 rows on prefixes of about 20k (1/5, 1/10, 1/20).

**Measurement.** 50 development prefixes, one seed, flip 13 to 20 tasks between any two pairs, so one reading cannot resolve a few-task gain. From now: candidates are read at 2,000 rows only, with two seeds (100 runs per candidate), arm A's final included as the baseline.

**Experiments, one at a time on one GPU, more only if a GPU is idle:**

| # | Change | Hypothesis | State |
|---|---|---|---|
| Baseline | Arm A's final pair (`w1A_final`, 300 runs) | — | Development at 1/10, two seeds: 20 + 20 = **40 of 100** solved |
| H1 | Evaluation-length runs only (prefix ≥ 19k tokens, 320 runs over all waves), from the same start as arm A | The length mismatch costs 1/10 | Development at 1/10: 22 + 21 = **43 of 100**, fewest continuation tokens (≈13.1k against ≈14.5k). Against arm A: better on 11 prefixes, worse on 9 (p 0.82); against H0: 9 and 13 (p 0.52) |
| H0 | Second + third waves on top of arm A (608 runs) | More trajectories | Development at 1/10: 22 + 28 = **50 of 100**; better on 15 prefixes, worse on 8 (sign test p 0.21). Best so far, not yet separable |
| H2 | Full-context distillation (match the untuned 9B reading the whole prefix) instead of one sampled continuation | Denser target than one sample | Development at 1/10: 21 + 21 = **42 of 100**, no better than arm A (40) and below H0 (50) |
| H3 | Second-seed 27B trajectories (780 of 990 new solved continuations) on top of H0 | More trajectories again | Packet built; training failed at start on the cluster's storage check (13:17Z); parked until the 9B-prefix results say whether 27B-prefix training is still the direction |
| H4 | Train on the 27B's own continuation text (the rest of each pool run) instead of the 9B's | Your candidate if H0–H2 do not pan out; also far more data, since every pool run past the cut has one | Waits for H0–H2 |

**Training data rule (your call, Oct 14 ≈16:30Z):** training may use the evaluation repositories; only the exact evaluation and development tasks are left out. The paper pitches domain specialization. Every pair so far was trained under the old repository rule.

Training keeps drawing sizes between 1/20 and 1/5; candidates are read at 1/10 only until one reaches a good point (your call, Oct 14 ≈09:20Z).

Not without your word: rates above 2e-5; per-layer additions (the engine step).

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M4 — Extras on request</span>
    <span class="card-oneliner">More ratios, the 27B summary, appendix comparisons.</span>
    <span class="card-badge">TBD</span>
  </summary>

#### Planning Overview

Each item needs extra training or extra resume runs, so each starts only after M3 and only if it defends a sentence in the text.

</details>

## The engine step in more detail

**Why it is in this plan.** The 2B → 9B model feeds the reader in two places: rows at the input, and small additions to the hidden state entering chosen layers. The serving engine carries only the first today. Serving both in a production engine is a claim of the paper's earlier sections; whether the compaction experiment itself needs it is what probe P decides.

**Where things stand.**

- The fork `amlatyrngom/vllm` and its branch `ac-interventions-v0.28.0` exist; the branch has no commits beyond upstream v0.28.0. That fork and `amlatyrngom/activation_context` are both public, so engine commits pushed there are visible before the paper.
- The local copy under `IB/REPOS/vllm` matches the branch but has lost its git metadata; work starts from a fresh clone.
- The source-level specification of Sept 17 (`IB/ARTIFACTS/VLLM_ACTIVATION_INPUTS/PLAN.md`) was reviewed and not built.

**What gets built, in order.**

| Step | Intent |
|---|---|
| Request | A request may carry, beside its tokens and input rows, positions and one small tensor per chosen layer |
| Lifetime | The engine keeps them with the request across chunked prefill, preemption and resume, and frees them at the end |
| Cache identity | From the first cut: two prompts with the same tokens but different additions never share cached state, and unchanged earlier blocks stay reusable |
| Model | The Qwen 3.5 forward adds the scheduled rows to the hidden state entering each named layer |
| Install | The project environment pins the fork at an exact commit |

**For the paper.** One small table for the systems part: agreement with the trainer's forward pass, throughput with and without per-layer additions, and cache reuse across an agent's turns.

## Beat 1 — Train and test selection, task limits

**Benchmark.** Long Code Arena bug localization in Java, Python and Kotlin. The agent gets a repository at the commit before a fix and the issue text, and must name the changed file. Scoring is exact match on the path; no judge.

**Split.** Evaluation side: the 150 official test tasks, plus repositories we hold out of `train` until the side has about 1,000 tasks. Training side: every other `train` repository. No repository on both sides; the split is frozen in a file before the pool runs. In the text: "the official test set, extended with held-out training repositories".

**Pool run (27B).** Single agent, medium reasoning, no subagents, no compaction, the run ends at 50,000 tokens. The token cap is the binding limit; the time limit is generous so it does not decide outcomes.

**The window.** A run qualifies when it passed 20,000 tokens and ended under 50,000, whatever its outcome. The prefix is everything up to the first turn boundary past 20,000, the rule the harness itself uses for compaction.

**Resume run (untuned 9B).** Same tools, sandbox restored to the state at the cut, 30,000 continuation tokens for every variant. No answer inside the budget counts as unsolved and as the whole budget spent.

**Counts.** 100 evaluation prefixes for now (200 later if wanted), taken in the split's fixed random order from the qualifying evaluation-side runs (about 600 runs at the pilot's 17 % yield); evaluation-side runs left over may serve training when their repository holds none of the 100; 50 more from training repositories as the development panel. Training cuts: every successful training-side run, cut at several points from 8k upward.

## Beat 2 — Trajectory synthesis: 27B or 9B

| Text | Writer | Why |
|---|---|---|
| The shared prefix | 27B (accepted) | Generator and banks exist; more tasks stay solvable; every variant shares it |
| The text-summary baseline | the untuned 9B (accepted, first) | What a harness running this reader would actually produce |
| Training continuations | the 27B's successful runs | Already the bank's format; the encoder learns what a useful continuation needs |
| A second summary point by the 27B | on request | A stronger summary to compare against |

New synthesis is bug localization only: about 3,700 single-agent 27B runs over the pool. The summary uses the harness's existing five-section compaction prompt.

## Beat 3 — Training targets

> Given the task statement and AC rows in place of the prefix, the reader should produce the continuation it would have produced with the whole prefix in view.

- **Start.** The `lc3i` reconstruction pair with its reader adapter, with or without per-layer additions as probe P selects. Encoder and reader adapter both train. Reconstruction teaches the rows to hold the prefix; this stage teaches the reader to act on them.
- **Continuation text comes from the 9B.** The 27B writes the prefix; the continuation the reader learns from is the untuned 9B's own successful run on the whole prefix. On the 41 pilot prefixes both models spend the same total (8,544 against 8,461 tokens) and solve about as many (28 against 26), but the 27B writes 831 tokens per turn in 6.9 turns and the 9B 346 in 13.0. Training on 27B text would teach the reader a longer-winded turn and blur what the AC rows contribute.
- **Two arms, same data.** Arm A matches the reader that sees the raw prefix (KL): it supports "as good as the whole context" and cannot exceed it. Arm B is plain text loss on that 9B continuation; the 27B's own continuation stays available as a third target at no extra cost if a stronger teacher is wanted. Evidence on which is better is split between earlier 9B and recent 4B runs, so both run, one per GPU.
- **Ratios.** Each example draws its ratio between 1/20 and 1/5. The start pair knows 1/10 only, so the first held panel reports the three sizes separately.
- **Data.** Bug-localization cuts only. Math and search traces enter only if this proves too thin.
- **Known gap.** The earlier continuation run reached 0.787 loss per token against 0.811 for recent text at equal size, on only 3,688 histories and never on bug localization. The margin over recent text is the main scientific risk here.

## Beat 4 — Plots

One script, one results file, written in M0. If a number is not in the figure or its table, it is not a goal.

### Main text: `fig:evalCompaction`, three panels

| Panel | x axis | y axis | Series |
|---|---|---|---|
| (a) Quality | size of what replaces the prefix (log scale: 1,000, 2,000, 4,000 of 20,000); "none" and "full" as dashed reference lines | tasks solved after resuming, out of 100, with 95 % intervals | recent-text line; AC rows line; AC rows + 1k recent text; the 9B text summary as one point at its measured size |
| (b) Cost to finish | same | mean continuation tokens until the answer; unsolved runs count the whole budget | same series |
| (c) Cost to compact | — | seconds to produce the compaction for one prefix on one GPU (log scale) | text summary by the 9B; AC rows by the 2B encoder; truncation is zero |

Panels (a) and (b) together are the paper's "continuation tokens required for accuracy-preserving resumption": a variant wins only if it is at the full-context line in (a) and at or below it in (b).

### Appendix and elsewhere

| Item | Cost | Status |
|---|---|---|
| Tasks solved within N continuation tokens, one curve per variant | free, same runs | planned |
| Length histogram of the 27B pool with the window marked | free, same runs | planned |
| Table of every number in the figure | free | planned |
| Input rows alone against input rows plus per-layer additions (loss) | probe P | planned |
| Engine table: agreement, throughput, cache reuse | from milestone E | planned, for the systems part |
| Arm A against arm B | already run in M2 | maybe |
| More ratios; the 27B summary; a 9B-written prefix | extra runs | on request |

### Naming to settle in the text

The snapshot says "prefix truncation" in Comparisons and "suffix truncation" in Results. Both mean keeping the most recent part of the trajectory; the figure labels it "recent text".

## Risks I see now

- **No gap to close.** If the untuned 9B does about as well with no context as with the full prefix, no compaction method can be told apart. M0 measures this first.
- **Rows alone may fall short.** The one diagnostic we have points that way for pairs trained with both channels. Probe P settles it in hours rather than after the pool is built.
- **Engine time.** If the engine step is on the critical path, it is real systems work in a large codebase; the baseline half of the figure still lands meanwhile.
- **Margin over recent text.** See the known gap in Beat 3.
- **Where agent sandboxes run.** Earlier 27B campaigns ran on rented machines; sandboxed runs on the CSAIL GPU nodes are not yet checked.
