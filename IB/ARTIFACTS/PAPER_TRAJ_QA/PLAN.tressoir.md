# Paper — trajectory question answering (first look)

First, intent-level look at trajectory Q/A, written Oct 14 00:12Z while you are away. It says what the repository already has, what is missing for the paper's multi-agent section, and what I am doing with free GPU slots meanwhile. Compaction keeps priority on every slot.

## Executive Summary

### Goal

The "Multi-Agent Collaboration" section of the snapshot has a synthesizer that sees each solver's trajectory through vectors, on hard math or on Musique. Trajectory Q/A is the task that teaches this: the reader answers a question about a trajectory from AC rows of it instead of its text.

`Agent.run_subagent` · data flow of the section

```text
solver 1 ─ trajectory ─ encoder ─ AC rows ┐
solver 2 ─ trajectory ─ encoder ─ AC rows ┼─ synthesizer (9B reader) ─ final answer
solver 3 ─ trajectory ─ encoder ─ AC rows ┘      asks: what did each solver establish, and how well?
```

### What exists

Found by a read-only research pass; not re-run by me.

| Piece | State |
|---|---|
| Question bank | 112,007 questions (coding 65,017, research 20,523, math 26,467) on **public** traces, written by the 27B. No Musique, DeepMath or IMO trajectories. |
| Pipeline | Material windows, question writing, training, decoded-answer cohort: all exist and ran in September. |
| Our own trajectories | 27B runs on Musique, DeepMath and other tasks, including solver segments; no questions written over them. |
| Harness | A subagent's trajectory can already return to its caller as an AC part at 1/20. |
| Multi-agent program | Missing. No three-solver plus synthesizer program; the earlier parallel program was dropped. |
| all@3, any@3, mean@3 | Missing. Never measured. |
| IMO | Missing. No loader, no DeepMath decontamination against it. |

### Earlier results

2B to 9B, 1/16, cross-entropy, 100 held questions (September):

| Reader sees | Held answer loss per token | Judged correct of 100 |
|---|---|---|
| Full trajectory text | 0.299 | 96 |
| Nothing | 1.772 | 26 |
| AC rows, starting pair | 0.945 | 40 |
| AC rows, after 2,000 blind questions | 0.908 | 41 |
| AC rows, after compaction training | 0.880 | 48 |
| AC rows, blind, 7,200 updates | 0.684 | not judged |
| AC rows, question-aware, 16,000 questions | 0.308 | not judged |

So question-aware encoding reached the full-text loss; blind encoding did not, and its answers were never judged past the early points. Held questions shared trajectories with training, so none of this is an unseen-trajectory result.

### What I ran meanwhile (all done)

| Item | Why it needs no decision | Slot |
|---|---|---|
| Check that the trajectory Q/A pipeline still runs from today's rows-only pair | Any choice below needs it | CPU job on the cluster |
| A first rows-only run on the existing bank: blind, sizes 1/20 to 1/5 | Tells whether rows only is enough for this task too | The H200 that compaction leaves idle |
| all@3, any@3, mean@3 for the untuned 9B on Musique and on math, three seeds | Your todo says the variant choice depends on these | An EC2 node when compaction has no 9B work for it |

### Boundaries

- **Both benchmarks, Musique and DeepMath** (your answer, Oct 14 12:48Z). Nothing for trajectory Q/A runs until your green light (defaults below).
- **Untuned 9B baselines, three seeds, 40k cap** (measurements only, Oct 14 09:43Z): Musique (100 validation tasks, solved = F1 1.0) all@3 0.26, any@3 0.43, mean@3 0.34, with 63 of 300 runs reaching 40k; DeepMath (100 tasks, difficulty ≥ 6) all@3 0.56, any@3 0.87, mean@3 0.73. Full table in `PLAN.md`.
- **Synthesis over the three runs** (Oct 14 10:45Z, one synthesizer run per task, 64k cap): with only the three final answers, the untuned 9B solves 0.40 on Musique (majority vote 0.33) and 0.79 on DeepMath (majority 0.73). With summaries and answers it solves 0.37 and 0.82. Either way it spends about a quarter of the three runs' tokens. It almost never solves a task that no solver solved. **As one call with no tools** (12:06Z), the same scores within noise: summaries + results 0.40 and 0.83 (beating majority vote 7–0 and 11–1), results only 0.38 and 0.81. With summaries it costs 3.5 % (Musique) and 31 % (DeepMath) of the three runs' tokens. Full tables in `PLAN.md`.
- **First try with AC rows** (Oct 14 14:02Z; T1 pair, not trained for synthesis): the three runs at 1/10 plus their final answers, one call, solve 0.36 on Musique and 0.80 on DeepMath. That is slightly below summaries + results (0.40 and 0.83), with no difference significant. Part of the gap is format: T1's adapter skips its thinking and the `Final answer:` line. Training on our own runs, with full replies as targets, comes next.
- **Every run feeding trajectory Q/A or agent collaboration stops at 40k tokens, with compaction off** (your rule, Oct 14), so these experiments never overlap compaction. The untuned-9B @3 baselines ran under a 62k cap and are re-scored exactly at 40k (`kit/score_at3_cap.py`; the cap is a plain stop). At seed 0, 21 of 100 Musique runs reach 40k before answering.
- Compaction first; these items stop the moment it needs their slot.
- Rows only, 2B encoder, 9B reader, as in compaction.
- Answers are judged on decoded text, never on loss alone.

## Your Answers So Far

| Question | Your answer | What it sets |
|---|---|---|
| What trajectory Q/A is for | The synthesizer's training task. Treat it as a first try: if the synthesizer is below summaries, add a dedicated SFT pass on valid 27B summaries plus single-call completions | Order: Q/A training first, then the synthesis reading, then the SFT pass only if needed |
| Does the encoder see the question | Blind. Make it question-aware only for a good reason, such as RAG later | One encoding per trajectory serves every question |
| Where the questions come from | The bank first, plus Musique and DeepMath training tasks of our own | Training mixes the bank with our own solver trajectories |
| What was started while you were away | You asked "What items?" They were the T1 run on the bank and the @3 measurement. Both are done (results above); nothing is running | — |

From your other notes:

- **Single call.** The synthesizer gets no tools, matching the single-call baselines above.
- **Multi-trajectory questions** go into Q/A training.
- **Both benchmarks.**
  - Evaluation stays on the current 100 Musique and 100 DeepMath tasks.
  - I also draw a reserve of 100 more per benchmark now, under the same rules, never trained on.
  - Training uses Musique's training split, and DeepMath problems disjoint from both evaluation sets.

## Green Light Defaults

Your second-round answers set the remaining choices. Nothing for trajectory Q/A runs before your green light.

| Item | Setting |
|---|---|
| Q/A target | SFT (cross-entropy), as in T1 and the September runs. The answer is written by the 27B, which reads the full trajectory text(s) (the "oracle"). The reader sees the question, AC rows and the solvers' final answers. KL is a later variant, only if compaction's H2 shows a gain |
| Multi-trajectory questions | 2–3 runs of one task, written by the 27B from all their full texts. Runs are chosen so the texts fit the 27B's context on the node (to be checked before building) |
| Solver runs (AC inputs) | Untuned 9B, three seeds, 40k, 500 tasks per benchmark first |
| SFT pass targets | 27B summaries of each 9B run, then a 27B single call over the three summaries and final answers. Kept when correct, thinking included, two samples per task |
| Evaluation | Current 100 + 100 tasks at 1/10 rows per run; a reserve of 100 per benchmark. All three sizes once hill-climbing is good enough |
| Nodes | Trajectory Q/A takes the EC2 nodes first. Compaction keeps one CSAIL GPU on the side. The SFT pass is prepared in parallel when GPUs allow |

**Were 500 tasks enough before?** T1 trained on 8,400 bank questions over 7,477 distinct trajectories, about one question per trajectory. 500 tasks with three seeds gives 1,500 trajectories per benchmark, fewer trajectories but several questions on each. The bank stays in the mix for variety.

## Milestones

All intent only, except that a first run of T1 is already training on a free GH200 (started Oct 14 00:46Z).

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">T0 — Today's state</span>
    <span class="card-oneliner">Does the old pipeline run from the rows-only pair, and how different are three independent solvers?</span>
    <span class="card-badge">TBD</span>
  </summary>

#### Planning Overview

Run the existing preparation on the cluster's tree with the rows-only pair as the start, and measure all@3, any@3 and mean@3 for the untuned 9B on Musique and on a math set with three seeds per task. The second item is what decides between the math and the search variant of the section.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">T1 — First rows-only pair on the existing bank</span>
    <span class="card-oneliner">Blind encoding, sizes 1/20 to 1/5, decoded answers on a held cohort.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Planning Overview

Train from the rows-only pair on the existing questions and judge decoded answers against full text and no context. This is a reading on public traces, not the paper's result.

**Started Oct 14 00:46Z** on the second GH200 slot (job 2653843): 8,000 questions from the existing bank (coding 4,648, math 1,927, research 1,425), encoder blind, sizes drawn between 1/20 and 1/5, 1,000 updates, rows only, from the rows-only pair. Held: 64 questions, each read at 1/20, 1/10 and 1/5.

| Held loss per token, before training | Value |
|---|---|
| No context | 0.665 |
| AC rows at 1/20 | 0.526 |
| AC rows at 1/10 | 0.460 |
| AC rows at 1/5 | 0.451 |
| Full trajectory text | 0.278 |

Decoded answers are not judged yet; that is the reading that counts.

#### Completion report (Oct 14, 07:52Z)

**What landed.** The run finished (job 2653843, 1,000 updates, 5 h 32 min). Held loss per token, 1/20 | 1/16 | 1/10 | 1/5: 0.526 | 0.523 | 0.460 | 0.451 at the start, **0.253 | 0.226 | 0.186 | 0.162** at the end. The 64 held questions were decoded (job 2659071, one GH200) for the untuned 9B with the full trajectory text and with nothing, and for the starting and trained pairs at four sizes. Five blind Sonnet judges graded 368 distinct answers against the gold answers (rubric: correct, partial, incorrect).

Judged correct:

| Reader sees | All 64 | Unseen trajectories (52) | Trajectory also in training (12) |
|---|---|---|---|
| Untuned 9B, full trajectory text | 51 | 41 | 10 |
| Untuned 9B, nothing | 16 | 11 | 5 |
| AC rows, starting pair, 1/20 | 18 | 13 | 5 |
| AC rows, starting pair, 1/10 | 19 | 14 | 5 |
| AC rows, starting pair, 1/5 | 17 | 12 | 5 |
| **AC rows, trained pair, 1/20** | **37** | **29** | 8 |
| AC rows, trained pair, 1/16 | 40 | 33 | 7 |
| **AC rows, trained pair, 1/10** | **49** | **40** | 9 |
| **AC rows, trained pair, 1/5** | **51** | **43** | 8 |

Paired over the same 64 questions (McNemar): trained against starting pair at 1/5, 35 questions gained and 1 lost (p ≈ 1e-9); trained 1/5 against full text, 8 and 8 (p = 1.0); trained 1/10 against full text, 7 and 9 (p = 0.80); trained 1/20 against full text, 6 and 20 (p = 0.009). The starting pair is no better than no context at any size.

**Drifts.**

- Held questions were drawn by question, not by trajectory: 12 of the 64 sit on a trajectory that other training questions also used (checked with `checks/held_overlap.py`, CPU job 2660594). The result holds on the 52 unseen trajectories (43 at 1/5 against 41 with full text).
- The untuned full-text reader did not submit an answer within the 1,024-token decode limit on 9 questions (51 of 55 submitted answers correct). The trained reader always answered, and it also learned the answer style, so part of the gap to the starting pair is format.
- The judges wrote one generic reason per verdict instead of a sentence each; a spot check of ten verdicts in one batch agreed with the gold.

**Focused actual diffs.** No product code. New on CSAIL (copies in `IB/TMP/PAPER_TRAJ_QA/`): `review/review_snapshot.py`, `wrappers/make_review_pair.py`, `jobs/{review_traj_qa_gh200,make_final_pair_cpu,check_review_cpu}.sbatch`, `checks/held_overlap.{py,sbatch}`; pair `pairs/tqa_blind_ro_final`. Local: `judge/{judge_answers.py,judge_prompt.md}`, `judge/keys/t1/scores_t1.json`, `judge_batches/t1/`.

**Validation.** CPU check of the review inputs (job 2657939); the decode ran on GPU without errors for all three reviews; `judge_answers.py score` merged every verdict (no missing or unknown ids).

**What it means for the paper.** On public traces with blind encoding, a rows-only 2B → 9B pair trained on 8,000 questions answers as well from 1/5 or 1/10 of a trajectory as the untuned 9B does from all of it. This is still the existing bank, not the paper's Musique, math or IMO trajectories (T2, T3).

#### Planned Changes

No product code changes. New files on the cluster only, with local copies:

`IB/TMP/PAPER_TRAJ_QA/cluster/` · packet wrapper and job files (folder `/data/scratch/ngom/activation/paper_traj_qa_v1/` on CSAIL)

```text
wrappers/build_traj_packet.py     trajectory-only packet from the existing bank: blind view, sizes 1/20 to 1/5, rows only, rates 2e-5
wrappers/check_traj_packet.py     counts by task, sizes, held groups, training settings
jobs/build_traj_packet.sbatch     CPU job
jobs/train_traj_qa_gh200.sbatch   one GH200, the existing trainer unchanged
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">T2 — Questions over our own agents' trajectories</span>
    <span class="card-oneliner">Math and search runs of our agents, questions written by the 27B.</span>
    <span class="card-badge">TBD</span>
  </summary>

#### Planning Overview

Write questions over solver trajectories on the paper's tasks, with a split by task so that held trajectories are unseen. Depends on the source decision.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">T3 — Three solvers and a synthesizer</span>
    <span class="card-oneliner">The program and the evaluation the section describes.</span>
    <span class="card-badge">TBD</span>
  </summary>

#### Planning Overview

Build the program: three independent solvers, each trajectory returned as an AC part, one synthesizer. Compare against a synthesizer that sees only the three final answers and one that sees the full text.

</details>

## Risks I see now

- The bank's trajectories are not our agents'; a pair trained on it may not transfer to solver trajectories.
- Blind encoding was well short of full text in September; rows only starts further back.
- The multi-agent program and the IMO data do not exist yet; they are the longer pole, not the training.
