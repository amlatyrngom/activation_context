# Compaction: best results so far (Oct 14, updated Oct 16 06:30Z)

For the writing session. Every number below was read from the run outputs. The full history is in `PLAN.md`.

**Status:** you accepted these results as the MVP (Oct 14). Hill-climbing continues on the side.

- **Canonical MVP figure (user, Oct 18; stands unless the PI's feedback this week changes it):** `figs/fig_compaction_mvp.{pdf,png}`, numbers in `figs/fig_compaction_mvp.json`.
  - Panels (a) solved and (b) continuation tokens: eval-200 (9B-written prefixes, 50k continuation budget), seeds 2–4, AC = H8a at 1/5 and 1/10, against full context, no context and the 9B text summary.
  - x-axis "Continuation Difficulty": Mean (all 200), P50 (hardest 50%), P75 (hardest 25%). Difficulty is the original 9B run's tokens after the cut. AC 1/20, truncation and the noisy P90 group (20 tasks) are left out on purpose.
  - Panel (c) compaction duration is unchanged from the frozen-100 figure, so it was timed on 27B-written prefixes. Say so in the caption, or re-time it on the 9B prefixes when a node is up.
  - Rebuild: `scripts/make_fig_compaction_v4.py` (copy; it runs from `IB/TMP/PAPER_COMPACTION/kit`, next to `heldout_quantiles_cache.pkl`) with `--style paper --ratios 1/5,1/10 --no-trunc --upto P75 --kind bar --values --fig-width 10`.
- **Earlier figures:** `figs/frozen100_h0_fig.*` (frozen-100 set, H0 pair) and arm A's `figs/frozen100_w1A_fig.*`.
- **Draft text:** `compaction_results_draft.tex` uses the H0 numbers below; its header lists the snapshot sentences the data refutes.

## Headline: frozen evaluation set (100 prefixes)

The setup:
- **Tasks:** Long Code Arena bug localization, 46 Python, 36 Java and 18 Kotlin, from evaluation-side repositories.
- **Prefix:** written by the 27B and cut at the first tool result past 20k tokens, inside runs of 20k–50k tokens. The mean prefix is 19.7k tokens. A version on 9B-written prefixes is running (below).
- **Resuming agent:** Qwen 3.5 9B with thinking on, given a 30k continuation budget. An unanswered run counts the whole budget.
- **AC rows:** from the **H0 pair**: arm A plus 608 more training runs, chosen on development numbers by the rule fixed in advance. One pair goes to the frozen set; H0 is the second, after arm A.

| What replaces the prefix | Size (tokens) | Solved [95 % interval] | Answered | Continuation tokens |
|---|---|---|---|---|
| Full prefix | 19,688 | 55 [45, 65] | 91 | 11,269 |
| **AC rows 1/5 (H0)** | 4,032 | **47** [37, 57] | **88** | **11,257** |
| **AC rows 1/10 (H0)** | 2,032 | **47** [37, 56] | **82** | **12,342** |
| **AC rows 1/20 (H0)** | 1,032 | **39** [30, 49] | **78** | **13,463** |
| Text summary (the 9B's own) | 426 | 45 [36, 55] | 67 | 18,776 |
| Recent text 1/5 (truncation) | 3,992 | 38 [28, 48] | 55 | 20,948 |
| Recent text 1/10 | 1,975 | 30 [21, 39] | 44 | 23,885 |
| Recent text 1/20 | 978 | 21 [13, 29] | 32 | 24,669 |
| Nothing | 32 | 16 [9, 23] | 24 | 28,259 |
| *Arm A's AC rows, for reference: 1/5, 1/10, 1/20* | | *44, 34, 30* | *86, 78, 69* | *11,984, 12,632, 16,837* |

Paired over the same 100 prefixes (difference in solved, bootstrap 95 % interval, prefixes won/lost, sign test; then difference in continuation tokens):

| AC (H0) against | Solved | Continuation tokens |
|---|---|---|
| Full prefix, at 1/5 | −8 [−16, +1], 6/14, p 0.12 | −12 [−1,471, +1,352] |
| Full prefix, at 1/10 | −8 [−16, 0], 5/13, p 0.10 | +1,073 [−392, +2,519] |
| Summary, at 1/5 | +2 [−8, +12], 14/12, p 0.85 | −7,519 [−9,685, −5,379] |
| Summary, at 1/10 | +2 [−8, +12], 14/12, p 0.85 | −6,434 [−8,439, −4,512] |
| Summary, at 1/20 | −6 [−17, +4], 11/17, p 0.35 | −5,313 [−7,657, −3,024] |
| Truncation, same size, 1/5 | +9 [−2, +20], 22/13, p 0.18 | −9,691 [−11,643, −7,639] |
| Truncation, same size, 1/10 | **+17 [+6, +28], 25/8, p 0.005** | −11,543 [−13,556, −9,509] |
| Truncation, same size, 1/20 | **+18 [+6, +29], 28/10, p 0.005** | −11,206 [−13,404, −8,976] |

**What the paper can say:**
- **Against full context:** at 1/5 and 1/10, AC rows finish at the cost of the full prefix (no significant token difference). They solve 8 fewer, which is not significant (the intervals touch 0).
- **Against the summary:** at 1/5 and 1/10, AC rows solve as many as the 9B's own summary (+2, not significant). They need 6.4k–7.5k fewer continuation tokens, which is significant, and they are about 5.5 times faster to produce.
- **Against truncation:** AC rows beat truncation of the same size at 1/10 and 1/20 on solved (significant) and at every size on tokens.
- **Still not supported:** "an order of magnitude faster" (it is about 5.5 times), and "exceeds full context".

**Time to compact,** one run at a time on an RTX PRO 6000 Blackwell:
- The text summary is a 510-token decode taking 7.4 s [6.9, 7.9] over 30 runs.
- The AC rows are one 2B encoder pass taking 1.3 s [1.2, 1.5] over 90 runs. This was measured with arm A's pair; H0 has the same architecture, so the timing carries over.

## Training setup (for the methods paragraph)

- **Models:** a Qwen 3.5 2B encoder with adapter modules writes the AC rows; the Qwen 3.5 9B reader carries a LoRA adapter. Both are trained, and rows only reach the reader.
- **Start:** the reconstruction pair, then 1,000 updates of long-context reconstruction at 1/10, rows only. This start is why 300 continuation updates suffice.
- **Continuation training (arm A):**
  - Data: 300 solved runs from 186 training-side repositories.
  - Prefixes: the training-side 27B pool, cut past 20k tokens (or at 60 % of runs of 6k–20k tokens). The median training prefix is 8.7k tokens, against about 20k at evaluation.
  - Targets: the untuned 9B's own continuation with the full prefix in view, kept only when it solved the task.
  - Sizes: each run is trained at 8 sizes, drawn log-uniformly between 1/20 and 1/5 of its own prefix.
  - Schedule: cross-entropy, learning rates 2e-5 (encoder) and 2e-5 (reader LoRA), cosine decay to 10 %, 50 warm-up updates, 8 examples per update, 300 updates (3 h 43 min on one H200).
  - Exclusions: evaluation and development repositories are excluded from training.

## Hill-climb on the development set (1/10 only, two seeds)

50 development prefixes from 45 repositories, at 2,000 rows (1/10), seeds 0 and 1, so 100 runs per pair. One-seed readings flip 13–20 of 50 between any two pairs, which is why every pair is read twice.

| Pair | What changed | Solved of 100 | Continuation tokens (mean) | Paired against arm A, per prefix |
|---|---|---|---|---|
| Arm A (earlier figure's pair) | 300 runs | 40 (20 + 20) | ≈14.5k | — |
| **H0** | + 608 more runs (waves 2 and 3, new tasks) | **50** (22 + 28) | ≈14.6k | better on 15, worse on 8 (p 0.21) |
| H1 | evaluation-length runs only (320 runs, prefix ≥ 19k), from the same start as arm A | 43 (22 + 21) | ≈13.1k | better on 11, worse on 9 (p 0.82) |
| H2 | KL distillation from the base 9B reading the full prefix | 42 (21 + 21) | | |
| H3 | second-seed 27B runs (780) on top of H0 | packet built; first start failed on CSAIL storage (Oct 14 13:40Z), not yet trained | | |

- **H0 is the best so far,** and it went to the frozen set by the rule fixed in advance. There it gains on arm A at every size: 47 against 44 at 1/5, 47 against 34 at 1/10, 39 against 30 at 1/20.
- **H1** shows that training only at evaluation length gives the cheapest finishes, but not more solves.

## Running: 9B-written prefixes at a 50k limit (your requests, Oct 14)

The same benchmark on prefixes written by the untuned 9B, so the prefix writer and the resuming agent are the same model: 200 evaluation prefixes (none shared with development), 3 seeds averaged per prefix, at a 50,000-token continuation limit. The 30k reading is a replay of the same runs, so the two readings differ only by the limit. Seed 0 on all 200 ≈21:00–22:00Z; all ≈01:00–02:00Z Oct 15.

**The argument (yours, Oct 14):** what 50k won't erase is the cost. A variant that catches up does it by spending more continuation tokens. So if solved converges, the argument is on tokens: AC and full reach the answer early, the others pay to rebuild the context. Only a solved gap that persists at 50k means the context itself is missing. Failures are not explained by the limit.

First reading (30k runs, seed 0, first 100 prefixes; one seed on 100 is about ±10):

| What replaces the prefix | Solved | Answered | Continuation tokens |
|---|---|---|---|
| Full prefix | 60 | 79 | 16,396 |
| AC rows 1/5 (H0) | 61 | 77 | 17,384 |
| AC rows 1/10 (H0) | 57 | 72 | 18,113 |
| AC rows 1/20 (H0) | 53 | 65 | 19,449 |
| Recent text 1/5 | 55 | 67 | 20,556 |
| Nothing | 38 | 44 | 25,681 |

**Seed 0, all 200 prefixes (Oct 15 00:40Z), 20k cut.** Paired on the same 200 tasks; "nothing" is the run shared with the fraction scan.

| What replaces the prefix | Solved, 50k | Tokens per run, 50k | Solved, 30k | Tokens per run, 30k |
|---|---|---|---|---|
| Full prefix | 146 | 17,976 | 134 | 15,241 |
| AC rows 1/5 (H0) | 143 | 18,021 | 127 | 15,524 |
| 9B summary | 140 | 20,222 | 121 | 16,736 |
| Nothing | 127 | 33,335 | 71 | 26,009 |

- At 50k, solved converges (146 / 143 / 140 / 127). The difference is cost:
  - AC 1/5 spends the same as full (−45 tokens per run).
  - AC 1/5 saves 2,200 per run against the summary; full saves 2,245.
  - AC 1/5 saves about 15,300 per run against nothing.
- Paired solved at 50k: full − summary +6 (16 vs 10, p 0.33); AC − summary +3 (p 0.70); full − AC +3 (p 0.70).
- At 30k: full − summary +13 (20 vs 7, p 0.02); AC − summary +6 (p 0.36).
**Seeds 0 and 1 averaged per prefix (Oct 15 02:40Z; `kit/two_seed_report.py`).** Seed 1 is less favourable to AC than seed 0.

| What replaces the prefix | Solved, 50k (mean@2) | Tokens per run, 50k | Solved, 30k | Tokens per run, 30k |
|---|---|---|---|---|
| Full prefix | 146.0 | 17,345 | 134.0 | 14,999 |
| AC rows 1/5 (H0) | 141.5 | 18,991 | 121.5 | 15,998 |
| 9B summary | 141.5 | 19,850 | 122.5 | 16,371 |

- **50k:**
  - AC − summary: 0.0 solved, −860 tokens per run.
  - Full − AC: +4.5 solved (p 0.32), −1,645 tokens.
  - Full − summary: +4.5 solved, −2,505 tokens.
- **30k:** full is ahead of both, by 34 prefixes to 14 (p 0.01). AC ≈ summary.
- **Reading:** solved converges at 50k, as predicted. On cost, AC sits between full and summary: it recovers about a third of full's token advantage over the summary. Seed 2 follows (AC running).

**All three seeds, H0 (Oct 15 04:40Z; mean@3 per prefix; 200 prefixes; 0 error rows).** This is the H0 headline table.

| What replaces the prefix | Solved, 50k | Tokens per run, 50k | Solved, 30k | Tokens per run, 30k |
|---|---|---|---|---|
| Full prefix | 145.3 | 17,359 | 131.7 | 14,955 |
| AC rows 1/5 (H0) | 141.7 | 19,363 | 121.7 | 16,249 |
| 9B summary | 139.3 | 20,551 | 120.7 | 16,842 |

- **50k:**
  - AC − summary: +2.3 solved (p 0.69), −1,188 tokens per run (fewer on 113 of 200 prefixes, p 0.08).
  - Full − AC: +3.7 solved (p 0.47), −2,005 tokens (p < 0.001).
  - Full − summary: +6.0 solved (p 0.06), −3,192 tokens (p < 0.001).
- **30k:** full leads both by 10–11 solved (p ≤ 0.01). AC ≈ summary.
- **Reading:** solved converges at 50k. On cost, H0's AC recovers about 37 % of full's token advantage over the summary.
- **Repository overlap is allowed (root canon, Oct 14):** 114 of H5's 156 training runs and H6's 209 evaluation-side runs come from repositories that also hold eval-200 prefixes, but never the same tasks. Training excludes only the exact evaluation and development tasks; the paper pitches domain specialization. So the H5/H6 results stand as measured. (A 12:30Z note calling this leakage was my mistake and is withdrawn.)
- **Hill-climb, H5** (H0 + 156 9B-written runs, 1 epoch):
  - On the 9B dev set (75 prefixes, 2 seeds) it matches full: 1/5 at 56.0 solved and 15,830 tokens, against full 56.0 and 15,515.
  - On these 200 prefixes the gain is smaller. At 1/10 (seeds 0–1, 50k), H5 solves 136.5 against H0's 130.5 at 1/10 (+6.0, p 0.36), with 1,324 fewer tokens per run.
  - That only brings H5 at 1/10 to the level of H0 at 1/5: 3 seeds, 138.3 / 19,346 against 141.7 / 19,363.
  - **H5 at 1/10, 3 seeds (50k):** 138.3 solved, 19,346 tokens, against H0 at 1/10's 128.0 and 21,665.
    - +10.3 solved (48 vs 30 prefixes, p 0.05), −2,320 tokens. At 30k: +11.0 (p 0.01).
    - H5 lifts the 1/10 rate up to the 1/5 level, so the 9B-written material helps most where compression is stronger.
  - **H5 at 1/5, 3 seeds (50k):** 138.3 solved, 18,370 tokens.
    - Against H0 at 1/5 (141.7 / 19,363): −3.3 solved (p 0.32), −994 tokens.
    - Against the summary: −1.0 solved, −2,182 tokens.
    - Full still leads H5 1/5 by +7.0 solved (p 0.01) and 1,011 tokens.
    - At 30k: H5 1/5 127.0 against H0 1/5 121.7 and summary 120.7.
  - **H3** (the earlier recipe on 27B-written material), 1/5, 3 seeds, 50k: 139.7 solved, 20,632 tokens. Against H0: −2.0 solved, +1,268 tokens. No gain; it sits at the summary's level.
  - **H5x2** (H5's material, 2 epochs), 1/5, 3 seeds, 50k: 140.3 solved, 18,036 tokens, against H0 141.7 / 19,363 (−1.3, 24/28 prefixes, p .68; −1,328 tokens) and summary 139.3 / 20,551. At 30k: 127.7 against H0 121.7 (+6.0, p .49). A second epoch does not change solved; it ties H5.
  - **H6** (866 runs, mixed cuts), 1/5, 3 seeds, 50k: 138.0 solved, 19,600 tokens, against H0 141.7 / 19,363 (−3.7, 23/36 prefixes, p .12; +236 tokens) and summary 139.3 / 20,551. At 30k: 119.0 against H0 121.7. More, longer material made the AC no better; it sits at the summary's level.
  - Likely cause: H5's training prefixes are short (median 7.8k vs 20k here). H6 and H6L (eval-length material) are training.

These prefixes take in easier tasks than the 27B set (only 26 tasks shared; on those, "nothing" is 7 against 3), so absolute numbers do not compare across the two sets; compare variants within a set.

## For the writing session: the simple presentation (Oct 16 ≈11:00Z)

The user (10:40Z) asked for simpler presentation: no "rank on seeds X, read on seeds Y" in the paper or the talk.

**Proposal:**
- **Seeds:** one set of 3 seeds (2–4, which every method has).
- **One AC model** (H7S) across all three ratios.
- **"Hard tasks":** those where the original 9B run, the one the prefix was cut from, still spent the most tokens after the cut. In plain words, the agent still had a long way to go.

That criterion involves none of the compared runs, so no method is penalised by the selection.
- Ranking on the summary−full gap with the same seeds pushes the summary below "nothing" (52.7% vs 62.0% on the top 25%), because it picks the tasks where the summary happened to fail.
- Ranking on full-context cost does the same to full context.

Solved share / mean continuation tokens at 50k; 200 tasks; mean of seeds 2–4. Script: `kit/make_fig_compaction_v3.py`. Figures: `figs/fig_compaction_simple_top25.*` and `figs/fig_compaction_simple_top50.*`.

| What replaces the prefix | All 200 | Hardest 50% | Hardest 25% | Hardest 10% |
|---|---|---|---|---|
| Full prefix | 71.7% / 17.3k | 63.0% / 23.8k | 62.7% / 28.1k | 70.0% / 29.9k |
| AC 1/5 | 70.3% / 18.4k | 61.0% / 25.6k | 61.3% / 29.2k | 70.0% / 31.4k |
| AC 1/10 | 69.3% / 18.8k | 59.7% / 25.9k | 58.7% / 30.5k | 61.7% / 30.8k |
| AC 1/20 | 64.5% / 21.1k | 56.3% / 28.5k | 53.3% / 33.0k | 55.0% / 33.1k |
| 9B text summary | 68.7% / 21.2k | 57.0% / 29.3k | 55.3% / 34.4k | 63.3% / 36.0k |
| Nothing | 63.2% / 32.7k | 49.3% / 38.6k | 50.0% / 39.5k | 50.0% / 39.6k |
| *AC before data selection (H0): 1/5, 1/10, 1/20, hardest 25%* | | | *56.0%, 49.3%, 44.0%* | |

- **The story in one line:** at 1/5 and 1/10, AC solves about as well as the full prefix and better than the summary, with 4–5k fewer continuation tokens than the summary on the hard tasks. At 1/20 it ties the summary.
- **Data selection** (H0 → H7S, hardest 25%): +5.3 at 1/5, +9.4 at 1/10, +9.3 at 1/20.
- **Optional extra column:** "hardest 25% by the summary−full gap" (67.3 / 52.7 / 68.7 / 68.7 / 64.7 for full / summary / AC 1/5 / 1/10 / 1/20). It is the user's original "optimistic" cut, and the most flattering to AC. It needs a footnote, though: the summary is penalised by construction.
- **Error bars:** on 50 tasks they are about ±10 points. Paired differences are tighter; see the next section.

**Per-ratio specialists (user request, Oct 16; evals seeds 2–4).** R10 and R20 start from H5 like H7S and train on the same H7 material and epochs, but at exactly one ratio. H7S mixes ratios.

| Model, ratio | All 200 | Hardest 50% | Hardest 25% | Median tokens |
|---|---|---|---|---|
| H7S (mixed), 1/10 | 69.3% / 18.8k | 59.7% | 58.7% | 15.5k |
| R10 (1/10 only), 1/10 | 67.7% / 18.9k | 55.7% | 55.3% | 15.2k |
| H7S (mixed), 1/20 | 64.5% / 21.1k | 56.3% | 53.3% | 17.9k |
| R20 (1/20 only), 1/20 | 66.2% / 20.3k | 56.3% | 56.0% | 16.8k |
| H7S (mixed), 1/5 | 70.3% / 18.4k | 61.0% | 61.3% | 15.3k |
| R5 (1/5 only), 1/5 | 69.7% / 18.6k | 60.7% | 59.3% | 14.2k |

- Per-task over the 3 seeds: R10 vs H7S wins 47 and loses 57 (p .38); R20 vs H7S wins 69 and loses 59 (p .43); R5 vs H7S wins 43 and loses 47 (p .75).
- Reading: specializing to one ratio gives no reliable gain at any ratio. One mixed-ratio model suffices, which is the simpler story.

**H8a: H7's recipe on 1.75× the material (Oct 17; evals seeds 2–4).** H8a uses 3,393 kept prefixes (H7S used 1,942). It has the same init (H5), the same selection, 1 epoch and the same rates. Cells give solved share / mean continuation tokens.

| Ratio | Model | All 200 | Hardest 50% | Hardest 25% | Median tokens |
|---|---|---|---|---|---|
| 1/5 | H7S | 70.3% / 18.4k | 61.0% | 61.3% | 15.3k |
| 1/5 | H8a | 71.3% / 17.1k | 61.7% | 62.0% | 12.6k |
| 1/10 | H7S | 69.3% / 18.8k | 59.7% | 58.7% | 15.5k |
| 1/10 | H8a | 71.0% / 18.3k | 59.7% | 60.0% | 15.1k |
| 1/20 | H7S | 64.5% / 21.1k | 56.3% | 53.3% | 17.9k |
| 1/20 | H8a | 65.7% / 20.0k | 54.3% | 52.7% | 17.4k |
| — | Full prefix | 71.7% / 17.3k | 63.0% | 62.7% | 13.2k |

- **Per task over 3 seeds, H8a vs H7S:** 1/5 wins 46 and loses 40; 1/10 wins 53 and loses 43; 1/20 wins 68 and loses 61. None is significant.
- **Against full context:** at 1/5, 37 vs 39; at 1/10, 43 vs 47. H8a is statistically indistinguishable from the full prefix at 1/5 and 1/10.
- **Against the summary:** at 1/5, 51 vs 35 (p .11); at 1/20, 51 vs 69 (p .12).
- **Reading:** more material of the same kind gives small, consistent gains of about 1 point and about 1k fewer tokens. These are not significant on 200 tasks.

**Full H8: H7's recipe on 2.2× the material (Oct 17 20:30Z; evals seeds 2–4, all 200 rows, no error rows).** Full H8 uses 4,276 kept prefixes, with the same init, recipe and 1 epoch as H8a (1,100 updates; loss 0.55 to 0.33).

| Ratio | Model | All 200 | Hardest 50% | Hardest 25% | Median tokens |
|---|---|---|---|---|---|
| 1/5 | H8a | 71.3% / 17.1k | 61.7% | 62.0% | 12.6k |
| 1/5 | Full H8 | 70.5% / 17.4k | 60.3% | 61.3% | 13.7k |
| 1/10 | H8a | 71.0% / 18.3k | 59.7% | 60.0% | 15.1k |
| 1/10 | Full H8 | 68.7% / 17.6k | 55.7% | 54.0% | 13.8k |
| 1/20 | H8a | 65.7% / 20.0k | 54.3% | 52.7% | 17.4k |
| 1/20 | Full H8 | 66.8% / 19.6k | 58.3% | 57.3% | 16.7k |
| — | Summary | 68.7% / 21.2k | 57.0% | 55.3% | 17.3k |

- **Per task over 3 seeds, full H8 vs H8a:** at 1/5, 39 won and 44 lost; at 1/10, 42 and 56 (p .19); at 1/20, 61 and 54. None is significant.
- **Against the summary:** at 1/5, 53 vs 42 (p .30); at 1/10, 52 vs 52; at 1/20, 53 vs 64.
- **Reading:** past 1.75× the material, more best-of-3 continuations at the same recipe do not move eval-200. H8a stays the compaction model for the paper. Only at 1/20 is full H8 ahead of both H8a and H7S, and that is within noise.

## Held-out difficulty quantiles and H7S (Oct 16 06:20Z): the final figure's view

Same 200 prefixes, 50k. Tasks are ranked by how much more the summary spends than full, on seeds 0–1, and every method is read on seeds 2–4. Ranking and reading on the same seeds overstates the gap through regression to the mean (≈14k in-sample, ≈8k held out). Solved share / mean continuation tokens; only complete 200-row runs. Script `kit/heldout_quantiles.py`, numbers in `heldout_quantiles_0116.json`.

| What replaces the prefix | Seeds read | All 200 | Top 50% | Top 25% | Top 10% |
|---|---|---|---|---|---|
| Full prefix | 2–4 | 71.7% / 17.3k | 69.7% / 17.6k | 64.0% / 20.5k | 60.0% / 22.6k |
| 9B summary | 2–4 | 68.7% / 21.2k | 67.0% / 23.5k | 57.3% / 28.3k | 50.0% / 31.7k |
| Nothing | 2–4 | 63.2% / 32.7k | 62.3% / 34.1k | 53.3% / 37.8k | 46.7% / 41.0k |
| AC 1/5, H0 | 2–4 | 69.2% / 20.2k | 68.7% / 21.3k | 63.3% / 26.2k | 56.7% / 31.2k |
| AC 1/5, H5 | 2–4 | 69.7% / 18.2k | 71.0% / 18.3k | 66.0% / 21.8k | 61.7% / 24.4k |
| AC 1/10, H0 | 2–4 | 63.7% / 21.8k | 59.3% / 23.0k | 50.7% / 27.9k | 50.0% / 29.3k |
| AC 1/10, H5 | 2–4 | 68.8% / 19.5k | 65.7% / 20.6k | 57.3% / 23.4k | 53.3% / 26.3k |
| **AC 1/10, H7S** | 2–4 | 69.3% / 18.8k | **70.3% / 18.9k** | **64.7% / 22.4k** | **61.7% / 23.4k** |
| AC 1/20, H0 | 2–4 | 60.3% / 23.5k | 60.3% / 24.7k | 52.7% / 29.3k | 41.7% / 33.5k |
| AC 1/20, H5 | 2–4 | 65.3% / 21.6k | 63.0% / 22.3k | 54.0% / 27.0k | 43.3% / 29.8k |
| AC 1/20, H7S | 2–4 | 64.5% / 21.1k | 64.3% / 22.4k | 56.0% / 26.8k | 55.0% / 27.3k |

- **H7S** is H5 plus one epoch on best-of-3 shortest-correct 9B continuations of multi-cut training prefixes: 1,942 prefixes kept, 4,016 items.
- **AC 1/10, H7S against H5 at 1/10** (paired over tasks, seeds 2–4, 95% bootstrap):
  - Top 25%: +7.3 solved points [+1.3, +13.3], −1.1k tokens.
  - Top 50%: +4.7 [−0.3, +9.7], −1.7k [−3.0, −0.5].
  - Top 10%: +8.3, −2.9k [−5.3, −0.4].
- **AC 1/10, H7S against full:** at par on the hard subsets. Top 25%: +0.7 [−5.3, +7.3] solved, +1.9k [−0.7, +4.4] tokens. Over all 200 tasks: −2.3 [−5.7, +1.0] solved, +1.5k tokens. At 1/10 it beats the summary by ≈6k tokens on the top 25%.
- **AC 1/20, H7S against H5 at 1/20** (seeds 2–4; seed 4 undid most of the seeds 2–3 gain):
  - All: −0.8 [−4.7, +3.0].
  - Top 50%: +1.3 [−3.7, +6.3].
  - Top 25%: +2.0 [−5.3, +9.3].
  - Top 10%: +11.7 [+0.0, +23.3], −2.5k tokens.
  - Against full it is still behind: top 25% −8.0 [−15.3, 0.0] solved, +6.3k tokens. 1/20 is not at par yet.
- **AC 1/5, H7S, seeds 2–4: no gain at the mildest ratio.** All 70.3% / 18.4k; top 25% 59.3% / 21.5k; top 10% 51.7% / 23.8k. Paired against H5 at 1/5: all +0.7 [−2.8, +4.2]; top 25% −6.7 [−14.7, +0.7]; tokens about equal.
  - H7S's gain is at 1/10 and 1/20.
  - On the top 25%, H7S reads 59.3 / 64.7 / 56.0 at 1/5 / 1/10 / 1/20, against H5's 66.0 / 57.3 / 54.0, full 64.0 and summary 57.3.
  - The 1/5 < 1/10 inversion within H7S is on 50 tasks; the intervals are about ±7 points.
- **H7M** uses median-correct instead of shortest-correct targets at r ≤ 1/10; everything else matches H7S. It is no better where its targets differ, so shortest-correct selection stays the rule.
  - Held-out top 25% at 1/10: 62.7% / 23.7k, against H7S's 64.7% / 22.4k.
  - Held-out top 25% at 1/20: 53.3% / 27.3k, against H7S's 56.0% / 26.8k.
  - In the simple table (hardest 25%), H7M against H7S: 54.0 vs 58.7 at 1/10 and 52.0 vs 53.3 at 1/20.
  - At 1/5, where both use the same kind of targets, three seeds give H7M 66.7% held-out and 62.0% simple, against H7S's 59.3% and 61.3%. The two are equal in the simple table. The held-out ranking's 7-point gap at the same targets shows its noise (about ±7 points on 50 tasks), which is another reason to prefer the simple table.
- **Also tried, no gain over H5 (seed 2 only):** H6fromH5 at 1/5, and H6L at 1/5 and 1/10.
- **Next:**
  - H7M (median-correct targets at 1/10 and lower) evaluates ≈08:00–11:00Z.
  - H8 scales H7's selection to ≈5,600 prefixes, ≈2.9×.
- **Figure draft:** `figs/fig_compaction_v2_draft.*` from `kit/make_fig_compaction_v2.py`. Panel (a) is continuation-token quantiles; panel (b) is solved on the top 25% (top 50% with `--top 50`).

## Where to cut: fraction scan (Oct 14, 9B prefixes, seed 0)

Same 200 tasks, each run cut at 50 %, 65 % or 80 % of its own length (prefix medians 17.3k, 21.9k, 26.5k tokens). Full prefix against the 9B's own summary, paired on the same tasks; "nothing" does not depend on the cut.

| Cut | Solved, full − summary, 30k | Tokens, full − summary, 30k | Solved, full − summary, 50k | Tokens, full − summary, 50k |
|---|---|---|---|---|
| 50 % | +3 (25 vs 22, p 0.77) | +745 | +2 | +469 |
| **65 %** | **+17 (27 vs 10, p 0.01)** | −1,245 | +3 | **−3,025** |
| 80 % | +9 (21 vs 12, p 0.16) | −1,501 | 0 | −2,002 |

- Solved out of 200 at 50k: full 139 / 144 / 145, summary 134 (of 197) / 141 / 145, nothing 127 (33,335 tokens per run).
- At 30k: full 124 / 139 / 141, summary 118 / 122 / 132, nothing 71.
- The summary loses most at the 65 % cut: solves at 30k, tokens at 50k. At 50k solved converges (your prediction), so the argument is cost: context saves about 12–21k tokens per run against nothing, and full saves 2–3k against the summary at the 65–80 % cuts.
- AC (H0) on the 65 % and 80 % cuts is running (seed 0). The first try ran out of GPU memory and was discarded; the rerun is due ≈02:00–03:00Z.
