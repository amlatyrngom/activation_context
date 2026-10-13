# Match-term ablation and real-like questions: results

Results of the two pieces of work approved on Oct 8 at about 16:33Z (plan: `REAL_LIKE_QUESTIONS_PLAN.tressoir.md`). The ablation says the gain of the match term is its KL part alone. The question work produced a judged set of 2,423 questions for the 50 evaluated articles. The best pair trained without any real question reaches 72.0 %, and it does so with the simplest questions: what matters is the text the reader is matched on, not how real the question looks.

All accuracies are on the 894 benchmark questions of the 50 evaluated QuALITY articles, AC rows at 1/20 of the article, 480 new tokens, one-argument answer call. One number is ±1.6 points; a difference of two is about ±2.3. "Teacher" is the untrained reader given the whole article: 75.3 %.

## The three answers

1. **Ablation.** KL to the teacher carries the whole gain. Cosine adds nothing, with or without KL. The text loss on our synthetic items costs 4 to 7 points when it is trained next to KL.
2. **Real-like questions.** None of the ways we tried of prompting the 27B writer gave questions that are as hard with the article as real ones while staying sound. Every variant that reached the real numbers did so through defective items, which the 27B judge then removed. What can be matched is how guessable a question is without the article.
3. **Training.** Under KL only, with the teacher's own explained answer as the text to match, questions on the evaluated articles reach 71 to 72 % whether they come from the judged final set (71.1 %) or from the rules-only generator (72.0 %). That equals real questions under the same loss (71.1 %). Question realism only shows when training on other articles: rules only 60.5 %, outsider-guess options 66.8 %, real 71.1 %. Real questions trained with their keys still lead (73.6 %). More questions do not help: the recipe settles at 71 to 72 %, about 85 % of what the teacher gets right.

**Added Oct 11: the 75 % line is not a ceiling, and the best pair without real questions is now at 75.1 % with a required answer call.** Details are in the last five sections before "Checkpoints".

- **Why the reader with the article stops near 75 %.** Untrained, it is right on 75.3 % by itself and on 76.4 % when it is asked for its letter at once. After the text-loss training on our synthetic explained answers the same reader with the whole article is right on 79.9 % asked at once, and on 81.0 % when it reasons first. Training the reader moves the line; the reasoning it writes adds about one point. The gain is there after the first text-loss training and does not grow with more cycles. It recovers the misses that have a clear key; what it still misses is mostly questions whose key is arguable.
- **With rows.** Three KL cycles toward that reader (reader frozen, only the encoder side trained), then three more on the questions where a pair leaves its teacher, averaged: 75.1 % asked at once, 75.2 % with reasoning and the letter asked for at the cap (69.5 % as decoded by itself, because 104 replies reach the cap). Asked the same way, the untrained reader with the whole article is at 76.4 %, so the pair is 1.3 points under it at 1/20 of its tokens. The KL average pair stands at 71.5 %, the text-loss average pair at 67.4 % under the same required call.
- **What is left between rows and article (75 against 80).** The rows follow their teacher on 92 % of fresh questions from our generator and on 81 % of the benchmark's questions of the same kind. The pair does not memorize questions. The 5 points are 86 questions on which the trained readers agree among themselves and the rows give another letter. Nothing in the question stage closes them: not more cycles, not mined questions, not a cycle on the real questions of the training split (75.2 %), not twice the rows. That points at the channel itself (encoder, reconstruction), not at the questions.

## Ablation of the match term

Every arm continues the question run's final pair (60.0 %) for 631 updates on the same 5,048 synthetic items. The teacher is the untrained reader with the whole article, run live along each item's fixed answer text. KL is the exact divergence over the whole vocabulary at every answer position. Cosine is one minus the cosine of the reader's final hidden vector at those positions. Weight 0.5.

| Arm | Right on 894 | Hard / easy | 200 questions: rows / wrong rows / no rows | Answers open with |
|---|---:|---:|---:|---|
| Text only (control) | 60.7 % | 58.1 / 65.3 | 60.5 / 39.1 / 43.0 | reasoning text |
| Text + cosine | 60.1 % | 57.2 / 65.0 | 57.0 / 40.6 / 42.5 | reasoning text |
| Text + KL | 62.3 % | 59.0 / 68.1 | 58.0 / 40.6 / 40.0 | reasoning text, a quarter bare calls |
| Text + KL + cosine | 65.9 % | 62.9 / 71.2 | not decoded | mixed |
| **KL only** | **69.6 %** | 65.5 / 76.7 | 67.5 / 42.2 / 40.0 | bare call |
| **KL + cosine** | **70.0 %** | 66.9 / 75.5 | 65.0 / 35.9 / 39.5 | bare call |

- **The gain goes through the rows.** With wrong rows or no rows every arm stays near 40 %.
- **Two arms closed earlier:** the teacher with its adapter (61.2 %) and weight 2.0 (67.7 % against 65.8 % at weight 0.5 on the same items, inside noise).

## What KL only does

It copies the teacher. The key of an item is never used.

| Pair | Right | Same letter as the teacher | Right where the teacher is right (673) | Right where the teacher is wrong (221) |
|---|---:|---:|---:|---:|
| Start (question run final) | 536 | 548 | 466 | 70 |
| Text only | 543 | 551 | 474 | 69 |
| Text + KL + cosine | 589 | 615 | 521 | 68 |
| KL only, older items | 622 | 692 | 568 | 54 |
| KL only, final set | 636 | 701 | 578 | 58 |
| KL only, rules-only set with explained text | 644 | 713 | 588 | 56 |
| KL only, final set, both batches one after the other | 633 | 708 | 577 | 56 |
| KL only, all 5,600 new questions in one run | 636 | 724 | 587 | 49 |
| KL only, real questions (other articles) | 636 | 678 | 567 | 69 |
| KL only, `g1x` questions (other articles) | 597 | 632 | 530 | 67 |
| The `g1x` pair, then KL only on the older items | 618 | 689 | 562 | 56 |
| Real questions, text + KL + cosine | 658 | 681 | 579 | 79 |
| The same, then KL only on own-article items | 658 | 700 | 588 | 70 |

- **The teacher is the ceiling.** The best pairs recover 85 to 87 % of the teacher's right answers and repeat its wrong letter on about half of its 221 errors.
- **Questions on other articles help where the teacher is wrong** (67 to 69 of 221), although no key is used. Items on the evaluated articles help where the teacher is right.
- **Adding KL only on top of the real finish changes nothing in total** (73.6 % before and after): nine questions gained where the teacher is right, nine lost where it is wrong.
- **The last stage decides.** Starting the own-article KL-only stage from the `g1x` pair gives 69.1 %, the same as starting it from the 60.0 % pair (69.6 %): what the other-article stage had gained where the teacher is wrong (67) is gone (56).
- **More questions of the same kind add nothing.** A second batch of 1,813 new questions on top of the 71.1 % pair gives 70.8 %: 36 questions gained, 39 lost. The recipe is at its level after about 450 updates on about 1,800 questions. All 5,600 new questions in one run of 1,398 updates give 71.1 %: that pair agrees with the teacher most (724 of 894) and so also repeats more of its errors (right on 49 of the 221, the lowest).
- **More of the same drifts down.** Continuing the 69.6 % pair with KL only on 12,593 older items read 67.5, 64.0 and 62.5 % on the 200-question panel at updates 0, 300 and 600. Stopped at update 1,172.
- **An item helps in proportion to the answer positions on which student and teacher still disagree.** Bare-call choice items, one informative position each, gave 64.8 %; the same loss on items with reasoning or prose text gave 69.6 %.

## Training tests

All from the 60.0 % pair unless the row says otherwise.

| Items | Articles | Loss | Updates | Right on 894 |
|---|---|---|---:|---:|
| 2,523 real questions | 150 other | text + KL + cosine | 946 | 73.6 % |
| 2,523 real questions | 150 other | KL only | 946 | 71.1 % |
| 2,523 real questions, reader frozen | 150 other | text | 946 | 69.0 % |
| 2,374 `g1x` questions (outsider-guess wrong options) | 150 other | KL only | 890 | 66.8 % |
| 2,470 rung-1 questions (rules only) | 150 other | text + KL + cosine | 926 | 59.8 % |
| 2,470 rung-1 questions (rules only) | 150 other | KL only | 926 | 60.5 % |
| 5,048 older items with reasoning or prose text | the 50 evaluated | KL only | 631 | 69.6 % |
| 5,386 older choice items, bare call | the 50 evaluated | KL only | 673 | 64.8 % |
| **Final set, first batch: 1,814 questions, teacher's explained answer as text, plus prose twins** | the 50 evaluated | KL only | 453 | **71.1 %** |
| Rules-only set, same treatment: 1,973 questions | the 50 evaluated | KL only | 492 | **72.0 %** |
| The 71.1 % pair, then the second batch (1,813 questions) | the 50 evaluated | KL only | 453 | 70.8 % |
| All new questions in one run: rules-only set and both judged batches, 5,600 questions with explained text plus prose twins | the 50 evaluated | KL only | 1,398 | 71.1 % |
| The `g1x` pair (66.8 %), then the older items | the 50 evaluated | KL only | 631 | 69.1 % |
| The real-question pair (73.6 %), then the older items | the 50 evaluated | KL only | 631 | 73.6 % |

- **Real against ours, like for like (other articles, KL only):** real 71.1 %, `g1x` 66.8 %, rules only 60.5 % (start 60.0 %). The outsider-guess wrong options are what makes our questions useful on other articles: they give about 60 % of the real-question gain, and rules-only questions give none under either loss. Asked as a bare answer call, rules-only questions are answered by the reader as the teacher answers them (KL 0.001 to 0.007 during training), so there is nothing to copy.
- **The fixed text matters more than the question.** With the teacher's two-or-three-sentence explanation as the text, the judged final set reaches 71.1 % in 453 updates and the rules-only set 72.0 % in 492. The two differ on 84 questions, 46 against 38, which is inside noise. So on the evaluated articles the simplest generator is enough, and by the ordering rule it is the one to describe in the paper.

## The question work

Detail of every round is in the plan document ("Probe results so far"). In short:

| What was tried | Untrained reader right with the article | Without | Sound? |
|---|---:|---:|---|
| Real questions (76 measurement articles) | 80.4 % | 43.6 % | – |
| Rules only (`g1`) | 97.8 % | 65.6 % | yes |
| Rules plus real examples, or mirroring a real set | 96 to 98 % | 65 to 72 % | yes |
| Asking for harder questions in several ways | 96 to 99 % | 72 to 78 % | yes |
| Wrong options = an outsider's guesses, restated in the story (`g1x`) | 95.9 % | 44.8 % | yes |
| Seven in-story wrong candidates, hardest three kept; stricter outsider grounding; the reader's own wrong answers as options | 79 to 90 % | 41 to 54 % | no: 30 to 43 % of items did not stand before the judge (a second right option, or a wrong option that restates the key) |
| The same after the 27B judge removed those | 94 to 98 % | 41 to 65 % | yes |

- **Showing real examples added nothing measurable**, so by the ordering rule the final generator uses none: rules, an outsider's guesses, a judge. For training on the evaluated articles even the last two turned out not to be needed (72.0 % with rules only).
- **Still told apart.** A classifier separates every variant from real questions (0.91 to 0.95), and the measurement proposer told 60 of 60 apart by reading.
- **What real questions have that ours lack:** they return to whole-story judgement, and their wrong options are near-misses a reader with the article still falls for. Ours are mostly single-passage look-ups.

## The released set

`IB/TMP/paper_runs/hillclimb/realism/final/release.g1xc.eval50.jsonl`, manifest beside it, built by `make_release.py`.

| Step | Count |
|---|---:|
| Questions written (two batches, six writer calls per article each) | 10,200 |
| Kept by the judge | 5,115 |
| After removing flagged items and repeated stems (the two exports) | 3,627 |
| After removing repeats across batches and items with a kept "partly correct" option | **2,423** |

- **Per article:** 28 to 67 questions, median 49.
- **Reader:** right on 96.2 % with the article, 44.7 % without (real: 80.4 / 43.6).
- **Right option longer / similar / shorter than the wrong ones:** 12 / 61 / 28 % (real: 26 / 54 / 20).
- **Kinds:** 521 "why", 120 judgement, 43 what-if, 28 negation; 2,110 answered from one place.
- **Defects, by the measurement proposer's reading of 80 items:** about 5 % after the "partly correct" items are removed (about 15 % before). Read against the article text only.
- **Each question carries** the teacher's explained answer (median 48 words), used as the text to match and as a prose twin.
- **Cost:** one batch is about 85 minutes on one GPU (write 19, outsider guesses 37, judge 28, explained answers 2).
- **The 894 benchmark questions of these articles were never read** by the writer, the judge, the proposers or any script.
- **The rules-only set** that trained the 72.0 % pair is beside it: `kl_g1.explained.eval50.jsonl`, 1,973 questions with the teacher's explained answer (reader right on 97.2 % with the article, 62.7 % without). It has not been through the judge.

## Answer text, loss, and a forced mix of question kinds (Oct 9)

All arms below start from the 60.0 % pair and train one pass on questions about the 50 evaluated articles; scores are on all 894 benchmark questions (±1.6). The benchmark questions themselves are never trained on.

**Who writes the answer text, and which loss.** Same 1,973 rules-only questions, 492 updates. The teacher is the untrained reader with the whole article (75.3 %); "KL only" trains towards its next-token distribution along the given answer text and never uses the text as a target.

| Answer text | Text loss only | KL only |
|---|---:|---:|
| Teacher's own explained answer | 64.8 % | 72.0 % |
| 27B writer's explained answer, same instruction | 68.8 % | 71.8 % |

| Arm | Right on 894 | Hard / easy | Same letter as the teacher | Teacher right (673): right | Teacher wrong (221): right |
|---|---:|---|---:|---:|---:|
| Teacher text, KL only | 72.0 % | 67.6 / 79.8 | 712 | 588 | 56 |
| Writer text, KL only | 71.8 % | 67.6 / 79.1 | 695 | 578 | 64 |
| Writer text, text loss | 68.8 % | 64.4 / 76.4 | 637 | 538 | 77 |
| Teacher text, text loss | 64.8 % | 59.5 / 73.9 | 614 | 515 | 64 |

- **Under KL the text's author does not matter.** Per question the two KL arms differ on 62 questions, 30 one way and 32 the other; they give the same letter on 815 of 894.
- **KL beats text loss on the same text**: by 7 points on the teacher's text (51 vs 116 per question) and by 3 points on the writer's (63 vs 90).
- **Under text loss the writer's text is better than the teacher's** (80 vs 44 per question), and it is right most often where the teacher is wrong (77 of 221).
- Per-article accuracy over the 50 articles, lowest / quartiles / highest: teacher 50 / 68, 76, 84 / 95; teacher text KL 38 / 63, 73, 81 / 94; writer text KL 38 / 65, 73, 79 / 94; writer text, text loss 44 / 60, 69, 77 / 89; teacher text, text loss 35 / 58, 67, 74 / 89.

**A forced mix of question kinds.** Left alone, the writer asks about one place in the article nine times out of ten. In the forced mix each writer call gets one assignment: one passage, two distant passages to connect, or the whole text; a code check keeps a question only when its copied evidence sentences lie in the assigned passages.

| Question set (teacher text, KL only) | Questions | Reader right with the article / without | Right on 894 |
|---|---:|---|---:|
| Rules only | 1,973 | 98 % / 66 % (same recipe, measured on 76 other articles) | 72.0 % |
| Forced mix, 12 calls per article | 1,873 | 95.9 % / 68.4 % | 71.3 % |

- The two sets score the same within noise (33 vs 40 per question). The mix was kept because it scales without repeating questions: one call per assignment, never two calls for the same one.
- Two-passage calls keep 38 to 41 % of what they write (one of the two copied sentences usually comes from elsewhere). Given only one of the two passages, the reader still answers 71 % and 90 % of the kept two-passage questions; given both, 96 %.
- **Large batch for the runs from reconstruction:** 56 independent calls per article (15 passages, 36 passage pairs, 5 whole-text aspects) wrote 17,300 questions; the passage check and the code dedup kept 10,260, and 9,788 remain after dropping flagged and repeated ones (153 to 264 per article; reader 95.9 % with the article, 63.4 % without).

**From reconstruction only (Oct 10).** Two runs start from the reconstruction-only pair, which has never seen a question, and train one pass on the 9,788 questions with the writer's text: 2,446 updates at a constant rate with a linear landing over the last 500. One uses KL only, the other text loss only.

| Share of the run | Update | KL only | Text loss only |
|---:|---:|---:|---:|
| 20 % | 489 | 62.5 % | 53.7 % |
| 40 % | 978 | 64.4 % | 53.9 % |
| 60 % | 1,467 | 67.3 % | 62.9 % |
| 80 % | 1,956 | 65.4 % | 63.0 % |
| Final | 2,446 | **67.4 %** | **61.7 %** |

| Final pair | Hard / easy | Same letter as the teacher | Median article (quartiles) | 200 questions: rows / no rows / wrong rows |
|---|---|---:|---|---|
| KL only | 61.4 / 77.9 | 683 | 68 % (58, 78) | 64.0 / 40.0 / 43.8 % |
| Text loss only | 56.5 / 70.9 | 587 | 61 % (55, 71) | 55.5 / 44.5 / 34.4 % |

- **KL only ends 5.7 points above text loss** (123 vs 72 per question, z 3.7), and it is ahead at every checkpoint.
- **KL climbs early and flattens.** It is at 62.5 % after a fifth of the run and reaches its final level at 60 %. The last two fifths, the landing included, add nothing measurable (47 vs 46 per question between 60 % and the end).
- **Text loss sits still, then jumps.** It is flat at 54 % for the first two fifths (87 gained, 85 lost between them), gains 9 points in the third, and stays there. Its landing adds nothing either (71 gained, 81 lost).
- **Starting from reconstruction only costs 4 to 7 points against the short runs above**, which started from a pair that had already been through one question-tuning run: 67.4 % against 71.8 % for KL, 61.7 % against 68.8 % for text loss, with five times the questions and updates.
- The 80 % checkpoints were decoded on another machine type than the others; a check on one checkpoint showed 9 of 894 letters differing between the two.

## Restart cycles on fresh questions, and averaging their ends (Oct 10)

The two runs from reconstruction only end at 67.6 % (KL only) and 62.8 % (text loss) when their final pairs are decoded in batches, the path used for every number in this section (one machine per arm, ±1.6). From each final pair, training restarts four times. Each restart is one short pass over questions never used before: about 2,240 to 2,350 new questions, 560 to 587 updates, a 50-update warm-up, a constant rate of 2e-5, a linear landing over the last 150 updates, and a fresh optimizer. The new questions come from two more batches of the forced mix, filtered against everything written earlier (same question, near-same question, evidence sentences already used).

| Pair | KL only | Text loss only |
|---|---:|---:|
| Start (end of the 2,446-update run) | 67.6 % | 62.8 % |
| After restart cycle 1 | 68.0 % | 61.4 % |
| After cycle 2 | 69.5 % | 66.4 % |
| After cycle 3 | 70.1 % | 67.8 % |
| After cycle 4 | 70.9 % | 64.4 % |
| Weights averaged: ends of cycles 3 and 4 | 71.3 % | 66.4 % |
| **Weights averaged: ends of cycles 2, 3 and 4** | **71.4 %** | **69.2 %** |
| Weights averaged: start and all four cycle ends | 70.8 % | 68.0 % |

| Against its own start, per question | Only this pair right | Only the start right | z |
|---|---:|---:|---:|
| KL only, after cycle 4 | 61 | 31 | +3.1 |
| KL only, average of cycles 2 to 4 | 56 | 22 | +3.8 |
| Text loss, after cycle 4 | 80 | 65 | +1.2 |
| Text loss, average of cycles 2 to 4 | 97 | 39 | +5.0 |

- **KL only rises at every cycle**, 3.3 points over the four. Two further cycles show that this is where it stops (next table).
- **Text loss swings from one cycle end to the next** (61.4, 66.4, 67.8, 64.4). Between two consecutive ends 145 to 169 of the 894 letters change; under KL 85 to 100 change.
- **Averaging the weights of the last three cycle ends gives the best pair of each arm** with no further training: 71.4 % for KL, 69.2 % for text loss. It gains about as many questions as the best single pair and loses far fewer of those the start had right. An average is the plain mean of every trained tensor of the pairs.
- **The gap between the losses shrinks from 4.8 points at the start to 2.2 between the two averages.**
- Over the five single pairs of an arm, 700 of 894 questions are answered right by at least one pair (both arms), and 194 by none. KL is right at all five on 524 questions, text loss on 442.
- **One guard was added.** A generated question whose text equals the text of a benchmark question, even about another article, is dropped before training ("What is the tone of the story?" appeared). Checked afterwards on the first cycle's data: none of its 4,696 trained items had such a text.

**The same schedule from the beginning, with no extra questions.** The two 2,446-update runs from reconstruction only are repeated with one change: the schedule is cut into five cycles of 489 updates, each with a 50-update warm-up, a constant rate, a linear landing over its last 150 updates, and a fresh optimizer. Questions, rates and length are those of the original runs.

| Update | KL only, original | KL only, five cycles | Text loss, original | Text loss, five cycles |
|---:|---:|---:|---:|---:|
| 489 | 62.5 % * | 62.8 % | 53.7 % * | 55.7 % |
| 978 | 64.4 % * | 63.6 % | 53.9 % * | 60.5 % |
| 1,467 | 67.3 % * | 65.2 % | 62.9 % * | 62.1 % |
| 1,956 | 65.4 % * | 66.7 % | 63.0 % * | 62.1 % |
| 2,446, final | 67.6 % | **67.2 %** | 62.8 % | **65.4 %** |
| Weights averaged: last two cycle ends | | 67.4 % | | 66.0 % |
| Weights averaged: all five cycle ends | | 66.4 % | | 63.0 % |

\* Decoded one at a time; that path read 0.2 (KL) and 1.1 (text loss) points lower than the batched one on the final pairs. Only the final row compares one decode path on one machine.

- **The schedule alone does nothing for KL** (36 questions only the cyclical pair has right, 39 only the original).
- **It adds 2.6 points to text loss** (70 against 46, z 2.2).
- **Averaging inside one run adds little.** The early cycle ends are much weaker, and averaging them in pulls the result down. Averaging pays after the restart cycles, where the ends sit at one level and differ mostly by swing.
- **Fresh questions are what moved KL in the restart cycles** (67.6 to 70.9 %, 71.4 % averaged). For text loss the schedule gives part of the gain; fresh questions and averaging give the larger part (62.8 to 69.2 %).

**Two more cycles: both arms are flat.** Cycles 5 and 6 use a third batch of fresh questions (1,706 and 1,705). Each arm runs them from the average of cycles 2 to 4; text loss also continues its chain from the end of cycle 4. All pairs are decoded on one machine; the KL average decoded there gives the same letter on all 894 questions as on the machine of the first table.

| Pair | KL only | Text loss only |
|---|---:|---:|
| Average of cycles 2, 3, 4 (start) | 71.4 % | 69.2 % |
| Cycle 5 from the average | 70.5 % | 68.6 % |
| Cycle 6 from that cycle 5 | 69.6 % | 61.6 % |
| Average of the start, cycle 5 and cycle 6 | 70.8 % | |
| Average of the start and cycle 5 | | 68.2 % |
| Chain continued from cycle 4: cycle 5 | lost with its machine | 66.6 % |
| Chain continued from cycle 4: cycle 6 | | 63.2 % |
| Average of cycles 3, 4, 5 of the chain | | 67.7 % |
| Average of cycles 2 to 5 of the chain | | 68.6 % |

- **Neither arm gains from more fresh questions of this kind.** KL loses 1.8 points over the two cycles; no new average beats the average of cycles 2 to 4 in either arm.
- **Text loss wanders.** Its six chain ends read 61.4, 66.4, 67.8, 64.4, 66.6, 63.2 %. A cycle started from the average holds the average's level once and loses 7 points in the next.
- **More KL copies the teacher more closely, errors included.** From the KL average to its cycle 6, the same letter as the untrained reader with the article goes from 717 to 722 questions, and the count right among the 221 questions that reader misses falls from 55 to 45.
- The first machine was a spot instance and was reclaimed during KL cycle 5 of the chain. The KL chain's single pairs went with it; the KL average and all score files were already copied.

## Why the reader with the whole article stops near 75 % (Oct 10)

The reference of every table here is the untrained 9B reader with the whole article, answering at once with the bare call: 673 of 894 right. Its 221 errors were read against the articles by eight judges (Sonnet, one reading per question), and compared with two other small readers given the article (the 4B reader with and without thinking).

| What the question needs | Errors | Judge finds the key clear |
|---|---:|---:|
| A judgment over the whole story (motive, tone, most likely, best summary) | 95 | 21 |
| Combining distant parts, counting, order of events | 65 | 40 |
| One stated detail | 42 | 31 |
| Other, mostly "the text does not say" | 13 | 6 |
| Not / except | 6 | 5 |
| **All** | **221** | **103** (arguable 100, doubtful 18) |

| Why the wrong option attracted | Errors |
|---|---:|
| Partly true: true of another character, another moment, or an early state the story overturns | 61 |
| Sensible in general, nothing in the text supports it | 61 |
| Repeats words or events of the text without answering the question | 55 |
| The literal reading where a twist or irony points elsewhere | 19 |
| No answer: the reader starts a worked list and runs out of tokens (12 of 17 are count or order questions) | 17 |
| Miscount, other | 8 |

- **It is not missing information.** At least one of the three small readers with the article is right on 781 questions (87 %). All three are wrong on 113, with the same wrong option on 64.
- **It is one-pass answering.** The reader takes an option that fits something in the story and does not check the four against each other. Thinking adds 3.4 points to the 4B reader.
- **About half of the errors sit where the key is a matter of reading.** That half moves little with anything we tried.
- **Practice moves the rest.** The same 894 questions, readers given the article:

| Reader, given the article | Right | Of the 103 clear | Of the 100 arguable | Of the 18 doubtful |
|---|---:|---:|---:|---:|
| Untrained | 75.3 % | 0 | 0 | 0 |
| After the synthetic-only question run | 78.5 % | 43 | 22 | 2 |
| After the finish on real training-split questions (a labelled diagnostic) | 81.8 % | 54 | 29 | 2 |

**Our questions do not contain these cases.** On the 44,632 distinct synthetic choice questions written so far, the untrained reader with the article picks the key on 93.9 % (96.2 % on the cycle-5 set), against 75.3 % on the benchmark; 54.6 % are right with nothing in context. The cycle sets hold 4.5 % judgment questions; 43 % of the benchmark errors are of that kind. And KL has the one-pass reader as its target, so on the few hard items it teaches the reader's answer.

**A hard pool.** In the three forced-mix batches the reader with the article misses 790 questions. The 27B writer answered each of them twice, with reasoning, with the article and without the key:

| Blind writer, two passes, on the 790 (788 answered twice) | Questions |
|---|---:|
| Gives the key both times | 484 |
| ... and names no second defensible option in either pass | 437 |
| Gives the reader's option both times (the key is probably wrong) | 160 |
| The two passes disagree | 110 |

Four in ten of the questions the reader misses have a wrong or contestable key, so "hard" items are usable only after this check. The pool is the 423 of the 437 that have a written explained answer (846 rows with their prose twins).

| Pair, 894 questions | Right | Of the 673 the untrained reader has right | Of the 221 it misses |
|---|---:|---:|---:|
| KL average of cycles 2 to 4 (start of the next two) | 71.4 % | 583 | 55 |
| Plus the cycle-5 questions, KL only, 426 updates | 70.5 % | 578 | 52 |
| Plus the hard pool alone, text loss, 105 updates | 67.0 % | 530 | 69 |
| Text-loss average of cycles 2 to 4 | 69.2 % | 545 | 74 |

- **The hard questions carry what the others lack.** 105 updates on them move 14 more of the 221 to right.
- **Alone they cost 53 of the 673.** Every text-loss pair shows that loss; KL holds those questions.
- **So each loss gets its own items in one run:** KL on the regular questions, the key on the hard pool. The packet builder now takes a row flag for that (a flagged item is built without its in-context view, so the trainer gives it no match term and keeps its text loss; no change to the training package).

**First cycle with both losses: the balance was wrong.** Same start, the cycle-5 questions by KL plus the hard pool on its key (a fifth of the examples), match weight 0.5 as in every KL run:

| Pair, 894 questions | Right | Of the 673 the untrained reader has right | Of the 221 it misses |
|---|---:|---:|---:|
| Cycle-5 questions, KL only | 70.5 % | 578 | 52 |
| Cycle-5 questions by KL, hard pool on the key | 68.6 % | 542 | 71 |
| Text-loss average of cycles 2 to 4 | 69.2 % | 545 | 74 |

- **It behaves like a text-loss pair.** It gains 16 of the 221 and loses 41 of the 673.
- **The log shows why.** Over the cycle the KL to the reader with the article stays at 0.19 on the regular questions; in the KL-only twin on the same questions it falls from 0.18 to 0.14. The text loss falls from 0.76 to 0.61. One flagged example weighs about 0.7 in the loss and one matched example about 0.5 x 0.19, so a fifth of the examples carries about two thirds of the loss.
- **A heavier KL term does not rescue it.** The same cycle with the match weight at 5.0 and a larger hard pool (699 confirmed questions: 276 more from a batch of 16,258 questions on the 76 other training-split articles, where the reader misses 555 and the two-pass check confirms 288):

| Pair, 894 questions | Right | Of the 673 the untrained reader has right | Of the 221 it misses | Same letter as that reader |
|---|---:|---:|---:|---:|
| Cycle-5 questions, KL only | 70.5 % | 578 | 52 | 718 |
| Plus 699 hard questions on the key, match weight 5 | 68.3 % | 551 | 60 | 678 |
| Plus 423 hard questions on the key, match weight 0.5 | 68.6 % | 542 | 71 | 650 |
| Text-loss average of cycles 2 to 4 | 69.2 % | 545 | 74 | 642 |

- **The pairs lie on one line.** The further a pair moves from the answers of the reader with the article, the more it loses on the 673 and the more it gains on the 221, and the loss is the larger of the two at every point. Questions picked because that reader fails them push the pair away from it, and the pair gets most of its right answers by following it.
- **So the KL-only average of cycles 2 to 4 (71.4 %) stays the best pair,** and the hard questions do not pay as a second loss on the pair. What is left of the plan is the other side: a better reader with the article as the KL target. From these counts (583 of 673 kept, 55 of 221 found), a target at 78.5 % would give about 73 % with rows.

**The trained readers as teachers (Oct 11).** Each averaged pair's reader, with its adapter, given the whole article in place of the rows (standard cap of 480 new tokens):

| Reader, given the article | Right | Of the 673 | Of the 221 | Of the 103 clear | No answer |
|---|---:|---:|---:|---:|---:|
| Untrained (the teacher of every KL run so far) | 75.3 % (673) | 673 | 0 | 0 | 17 |
| Reader of the KL average pair | 77.9 % (696) | 649 | 47 | 29 | 0 |
| Reader of the text-loss average pair | 75.3 % (673) | 597 | 76 | 51 | 88 |
| The same reader, cap of 1,024 new tokens | 77.6 % (694) | 608 | 86 | 59 | 59 |

- **Both are better teachers than the untrained reader on the questions it misses.** The text-loss reader is right on 83.4 % of the 806 questions it answers.
- **The text-loss reader deliberates past the cap.** It writes its reasoning before the call, and on 88 questions the 480 tokens end first; at 1,024 tokens 59 are still open. The open outputs are not loops: the reader weighs the options, doubts its reading and starts again.
- **For the best pair, the whole gap between rows and article is on the 673.** With the article its reader has 649 of them, with rows 583. On the 221 the rows are not behind (55 against 47). So the limit of that pair is what the rows carry to its own reader.

**A cycle toward a fixed trained teacher.** The trainer has an option that makes the teacher the reader with its current adapter and the article. With the reader's rate at 1e-12 the reader stays as it is, the teacher is fixed, and only the encoder side trains (2e-5). No change to the training package. First run: from the text-loss average pair, the cycle-5 questions, 426 updates. The KL to the teacher falls from 0.67 to 0.19.

| Pair, 894 questions | Token cap | Right | Of the 673 | Of the 221 | No answer |
|---|---:|---:|---:|---:|---:|
| KL average of cycles 2 to 4 | 480 | 71.4 % (638) | 583 | 55 | 1 |
| Text-loss average of cycles 2 to 4 (the start) | 480 | 69.2 % (619) | 545 | 74 | 1 |
| Fixed teacher, cycle 5 | 480 | 68.1 % (609) | 546 | 63 | 100 |
| Fixed teacher, cycle 5 | 1,024 | 70.4 % (629) | 558 | 71 | 66 |

- **The pair moves onto its teacher, habits included.** Where it answers within 480 tokens it is right on 76.7 % (794 questions); where pair and teacher both answer (738) they give the same letter on 622. It also copies the long deliberation: 100 questions end without an answer, against 1 for its start.
- **A larger cap gives back 20 answers, not all.** At 1,024 tokens 66 questions are still open. The pair is then level with the KL average (78 questions gained, 87 lost against it), not above it.
- **Where both answer, the pair is close to its teacher.** At the 1,024 cap pair and teacher both answer 789 questions and give the same letter on 653; the teacher is right on 658 of them, the pair on 607. Of the 103 clear-key errors of the untrained reader the pair has 46 right, the KL average 29.
- **The same cycle from the KL average pair changes nothing.** Its reader with the article answers every question at once and is right on 77.9 %. With that reader as the fixed teacher:

| Pair, 894 questions | Right | Of the 673 | Of the 221 | Same letter as this reader with the article |
|---|---:|---:|---:|---:|
| KL average of cycles 2 to 4 (the start) | 71.4 % (638) | 583 | 55 | 753 |
| Cycle 5, untrained teacher, both sides train | 70.5 % (630) | 578 | 52 | 748 |
| Cycle 5, fixed trained teacher, encoder side only | 70.5 % (630) | 576 | 54 | 748 |

- **So the teacher is not what limits the pair.** A teacher at 77.9 % in place of 75.3 % leaves the pair where it was. It gives its reader-with-article's letter on about 750 of 894 questions either way.

## What the rows lose (Oct 11)

Take the KL average pair's reader with the article as the reference: 696 questions right. Five KL pairs of that line (the average, the two cycles after it, their average, the fixed-teacher cycle) with rows:

| Of the 696 the reader has right with the article | Questions |
|---|---:|
| Right with rows in all five pairs | 565 |
| Right in some | 67 |
| Right in none | 64 (the same wrong letter in all five on 58) |

The 64 are a stable set, not noise. Six judges (Sonnet) read them against the articles, mixed with 64 questions from the same articles that all five pairs keep, without knowing which was which:

| Judged property | The 64 lost | The 64 kept |
|---|---:|---:|
| A faithful 300-word summary would be enough: yes | 8 | 19 |
| ... no | 21 | 8 |
| The evidence is one sentence | 13 | 4 |
| The deciding fact is a number, a time or exact wording | 14 | 5 |
| The wrong option is a close value or variant of the right one | 8 | 3 |
| Needs one stated detail | 27 | 24 |
| Needs a judgment over the whole story | 15 | 18 |
| Needs combining distant parts | 14 | 15 |

- **The rows keep the gist and lose fine detail.** The lost questions are not a different kind of question; they turn on a single sentence, a number or the wording.
- **The training log says the same.** The pair that starts the question stage reconstructs whole articles almost word for word (0.002 per token). The question stage trains no reconstruction, and after it the same measure stands at 0.6 to 0.75.
- **Decoding the same pair at 1 row per 10 tokens, with no training at that ratio, gives 70.5 %** (71.4 % at 1/20). The pair tolerates the other ratio; more rows alone do not bring the detail back.
- **Training at 1 row per 10 tokens does not help either.** The cycle-5 questions, KL only, from the KL average pair, once with rows at 1/20 and once at 1/10 (a diagnostic; the paper setting stays 1/20):

| Pair, 894 questions | Right | Of the 673 | Of the 221 | Of the 64 lost | Of the 64 kept |
|---|---:|---:|---:|---:|---:|
| KL average of cycles 2 to 4, rows at 1/20 | 71.4 % (638) | 583 | 55 | 0 | 64 |
| The same pair decoded at 1/10 | 70.5 % (630) | 577 | 53 | 10 | 60 |
| Cycle 5 at 1/20 | 70.5 % (630) | 578 | 52 | 0 | 64 |
| Cycle 5 at 1/10 | 69.6 % (622) | 576 | 46 | 8 | 61 |

- **Twice the rows bring back 8 to 10 of the 64 and lose as many elsewhere.** The limit is not the number of rows. What the encoder writes into them is the open point.
- **Reconstruction mixed into the cycle restores the text in the rows and does not help the answers.** The same cycle at 1/20 with 1,136 reconstruction requests on the 50 articles added (a quarter of the examples):

| Pair, 894 questions | Right | Of the 673 | Of the 221 | Of the 64 lost | Whole-article reconstruction, per token |
|---|---:|---:|---:|---:|---:|
| KL average of cycles 2 to 4 (the start) | 71.4 % (638) | 583 | 55 | 0 | 0.753 |
| Cycle 5, questions only | 70.5 % (630) | 578 | 52 | 0 | 0.789 |
| Cycle 5 with reconstruction mixed in | 67.7 % (605) | 551 | 54 | 4 | 0.031 |

- **The text is back in the rows after 250 updates (0.053) and the pair answers worse.** It loses 27 of the 673 and gives its reader-with-article's letter on 715 questions instead of 748. During the cycle the KL on the questions stays near 0.20 where the questions-only cycle falls to 0.14.
- **This repeats the Oct 8 finding from the other side.** Then, the pair that only reconstructs recited 14 of the 50 articles word for word and the rest with 0.08 % of the words different, and answered 61 %. Having the content is not the limit; reaching it while answering is.

**Three ways out are now closed:** a better reader as the KL target, more rows, and keeping the text in the rows. What is left is the pressure to read the rows at all.

**Two thirds of our training questions need no rows.** In every cycle file the untrained reader with nothing in context picks the key on 65 to 66 % of the choice questions (on the 894 benchmark questions: 30.6 %, with 288 left unanswered). Those items train the pair to match a teacher it already matches. A cycle on 1,704 questions that the reader has right with the article and wrong with nothing (and their prose twins), with the same number of rows and updates as the questions-only cycle, does not move the score either:

| Pair, 894 questions | Right | Of the 673 | Of the 221 | Of the 64 lost |
|---|---:|---:|---:|---:|
| KL average of cycles 2 to 4 (the start) | 71.4 % (638) | 583 | 55 | 0 |
| Cycle on all questions of cycle 5 | 70.5 % (630) | 578 | 52 | 0 |
| Cycle on non-guessable questions only | 70.6 % (631) | 574 | 57 | 6 |

**Where this leaves the direct-answer pair.** Five single cycles from the same start (another teacher, twice the rows, reconstruction mixed in, non-guessable questions, the plain cycle) end between 67.7 and 70.6 %, none above the plain cycle. One more cycle of anything of this kind does not move a pair that answers at once.

**The one lead that is left is the pair that deliberates.** The fixed-teacher pair from the text-loss reader writes its reasoning before the call. On the 828 questions it answers within 1,024 tokens it is right on 76.0 %; the KL average is right on 73.2 % of the same questions. Its 66 unanswered questions cost it the lead: the KL average has 32 of them right. If the deliberating pair were asked for its letter when the tokens run out, it would stand at 73 to 74 % with a 40 to 50 % hit rate on those 66.

## Asking for the letter when the tokens run out (Oct 11)

One rule added to the decode: when a reply reaches the token cap without the `submit_answer` call, the reader gets its own text back with the call started for it and writes the letter. One short pass per unanswered question, greedy, same pair, same rows or article. A reader that calls by itself is untouched (the KL average pair has 1 such question in 894).

| Reader, 894 questions | Token cap | Right as decoded | No answer | Right with the letter asked for | Of the 673 | Of the 221 |
|---|---:|---:|---:|---:|---:|---:|
| KL average pair, rows (answers at once) | 480 | 71.4 % (638) | 1 | 71.4 % | 583 | 55 |
| Deliberating pair, rows | 480 | 68.1 % (609) | 100 | **72.8 % (651)** | 576 | 75 |
| Deliberating pair, rows | 1,024 | 70.4 % (629) | 66 | **73.3 % (655)** | 576 | 79 |
| Deliberating pair after a second fixed-teacher cycle, rows | 480 | 68.8 % (615) | 108 | **73.7 % (659)** | 571 | 88 |
| Deliberating pair after a third fixed-teacher cycle, rows | 480 | 67.7 % (605) | 101 | **72.8 % (651)** | 573 | 78 |
| **Average of the three fixed-teacher cycle ends, rows** | 480 | 69.0 % (617) | 91 | **74.3 % (664)** | 583 | 81 |
| Its teacher: the text-loss reader with the article | 480 | 75.3 % (673) | 88 | **81.0 % (724)** | 629 | 95 |
| The same | 1,024 | 77.6 % (694) | 59 | **81.3 % (727)** | 629 | 98 |

- **A reader trained on our synthetic explained answers passes the 75 % line by six points when it has the article.** 81 % is the level of the diagnostic that finished on real training-split questions (81.8 %), reached here without a real question. The plain decode hid it because one reply in ten ran out of tokens. (Later on Oct 11: asked for its letter at once this reader is at 79.9 %, so five of the six points do not need the reasoning; see "Asked for the letter at once".)
- **So the 75 % of the untrained reader is trained away on the reader's side.** Of the 221 questions the untrained reader misses, this reader has 95 to 98 right. (Later on Oct 11: asked at once it stands at 79.9 %, so the training of the reader carries the gain and the reasoning adds about one point; see "Asked for the letter at once".)
- **With rows, the pair trained toward that reader for one cycle stands at 72.8 to 73.3 %,** the best pair so far that started from reconstruction and saw no real question (the KL average: 71.4 %; 82 to 84 questions gained against it, 67 to 69 lost). It gains on the 221 (75 to 79 against 55) and holds the 673 almost level (576 against 583).
- **The letters asked for are right 39 to 42 % of the time with rows and 59 % with the article.** These are the questions the reader found hard enough to keep deliberating.
- **The cap matters little once the letter is asked for:** 72.8 % at 480 tokens, 73.3 % at 1,024.
- **Little of the 81 % sits in the reader's weights.** This reader has trained on synthetic questions about these 50 articles. With nothing in context (no rows, no article) it is right on 43.2 % of the 894 as decoded and 44.4 % with the letter asked for; the untrained reader with nothing in context is right on 30.6 % (274) and leaves 288 unanswered; if it guessed those at the rate of chance it would stand near 39 %.
- **Not yet measured with the rule:** the untrained reader with the article (17 unanswered; the rule can add at most 1.9 points to its 75.3 %) and the readers of the other pairs.
- **The distance from rows to article is 7 to 8 points on this line (73 to 74 against 81), and a second cycle toward the same teacher does not close it.** Cycle 6 (`hccft2_sft_c6_run_u425`) ends at 73.7 % with the letter asked for: 58 questions gained and 50 lost against cycle 5 (z 0.8), 94 gained and 73 lost against the KL average (z 1.6). It picks the teacher's letter on 699 of 894 (cycle 5: 709; the KL average pair: 680) and is right on 608 of the 724 the teacher has right (cycle 5: 609). The gain is among the 221 (88 against 75), outside what the teacher has right. The two cycle ends differ on 108 questions, and 709 questions are right in at least one of them, so the next step is the average of the cycle ends. Cycle 7 (`hccft2_sft_c7_run_u560`, the questions of cycle 4) ends at 72.8 %: 51 gained and 59 lost against cycle 6, same letter as the teacher on 706. The three cycle ends stand at 72.8, 73.7 and 72.8 %, each 1.4 to 2.3 points above the KL average; 568 questions are right in all three, 729 in at least one, and the letter two of the three agree on is right on 664 (74.3 %). Cycle 8 was stopped at its start to free the node.
- **The average of the three cycle ends is the best pair so far that saw no real question: 74.3 % (664) with the letter asked for** (`hccft_sft_avg_c5_to_c7_run_u425`; only the encoder side differs between the ends, the reader is the same frozen adapter). Against the KL average pair it gains 92 questions and loses 66 (z 2.1). It holds all of the 673 that the KL average holds (583) and adds 26 of the 221 (81 against 55). It picks its teacher's letter on 716 of 894 and is right on 619 of the 724 its teacher has right. Its 91 asked letters are right 47 times. It is one point under the untrained reader with the whole article (75.3 %) and above the pair trained on real questions with their keys (73.6 %, decoded without the rule).

**The gain sits where the reader reasons.** The untrained reader with the article writes nothing before its call on 98 % of the questions. The text-loss reader writes reasoning on 530 of the 894 and answers at once on 364, and it picks the harder questions to reason on:

| Questions, readers with the article | Questions | Text-loss reader, letter asked for | Untrained reader | KL average reader |
|---|---:|---:|---:|---:|
| It wrote reasoning before the call | 530 | **76.4 %** | 68.1 % | 71.7 % |
| Of those: it called by itself | 443 | 79.9 % | 72.0 % | 76.3 % |
| Of those: it reached the cap and was asked | 88 | 59.1 % | 48.9 % | 48.9 % |
| It answered at once | 364 | 87.6 % | 85.7 % | 86.8 % |
| All | 894 | 81.0 % | 75.3 % | 77.9 % |

Of the 51 questions it gains on the untrained reader, 44 are among those it reasoned on.

**Which of the 221 the reasoning recovers.** The 221 questions the untrained reader misses with the article were judged on Oct 10 (is the key clear; what drew the reader to its wrong option). Right answers among them, token cap 480, letter asked for:

| Judged class of the 221 | Questions | KL average reader, article | Text-loss reader, article | Deliberating pair, rows |
|---|---:|---:|---:|---:|
| Key is clear | 103 | 29 | **63** | 48 |
| Key is arguable | 100 | 17 | 27 | 23 |
| Key is doubtful | 18 | 1 | 5 | 4 |
| Wrong option was a plausible guess | 61 | 10 | **32** | 29 |
| Wrong option echoes the text | 55 | 12 | **25** | 16 |
| Wrong option is true but partial | 61 | 8 | 15 | 12 |
| Needs distant parts combined | 64 | 18 | **33** | 21 |
| Needs a judgment on the whole story | 95 | 17 | 31 | 28 |
| Needs one stated detail | 42 | 8 | 20 | 16 |
| All | 221 | 47 | 95 | 75 |

- **Reasoning before the call recovers the misses that had a clear key:** 63 of 103, against 29 for the reader that answers at once. It stops the plausible guess (32 of 61) and the option that repeats words of the text (25 of 55), and it combines distant parts (33 of 64).
- **It does not recover the arguable keys** (27 of 100) or the true-but-partial options (15 of 61). Of the 126 questions of the 221 that the text-loss reader still misses, 86 have an arguable or doubtful key and 40 a clear one.
- **With rows the deliberating pair keeps most of the gain on plausible guesses (29 of 61) and loses it where the text itself is needed:** options that echo the text (16 against 25) and distant parts (21 against 33). This is the same loss as in "What the rows lose".

## Asked for the letter at once: the written reasoning is not what carries the gain (Oct 11, late)

"Asked at once" means the answer call is started right after the prompt and the reader writes only the letter, 12 tokens. That is what a harness with a required answer call does. The finished decodes were also read as if the token cap had been lower: a reply longer than the cap is cut there and the letter is asked for. Check of the method: the KL average pair, which answers at once by itself, gives the same letter on 885 of 894 questions when asked this way (71.5 % against 71.4 %); the untrained reader with the article gives the same letter on 873 of the 877 questions it answered by itself.

| 894 questions | Asked at once | Cap 120 | Cap 240 | Cap 480 | Decoded by itself, cap 480 |
|---|---:|---:|---:|---:|---:|
| Averaged mined pair, rows at 1/20 | **75.1 % (671)** | 75.3 % | 75.3 % | 75.2 % (672) | 69.5 % (621), 104 cut |
| Text-loss average pair, rows (its start two stages back) | 67.4 % (603) | – | – | 69.2 % | 69.2 % (619), 1 cut |
| KL average pair, rows | 71.5 % (639) | – | – | 71.4 % | 71.4 % (638), 1 cut |
| Text-loss reader with the article | **79.9 % (714)** | – | 80.8 % (722) | 81.0 % (724) | 75.3 % (673), 88 cut |
| Untrained reader with the article | **76.4 % (683)** | – | – | – | 75.3 % (673), 17 without an answer |

- **The like-for-like line of the untrained reader with the article is 76.4 %, not 75.3 %.** Asked at once it is right on 10 of the 17 questions it had left unanswered. Under a required answer call the best pair with rows (75.1 %) is 1.3 points under it: 66 questions gained, 78 lost, inside noise.
- **With rows the reasoning text adds nothing.** The averaged mined pair is right on 75.1 % asked at once and on 75.2 to 75.3 % with 120, 240 or 480 tokens of its own reasoning first (42 questions gained and 43 lost between at once and 480). A required answer call gives the full score in 12 tokens; no rule about the cap is needed.
- **With the article the reasoning adds one point at most.** The text-loss reader is right on 79.9 % asked at once and on 81.0 % with its reasoning (33 gained, 43 lost the other way round; z 1.1). The reasoning moves answers: it has 95 of the 221 instead of 74 and 629 of the 673 instead of 640. By judged class of the 221: with a clear key 63 of 103 with reasoning and 46 at once; with an arguable key 27 and 25 of 100.
- **So what takes the reader with the article from 76 to 80 % is its training, not the act of reasoning.** Against the untrained reader asked the same way it gains 64 questions and loses 33 (z 3.1). This corrects the reading of the section before ("trained away by reasoning before the call"). Most of it needs the article: of the 64 questions it gains, the trained reader has 23 right with nothing in context (36 %), against 45 of all 211 questions the untrained reader misses (21 %). So a part, on the order of ten questions or one point, may be a better guess at how such questions go and not better reading.
- **The gain of the fixed-teacher stage is in the rows.** Asked at once, the averaged mined pair gains 104 questions on the text-loss average pair and loses 36 (z 5.7); against the KL average pair 78 and 46 (z 2.9). It picks the letter of the text-loss reader with the article (asked at once) on 764 of 894 questions, 85.5 %; the KL average pair on 720, the text-loss average pair on 655.
- **The distance from rows to article under the same harness and reader is 5 points:** 75.1 % against 79.9 %. The KL line has the same distance to its own teacher: 71.5 % against 76.4 %.

**The reader's gain appears with the first text-loss training and then stays.** Readers with the whole article, asked at once:

| Reader | Right | Of the 673 | Of the 221 |
|---|---:|---:|---:|
| Untrained | 76.4 % (683) | 671 | 12 |
| KL average reader (trained only to match the untrained reader) | 77.6 % (694) | 649 | 45 |
| Text-loss run before the restart cycles, average of epochs 3 to 5 | 80.0 % (715) | 647 | 68 |
| Text-loss cycle 1 | 79.9 % (714) | 643 | 71 |
| Text-loss cycle 4 | 78.9 % (705) | 640 | 65 |
| Text-loss average of cycles 2 to 4 (the teacher of the fixed-teacher stage) | 79.9 % (714) | 640 | 74 |
| One and two more plain text-loss cycles | 79.3 % (709), 79.9 % (714) | 641, 644 | 68, 70 |

- **Six readers along the text-loss path stand between 78.9 and 80.0 %.** More text-loss cycles on our explained answers will not raise the teacher. For comparison, a reader trained on the real questions of the training split reached 81.8 % with the article.
- **They differ on single questions.** Two of them give the same letter on 837 to 857 questions. At least one of four is right on 745 (83.3 %), all four on 677; the majority letter of four is right on 712, no better than one.

**Where the 5 points between rows and article sit.** Six text-loss readers with the article (the table above without the untrained and the KL reader) give one and the same letter on 785 of the 894 questions and differ on 109. The averaged mined pair with rows against its teacher, both asked at once:

| Questions | Count | Teacher with the article right | Pair with rows right | Pair gives the teacher's letter |
|---|---:|---:|---:|---:|
| The six readers give one letter | 785 | 663 (84.5 %) | 621 (79.1 %) | 699 (89.0 %) |
| The readers differ | 109 | 51 (46.8 %) | 50 (45.9 %) | 65 (59.6 %) |

- **The whole distance is 86 questions.** Where the readers differ among themselves, rows and article score alike, near chance for both. Where the readers are sure, the pair leaves their letter on 86 questions: the teacher is right on 60 of them and the pair on 18. That is 42 questions, the 4.7 points between 75.1 and 79.9 %.
- **So the target is narrow:** questions on which the reader with the article is stable and the rows still give another letter. That is 11 % of the stable questions. For comparison, the KL average pair left its own teacher on 8 % of our generator's fresh questions that need the article (next section); that count did not separate stable from unstable questions.

**Cycle ends are stable when the letter is asked at once.** The three cycles on mined questions, rows at 1/20:

| How the letter is obtained | Right, cycle 5 / 6 / 7 | Letter changes 5 to 6 | 6 to 7 | Right in all three | In at least one | Average of the three ends |
|---|---:|---:|---:|---:|---:|---:|
| Asked at once | 73.9 / 73.8 / 73.9 % | 49 | 46 | 628 | 692 | 75.1 % |
| Reasoning first, letter asked for at the cap | 74.6 / 73.8 / 75.7 % | 111 | 120 | 598 | 731 | 75.2 % |

- **More than half of the answer churn between cycle ends is in the written reasoning.** A further cycle moves 5 % of the letters when asked at once and 13 % with reasoning.
- **The ends no longer rise.** Three cycles on mined questions end within one question of each other. Averaging the three ends is worth 1.2 points; the average of the three plain fixed-teacher ends stands at 74.0 % asked at once, and the average of all six ends at 74.3 %, so averaging more ends adds nothing.
- **More rows without reasoning do not help.** The averaged pair read at 1/10, a ratio it was not trained at, is right on 73.4 % asked at once (76.3 % with reasoning). After one cycle at 1/10 it is right on 75.3 % at 1/10 and on 74.9 % at 1/20, asked at once.

**A cycle on real questions does not close the distance either (diagnostic, Oct 12).** One fixed-teacher cycle from the averaged mined pair on the 2,523 real questions of the training split (150 other articles, 315 updates). Their keys are not trained on: the encoder side is trained only toward the teacher's distribution on them. Rows at 1/20, asked at once:

| Pair | Right | Same letter as the teacher with the article |
|---|---:|---:|
| Averaged mined pair (the start) | 75.1 % (671) | 764 |
| One cycle on real training-split questions (`hccftreal_kl_c8_run_u315`) | 75.2 % (672) | 750 |
| One cycle on mined questions of our own, trained at 1/10 | 74.9 % (670) | 748 |
| Average of all six fixed-teacher cycle ends | 74.3 % (664) | 754 |

- **Questions written by the benchmark's own annotators, on other articles, move nothing:** 34 questions gained, 33 lost, and no more agreement with the teacher.
- **Five things have now left the 5 points where they are:** more fixed-teacher cycles, cycles on mined questions, real questions on other articles, twice the rows, and averaging more ends. A sixth, training the reader too, made it worse (below). What the question stage can do for this encoder and this reader looks done at about 75 %.

**Letting the reader train in this stage loses the gain (Oct 12).** Every fixed-teacher cycle so far trained the encoder side only. One cycle from the averaged mined pair with the same options but the reader trained too, at the recipe's rate (`hccftrd_kl_c8_run_u560`, 560 updates on plain questions). The teacher is then the reader with its current adapter and the article. Asked at once:

| Pair | Rows at 1/20 | The same reader with the article | Distance |
|---|---:|---:|---:|
| Averaged mined pair, reader frozen since the text-loss stage (the start) | 75.1 % (671) | 79.9 % (714) | 4.8 |
| One more cycle, reader frozen (two such cycles above) | 74.9 %, 75.2 % | 79.9 % | 4.7 to 5.0 |
| One more cycle, reader trained too | **71.7 % (641)** | 78.9 % (705) | 7.2 |
| KL average pair (reader trained throughout, teacher: the untrained reader) | 71.5 % (639) | its teacher: 76.4 % | 4.9 |

- **With rows the pair loses 3.4 points in one cycle** (22 questions gained, 52 lost, z 3.5) while its KL to the teacher on our training questions keeps falling (0.174 to 0.112 per token over the cycle). The reader with the article barely moves (14 gained, 23 lost).
- **So the reader fits our questions, not the rows.** A reader that is free to train learns to give the teacher's answer on questions of our generator and gets worse at reading the rows on the benchmark's. Decoded by itself the pair also stops reasoning and answers at once (71.6 %, none unanswered).
- **The frozen reader is part of what the fixed-teacher stage is.** The stage differs from the KL-only line in two ways, a better teacher and a frozen reader. This cycle keeps the teacher and frees the reader, and lands on the KL-only line's score. Train the encoder side alone in that stage.

## Rows agree with the teacher on our questions, less on the benchmark's (Oct 11)

Every KL pair picks its teacher's letter on about 80 % of the 894 questions, whatever it was trained on. The KL average pair (trained on the questions of cycles 2 to 4; teacher: the untrained reader with the article) was decoded with rows on 1,800 choice questions of our own generator: 900 from the file of cycle 4, which it trained on, and 900 from the file of cycle 6, which it never saw. Half of each set are questions the untrained reader has wrong with nothing in context.

| Questions, KL average pair with rows | Questions | Same letter as the teacher |
|---|---:|---:|
| Benchmark, all | 894 | 80.3 % |
| Benchmark, wrong with nothing in context | 620 | 74.7 % |
| Our generator, trained on (cycle 4), wrong with nothing in context | 450 | 86.9 % |
| Our generator, never seen (cycle 6), wrong with nothing in context | 450 | 88.7 % |
| Benchmark, wrong with nothing in context and the teacher is right | 424 | **80.7 %** |
| Our generator, trained on, the same two conditions | 416 | 90.9 % |
| Our generator, never seen, the same two conditions | 411 | **92.0 %** |
| Benchmark, right with nothing in context | 274 | 93.1 % |
| Our generator, never seen, right with nothing in context | 450 | 96.2 % |

- **The pair does not memorize questions.** Questions it trained on and fresh ones from the same generator come out alike (90.9 and 92.0 %). More questions of the same kind cannot help, which is what the cycles showed.
- **The rows carry what our generator asks for, and less of what the benchmark asks for.** Among questions that need the article and that the teacher gets right, the pair follows the teacher on 92 % of ours and 81 % of the benchmark's. Eleven points on 424 questions are about 47 questions, 5 points of the 894.
- **Our questions are also easier with the article:** the untrained reader with the article has the key on 91 to 92 % of our questions that need the article, and on 68 % of the benchmark's.
- **So the distance from rows to article is a distance between question sets, not a limit the pair reaches on every kind of question.** The training questions have to ask what the benchmark's ask: the judged read above says one sentence of evidence, a number, a time or exact wording, and wrong options with close values.
- **On all 11,247 distinct questions of ours that need the article, the pair leaves its teacher on 1,475 (13.1 %).** The teacher has the key on 1,073 of them; the pair has the key instead on 310. The rate depends on how the question was written: 10.8 % for the rules-only generator (3,579 questions), 16.8 % for the judged final set (1,760), 18 to 19 % for the variant whose wrong options are an outsider's guesses (1,914). The benchmark's rate on such questions is 25 %. Questions built to look real are closer to the benchmark on this measure too.

**A cycle on the questions where the pair leaves its teacher.** From the KL average pair, KL only, the same rates, 3,413 rows: 1,175 of those questions with their prose twins (300 held out), filled with fresh questions of cycle 5's file to the size of the plain cycle.

| Single cycle from the KL average pair (71.4 %, 638) | Right on 894 | Of the 673 | Of the 221 | Same letter as the teacher | Held-out questions it had left (300): teacher's letter now | Held-out questions it had followed (600): still the teacher's letter |
|---|---:|---:|---:|---:|---:|---:|
| Plain cycle, all questions of cycle 5 | 70.5 % (630) | 578 | 52 | 719 | 14.7 % | 97.0 % |
| Questions that need the article only | 70.6 % (631) | 574 | 57 | 707 | – | – |
| **Questions where the pair left its teacher** (`hccmine_kl_c5_run_u425`) | **71.8 % (642)** | 589 | 53 | **729** | **23.7 %** | 98.3 % |

- **It is the first single cycle on this line that does not end under its start.** Against the plain cycle it gains 36 questions and loses 24 (z 1.5); against its start 31 and 27. One cycle is not decisive on the benchmark.
- **It does teach the pair to follow its teacher where it did not.** On 300 such questions kept out of training, the pair now picks the teacher's letter on 23.7 %; the plain cycle on 14.7 % (a question chosen because one pair missed it comes out better for any other pair, so the plain cycle is the control). On the benchmark it picks the teacher's letter on 729 questions, the most of any KL pair (690 to 719).
- **Most of those questions stay missed, even the ones trained on.** After the cycle the pair was decoded on the whole pool again: it leaves its teacher on 1,178 of the 11,247 questions (10.5 %; before: 1,475, 13.1 %). Of the 1,175 questions it trained on it now follows the teacher on 445 (38 %); of the 300 held out on 71 (24 %); 219 questions it used to follow are new misses. One pass over a missed question fixes it in two cases out of five. They are a hard residue, not a gap that one pass of the right questions closes.
- **More rows bring some of them back without any training, but not most.** The same KL average pair, trained at 1 row per 20 tokens, decoded on the 300 held-out misses with more rows: it follows the teacher on 23.0 % at 1/10 and on 30.3 % at 1/5 (14.7 % is what a plain cycle brings back by chance; the mined cycle 23.7 %). On the 600 questions it followed, it still follows on 97.5 % and 96.0 %. Part of the residue is room in the rows; seven in ten stay missed with four times the rows.
- **On the deliberating line the same questions hold the level of the average and raise agreement with the teacher.** One fixed-teacher cycle on the mined file from the 74.3 % average pair (`hccftm_sft_c5_run_u425`) ends at **74.6 % (667)** with the letter asked for (68.7 % as decoded, 101 replies at the cap): 41 questions gained and 38 lost against its start, 97 and 68 against the KL average (z 2.3). It picks its teacher's letter on 725 questions (start: 716) and is right on 627 of the 724 the teacher has right (start: 619). Plain cycles on this line ended 0 to 1.5 points under the average of their ends.
- **Three rounds and their average bring the line to 75 %.** Fixed-teacher cycles on the mined files, each from the end of the one before, letter asked for at the cap:

| Pair, 894 questions | As decoded | No answer | Letter asked for | Of the 673 | Of the 221 | Same letter as the teacher | Right of the 724 the teacher has right |
|---|---:|---:|---:|---:|---:|---:|---:|
| Average of the three plain fixed-teacher ends (the start) | 69.0 % (617) | 91 | 74.3 % (664) | 583 | 81 | 716 | 619 |
| Round one (`hccftm_sft_c5_run_u425`) | 68.7 % (614) | 101 | 74.6 % (667) | 590 | 77 | 725 | 627 |
| Round two (`hccftm_sft_c6_run_u426`): the questions still missed after a round | 67.2 % (601) | 115 | 73.8 % (660) | 581 | 79 | 718 | 618 |
| Round three (`hccftm_sft_c7_run_u425`): round one's questions, other fill | 69.6 % (622) | 101 | 75.7 % (677) | 592 | 85 | 726 | 629 |
| **Average of the three rounds' ends** (`hccftm_sft_avg_c5_to_c7_run_u425`) | 69.5 % (621) | 104 | **75.2 % (672)** | 591 | 81 | 729 | 630 |

- **The averaged pair equals the untrained reader with the whole article (75.3 %) at 1/20 of its tokens.** Against the KL average pair it gains 96 questions and loses 62 (z 2.7); against the average it started from 45 and 37 (z 0.9). Single ends bounce between 73.8 and 75.7 %, as on every line; the average is the pair to keep. The gain over its start is small and steady in every column: 8 more of the 673, 13 more questions with the teacher's letter, 11 more of what the teacher has right.
- **Twice the rows do not move it, with or without training.** The averaged pair decoded with rows at 1/10 (it was trained at 1/20), and after one more fixed-teacher cycle at 1/10 on the still-missed questions (`hccftm10_sft_c8_run_u426`):

| Averaged mined pair, 894 questions | As decoded | No answer | Letter asked for | Of the 673 | Of the 221 | Same letter as the teacher |
|---|---:|---:|---:|---:|---:|---:|
| Rows at 1/20, as trained | 69.5 % (621) | 104 | 75.2 % (672) | 591 | 81 | 729 |
| Rows at 1/10, not trained at that ratio | 69.0 % (617) | 110 | 76.3 % (682) | 595 | 87 | 727 |
| Rows at 1/10, after a cycle at 1/10 on the still-missed questions | 68.8 % (615) | 99 | 74.8 % (669) | 591 | 78 | 731 |

- The three rows are within noise of each other (52 gained and 42 lost between the first two; 43 and 56 between the last two), and the pair follows its teacher on 727 to 731 questions in all three. On the benchmark, the room in the rows is not what holds the pair at 75 %.
- **The replies that reach the cap are deliberation, not loops.** Of the 102 capped replies of the averaged pair, 2 repeat sentences; the others weigh the options and start over ("Wait, let's re-read carefully"). They are the hard questions: the letter asked for is right on half of them.

**What each stage adds on this line, and what the rule adds.** All pairs from reconstruction, no real question, rows at 1/20, cap 480:

| Stage | As decoded | No answer | Letter asked for | Of the 673 | Of the 221 |
|---|---:|---:|---:|---:|---:|
| Text-loss cycles, average of three ends (`hcc_sft_avg_c2_c3_c4_run_u587`) | 69.2 % (619) | 1 | 69.2 % (619) | 545 | 74 |
| One more plain text-loss cycle from it | 68.6 % (613) | 1 | 68.6 % | 535 | 78 |
| One fixed-teacher cycle from it (teacher: its own reader with the article) | 68.1 % (609) | 100 | 72.8 % (651) | 576 | 75 |
| Three fixed-teacher cycles, averaged | 69.0 % (617) | 91 | 74.3 % (664) | 583 | 81 |
| Three more on the questions where a pair leaves its teacher, averaged | 69.5 % (621) | 104 | **75.2 % (672)** | 591 | 81 |

- **The fixed-teacher stage is worth six points as decoded with the rule, and almost eight under a required answer call.** The text-loss pair with rows answers inside the cap (1 reply of 894 does not), so the rule gives it nothing: 69.2 %. Trained toward its reader with the article, the rows make the reader deliberate as it does with the article: about 100 replies reach the cap. As decoded the line stays at 68 to 69.5 %; with the letter asked for at the cap it goes from 69.2 to 75.2 % (110 questions gained, 57 lost, z 4.1). Asked at once, from 67.4 to 75.1 % (104 gained, 36 lost, z 5.7).
- **The pair has to be made to answer, but the rule about the cap is only one way.** Decoded by itself, the deliberating line stays below the pairs that answer at once (72.0 % and 71.4 %), and a cap of 1,024 tokens does not help (cycle 5: 70.4 % as decoded, 66 replies still open). A required answer call does it with no cap at all and the same score (see "Asked for the letter at once").

## Checkpoints

| Pair | Right on 894 | Real questions used | Where |
|---|---:|---|---|
| Real questions, text + KL + cosine | 73.6 % | yes | CSAIL `pairs/realm_u946` |
| The same, then KL only on own-article items | 73.6 % | yes | CSAIL `hc_synmatch_rk_run`; second node `pairs/hc_synmatch_rk_run_u631` |
| From reconstruction only: text-loss cycles, then three KL cycles toward the text-loss reader with the article (reader frozen), average of the three ends; letter asked for at the token cap | 74.3 % (69.0 % as decoded) | no | second node and CSAIL `pairs/hccft_sft_avg_c5_to_c7_run_u425` |
| **The pair above, then three fixed-teacher cycles on the questions where a pair leaves its teacher, average of the three ends** | **75.1 % with a required answer call**; 75.2 % with the letter asked for at the token cap; 69.5 % decoded by itself | no | second node and CSAIL `pairs/hccftm_sft_avg_c5_to_c7_run_u425` |
| Diagnostic: the pair above, then one fixed-teacher cycle on the real training-split questions (keys not trained on) | 75.2 % with a required answer call | yes | second node `pairs/hccftreal_kl_c8_run_u315` |
| The averaged mined pair, then one fixed-teacher cycle with the reader trained too | 71.7 % with a required answer call; 71.6 % decoded by itself | no | second node `pairs/hccftrd_kl_c8_run_u560` |
| **Rules-only set with explained text, KL only** | **72.0 %** | no | second node `pairs/hc_fin1g_run_u492`; CSAIL and first node `pairs/fin1g_u492` |
| From reconstruction only, KL only, four restart cycles, average of the last three ends | 71.4 % (batched decode) | no | first node `pairs/hcc_kl_avg_c2_c3_c4_run_u587` |
| Final set with explained text, KL only | 71.1 % | no | first node `pairs/fin1_u453`; second node `pairs/hc_fin1_run_u453`; CSAIL `pairs/fin1_u453` |
| KL + cosine, older items | 70.0 % | no | CSAIL `hc_synmatch_kc_run` |
| KL only, older items | 69.6 % | no | CSAIL and first node `pairs/synk2_final` |
| From reconstruction only, text loss only, four restart cycles, average of the last three ends | 69.2 % (batched decode) | no | second node `pairs/hcc_sft_avg_c2_c3_c4_run_u587` |
| From reconstruction only, writer text, KL only | 67.4 % | no | first node `pairs/hc_recw_kl_run_u2446` |
| From reconstruction only, writer text, text loss | 61.7 % | no | second node `pairs/hc_recw_sft_run_u2446` |

Paths on the Sky nodes are under `/root/activation_artifacts/long_recon_ec2/`; on CSAIL under `/data/scratch/ngom/activation/paper_v1/`. The two Sky nodes are spot nodes.

## What I would do next

Rewritten on Oct 11 and again on Oct 12 after the passes with a required answer call. The earlier list asked for a better teacher, keys where the teacher is wrong, the cause of the 80 % agreement, and items chosen by measured disagreement; all four were run.

- **Use a required answer call as the harness.** Asked for its letter at once, the best pair is right on 75.1 % and the reader with the article on 79.9 %; the reasoning text and the rule about the token cap are not needed, and an answer costs 12 tokens instead of up to 480. The other pairs do not change under it (the KL average pair: 71.5 against 71.4 %). It needs a decision, since the decode so far let the reader call by itself; both columns are in the tables.
- **Keep the three stages.** Text-loss cycles for the reader, fixed-teacher KL cycles for the rows (the reader with its adapter and the article as teacher, reader frozen), cycle ends averaged. Under the required call the stages read 67.4 % (text loss alone), 75.1 % (after the fixed-teacher stages); the KL-only line 71.5 %.
- **Keep the reader frozen in the fixed-teacher stage.** With the reader trained too, one cycle takes the pair from 75.1 to 71.7 % with rows; the reader fits our questions and reads the rows worse on the benchmark's.
- **Stop tuning questions for this pair.** The reader with the article stays at 79 to 80 % from the first text-loss training on, whatever is added. The rows stay 5 points under it through more cycles, mined questions, real questions of the training split, twice the rows and more averaging. The earlier advice to build the next question batch like the benchmark's is withdrawn: the real questions themselves did not help.
- **The 5 points are 86 questions** on which the trained readers agree with one another and the rows give another letter. Closing them needs a change in the channel, not in the questions: a larger encoder, or a reconstruction stage that ends lower. Both are new reconstruction runs, days and not hours; the encoder-size arms of the paper run are the place to read it.
- **Mined questions stay the default choice of questions,** because the cycle on them is the best single cycle on both lines; three rounds and their average added one point (74.0 to 75.1 % asked at once).
- **A late switch to more rows does not help.** The best pair scores the same at 1/10 as at 1/20 when asked at once, after a cycle at 1/10 too (75.3 %). Its reconstruction stage ran at 1/20 only, so this does not say what a pair built at 1/10 from the start would reach.
- **Real training-split questions with their keys** gave the earlier 73.6 % pair. Used without keys on the current pair they add nothing. Whether they are admissible matters less than it seemed.

## Not done, and cautions

- The scorecard script compares with real questions and so only runs on training-split articles; the released set's numbers are direct counts.
- The twin set of 2,500 on the 150 other articles was run with the `g1x` variant before the judge existed (2,374 questions), not with the final judged variant.
- Blind reading and the defect rates rest on samples of 60 to 80 items.
- "The explained text does the work" is not isolated: the 69.6 % arm (older items, the writer's reasoning notes as text) and the 71 to 72 % arms (new questions, the teacher's explanation as text) differ in their questions too. The explained answers for the older items are written (`qmain_s2.explained.jsonl`, beside the released set) but that arm was not run.
- The arm "the `g1x` pair, then the final set" never got a CSAIL GPU and was cancelled (job 2607469). The same test on the older items says what to expect: the last stage decides.
- Mixing other-article and own-article items in one run, as opposed to one stage after the other, has not been tried.
- The storage check on the first node stopped two arms at their start; snapshots of closed, scored arms were deleted there to make room (final pairs kept).
