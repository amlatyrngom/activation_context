# Real-like synthetic questions, and what each part of the match term is worth

**Results are in `REAL_LIKE_QUESTIONS_RESULTS.tressoir.md` (Oct 9).** This document is the plan as approved, with the probe rounds it produced. Two pieces of work for the next 16 to 24 hours, to be greenlit together. The main one makes the 27B writer's questions behave like the human-written QuALITY questions, then generates up to 2,500 of them for the 50 evaluated articles. The side one (in the card below) takes the match term apart. Approved by the user on Oct 8 at about 16:33Z, with one rule added: simplest method first (see the ladder below).

## Where we stand

All numbers are on the 894 benchmark questions of the 50 evaluated articles, AC rows at 1/20 of the article length. "Match term" is the extra pull toward the untrained reader that reads the whole article.

| Pair | Real questions used in training | Right |
|---|---|---:|
| Untrained reader given the whole article (the target) | – | 75.3 % |
| Question run, final (our synthetic items only) | no | 60.0 % |
| The same + 631 more updates with the match term | no | 65.9 % |
| Other variants of that (other articles' items, either order of the two) | no | 65.5 to 66.2 % |
| + 600 updates on 2,523 real questions of 150 other articles | yes | 70.2 % |
| + the same with the match term | yes | **73.6 %** |

Synthetic-only training has stalled near 66 %. The only thing that has moved it further is real questions, which were written by people about other articles. So the open question is what real questions have that ours lack, and whether the writer can be made to supply it.

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">Side work: the match term taken apart</span>
    <span class="card-oneliner">Four new arms and one control, about 1 h 45 each, first results about 2 hours after the go</span>
    <span class="card-badge">Ablation</span>
  </summary>

### What is compared

Every arm continues the question run's final pair for 631 updates on the same 5,048 synthetic items as the 65.9 % row. Only the loss differs. The teacher is always the untrained reader with the whole article, run live along the item's fixed answer text; "KL" is the exact divergence over the whole vocabulary at every answer position; "cosine" is one minus the cosine between student and teacher on the reader's final hidden vector at the same positions.

| Arm | Text loss | KL | Cosine | Status |
|---|:-:|:-:|:-:|---|
| KL only | – | yes | – | done: 69.6 % |
| KL + cosine | – | yes | yes | done: 70.0 % |
| Text + KL | yes | yes | – | done: 62.3 % |
| Text + KL + cosine | yes | yes | yes | done: 65.9 % |
| Text + cosine | yes | – | yes | done: 60.1 % |
| Text only (control I added) | yes | – | – | done: 60.7 % |

- **"KL only" is the repaired KL.** The package's older KL loss stores the top 32 probabilities of the reader as it is when the run starts. Here the arm uses the full vocabulary and the untrained reader, as agreed. The older loss stays in the package untouched; other code depends on it.
- **The control** is the same 631 updates with no match term. Without it the "+6 points" is only measured against the starting pair.
- **Two new switches** (text loss off, KL off) are written. With neither named, training is bit-identical to before on CPU tests, in one and in two processes; each arm's gradient equals the sum of exactly its parts. Not yet run on a GPU.
- **Started:** the text-only control on CSAIL at 16:34Z (`hc_synmatch_t_run`).

### Results (complete, Oct 9, 01:05Z)

All 894 questions, rows at 1/20; the starting pair scores 60.0 %.

| Arm | Right | Hard / easy | Note |
|---|---:|---:|---|
| Text only (control) | 60.7 % | 58.1 / 65.3 | the text loss alone adds nothing |
| Text + cosine | 60.1 % | 57.2 / 65.0 | cosine alone adds nothing; its KL, not trained there, stays near 0.8 |
| Text + KL | 62.3 % | 59.0 / 68.1 | 2.5 % of answers missing; a quarter are bare answer calls |
| Text + KL + cosine | 65.9 % | 62.9 / 71.2 | the earlier arm |
| **KL only** | **69.6 %** | 65.5 / 76.7 | every answer is the bare answer call |
| **KL + cosine** | **70.0 %** | 66.9 / 75.5 | 891 of 894 answers are the bare answer call |

- **The answer to the ablation.** KL carries the whole gain: the two arms without the text loss stand 9 points above text only. Cosine changes nothing, with or without KL. Adding the text loss to KL costs 4 to 7 points. Each number is ±1.6.
- **KL only copies the teacher.** It picks the teacher's letter on 692 of 894 questions (start pair 548), is right on 568 of the 673 the teacher gets right, and repeats the teacher's wrong letter on 124 of the other 221. Its ceiling is therefore the teacher's 75.3 %.
- **The answer key is not used by KL only.** A second right option or a wrong key cannot hurt it; an item helps only if the student and the teacher still disagree on it.
- **The teacher-with-adapter arm** scored 61.2 % and is closed. **Weight 2.0** scored 67.7 % against 65.8 % at 0.5 on the same items, inside noise.

### What is reported per arm

- Final pair on all 894 questions with the rows (noise about ±1.6 points).
- On the 200-question panel: rows, wrong rows, no rows.
- Whether answers still open with reasoning text. Without the text loss the reader may drift back to bare answer calls, because nothing but the teacher shapes the first positions.
- KL and cosine values over training, also for the part that is not trained.

### Where and when

| Place | Arms | Time |
|---|---|---|
| CSAIL, one free GPU | Text-only control, at the go | about 1 h 40 plus decode |
| First node, both GPUs, from about 17:25Z | Text + KL, KL only, Text + cosine, KL + cosine, in this order | about 1 h 45 each, about 7 h in all |
| Second node, spare GPU | The arms' decodes, so the first node never pauses | about 12 min each |

- **The weight-2.0 arm finishes first** (first node, ends about 17:25Z). The ablation stays at weight 0.5 whatever it shows, because the finished reference arm is at 0.5. If 2.0 is clearly better (3 points or more on the 894), one more arm reruns the best combination at 2.0.
- **The teacher-with-adapter arm** (second node, ends about 17:25Z) is a separate question: same loss as the 65.9 % row, other teacher.
- **Not included:** matching a middle layer. It would be a real extra constraint, needs new code, and waits for these results.

</details>

## What separates them today

Both proposers read real and synthetic questions on the same 76 training-split articles (1,278 real, 2,283 of our choice items). What a reader sees at once:

| | Real | Ours |
|---|---|---|
| Same story | "Why did the bank robbers end up crashing?" | "How did the physical state of the getaway car's tires change between the moment the robbers fled the bank and the moment the owner inspected the vehicle at the police garage?" |
| Same story | "How did Glmpauszn come to Earth?" | "In the letter dated June 17, Glmpauszn identifies the source of his confusion and impaired mission as what?" |

- **Our questions retell the scene and say where to look.** Real ones are short and plain, and many ask for a judgement about the whole story: "What theme could be taken from this story?", "Which word least describes Ivan?", "What would have likely happened if…".
- **Real wrong options are near-misses from the same story:** a swapped character, a neighbouring number, a statement that is almost true. Ours, where written without the article, are paraphrases of the right option or inventions foreign to the story.
- **Some of our "NOT" items are broken:** the wrong options written without the article are also absent from the story, so three options are right.
- **A simple classifier separates the two almost perfectly** (0.99, where 0.5 is a coin flip and two random halves of the real questions give 0.53). Swapping in the wrong options first written with the article leaves that at 0.99: the no-article options explain the word-overlap gap, not the detectability. Question length and wording do.

## What "near-real" will mean

A batch counts as near-real when it passes this scorecard on the measurement articles, where real questions exist for the same articles. The 894 benchmark questions are never used for tuning. The tolerances are the measurement proposer's judgement and mine, not derived from data.

| # | Measure | Real | Ours now | Near-real if |
|---|---|---|---|---|
| 1 | Untrained reader right, with / without the article | 80.5 % / 44.8 % (measured Oct 8 on the 2,523 real training questions; hard 72.8 / 39.9, easy 88.1 / 49.6) | 95 % / 46 % | 75 to 87 % / 39 to 49 % |
| 2 | Content words found in the article: wrong options / right option | 55.6 % / 56.8 % | 39.5 % / 59.3 % | wrong 50 to 60 %, gap to right at most 4 points |
| 3 | Right option longer / similar / shorter than the wrong ones (2 words) | 26 / 54 / 20 % | 13 / 68 / 18 % | each within 5 points |
| 4 | Classifier, real against synthetic, held-out articles | 0.53 (real against real) | 0.99 | at most 0.75, with no single feature carrying it |
| 5 | Question words: median; share with 8 or fewer; share above 20 | 10; 33.5 %; 3.7 % | 18; 0.4 %; 30.7 % | 9 to 12; at least 25 %; at most 8 % |
| 6 | Option words: mean; items averaging under 5 | 9.7; 22.1 % | 15.0; 5.1 % | 8.5 to 11; 17 to 27 % |
| 7 | Kinds: "why" opener; negation; best / most / least / main; would / likely | 20.7 %; 14.9 %; 13.4 %; 8.7 % | 7.7 %; 9.3 %; 0.8 %; 1.8 % | each within a third of the real value |
| 8 | Blind reading: a judge picks the human-written set out of two | – | not measured | about 6 in 10 or worse |

- **Rows 1 to 3 matter most.** Row 1 is the only measure of difficulty, not style. Rows 2 and 3 are the shortcuts a trained reader can exploit, and the earlier analysis tied the gain from real questions to them.
- **Row 1's real side is now measured** with the same check the synthetic items get. It is 5 points easier with the article than the benchmark figure (75 %) used in the first draft, so the band moved with it.
- **Row 4 is read together with its strongest features**, because cosmetic edits can lower it. We will not add typos or lower-case options to fool it, and we will not filter items with the classifier and then report the same classifier.
- **The scorecard is a proxy.** Rebuilt options once matched the word statistics and did not help training. The test that counts is the training test at the end.

## The articles and who sees what

| Set | Articles | Use |
|---|---:|---|
| Example pool | 74 training-split articles, 1,245 real questions | Real questions shown to the writer |
| Measurement set | 76 training-split articles, 1,278 real questions, 2,283 of our present choice items | Probes and scorecard; never shown as examples |
| Evaluated articles | 50, with their 894 benchmark questions | Final generation; their questions are never read by anything |

The writer never sees a real question about the article it is writing for.

## Ideas for the generator

**Ordering rule (the user's).** Among methods that work, the simplest to explain and the most general comes first, so the result cannot be read as tuned to the benchmark. Some fit to the benchmark is acceptable and will be named as co-design of harness, model and training data. In practice the variants form a ladder, and we take the lowest rung that passes; a higher rung is used only if it beats the one below by more than the noise.

| Rung | What the writer is given | How specific to QuALITY |
|---|---|---|
| 1 | Rules only: reading-test framing, short plain questions, wrong options from the story by named recipe | none: no real question is shown |
| 2 | Rung 1 plus ten real questions from ten other articles | examples of the format |
| 3 | Rung 1 plus two complete real sets, without their articles | examples of whole sets |
| 4 | An example article with its real set, mirrored question by question | the mix of kinds is copied from real sets |

Selection follows the same rule: first only validity (the 27B confirms the key without seeing it), then "the question must need the article" (drop what the reader answers without it), and only last matching the difficulty profile of real questions.

The designs below are described in the order the design proposer ranked them by expected effect; the ladder above is the order in which they are tried and adopted.

The list is ranked after the design proposer's report. That proposer read whole real sets beside ours and found gaps the word counts had missed (added to the scorecard above): our question stems are long, quote the article and point at "specific" lines, so they can be answered by searching for a phrase; real sets are short, plain and full of judgement questions ("which word least describes…", "what isn't likely to happen next", "why is it ironic…").

1. **Mirror a real set, shown with its article (top design).** One call shows an example article from the pool, its complete real question set with keys and hard marks, then the target article. The writer writes one question per example question: the same kind of question, as hard to answer by skimming, with options of the same shape, about the target. The mix of kinds, the "why" and "not" shares, the option lengths and the hard share then follow the real mix by construction. Each target article is paired with 9 different example articles.
2. **All wrong options written with the article, in the same call.** Today two of three are written without it, the largest measured gap. Each wrong option comes with a note on which reader would pick it ("skimmed the ending", "confuses two characters", "half of the reason").
3. **Short plain stems that do not quote the text**, and no equal-length rule. Enforced by code as well: stems holding a four-word run of the article, or phrases like "according to the passage", are dropped.
4. **Select by behaviour, not only by validity.** Write three times what is needed, then keep 2,500 so that the untrained reader's accuracy with and without the article matches real. Items the reader gets wrong with the article are kept only if the 27B, reasoning with the article and without the key, picks the key twice (two seeds, two option orders) and names no second defensible option.
5. **Kind slots without an example article (fallback).** The target article plus ten real questions from ten different articles, one imitated per slot. Half the reading per call, but the writer never sees how wrong options relate to a story.
6. **Real sets shown without their articles (cheapest).** Expected to fix form but not the options' relation to the story. Kept as a probe arm to learn whether the example article is needed.
7. **A separate wrong-option step**, rewriting the wrong options as a skimmer's picks. About 7,500 more short calls. Only if the first design's options stay guessable; an earlier probe of with-article options alone left them 65 to 70 % guessable without the article.
8. **Writer reasoning while writing.** Probe arm only, since it costs about ten times more.
9. **A revise pass by the writer on its own draft.** Rejected as the generator: it needs reasoning and a writer tends to defend its draft. A critique of flagged items only is the cheaper form.

Where the proposers differ: the measurement proposer would show 8 to 12 real questions from different articles per call (design 5) and give named recipes for the three wrong options: one true statement of the story that answers something else or names another character, one with a single detail changed, one natural misreading the story contradicts. The design proposer wants the whole set with its article (design 1). Both are in the first probe round, and the recipes go into both. "Except" items are built as three true statements and one false, never with options written without the article.

Two cautions from the design proposer that I take on:

- **Fifty questions per article is three times a human set** (real sets have 11 to 20, median 17), which pushes toward trivia. Real sets do return to the main points with other options, and nine example sets per article give that naturally. The plan keeps "up to 2,500" as a ceiling: if the kept items per article thin out below real quality, the release is smaller.
- **Choosing items the reader gets wrong also enriches wrong keys**, and the 27B checking its own key is not an independent check. The share of such items is capped at what survives the double check, even if that is 15 % and not 25 %, and I read about 50 of the kept hard items myself.

The 894 benchmark questions are not used as a filter either: removing candidates that resemble them would shape training data with test data. Collisions are only counted afterwards, for the record.

## The probe loop

One round takes about 30 to 40 minutes on one GPU of the second node.

1. **Step zero, once:** the untrained reader answers the 2,523 real training questions with and without the article. This gives the real target profile from a clean source, overall and per kind of question.
2. The writer produces about 170 candidates per variant on 10 measurement articles.
3. The untrained reader answers them with and without the article; the 27B answers a sample without the key.
4. The scorecard and the classifier are computed.
5. The proposers and I each read 30 items beside the real questions of the same articles and name what still gives them away.
6. The prompt or format is changed and the next round starts.

First round, in ladder order:

| Variant | Dropped if |
|---|---|
| Today's prompt | reference only |
| Rung 1: rules only | – (the floor every other rung must beat) |
| Rung 2: plus ten real questions | it does not beat rung 1 by more than noise |
| Rung 3: plus two whole real sets | it does not beat rung 2 |
| Rung 4: mirror a real set with its article | it does not beat rung 3; or the reader without the article stays above 55 %; or the 27B agrees with the key on under 80 % |

Held for later rounds, only if no rung passes: writer reasoning while writing, a separate wrong-option step, kind slots.

- **Stop rule:** the scorecard passes and the two proposers and I each fail to tell the sets apart better than about 6 in 10. Otherwise stop after eight rounds and take the best variant, saying plainly which rows it misses.
- **Copy guard, checked every round:** names that occur only in the example article must not appear; stem and options must not overlap the mirrored example beyond about 60 % of words once names are masked; no stem skeleton may dominate an article's set.

## Probe results so far

Rounds use all 76 measurement articles, one writer call each (about 1,290 questions per variant); ten articles proved too noisy. The reader is the untrained 9B. Real questions on the same articles are the first column.

| Variant | What the writer is given or asked | Reader right, with the article | Reader right, without | Right option longer / similar / shorter | Wrong / right options' words found in the article |
|---|---|---:|---:|---:|---:|
| Real questions | – | 80.4 % | 43.6 % | 26 / 54 / 20 % | 56 / 57 % |
| Rung 1 (`g1`) | rules only | 97.8 % | 65.6 % | 36 / 62 / 2 % | 59 / 74 % |
| Rung 2 (`g2`) | + ten real questions of other articles | 97.2 % | 64.7 % | 37 / 60 / 3 % | – |
| Rung 3 (`g3`) | + two whole real sets | 97.5 % | 65.7 % | 35 / 61 / 4 % | – |
| Rung 4 (`g4`) | mirror a real set shown with its article | 96.2 % | 71.9 % | 40 / 57 / 3 % | 53 / 58 % |
| `g1p` | rung 1, answer note and wrong options written before the right one | 98.5 % | 71.6 % | 40 / 46 / 14 % | 59 / 74 % |
| `g1h` | `g1p` + at most a third answerable from one place | 96.8 % | 75.2 % | 50 / 40 / 10 % | 58 / 68 % |
| `g1m` | `g1h` in two steps: twenty misreadings of the article first | 96.3 % | 78.1 % | 58 / 33 / 9 % | 52 / 67 % |
| `g1x` | question and key with the article; wrong options are an outsider's guesses made without it, restated in the story | 95.9 % | 44.8 % | 5 / 71 / 24 % | 44 / 72 % |

- **Rules alone fix the form of the question** (length, kinds). Showing real questions (rungs 2 and 3) adds nothing measurable, so by the ordering rule they are dropped. Mirroring (rung 4) fixes the mix of kinds only.
- **No wording moves the two reader numbers toward real.** Asking for harder questions makes the right option longer and easier to guess without the article. The 27B's key was sound throughout (58 or 59 of 60 on its own check, at most 2 of 60 with a second defensible option).
- **Outsider guesses reach the real guessing level** (44.8 against 43.6 %), but those wrong options are less in-story than real ones (44 against 56 % of their words in the article). Options written with the article are the mirror image: in-story and implausible. Real wrong options are both.
- **Round 3, running:** seven wrong candidates written with the article, each tied to a sentence it twists; the three the no-article reader finds most likely are kept (`g1s`). And a stricter grounding of the outsider's guesses (`g1x2`).
- **With the article the reader stays at 96 to 98 % on everything we write** (real: 80 %). About two thirds of the real questions the reader fails are worth reproducing by the measurement proposer's reading; nothing tried so far produces them.
- **Caution carried from before:** our older items already matched the real guessing level (46 %) and still trained worse than real questions. The training tests decide, not the scorecard.

**After the 27B wrong-option judge (round 5).** The judge asks, without the article, whether an option says the same as the key, and, with the article and the key hidden, whether each option is correct. Its defect rates match the measurement proposer's own reading of 40 items per variant, so it is not lenient.

| Variant | Stands as written | Kept after replacing options | Reader right with the article | Without |
|---|---:|---:|---:|---:|
| Seven in-story candidates, three kept by the no-article reader (`g1s`) | 70 % | 1,138 of 1,275 | 98.5 % | 64.9 % |
| The same pool plus the 9B reader's own answers (`g1rs`) | 57 % | 1,117 of 1,275 | 96.5 % | 57.9 % |
| Outsider's guesses, stricter grounding (`g1x2`) | 59 % | 753 of 1,281 | 95.1 % | 44.1 % |
| Outsider's guesses as a pool (`g1xc`) | – | 683 of 1,292 | 94.6 % | 44.9 % |

- **Every variant that reached the real reader numbers before the judge did so through defective items** (a wrong option that restates the key, or a second true option). Before the judge `g1x2` read 79.2 / 42.5 % and `g1rs` 80.1 / 41.4 %.
- **Clean items stay easy with the article** (94 to 98 %; real 80 %). Outsider guesses do give the real guessing level (44 %) on clean items.
- **Blind reading** by the measurement proposer told `g1s` from real in 60 of 60; the classifier stays at 0.91 to 0.95 for every variant.

**Training tests.**

| Arm | Questions | Loss | Right on 894 |
|---|---|---|---:|
| Reference | 2,523 real questions, 150 other articles | text + KL + cosine | 73.6 % |
| Training test 1 | 2,470 rung-1 questions, same articles | text + KL + cosine | 59.8 % (start 60.0 %) |
| Done | real questions, same articles | KL only | 71.1 % (hard 66.0 / easy 80.1) |
| Done | 2,374 `g1x` questions, same articles | KL only | 66.8 % (hard 63.7 / easy 72.1) |
| Done | rung-1 questions, same articles (control) | KL only | 60.5 % (start 60.0 %) |
| Done | final set for the 50 evaluated articles (first batch, 1,814 questions), the reader's explained answer as text, 453 updates | KL only | **71.1 %** (hard 67.1 / easy 78.2) |
| Done | 5,386 older bare-call items, the 50 evaluated articles | KL only | 64.8 % (text only on the same items: 60.0 %) |

Under KL only, real questions on other articles (71.1 %) and our older synthetic items on the evaluated articles (69.6 %; 70.0 % with cosine) are 1.5 points apart, inside noise. Under the text loss they were 13.6 points apart. Both KL-only pairs are right on the same number of the 673 questions the teacher gets right (567 and 568); the real-question pair is right on 69 of the 221 the teacher gets wrong, the synthetic one on 54.

The `g1x` questions give about 60 % of the real-question gain under KL only (+6.8 against +11.1 over the 60.0 % start). Questions on other articles, real or ours, help where the teacher is wrong (69 and 67 of 221 right, against 54 for own-article items), although KL only never uses a key; own-article items help where the teacher is right (568 of 673, against 530 for `g1x`). A combined arm tests whether the two add up.

Rung-1 questions gave nothing: the reader already answers them as the teacher does (text loss 0.002 during training), so they carry no signal. The two KL-only arms on the 150 other articles are the like-for-like test of real against synthetic questions under the loss that works.

## Final generation

- About 7,500 candidates for the 50 evaluated articles with the winning variant: 9 calls per article, 450 calls, under one GPU-hour by the proposer's estimate. All 150 training-split articles can serve as examples here, since measuring is over.
- Filters in order of cost: form and copy checks by code; reader check with and without the article; the 27B with reasoning answers without the key and names any second defensible option (about 4 GPU-hours for about 5,000 survivors); a second such pass for items the reader got wrong.
- Selection of up to 50 per article: difficulty matched to the real profile from step zero, at most 4 items per evidence passage, at least a fifth needing the whole text, kinds near the real mix, no near-duplicates.
- Release gates on the kept set: the scorecard rows that need no model.
- A twin of 2,500 on the 150 training-split articles, same pipeline. It allows a like-for-like test against the real questions.

## Training tests

Each is 600 to 950 updates from the question run's final pair, with the match term at 0.5, decoded on all 894.

| Arm | Question | Compared with |
|---|---|---|
| Twin: synthetic questions on the 150 other articles | Do they do what the real ones do? | 73.6 % (real, same articles) |
| Main: 2,500 on the 50 evaluated articles | The paper arm | 65.9 % (our present items) |
| Both, one after the other | Do they add up? | the two above |

The work has succeeded if the twin lands within about 3 points of 73.6 %. That would mean the gain came from the kind of question, not from human authorship.

## Time and GPUs

| Hours after the go | Second node | First node | CSAIL |
|---|---|---|---|
| 0 to 1 | Teacher-with-adapter arm ends; real-question reader measurement | Weight-2.0 arm ends | Text-only control |
| 1 to 7 | Probe rounds (writer on one GPU); ablation decodes on the other | Ablation arms | Decodes |
| 7 to 10 | Final generation and review, both GPUs | Review shards | – |
| 10 to 15 | Training test (main) | Training test (twin) | Decodes |
| 15 to 16 | 894-question decodes, write-up | 894-question decodes | – |

## Decided

- **Go for both pieces of work** (user, Oct 8, about 16:33Z), with the ordering rule above.
- The twin set on the 150 other articles stays in, as no objection was raised.
