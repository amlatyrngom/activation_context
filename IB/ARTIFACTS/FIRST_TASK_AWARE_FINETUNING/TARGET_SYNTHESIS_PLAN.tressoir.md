# Generating the right training targets: quote-only and better question/answer

Two next steps, both about what the 27B writer produces and what the reader is then trained to output. Part 1 is a quote-only run of about 3,000 updates on quotes that cover the articles evenly. Part 2 is a question/answer synthesis that covers the articles, keeps hard questions and labels what each question needs. Six decisions are settled (table near the end); the sizes are set (both runs 3,000 updates plus a 500-update cooldown) and the GPU plan (D9). Phase 1, the probes, started at 19:25Z. Your green light covers everything up to and including training and decoding. Old questions are not carried over. A probe phase comes before each synthesis, for the quote requests as well as for the questions, and each may end with several prompts used side by side.

## Why these two

Today's runs on the 50 evaluated QuALITY articles (written 18:10Z, Oct 7):

| Run | Right rows | Wrong rows, same questions | No rows |
|---|---:|---:|---:|
| Starting pair (reconstruction only), 200 questions | 41.5 % | 35.9 % (64) | 40.5 % |
| Question run on our synthetic items, final, 200 questions | 54.5 % | 43.8 % (64 paired: 56.2 % right) | 37.5 % |
| Diagnostic trained on real questions of other articles, 200 questions | 59.0 % | 43.8 % (64 paired: 60.9 % right) | 39.5 % |
| Untrained reader given the article text | 72.0 % | – | 40.0 % |

- The same rows answer 95.8 % of our own held synthetic choice questions (48 items, no wrong-rows control yet) but 54.5 % of the benchmark's. The gap to the real-question run sits on hard questions (57.1 % against 42.1 % at update 500).
- The quote-only run (573 updates) cut held-quote loss from .511 to .176 per token, but reconstruction got four times worse (.041 → .167) and nothing else moved.
- Our 9,800 QuALITY-50 quotes cover only 40 % of the article text; 20 % is quoted twice or more. The writer returns to the same quotable sentences.

So the targets are the suspect: too local, too redundant, and filtered by what the frozen reader already gets right.

## The shared idea: we assign the place

Both parts use the same covering rule. The writer no longer chooses where in the article an item comes from.

```text
article ──cut at paragraph ends──▶ units (cut points shift each round)
                                     │
        every unit gets exactly one item per round  =  one covering pass
                                     │
        round 1, round 2, round 3 …  =  each place covered 1×, 2×, 3× …
```

The assignment follows from the round number alone, so parallel jobs need no shared bookkeeping. Every release reports a coverage line: the share of article text under at least one, two and three items.

| Corpus | Documents | Quote units (64 to 256 tokens, mean 160) | Question units (about 130 tokens) |
|---|---:|---:|---:|
| QuALITY-50 | 50 | about 2,000 per pass | 2,509 per pass |
| LongHealth | 20 | not cut yet | 1,236 per pass |

## Part 1 — Quote-only

### Step A: writing the items (27B writer)

```text
INPUT to the writer, one call per group of about 8 units of one article
  ┌──────────────────────────────────────────────────────────────┐
  │ the whole article (same text every call, so it is cached)    │
  │ the assigned spans, each 64 to 256 tokens, numbered          │
  │ later rounds: the requests already written for each span     │
  └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
OUTPUT of the writer, as JSON
  one REQUEST per span: a short request for that passage
  without using its words
  e.g. "Quote what the crates in the car were labelled."
                              │
                              ▼
CHECKS, no model needed
  · the quote is the span itself → verbatim and covering by construction
  · the span occurs once in the article
  · request and quote share few words (it must not give the answer away)
  · the request does not mention "the marked passage" or the prompt
  · not a near-duplicate of an earlier request for the same span
```

**Multi-quote items.** One request answered by two or three quotes from separate places. Random spans rarely share a topic, so here the writer gets a set of about a dozen least-covered units and picks two or three that belong together, then writes the request. Each quote is checked to be verbatim, unique and inside the assigned set.

### Step B: the training example

```text
article ──▶ encoder (Qwen3.5-2B) ──▶ AC rows, 1 per 20 tokens ─┐
                                                               ▼
reader input (Qwen3.5-9B + adapter):  system prompt
                                      "Task: <request> … call submit_answer"
                                      AC rows in place of the article text
                                                               │
                                                               ▼
reader target:   <tool_call> submit_answer( <the quote, verbatim> ) </tool_call>
                 multi-quote: one quote per line inside the same single argument
                 no reasoning text before the call
loss:            80 % on the quote tokens, 20 % on the 26 fixed wrapper tokens
```

### Step C: the run

| Item | Plan |
|---|---|
| Span | we assign it; the writer writes the request (decided) |
| Multi-quote | about a third of the quote examples (decided) |
| Reconstruction | about 10 % of the examples (decided) |
| Corpus | QuALITY-50 first (decided) |
| Size | 3,500 updates = 28,000 examples; breakdown in the sizes decision at the end |
| Starting pair | the tuning pair at about 3× exposure (same as today's quote run), unless you prefer the question-run pair |
| Held-out | 2 % of each article as stretches of about 250 tokens; a unit touching one never trains, in any pass |
| Panel | held quotes, held prose, reconstruction, benchmark questions with right / no / wrong rows |
| Decode | exact match and word overlap of held quotes with right rows, **wrong rows** and no rows; the untrained reader with the article as ceiling |

Reference before any quote training: the starting pair reproduces 1 of 48 held quotes exactly (word overlap .38).

### Probe phase for the quote requests

One prompt gives one kind of request. Measured on the first covering pass (9,388 requests) and the part of the second that is written (6,221):

| What | Today's single prompt |
|---|---|
| Request length | 23.7 words, for a quote of 23.7 words: the request is a full retelling of the passage |
| Opening | "Quote the sentence" 32 %, "Quote the line" 10 %, "Quote the description" 6 % |
| Quote's content words that appear in the request | 21 % |
| Two passes over the same place | the second request is the first one reworded, as below |

```text
pass 1: Quote Dan's internal reasoning for preferring darkness, specifically regarding the tactical advantage
        his eyes would have against intruders.
pass 2: Quote the sentence explaining Dan's tactical reason for preferring the dark and his subsequent movement
        to his sleeping area.
```

So three passes would show the reader each passage three times behind nearly the same request. The fix is the same as for questions: several request styles, tuned by measuring, and every pass gives a passage a different style.

```text
about 320 units from 10 of the 50 articles, in three size groups (32-64, 64-128, 128-256 tokens), the same units for every style
        │
        ▼
each style writes one request per passage  ──▶  quality gate (below)  ──▶  one table: styles as rows
        │
        ▼
keep the styles that still lead back to the passage and differ from each other; rewrite the rest; repeat
        │
        ▼
full passes: style of a passage = (passage number + pass number) cycled over the kept styles
```

| Style to try in cycle 1 | Example for the passage above |
|---|---|
| Full description (today) | the reference |
| Short pointer, at most 10 words | Quote Dan's reason for preferring the dark. |
| A question, answered in the text's words | Why does Dan want the lights off? Answer with the exact words of the text. |
| By its role in the story | Quote where Dan first prepares for intruders. |
| By what happens around it | Quote what Dan thinks just before he goes to his sleeping area. |
| By one detail only | Quote the lines that mention Dan's eyes. |
| By what it shows, not what it says | Quote the lines that show Dan expects to be attacked. |
| By speaker and listener (dialogue only) | Quote Dan's answer to the guard. |

| Measure per style | Wanted |
|---|---|
| Leads back: the 27B, given article and request, quotes the assigned passage | high; this is what vaguer styles pay with |
| Request length against quote length | well under today's 1.0 |
| Quote's content words in the request | under today's 21 % |
| Most common opening | no opening above about 10 % |
| Word overlap between two styles' requests for the same passage | low |
| Untrained reader with the article gives the exact quote | the ceiling for the run |

Multi-quote requests get the same treatment with two or three styles (description, question, short). One cycle is about 2,600 requests plus the gate: about 15 minutes on one GPU.

Every measure is reported per size group. The unit size stays negotiable on that evidence: if the 27B writes too many overlapping requests at the chosen size (requests for one unit that repeat each other, or that fit a neighbouring unit as well), the lower end moves down, towards 32 to 64 tokens (your note, 19:30Z).

### Quality gate for quotes (proposed)

A request is only good if it leads back to its passage. One more pass makes that a check instead of a hope:

```text
27B gets: the article + the request (NOT the span)  ──▶  it quotes what it thinks is asked for
keep the item when its quote is the assigned passage (word overlap of at least .8)
```

It costs one short call per quote, with the article cached. It also removes requests that fit a neighbouring passage equally well.

### Status (19:30Z)

- **Built:** the covering assignment at sentence size, the request-only writer prompt, held-out sentences, the coverage line. Three sentence-size passes of QuALITY-50 are written (single request style). They are the reference row of the probe, not training data.
- **Written on the container, never run:** the probe script with eight request styles, the lead-back check and its table; multi-quote grouping (12 candidate units from three far-apart parts of the article, two or three grouped per request).
- **Not built:** the cut into 64-to-256-token units at paragraph ends, held-out stretches, the three size groups in the probe.
- **Earlier quotes** (9,640 for QuALITY-50, written the old way, 40 % coverage) are not used.

## Part 2 — Better question/answer

Old questions are not carried over: about 40,000 QuALITY-50 items written the old way stay on disk as a reference, and the new release is written from scratch.

### Probe phase first

Asking the writer for whole-article questions is what we already did, and its own review labelled 1 % of them that way. So the prompts have to be found by measuring. The probe may well end with several prompts used side by side, one per kind of question.

```text
12 training-split articles (never the 50 evaluated ones; real questions exist for them as the yardstick)
        │
        ▼
each prompt variant writes about 200 items   ──▶  reader check  ──▶  review (answer + needs label)
        │
        ▼
one table per cycle: variants as rows, the measures below as columns, real questions as the reference row
        │
        ▼
keep, per kind of question, the variant with the best yield; drop or rewrite the rest; repeat (2 to 3 cycles)
```

| Variant to try in cycle 1 | What it changes |
|---|---|
| Today's prompt | the reference |
| Assigned unit, local | one 90-word unit per item, evidence must lie inside it |
| Assigned pair, multi-passage | two far-apart units; the question must need both |
| Whole-article by question type | motives, how a character changes, theme, tone, what the title refers to, overall course of events |
| Whole-article, writer reasons first | the same with medium reasoning for the writer |
| Notes first | the writer first writes notes on the whole article (people, arc, themes), then asks from the notes |
| Skim-proof self-test | after writing, the writer names the single passage that would answer it; if one exists it rewrites |
| Wrong-option variants | how the wrong options are made, to keep items hard to guess without the article |

| Measure per variant | Target |
|---|---|
| Label profile from the review | real questions: 60 % local, 23 % multi-passage, 16 % whole-article |
| 27B gives the key again | at least the 87.5 % it reaches on real questions |
| Guessable without the article (frozen reader) | no higher than real questions (44 % in the earlier probe) |
| Near-duplicates within an article | low, and how fast it rises with more passes |
| Evidence inside the assigned place | high for local and multi-passage |
| Distinct kept items per article | this sets how many items one article can give |
| Cost per kept item | GPU-seconds, review included |

One cycle is about 1,600 items written and reviewed: under an hour on four GPUs.

### The flow

```text
 ① WRITE (27B writer)                      IN:  whole article
                                                assigned paragraph unit(s)
                                                asked kind: local / multi-passage / whole-article
                                                questions already asked about this unit
                                           OUT: question · correct answer · wrong options (choice)
                                                evidence sentence(s), verbatim · short reasoning
        │  checks, no model: evidence is verbatim AND inside the assigned unit(s);
        │                    format; no near-duplicate
        ▼
 ② READER CHECK (frozen 9B)                IN:  question with the article, and without it
                                           OUT: its answers → "guessable" flag, reader's own prose answer
        │  no longer decides what is kept
        ▼
 ③ REVIEW (27B, medium reasoning)          IN:  article · question · options — NOT the key
                                           OUT: its own answer letter
                                                needs: common-sense | local | multi-passage | whole-article
        │  keep a choice item when the 27B's answer is the key
        ▼
 ④ RELEASE                                 whole covering passes only
                                           mix by "needs" label, aimed at the profile of real questions
                                           3 % held out · coverage line · counts per label
        ▼
 ⑤ TRAINING EXAMPLE                        reader input:  question (+ options) + AC rows
                                           reader target: short reasoning text, then the answer call
                                           plus 20 % reconstruction examples of the same articles
```

### What changes against today's synthesis

| Today | Proposed |
|---|---|
| Writer picks the place inside a random window | Place is assigned; evidence must lie inside it |
| A choice item is kept only if the frozen 9B answers it correctly with the article | Kept if the 27B, asked again without the key, gives the key |
| No notion of difficulty | Every item carries a `needs` label from the review |
| Mix fixed by style only (prose 50 %, choice 25 %, quote 25 %) | Same style mix, plus a mix over `needs` labels |
| Later rounds rephrase earlier questions | Writer is shown what was already asked about the unit |

Whole-article questions have no single place. They are asked per article, not per unit, and are governed by the `needs` label and the near-duplicate check.

### The `needs` label and its calibration

The review is one call per item. The 27B must choose the least that is enough: `common-sense` (no document needed), `local` (one place), `multi-passage` (two or more separate places), `whole-article` (the text as a whole).

The same call ran on real training-split QuALITY questions, which carry the annotators' hard / easy label. First result (18:36Z; 598 real questions, 600 of ours):

| What the question needs | Real questions | Ours, QuALITY-50 | Ours, QuALITY-150 | Hard share among real |
|---|---:|---:|---:|---:|
| common-sense | 0.7 % | 0.3 % | 0.7 % | (4 questions) |
| local | 59.7 % | 83.5 % | 82.0 % | 45.1 % |
| multi-passage | 23.4 % | 15.2 % | 16.2 % | 55.7 % |
| whole-article | 16.2 % | 0.9 % | 1.1 % | 52.6 % |

- **The profile differs clearly.** Real questions are 40 % non-local; ours are 16 %. Whole-article questions are one in six among real questions and about one in a hundred among ours, although a quarter of our choice items were asked to need the whole document.
- **The label is a weak difficulty signal.** Hard questions are 45 % of local and 53–56 % of the other two labels. That is a 7 to 10 point difference, well short of the 20 points I had set as the bar. Treat the label as a description of what a question draws on, not as a hard / easy predictor.
- The 27B gives the key on 87.5 % of real questions and on 95 % of ours.

The real profile (60 / 23 / 16) is the target mix for the next synthesis. Real questions are used only for this; the writer never sees them (decided).

### Training target, two variants

| Variant | Target of a training item | State |
|---|---|---|
| Writer's answer | the 27B's reasoning and answer | used in today's question runs |
| Frozen reader's answer | what the untrained 9B itself writes when given the article | training on CSAIL since 18:09Z (1,284 updates). The frozen reader writes no reasoning with this prompt (57 of 8,222 items have any), so these targets are bare answer calls |

A known weakness in both: reasoning and answer share 80 % of the loss by token count, so a choice item's answer letter carries about 1.4 % of its example's loss. Decided: the answer gets its own fixed share from the next packet on; the split is part of the sizes decision.

### Status

- **Built and working on the GPU:** the review call (own answer + `needs` label), the keep rule based on it, release counts per label, the table script. It reviews about 0.5 items per second per GPU.
- **Not built:** the probe variants, assigned units for questions, "already asked" lists, mixing by label.
- **Existing items** (about 40,000 for QuALITY-50) are not reused.

## Order of work

1. Probes for the quote requests and for the questions, side by side; the prompt document grows with each cycle.
2. Bulk writing: quote passes with the lead-back check; question stage 1 (about 10,000 items).
3. The quote run and the question run start on the two EC2 nodes; question stage 2 is written meanwhile.
4. The stage-border checkpoint of the question run is the like-for-like comparison with today's 54.5 % (today's release: 8,222 items).

The GPU plan and the times are in "GPUs: who does what, in which order" below.

## Decided (your answers, 18:27Z)

| # | Question | Answer |
|---|---|---|
| D1 | Who chooses the quote span? | We assign it; the writer writes the request |
| D2 | Multi-quote share of the quote run | About a third |
| D3 | Reconstruction examples in the quote run | Yes, about 10 % |
| D4 | Corpora | QuALITY-50 first |
| D5 | Real training-split questions | Measure and set the label mix only; never shown to the writer |
| D6 | Answer letter of a choice item | Gets its own fixed share of the loss |
| D8 | Size of a covering quote unit | 64 to 256 tokens, no size test (19:20Z); the lower end may go down towards 32 to 64 if the probe shows overlapping requests (19:30Z) |
| D7 | Sizes | As in the sizes table: both runs 3,000 + 500 updates; 22,400 fresh question items in two stages; lead-back check for quotes |
| D9 | GPUs and order | As in the GPU section; phase 1 started 19:25Z; the frozen-answer run keeps running |

## Covering unit: 64 to 256 tokens (decided 19:20Z)

A covering quote unit is 64 to 256 tokens, no size test first. The lower end is negotiable down towards 32 to 64 tokens if the probe shows the 27B writing overlapping requests; the probe reports its measures per size group for that.

```text
article ──▶ paragraphs (blank lines; 143 per article, 45 tokens on average)
        ──▶ merge neighbours until a drawn minimum between 64 and 200 tokens is reached, never above 256
            (a paragraph longer than 256 tokens is cut at sentence ends)
        ──▶ units; every paragraph is in exactly one unit; the draw changes with the pass, so borders move
```

| Measured on QuALITY-50 | Value |
|---|---|
| Units per pass | about 2,000 (40 per article) |
| Unit length | 160 tokens on average; a tenth below 100, a tenth above 226 |
| Single-quote examples in the run | 16,800 = about 8.4 passes, each with a different request style and other borders |
| Multi-quote examples | 8,400; two or three pieces from the short end (64 to 128 tokens each), so a target stays under about 380 tokens |

Why larger than sentences, from today's quote run: of 34 missed held quotes, 23 were the right content written approximately and 9 were real text from another place. The reader mostly finds the place and then fails to copy; longer targets train the copying.

What follows from it:

- A request is short against its quote and names the borders ("the exchange where …, from … until …").
- Held-out text is held-out stretches of about 250 tokens, 2 % of each article, instead of single sentences.
- The measure is word overlap and the longest exactly copied run; exact match is reported but will be rare.
- New control in the decode: passages the run trained on, asked with wrong AC rows. If the reader still quotes them, it learned them by heart and not from the rows.

## GPUs: who does what, in which order

State at 19:15Z:

| Where | GPUs | Doing now | Free |
|---|---|---|---|
| EC2 node 1 (the QuALITY node, `qa-r10`) | 2 RTX PRO 6000 | idle | now |
| EC2 node 2 (the LongHealth node, `qa-lh`) | 2 RTX PRO 6000 | final decode of the LongHealth question run on one GPU | about 19:30Z |
| CSAIL | 2 H200 | frozen-answer question run, update 640 of 1,284 | about 20:15Z, or now if stopped |
| ORCD | up to 4 RTX PRO 6000, can be preempted | full decode of today's final QuALITY pair on one; the sentence-size quote passes have finished | 3 now, the 4th within the hour |

Ten GPUs in all. The two training threads go to the EC2 nodes, two GPUs each; until their packets exist, the EC2 nodes write data like the others.

```text
PHASE 1  probes                      about 1 h 15 after the go
  ORCD 1 GPU ............ quote probe: units in three size groups, 8 request styles, lead-back check (15 min per cycle, 2 cycles)
  ORCD 2-3 + EC2 1 + EC2 2  question probe: 8 prompt variants over 6 GPUs, write → reader check → review
                          (25 min per cycle, 2 to 3 cycles)
  CSAIL ................. frozen-answer run finishes (20:15Z)
  result ................ the prompt document for you (below), updated after every cycle

PHASE 2  bulk writing on all 10       about 1 h 15 to the two first packets
  CSAIL 2 + EC2 node 1 .. quote passes in mixed styles, multi-quote, lead-back check   (about 4 GPU-hours)  → quote packet
  ORCD 4 + EC2 node 2 ... question stage 1: write, reader check, review                (about 7 GPU-hours)  → stage-1 packet

PHASE 3  the two training threads
  EC2 node 1 ............ quote run, 3,000 + 500 updates                               (about 6 hours)
  EC2 node 2 ............ question run, starts on stage 1, goes on with stage 2        (about 5.5 hours)
  ORCD 4 + CSAIL 2 ...... question stage 2 (about 10 GPU-hours, 1 h 45), then the checkpoint decodes of both runs,
                          then LongHealth writing
```

| If the go comes at 19:45Z | Time |
|---|---|
| Probes done, prompt document complete | about 21:00Z |
| Quote packet and question stage-1 packet | about 22:15Z |
| Both runs start | about 22:30Z |
| Question stage 2 ready | about 00:15Z, before the question run reaches the stage border (about 00:50Z) |
| Stage-border checkpoint of the question run decoded (the 10,000-item point) | about 01:30Z |
| Both runs end | about 04:00 to 04:30Z |

### What changed by 20:30Z

The plan above assumed ten GPUs. Tonight there are six:

| Where | GPUs | State at 20:30Z |
|---|---|---|
| EC2 `qa-lh` | 2 RTX PRO 6000 | GPU 1 writes quote rounds 600 to 605 (one round = 14 minutes, 88 % of requests kept). GPU 0 finishes the question bulk test, then writes question shards 0 and 1. |
| EC2 `qa-q2` (new) | 2 RTX PRO 6000 | Replaces `qa-r10`, which its idle timer shut down at 19:16Z. Created 20:18Z, models downloading; then quote rounds 606 to 609 and the multi-quote rounds. |
| CSAIL | 2 H200 | Question shards 2 to 5 running, shards 6 to 9 start when those end. |
| ORCD | 0 | Our jobs wait behind about 1,500 others with far higher priority. Kept in the queue as a bonus, nothing is planned on them. |

- Both EC2 nodes now carry a sleeping Sky job, so the idle timer cannot shut them down during work started by plain ssh.
- The probes ended at 20:10Z; the prompt set in use is in `PROBE_PROMPTS.tressoir.md`.
- The quote run trains on `qa-q2`, the question run on `qa-lh`. Both are expected to start between 22:15Z and 22:45Z and to end between 04:00Z and 04:30Z.
- Checkpoints are copied off the spot nodes as they appear and decoded on CSAIL, not on ORCD.

### Correction at 21:20Z: ORCD did run

Three ORCD jobs started at about 20:28Z (three RTX PRO 6000 on one node); I noticed at 21:08Z. So tonight there are nine GPUs, not six.

| Where | What it wrote or runs | Consequence |
|---|---|---|
| ORCD, two quote lanes | Single-quote rounds 606 to 609, complete at 21:14Z; then 610 and 611 as a reserve | The ten single-quote rounds are covered: 600 to 605 on `qa-lh`, 606 to 609 on ORCD. |
| ORCD, one question job | Question shard 6, rounds 800 and 801 | CSAIL no longer writes shard 6 for stage 1; duplicate jobs were cancelled on both clusters. |
| ORCD, waiting | Three lanes for multi-quote rounds 731 to 745 with the current script; the untrained reader with the article on the 894 benchmark questions | The multi-quote lanes help only if they start before about 22:15Z. After the quote release is cut, ORCD takes hill-climb decodes and arms. |

- Start times now expected: question run about 22:00Z on `qa-lh`, quote run between 22:25Z and 22:45Z on `qa-q2`.
- Checkpoint decodes go to CSAIL first; ORCD is the second place when its jobs start.
- Question stage 2 is written by CSAIL and by whichever EC2 GPU is idle before its run starts.

Points behind the plan:

- **The frozen-answer run** has one hour left and CSAIL is not needed before phase 2, so I would let it finish. Stopping it now would free two GPUs for the probes, which do not need them.
- **The question run starts on stage 1** instead of waiting for all 22,400 items. This saves about 1 h 45, and its checkpoint at the stage border is the 10,000-item point you asked for. Every item is seen once. To check before the go: that the trainer can take the second stage into the same run without restarting the optimizer; if not, the run continues as a second run from the stage-1 pair.
- **Decodes never run on a training node** (a decode beside training slows it by about 60 %); checkpoints are copied to ORCD or CSAIL and decoded there.
- **ORCD can be preempted.** Its work is written in whole-article batches with claims, so a preempted job loses a few minutes.

### The prompt document

During the probes I keep `PROBE_PROMPTS.tressoir.md` in this folder, one section per probe:

- one table per cycle (styles or variants as rows, the measures as columns, real questions as the reference row);
- one card per prompt, closed by default: the full prompt text as sent to the 27B, what changed against the previous cycle, and five written items with the review's verdict;
- a last section with the prompt set I propose for the bulk writing and the share of each prompt.

**Decided (19:20Z):** allocation and order as shown; go for phase 1 given in the conversation ("Go"); the frozen-answer run keeps running ("keep that going"). Phase 1 started 19:25Z.

**Green light (19:30Z):** it covers everything up to and including training and decoding. So after the probes the bulk writing starts without waiting, with the prompt set chosen from the probe numbers and shown in the prompt document; then the two runs and their decodes. Following the plan as shown, the question run starts on stage 1 and continues on stage 2. You can still redirect at any point; the prompt document is there for that.

## Sizes (decided)

Your numbers (19:10Z): both runs are 3,000 updates plus a 500-update cooldown; old questions are not carried over; releases are staggered and the 10,000-item point is a checkpoint of its own.

| | Recommended | Note |
|---|---|---|
| **Quote run, length** | 3,000 updates + 500 cooldown = 28,000 examples | about 6 hours on two GPUs with the longer targets |
| Quote examples | 16,800 single (about 8.4 passes of 64-to-256-token units) and 8,400 multi-quote | a third multi-quote |
| Reconstruction | 2,800 examples | 10 % |
| Quality gate | keep a quote only if the 27B, given the request, finds its passage | passes are written with a margin |
| Quote probe | about 320 units of 64 to 256 tokens × about 8 request styles, 1 to 2 cycles | before the training passes are written |
| Held-out text | 2 % of each article | as stretches, not single sentences, once units are larger |
| Panel and checkpoint | every 250 updates | 14 panels |
| **Question run, length** | 3,000 updates + 500 cooldown = 28,000 examples | about 5 hours on two GPUs |
| New question items | 22,400, all written fresh | 448 per article |
| Reconstruction | 5,600 examples (20 %, as in today's question runs) | say if you want 10 % as for quotes |
| Style mix | prose 11,200 · choice 5,600 · quote 5,600 | your earlier 50 / 25 / 25; the quotes come from the covering passes |
| Label mix of the 16,800 written questions | local 10,080 · multi-passage 3,860 · whole-article 2,690 | the real profile 60 / 23 / 16 |
| **Staggered releases** | stage 1 = about 10,000 items, a whole release: complete covering passes, the full style and label mix, its own held-out part. Stage 2 = all 22,400, containing stage 1 | stage 1 is close to today's release (8,222 items, 1,288 updates), so a run on it shows what better questions give at equal size, before more data is added |
| Loss split of a choice or prose item | answer 30 %, reasoning 50 %, wrapper 20 % | D6 |
| **Question probe** | 12 training-split articles, about 8 variants of about 200 items, 2 to 3 cycles | before any full synthesis |

**The open risk in the question numbers.** 22,400 items is 448 per article, and 54 of them whole-article questions per story. I doubt one article gives that many distinct whole-article questions. The probe measures the distinct yield per article. If it is lower, I recommend keeping the 3,500 updates and passing over the items more than once, rather than padding with near-duplicates.

**Decided (19:20Z):** the table as shown. Whether to repeat items or shorten the run if an article gives too few distinct questions stays open until the probe has measured the yield.
