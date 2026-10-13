# TARGET_SYNTHESIS_PLAN (agent-facing source; human projection: TARGET_SYNTHESIS_PLAN.tressoir.md)

Written 18:10Z Oct 7 at the user's request ("summary of the next two things … generating the right training targets";
"1. Quote-only: inputs/outputs of the model, the flow. 2. Better Q/A: same"). Answers land in
`TARGET_SYNTHESIS_PLAN.interactions.json` (keys `tsp.*`); integrate them here and in the projection, then remove resolved controls.

## Evidence behind it (QuALITY-50, 200 questions unless noted)
- start pair 41.5 / 35.9 (64) / 40.5 (right / wrong / no rows); synthetic-trained question run final 54.5 / 43.8 / 37.5
  (paired 64: 56.2 vs 43.8); real-question diagnostic u450 59.0 / 43.8 / 39.5; untrained reader with article 72.0.
- held synthetic choice 95.8 % (no wrong-rows control). Real-vs-synthetic-trained gap is on hard questions (57.1 vs 42.1 at u500).
- quote-only run (573 updates): held quote .511 → .176, reconstruction .041 → .167, prose / benchmark flat.
- 9.8k QuALITY-50 quotes cover 39.9 % of article text, 20.0 % twice or more.

## Shared mechanism: assigned places (NOT built)
- Per round, cut each document at sentence ends into units with round-shifted cut points; every unit gets exactly one item per
  round (deterministic from the round number; no shared ledger between lanes). Releases report coverage (≥1, ≥2, ≥3 items).
- Sizes: QuALITY-50 11,403 quote units (≥12 words, mean 20) and 2,509 paragraph units (~90 words) per pass; LongHealth 3,646 / 1,236.

## Part 1 — quote-only
- Writer call: whole article + ~8 numbered assigned spans (+ earlier requests per span) → JSON, one request per span. Quote = the
  span (verbatim by construction). Checks: span unique, low word overlap request/quote, no prompt reference, near-duplicate.
- Multi-quote: writer picks 2–3 related units out of ~12 least-covered ones, writes one request; quotes one per line in the
  single `submit_answer` argument; each verified verbatim, unique, inside the assigned set.
- Training example: request + AC rows → bare call with the quote; loss 80 % quote tokens / 20 % wrapper; no reasoning.
- Run: 3,000 updates + cooldown = 24–30k examples = 2.1–2.6 passes; start pair `pq50tune_u420` unless told otherwise; hold out
  UNITS (3 %); panel + exact-match decode with right / wrong / no rows; untrained reader with article as ceiling.
- State: old-prompt quote kit `kit_i` READY on ORCD, lane 25174117 HELD on purpose (do not release before the new prompt).

## Part 2 — better question/answer
- Flow: write (assigned unit(s), asked kind, already-asked list) → reader check (flags only) → review by the 27B, medium
  reasoning, no key (`writer_pick`, `needs`) → release (whole passes, mix by `needs`, 3 % held) → training example (reasoning
  text then call; + 20 % reconstruction).
- Built today in `IB/TMP/paper_runs/synth/synth_slice.py`: `review`, `--keep writer`, `needs_by_style` in the manifest,
  `needs_table.py`. First run: ORCD job 25179764 (600 real training-split + 600 synthetic). GPU path untested before that job.
- Not built: assigned units for questions, asked-kind prompt, already-asked lists, mixing by label.
- Target variants: writer's answer (today's runs) vs frozen reader's own output (packet `qq50_fz4`, in preparation).
- Known weakness: a choice item's letter carries ≈1.4 % of its example's loss.

## Decided by the user (interaction file read 18:30Z Oct 7; resolved controls and keys removed)
D1 we assign the quote span, the writer writes the request · D2 multi-quote ≈ a third of the quote examples · D3 the quote
run carries ≈10 % reconstruction examples (free response "Add something like 10%.") · D4 QuALITY-50 first · D5 real
training-split questions only to measure and to set the label mix, never shown to the writer · D6 a choice item's answer gets
its own fixed share of the loss.

## User direction 18:45Z Oct 7 (after the PI meeting; "we have time now … longer horizons, with parallelism helping significantly")
- Old questions are NOT carried over ("not sure we should preserve the old ones, if things are being asked differently").
- Quote run: 3,500 high-quality updates. Question run: aim for 5,000 high-quality updates.
- A PROBE PHASE comes before the question synthesis ("might result in many different prompts") to get an appropriate
  distribution of questions.
- The user rejected the tool call that would have published the multi-quote kit `kit_k` on ORCD and submitted a second lane
  (interrupt to give this direction). Multi-quote is coded and self-tested locally only; do not publish/submit without a go.

- 18:51Z: "(even the quote prompts might need multi-prompt tuning to have actual diversity per passage.)" → QUOTE PROBE added
  to Part 1. Measured on covering round 400 (9,388 requests) / 401 (6,221, partial): request 23.7 words for a 23.7-word quote;
  openings "Quote the sentence" 32 %, "Quote the line" 10 %, "Quote the description" 6 %; 21 % of the quote's content words
  are in the request; requests for nested units of passes 1 and 2 are rewordings of each other (content-word Jaccard mean .22,
  25 % ≥ .3). Probe: 400 units from 10 of the 50 articles, the SAME units for every style; styles: full description (today),
  short pointer ≤10 words, question answered in the text's words, role in the story, by neighbouring events, one detail,
  by what it shows, speaker/listener; measures: round-trip rate (27B quotes the assigned unit from article + request),
  request/quote length, content-word reuse, top opening share, cross-style overlap per unit, untrained-reader ceiling. Full
  passes then assign style = (unit + round) cycled over the kept styles. Rounds 400–402 (single style) become the probe's
  reference row, NOT training data.

## Open decision (projection control `tsp.sizes2.*`; the earlier `tsp.sizes.*` keys were never answered and are retired)
D7 sizes. Quote run 3,500 updates total = 28,000 examples, decay over the last 875 (≈16,650 single = the 24- and 28-word
passes; ≈8,400 multi-quote; 2,800 reconstruction); proposed quality gate: 27B gets article + request, must quote the assigned
unit (F1 ≥ .8). Question run 5,000 updates = 40,000 examples (32,000 fresh items: prose 16,000 / choice 8,000 / quote 8,000;
8,000 reconstruction = 20 %); label mix of the 24,000 written questions 14,400 / 5,500 / 3,900 (real profile 60 / 23 / 16);
loss split 30 / 50 / 20. Probe: 12 training-split articles, ≈8 prompt variants × ≈200 items, 2–3 cycles; measures: label
profile, 27B key agreement (≥87.5 %), guessable share (≤ ≈44 %), near-duplicates, evidence in the assigned place, distinct
yield per article, cost. Flagged risk: 640 items per article (78 whole-article) is probably more than an article yields →
recommend repeating items over shortening the run. Options: as shown · cooldown on top · repeat-if-short · shorten-if-short ·
quote gate yes · other.

## User direction 19:10Z Oct 7 ("Still planning, not green lighted") — supersedes the numbers of the D7 paragraph above
- Question run back to ≈3,500 updates (3,000 + 500 cooldown). Assumed the same shape for the quote run. Control
  `tsp.sizes2.cooldown-on-top` removed as answered (key retired). 28,000 examples each; question run: 22,400 fresh items
  (prose 11,200 / choice 5,600 / quote 5,600; written questions 16,800 → 10,080 / 3,860 / 2,690), 5,600 reconstruction.
- Releases are staggered; "the ~10000 items point should be a meaningful checkpoint" → stage 1 ≈ 10,000 items as a whole,
  balanced release (comparable in size to today's 8,222-item release), stage 2 = all, containing stage 1.
- Unit size questioned: "Is 1-2 sentences the right granularity? … more thinking like ~128-256 tokens, or 256-512 … really not
  sure". New open decision D8 (`tsp.unit.*`): test three sizes (400 updates each) · 128–256 · 256–512 · mixed.
  Facts: article 6,480 tokens = 324 rows, 143 blank-line paragraphs of ≈45 tokens; 25,200 quote examples = 2.7 passes at
  sentence size, 15 at 128–256 (34 units per article), 30 at 256–512 (17 per article). Held quotes of the 573-update quote run
  (48): 14 exact; of 34 misses 9 real text from a nearby/other place, 2 right start with other borders, 23 not article text
  (right content written approximately) → copying, not finding, is the main failure (hint only).
- NOTHING is green-lighted: no probe, pass or run may be launched. `IB/TMP/paper_runs/synth/quote_probe.py` (request styles +
  lead-back gate + table) is written locally, never run.

## User direction 19:20Z Oct 7
- D8 DECIDED: "Let's just go with 64-256. It's fine." → covering quote units 64–256 tokens, no size test. Controls `tsp.unit.*`
  removed (keys retired). Simulated cut (blank-line paragraphs merged to a drawn minimum 64–200 tokens, cap 256): ≈2,000 units
  per pass on QuALITY-50 (40 per article), mean 160 tokens, p10 96, p90 226. 16,800 single examples ≈ 8.4 passes; multi-quote
  from 64–128-token pieces (my choice, stated in the projection). `units_of` must be rewritten for this (NOT done); held-out
  becomes ≈250-token stretches; `quote_probe.py` samples sentence units today and must switch too.
- "during/at the end of the probe phase, you should produce a well-formed .tressoir.md file (with cards, reveal-on-demand
  details, etc to avoid overflow) so that I know what the prompts end up looking like" → `PROBE_PROMPTS.tressoir.md` in this
  folder: table per cycle, one closed card per prompt (full text, change against the previous cycle, five items with verdict),
  proposed prompt set with shares.
- "gpu allocation and ordering … maximize parallelism across ec2, csail, orcd, before starting the two main training threads on
  EC2 2xrtx per node. Add a decision item" → section "GPUs: who does what, in which order" + D9 (`tsp.gpu.plan`, `.go`,
  `.bulk-without-review`, `.stage1-start`, `.stop-frozen`, `.feedback`). Both training threads run on the EC2 nodes
  (node 1 `qa-r10` quote run, node 2 `qa-lh` question run); CSAIL + ORCD write data and decode. STILL NOT GREEN-LIGHTED: the
  go is the `tsp.gpu.go` control or the user's word.
- To check before the go: can `run_channel_mixture.py train` take a second data stage into a running run without resetting the
  optimizer (else: second run from the stage-1 pair).

- 19:30Z user note: "(64-128 is negotiable down to something like 32-64 if we see the 27B model writing too many overlaps
  during the probe.)" My reading: the lower end of the unit size (and the 64–128 multi-quote pieces) may move down if the probe
  shows overlapping requests (requests for one unit repeating each other, or fitting a neighbouring unit). → the quote probe
  samples units in three size groups (32–64, 64–128, 128–256 tokens) and reports every measure per group.

## Answers read 19:20Z Oct 7 (interaction file: `tsp.sizes2.recommended` true, `tsp.gpu.plan` true; chat: "Ok keep that going [the frozen-answer run]. I've responded. Go.")
- D7 DECIDED: sizes table as shown. D9 DECIDED: allocation and order as shown; GO for phase 1 (probes). Controls `tsp.sizes2.*`
  and `tsp.gpu.*` removed, keys retired, stored answers cleared.
- NOT answered (left unchecked): bulk writing without the user's prompt review; question run starting on stage 1. Treated as
  "no" (stop after phase 1 for the review; question run waits for the whole release) until answered → D10 (`tsp.next.*`).
  Also open: repeat items vs shorten the run if the distinct yield per article is too low (after the probe).
- Phase 1 started 19:25Z: ORCD job 25188215 `quote_probe` (code `paper_synth/probe/code_v1`, output `paper_synth/probe/quote`).

- 19:30Z user: "(note to make: my greenlight covers up to and including training + decoding.)" → phases 1–3 are all
  authorized: probes, bulk writing (no wait for a prompt review), the quote run and the question run on the EC2 nodes, and
  their decodes. D10 withdrawn (controls `tsp.next.*` removed, keys retired). My call under the plan as shown: the question run
  starts on stage 1 and continues on stage 2. No open controls in the projection.

## Label result (18:36Z)
real (598) local 59.7 / multi-passage 23.4 / whole-article 16.2 / common-sense 0.7 %; ours QuALITY-50 83.5 / 15.2 / 0.9 / 0.3;
hard share among real per label 45.1 / 55.7 / 52.6 % (weak signal).

## State at 18:50Z
Covering quotes built (`units_of`, `write --covering`, held sentences, coverage in the manifest); ORCD lane 25174117 writing
QuALITY-50 rounds 400–402 (first batch: 1,864 items, 18 rejects, ≈20 min per pass). Review works on the GPU (synthetic:
local ≈80 %, multi-passage ≈18 %). Multi-quote coded locally (not on a cluster); quote quality gate, probe variants and located questions NOT built. Frozen-answer run training on CSAIL (2586459).

## Order
1 read the first labelling result · 2 build the covering assignment · 3 quotes: 2–3 passes, packet, 3,000-update run ·
4 questions: assigned units, kinds, mix by label, then a question run against today's 54.5 %.
