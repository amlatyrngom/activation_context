# Gradient clipping: fixed 100-question readout

The candidate clips encoder/AC and reader gradients separately to norm 1. The control clips their combined norm to 1. Both use the original P2 reconstruction pair, a 2B encoder and 9B reader, 16,000 unique 27B-synthesized questions × 1 epoch, 1/8 compression, the same learning rates and data order. Separate clipping permits a combined norm up to √2.

The original RTX training endpoints give held 400 NLL 0.64627→0.62936 and fixed training 32 NLL 0.52183→0.42966. Separate clipping reduces extreme gradient spikes, but this modest NLL change alone does not establish better answers.

## Matched GPU readout

Both AC finals were rescored on L40S using the same frozen evaluator, original 400 order/batch partners, exact source evidence, gold tokens/masks and fixed 100 selection. Greedy decoding disables thinking, uses at most 160 new tokens and stops at chat-end or inherited EOS.

| View | Joint clipping | Separate clipping |
|---|---:|---:|
| Held400 mean gold NLL |0.64611|0.62864|
| Fixed100 mean gold NLL |0.67955|0.69331|
| Fixed 50 from 9B generator |0.70510|0.69372|
| Fixed 50 from 27B generator |0.65400|0.69291|

These are equal-question means; lower is better.198 of 400 question losses favor separate clipping. The fixed 100 subset reverses the mean direction, so it must not replace the full400 panel as the NLL summary. Actual answer judgments are supplied in the HTML and evidence file when the full review is complete.

The candidate’s L40S replay differs from its RTX training panel by−0.00072 in mean NLL,0.00973 mean absolute per-question difference and0.2647 maximum. The maximum belongs to a two-token answer outside fixed 100. Input, mask, checkpoint and source checks match; the backend cause is unproven. The control’s mean replay difference is−0.00016. We preserved both RTX outputs and reran the control on L40S before comparing answers.

Historical frozen P2 full-context/no-context answers remain unchanged and explicitly labeled RTX. They are orientation columns. Full context includes the same retrieved distractors as AC. The two clipping columns use L40S; this controls GPU type, without claiming bitwise determinism across devices or identifying the cause of the earlier drift.

## Source review and limits

All 100 questions remain in NLL. The pre-existing source-only review fixes 87 supported references and 13 concerns for the optional semantic summary. Every candidate answer is reviewed against the source. For the L40S control, 81 byte-identical response/answer pairs reuse their original judgments; 19 changed responses receive a new source review. Historical annotations are preserved, with known inconsistencies disclosed. Agent judgments are inspectable, not an independent benchmark.

This is one seed and a repeatedly examined development set. No reserved confirmation questions were used. The matched-L40S source review gives control 52 correct / 12 partial / 23 incorrect versus separate clipping 51 correct / 16 partial / 20 incorrect on the fixed 87 supported references. Seven answers become fully correct and eight lose that status. Both arms finish all 100 answers at EOS with parseable answer calls. There is no clear semantic improvement, so the main training recipe retains joint clipping.

## Receipts

- Candidate job 23493122: complete in 6m23s, 25.48 GiB peak allocated; control 23494064: complete in 6m06s, 25.48 GiB peak allocated.
- Original source/evaluator/selection preserved; each final pair is validated before scoring. Control output/cache use new directories, so historical RTX evidence was not overwritten.
- Local review inputs, source annotations, exact generations and hashes are in `IB/TMP/FIRST_TASK_AWARE_FINETUNING/group_clipping_cohort_review/`; numerical replay investigation is `GROUP_CLIP_L40S_REPLAY_REVIEW.md` in the parent TMP folder.
- The companion HTML is a static snapshot without a refresh timer or automatic scrolling. Standalone Chromium checks passed: 100 questions / 400 answer cards, all source filters, search, disclosures, feedback/Escape, repeated render events, and no desktop/mobile overflow or script errors. Actual VSCode CSP/morph behavior is not tested here.
