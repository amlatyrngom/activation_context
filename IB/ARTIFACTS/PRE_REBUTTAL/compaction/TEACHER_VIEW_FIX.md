# KL-teacher view fix for compaction stage 1 (2026-11-17)

Status: code and data prepared; **no training started** (the user reviews first). Background:
`AGENT_EVAL_DEV75.md` §5 (the res pair writes `[/shell]` instead of `<tool_call>`) and §8 (the probe: the KL stage, not lcrec).

## The bug

`ActivationContextTrainer.build_example` builds the KL teacher's history with `prepare_training_history`. That function renders
every message before the first target with `_training_template_message(..., reasoning_as_text=True)`. The agent records each
turn's reasoning inline, ending in `</think>`, so in the teacher's own input every prefix turn reads:

```
<think>\n\n</think>\n\n<reasoning text>\n[/Reasoning]\n\n<tool_call>...
```

The untuned 9B copies the pattern. At the first continuation turn it closes its thinking with `[/` in 44 % of the dev-75
prefixes (mean log-prob −1.43). KL stage 1 trains the pair toward that teacher, so both KL pairs carry the habit (probe, §8).

The agent eval's `full` arm never reads this view. At serving the prompt is the head followed by each prefix turn's recorded
tokens, with a real `</think>`, and that arm showed 0 garbles.

## The fix (opt-in per item; the corrected view is the new default for compaction packets)

**Main code**, two files, 26 added lines / 6 removed (commit `6f992e9`):
- `activation/ac_model/ac_model_utils.py`, `prepare_training_history(..., history_ids=None)`. This is a new keyword argument
  holding the recorded tokens of every assistant message before `start`.
  - With it, the history is rebuilt as the agent built it at run time: the messages before the first assistant turn by the
    template, each earlier turn as its recorded tokens, and the messages between turns by the template's continuation wrapper
    (`dialect.continuation_tokens` + `join_continuation`, the calls `Agent._continuation` makes).
  - Earlier turns are context only; positions are recorded for targets alone.
  - Without it, nothing changes.
- `activation/ac_model/ac_model_training.py`, `build_example`. The teacher-view call passes
  `history_ids=item.info.get("in_context_history_ids")`. The student view, the encoder input and the no-context reference are
  untouched.

**Encoder side: unchanged.** The AC part still renders past reasoning as `[Reasoning]` text. The probe shows the pairs do not leak
`[/` from it: the recon-only pair and the start pair both stay at −18 on the panel. The escaping keeps the template from
stripping earlier thoughts from the encoder's input.

**Packet tools** (`compaction/stage1/`):
- `export_pool_material.py`: every material line now also holds `prefix_assistant_token_ids` (each prefix assistant step's
  recorded `token_ids`). This field is additive; the rest of the line is unchanged. Re-exporting `eval_v3_c20` gives 278 of 278
  lines identical apart from this field.
- `prepare_agent_continuation.py` and `build_dev_panel.py`: `build_item` sets `info["in_context_history_ids"]` from that field
  by default, and refuses material without it. `--escaped-teacher-prefix` rebuilds the old view, so old packets stay
  reproducible. The recipe's `arm` and `panel.json` record `teacher_view: recorded | escaped`.
- Old packets carry no `in_context_history_ids` and build exactly as before (the same teacher ids, hence the same teacher-cache
  keys).

**Important for training:** the ORCD bundle `cmp_stage1_code_v1` (commit 6813698) predates this change. That trainer silently
ignores the new info field and would rebuild the escaped view. A corrected run needs a new code bundle cut from main at or
after `6f992e9`.

## Checks

**Token identity with the served `full` arm** (`stage1/check_teacher_view.py`, CPU; `cmp_agent_eval/report/teacher_view_tokens.json`):
- For all 75 dev-75 prefixes, the corrected teacher prompt up to the first continuation turn is token-identical to the
  `prompt_token_ids` the agent eval's `full` arm sent (`cdv1base_s2_full`): 75 of 75.
- The continuation part is identical between the old and new views: 75 of 75.
- The prompts hold `[/Reasoning]` 0 times in the new view and 1954 times in the old one (once per prefix turn). Both hold
  `</think>` 1954 times.
- Mean prompt length: 20,696.2 tokens new vs 20,825.3 old.

**New data, all under new names** (S3 `.../compaction/stage1/`; local `IB/TMP/PRE_REBUTTAL/compaction_stage1/`):
- `material/eval_v4tv_c20.jsonl`: the dev-side material, re-exported with the token ids.
- `panel_dev75_tv/`: the dev-75 proxy panel with the corrected teacher view. Student views are unchanged, with the same groups,
  rows and targets (497,993 target tokens). Teacher tokens: mean 31,867, vs 31,996 in `panel_dev75`.
- `material/s1kl_tvfix_small_v1_c20.jsonl` (the first 64 of 173 usable runs of `pc_pool9_h6_trA`, build_stage1's cut rule) and
  `packets/s1kl_tvfix_small_v1/` (64 KL items, `teacher_view: recorded`, start pair R_res_rq9fr). This is a small check packet,
  not for training.
- `proxy_ref_tv/`: `cmp_proxy_eval.py --references` with the start pair on `panel_dev75_tv` (new teacher targets
  `teacher_dev75_tv.pt` and the new NLL_none / NLL_full references). See the table below.

**Old vs corrected teacher, both scored by the untuned 9B** (`agent_eval/teacher_view_compare.py` on node cdv4;
S3 `.../agent_eval_dev75/probe/tvcompare_v1.json`):

| | dev-75 panel (75 prefixes) | small packet (64 runs) |
|---|---|---|
| target tokens | 497,993 | 462,379 |
| NLL_full old → new | 0.50567 → 0.49915 | 0.51653 → 0.51013 |
| KL(old ‖ new) per token, mean | 0.0081 | 0.0078 |
| items with mean KL > 0.005 / 0.01 / 0.02 / 0.05 | 70 / 29 / 10 / 3 | 56 / 23 / 5 / 1 |
| items with first-turn KL > 0.01 / 0.05 / 0.1 / 0.5 | 74 / 35 / 9 / 0 | 64 / 26 / 9 / 0 |
| per-item mean KL: median / 90th pct / max | 0.0090 / 0.0227 / 0.1006 | 0.0088 / 0.0168 / 0.0925 |
| top-1 agreement (token mean) | 97.85 % | 97.81 % |
| top-32 overlap (token mean; the trainer stores the top 32) | 94.32 % | 94.32 % |
| `</think>` positions: mean lp `</think>` / `[/`, % `[/` > `</think>` (old) | -0.085 / -7.83, 2.84 % | -0.073 / -8.15, 1.86 % |
| same (new) | -0.023 / -19.71, 0.00 % | -0.020 / -19.75, 0.00 % |
| first continuation turn only (old) | -0.731 / -1.43, 44.0 % | -0.800 / -1.79, 32.8 % |
| first continuation turn only (new) | -0.005 / -19.25, 0.00 % | -0.037 / -19.17, 0.00 % |

The recomputed old NLL_full on the dev-75 panel (0.505667) matches the stored proxy reference
(`proxy_start_pair.json`: 0.505665).

**What changes downstream.**
- The leak is gone: `[/` never outranks `</think>` in the corrected view, and its mean log-prob falls from about −8 to about −20.
- The teacher's other targets move a little everywhere (top-1 agreement 97.8 %, top-32 overlap 94.3 %), most at the first
  continuation turn.
- NLL_full drops by 0.0065, so the dev-75 proxy's denominator NLL_none − NLL_full grows from 0.1392 to 0.1457. Rescaling
  existing proxies with their unchanged student NLLs (the student and none views do not change) lowers gap closed by about 4.5 %:

  | pair | r10 gap closed (old reference → new) | r5 gap closed (old reference → new) |
  |---|---|---|
  | S1_res_kl_both_c20 | 0.510 → 0.487 | 0.608 → 0.581 |
  | S1_lcrec100_kl_both_c20 | 0.605 → 0.578 | 0.710 → 0.678 |
  | R_cmp_lcrec | −1.057 → −1.009 | −0.967 → −0.923 |

  Proxy numbers before and after the fix are not comparable without this rescaling.
