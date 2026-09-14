# Teacher probe variants: countermeasures and reasoning effort

The same 30 scored tasks (5 per benchmark, seed 0) on the oracle `unsloth/Qwen3.8-27B-NVFP4`, one RTX PRO 6000 per variant, run concurrently. v2 is the earlier probe for reference (it also ran 5 BRIGHT tasks, excluded from the per-benchmark rows below where marked). Every run of every variant is in its cache under `IB/TMP/AGENT_AC_PRETRAINING/node_<variant>/`.

## Variants

| Variant | Setting | Uncached wall | Engine context | Runs |
| --- | --- | --- | --- | --- |
| v2 (before) | non-thinking, 40 turns per segment, BRIGHT included, no countermeasures | 696 s | 56000 | 35 |
| base | non-thinking, 100 turns total, countermeasures | 1310 s | 64000 | 30 |
| low | thinking, reasoning effort low, 8k-token turns | 1878 s | 64000 | 30 |
| medium | thinking, reasoning effort medium, 8k-token turns | 2486 s | 64000 | 30 |
| xhigh | thinking, reasoning effort xhigh (the template default), 8k-token turns | 3418 s | 64000 | 30 |

## Outcomes

| Variant | Submitted | Max turns | Other finishes | Mean score (scored) | Mean turns | Max turns seen | Prompt tokens | Cache hits | Generated tokens | Compactions | Subagent calls | Parallel calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2 (before) | 28 | 7 | - | 0.43 | 22.3 | 77 | 7,232,338 | 75% | 62,867 | 7 | 0 | 0 |
| base | 20 | 1 | max_tool_errors 9 | 0.54 | 23.0 | 138 | 6,270,732 | 76% | 59,752 | 4 | 1 | 0 |
| low | 28 | 2 | - | 0.69 | 17.1 | 106 | 4,854,257 | 75% | 258,237 | 11 | 5 | 4 |
| medium | 29 | 0 | max_duration 1 | 0.76 | 13.6 | 88 | 3,556,358 | 71% | 189,234 | 9 | 4 | 7 |
| xhigh | 27 | 1 | max_duration 2 | 0.62 | 12.1 | 100 | 3,608,812 | 74% | 291,248 | 9 | 8 | 5 |

## Pathologies

Identical-call loops: runs whose longest streak of byte-identical consecutive calls is 3 or more (the countermeasure returns a note instead of re-running). Wrong argument names now run through the alias table, so `Bad arguments` counts only what aliasing could not resolve.

| Variant | Runs with a loop (longest streak) | Repeated-call notes | Bad arguments | Empty submits refused | Parse errors | No-tool nudges | Turns with visible text | Thinking chars per turn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2 (before) | 7 (32, 19, 18, 15, 14, 12, 3) | 0 | 16 | 0 | 1 | 0 | 26 / 779 | 0 |
| base | 9 (10, 10, 10, 9, 8, 8, 7, 6, 4) | 68 | 0 | 0 | 0 | 1 | 11 / 690 | 0 |
| low | 0 (-) | 2 | 1 | 0 | 0 | 1 | 47 / 514 | 1,414 |
| medium | 0 (-) | 0 | 2 | 0 | 0 | 0 | 53 / 407 | 1,336 |
| xhigh | 0 (-) | 0 | 0 | 0 | 0 | 3 | 58 / 362 | 2,132 |

## Per benchmark

Mean score / submitted of 5 / mean turns / generated tokens per run.

| Benchmark | v2 (before) | base | low | medium | xhigh |
| --- | --- | --- | --- | --- | --- |
| hotpotqa | 0.75 / 5 / 11 / 432 | 0.75 / 5 / 3 / 140 | 0.88 / 5 / 3 / 624 | 0.82 / 5 / 3 / 666 | 0.82 / 5 / 3 / 983 |
| lca_bug_localization | 0.20 / 1 / 32 / 2,292 | 0.20 / 1 / 52 / 2,735 | 0.40 / 3 / 59 / 32,472 | 0.80 / 4 / 48 / 26,030 | 0.40 / 2 / 40 / 34,861 |
| musique | 0.51 / 4 / 22 / 915 | 0.60 / 3 / 27 / 1,088 | 1.00 / 5 / 15 / 3,400 | 0.93 / 5 / 6 / 1,541 | 0.60 / 5 / 6 / 4,622 |
| quality | 0.60 / 5 / 19 / 770 | 0.80 / 5 / 4 / 222 | 0.60 / 5 / 4 / 1,396 | 0.80 / 5 / 4 / 712 | 0.60 / 5 / 5 / 2,713 |
| narrativeqa | 0.09 / 4 / 36 / 1,877 | 0.26 / 3 / 32 / 1,668 | 0.26 / 5 / 15 / 3,891 | 0.23 / 5 / 17 / 5,129 | 0.30 / 5 / 13 / 4,500 |
| dapo_math | 0.40 / 4 / 16 / 4,494 | 0.60 / 3 / 20 / 6,097 | 1.00 / 5 / 6 / 9,864 | 1.00 / 5 / 5 / 3,769 | 1.00 / 5 / 6 / 10,570 |

## Per task

Score and finish per task across variants (tasks in benchmark order).

| Benchmark | Task | v2 (before) | base | low | medium | xhigh |
| --- | --- | --- | --- | --- | --- | --- |
| hotpotqa | `5a8b57f25542995d1e6f` | 1.00 submitted (2t) | 1.00 submitted (2t) | 1.00 submitted (3t) | 1.00 submitted (2t) | 1.00 submitted (2t) |
| hotpotqa | `5adbf0a255429947ff17` | 1.00 submitted (2t) | 1.00 submitted (2t) | 1.00 submitted (3t) | 1.00 submitted (2t) | 1.00 submitted (2t) |
| hotpotqa | `5a8e3ea95542995a26ad` | 0.75 submitted (3t) | 0.75 submitted (3t) | 0.75 submitted (2t) | 0.75 submitted (2t) | 0.75 submitted (3t) |
| hotpotqa | `5a8c7595554299585d9e` | 0.00 submitted (3t) | 0.00 submitted (3t) | 0.67 submitted (4t) | 0.35 submitted (4t) | 0.35 submitted (5t) |
| lca_bug_localization | `thealgorithms__pytho` | 1.00 submitted (2t) | 1.00 submitted (2t) | 1.00 submitted (4t) | 1.00 submitted (3t) | 1.00 submitted (3t) |
| musique | `2hop__481349_302087` | 1.00 submitted (4t) | 1.00 submitted (4t) | 1.00 submitted (4t) | 1.00 submitted (4t) | 1.00 submitted (4t) |
| musique | `2hop__701895_752697` | 1.00 submitted (4t) | 1.00 submitted (75t) | 1.00 submitted (6t) | 1.00 submitted (5t) | 1.00 submitted (4t) |
| quality | `d56da82677ef-2` | 0.00 submitted (3t) | 0.00 submitted (3t) | 0.00 submitted (3t) | 0.00 submitted (3t) | 0.00 submitted (4t) |
| narrativeqa | `01502137e4276712d118` | 0.00 submitted (4t) | 0.33 submitted (5t) | 0.29 submitted (4t) | 0.31 submitted (4t) | 0.31 submitted (6t) |
| quality | `956168545a8b-2` | 0.00 submitted (4t) | 1.00 submitted (5t) | 0.00 submitted (4t) | 1.00 submitted (5t) | 0.00 submitted (3t) |
| quality | `224f98498d2c-1` | 1.00 submitted (5t) | 1.00 submitted (3t) | 1.00 submitted (6t) | 1.00 submitted (3t) | 1.00 submitted (4t) |
| dapo_math | `485b39f5-e71b-42b3-9` | 1.00 submitted (3t) | 1.00 submitted (5t) | 1.00 submitted (4t) | 1.00 submitted (3t) | 1.00 submitted (5t) |
| dapo_math | `186c94eb-3f6a-43b1-b` | 1.00 submitted (3t) | 1.00 submitted (3t) | 1.00 submitted (3t) | 1.00 submitted (3t) | 1.00 submitted (2t) |
| quality | `956168545a8b-1` | 1.00 submitted (6t) | 1.00 submitted (5t) | 1.00 submitted (2t) | 1.00 submitted (3t) | 1.00 submitted (4t) |
| musique | `2hop__259228_793698` | 0.00 submitted (10t) | 0.00 max_tool_errors (25t) | 1.00 submitted (9t) | 0.67 submitted (10t) | 0.00 submitted (10t) |
| musique | `2hop__252311_366220` | 0.57 submitted (19t) | 1.00 submitted (6t) | 1.00 submitted (6t) | 1.00 submitted (4t) | 1.00 submitted (5t) |
| narrativeqa | `00fb61fa7bee266ad995` | 0.22 submitted (16t) | 0.00 max_tool_errors (43t) | 0.15 submitted (13t) | 0.00 submitted (13t) | 0.22 submitted (10t) |
| hotpotqa | `5a85ea095542994775f6` | 1.00 submitted (47t) | 1.00 submitted (4t) | 1.00 submitted (3t) | 1.00 submitted (3t) | 1.00 submitted (3t) |
| dapo_math | `b426d104-244d-4831-a` | 0.00 submitted (10t) | 0.00 max_tool_errors (18t) | 1.00 submitted (10t) | 1.00 submitted (7t) | 1.00 submitted (8t) |
| lca_bug_localization | `electron__electron__` | 0.00 max_turns (40t) | 0.00 max_tool_errors (44t) | 1.00 submitted (27t) | 1.00 submitted (83t) | 0.00 max_duration (34t) |
| musique | `2hop__460946_294723` | 0.00 max_turns (71t) | 0.00 max_tool_errors (26t) | 1.00 submitted (52t) | 1.00 submitted (7t) | 0.00 submitted (8t) |
| narrativeqa | `00fb61fa7bee266ad995` | 0.00 max_turns (40t) | 0.67 submitted (15t) | 0.67 submitted (5t) | 0.67 submitted (15t) | 0.67 submitted (4t) |
| quality | `d56da82677ef-1` | 1.00 submitted (77t) | 1.00 submitted (6t) | 1.00 submitted (4t) | 1.00 submitted (4t) | 1.00 submitted (10t) |
| lca_bug_localization | `electron__electron__` | 0.00 max_turns (40t) | 0.00 max_tool_errors (33t) | 0.00 submitted (60t) | 1.00 submitted (88t) | 1.00 submitted (46t) |
| narrativeqa | `042bb7019f583cebe329` | 0.00 submitted (58t) | 0.00 max_tool_errors (89t) | 0.00 submitted (45t) | 0.00 submitted (42t) | 0.00 submitted (35t) |
| lca_bug_localization | `keras-team__keras__1` | 0.00 max_turns (40t) | 0.00 max_tool_errors (44t) | 0.00 max_turns (106t) | 1.00 submitted (12t) | 0.00 max_duration (15t) |
| lca_bug_localization | `keras-team__keras__1` | 0.00 max_turns (40t) | 0.00 max_turns (138t) | 0.00 max_turns (100t) | 0.00 max_duration (52t) | 0.00 max_turns (100t) |
| narrativeqa | `01502137e4276712d118` | 0.24 submitted (64t) | 0.31 submitted (7t) | 0.20 submitted (8t) | 0.17 submitted (11t) | 0.29 submitted (9t) |
| dapo_math | `6ff0b17f-7e5c-4ae9-b` | 0.00 max_turns (40t) | 1.00 submitted (15t) | 1.00 submitted (9t) | 1.00 submitted (4t) | 1.00 submitted (8t) |
| dapo_math | `9a9b6eb4-a1cb-49d1-8` | 0.00 submitted (26t) | 0.00 max_tool_errors (57t) | 1.00 submitted (5t) | 1.00 submitted (6t) | 1.00 submitted (6t) |

## Reading

- **Thinking removes the degeneration.** Non-thinking runs (v2 and base) loop on byte-identical calls; with thinking on, no run repeated a call three times. The repeated-call note does not break a non-thinking loop: the model re-issues the same call against the note (68 notes in base), so those runs end at max_tool_errors after 8 notes instead of burning 30 more turns. It is a budget saver, not a cure.
- **Scores.** Mean over the 30 scored tasks: base 0.54, low 0.69, medium 0.76 (v2 0.43 under the old budget). Medium solves 4 of 5 bug-localization tasks that no non-thinking run solved, and every math and MuSiQue task. QuALITY and NarrativeQA do not move: the remaining errors there are reading and answer-phrasing errors (NarrativeQA is token F1 against short gold answers).
- **Delegation and parallel calls appear only with thinking** (low 5 subagent calls and 4 parallel calls, medium 4 and 7, none in 60 non-thinking runs). The briefs follow the TASK / CONTEXT / TO PRODUCE / DON'T DO format from the tool description.
- **Cost.** Thinking generates 3 to 4 times the tokens (189k to 258k against 60k) in fewer turns (13.6 and 17.1 mean turns against 23.0), and the uncached pass is bounded by the longest run: 1310 s base, 1878 s low, 2486 s medium (one medium bug-localization run hit the 1800 s duration cap at 52 turns). Prompt tokens drop with thinking because runs are shorter; cache hit rates stay around 75%.
- **Thinking length** is about 1,400 characters per turn (roughly 350 tokens) at both efforts; the 8,192-token turn was never the binding limit. Visible text outside the think block is rare (about one turn in ten), the model goes from thinking straight to the calls.
- **Aliasing worked**: 16 wrong-argument-name errors in v2, 0 to 2 across the three new variants (the rest resolved by the alias table or the lone-key rule). No empty submission was attempted.
- **xhigh (the template default) scores 0.62, below medium**, at 1.5 times medium's generated tokens and 3418 s of wall. Its thinking averages 2,100 characters per turn and 5 turns hit the 8,192-token turn limit (the reply is cut before the call, hence the 3 nudges). Two bug-localization runs died at the 1800 s duration cap (they were at 2123 and 2220 s when the between-turn check fired) and a third at 100 turns; it also missed two MuSiQue tasks that medium solved. Longer thinking per turn is slower per turn at this concurrency, and the budget binds before the extra thought pays off.
- **Two tasks stay unsolved everywhere**: the QuALITY question `d56da82677ef-2` (all variants answer C, gold B) and the keras EfficientNet localization (max turns or duration in every variant).

