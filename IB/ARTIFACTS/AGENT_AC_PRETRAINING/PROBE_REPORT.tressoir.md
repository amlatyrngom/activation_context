# Teacher probe v2

The first oracle rollouts over the slice 4a benchmark suite: 5 tasks per benchmark, one seed, on `unsloth/Qwen3.8-27B-NVFP4` (engine context 56,000) with the default tools plus the general subagent and semantic search, in the default image, on one RTX PRO 6000. Run uncached and then from the cache under `teacher_probe_v2`. This page is the artifact you asked to refine the interfaces and the report on; the numbers come from `probe.json`, every run links its trajectory file, and the observations and questions at the end are mine.

## Headline

| | |
| --- | --- |
| Tasks | 35 (hotpotqa 5, musique 5, dapo_math 5, lca_bug_localization 5, quality 5, narrativeqa 5, bright 5) |
| Uncached pass | 696 s wall, all tasks concurrent; longest run 391 s, mean 175 s |
| Cached pass | 0.1 s, 35 of 35 runs served from the cache |
| Turns | 779 agent turns, 7 compactions, 0 subagent calls |
| Tokens | 7,232,338 prompt tokens read (75% served from the prefix cache), 62,867 generated |
| Throughput | 90.4 generated tokens/s of wall time; 2,648 uncached prompt tokens/s: the pass is prefill-bound at 35 concurrent agents |
| Budgets | 40 turns per segment, 1800 s per task, compaction at 32,768 tokens, 4,096-token turns |
| Suite load | 28.6 s on the node (datasets streamed, one repository tarball per repo task) |

## Per benchmark (uncached pass)

| benchmark | finish reasons | turns mean / max | prompt tokens mean | generated mean | compactions | search | shell | python | score | framings |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hotpotqa | submitted 5 | 11.4 / 47 | 88,930 | 432 | 1 | 30 | 0 | 0 | 0.75 | minimal 3, guided_search_then_verify 2 |
| musique | max_turns 1, submitted 4 | 21.6 / 71 | 176,085 | 915 | 1 | 81 | 0 | 0 | 0.51 | minimal 3, guided_search_then_verify 2 |
| dapo_math | submitted 4, max_turns 1 | 16.4 / 40 | 115,604 | 4,494 | 0 | 0 | 4 | 74 | 0.40 | minimal 3, guided_plan_solve_validate 2 |
| lca_bug_localization | submitted 1, max_turns 4 | 32.4 / 40 | 358,966 | 2,292 | 0 | 4 | 157 | 0 | 0.20 | minimal 3, guided_sequential_localize 2 |
| quality | submitted 5 | 19.0 / 77 | 139,134 | 770 | 1 | 3 | 48 | 0 | 0.60 | minimal 3, guided_delegate_sections 2 |
| narrativeqa | submitted 4, max_turns 1 | 36.4 / 64 | 389,472 | 1,877 | 2 | 7 | 119 | 0 | 0.09 | minimal 3, guided_delegate_sections 2 |
| bright | submitted 5 | 18.6 / 44 | 178,276 | 1,794 | 2 | 67 | 10 | 0 | unscored | minimal 3, guided_iterative_search 2 |

Turns count every segment of a run: `max_turns` applies per segment and the counter restarts after a compaction, which is why runs reach 64 to 77 turns. Prompt tokens are the sum over turns of what the engine read; three quarters came from the prefix cache.

## Runs

One line per task, linking the uncut trajectory. Tool counts include the members of parallel calls.

**hotpotqa**

- [5a8b57f25542995d1e6f1371](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/784338ea.json) · minimal · submitted · 2 turns · 3,869 in (0 cached) / 112 out · 0 compactions · semantic_search:2,submit_answer:1 · score 1.00 · answer 'Yes' (gold 'yes') · 63 s
- [5a8c7595554299585d9e36b6](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/f4bc8b22.json) · minimal · submitted · 3 turns · 7,517 in (0 cached) / 112 out · 0 compactions · semantic_search:2,submit_answer:1 · score 0.00 · answer '' (gold 'Chief of Protocol') · 68 s
- [5a85ea095542994775f606a8](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/14490c30.json) · guided_search_then_verify · submitted · 47 turns · 423,055 in (336,000 cached) / 1,671 out · 1 compactions · semantic_search:22,compact:1,submit_answer:1 · score 1.00 · answer 'Animorphs' (gold 'Animorphs') · 205 s
- [5adbf0a255429947ff17385a](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/335bdf87.json) · minimal · submitted · 2 turns · 4,011 in (0 cached) / 147 out · 0 compactions · semantic_search:2,submit_answer:1 · score 1.00 · answer 'No' (gold 'no') · 65 s
- [5a8e3ea95542995a26add48d](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/329e3dee.json) · guided_search_then_verify · submitted · 3 turns · 6,200 in (0 cached) / 120 out · 0 compactions · semantic_search:2,submit_answer:1 · score 0.75 · answer 'New York City' (gold 'Greenwich Village, New York City') · 68 s

**musique**

- [2hop__460946_294723](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/274ee031.json) · minimal · max_turns · 71 turns · 669,363 in (524,800 cached) / 2,838 out · 1 compactions · semantic_search:48,compact:1 · score 0.00 · answer '' (gold 'Miquette Giraudy') · 265 s
- [2hop__252311_366220](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/d670a4b9.json) · minimal · submitted · 19 turns · 134,004 in (81,600 cached) / 954 out · 0 compactions · semantic_search:18,submit_answer:1 · score 0.57 · answer 'Mike Medavoy and Michael Lynton' (gold 'Mike Medavoy') · 165 s
- [2hop__701895_752697](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/ebd3607d.json) · guided_search_then_verify · submitted · 4 turns · 13,444 in (1,600 cached) / 172 out · 0 compactions · semantic_search:3,submit_answer:1 · score 1.00 · answer 'Tamaulipas' (gold 'Tamaulipas') · 80 s
- [2hop__259228_793698](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/aab463b6.json) · minimal · submitted · 10 turns · 54,714 in (25,600 cached) / 455 out · 0 compactions · semantic_search:9,submit_answer:1 · score 0.00 · answer 'Bonn, Germany' (gold 'Cologne') · 115 s
- [2hop__481349_302087](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/3494e319.json) · guided_search_then_verify · submitted · 4 turns · 8,901 in (0 cached) / 155 out · 0 compactions · semantic_search:3,submit_answer:1 · score 1.00 · answer 'Bombardier Inc.' (gold 'Bombardier Inc.') · 78 s

**dapo_math**

- [9a9b6eb4-a1cb-49d1-8c1e-62ea](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/b42dfb24.json) · minimal · submitted · 26 turns · 226,103 in (158,400 cached) / 9,973 out · 0 compactions · python:21,shell:4,submit_answer:1 · score 0.00 · answer '81' (gold '34') · 391 s
- [b426d104-244d-4831-a2c4-cd75](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/9d593802.json) · minimal · submitted · 10 turns · 40,531 in (16,000 cached) / 2,975 out · 0 compactions · python:9,submit_answer:1 · score 0.00 · answer '21' (gold '113') · 244 s
- [6ff0b17f-7e5c-4ae9-b5e9-63eb](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/16d6d887.json) · guided_plan_solve_validate · max_turns · 40 turns · 302,062 in (204,800 cached) / 8,414 out · 0 compactions · python:40 · score 0.00 · answer '' (gold '-3') · 326 s
- [186c94eb-3f6a-43b1-b2a4-f7de](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/159ec885.json) · minimal · submitted · 3 turns · 4,423 in (0 cached) / 546 out · 0 compactions · python:2,submit_answer:1 · score 1.00 · answer '3' (gold '3') · 96 s
- [485b39f5-e71b-42b3-931f-219b](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/b812e152.json) · guided_plan_solve_validate · submitted · 3 turns · 4,899 in (0 cached) / 560 out · 0 compactions · python:2,submit_answer:1 · score 1.00 · answer '37' (gold '37') · 95 s

**lca_bug_localization**

- [thealgorithms__python__295__](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/32c7d79e.json) · minimal · submitted · 2 turns · 2,884 in (0 cached) / 80 out · 0 compactions · shell:1,submit_answer:1 · score 1.00 · answer 'Project Euler/Problem 01/sol2.py' (gold 'Project Euler/Problem 01/sol2.py') · 65 s
- [electron__electron__8668__85](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/756ec480.json) · minimal · max_turns · 40 turns · 266,866 in (168,000 cached) / 2,932 out · 0 compactions · shell:38,parse_error:1,semantic_search:1 · score 0.00 · answer '' (gold 'script/lib/config.py') · 261 s
- [electron__electron__8640__86](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/8c74305a.json) · guided_sequential_localize · max_turns · 40 turns · 561,359 in (446,400 cached) / 2,454 out · 0 compactions · shell:38,semantic_search:2 · score 0.00 · answer '' (gold 'script/lib/config.py') · 250 s
- [keras-team__keras__18352__15](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/6a2274c9.json) · minimal · max_turns · 40 turns · 412,135 in (297,600 cached) / 2,916 out · 0 compactions · shell:40 · score 0.00 · answer '' (gold 'keras/applications/mobilenet_v3.py') · 268 s
- [keras-team__keras__17498__17](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/7c2bd568.json) · guided_sequential_localize · max_turns · 40 turns · 551,588 in (440,000 cached) / 3,076 out · 0 compactions · shell:40,semantic_search:1 · score 0.00 · answer '' (gold 'keras/applications/efficientnet.py') · 268 s

**quality**

- [d56da82677ef-1](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/3f1eddef.json) · minimal · submitted · 77 turns · 617,070 in (494,400 cached) / 2,767 out · 1 compactions · shell:37,compact:1,submit_answer:1 · score 1.00 · answer 'B' (gold 'B') · 260 s
- [d56da82677ef-2](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/c852c079.json) · minimal · submitted · 3 turns · 5,951 in (0 cached) / 111 out · 0 compactions · shell:2,submit_answer:1 · score 0.00 · answer 'C' (gold 'B') · 71 s
- [956168545a8b-1](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/b51ac1ba.json) · guided_delegate_sections · submitted · 6 turns · 45,755 in (24,000 cached) / 389 out · 0 compactions · shell:3,semantic_search:2,submit_answer:1 · score 1.00 · answer 'B' (gold 'B') · 99 s
- [956168545a8b-2](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/ba4dcb15.json) · minimal · submitted · 4 turns · 6,778 in (0 cached) / 246 out · 0 compactions · shell:3,submit_answer:1 · score 0.00 · answer 'C' (gold 'A') · 87 s
- [224f98498d2c-1](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/c80b7eda.json) · guided_delegate_sections · submitted · 5 turns · 20,117 in (4,800 cached) / 337 out · 0 compactions · shell:3,semantic_search:1,submit_answer:1 · score 1.00 · answer 'D' (gold 'D') · 92 s

**narrativeqa**

- [00fb61fa7bee266ad995e52190eb](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/eb1dde6a.json) · minimal · submitted · 16 turns · 113,893 in (70,400 cached) / 983 out · 0 compactions · shell:15,submit_answer:1 · score 0.22 · answer 'One of the children (a child actor)' (gold 'THE ACTOR WEARING THE BLACK CLOAK') · 164 s
- [00fb61fa7bee266ad995e52190eb](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/14fbfc62.json) · minimal · max_turns · 40 turns · 619,476 in (505,600 cached) / 2,557 out · 0 compactions · shell:40 · score 0.00 · answer '' (gold 'THE GODDESS DIANA') · 256 s
- [01502137e4276712d118bd4bdaf4](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/2f2b416c.json) · guided_delegate_sections · submitted · 64 turns · 670,894 in (537,600 cached) / 3,022 out · 1 compactions · shell:33,semantic_search:3,compact:1,submit_answer:1 · score 0.24 · answer 'Because his planned opponent Mamby got the flu and was replaced by Mik' (gold 'Opponent was in a weight class of 20 pou') · 269 s
- [01502137e4276712d118bd4bdaf4](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/033e5b86.json) · minimal · submitted · 4 turns · 9,489 in (0 cached) / 194 out · 0 compactions · shell:3,submit_answer:1 · score 0.00 · answer 'His head is split open (bleeding) from the police beating him with fla' (gold 'Broken hand') · 84 s
- [042bb7019f583cebe3290795149f](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/a2d7147e.json) · guided_delegate_sections · submitted · 58 turns · 533,609 in (412,800 cached) / 2,630 out · 1 compactions · shell:28,semantic_search:4,compact:1,submit_answer:1 · score 0.00 · answer 'She is pregnant with his baby (a child)' (gold 'A pregnancy.') · 262 s

**bright**

- [bright:biology:0](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/406b4522.json) · minimal · submitted · 32 turns · 288,753 in (203,200 cached) / 3,179 out · 1 compactions · semantic_search:27,shell:4,compact:1,submit_answer:1 · answer 'Your "light = heat association" idea is a plausible-sounding alternati' (gold "$\\begingroup$\n\nThey're looking for and d") · 358 s
- [bright:biology:1](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/90fe8190.json) · minimal · submitted · 3 turns · 7,307 in (0 cached) / 396 out · 0 compactions · semantic_search:2,submit_answer:1 · answer 'What you noticed is the nasal cycle — a normal, unconscious physiologi' (gold "$\\begingroup$\n\nApparently you're not the") · 93 s
- [bright:biology:2](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/f574342d.json) · guided_iterative_search · submitted · 44 turns · 456,919 in (358,400 cached) / 3,230 out · 1 compactions · semantic_search:25,shell:2,compact:1,submit_answer:1 · answer 'No — there is no true RGB equivalent for smell. RGB works because huma' (gold '$\\begingroup$\n\nThere are about 100 [ (Pu') · 268 s
- [bright:biology:3](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/6cce921f.json) · minimal · submitted · 7 turns · 58,120 in (30,400 cached) / 1,197 out · 0 compactions · semantic_search:7,shell:2,submit_answer:1 · answer 'Yes — animals do exhibit "paw-ness"/limb dominance, and your cat is pl' (gold '$\\begingroup$\n\n###  Short Answer\n\nYes. h') · 163 s
- [bright:biology:4](../../TMP/AGENT_AC_PRETRAINING/node/probe_v2/uncached/trajectories/dc7d70d4.json) · guided_iterative_search · submitted · 7 turns · 80,279 in (48,000 cached) / 967 out · 0 compactions · semantic_search:6,shell:2,submit_answer:1 · answer 'Yes — the corpus (the Color blindness article) identifies circumstance' (gold '$\\begingroup$\n\n**Short answer**  \nColor-') · 149 s

## Excerpts

The most compaction-heavy and the most search-heavy runs (no run used a subagent). Calls and results are cut; the linked trajectory files hold everything.

### Most compaction-heavy run

**Segment 1** (23 turns, compacted)
- turn:  -> `semantic_search({"query": "science fantasy young adult series first person companion books enslaved worlds alien species", "top_k": 10})`
  - result: [wiki/Animorphs:0] Animorphs Animorphs is a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic. It is told in first person, with all six main characters taking turns narrating the books through their own perspectives. Horror, war, dehumanization, sanity, morality, inn…
- turn:  -> `semantic_search({"query": "Animorphs companion books Andalite Chronicles Ellimist Chronicles Megamorphs enslaved worlds alien species", …)`
  - result: [wiki/The Hork-Bajir Chronicles:0] The Hork-Bajir Chronicles The Hork-Bajir Chronicles is the second companion book to the "Animorphs" series, written by K. A. Applegate. With respect to continuity within the series, it takes place before book #23, "The Pretender", although the events told in the story occur between the time of "The Ellimist Chronicles" and "The Andalite Chronicles". The book is i…
- turn:  -> `semantic_search({"query": "Animorphs companion books narrating stories of enslaved worlds and alien species", "top_k": 8})`
  - result: [wiki/Animorphs:0] Animorphs Animorphs is a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic. It is told in first person, with all six main characters taking turns narrating the books through their own perspectives. Horror, war, dehumanization, sanity, morality, inn…
- turn:  -> `semantic_search({"query": "Animorphs companion books \"The Andalite Chronicles\" \"The Ellimist Chronicles\" enslaved worlds alien speci…)`
  - result: [wiki/The Hork-Bajir Chronicles:0] The Hork-Bajir Chronicles The Hork-Bajir Chronicles is the second companion book to the "Animorphs" series, written by K. A. Applegate. With respect to continuity within the series, it takes place before book #23, "The Pretender", although the events told in the story occur between the time of "The Ellimist Chronicles" and "The Andalite Chronicles". The book is i…
- turn:  -> `semantic_search({"query": "Animorphs spin-off companion books \"The Yeerk War\" Megamorphs \"narrating the stories of enslaved worlds an…)`
  - result: [wiki/Animorphs:0] Animorphs Animorphs is a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic. It is told in first person, with all six main characters taking turns narrating the books through their own perspectives. Horror, war, dehumanization, sanity, morality, inn…
- turn:  -> `semantic_search({"query": "The Andalite Chronicles companion book Animorphs Yeerks enslaved worlds alien species narrating stories", "to…)`
  - result: [wiki/The Hork-Bajir Chronicles:0] The Hork-Bajir Chronicles The Hork-Bajir Chronicles is the second companion book to the "Animorphs" series, written by K. A. Applegate. With respect to continuity within the series, it takes place before book #23, "The Pretender", although the events told in the story occur between the time of "The Ellimist Chronicles" and "The Andalite Chronicles". The book is i…
- turn:  -> `semantic_search({"query": "Animorphs companion books \"narrating the stories\" enslaved worlds alien species first person", "top_k": 8})`
  - result: [wiki/Animorphs:0] Animorphs Animorphs is a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic. It is told in first person, with all six main characters taking turns narrating the books through their own perspectives. Horror, war, dehumanization, sanity, morality, inn…
- turn:  -> `semantic_search({"query": "Animorphs companion books \"The Andalite Chronicles\" \"The Ellimist Chronicles\" \"The Hork-Bajir Chronicles…)`
  - result: [wiki/The Hork-Bajir Chronicles:0] The Hork-Bajir Chronicles The Hork-Bajir Chronicles is the second companion book to the "Animorphs" series, written by K. A. Applegate. With respect to continuity within the series, it takes place before book #23, "The Pretender", although the events told in the story occur between the time of "The Ellimist Chronicles" and "The Andalite Chronicles". The book is i…
- turn:  -> `semantic_search({"query": "Animorphs companion books \"The Yeerk War\" Megamorphs first person narrating enslaved worlds alien species",…)`
  - result: [wiki/Animorphs:0] Animorphs Animorphs is a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic. It is told in first person, with all six main characters taking turns narrating the books through their own perspectives. Horror, war, dehumanization, sanity, morality, inn…
- turn:  -> `semantic_search({"query": "Animorphs companion books \"The Yeerk War\" Megamorphs first person narrating enslaved worlds alien species",…)`
  - result: [wiki/Animorphs:0] Animorphs Animorphs is a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic. It is told in first person, with all six main characters taking turns narrating the books through their own perspectives. Horror, war, dehumanization, sanity, morality, inn…
- turn:  -> `semantic_search({"query": "Animorphs companion books \"The Yeerk War\" Megamorphs first person narrating enslaved worlds alien species",…)`
  - result: [wiki/Animorphs:0] Animorphs Animorphs is a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic. It is told in first person, with all six main characters taking turns narrating the books through their own perspectives. Horror, war, dehumanization, sanity, morality, inn…
- turn:  -> `semantic_search({"query": "Animorphs companion books \"The Yeerk War\" Megamorphs first person narrating enslaved worlds alien species",…)`
  - result: [wiki/Animorphs:0] Animorphs Animorphs is a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic. It is told in first person, with all six main characters taking turns narrating the books through their own perspectives. Horror, war, dehumanization, sanity, morality, inn…
- turn:  -> `semantic_search({"query": "Animorphs companion books \"The Yeerk War\" Megamorphs first person narrating enslaved worlds alien species",…)`
  - result: [wiki/Animorphs:0] Animorphs Animorphs is a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic. It is told in first person, with all six main characters taking turns narrating the books through their own perspectives. Horror, war, dehumanization, sanity, morality, inn…
- turn:  -> `semantic_search({"query": "Animorphs companion books \"The Yeerk War\" Megamorphs first person narrating enslaved worlds alien species",…)`
  - result: [wiki/Animorphs:0] Animorphs Animorphs is a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic. It is told in first person, with all six main characters taking turns narrating the books through their own perspectives. Horror, war, dehumanization, sanity, morality, inn…
- ... 9 more turns
**Segment 2** (24 turns)
- turn:  -> `submit_answer({"answer": "Animorphs"})`
  - result: Answer submitted: Animorphs
- finish: submitted; answer 'Animorphs'

### Most search-heavy run

**Segment 1** (31 turns, compacted)
- turn:  -> `semantic_search({"query": "Green performer spouse", "top_k": 10})`
  - result: [wiki/List of show business families:0] List of show business families Actress / director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phyl…
- turn:  -> `semantic_search({"query": "\"Green\" performer married spouse wife husband musician", "top_k": 10})`
  - result: [wiki/List of show business families:0] List of show business families Actress / director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phyl…
- turn:  -> `semantic_search({"query": "Green singer actor \"Green\" biography", "top_k": 10})`
  - result: [wiki/Green Lake (Chisago City, Minnesota):0] Green Lake (Chisago City, Minnesota) Green Lake is a lake in Chisago City, Minnesota, United States. This lake is sometimes also referred to as "Big Green Lake" because it is connected to Little Green Lake by a channel. Green Lake was named from the fact its waters are green from the high algae content. [wiki/Green:0] Green For the ancient Egyptians, g…
- turn:  -> `semantic_search({"query": "Green singer actor \"Green\" biography", "top_k": 10})`
  - result: [wiki/Green Lake (Chisago City, Minnesota):0] Green Lake (Chisago City, Minnesota) Green Lake is a lake in Chisago City, Minnesota, United States. This lake is sometimes also referred to as "Big Green Lake" because it is connected to Little Green Lake by a channel. Green Lake was named from the fact its waters are green from the high algae content. [wiki/Green:0] Green For the ancient Egyptians, g…
- turn:  -> `semantic_search({"query": "\"Green\" is a performer singer songwriter", "top_k": 10})`
  - result: [wiki/List of show business families:0] List of show business families Actress / director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phyl…
- turn:  -> `semantic_search({"query": "\"Green\" is an American singer", "top_k": 10})`
  - result: [wiki/Equity Music Group:0] Equity Music Group Equity Music Group was an American country music record label founded in 2003 by singer Clint Black. The label was distributed by Koch Entertainment (now E1 Entertainment). [wiki/Grant's First Stand:0] Grant's First Stand Grant's First Stand is the debut album by American jazz guitarist Grant Green featuring performances by Green recorded and released…
- turn:  -> `semantic_search({"query": "Green (disambiguation)", "top_k": 10})`
  - result: [wiki/Green Lake (Chisago City, Minnesota):0] Green Lake (Chisago City, Minnesota) Green Lake is a lake in Chisago City, Minnesota, United States. This lake is sometimes also referred to as "Big Green Lake" because it is connected to Little Green Lake by a channel. Green Lake was named from the fact its waters are green from the high algae content. [wiki/Green:0] Green For the ancient Egyptians, g…
- turn:  -> `semantic_search({"query": "\"Green\" is an American singer songwriter born", "top_k": 10})`
  - result: [wiki/Equity Music Group:0] Equity Music Group Equity Music Group was an American country music record label founded in 2003 by singer Clint Black. The label was distributed by Koch Entertainment (now E1 Entertainment). [wiki/Ulrich Franzen:0] Ulrich Franzen Ulrich Joseph Franzen (January 15, 1921 – October 6, 2012) was a German-born American architect known for his "fortresslike" buildings and Br…
- turn:  -> `semantic_search({"query": "Green (singer)", "top_k": 10})`
  - result: [wiki/List of show business families:0] List of show business families Actress / director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phyl…
- turn:  -> `semantic_search({"query": "Green (singer)", "top_k": 10})`
  - result: [wiki/List of show business families:0] List of show business families Actress / director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phyl…
- turn:  -> `semantic_search({"query": "\"Green\" is a Canadian singer", "top_k": 10})`
  - result: [wiki/List of show business families:0] List of show business families Actress / director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phyl…
- turn:  -> `semantic_search({"query": "Green (singer)", "top_k": 10})`
  - result: [wiki/List of show business families:0] List of show business families Actress / director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phyl…
- turn:  -> `semantic_search({"query": "\"Green\" is a singer known for", "top_k": 10})`
  - result: [wiki/List of show business families:0] List of show business families Actress / director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phyl…
- turn:  -> `semantic_search({"query": "\"Green\" is an American singer and songwriter", "top_k": 10})`
  - result: [wiki/Equity Music Group:0] Equity Music Group Equity Music Group was an American country music record label founded in 2003 by singer Clint Black. The label was distributed by Koch Entertainment (now E1 Entertainment). [wiki/Grant's First Stand:0] Grant's First Stand Grant's First Stand is the debut album by American jazz guitarist Grant Green featuring performances by Green recorded and released…
- ... 17 more turns
**Segment 2** (40 turns)
- turn:  -> `semantic_search({"query": "Grant Green jazz guitarist biography born died married", "top_k": 10})`, `semantic_search({"query": "Green (musician) disambiguation list of people named Green", "top_k": 10})`
  - result: [wiki/Grant's First Stand:0] Grant's First Stand Grant's First Stand is the debut album by American jazz guitarist Grant Green featuring performances by Green recorded and released on the Blue Note label in 1961. Earlier recordings made by Green for Blue Note were released as "First Session" in 2001. [wiki/The Main Attraction (album):0] The Main Attraction (album) The Main Attraction is an album b…
  - result: [wiki/Green Lake (Chisago City, Minnesota):0] Green Lake (Chisago City, Minnesota) Green Lake is a lake in Chisago City, Minnesota, United States. This lake is sometimes also referred to as "Big Green Lake" because it is connected to Little Green Lake by a channel. Green Lake was named from the fact its waters are green from the high algae content. [wiki/Green (Steve Hillage album):0] Green (Steve…
- turn:  -> `semantic_search({"query": "\"Green\" was born in 1935 jazz guitarist from Washington D.C.", "top_k": 10})`, `semantic_search({"query": "jazz musician named Green wife married personal life", "top_k": 10})`
  - result: [wiki/The Main Attraction (album):0] The Main Attraction (album) The Main Attraction is an album by American jazz guitarist Grant Green featuring performances recorded in 1976 and released on the Kudu label. [wiki/Grant's First Stand:0] Grant's First Stand Grant's First Stand is the debut album by American jazz guitarist Grant Green featuring performances by Green recorded and released on the Blue…
  - result: [wiki/The Odd Life of Timothy Green:0] The Odd Life of Timothy Green The Odd Life of Timothy Green is a 2012 American fantasy comedy-drama film starring Jennifer Garner, Joel Edgerton, Dianne Wiest, CJ Adams, Rosemarie DeWitt, Ron Livingston, David Morse and Common, written and directed by Peter Hedges and produced by Walt Disney Pictures. Based on a concept by Ahmet Zappa, the film is about a mag…
- turn:  -> `semantic_search({"query": "Green is a singer songwriter musician known for stage name", "top_k": 10})`, `semantic_search({"query": "performer \"Green\" married to spouse wife husband personal life", "top_k": 10})`
  - result: [wiki/Green (Steve Hillage album):0] Green (Steve Hillage album) Green is the fourth studio album by British progressive rock musician Steve Hillage. Written in spring 1977 at the same time as his previous album, the funk-inflected "Motivation Radio" (1977), "Green" was originally going to be released as "The Green Album" as a companion to "The Red Album" (the originally intended name for "Motivat…
  - result: [wiki/List of show business families:0] List of show business families Actress / director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phyl…
- turn:  -> `semantic_search({"query": "Grant Green jazz guitarist 1935 1979 biography discography", "top_k": 10})`, `semantic_search({"query": "\"Green\" actor actress performer film stage name", "top_k": 10})`
  - result: [wiki/Grant's First Stand:0] Grant's First Stand Grant's First Stand is the debut album by American jazz guitarist Grant Green featuring performances by Green recorded and released on the Blue Note label in 1961. Earlier recordings made by Green for Blue Note were released as "First Session" in 2001. [wiki/The Main Attraction (album):0] The Main Attraction (album) The Main Attraction is an album b…
  - result: [wiki/Miquette Giraudy:0] Miquette Giraudy Miquette Giraudy (born 9 February 1953, Nice, France) is a keyboard player and vocalist, best known for her work in Gong and with her partner Steve Hillage. She and Hillage currently form the core of the ambient band System 7. In addition to her performances in music, she has also worked as an actress, film editor and writer. In each role, she has used di…
- turn:  -> `semantic_search({"query": "\"Green\" is a singer, songwriter and performer", "top_k": 10})`, `semantic_search({"query": "stage name Green performer musician \"born\"", "top_k": 10})`
  - result: [wiki/List of show business families:0] List of show business families Actress / director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phyl…
  - result: [wiki/Green (Steve Hillage album):0] Green (Steve Hillage album) Green is the fourth studio album by British progressive rock musician Steve Hillage. Written in spring 1977 at the same time as his previous album, the funk-inflected "Motivation Radio" (1977), "Green" was originally going to be released as "The Green Album" as a companion to "The Red Album" (the originally intended name for "Motivat…
- turn:  -> `semantic_search({"query": "\"Green\" is a singer and songwriter born in 1980s debut album", "top_k": 10})`, `semantic_search({"query": "performer known as Green music artist married", "top_k": 10})`
  - result: [wiki/Grant's First Stand:0] Grant's First Stand Grant's First Stand is the debut album by American jazz guitarist Grant Green featuring performances by Green recorded and released on the Blue Note label in 1961. Earlier recordings made by Green for Blue Note were released as "First Session" in 2001. [wiki/Green (Steve Hillage album):0] Green (Steve Hillage album) Green is the fourth studio album …
  - result: [wiki/List of show business families:0] List of show business families Actress / director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phyl…
- turn:  -> `semantic_search({"query": "Green (singer) American rapper or singer born 1980s", "top_k": 10})`, `semantic_search({"query": "\"Green\" is an English musician or British performer", "top_k": 10})`
  - result: [wiki/Equity Music Group:0] Equity Music Group Equity Music Group was an American country music record label founded in 2003 by singer Clint Black. The label was distributed by Koch Entertainment (now E1 Entertainment). [wiki/List of show business families:0] List of show business families Actress / director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is married to …
  - result: [wiki/Green (Steve Hillage album):0] Green (Steve Hillage album) Green is the fourth studio album by British progressive rock musician Steve Hillage. Written in spring 1977 at the same time as his previous album, the funk-inflected "Motivation Radio" (1977), "Green" was originally going to be released as "The Green Album" as a companion to "The Red Album" (the originally intended name for "Motivat…
- turn:  -> `semantic_search({"query": "\"Green\" is a singer and songwriter who was born in 1980s debut album", "top_k": 10})`, `semantic_search({"query": "performer known as Green music artist married", "top_k": 10})`
  - result: [wiki/Grant's First Stand:0] Grant's First Stand Grant's First Stand is the debut album by American jazz guitarist Grant Green featuring performances by Green recorded and released on the Blue Note label in 1961. Earlier recordings made by Green for Blue Note were released as "First Session" in 2001. [wiki/Green (Steve Hillage album):0] Green (Steve Hillage album) Green is the fourth studio album …
  - result: [wiki/List of show business families:0] List of show business families Actress / director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phyl…
- turn:  -> `semantic_search({"query": "\"Green\" is a singer and songwriter who was born in 1980s debut album", "top_k": 10})`, `semantic_search({"query": "performer known as Green music artist married", "top_k": 10})`
  - result: [wiki/Grant's First Stand:0] Grant's First Stand Grant's First Stand is the debut album by American jazz guitarist Grant Green featuring performances by Green recorded and released on the Blue Note label in 1961. Earlier recordings made by Green for Blue Note were released as "First Session" in 2001. [wiki/Green (Steve Hillage album):0] Green (Steve Hillage album) Green is the fourth studio album …
  - result: [wiki/List of show business families:0] List of show business families Actress / director / singer Phylicia Rashād is the older sister of performer Debbie Allen, who is married to former NBA basketball player, Norm Nixon. Phylicia Rashād is the former spouse of both Victor Willis, former lead singer of the group Village People, and former NFL football player turned sportscaster, Ahmad Rashād. Phyl…
- finish: max_turns; answer 'None'

## Observations

- **Mechanics hold.** Every setup ran (documents written, repositories copied in), every benchmark loaded on the node, all 35 runs finished by a budget or a submission, none errored, and the second pass found all 35 in the cache. Compaction fired 7 times in the longest runs and the model always complied with the compaction demand.
- **The oracle never delegated.** Zero subagent calls in 35 runs, including the 14 guided framings that ask for delegation explicitly, and the system prompt that encourages it. It also never used parallel_tool_call. The 27B prefers to search and read itself; the subagent channel of the AC items gets no teacher data under these prompts.
- **Tool habits by family.** Multi-hop and BRIGHT are search-only (30 to 81 searches per benchmark); repository and long-document tasks are shell-only (sed, grep, cat over the file or tree); math is python-only. Assistant turns carry almost no text: the model emits bare tool calls, so the trajectories are call-result chains.
- **Repository tasks do not converge.** Four of five hit the 40-turn limit with 38 to 40 shell calls and no submission; the one submission was the trivial task. The model reads file after file and never commits to an answer; it also tried git in a tree without history and once called shell with a wrong argument name.
- **Long documents.** QuALITY scored 0.60 with mostly short runs; NarrativeQA scored 0.09 with 16 to 64 turns of sequential reading over 360k-character books. NarrativeQA drives compaction well and answers badly.
- **Looping.** The longest MuSiQue run repeated near-identical searches for 71 turns; the compaction summary did not stop the loop.
- **Cost shape.** 7,232,338 prompt tokens for 62,867 generated: teacher rollouts are prefill-bound, and the prefix cache carried 75%. At 35 concurrent agents the node was far from saturated (2.6k uncached prompt tokens/s); the campaign should run 64 or more agents per GPU.

## Questions for you

1. **Delegation.** Should the guided framings require delegation (a brief that must be issued before the answer, or a tool-count rule enforced by the loop), should a share of tasks run with `shell`/`python` removed so search and delegation are the only routes, or do we accept that the 27B rarely delegates and source the subagent channel differently?
2. **Repository family.** Raise the turn budget, add a nudge near the limit to submit the best guess, or reformulate the task (issue plus a candidate list)? As is, the family yields long shell-only runs with no success signal.
3. **NarrativeQA.** Keep it as a compaction driver (everything is teacher for AC items) despite near-zero scores, or trade its share for QuALITY?
4. **Report.** What should this page add or drop: per-turn latency, prompt-length histograms, answer-versus-gold diffs, the framing text itself, cache-hit ratios per benchmark?
5. **Interfaces.** The suite exposes `load_suite`, `configs_for_tasks` and per-task `env_setups` specs; the campaign runner takes a `ProbeSpec`. Say what you want shaped differently before M4 builds the campaign on them.
