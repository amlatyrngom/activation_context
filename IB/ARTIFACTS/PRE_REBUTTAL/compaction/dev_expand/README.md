# Dev-206: a second compaction dev set (2026-11-17)

Why: dev-75 × 3 seeds (225 runs per arm) resolves about ±4.4 points (95 % CI of the S1_lcrec100 ac_4000 − full difference,
[−12.44, −3.56]), too coarse for 2–3-point decisions (user, 2026-11-17: use more dev data).

What: every seed-0 run of `pc_pool9_h6_eval` (evaluation side, evaluation order 600–1009) that qualifies for the eval-200 cut
(`kit/cut_prefixes.py`, 20k–50k tokens) and is outside the 322 tasks of `kit/exclude_h7.json`: **206 prefixes, 206 tasks**
(py 113, java 56, kt 37; original run solved 138; 91 repositories, 174 of the 206 runs from repositories in eval-200, as dev-75's
63/75). Prefix tokens: min 20,000, median 20,340, max 23,689; all 206 match the inventory table
(`IB/TMP/PRE_REBUTTAL/inventory_compaction/derived/runs_pc_pool9_h6_eval.jsonl`).

Rejected first idea: the 56 qualifying frozen-100 prefixes; 53 of them are eval-200 (test) tasks.

Consequences:
- **Training exclusion from now on:** `exclude_h7_dev206.json` (528 task ids = the 322 + dev206), task level, all seeds.
  `make_dev206.py` writes both files.
- Runs F and M (`S1_tv_kl_both_c20_s012`, `S1_tv_kl_both_c162024_s012`) trained on dev206 tasks; they are compared on dev-75 only.
- Agent evals: `agent_eval/dev_chain.sh NODE prefixes_dev9_206 ...` (DEVSET/DEVN in `dev75_node.sh`); tags `d206*`.

Files: prefixes on S3 `compaction/pc_inputs/prefixes_dev9_206/` (local `IB/TMP/PRE_REBUTTAL/dev_expand/prefixes_dev9_206/`);
`dev206_tasks.json`; `exclude_h7_dev206.json` (also S3 `compaction/pc_inputs/kit/`).
