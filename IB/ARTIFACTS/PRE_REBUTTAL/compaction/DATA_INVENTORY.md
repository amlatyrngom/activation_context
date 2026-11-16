# Compaction rerun (§3.3): data inventory (step C0 of CLEAN_RECIPES_PLAN §3)

Date: 2026-11-15. Scope: everything the clean-recipe rerun needs (9B agent runs on training tasks, the old H7/H8
material, the splits and exclusions, dev-75 and eval-200, and the evaluation tooling). Every number below was read from a
file or computed by streaming one; "unknown" means I did not read it.

Short names used below:
- **Pool run**: one untuned Qwen3.5-9B agent run that solves a task from scratch.
- **Cut**: the eval-200 rule from `kit/cut_prefixes.py`. A run qualifies when it is between 20,000 and 50,000 tokens
  (head included), did not hit the trajectory cap, and did not end in `error` or `task_unavailable`. It is cut at the
  first step boundary at or past 20,000 tokens that is not an assistant turn, and something must remain after the cut.
- **Original continuation**: the pool run's own steps after the cut.

Paths starting with `node_sync/` or `kit/` are under `IB/TMP/PAPER_COMPACTION/`. `S3:` means
`s3://general-bucket-aln/activation-context-data/`.

## 1. The 9B pool runs

Every pool was written by `kit/run_pool.py`, called by `kit/pool_node.sh`, `pool9c_node.sh` or `pool9h6_node.sh`. The
settings are the same for all of them:
- Model: untuned `Qwen/Qwen3.5-9B` in vLLM 0.28 (the logs say so; the rows call the model "oracle").
- Seed 0, one run per task.
- Sampling: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, 8,192 tokens per turn, thinking on.
- 50,000-token trajectory cap, engine context 72k, 150 turns, no compaction tool.

These sampling settings are the same as in the eval runs (`pc_r9d50_s2_full`, `pc_dev9_s0_full`), which I checked.

**Format.** Each line of `rollouts.jsonl` is one run. It holds:
- `agent_config`, which contains `dataset_task.task_id`, `gold_answer` and `metadata`
  (`side`, `repo`, `language`, `benchmark`).
- `prompt_messages` (system + user), `prompt_token_ids`, and `trajectory`. Each step of `trajectory` has `role`,
  `messages` and `token_ids`; assistant steps also have the sampled `logprobs` and `think_end`.
- `score` (1.0 = solved), `finish_reason`, `answer`, `seed` and `config_key`.

Total tokens = `len(prompt_token_ids)` plus the `token_ids` of every step. Every run in every pool has both token ids and
messages.

| Pool (cache id) | Side, task range | Runs | Solved | ≥ 20k tokens | Qualify for the 20k cut | …of which solved | Local copy | S3 copy |
|---|---|---|---|---|---|---|---|---|
| `pc_pool9_h6_trA` | training 0–369 | 370 | 234 | 237 | 173 | 113 | `node_sync/h6_pool/` and `node_sync/pc-b_live/ROLLOUTS/` | yes |
| `pc_pool9_h6_trB` | training 370–899 | 530 | 324 | 351 | 270 | 156 | `node_sync/h6_pool/` and `node_sync/pc-a_live/ROLLOUTS/` | yes |
| `pc_pool9_h6_trC1` | training 900–1343 | 444 | 296 | 286 | 215 | 151 | `node_sync/pc-b_live/ROLLOUTS/` | yes |
| `pc_pool9_h8_D1` | training 1344–1787 | 444 | 281 | 278 | 204 | 131 | **none** | `S3:node_final_20261018/pc-b/SYNC/ROLLOUTS/pc_pool9_h8_D1/` (356 MB) |
| `pc_pool9_h8_D2` | training 1788–2231 | 444 | 304 | 272 | 199 | 148 | **none** | `S3:node_final_20261018/pc-b/SYNC/ROLLOUTS/pc_pool9_h8_D2/` (343 MB) |
| `pc_pool9_h6_trC4` | training 2232–2675 | 444 | 289 | 280 | 200 | 143 | `node_sync/pc-a_live/ROLLOUTS/` | yes |
| **Training side, total** | **all 2,676 training tasks** | **2,676** | **1,728** | **1,704** | **1,261** | **842** | | |
| `pc_pool9_eval_v3` ("9B pool v3") | evaluation, order 0–599 | 600 | 382 | 389 | 278 | 180 | `node_sync/pc-a_live/ROLLOUTS/` | yes |
| `pc_pool9_h6_eval` | evaluation, order 600–1009 | 410 | 252 | 281 | 206 | 138 | `node_sync/h6_pool/` and `node_sync/pc-b_live/ROLLOUTS/` | yes |

Notes on the table:
- **S3 copies.** All pools are on S3 under `S3:node_final_20261018/pc-{a,b}/SYNC/ROLLOUTS/`. The ones held locally are
  also under `S3:host_ib_tmp_20261109/PAPER_COMPACTION/node_sync/`.
- **`pc_pool9_eval_v3` is the shards joined.** It has exactly the byte size of its four 150-task shards `_q0` … `_q3`
  added together (493,027,254 bytes), so it is their concatenation. eval-200 and dev-75 are cut from it.
- **Qualifying training-side runs (1,261).**
  - Outcome: 1,080 submitted and 181 `max_tool_errors`; 842 solved.
  - Languages: py 893, java 328, kt 40. Eval-200 is py 89 / java 77 / kt 34, so Kotlin is thin on the training side.
  - Prefix: median 20,377 tokens, maximum 27,703.
  - Original continuation: median 11,920 tokens (6,526 of them sampled assistant tokens), so a median run has 6,526
    target tokens for a KL loss.
  - Totals over the 1,261 runs: 15.85 M continuation tokens, of which 8.99 M are assistant tokens.
- **Capped runs.** Another 427 training-side runs pass 20k but end at the 50k cap (unsolved; about 30k tokens of
  continuation after the cut). The eval rule leaves them out. The H7/H8 prefix cutter kept them. The H7/H8 prefix indexes
  confirm the count: 1,688 training-side runs have a valid boundary past 20k (= 1,261 + 427).
- **Same cutter as eval-200.** My implementation of the cut reproduces the eval-200 index exactly. On
  `pc_pool9_eval_v3` it gives 278 qualifying runs (180 solved), and for every eval-200 and dev-75 entry the prefix and
  continuation token counts match the index.

Per-run tables (one JSON line per run: task, side, repo, solved, finish reason, total tokens, whether it qualifies, prefix
and continuation tokens at the cut, membership in the exclusions): `IB/TMP/PRE_REBUTTAL/inventory_compaction/derived/runs_<pool>.jsonl`.
Scripts: `IB/TMP/PRE_REBUTTAL/inventory_compaction/scripts/`.

## 2. The old H5–H8 training material (what an item is)

The pipeline has two stages:
1. A **material** line (one JSON per kept continuation), written by `export_*_material.py`.
2. A **trainer packet** (`prepare_agent_continuation*.py`), which picks the sizes and the loss.

A material line holds:
- `task_id`, `repo`, `language`, `side`, `solved`, `finish_reason`.
- `head_messages` (system + task) and `prefix_messages`.
- `continuation_messages` and `target_token_ids` (one list per assistant turn of the continuation).
- `tools`, `chat_template_kwargs`, `prefix_tokens`, and `tokens` (`head`, `prefix`, `continuation_written`,
  `continuation_context`, `continuation_turns`).
- `teacher`, which is a summary of the ORIGINAL pool run (outcome, total and continuation tokens, cut level); it does not
  hold the original messages.

No size or ratio is stored in the material line. The packet builder draws r per item: rows = round(prefix_tokens × r),
log-uniform in [1/20, 1/5]. The builder already has a `--loss kl` option ("top-k KL to the starting reader reading the
teacher view") and a `--held-repos` panel. Its limits:
- reader history 72,000 tokens;
- encoder input 65,536 positions;
- at most 4,096 rows, so at r = 1/5 a prefix over 20,480 tokens is clamped.

| Material | Script | Continuation used as target | Where | Content |
|---|---|---|---|---|
| H5: `pool9_v3_h5.jsonl` | `export_pool_material.py` | **original** pool continuation, solved runs only | `node_sync/material/` (11 MB) and S3 mirror | 156 runs, all evaluation side (eval_v3 minus the 322 excluded tasks); cut at min(20k, f × length) with f in [0.5, 0.8], median prefix 8.1k |
| H6: `h6_mix.jsonl`, `h6_long.jsonl` | `export_pool_material.py` | **original**, solved only | `node_sync/material/` | mix: 866 lines (483 training + 383 evaluation side); long: 336 lines (221 + 115), all cut at 20k |
| H7: `h7_material_pc{a,b}.jsonl` | `export_h7_material.py` via `h7_export_node.sh` | **re-sampled**: 3 new full-prefix continuations by the untuned 9B per prefix (`run_resume_k.py`); keep the shortest solved + up to 2 within 1.3× | `S3:node_final_20261018/pc-a/pc/h7_material/` (595 MB), `…/pc-b/pc/h7_material/` (332 MB) | pca 1,811 lines / 2,560 items; pcb 1,034 / 1,466 |
| **H8a**: `h8a_material_pc{a,b}.jsonl` | `export_h7_material.py` via `h8_export_node.sh` (snapshots of prefixes with all 3 samples) | **re-sampled**, as H7 | `S3:node_final_20261018/pc-a/pc/h8_material/h8a_material_pca.jsonl` (758 MB), `…/pc-b/pc/h8_material/h8a_material_pcb.jsonl` (866 MB) | 4,676 prefixes read, 3,393 kept (1,588 + 1,805), 4,957 lines, 7,034 items in the summaries (the digest says the packet had 6,957); lines by side: 3,990 training + 967 evaluation |
| H8 full: `h8full_material_pc{a,b}.jsonl` | same | **re-sampled** | same folders (903 MB + 1,169 MB) | 4,276 kept prefixes, 6,274 lines, 8,870 items; 1,158 evaluation-side lines |

How H7/H8 got their prefixes and continuations:
- **Prefixes:** `kit/cut_prefixes_multi2.py` cut the pool runs at four levels, 16k / 20k / 24k / 28k (c16 … c28), and
  kept capped runs. The prefix folders are `S3:node_final_20261018/pc-{a,b}/pc/prefixes_{h7b_trA,h7b_trB,h7b_C1,h7b_C4,h8_D1,h8_D2,h7b_eval,h7b_v3}_c{16,20,24,28}/`.
  I copied their `index.json` files to `IB/TMP/PRE_REBUTTAL/inventory_compaction/s3_small/`.
- **Continuations:** the 3-sample caches are `S3:node_final_20261018/pc-{a,b}/SYNC/ROLLOUTS/h7k3_prefixes_*_full/rollouts.jsonl`.
  There are 24 caches, about 0.14–0.95 GB each. Some cut levels were never run, for example trA c28, D1 c24/c28,
  D2 c20/c28 and v3 c24/c28. The snapshots of these caches (`h8_material/snap_*`, about 31 GB) are also on S3.

**Is the original continuation kept?** In H7/H8: **no.** The target is a new continuation re-sampled by the 9B, and the
original run is only summarised in `teacher`. The original continuation can always be rebuilt from the pool cache: the
prefix index gives `prefix_steps`, and the rest of that run's `trajectory` is the original continuation. In H5/H6 the
target IS the original continuation, but only for solved runs.

**The trained H8a trainer packet** (built by `prepare_agent_continuation_h8.py`) and its `recipe.json` were not found
on S3 or locally. PAPER_CANON says they are on CSAIL only. The trained pairs are on S3 under
`S3:node_final_20261018/pc-{a,b}/pc/pairs/` (`h8a_final`, `h8full_final`, `h5_final`, …). The new recipe does not need
any of these.

## 3. Splits and exclusion

**`IB/TMP/PAPER_COMPACTION/split.json`** (seed 20261013, `kit/build_split.py`) is a **repository-level** split of LCA
bug localization (py, java, kt), with 3,686 tasks:
- **evaluation side:** 1,010 tasks from 189 repositories (py 488, java 357, kt 165; the 99 official test tasks are all
  here). These tasks have an `evaluation_order`.
- **training side:** 2,676 tasks from 615 repositories (py 1,883, java 715, kt 78).
- No repository is on both sides.

**`split_train_shuffled.json`** has the same tasks and sides plus a `training_order` (the shuffled order the training
pools ran in).

**`kit/exclude_h7.json`** holds 322 task ids: eval-200 ∪ dev-75 ∪ frozen-100. Its rule says "task level only".
**All 322 are evaluation-side tasks.** The frozen-100 set comes from the 27B pool `pc_pool_eval_v1`; the 9B pool v3 ran
the same tasks.

Checks I ran:
- **The six training-side pools are clean.** They hit 0 of the 322 excluded ids and contain 0 runs from any eval-200
  repository.
- **The old H5–H8 material was not repository-clean.** It also trained on evaluation-side tasks outside the 322 (from
  `pc_pool9_eval_v3` and `pc_pool9_h6_eval`). Of the 206 qualifying `h6_eval` runs, 174 come from repositories that are
  in eval-200. H8a material has 967 evaluation-side lines. This is allowed by the old task-level rule (the user's choice
  of Oct 14, "domain specialization"), but a clean rerun should not do it.

## 4. Held-out prefixes and their original continuations

| Set | Index | Source runs | Prefixes | Original pool run solved | Prefix tokens (median, range) | Original continuation (median, range) | Original continuation available? | Extra full-prefix continuations |
|---|---|---|---|---|---|---|---|---|
| **dev-75** (`prefixes_dev9_75`) | `node_sync/prefixes_dev9_75/index.json` (+ 75 prefix files); `S3:node_final_20261018/pc-{a,b}/pc/prefixes_dev9_75/` | `pc_pool9_eval_v3`, seed 0 | 75 (py 36, java 26, kt 13) | 47 (59 submitted, 16 `max_tool_errors`) | 20,379 (20,007–26,180) | 9,891 (303–25,558); 5,810 assistant tokens | **Yes, all 75**: steps after `prefix_steps` in `pc_pool9_eval_v3/rollouts.jsonl` (token counts match the index) | `node_sync/dev9_results/pc_dev9_s{0,1}_full/` (75 rows each, untuned 9B with full prefix, 50k settings); also s0/s1 summary, AC (H0, H5) runs |
| eval-200 (`prefixes_eval9c_200`) | `node_sync/pc-{a,b}_live/pc/prefixes_eval9c_200/index.json`; `S3:pre_rebuttal_20261109/compaction/pc_inputs/prefixes_eval9c_200/` | `pc_pool9_eval_v3`, seed 0 | 200 (py 89, java 77, kt 34) | 132 (171 submitted, 29 `max_tool_errors`) | 20,439.5 (20,007–25,983) | 11,359.5 (224–29,690); 6,596 assistant tokens | **Yes, all 200** (same source) | `pc_r9d50_s{0..4}_full` (5 seeds) in `node_sync/pc-{a,b}_live/ROLLOUTS/` |

What the dev-75 index says about itself:
- Its source is "prefixes_eval9c_200 left_over minus the 27B frozen 100", that is, the qualifying v3 runs beyond the
  first 200 in evaluation order (278 = 200 + 75 + 3 frozen).
- 63 of the 75 dev prefixes come from repositories that are also in eval-200. Dev-75 shares no task with eval-200.
- Dev-75 was used earlier to promote H5, so it is a seen development set, not a fresh one. It is still held out from
  every training set.

The prefix files hold only the prefix (`prompt_token_ids` + `steps`). To compute the proxy, rebuild the continuation from
the pool cache by task id.

## 5. Evaluation tooling (for C4 dev and C5 test)

The whole kit lives in **`IB/TMP/PAPER_COMPACTION/kit/`**. It is git-ignored (under `IB/TMP`) and mirrored at
`S3:host_ib_tmp_20261109/PAPER_COMPACTION/kit/` and `S3:pre_rebuttal_20261109/compaction/pc_inputs/kit/`.
- **`run_resume.py`** is the entry point. It takes `--split split.json --prefixes <folder> --variants …`, with variants
  `full`, `none`, `summary_write`, `summary`, `recent_N` and `ac_<rows>`.
  - The resume logic is in `resume_program.py` (`ResumeFromPrefix`).
  - It writes `ROLLOUTS/<tag>_<variant>/rollouts.jsonl` and can resume an interrupted run.
  - AC arms need `--ac-checkpoint PAIR/ac --reader-checkpoint PAIR/reader --rows-only`; serving is Input Embeddings only.
  - `--uncapped-summary` selects `uncapped_summary.py`.
- **`resume_node.sh`** (+ `node_env.sh`) is the node wrapper.
- **The paper's 50k settings** (`run9d_unit.sh`, `run9h_unit.sh`):
  `--budget 50000 --trajectory-cap 90000 --max-model-len 100000 --max-duration 6000 --seed {2,3,4}`.
  - Use 96 (`run9d`) or 40 agents per GPU for text arms and 32 for AC arms.
  - Variant tags `ac_4000` / `ac_2000` mean 1/5 / 1/10 on a 20k prefix.
- **`run_resume_k.py`** runs k samples per prefix (the H7/H8 data collection).
- **Reporting:**
  - `view_budget.py` (replay at a smaller budget);
  - `heldout_quantiles.py` (difficulty groups);
  - `two_seed_report.py`;
  - `IB/ARTIFACTS/PAPER_COMPACTION/scripts/make_fig_compaction_v4.py` (figure).
- **Latest working launcher:** `IB/ARTIFACTS/PRE_REBUTTAL/compaction/cmp_node.sh`. It ran the uncapped-summary arm
  (seeds 2–4) on a SkyPilot spot node. It pulls inputs from S3 (`pc_inputs/`, `repo_cache.tar.gz` of 5.4 GB) and mirrors
  the rollout cache to `S3:pre_rebuttal_20261109/compaction/ROLLOUTS/` every 10 minutes. Results:
  `pc_r9d50u_s{2,3,4}_{summary,summary_write}`.
- **Gap for the new recipe:** the AC arms of `run_resume.py` serve input rows through vLLM. A Hidden Residuals pair
  needs the vLLM residual-write work described in CLEAN_RECIPES_PLAN §7 before C4/C5 can run.

## 6. What is missing or not local

- `pc_pool9_h8_D1` and `pc_pool9_h8_D2` (888 training tasks): S3 only. I read them by streaming them from S3.
- All H7/H8 material, the h7k3 continuation caches and the H7/H8 prefix folders: S3 only (`node_final_20261018`).
- The H8a trainer packet and `recipe.json`: CSAIL only (per PAPER_CANON; not found on S3).
- Exact count of training-side re-sampled continuations in the h7k3 caches: unknown. The summaries do not split by side,
  and I did not stream the 15 GB of caches.
- Nothing needed for the recommendation below is cluster-only.

## 7. Recommendation

**Stage 1 training material: the original continuations of the six training-side pools.** These are `pc_pool9_h6_trA`,
`_trB`, `_trC1`, `_trC4`, `pc_pool9_h8_D1` and `_D2`: 2,676 runs covering every training-side task once.
1. **Cut them with the eval-200 rule** (`cut_prefixes.py` logic: window at 20k, run under 50k, not capped). This gives
   **1,261 prefixes** (842 solved). Median prefix 20.4k, median continuation 11.9k tokens (6.5k assistant target tokens).
   No eval-side repository or task appears in them.
2. **Keep unsolved runs.** The loss is KL to the teacher on the teacher's own sample, so success does not matter. Keeping them gives
   1,261 prefixes instead of 842 and avoids a success filter that §3.1 does not have.
3. **Optional:** add the 427 capped runs (continuation up to the 50k cap), for **1,688 prefixes** at the 20k cut.
4. **Optionally add more cut points from the same runs.** At 16k / 24k (`prefixes_h7b_*` / `prefixes_h8_*` indexes:
   1,914 / 1,475 training-side prefixes, capped runs included) these give roughly 5,000 prefixes in all. They are
   correlated, but each one comes from a real 9B run.
5. **How many updates that is.** With one item per prefix and r drawn from {1/5, 1/10}:
   - **≈1,260–1,690 items ≈ 160–210 updates** at batch 8;
   - about 5,000 items (about 630 updates) with the extra cuts.
6. **Build path.** `export_pool_material.py` with `--window-min-total 20000 --no-solved-only` builds the original-
   continuation material. Its `trajectory_cap` filter must be relaxed if the capped runs are wanted. Then
   `prepare_agent_continuation.py --loss kl --sides training` builds the packet.
7. **Do not use** the H7/H8 re-sampled "shortest-correct" material as stage-1 targets. They are selected samples, not the
   run's own continuation, and part of them is evaluation-side.

**Proxy held-out set: dev-75.** Use the original continuations from `pc_pool9_eval_v3`, which exist for all 75 prefixes.
Compute KL/NLL for three views:
- frozen reader + AC rows;
- untuned no-context;
- untuned full prefix.

`pc_dev9_s{0,1}_full` give two more teacher-policy continuations per dev prefix if a lower-variance estimate is wanted.
Dev-75 is evaluation-side and shares repositories with eval-200 (63/75), so it measures transfer to unseen repositories. **Correction (2026-11-16):** that is only because the first packets trained on the training side only; the rule (ROOT_CANON) is item-level, so evaluation-side tasks outside the 322 train too, and dev-75 then measures same-repository specialization.
For a second check that does not touch the evaluation side, the packet builder's `--held-repos` panel (a few
training-side repositories held out) works. Eval-200 stays untouched; its original continuations (all 200) exist if a
final likelihood readout is ever wanted.
