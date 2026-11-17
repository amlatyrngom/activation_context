# Compaction agent evaluation on dev-75 (pre-rebuttal C4) — Hidden Residuals pairs in vLLM

Started 2026-11-16 ~20:00Z. Dev-75 only (`IB/TMP/PAPER_COMPACTION/node_sync/prefixes_dev9_75/`, S3 copy
`pre_rebuttal_20261109/compaction/pc_inputs/prefixes_dev9_75/`); eval-200 (test) is not fetched or run.

## 1. Wiring (code: `compaction/agent_eval/`, commit 795cda1)

- **`run_resume_res.py`** wraps the kit's `run_resume.py` (kit = `IB/TMP/PAPER_COMPACTION/kit/`, identical to the S3 copy
  `pc_inputs/kit/`). Text arms run unchanged. For the `ac_<rows>` arms without `--rows-only`:
  - the AC model is built from the fields saved in `ac_modules.pt` (run_qa.pair_setup's rule) and the saved model ids must be
    [Qwen/Qwen3.5-2B, Qwen/Qwen3.5-9B];
  - it checks and logs: checkpoint intervention layers = loaded layers = the engine's `hf_overrides` layers, the engine
    architecture `Qwen3_5ResForConditionalGeneration`, and `inputs_embeds_size = (1 + k) · 4096`; a failed check stops the run;
  - a reader adapter whose LoRA B matrices are all zero (frozen reader) is not sent to the engine, so the AC arm reads with the
    untuned 9B exactly like the text arms;
  - a monitor line every 60 s: host memory, process-tree RSS, engine requests/min, request payload (sent, and what the stock
    full-length tensor would have been), prompt / cached / output tokens.
  - `--rows-only` keeps the kit's old behavior (rows alone).
- **Payload fix (main code, small):** a deep model's request tensor now stops after its last AC row
  (`Agent.prepare_request`), and `activation.vllm_residuals.allow_short_prompt_embeds` (run by the plugin in every vLLM
  process) relaxes vLLM's one length-equality check. vLLM's runner already copies rows only where the tensor has them and embeds
  every token-id position itself; the prefix-cache hash of a block past the end hashes an empty slice (same on every turn).
  Why: the stock rule sends the whole prompt as a (1 + 8) · 4096-wide bf16 tensor every turn, 73,728 bytes per token, so late in
  a 50k-token continuation one request would be about 4 GB (and is SHA-256 hashed by the engine core every turn). With the cut,
  the payload stays at the first prompt's size (head + rows + instructions). Rows-only models are untouched.
- **`first_turn_check.py`**: first prompt of the ac arm built as run_resume builds it; vLLM greedy with the cut tensor vs the
  full-length tensor (must agree) vs deltas zeroed (scale); then HF with the intervention hooks on the same ids/rows/deltas.
- **`dev75_node.sh`**: SkyPilot spot node job (pattern of `cmp_node.sh` / `pool_seed_node.sh`), paper 50k settings, rollout
  caches mirrored to S3 `compaction/agent_eval_dev75/ROLLOUTS/` every 10 min, DONE markers.
- **`report.py`**: per-arm tables and paired comparisons (bootstrap over prefixes, McNemar exact).

## 2. Existing dev-75 arms (inventory)

All 75 rows each, untuned Qwen3.5-9B reader, 50k settings (budget 50,000, trajectory cap 90,000, max duration 6,000 s), run in
October (local `IB/TMP/PAPER_COMPACTION/node_sync/dev9_results/`; S3 `host_ib_tmp_20261109/PAPER_COMPACTION/node_sync/` and
`node_final_20261018/pc-{a,b}/SYNC/ROLLOUTS/`):

| cache | arm | seeds |
|---|---|---|
| `pc_dev9_s{0,1}_full` | full prefix | 0, 1 |
| `pc_dev9_s{0,1}_summary_write` / `_summary` | summary, **capped** at 500 words (the paper's old arm) | 0, 1 |
| `pc_dev9_s{0,1}_ac_{4000,2000}` | old AC pair H0, rows only | 0, 1 |
| `pc_dev9h5_s{0,1}_ac_{4000,2000}` | old AC pair H5, rows only | 0, 1 |
| `pc_dev9h5x2_s{0,1}_ac_{4000,2000}` | old H5x2 | 0, 1 |
| `pc_dev9h3_s0_ac_{4000,2000}` | old H3 | 0 |

Missing on dev-75: **no-context (`none`) at any seed**, **uncapped summary at any seed**, and every arm at seeds 2–4.

## 3. Smoke and agents per GPU (one spot g7e.4xlarge: 1 × RTX PRO 6000, 16 vCPU, 124 GiB RAM; ap-south-1)

**First-turn check** (`first_turn_check.py`, `S1_lcrec100_kl_both_c20` final, ac_4000, first 2 dev prefixes, greedy 48 tokens,
fp8 KV; S3 `agent_eval_dev75/checks/first_turn_S1_lcrec100_kl_both_c20_r4000.json`). The engine got the residual architecture,
layers [3, 7, …, 31] and inputs_embeds_size 36,864.

| prefix | prompt tokens | request tensor (cut / full length) | cut vs full length | vLLM vs HF hooks: max / mean \|Δ logprob\| (top-20), top-1, common greedy prefix | deltas zeroed: max / mean \|Δ\| |
|---|---|---|---|---|---|
| embulk__embulk__761__757 | 6,340 | 464.9 / 467.4 MB | identical (0.0, 48/48) | 0.250 / 0.075, agree, 24/48 | 5.22 / 1.97 |
| pinpoint-apm__pinpoint__2078__2077 | 4,814 | 352.4 / 354.9 MB | identical (0.0, 48/48) | 0.135 / 0.059, agree, 40/48 | 3.20 / 1.99 |

The vLLM–HF difference is the size the vLLM residual README measured (0.37–0.50 max, bf16 KV); the deltas move the log-probs by 3–5.

**End to end** (4 prefixes, seed 2, 50k settings, tag `cdv1smoke_s2`): ac_4000 4/4 solved, mean continuation 11,812 tokens;
ac_2000 3/4, 10,182 tokens; all 8 runs submitted, no errors.

**Load** (ac_4000, dev prefixes 5–75 = 71 runs, seed 2, throwaway tags `cdv1load{16,32,48}_s2`; monitor lines every 60 s):

| agents / GPU | peak host memory used | peak process-tree RSS | steady output tokens/s | wall time | solved | max_tool_errors / no_tool_call |
|---|---|---|---|---|---|---|
| 16 | 44.5 GiB | 34.9 GiB | ~600 | ~19 min | 48 / 71 | 4 / 5 |
| 32 | 57.8 GiB | 51.3 GiB | ~950 | ~16 min | 51 / 71 | 2 / 3 |
| 48 | 70.5 GiB | 59.3 GiB | ~1,100–1,240 | ~16 min | 42 / 71 | 7 / 6 |

Request payload with the cut (all three loads): mean 377–411 MB, max 487 MB for the compaction part alone (1.2–1.8 GB when the
harness adds a later tool-output AC part, which happens in a few runs as in the paper's runs). The stock full-length tensor
would have averaged 1.20–1.23 GB per request with a maximum of 4.05 GB, about 3× the memory at every load.
**Choice: 32 agents per GPU on g7e.4xlarge** (the paper's AC setting); 48 is no faster end to end and shows more tool-error
and no-tool-call finishes (the canon's sandbox limit is about 40 agents). At 32, a g7e.2xlarge (64 GB) would be tight.

## 4. Dev-75 results: `S1_lcrec100_kl_both_c20` final, seeds 2, 3, 4 (2026-11-16 22:15Z – 11-17 00:20Z)

Runs: tags `cdv1lcrec100_s{2,3,4}` (ac_4000, ac_2000; node cdv1, 32 agents) and `cdv1base_s{2,3,4}` (full, none,
summary_write, summary; seed 2 on cdv2, seeds 3–4 on cdv3, 40 agents). cdv2 was spot-reclaimed during seed 3 before its
first mirror, so seed 3 was rerun from scratch on cdv3 under the same tag (nothing had been saved). All 18 caches have 75 rows
and a DONE marker on S3 `compaction/agent_eval_dev75/ROLLOUTS/`; summaries in `.../PAPER_COMPACTION/`; tables below are
`report.py` output, also in S3 `.../agent_eval_dev75/report/` (local `IB/TMP/PRE_REBUTTAL/cmp_agent_eval/report/`).

**Seeds.** 2, 3, 4 for every arm, all run fresh with today's code: the October dev-75 baselines are seeds 0/1, came from older
code, and have no `none` and no uncapped summary at any seed, so pairing the new AC arms with them would mix code versions.
The October seed 0/1 caches are kept as a cross-check (end of this section).

Per arm (225 runs = 75 prefixes × 3 seeds; solved = score ≥ 1; continuation tokens = sampled + tool tokens after the prefix,
capped at the 50,000 budget, the whole budget when no answer was submitted):

| arm | seeds | runs | solved % | answered % | mean continuation tokens | errors | finish reasons |
|---|---|---|---|---|---|---|---|
| full | 2,3,4 | 225 | 75.56 | 93.8 | 15795 | 0 | {"submitted": 211, "max_tool_errors": 11, "continuation_budget": 3} |
| none | 2,3,4 | 225 | 58.67 | 74.7 | 32676 | 0 | {"submitted": 168, "max_tool_errors": 30, "continuation_budget": 27} |
| summary (uncapped) | 2,3,4 | 225 | 70.22 | 88.4 | 18304 | 0 | {"submitted": 199, "max_tool_errors": 16, "continuation_budget": 10} |
| ac_4000 | 2,3,4 | 225 | 67.56 | 87.1 | 17677 | 0 | {"no_tool_call": 10, "submitted": 196, "max_tool_errors": 15, "continuation_budget": 4} |
| ac_2000 | 2,3,4 | 225 | 59.56 | 80.0 | 21030 | 0 | {"no_tool_call": 23, "submitted": 180, "max_tool_errors": 13, "continuation_budget": 9} |

Solved % per seed (2 / 3 / 4): full 77.33 / 78.67 / 70.67; none 57.33 / 58.67 / 60.00; summary 73.33 / 73.33 / 64.00;
ac_4000 62.67 / 65.33 / 74.67; ac_2000 58.67 / 62.67 / 57.33. The summary-writing turn (summary_write, not counted in the
summary arm's continuation tokens) averaged 460.6 / 599.3 / 527.6 tokens.

Paired comparisons (pairs = (prefix, seed); Δ = A − B; 95 % bootstrap over prefixes, 10,000 draws; McNemar exact on the
discordant pairs, treating the 3 seeds of a prefix as independent pairs):

| A − B | pairs | prefixes | solved A % | solved B % | Δ solved (pts) [95 % CI] | A✓B✗ / A✗B✓ | McNemar p | Δ tokens [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| ac_4000 − full | 225 | 75 | 67.56 | 75.56 | -8.00 [-12.44, -3.56] | 11 / 29 | 0.006 | +1882 [+47, +3697] |
| ac_4000 − none | 225 | 75 | 67.56 | 58.67 | +8.89 [+1.78, +16.44] | 43 / 23 | 0.019 | -14998 [-17980, -12056] |
| ac_4000 − summary | 225 | 75 | 67.56 | 70.22 | -2.67 [-7.56, +2.22] | 15 / 21 | 0.405 | -627 [-2652, +1353] |
| ac_2000 − full | 225 | 75 | 59.56 | 75.56 | -16.00 [-22.67, -9.78] | 11 / 47 | 0.000 | +5234 [+3112, +7425] |
| ac_2000 − none | 225 | 75 | 59.56 | 58.67 | +0.89 [-7.56, +9.33] | 38 / 36 | 0.908 | -11646 [-14630, -8642] |
| ac_2000 − summary | 225 | 75 | 59.56 | 70.22 | -10.67 [-16.44, -4.44] | 12 / 36 | 0.001 | +2725 [-9, +5404] |
| ac_4000 − ac_2000 | 225 | 75 | 67.56 | 59.56 | +8.00 [+2.22, +13.78] | 39 / 21 | 0.027 | -3352 [-5953, -889] |
| summary − none | 225 | 75 | 70.22 | 58.67 | +11.56 [+3.11, +20.00] | 45 / 19 | 0.002 | -14371 [-17259, -11560] |
| full − summary | 225 | 75 | 75.56 | 70.22 | +5.33 [+0.44, +10.67] | 22 / 10 | 0.050 | -2509 [-4423, -718] |

Reading: ac_4000 sits between none and full: above none (+8.89 pts, p = 0.019), below full (−8.00 pts, p = 0.006), and not
distinguishable from the uncapped summary (−2.67 pts, CI [−7.56, +2.22]) at about the same continuation tokens. ac_2000 is at
the no-context level (+0.89 pts vs none) and clearly below summary and full. Seed spread is large (ac_4000 62.67–74.67 %,
full 70.67–78.67 %), so single-seed comparisons on dev-75 are not reliable.

**October cross-check (seeds 0, 1, rows-only pairs, `report.py` on `IB/TMP/PAPER_COMPACTION/node_sync/dev9_results/`):**

| arm | seeds | runs | solved % | answered % | mean continuation tokens |
|---|---|---|---|---|---|
| full (Oct) | 0,1 | 150 | 74.67 | 94.0 | 15515 |
| summary capped at 500 words (Oct) | 0,1 | 150 | 73.33 | 90.7 | 16984 |
| H5 rows-only ac_4000 (Oct) | 0,1 | 150 | 74.67 | 94.7 | 15830 |
| H5 rows-only ac_2000 (Oct) | 0,1 | 150 | 74.00 | 92.7 | 16860 |

H5 − full (Oct): ac_4000 +0.00 pts [−7.33, +7.33], ac_2000 −0.67 pts [−7.33, +6.00]. Today's full arm (75.56 %) matches
October's (74.67 %), so the harness has not drifted; the new res pair is below what the old rows-only H5 pair reached on the
same prefixes.

## 5. Tool-call format failures in the AC arms (model, not engine)

The res-pair AC arms end runs with `no_tool_call` (10 and 23 of 225) and show a specific broken call that no text arm and no
October AC arm shows: after a line of prose the reader writes `[/shell]` / `[/path]` and then the rest of a call
(`<parameter=...>…</function></tool_call>`) without the opening `<tool_call>\n<function=…>`, so the parser finds no call.
Counted over every assistant turn (`IB/TMP/PRE_REBUTTAL/cmp_agent_eval/report/format_failures.md`):

| cache | assistant turns | turns without a parsed call | of which `\n[…]` garble | runs with ≥ 1 garble (solved) | first turn garbled |
|---|---|---|---|---|---|
| full / none / summary, seeds 2–4 (9 caches) | 1,819–3,497 each | 0–1 each | 0 | 0 | 0 |
| ac_4000 s2 / s3 / s4 | 1901 / 2038 / 1918 | 28 / 48 / 29 | 25 / 40 / 26 | 11 (5) / 16 (7) / 16 (10) | 8 / 13 / 14 |
| ac_2000 s2 / s3 / s4 | 2228 / 2056 / 1762 | 49 / 28 / 61 | 40 / 25 / 52 | 19 (9) / 12 (4) / 20 (8) | 13 / 8 / 10 |
| H5 rows-only ac_4000 / ac_2000, Oct s0, s1 | 2070–2413 each | 0–1 each | 0 | 0 | 0 |

So 43 of 225 ac_4000 runs and 51 of 225 ac_2000 runs hit the garble at least once (22 and 21 of those were still solved).

**Engine or model?** `garble_check.py` (`dev75_node.sh garble`, a node job like `check`; S3 `.../agent_eval_dev75/checks/garble_cdv1lcrec100_s2_ac_{2000,4000}.json`)
rebuilt the first prompt of 6 (ac_2000) + 4 (ac_4000) garbled first turns from seed 2 (the rebuilt token ids equal the recorded
ones in all 10), teacher-forced the turn vLLM sampled through the HF reader with the intervention hooks, and compared per-token
log-probs with the ones vLLM recorded:

- whole turn: mean |Δ log-prob| 0.012–0.039 per token, max 0.11–0.37 (turn totals e.g. −172.18 vLLM vs −172.38 HF);
- at the garble token `[/`: vLLM −0.142 … −2.139, HF −0.143 … −1.927; HF's top choices there are `[/` and `</think>`
  (`[/` is HF's top-1 in 7 of 10, `</think>` in the other 3), and HF gives `<tool_call>` only −4.34 … −6.95.

The HF reference with the same rows and deltas produces the broken call with the same probability, so it is the pair's
behavior (the frozen reader reading these deltas), not the vLLM residual path. The high `</think>` mass at the same point
suggests the reader believes it is still inside a thinking block when it starts the call.

## 6. Differences from the paper's settings

- Same: dev-75 prefixes, untuned Qwen3.5-9B reader for text arms, budget 50,000, trajectory cap 90,000, max model length
  100,000, max duration 6,000 s, fp8 KV cache (the recommended setting), kit `run_resume.py` harness.
- AC pair: a Hidden Residuals `res` pair (rows + per-layer deltas at layers 3, 7, …, 31) served through the vLLM residual path,
  reader adapter all zeros so the AC arm reads with the untuned 9B; the paper's AC arm used a rows-only pair through the kit's
  `--rows-only` path.
- Summary arm is uncapped (`--uncapped-summary`); the paper's / October's was capped at 500 words.
- Seeds 2, 3, 4 (paper / October: 0, 1).
- Hardware: spot g7e.4xlarge (1 × RTX PRO 6000); 32 agents / GPU for AC arms, 40 for text arms.
- As in the paper-era runs, the harness occasionally adds a later tool-output AC part in AC arms.
- Dev-75 has been used for model selection; these are dev numbers, not test numbers. Eval-200 was not run.

## 7. Not done

- The larger run `S1_lcrec_kl_both_c20_s012` final was not staged under `compaction/stage1/runs/S1_lcrec_kl_both_c20_s012/`
  when these runs finished (checked 2026-11-17 ~00:20Z), so its AC arms were not run; `compaction/agent_eval/ac_chain.sh <node>
  S1_lcrec_kl_both_c20_s012 <new short tag>` runs them (needs one spot node).
- Both GPU nodes (cdv1, cdv3) were torn down at ~00:20Z.

## 8. Where the `[/` comes from: teacher-forced probe (2026-11-17, coordinator follow-up)

Hypothesis given: the encoder's view escapes reasoning delimiters to `[Reasoning]` / `[/Reasoning]`
(`ac_model_utils._training_template_message`, `reasoning_as_text`), and the long-context reconstruction stage (target = that
rendering) taught the pair to make the reader write `[/`.

**(a) Is `[/Reasoning]` in the encoder input?** Yes (host CPU check; `render_with_part_sentinels` with the Qwen3.5-2B tokenizer).
Every assistant message of the 75 dev-75 prefix files carries its reasoning inline, ending in `</think>` (1954 / 1954; no
`<think>`, no reasoning field). The served AC part renders one `[/Reasoning]` per assistant message (first 10 prefixes: 258 / 258).
The same holds for the stage-1 KL items (`s1kl_both_c20_v1`, first 200: 5334 / 5334) and the lcrec items (`lcrec_v1`, first 200:
4986 / 4986). The lcrec reader targets hold `[/Reasoning]` 4986 times and `</think>` never. The KL targets (original
continuations) hold `</think>` 4373 times and `[/Reasoning]` never.

**(b) Teacher-forced log-probs.** `agent_eval/probe_think.py` (run by `dev75_node.sh probe` in the stage-1 code bundle's
environment, node cdv4) scores the stage-1 dev-75 proxy panel with the trainer's own eval forward. That panel is the 75 original
continuations: r5 = 4,000 rows, r10 = 2,000 rows, plus the pair-independent `none` view and the teacher's in-context view (`full`).
At the 1478 positions whose gold is `</think>` and the 1478 whose gold is `<tool_call>`, it records the reader's log-probs
(S3 `.../agent_eval_dev75/probe/`, tables in `IB/TMP/PRE_REBUTTAL/cmp_agent_eval/report/probe_*`):

| pair | view | mean lp `</think>` | mean lp `[/` | % `[/` > `</think>` | % `[/` top-1 | mean lp `<tool_call>` | mean lp `[/` (call) | % `[/` > `<tool_call>` |
|---|---|---|---|---|---|---|---|---|
| frozen 9B | full (teacher's in-context view) | -0.085 | -7.83 | 2.8 | 2.8 | -0.110 | -23.80 | 0.0 |
| frozen 9B | none | -0.036 | -18.90 | 0.0 | 0.0 | -0.030 | -24.15 | 0.0 |
| R_res_rq9fr (start) | r5 / r10 | -0.069 / -0.054 | -18.01 / -18.30 | 0.0 / 0.0 | 0.0 / 0.0 | -0.045 / -0.039 | -23.73 / -23.98 | 0 / 0 |
| S1_res_kl_both_c20 (KL, no lcrec) | r5 / r10 | -0.091 / -0.079 | -5.23 / -5.83 | 2.8 / 2.2 | 2.7 / 2.2 | -0.100 / -0.094 | -20.40 / -21.17 | 0 / 0 |
| R_cmp_lcrec (recon only) | r5 / r10 | -0.038 / -0.036 | -18.36 / -18.52 | 0.0 / 0.0 | 0.0 / 0.0 | -0.031 / -0.025 | -23.59 / -23.83 | 0 / 0 |
| S1_lcrec100_kl_both_c20 (lcrec, then KL) | r5 / r10 | -0.084 / -0.088 | -5.11 / -5.52 | 1.8 / 2.8 | 1.6 / 2.7 | -0.105 / -0.102 | -19.63 / -20.27 | 0 / 0 |

By position of the `</think>` in the continuation (mean lp `[/`, % above `</think>`):

| pair, view | 1st (n = 75) | 2nd–5th (279) | 6th–20th (738) | 21st+ (386) |
|---|---|---|---|---|
| full (teacher) | -1.43, 44.0 % | -5.13, 2.9 % | -7.76, 0.1 % | -11.16, 0.0 % |
| R_res_rq9fr r5 | -13.31, 0 % | -17.61, 0 % | -18.27, 0 % | -18.73, 0 % |
| R_cmp_lcrec r5 | -14.00, 0 % | -17.74, 0 % | -18.70, 0 % | -19.00, 0 % |
| S1_res_kl_both_c20 r5 / r10 | -0.90, 54.7 % / -1.02, 44.0 % | -3.74, 0.4 % / -4.17, 0.0 % | -5.32 / -5.96, 0 % | -6.98 / -7.71, 0 % |
| S1_lcrec100_kl_both_c20 r5 / r10 | -1.04, 30.7 % / -1.00, 45.3 % | -3.74, 1.1 % / -3.95, 2.5 % | -5.32 / -5.74, 0 % | -6.50 / -7.11, 0 % |

**Reading.**
- The hypothesis is not supported as stated. The recon-only pair puts no more mass on `[/` than the start pair or `none`.
- Both KL stage-1 pairs do, with or without lcrec. The mass is largest at the first continuation turn, matching the agent arms'
  first-turn garbles.
- It matches the KL teacher. `prepare_training_history` renders every message before the first target with
  `reasoning_as_text=True`, so the teacher's own in-context view of head + prefix shows each turn closed by `[/Reasoning]`. The
  frozen 9B copies that at the first continuation turn (`[/` above `</think>` in 44 % of prefixes), and KL distils it into the
  pair.
- The agent eval's `full` arm is not this view: at serving it reads the recorded ids with a real `</think>` and had 0 garbles.
- `[/` never outranks `<tool_call>` at call positions; the break is at the end of thinking, one token earlier.
- Not checked: the stored teacher targets themselves (CSAIL/ORCD-made `.pt` files, not loaded) and the larger run.
