# Paper canon: evidence map for the submitted paper

"Activation Context: Harness/Model Co-Design at the Architectural Level" (MLSys 2027 research track,
submitted 2026-10-30; reviews 2027-01-18, responses due 2027-01-21, notifications 2027-02-28). Paper =
`paper/` submodule at `16a0260` (body `main.tex` + `00`–`06`, separate `appendix.tex`); build with
`IB/ARTIFACTS/PAPER_WRITING/build_pdf.sh` (body 10 pages, references from page 11; appendix 3 pages).
Mapped 2026-11-09 from the paper, the figure scripts and their data. When a number here and a file
disagree, the file wins; fix this page.

Path conventions: `PW/` = `IB/ARTIFACTS/PAPER_WRITING/`, `AS/` = `PW/appendix_sources/` (its `README.md` maps every appendix row to a file). Local `IB/TMP/...` paths are given as-is; they are archived (2026-11-09) at `s3://general-bucket-aln/activation-context-data/host_ib_tmp_20261109/`. Storage flags:
- **[TMP]**: only under `IB/TMP`.
- **[S3]**: only on S3. `STATE.md` writes "S3 rq_runs/…" without a bucket; the bucket prefix is `s3://general-bucket-aln/activation-context-data/` (checked: `rq_runs/` is there).
- **[REMOTE]**: only on CSAIL, ORCD or EC2 nodes.
- **[HARDCODED]**: the figure script types the numbers in rather than reading a file.

Figure PDF provenance was checked by byte size and MediaBox against the script outputs. Matches by md5: fig 1, 2, 3, 5 and 7. Figs 4 and 6 have the same byte size and differ only in CreationDate.

---

## Section 1 (Introduction) and Abstract: headline numbers

Every headline number is repeated in Section 3; the source is listed there.

| Claim | Value(s) | Source (Section) |
|---|---|---|
| Retrieval: ACs top-100 vs text top-100 | 52.9 vs 49.2 %, 1.3k vs 12.0k tokens | 3.1 |
| Retrieval: matches text top-200 with "about 19×" / abstract "9–19×" fewer tokens | 24,351 / 1,306 = 18.6; 24,351 / 2,559 = 9.5 | 3.1 |
| Corpus: matches full context 79.5 % after 0.5M tokens; 81.5 % after 1M; AM 75.3 %; Cartridges 69 % at 100M | 79.5 / 81.5 / 75.25 / 69.0 | 3.2 |
| Compaction: AC 5× P75 62.0 vs summary 55.3 vs full 62.7; 5.5× faster | 7.42 s / 1.34 s | 3.3 |
| Multi-agent: difficulty ≥ 9: AC 85.7 vs results-only 82.4 vs summaries 83.5 | | 3.4 |

**Fig. 1** (`fig:overview`, `figures/fig_overview.pdf`) is a conceptual diagram. Script: `PW/fig1_mocks/make_fig1_r4.py`, output `fig1_r4.pdf` (md5 match). No data.

## Section 2 (Approach)

- **Fig. 2** (`fig:messages`, `fig_messages_v3.pdf`) is a message-format diagram from `PW/fig_messages/make_fig_messages_v3.py` (md5 match). No data.
- **Listing 1** (`lst:engine`) and **Table 1** (`tab:writemodes`) are static LaTeX. No data.
- Two claims in Section 2 matter for the next step:
  - "Residual heads start at zero."
  - The no-context penalty $\lambda\mathcal{R}$ is applied to a fraction $\rho$ of each batch. The appendix gives p = 0.25 (`AS/00_shared/trainer_config_defaults.excerpt.txt` l.111-112, 229-231).

---

## Section 3.1 Retrieval Q/A (HotpotQA): Fig. 3 (`fig:evalRetrievalQA`, `fig_retrieval_v3.pdf`)

**Script.** `PW/figs_compact_1028/make_fig_retrieval_v3c.py` writes `figs_compact_1028/fig_retrieval_v3.pdf` (md5 match). It is a compact copy of `PW/fig3_v3_final/make_fig_retrieval_v3.py`. It imports these modules:
- `PW/mocks_fig3_v3/make_fig3_v3.py` and `make_fig3_s2_variants.py` (the `l2t` panel; select series `SEL`);
- `PW/fig3_ideal_final/make_fig_retrieval_ideal.py` (`IDEAL`);
- `PW/mocks_r2/make_r2.py` (`RQ`).

All panel values are [HARDCODED] in these modules.

**Panel (a): accuracy vs mean reader input tokens** (eval500, mean of 3 sampled passes unless noted)

| Arm | k=20 | k=100 | k=200 | Tokens (k: 20/100/200) |
|---|---|---|---|---|
| Text, BM25 (untuned 9B, thinking off) | 43.93 | 49.2 | 52.67 | 2,396 / 11,996 / 24,351 |
| AC, BM25 (Arm G) | (49.4, 1 pass; not plotted) | **52.87** | **54.07** | 331 / 1,306 / 2,559 |
| Text, ideal recall | 66.33 | 64.07 | 63.80 | 2,402 / 11,996 / 24,350 |
| AC, ideal recall | 67.13 | 63.67 | 62.87 | 331 / 1,306 / 2,559 (BM25 counts reused) |
| Select → text (untuned 2B picks ≤ 2/20; `sel{K}r`) | – | 39.13 | 42.67 | – / 1,272 / 2,461 |
| Gold reference / No context | 67.73 / 23.07 | | | gold 330, none 82 prompt tokens |

- BM25 both-gold recall@20/100/200 is 52.0 / 65.2 / 72.0 (Section 3.1 text). Source: reranker probe `IB/TMP/RERANKER_PROBE/` [TMP] (STATE l.1101).
- Unplotted point: text top-400 = 53.47 at 49,407 tokens (`AS/01_retrieval/text_baselines_ec2_text_nothink_e500_top400__eval500.scores.json`).

**Panel (b): TTFT at k=200, one request, medians over 100 requests (RTX PRO 6000)** [HARDCODED in `V.BARS`]

| Arm | TTFT (ms) | Stages (ms) | TPOT (ms/tok) |
|---|---|---|---|
| Text | 1,545.5 | prefill 1,545.5 | 17.7 |
| Select → text | 1,002.2 | selector prefill 562.2, selector decode 261.6, reader prefill 140.8 | 16.2 |
| AC | 808.0 | encode 665.7, prefill 143.6 | 16.2 |

- Ratios: 1.9× faster than text (1,545.5 / 808) and 1.2× faster than select.
- Source: sky job 33 "main9b replay" (STATE l.1184), raw at `S3 rq_runs/20261027_systems_timing/v3/` [S3].
- `IB/TMP/PAPER_SYSTEMS/results.json` [TMP] holds the earlier job-30 numbers instead: text 1,543.1, select 1,213.7 (`decode_select_max`), AC 807.2. **The plotted select bar (1,002.2) is not in any local file.**
- Timing used **untrained, merged** weights (setup B in `IB/TMP/PAPER_SYSTEMS/README.md`).

**Data files**
- Frontier: `IB/ARTIFACTS/PAPER_CORPUS_RETRIEVAL_QA/figs/results.json` (`eval500`: text, ours, gold, none, tokens). Token counts come from `figs/tok_counts.py`.
- Text, gold and no-context scores: `AS/01_retrieval/text_baselines_ec2_text_nothink__eval500.scores.json`.
- Select scores: `AS/01_retrieval/pr_select__eval500.scores.json`. It includes **unreported variants**:
  - 9B self-select `sel200r9` = 47.73 at 680 tokens;
  - uncapped 2B `sel200u` = 49.13 at 9,551 tokens;
  - summarise `sum200` = 30.87.
- Per-item AC outputs [S3]:
  - `S3 rq_runs/20261021/evals_ec2/trackA_armG/` (`out/e500_10x20_armG_mean3.jsonl`). STATE l.1045-1048 has the pass-0 and passes-1–2 split, e.g. 5×20 pass 0 = 52.4.
  - Ideal recall: `S3 rq_runs/20261027_ideal_recall/`.
  - Select: `S3 rq_runs/20261027_select_baseline/`.
- Question set: `IB/TMP/PAPER_RETRIEVAL_QA/m1/eval500.questions.jsonl` [TMP]. It has 500 questions (424 bridge, 76 comparison), drawn as `val[:500]` after a `Random(2)` shuffle.

**Models and checkpoints**
- Reader: Qwen3.5-9B + LoRA r128. Encoder: Qwen3.5-2B + LoRA r128. The 9B reader's LoRA is trained; the text arms use the untuned 9B.
- Write mode: Input rows plus residuals at layers 3, 7, …, 31 (`intervention_frequency -1`), mixer 2. Ratio 0.1; rows between 2 and 4,096; k/20 encoder calls of 20 passages each, question first.
- AC checkpoint is **Arm G `real_gold` epoch_005**: `S3 rq_runs/20261021/trackA_real/armG_final/` [S3]. The local config is only from the starting pair: `AS/01_retrieval/ec2_fresh_rq_pair_ac__ac_modules.pt.config.json`.
- Phase chain (STATE l.969–1027; **log only** in AS):
  1. `lc3i`: reconstruction at 1/10, 8,300 updates.
  2. `dense_pow1_tgt48_256_lc3i`: 4,000 updates, 2e-5 / 2e-5 cosine.
  3. Run A `sftd_dense_r10b`: SFT, 2,500 updates, 5e-5 / 2e-5 constant + 250-update cooldown.
  4. Track A `tA_s1`: KL to same-context references, about 2,250 updates.
  5. `tA_s2`: KL to same-context references (1–5 calls × 20).
  6. **Arm G**: KL toward the base 9B reading gold passages only. Top-32 log-probs, 4,000 items, **500 updates**, 8 per update, encoder 5e-5 / reader 2e-5, warm-up 20, then cooldown over the last 100 updates; λ = 1.0. Student input = BM25 **top-400 as 20 × 20** (`AS/01_retrieval/armG__real_gold__real_gold.run.json`).
- The appendix lists only the last stage's 500 updates.

---

## Section 3.2 Latent Corpus Compression (LongHealth): Fig. 4 (`fig:evalCorpusCompression`, `fig_corpus_compression.pdf`)

**Script.** `PW/figs_compact_1028/make_r2_compact.py` → `corpus("callout")` → `r2_callout_corpus.pdf`. Values are [HARDCODED] as `CC`, `CC_REF`, `CC_X`, with x = updates × 1,037 / 1e6.

**Data.** `IB/ARTIFACTS/PAPER_AM/results.json` and `PAPER_AM/evals/E_*.json`. All 400 questions, 1 sampled pass per point.

| Updates (training tokens/context, M) | 0 (0) | 500 (0.52) | 1000 (1.04) | 1500 (1.56) | 2221 (2.30) | Run / eval files |
|---|---|---|---|---|---|---|
| AC KV Non-Uniform | 75.75 | 79.5 | 81.5 | 81.25 | 81.75 | `G2n_all20_lr1e3`; `E_G2n_u{500,1000,1500}.json`, `E_G2n_all20_lr1e3_final.json` |
| AC KV Uniform | 55.75 | 72.0 | 74.5 | 75.0 | 75.75 | `U2n_all20_lr1e3`; `E_U2n_*.json` |
| AC Input + Residuals | 36.25 | 40.5 | 47.75 | 53.5 | 58.0 | `A_deepS_all20_lr1e3` / final `R_deepS` (CSAIL H200); `E_deepS_*`, `E_rows_u0.json` |
| AC Input | 36.25 | 44.25 | 42.75 | 48.0 | 46.75 | `A_rows_all20_lr1e3`; `E_rows_*.json` |

References (1 pass):

| Reference | Value | Note |
|---|---|---|
| AM Non-Uniform (AM's own code, R4 `t0.1_fastest`) | 75.25 | |
| Long context (full 60k) | 79.5 | |
| Gold (right patient only) | 83.75 | |
| Cartridges | 69.0 | **Read off AM paper Fig. 3** at 1/10, upper bound taken; placed at 100M tokens |

- Also in `results.json` but not plotted:
  - no context 36;
  - AM Uniform (AM's code) 59;
  - KV NU final, **3-pass mean 81.08** (the figure plots pass 0, 81.75), CI95 [77.58, 84.5];
  - Input-only continuation: 51.75 at 3,221 updates and 52.25 at 4,221.
- Checkpoints and tables:
  - Run on EC2 nodes `cqa-ec2r` and `cqa-ec2b`, now gone (`/root/am_v1/...`).
  - Backups at `s3://general-bucket-aln/activation-context-data/am_lh5_20261019/` [S3].
  - `teacher_v1_all.pt` was on EC2 only [REMOTE/S3].
  - No local AM log or JSON exists for the long-context, gold or no-context baselines (AS README; **log only**, STATE l.849, 857, 867, 888).
- Hyperparameters (appendix Table 3; `AS/02_corpus/G2n_all20_lr1e3.run.json`):
  - Qwen3-4B-Instruct-2507, frozen. Encoder-free All-Layer KVs at 0.1, initialised from AM's non-uniform fit (HighestAttnKeys + NNLS + LSQ).
  - KL to the one-patient view, top-32.
  - 17,764 self-study and multiple-choice pairs; LR 1e-3, warm-up 25, cosine, **no clip**, 8 per update, 2,221 updates, seed 0.
  - Eval: temperature 0.7, top-p 0.8, top-k 20, max_new 2,048.
- Not in the appendix:
  - Input: closed-form selected rows (`/root/am_v1/ablate/init_sel_rows.pt`).
  - Input + Residuals: the same rows plus zero deltas at layers 3, 7, …, 35 with `delta_scale` (`init_sel_deep_scaled.pt`; STATE l.876-880).
  - The LR 1e-3 was picked by a 125-update probe on the rows arm (STATE l.877).

---

## Section 3.3 Agent Trajectory Compaction (LCA Bug Localization): Fig. 5 (`fig:evalCompaction`, `fig_compaction.pdf`)

**Script.** `PW/figs_compact_1028/make_r2_compact.py` → `compaction("callout")` → `r2_callout_compaction.pdf` (md5 match). Values are [HARDCODED] as `CP` and `CP_TIME`. The upstream generator is `IB/ARTIFACTS/PAPER_COMPACTION/scripts/make_fig_compaction_v4.py --style paper --ratios 1/5,1/10 --no-trunc --upto P75 --kind bar`. It reads:
- `IB/TMP/PAPER_COMPACTION/heldout_quantiles_cache.pkl` [TMP];
- `IB/TMP/PAPER_COMPACTION/node_sync/pc-b_live/pc/prefixes_eval9c_200/index.json` [TMP];
- `figs/frozen100_h0_fig.json`.

Its output is `IB/ARTIFACTS/PAPER_COMPACTION/figs/fig_compaction_mvp.json`.

**Panels (a) and (b): % solved and mean continuation tokens (k), eval-200, seeds 2–4 averaged per task.** Groups: Mean n = 200, P50 n = 100, P75 n = 50. Difficulty = the **original untuned-9B run's tokens after the cut**; the caption says "full context reference".

| Arm | Solved Mean / P50 / P75 | Tokens (k) Mean / P50 / P75 |
|---|---|---|
| Full context | 71.67 / 63.0 / 62.67 | 17.29 / 23.83 / 28.07 |
| No context (not described in the text) | 63.17 / 49.33 / 50.0 | 32.71 / 38.55 / 39.51 |
| Text compaction (9B summary) | 68.67 / 57.0 / 55.33 | 21.17 / 29.30 / 34.36 |
| AC 5× (H8a, ~4,000 rows, `ac_4000`) | 71.33 / 61.67 / 62.0 | 17.08 / 24.14 / 28.49 |
| AC 10× (H8a, ~2,000 rows, `ac_2000`) | 71.0 / 59.67 / 60.0 | 18.31 / 25.47 / 29.07 |

**Panel (c): compaction time**

| Arm | Plotted (mean [CI]) | Exact | Runs |
|---|---|---|---|
| Text | 7.4 s [6.9, 7.9] | 7.4207 s (`compact_seconds`, last-turn seconds of the summary write) | 30 |
| AC | 1.3 s [1.2, 1.5] | 1.3382 s (`ac_encode_seconds`) | 90 |

- Source: `AS/03_compaction/frozen100_h0_fig.json`, tag `pc_timing_v1`. Measured on the **frozen-100 set (27B-written prefixes), arm A's pair, 30k budget**; the paper does not say so.
- Rollouts [TMP]: `IB/TMP/PAPER_COMPACTION/node_sync/{pc-a_live,pc-b_live}/ROLLOUTS/`:
  - `pc_r9d50_s{2,3,4}_{full,none,summary,summary_write}`;
  - `pc_r9h8a_s{2,3,4}_ac_{1000,2000,4000}`;
  - timing `pc_timing_v1_*`.
- Models and checkpoints:
  - Agent: Qwen3.5-9B, thinking on, temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, 8,192 tokens per turn, 150 turns, 50k continuation budget.
  - Encoder: Qwen3.5-2B + LoRA r128; reader LoRA r128; Input Embeddings (rows only).
  - **H8a** = `pairs/h8a_final` on **CSAIL** [REMOTE]; pc-a and pc-b nodes also hold a copy. Configs come from its start pair `h5_final` (`AS/03_compaction/h5_final_*`).
- Lineage:
  1. Reconstruction pair (**log only**).
  2. 1,000 updates of long-context reconstruction.
  3. arm A: 300 runs, 300 updates.
  4. H0: + 608 runs.
  5. H5: + 156 9B-written runs.
  6. H8a: RFT on 3,393 kept prefixes, best-of-3 shortest-correct continuations of the untuned 9B with full prefix; 6,957 items, **870 updates**, 2e-5 / 2e-5, warm-up 50, cosine.
  - The 870 updates are **log only** (STATE l.697, 717). The H8a `recipe.json` is on CSAIL only.
- Test set: the first 200 eval tasks that the untuned agent finished within 20k–50k tokens. Prefix cut at ≥ 20,000 tokens (20.0k–26.0k, mean 20.7k). Index: `AS/03_compaction/prefixes_eval9c_200__index.json`.
- Not reported: AC 1/20 (P75 52.7) and full H8 (`IB/ARTIFACTS/PAPER_COMPACTION/RESULTS_SO_FAR.md`).

---

## Section 3.4 Multi-Agent Synthesis (DeepMath-103K): Fig. 6 (`fig:evalMultiAgent`, `fig_multiagent.pdf`)

**Script.** `PW/figs_compact_1028/make_r2_compact.py` → `multiagent_r3("wide_dec", wide=True, decimals=True)` → `r3_wide_dec_multiagent.pdf`. Values are [HARDCODED] `MA` and `MA_CEIL`, copied from `paper/figures/tab_tqa_mvp.tex`. That table is not `\input` in the paper.

**Data.**
- Table: `IB/ARTIFACTS/PAPER_TRAJ_QA/tables/tables_unified_tb_sumres.txt`, "separate" layout, plus `tqa_unified_sumres.json`.
- Producer: `PAPER_TRAJ_QA/scripts/tqa_unified_table_sumres.py --sync-root <SYNC mirror>`. The SYNC mirror is on the pc nodes and CSAIL (`/data/scratch/ngom/activation/paper_traj_qa_v1/.../SYNC`) [REMOTE].

| Arm (table name) | 6–8 (n=272, 3 synth runs) | 8.5 (n=226, 5 runs) | 9–9.5 (n=74, 5 runs) | All (n=572) |
|---|---|---|---|---|
| Single Solver | 87.5 | 75.1 | 72.5 | 80.7 |
| Majority Vote (random tie-break, expected value) | 90.1 | 81.0 | 76.1 | 84.7 |
| Results-Only (`results`) | 92.4 | 81.2 | 82.4 | 86.7 |
| Results + Summary (`summary`) | 92.6 | 79.6 | 83.5 | 86.3 |
| Results + LoRA (`M1 no rows`) | 93.3 | 81.8 | 83.2 | 87.4 |
| Results + AC (`M1`) | 91.5 | 81.8 | **85.7** | 86.9 |
| Any of 3 (ceiling) | 94.1 | 87.6 | 90.5 | 91.1 |

- Synthesis gap (text): 6.6 / 12.5 / 18 (= ceiling − single).
- Paired sign tests are in the table file. Examples:
  - M1 vs results at ≥ 9: +22 / −10, p 0.050.
  - M1 vs M1-no-rows at 6–8: +4 / −18, p 0.004.
- The "merged" layout (n = 246 and 82 at 8.5 and ≥ 9) gives different numbers and is not used.
- Test sets:
  - `tqa_dmext_solve_v1`: 300 problems ≥ 6; the 272 below 8.5 form the 6–8 bucket.
  - `tqa_dmhard`: 300 problems ≥ 8.5 (226 at 8.5, 72 at 9, 2 at 9.5).
  - Samples: `AS/04_multiagent/{dmext,dmhard}__sample.json`. Training overlap 0 (`draws_check.json`).
- Rollout locations:
  - Solvers: dmhard `IB/TMP/PAPER_TRAJ_QA/node_sync/dmhard/rollouts.jsonl` [TMP, 290 MB]. dmext solver rollouts are **not local**.
  - AC and no-rows synthesis reads: partly local in `IB/TMP/PAPER_TRAJ_QA/m1/v2/reads/`:
    - dmh: `ac_m1_s1`, `s3`; `norows_m1_s0`, `s2`, `s3`;
    - dmx: `ac_m1_s1`; `norows_m1_s0`, `s2`.
    - The rest are [REMOTE].
  - Results-only `synth1call.json` for dmhard and dmext: **missing locally** (AS README). One summary run: `AS/04_multiagent/sumres_dmh_s0__synth1call.json`.
- Models and checkpoints:
  - Solvers: 3 × Qwen3.5-9B, thinking on, temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5; 8,192 tokens per turn, 100 turns, 40k per trajectory.
  - **M1 = pair `tqa_m1_v2`** (local copy `IB/TMP/PAPER_TRAJ_QA/round2/pairs/tqa_m1_v2/` [TMP, 2.4 GB]). Encoder 2B LoRA r128, reader 9B LoRA r128, ratio 0.1.
  - Init pair: `tqa_r2_b` (arm B: 1,234 updates on the `r2_bu` union, 9,976 examples).
  - M1 packet `m1v2_b`: 4,209 items over 786 tasks (DeepMath 3,804 + **MuSiQue 405 replay**), **527 updates**, 2e-5 / 2e-5, warm-up 50, cosine, 1 epoch. **Log only**: STATE l.716-717, 732. `recipe.json` is missing.
  - Targets: Qwen3.8-27B (NVFP4) syntheses, kept when correct (`keep_report.json`).
  - The **reconstruction phase is "missing"**: the appendix states it, but nothing backs it.

---

## Section 3.5 Training and Ablations (controlled HotpotQA): Fig. 7 (`fig:trainingablations`, `fig_sec45_v4.pdf`)

**Script.** `PW/sec45_v4/make_sec45_v4.py`, `variants("v2", suffix="_paper_compact")` with `V2_H = 1.6`, `TTFT_XMAX = 225` → `sec45_v4/figs/fig7_v2_paper_compact.pdf` (md5 match).
- Panels (a)–(c) read `PW/sec45_v4/data/<run>/{eval/*.summary.json, sync/<run>.held.jsonl}`. These are small copies from `s3://general-bucket-aln/activation-context-data/sec45_20261026/runs/<run>/`. Checkpoints are [S3]; some are local in `IB/TMP/PAPER_SEC45/runs/` [TMP]: R_* plus `S_kv_v4fr`, `S_res_v4fr`, `T_*_v4fr`.
- Panel (d) reads `/workspace/IB/TMP/PAPER_SYSTEMS/results.json`. That is a hardcoded container path; locally it is `IB/TMP/PAPER_SYSTEMS/results.json` [TMP].
- Text references:
  - Text (untuned) = 53.0 [HARDCODED];
  - Text + reader LoRA = last point of `data/T_text_lora/T_text_lora.dev200.scores.json` `acc_lenient`.

**Setup.** Reader: Qwen3-4B-Instruct-2507 + LoRA r128. Encoder: a 4B copy with its own LoRA r128. Reconstruction `R_*`: 2,000 × 8 FineWeb-Edu. Task: 4,000 × 8 on HotpotQA (32,000 items). LR 2e-5 / 2e-5, warm-up 50, cosine, seed 0, λ = 1.0. Eval: dev200 (top-20 BM25 with both golds injected), 1 greedy sample, max_new 128.

**Final dev200 accuracy (%)**

| Run | Input Emb. | Hidden Res. | All-Layer KVs |
|---|---|---|---|
| Full recipe `T_{m}_v4` (from `R_{m}_v4`) | 68.5 | 68.5 | 68.5 |
| From scratch `S_{m}_v4` | 21.0 | 21.5 | 67.0 |
| Reader frozen `T_{m}_v4fr` (from `R_{m}_v4fr`) | 19.5 | 62.0 | 66.0 |
| *Not in paper:* frozen from scratch `S_{m}_v4fr` | 18.5 | 22.0 | **71.5** |

- Reconstruction held-out NLL at u2000 (panel b text):
  - `R_in_v4` 0.0487;
  - `R_in_v4fr` 2.3041;
  - `R_in_v4fre4` (encoder LR 1e-4) 2.3051;
  - `R_res_v4` 0.0363; `R_res_v4fr` 0.2052;
  - `R_kv_v4` 0.00345; `R_kv_v4fr` 0.01167.
  - `T_in_v4fre4` was cancelled; it has only a u0 eval of 15.0.
- Encoder size (panel c): `T_in_e06ls` 60.0, `T_in_e17ls` 66.0, `T_in_v4` 68.5.
  - The 0.6B and 1.7B runs use Qwen3 0.6B / 1.7B with a linear skip, a least-squares projection init and **encoder LR 1e-4**, for both R and T.
  - Their R NLLs are 0.0740 and 0.0631.
- Text references:
  - Untuned 4B text top-20 = 53.0. This is a lenient re-score; the **strict** score is 27.0 with 58 % answered (`AS/05_ablations/data/untuned_text_text4b_sec45.dev200.scores.json`).
  - Gold 67.5 (strict 46.0); none 19.0.
  - Text + reader LoRA, lenient: u0 53.5, u1000 72.5, u2000 72.5, u3000 71.5, **u4000 70.5 (plotted)**.
- Panel (d), TTFT medians in ms (n = 100, fast engine, k = 20, RTX PRO 6000):

| Arm | TTFT (ms) | Weights |
|---|---|---|
| Input, 0.6B encoder | 48.4 | untrained |
| Input, 1.7B encoder | 64.7 | untrained |
| Input, 4B encoder | 114.7 | trained `S_*` pairs |
| Residuals, 4B | 116.9 | trained `S_*` pairs |
| KVs, 4B | 113.5 | trained `S_*` pairs |
| KVs, precomputed | 26.2 | trained `S_*` pairs |
| Text (+LoRA, merged) | 80.5 | |

---

## Appendix A (separate `appendix.tex`): Tables A.1–A.5

The row-to-file map is `AS/README.md`. Run IDs and the rows that only a `STATE.md` line backs ("log only") or that are missing:

| Table | Final run / checkpoint | Log-only or missing rows |
|---|---|---|
| A.1 Retrieval (`tab:hp-retrieval`) | Arm G `real_gold` epoch_005 (9B r128 / 2B r128; 500 updates; encoder 5e-5, reader 2e-5) | Reconstruction; KL stage settings (STATE l.969, 991, 996, 1001). The final `ac_modules.pt` is on S3. SFT and Track A have no `run.json`. |
| A.2 Corpus (`tab:hp-corpus`) | `G2n_all20_lr1e3` (2,221 updates, LR 1e-3, seed 0) | LC, gold and AM baselines; Cartridges. The Input / Input+Res inits are not listed. |
| A.3 Compaction (`tab:hp-compaction`) | H8a, eval set `eval9c_200`, seeds 2–4 | Reconstruction; 870 updates. Pair and recipe are on CSAIL. |
| A.4 Multi-agent (`tab:hp-multiagent`) | M1 `tqa_m1_v2` | QA phase; 527 updates. Reconstruction has **no source**. Results-only `synth1call` missing. |
| A.5 Ablations (`tab:hp-ablations`) | `{R,S,T}_{in,res,kv}_v4[fr]`, `T_in_e{06,17}ls`, `T_text_lora` | None. Adapter configs missing for the T and S v4 runs and for e06ls/e17ls (ranks are in each `run.json`). |

---

## Settings that differ between arms or are undocumented

### Retrieval (3.1)

1. **Presence penalty.** The text, gold, no-context and select arms use 1.5 (vLLM `SAMPLING_NO_THINK`: temperature 0.7, top-p 0.8, top-k 20). The AC arm uses **none** (HF generate; `AS/01_retrieval/kit_rq_eval.excerpt.txt` l.96). The decoding engines also differ: vLLM vs HF batched generate.
2. **Reader tuning.** The AC reader has a trained LoRA. The text arms use the **untuned** 9B. There is **no text + reader-LoRA baseline**. The AC was also distilled from the 9B *reading gold passages* (privileged teacher); no text arm gets equivalent training.
3. **AC training budget.** Training is spread over six stages (roughly 19k updates in total: lc3i 8,300 + dense 4,000 + Run A 2,500 + Track A ≈4,000 + Arm G 500). The appendix reports only Arm G's 500, and the schedules differ per stage.
4. **Train vs eval input size.** Arm G trained on BM25 top-400 (20 × 20 calls) and is evaluated at k = 100 / 200.
5. **Select baseline.**
   - Untuned 2B selector, greedy, 64-token cap: 163 selector length stops in `sel200r`.
   - Stronger variants exist but are unreported: 9B self-select 47.73, uncapped 2B 49.13.
6. **Passes.**
   - AC top-100 and top-200: pass 0 and passes 1–2 were run separately (different kit copies, `kit_c2` vs `kit_s20`).
   - AC top-20 and top-400: 1 pass.
   - Ideal-recall AC token counts reuse the BM25 counts.
7. **TTFT (Fig. 3b).** Untrained merged weights. The plotted select bar comes from a replay run (job 33) that is not in local `results.json`.

### Corpus (3.2)

8. **Different harnesses.** AM, long-context and gold use AM's code; AC uses ours, so the same closed-form fit scores 75.75 vs 75.25. Uniform: 55.75 ours vs 59 in AM's code.
9. **Passes.** All points are 1 sampled pass; one 3-pass final exists (81.08). The 81.5 vs 79.5 headline is 8 of 400 questions. Cartridges is read off a plot, not rerun.
10. **Undocumented inits.** The Input / Input+Res initialisation is missing from the appendix (closed-form selected rows; zero scaled deltas). Input+Res ran on CSAIL H200 (R_deepS rerun) while the others ran on EC2.
11. **No gradient clip** (other workloads clip at 1.0). LR 1e-3 vs 2e-5 elsewhere. The teacher is the gold one-patient view; the baselines are untrained by design.

### Compaction (3.3)

12. **Untuned text arms.** Summary and full context run on the untuned 9B; only the AC arm has a reader LoRA. The AC was RFT-trained on shortest-correct full-prefix continuations, and the summary arm gets no matching tuning.
    - Missing baselines: text + reader-LoRA, i.e. a summary-reading LoRA, or the same RFT on summary or truncated prefixes.
13. **Timing** is from a different set (frozen-100, 27B prefixes, 30k budget, arm A pair). Text time = the last-turn decode (≈510-token summary); AC time = one encoder pass.
14. **Caption vs code.** The caption defines difficulty by the full-context reference; the code uses the original run's post-cut tokens. The "No Context" arm is not introduced in the text.
15. **Fixed AC row counts.** 4,000 / 2,000 rows are used whatever the prefix length (20.0–26.0k), so the true ratio varies.
16. **Model selection on the test set.** H0 → H5 → H7S → H8a vs full H8 were compared on eval-200 itself (seeds 0–1 and 2–4; `RESULTS_SO_FAR.md`). The reported seeds 2–4 were also used to pick H8a.

### Multi-agent (3.4)

17. **Different code paths.**
    - AC and Results+LoRA arms: agent harness, `max_turns 1`, per-turn `max_tokens 8192`, program `max_tokens 64000`, trajectory cap 80,000.
    - Results-Only and Summary arms: `synth1call` vLLM, `max_tokens 64000`, `max_model_len 72000`.
    - Verify the effective generation cap of the AC arm.
18. **"Results + LoRA"** is the AC reader adapter used without ACs. It is not a LoRA trained on text synthesis, so there is no text-trained synthesizer baseline. The AC arm is distilled from 27B syntheses and the text arms are untuned.
19. **Write mode.** The `tqa_m1_v2` config has residual intervention heads at layers 3, 7, …, 31, but the appendix says Input Embeddings. Confirm whether rows-only was used at read time.
20. **Training data.** It includes 405 MuSiQue replay items (not in the appendix). The reconstruction claim has no source. Synthesis repeats are 3 on 6–8 and 5 on ≥ 8.5. The ≥ 9 bucket is 74 problems.

### Ablations (3.5)

21. **Generation cap and scoring.**
    - Cap: 128 new tokens for AC vs 2,048 for text.
    - Scoring: AC is strict ("Final answer:" line, HF decode); the text and text+LoRA arms are **lenient** (fallback to the last line ≤ 120 characters, vLLM). The untuned text strict score is 27.0, not 53.0.
22. **Text + LoRA.**
    - It has no no-context penalty, while the AC arms use λ = 1.0.
    - The paper reports its final 70.5, not the peak 72.5 at u1000–2000.
    - Untuned text is 53.0 in one run and 53.5 at T_text_lora u0.
23. **Encoder-size confounds.** The 0.6B and 1.7B encoders differ from the 4B in encoder LR (1e-4 vs 2e-5), init (least squares) and head (linear skip). All runs are one seed on 200 questions.
24. **Unreported best result.** `S_kv_v4fr` = 71.5 beats every reported number.
25. **TTFT pairs.** The TTFT arms use untrained (0.6B, 1.7B) or `S_*` pairs, not the `T_*_v4` checkpoints.

### Cross-workload

26. **Optimisation recipes all differ:**

| Workload | LR | Schedule | Updates |
|---|---|---|---|
| Retrieval | 5e-5 / 2e-5 | constant + cooldown | 500 |
| Compaction | 2e-5 / 2e-5 | cosine | 870 |
| Multi-agent | 2e-5 / 2e-5 | cosine | 527 |
| Corpus | 1e-3 | cosine, no clip | 2,221 |
| Ablations | 2e-5 | cosine | 2,000 + 4,000 |

27. **Evaluation protocols all differ:**

| Workload | Passes | Decoding |
|---|---|---|
| Retrieval | 3 | temperature 0.7 |
| Corpus | 1 | temperature 0.7 |
| Compaction | 3 seeds | temperature 1.0 |
| Multi-agent | 3 / 5 | temperature 1.0 |
| Ablations | 1 | greedy |

    - Hardware is mixed (H200, GH200, RTX PRO 6000) across arms, and only timing is pinned to one GPU type.
    - Hyperparameters were set by agents; the appendix says the trained baselines were "optimised to be fair", but only §3.5 has a trained text baseline.

### Also from the October log (details and line references in `IB/ARTIFACTS/HANDOFF_EC2/STATE_DIGEST_20261109.md`)

- **Stacked, ad hoc recipes.** Retrieval Arm G is SFT → KL stage 1 → KL stage 2 → gold-reference KL; stage 2 shrank from 26k to 22k items and ran 2,000 updates without a recorded reason. Compaction H7/H8 start from `h5_final` while H5/H6 start from H0. A clean single recipe per workload is the rerun target.
- **Rates that break the old "2e-5 everywhere" rule.** Retrieval encoder 5e-5; corpus 1e-3 (probed on the rows arm only and reused for KV arms; `C_G` continued at 5e-4); small-encoder arms 1e-4 with a least-squares "linear skip" head.
- **Frozen readers in the corpus arms** (`lr_reader` 0 everywhere); KV arms start from AM's closed-form cache, input arms from near no-context; input arms had not plateaued.
- **Unfinished or selection-tainted checkpoints.** The synthesis model `u245` is the update-245-of-326 snapshot (final save hit a quota) and was picked on the set it was reported on; the retrieval start pair 6459 is mid-run; the compaction final model was picked on the evaluation set.
- **Text + reader-LoRA (§3.5 only)** trained without the no-context penalty the AC runs use and was monitored on dev200, the set it is scored on; it peaked at 72.5 and the paper reports its final 70.5.
- **Scoring and caps.** Lenient vs strict answer parsing changes conclusions (TQA AC vs results-only went from p .035 to not significant; §3.5 text 53.0 lenient vs 27.0 strict). Retrieval hill-climb dev evals were capped at 4,096 tokens and cannot separate settings.
- **Small samples.** §3.5 is one seed on 200 questions (±3.5 points); compaction hard subsets are 50 tasks (±10); the multi-agent gain rests on 74 problems; the corpus lead over full context (+.0113) is inside its CI.
