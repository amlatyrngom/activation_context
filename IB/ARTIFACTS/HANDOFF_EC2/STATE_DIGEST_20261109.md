# STATE digest (2026-11-09)

What was worth keeping from the 400 KB `IB/STATE.md` log (Oct 14–27, 2026), extracted in three ranges.
Line references `(L123)` point into the verbatim archive `STATE_archive_20261109.md` next to this file.
Durable lessons and the user's standing preferences were promoted to `IB/CANON/ROOT_CANON.md`; the
figure-by-figure evidence and the fairness gaps are in `IB/CANON/PAPER_CANON.md`. S3 root throughout:
`s3://general-bucket-aln/activation-context-data/`; local `IB/TMP/X` is now archived at
`host_ib_tmp_20261109/X` under that root.

## STATE.md extract, lines 1–740

Covers the header (L1–106: condensed at Oct 14 14:08Z, plus later inserts dated Oct 19 and Oct 26–27) and the log from Oct 14 14:08Z to Oct 17 09:40Z. Dates in the log are inferred from rollovers (L118 "≈00:30Z Oct 15", L645 "Oct 17", S3 archive names). Model sizes: "9B" (solver/continuation), "27B" (oracle/writer, FP8 or NVFP4), "2B" (selection baseline). The log does not name the model family, except AM compaction: Qwen3-4B (L85). Pre-condense chronology: `IB/TMP/PAPER_COMPACTION/STATE_before_condense_20261014T1415Z.md`. Older: `IB/TMP/FIRST_TASK_AWARE_FINETUNING/STATE_before_paper_mvp_20261013.md`, git branch `paper-mvp-20261013` at 258fce6 (L4–5). The FIRST_TASK_AWARE_FINETUNING folder was later moved to S3 (see §3).

### 1. Final / headline results

**Compaction, 27B-written prefixes: frozen 100, H0 pair (w23A_final).** Columns are solved / answered / continuation tokens (L26–39).
- AC 1/5: 47/88/11.3k. AC 1/10: 47/82/12.3k. AC 1/20: 39/78/13.5k. Full prefix: 55/91/11.3k. Summary: 45/67/18.8k.
- Truncation 1/5, 1/10, 1/20: 38/55/20.9k, 30/44/23.9k, 21/32/24.7k.
- Vs summary: +2 solved (n.s.) and 6.4k–7.5k fewer tokens. Vs truncation: +17/+18 at 1/10 and 1/20 (p 0.005).
- Time to compact: summary 7.4 s vs AC encode 1.3 s. Earlier arm A frozen: 44/34/30.
- Writing files: `IB/ARTIFACTS/PAPER_COMPACTION/{RESULTS_SO_FAR.md,compaction_results_draft.tex,figs/frozen100_h0_fig.*}`; old figure `figs/frozen100_w1A_fig.*` (L19).
- Compaction dev (1/10, 2 seeds, of 100): arm A 40, H0 50, H1 43, H2 (w1KL_final) 42 (L40, L111).

**Compaction, 9B-written prefixes: eval-200 (`prefixes_eval9c_200`) at a 50k continuation budget. This became the paper's main compaction setting.**
- H0 3-seed headline, mean@3 over 200, solved / tokens (L371–375):
  - full 145.3 / 17,359; AC 1/5 141.7 / 19,363; summary 139.3 / 20,551.
  - AC − summary: +2.3 (p .69), −1,188 tokens (p .08). Full − AC: +3.7 (p .47), −2,005 tokens (p<.001).
  - At 30k: full 131.7, AC 121.7, summary 120.7.
- Earlier seed-0 and two-seed readouts (L247–250, L292–297). Seed 0's AC ≈ full did not hold on seed 1 (L296).
- H5 vs H0 at 1/10, 3 seeds: 138.3 / 19,346 vs 128.0 / 21,665, i.e. **+10.3 solved (48/30, p .05), −2,320 tokens**; at 30k +11.0 (p .01) (L477–480).
- H5 at 1/5, 3 seeds: 138.3 / 18,370 vs H0 141.7 / 19,363 (−3.3, p .32) (L462–466).
- Non-gains at 1/5, 3 seeds, 50k:
  - H3 139.7 / 20,632 vs H0 141.7 (L505).
  - H5x2 140.3 / 18,036 (L507).
  - H6 138.0 / 19,600 (p .12) (L509).
  - H6fromH5 and H6L: seed 2 only, no gain over H5 (L632).
- Held-out difficulty quantiles: ranked on seeds 0–1, read on seeds 2–4. Script `IB/TMP/PAPER_COMPACTION/kit/heldout_quantiles.py`, cache `heldout_quantiles_cache.pkl`, output `heldout_quantiles_0116.json` (L621–624, L631).
  - Top 25%: full 64.0%/20.5k, summary 57.3%/28.3k, H0 1/5 63.3%/26.2k, H5 1/5 66.0%/21.8k.
- **H7S (shortest-correct targets), 3 held-out seeds:** at 1/10, top 25% 64.7%/22.4k vs full 64.0%/20.5k vs H5 1/10 57.3%/23.4k. Paired vs H5: +7.3 pts [+1.3, +13.3] (L648). The gain at 1/20 mostly vanished with seed 4; the clear win is at 1/10 only (L651). At 1/5, H7S ≈ H5 (L650).
- H7M (median-correct targets) ≤ H7S (L656, L666, L668).
- Per-ratio specialists R5/R10/R20 ≈ H7S: R10 vs H7S +47/−57; R20 +69/−59; R5 +43/−47. Conclusion: one mixed-ratio model (L702, L706, L721).
- **H8a (1.75× H7 data) ≈ H7S at every ratio (n.s.), ≈1k fewer tokens** (L722, L724, L728):
  - 1/10: all 71.0%/18.3k vs full 71.7%/17.3k vs summary 68.7%/21.2k.
  - 1/5: 71.3%/17.1k.
  - 1/20: 65.7%/20.0k, still at or below the summary.
- **"Simple" paper view (user-requested).** Seeds 2–4, one AC model (H7S), hard tasks ranked by the original 9B run's tokens after the cut (L662).
  - Hardest 25%: full 62.7 / AC 1/5 61.3 / 1/10 58.7 / 1/20 53.3 / summary 55.3 / nothing 50.0.
  - Tokens: 28.1 / 29.2 / 30.5 / 33.0 / 34.4 / 39.5k.
  - Figures: `kit/make_fig_compaction_v3.py` → `IB/TMP/PAPER_COMPACTION/figs/fig_compaction_simple_top{25,50}.*`; draft v2 `figs/fig_compaction_v2_draft.*` (L642).
  - RESULTS_SO_FAR section: "For the writing session".
- **Fraction scan** (seed 0, 200 tasks per cut, cut at 50/65/80% of run length). Full − summary at 30k: 65% cut +17 (p 0.01). Summary keeps 94/75/87% of full's gain over nothing. Nothing at 50k: 127 solved, 33,335 tokens (L148).
- AC H0 on the 65% cut: AC − summary −2 (p .86), −984 tokens. The headline stays on the 20k cut (L261–266).

**Trajectory Q/A synthesis (TQA)**
- Single call, early (L42–50):

  | Variant | Musique F1=1 | DeepMath |
  |---|---|---|
  | majority | .33 | .73 |
  | text summaries + answers | .40 | .83 |
  | results-only | .38 | .81 |
  | T1 AC | .36 | .80 |

- Reserve set (100+100), after the lenient-extraction fix (`kit/score_synth_lenient.py`, `IB/TMP/PAPER_TRAJ_QA/node_sync/scores/lenient_*_v1.json`). Values are Musique F1=1 / F1≥.5 / DeepMath:
  - **u245 SFT, 2 seeds: .405 / .49 / .92**; results-only .40/.51/.90; majority .35/.45/.88 (L352–357, L490).
  - Rows-removed ≈ AC; answers-removed .38/.49/.92 (L369–370).
  - Additions on vs off: no effect (L446). 1a rebalanced: no gain (L442). 1b: 2.3% yield, closed (L484–492).
- u245 is selection-tainted (the reserve chose it). The clean set is extended reserve `tqa_extres_solve_v1`, 300 Musique (L660).
- **Musique extres, 3 seeds, F1=1** (`node_sync/r2read/extres_3seed/`):
  - B 108.7 ± 3.2 > u245 105.3 > B-RR 101.7 > RO 99.0 > majority 93; any@3 126 (L686, L734).
  - B vs B-RR +38/−17 (p .006); B vs RO +43/−14.
  - u245's edge over RO is mostly on unanimous tasks (+15/−0); mixed tasks tie (L686).
- **DeepMath extended reserve `tqa_dmext_solve_v1`** (300, difficulty ≥6), 3 seeds (L676, L684, L703, L705):
  - RO 269.3 ± 0.6, u245 266.7 ± 1.2, B 266.7 ± 4.7, B-RR 269.7 ± 1.2; majority 260, any@3 276.
  - No trained arm beats RO.
- **DeepMath hard `tqa_dmhard`** (300, ≥8.5), 3 seeds (L713–715, L723, L737; `node_sync/dmhard/tables_*.txt`):
  - **M1 250.0 ± 1.0 (.833), best**; B-RR 247.0; B 243.7; RO 243.7; u245 241.3; majority 237; any@3 265.
  - M1 vs RO +41/−22 (p .023); ≥9 +15/−3 (p .008).
  - M1-RR tables pending at L740 (other range).
- Headroom: Musique extres ceiling any@3 .420 vs majority .310 (L660). DeepMath ≥6 is near-saturated (L670–671). User target "0.8 on ≥8" = 66/82 (L677).
- TQA dev gate, registered rule: **NO-GO** for arm A (mean gain over RR 3.0 < 4) (L633). Pooled gain ≈+3.2/seed over 124 mixed tasks; the +13/−3 on reserve was small-n luck (L654).

**AM compaction** (Qwen3-4B LongHealth, 1/10; Oct 19 insert): MVP = G2n_all20_lr1e3 vs AM-own full context .795 / single patient .8375; 3-pass mean, SE ≈.013 (L71, L85).

### 2. Training protocols and hyperparameters

**Global**
- **Rates 2e-5 / 2e-5 everywhere** (AC modules / reader LoRA). "never 5e-5; no raise without the user's word"; per-layer additions only with the user's word (L97, L147).
  - Exception in the Oct 19 AM work: `G2n_all20_lr1e3` and side-arm "lr 3e-3 continuation, epu 16, delta_scale ×3" (L71–72). This contradicts the 2e-5 rule; the authorization is not in this range.
- Items/updates ratio is ≈8 in every packet (8,000→1,000; 1,301→163; 6,957→870; 4,209→527). This is inferred from the counts; batch size is never stated.
- Snapshots: 50% plus final. One TQA job had only an end-of-run save and was nearly lost (L636). H7 used CHANNEL_SNAPSHOT_PERCENT=100 (L603).
- Rows clamped at 4,096 (L669, L738).
- Pair = `ac_modules.pt` + `side_lora` + reader LoRA (`adapter_model.safetensors`, `adapter_config.json`); no optimizer.pt (L95).

**Compaction arms** (all 1 epoch unless noted; ratio set 1/5, 1/10, 1/20; eval tags ac_4000 / ac_2000 / ac_1000 = 1/5 / 1/10 / 1/20 on 20k prefixes)

| Arm | Init | Material | Updates | Notes |
|---|---|---|---|---|
| H0 (w23A_final) | — | trained on 27B-written prefixes, "training-side waves" | — | no recipe in this range (L318, L516) |
| H3 | w23A | packet `s1_from_w23A`, material `train_s1_780.jsonl` | 681 steps | sft .95→.61; no gain (L66, L200, L385) |
| H5 | H0 | 156 9B pool runs (89 repos), exporter cut min(20k, f×len), f∈[.5,.8]; prefix median 7.8k, only 1 of 156 at the full 20k cut | 144 | material too short vs eval (L174–175, L403) |
| H5x2 | H0 | same as H5, 2 epochs | 288 | sft .18 vs .37, possible overfit (L428) |
| H6 | H0 | 866 runs, mixed cuts, `--window-min-total` T=24000, prefix median 12.3k | ≈430 | (L406, L431) |
| H6L | H0 | 336 runs, all cut at 20k (median 19.4k) | ≈320–330 | (L430) |
| H6fromH5 | h5_final | 866 runs, 3,384 items | 423 | (L476, L498) |
| H7S | h5_final | k=3 9B continuations per prefix, shortest correct + up to 2 within 1.3×; cuts c16/c20/c24 (c28 also cut); 4,016 items | 502 | ratio drawn log-uniform r∈[1/20, 1/5], so it never saw exactly 1/10 (L594–603, L661) |
| H7M | h5_final | median-correct target at r≤1/10; 4,015 items, 1,966 at r≤1/10 | 502 | (L603) |
| R5/R10/R20 | h5_final | `--one-item-per-line --r-range r r`, 2,840 items at one ratio | 355 | R5 had 373 items clamped (L661, L669) |
| H8a | h5_final | H7S settings, 3,393 kept prefixes, 6,957 items | 870 | (L682, L697) |
| H8 full | h5_final | 4,276 kept prefixes, 8,793 items | 1,100 | still training at L738 |

- H7 material rules: `--nominal-prefix-tokens 0`, guess guard kept (L541). Exclusion is task-level only (`kit/exclude_h7.json` = eval-200 ∪ dev-75 ∪ frozen-100, 322 ids). There is no length/outcome filter (L535, L538).
- Mean kept H7 continuation is 15.8k, which is "well above the proposer's ≈10.5k" (L597).
- Held panel thin: 6 items, 2 repos (L605).
- **Init inconsistency:** H5/H6/H6L start from H0, but H6fromH5/H7/H8/specialists start from h5_final. H7S therefore had two stages of training (H0→H5→H7S). The specialist init was switched to h5_final to avoid a second-epoch confound (L661).

**Compaction eval protocol**
- 9B continues from the prefix. Variants: full, summary (summary_write + summary), nothing, truncation (recent_N), AC at ratios (L123, L145).
- 50k budget runs use `--budget 50000 --trajectory-cap 90000 --max-model-len 100000 --max-duration 6000`. The scan used cap 100000 / len 110000 (L123, L137).
- The 30k view is replayed from 50k runs with `kit/view_budget.py`, which stops at the end of the turn that crosses the limit (L123).
- Original 30k-only runs used engine 72k after the 52k assertion bug (L111–112).
- Agents per GPU: 40 (96 was ≈12× slower per token; >40 breaks sandboxes) (L211–212, L537). AC: encoder share 0.55, 32–64 agents. Share 0.4 + encode batch 48k + 96 agents OOMed (L180–184).
- `run_resume_k` groups k samples per prefix, ≈+25% throughput (L537).
- **Seeds and sets:**
  - eval-200 (`prefixes_eval9c_200`): 9B pool v3 over 600 eval tasks, 278 qualifying; first 200 in eval order: py 89 / java 77 / kt 34; prefix median 20,449, max 25,983; 0 overlap with dev (L135).
  - dev 75 (`prefixes_dev9_75`) (L173).
  - Headline seeds s0–s2. s3–s4 added later for power and for the held-out figure (L507, L510).
- The 1/20 arm was dropped for s0–s2 to save GPUs (L252, L305). The 1/20 numbers come from s2–4 only.
- Scoring rule: count finish_reason before reading any AC run (L185).

**TQA**
- 9B solver ×3 seeds, 40k cap, compaction off; 27B oracle writes Q/A from full texts (L14, L20).
- T1 → own_ro packet `tqa_own_ro_v1`: blind, rows only at 1/20–1/5, 8,000 items = 1,000 updates (Musique 4,556, DeepMath 3,444) (L126). Multi-trajectory questions were excluded (builder limitation).
- Last-mile SFT `synth_sft_v1` from tqa_own_ro_final:
  - 1,301 items, 163 updates × 2 epochs. Targets are 27B completions with thinking, ending in `Final answer:`, kept if correct (1,364 kept) (L126, L144).
  - The final save failed on quota, so **u245 (update 245 of 326) is a snapshot used as the model**. Epoch 2 was never rerun (L236–239).
- 1a rebalanced: 909 items, 114 updates (L325). Additions SFT: 326 updates, heads reset (L191, L348).
- Round 2:
  - Arm A: 1,405 ex, 162 updates, from tqa_own_ro_final (L576–580).
  - B: union r2_bu, 9,976 ex, 1,234 updates, 18 h time limit (L628, L637).
  - M1: 4,209 items (DeepMath 3,804, Musique 405 replay), 527 updates, init B (L716–717).
- Generation caps:
  - Synthesis read max_new 16,384. At a 16k cap the text arms collapse (results-only DeepMath .65), so cross-engine capped comparisons are not like for like (L209, L435).
  - Round-2 27B targets: 32k max new tokens (L547).
- Grounding filter loosened from 54–65% reject to Musique 10% / DeepMath 18% (L546).
- Scoring: lenient + last line; DeepMath re-scored with math-verify (L162–170, L670).

### 3. Durable lessons

- **SkyPilot autodown killed both EC2 nodes, disks included** (L129–131). The cause was a 30-min idle autostop + `--down` (`activation/cloud/sky.py:57,523`). nohup work does not count as activity, and the `sky exec -d 'sleep 86400'` keepalive expired. Rule: keepalive with logged expiry plus a pull loop every 10 min.
- **CSAIL /data/scratch** is shared 50T NFS with a per-user quota, invisible from the login node; it gives `Errno 122 Disk quota exceeded` (L236, L245). Silent job failures (exit 1, no traceback) were quota writes (L220, L223).
  - Trainer preflight needs 32 GiB free (L62). Moved to `/data/scratch-fast/ngom/activation/...` (3.8T, nfs-prod-aia-1) (L195–199).
  - Set TRITON_CACHE_DIR, HOME and TORCHINDUCTOR_CACHE_DIR to scratch-fast (L246).
  - The job guard CHANNEL_QUOTA_GIB was hardcoded and is now read from env (L604).
  - CSAIL mtimes are US Eastern (L635).
- **Agent rollouts record CUDA OOM as finish_reason "error", not as a crash**, so check error counts before reading (L181, L185).
- Side_base load race "Cannot copy out of meta tensor" (no lock in model_to_device) made a whole read error out (L730).
- AC reads' saved_tensors are 5–18G each and filled pc-b. Archive them to S3 after each table; never copy them between nodes; use df ≥40G guards (L691, L251).
- S3 archives:
  - `s3://general-bucket-aln/activation-context-data/container_archive_20261015/<path>` via `IB/TMP/ARCHIVE/s3_archive.py`, size-verified manifests `IB/TMP/ARCHIVE/*.manifest.json` (L564).
  - Archived there: SYNC_RUNS, long_recon_ec2, paper_runs, FIRST_TASK_AWARE_FINETUNING (25,893 files, 141.1 GiB) (L565–570).
  - Node tensors: `.../node_archive_20261016/pc-b/` via `node_s3_archive.py`, manifests `pc/s3_archive_manifests/` (L691, L695).
- `uv cache clean` freed 30.4 GiB (L566).
- 9B engine: the recommended length of 52k is short of 50k cap + 8k turn, so pass `run_pool.py --max-model-len` (L111–112). The cache key is task/variant/seed and tag-independent, so caches can be concatenated across tags (L145).
- Throughput:
  - TQA training: GH200 53.6 s/update vs H200 pair 21.1 (L550); later 36 s/update on the pair, which pushed a job past its 12 h limit (L636).
  - Pools ≈350 runs/GPU-h; H7 continuations ≈335 rows/GPU-h (L559, L646).
  - The 9B solver for TQA runs ≈800–1,200 rows/h/GPU (L548, L693).
- PyTorch eval path validated against vLLM: mean |Δ logprob| .014, p99 .14; rows cosine ≥.99995 (L206–207, L324).
- Process hygiene: `pgrep -f` self-match idled GPUs twice; kill by PID/PGID only (L101, L432). A `$PY` expanded on the wrong side broke a waiter (L135). A READY-path typo left a GPU idle (L689).
- Cluster rules:
  - ORCD/CSAIL only through `IB/TMP/AGENT_AC_PRETRAINING/ac_recon/{orcd,csail}_ssh.sh`; exit 97 means the user re-auths (L91).
  - One guarded call per ≥50 min (L93). Never overwrite CSAIL files; no `sbatch --export=ALL,...` (L95).
  - ≤2 GH200 jobs and ≤2 H200 GPUs on CSAIL. ORCD reservation 23493337 is off limits. Never touch i-0918b8f73e9a0fe03 (L88–89).

### 4. User decisions and preferences

- Push evaluation to MVP; "Gently warn against 1–2 % micro-optimizing" (L10). Compaction is "just good enough for our MVP, but more is better" (L11).
- 9B prefixes: "explaining that the prefix is from 27B, but the continuation from 9B just makes no sense" (L12). Benchmark 200 prefixes × 3 seeds (L118). Run at a 50k limit, report as if 30k (L123).
- **Data rules:**
  - Training may use evaluation repositories; only exact eval/dev tasks are excluded (L116).
  - "repository overlap is not leakage; only task overlap is" (L533).
  - "Make sure not to make training artificially harder." (L534)
- Fraction scan: "Instead of 20k, say we use the last 75% or 80% or 60%"; "publish early results here so I can pivot accordingly" (L137).
- Overnight hill-climb, verbatim: "any general hillclimbing you+some proposers can think of ... (use many gpus, more/less data, maybe faster turnarounds etc as you see fit; up to you)." (L147)
- Autonomy rule (Oct 19): never wait on the user; hyperparameter tweaks and new arms need no approval (L84).
- Cleanup: "When in doubt, upload to s3 first, then cleanup" (L563). Remove the obvious, ask before more (L121). Optimizer.pt deletions approved "Yes" ≈11:05Z Oct 16 (L664).
- Green light (12:00Z Oct 15) on hard-subset readout, H7, and 1/20 evals. Final-plot spec: token plot = continuation quantiles; solving plot = top-difficulty subset (L512).
- Simpler paper presentation, no rank-on-X/read-on-Y. DeepMath eval → 300, "go for 300". All synthesis numbers on 3 seeds (L662).
- Synthesis targets (L672):
  - 1–3 pt gains count as wins.
  - "0.8 on 8 and up would be perfect".
  - "you can always use 27B trajectories to push the number of available solves".
- "Targetting 8.5 and up, with some special tuning as needed, is perfectly fine." (L678)
- Per-ratio specialists requested (L658). Result: no gain (L721).
- Oct 19 priorities, verbatim (L70):
  - "The most important thing is the MVP: showing our trained nonuniform KV beating even the full-long-context reference."
  - Ablations "to the point where they plateau".
  - "I would not automatically shutdown EC2 until you can see clearly that csail is back and working."
- **Paper trims (notes only, not marked in tex):**
  - §2.5 Engine Implementation is a trim candidate (L79).
  - Select → text baseline in §4.1: "KEEP for now, but a good candidate for removal" (L80).
  - The §4.5 decode-baseline bullets are a separate, undecided decision (L80).
- **Unresolved or pending in this range:**
  - /data/scratch quota (L494).
  - 27B NVFP4 (22G) in pc-a's HF cache (L494, L664).
  - r2_a optimizer.pt (3.9G) not deleted (L664).
  - The three 11:30Z proposals (a/b/c) were batched and green-lit (L511–512).

### 5. Rerun and follow-up candidates, known weaknesses

- **Small samples, noisy CIs:**
  - Hard-subset figures use 50 tasks (top 25%), bootstrap CI ±10 pts (L642).
  - The 1/5 vs 1/10 inversion within H7S is "likely noise" (L650).
  - Seed-0 leads repeatedly vanished: AC ≈ full (L296), B on dmext (L703).
  - Dev overstated H5 (L466).
- **Headline AC vs summary is not significant** (+2.3, p .69). The token saving vs summary is p .08 (L373).
- **Inconsistent seeds across arms:**
  - 1/20 was dropped on s0–2 (L252, L305).
  - Quantile tables mix s2–3 and s2–4 (L622).
  - RO had s0 only at first on extres (L654).
- **Selection-tainted numbers:** u245 was chosen on the reserve set used to report it (L660). The dev gate and reserve were later dropped as "decision-used sets" (L678).
- **u245 is a 75%-through snapshot.** The final epoch-2 save failed and was never rerun (L236–239). The same applies to H5's first run and other quota failures.
- **Init and epoch confounds:** H7/H8 start from h5_final, but H5/H6 start from H0 (L661). H5x2 may have overfit (L428).
- **Material length mismatch:** H5 material is much shorter than the eval prefixes (L403). The H7 mean target is longer than intended (L597).
- **Not evaluated or incomplete:**
  - H3 dev (s0 only), H6L and H6fromH5 (seed 2 only) (L482, L632).
  - T1 pair on the reserve set (L142).
  - Ratio-0.2 TQA probe OOMed (L367).
  - AC on the 80% cut dropped (L305).
  - H8 full still training at L738.
  - 9B-written frozen-100 AC with arm comparisons beyond H0 — not in this range.
- **TQA structural limits:**
  - Readers only pick among solver answers. k=0 tasks are unrecoverable (L443, L492).
  - Headroom over majority is structural at k=3; raising k (k=6) is the knob (L671).
  - DeepMath ≥6 is near-saturated (L670).
  - On Musique, AC > text mainly by not overriding agreement (L686).
- **Additions (deep residual interventions):** stale heads in existing pairs (bit-identical to lc3i). Retrained, no effect (L191, L446).
- **Arbitrary or ad hoc knobs worth fixing:**
  - Encoder share (0.4 vs 0.55), agents per GPU (40/64/32/96) (L184, L211–217).
  - Grounding thresholds loosened ad hoc (L546).
  - B-mix rule (×2 oversample if A fails) (L616).
  - Exporter f∈[.5,.8] and the 20k cut (L174).
  - Ratio sampling log-uniform (L661).
  - 4,096-row clamp (L669).
  - Lenient vs strict parsing changes conclusions: AC vs RO went from p .035 to n.s. (L168).
- Uniform-vs-nonuniform AM ablations and "plateau" continuation (lr 3e-3, epu 16, delta_scale ×3) are only planned in the Oct 19 insert. Outcomes are outside this range (L72).

## STATE.md extract B — lines 741–959

Dates (inferred from neighbouring entries; the log has no date headings here): L741–754 = Oct 17; L755–792 = Oct 18; L793–893 = Oct 19; L894–959 = Oct 20 (2026). Workstreams: TQA (trajectory Q/A, DeepMath/Musique), compaction (H8a/H8F), Llama corpus Q/A (K15/R15b/D15b, axed L853), AM corpus (LongHealth, Qwen3-4B, the "corpus compression" paper figure), retrieval Q/A (HotpotQA).
S3 root = s3://general-bucket-aln/activation-context-data/.

### 1. Final / headline results

**TQA (9B solver x3 runs; reader reads, 300 tasks/set):**
- Musique extres M1 s0–2: 109.0 ± 3.0 (112/106/109) = B 108.7; vs RO +48/−18 (p <.001); math targeting cost nothing on Musique; on Musique the rows help (p .006) (L741, L746).
- dmhard (DeepMath ≥8.5) 3 seeds: M1 250.0 (.833), M1-RR (rows removed) 246.0 ± 2.0 (.820); M1-RR vs RO p .40, so M1's gain over RO goes with the rows (L744).
- dmext (6–8) 3 seeds: M1 268.3 ± 2.3 (.894), RO 269.3, B 266.7, majority 260, any@3 276; M1-RR 272.3 ± 0.6 (.908) > M1 (p .029) (L745–746).
- dmhard 5 seeds (node_sync/dmhard/tables_M1_5seed.txt): RO 244.6 ± 1.9 (.815), M1 248.2 ± 2.8 (.827), M1-RR 246.4 ± 1.7 (.821), majority 237, any@3 265; M1 vs RO +64/−46 (p .105; was p .023 on 3 seeds); ≥9 +22/−10 (p .050). Seeds 3–4 regressed for M1 (L750).
- Unified DeepMath table, separate layout (6–8 dmext n 272 / 8.5 dmhard n 226 / ≥9 dmhard n 74 / all): Results-Only 92.4/81.2/82.4/86.7; Results+AC (M1) 91.5/81.8/85.7/86.9; Results+LoRA (M1 rows removed) 93.3/81.8/83.2; Majority (random tie-break) 90.1/81.0/76.1; Results+Summary 92.6/79.6/83.5/86.3. M1 vs Summary all +76/−52 (p .042); M1 vs results ≥9 +22/−10 (p .050) (L763–765, L795). Files: IB/ARTIFACTS/PAPER_TRAJ_QA/tables/{tables_unified_tb_sumres.txt,tqa_unified_sumres.json}, scripts/tqa_unified_table_sumres.py, make_fig_tqa_unified_sumres.py; IB/ARTIFACTS/PAPER_FIGURES/tab_tqa_mvp.tex (L795, L797, L799).
- Canonical math figure IB/ARTIFACTS/PAPER_TRAJ_QA/figs/fig_tqa_mvp.{png,pdf,json} (six arms incl. Summary; y 70–95); old one fig_tqa_mvp_nosummary.* (L769, L799). Overleaf copies in IB/ARTIFACTS/PAPER_FIGURES/ (+README) (L770).
- Difficulty figure: M1−RO by bucket 6/6.5–7/7.5–8/8.5/≥9 = 0/−0.4/−2.2/+1.0/+5.4 pts (IB/TMP/PAPER_TRAJ_QA/figs/fig_tqa_difficulty_m1.*; 5-seed version fig_tqa_difficulty_m1_5seed.png) (L745, L750).

**Compaction (eval-200, seeds 2–4, solved % / continuation tokens):**
- Full H8 (H8F, 2.2x data): 1/5 70.5%/17.4k, 1/10 68.7%/17.6k, 1/20 66.8%/19.6k vs H8a 71.3/71.0/65.7; full context 71.7%/17.3k; summary 68.7%/21.2k. No gain over H8a; **H8a stays the paper's compaction model** (L752–754).
- Canonical figure IB/ARTIFACTS/PAPER_COMPACTION/figs/fig_compaction_mvp.{png,pdf,json} (bar chart, ratios 1/5 and 1/10, x = Mean/P50/P75 of "Continuation Difficulty" = teacher tokens after cut; AC = H8a); script IB/ARTIFACTS/PAPER_COMPACTION/scripts/make_fig_compaction_v4.py (`--style paper --ratios 1/5,1/10 --no-trunc --upto P75 --kind bar --values --fig-width 10`) (L757, L759–762).

**AM corpus (LongHealth 4 contexts x 5 patients, 400 MC Q, Qwen3-4B-Instruct-2507, 1/10):**
- AM's own code: nonuniform t0.1_fastest 71/79/79/72 = .7525; uniform 52/58/64/62 = .59; full 5-patient .795 (77/84/77/80); no context .360; single right patient .8375 (89/86/83/77) (L857, L867, L889).
- Our harness reproduces AM nonuniform: 69/78/81/75 = .7575 (L863).
- **MVP, G2n_all20_lr1e3 final** (trained nonuniform KV, 1 epoch v1 = 2,221 updates x 8): pass 0 .8175 (.86/.84/.80/.77); **3-pass mean .8108** (973/1,200; 95% CI .776–.845) (L896, L902). Full context vLLM 10 passes .7995; paired diff +.0113, CI −.0088…+.0315, P(diff ≤ 0) .14 — not significant (L930). vLLM Right Patient 3 passes .8617 (L921).
- Trained uniform U2n final .7575 (.77/.77/.72/.77) vs AM uniform .59 (L898).
- Curves (400 Q, 1 pass): G2n u0 .7575, u500 .795, u1000 .815, u1500 .8125, u2221 .8175; U2n u0 .5575, u500 .72, u1000 .745, u1500 .75, u2221 .7575 — both flat after ≈8k pairs (L907).
- AC Input (rows): u0 .3625, u500 .4425, u1000 .4275, u1500 .48, u2221 .4675; continuation C_rows s1000 (3,221 updates) .5175 (L907, L909–910, L937–938).
- AC Input + Residuals (deepS): u500 .405, u1000 .4775, u1500 .535, u2000 .56; u2221 = R_deepS rerun, eval pending at L942 (L930, L937, L948).
- Cartridges = .69 read off AM Fig. 3 (upper bound), not run (L888). Token budget: ours 2.30M/context (518.5 tok/pair) vs Cartridges ≥100M (≈43x) (L913).
- Files: IB/ARTIFACTS/PAPER_AM/results.json, evals/, figs/make_figs.py → fig_am_mvp, fig_am_curves, fig_am_curves_tokens_{log,break}; PAPER_FIGURES/fig_am_mvp.*, fig_corpus_compression_curves_tokens_mvp.* (L886, L896, L902, L914, L922). Full-context 10-pass file CSAIL am_ec2_out/vfull/vfull_p3_9.json + container tmp/vfull/ (L931). EC2 final backup S3 am_lh5_20261019/run2/{cqa-ec2r,cqa-ec2b2}/final_small (L917); ec2b refs am_lh5_20261019/ec2b/ (L867).

**Llama-3.1-8B corpus Q/A (abandoned L853; 200 MC, patients 1–10, 1/15, all/right/none views):** K15 KV final .455/.46/.03; K15c (+2,128 uniform-key pairs) .505/.47/.005; R15b rows final .505/.505/.02; D15b residual u400 .510; H1a (KV side LR 1e-4) ≈+10 pts over K15 at matched pairs (L808, L810, L820, L835, L812). S3 corpus_qa_20261019/{K15,K15c,R15b,D15b}/, corpus_hillclimb_20261019/ (L821, L826, L873).

**Retrieval Q/A text baselines (HotpotQA pooled 509,315; question-only BM25):**
- Gold-context 9B teacher (thinking) eval500 acc 73.2 / EM 65.6 / F1 82.3 (L932).
- eval500 3 passes, cap 12,288 (acc strict / lenient): thinking ON none 25.4, gold 73.07, top20 49.53, top400 59.0/59.13; thinking OFF none 23.07, gold 67.73/67.80, top20 43.93, top400 53.47/55.20 (L959). Copies IB/TMP/PAPER_RETRIEVAL_QA/text_baselines_ec2/.
- dev200 1 pass: ON none 24.5, gold 75.5, top20 57.0, top400 62.0; OFF 25.5 / 68.0 / 50.5 / 54.0 (L942, L958).

### 2. Training protocols and hyperparameters

**TQA:** M1 compression 1/10 per solver run (realized 0.100) (L764). Majority = first answer in seed order matched by another run, later random tie-break (L764–765). Summary arm = untuned 9B single turn, old summary-stage prompt, one resample (seed+1000) if no compact call; reads seeds dmh 0–4 / dmx 0–2 (L771) — note dmext has 3 seeds, dmhard 5.

**Compaction H8F:** 1,100 updates in 9:40, final loss 0.331; H8a = 1.75x data, H8F = 2.2x (L751–752).

**Llama corpus (K15 etc.):** unsloth/Llama-3.1-8B-Instruct; teacher = same model with the full right patient document; self-study via covering units 64–256 tok, 8 kinds, 25% MC, dedupe; 8,354 pairs (L781–785). K15: 2xH200, 4 ex/GPU = 8 global, recon 240 ex (30 updates), cap 6,160 pairs, LR 2e-5/2e-5 (proposed constant after warm-up), LoRA r128 (L779, L787, L798). R15b rows micro 8, D15b deltas micro 4 (micro 8 OOM), every-4th layers 3,7,…,35 (L790, L870). R15/D15 first ran with a 2,064-pair cap, restarted as R15b/D15b with K15's cap/order (L790). Hill-climb: lr_ac 1e-4 chosen; 2e-4 makes no-context answering leak (L808–809).

**AM corpus (all arms):**
- Data v1 per context 5,000 = 3,750 self-study + 1,250 eval-shaped (AM instruction, kept if parses), ≈1k/patient, near-dup filter vs 400 eval Qs; all-20 = 19,977; trained on 17,764 after dropping 2,213 truncated (`--keep-truncated` only added later) (L840, L853, L862, L875).
- Teacher: Qwen3-4B reading only the right patient; KL top-32 + remainder, precomputed (am_teacher.py) (L839–840, L855, L858).
- Recipe: lr_special 1e-3, epu 8, 1 epoch = 2,221 updates, cosine to 10%, ckpt 250, frozen reader (**lr_reader 0 for every corpus arm**), no gradient checkpointing (L858, L877, L884, L934).
- Inits: G2n = AM nonuniform closed-form t0.1_fastest; U2n = AM uniform t0.1_fastest-u-all; rows = closed-form selection (V most-attended tokens from 32 training user messages, reader input embeddings); deepS = rows + zero deltas at 3,7,…,35 with delta_scale 11.3…611.1 (L874, L879–880, L889).
- **LR choice was probe-based only for rows** (3e-4/1e-3/3e-3 on ctx0, 125 updates → 1e-3, L877); KV lr 1e-3 was set ad hoc from the first ctx0 run (L858) and reused for uniform/deepS. Residual layer set (every 4th) defaulted, user never chose (L868, L872).
- Continuations (≈2x data): `--handler-state`, fresh optimizer + cosine over 1 epoch of v1v2s1 (39,881 samples, 39,780 targets), `--keep-truncated`, ckpt 500; LR C_U 1e-3, **C_G 5e-4**, C_rows/C_deepS 1e-3; **C_deepS started from u2000, not u2221** (lost) (L893, L899, L901, L933). v2 data recipe differs from v1 (6 more seed kinds, temperatures 0.6/0.8/1.0; eval-shape agreement .769–.788 vs v1 .874–.922) (L872, L883, L894).
- Eval: AM protocol, sampling 0.7/0.8/20, max_new 2048, AM parse_model_choice; pass k seeds 1000k + context; 1 pass for snapshots, 3 for finals; rows evals batch 16, KV batch 8 (L840, L887, L906, L916). Full context measured with vLLM (prefix caching) rather than the HF harness (L916).

**Retrieval:** train12k (9,681 bridge + 2,319 comparison) → 10,388 kept teacher targets; eval500, dev200 (160 bridge/40 comparison) (L923, L928, L932). Ours = 20 contextual 2B calls x 20 passages at 1/20 (also 1/10, 1/40) (L924, L944). Start pairs PAIR6459 (epoch 1 step 6,459 of 21,982, not final) and PAIR75M (L927–928). hc: 800 items = 100 updates, reader 2e-5, encoder 5e-5/1e-4/2e-4 (L929). Full run full_nt_p75m_e5e5_r20 (2717966): 2xH200 DDP, enc 5e-5 / reader 2e-5, 1/20, KL micro-batch 2, checkpoint evals every 250, thinking off, cap 12,288 (L952, L958). Thinking-off sampling: baselines T 0.7/top_p 0.8/top_k 20/**presence 1.5**; READ_SAMPLING for teacher reads listed as T 0.7/top_p 0.8/top_k 20 (no presence mentioned) — check consistency (L953–954).

### 3. Durable lessons

- **CSAIL /data/scratch has a per-user quota** (writes die with "Disk quota exceeded" while the FS shows free space); write outputs to /data/scratch-fast/ngom/ (L925–926). **ORCD per-user scratch quota ≈1.0T** (Errno 122), unreadable over NFS (L814, L817).
- Bad CSAIL node aia-h200-12 (CUDA error 802); scripts ending in `echo` exit 0 and hide failures → use `--exclude` and check exit codes (L786, L948). `run_corpus_shared` bad exits don't requeue (L813).
- Slurm prints EDT (misread maintenance window by 4 h) (L805). `--time-min` shortens the job's actual end time (L788).
- Spot nodes vanish with their disks (cqa-ec2 at 22:01Z, cqa-ec2b at 02:14:52Z): back up checkpoint tables to S3 as they appear (L881, L900). 2x RTX PRO 6000 spot (g7e.12xlarge ≈$2.23–2.31/h) often unavailable; on-demand $8.29/h (L855, L875, L901).
- SkyPilot copies ~/.aws/credentials onto nodes; the `sky exec` wrapper passes `--secret-file .env` (L867, L901, L936). `sky setup` needs UV_PYTHON_INSTALL_DIR/UV_CACHE_DIR overrides (L800).
- vLLM: fp8 KV cache fails on RTX PRO 6000 ("FlashInfer backend is not available") → bf16 KV (L940); spawn needs a `__main__` guard (L916); 3 simultaneous cold FLA autotunes on GH200 fail (L935); GH200 OOM at eval batch 100 → 32 (L949).
- Full-context 60k-token eval in the HF harness ≈69 s/question (7.7 h/400 Q); vLLM with prefix caching takes minutes (L910, L918).
- Speed facts: per-example re-encoding was the cost; `share_part_encodings` 1.8–2.0x, exact (L787–789); kv_transfer MASKLESS + per-layer widths + no fp32 cast: 22–26 → 5.1–6.6 s/update (L861–862); AMKVTable left on CPU was the main slowdown (L859); capacity check once per distinct part saved ≈25 min setup (L879); a frozen zero LoRA runs in fp32 (L852); retrieval encode = 2B side decoder ≈79% (L946); KL micro-batching ≈10% and loss-identical (L943–945); retrieval setup is CPU-bound (build_example ≈15.5 min at 9,897 items) (L959).
- Trainer quirks: keeps only the latest mid-epoch checkpoint; the final checkpoint is the epoch folder, not step N (L864, L878); parallel arms share ACTIVATION_SYNC_ROOT/AC_MODELS/rag_poc → give each its own (L862); resume refuses a silent optimizer reset (L786); trainer sync writes optimizer.pt ≈10G per checkpoint (L925).
- Eval: sampling was refused with payload rows (fixed in hf_utils; code_v6) (L909); rows-channel answers are long → 1.3–1.6 h/400 Q at any batch (L908). Noise: 100-Q pass SE ≈.04, 400 Q ≈.02, 3 passes ≈.013 (L884); cross-hardware ±1 pt (L826).
- MC letter-prior collapse (B, then A) confounded early Llama accuracy; use uniform keys + balanced accuracy (L792, L802, L826).
- One job per GPU on EC2 (user) (L852). AM's paper reports compaction time only; per-context end-to-end ≈29 min (L858). AM nonuniform: real entries 10.0% of cache, per-layer padded 31.4%, block table 75% (L865).

### 4. User decisions and preferences

- TQA table arm names: Single Solver / Majority Vote / Results-Only / Results + LoRA / Results + AC; Majority = random tie-break; separate 6–8/8.5/≥9 layout (L764–765). "Queue Results + Summary. But don't put it in the figures yet." (L766), later added to figure (L799).
- Compaction figure: bar chart, 1/5+1/10, up to P75, no truncation; canonical "unless PI feedback this week changes it"; caption must say panel (c) was timed on 27B frozen-100 prefixes (L760–762).
- Corpus (Llama, later axed): Llama-3.1-8B for every arm; KV 1/15 only for MVP; "Don't even expect something else"; teacher with full right document ("let's not overindex on what they do"); no ratio change, "stick to 1/15" (L777–780, L831). "Axe them. We are not doing that anymore." (L853).
- "Do the orcd cleanup from a forked agent… Don't wait for me on things like this again. Just do it." (L831). Standing autonomy (L842).
- AM: "Use with your defaults, unless they are, like, absurdly suboptimal…" (L840); "27B is always Qwen 3.8 27B. Medium reasoning if no-reasoning… behaves oddly." (L839); optimize benchmark numbers on the baselines' terms, practicality secondary (L865); "if we are scoring 0.86, I'd only worry about more data for the baselines" (L872); "No third node. This is a case where it's better to wait." (L873); "I'll give you the freedom on how to pick the best starting point. A short fit at high lr seems to make sense." (L874); MVP = one bar chart on 400 Q (L891); "stop worrying about the compression confidence interval. The mvp is fine as is, and we'll extend if someone asks for it." (L931); "2x" means data, not epochs; resumed points on the same line (L938).
- "Once csail goes down, and ONLY while it's down, spawn EXACTLY one 2 x rtx ec2 spot instance…" (L800). GPU caps: "stick to 3 quanta gh200, 3 h200", "Yes also 3 shared", +1 EC2 spot node (L934).
- Priorities Oct 20: retrieval first; then redo the 4 AC corpus arms with reader LoRA; extend LoRA KV arms to ≈2x if not plateaued (L934, L941).
- Retrieval: N = 10 calls (correction), then top-20 x 20 calls at 1/20 (L922, L924); targets: MVP > text top400, worst case top20, aim gold, "ideally 70+" (L944); frozen trunk "way over the line of acceptable optimization. Let's not do that at all"; "The 7-8h speed is actually fine"; eval every ~250 updates (L950); "Prefer thinking off unless it's ridiculously worse"; "don't cap us at 4096, if baseline is not capped either"; "1.7k vs 69 makes it worth it." (L955, L957).
- **Never resolved in range:** ~/.aws/credentials on sky nodes, "user to decide" (L901, L936); residual every-4th vs every layer (L868, L872); lenient vs strict parsing for the paper (recommended lenient, L959); retrieval start pair's true final checkpoint not located (L927); reader-id match check for R15b/D15b (moot, L870/L873); free-weights plan D1–D4 paused (L834, L836); AM R1/R3 in our harness; truncation arms on eval-200 (L758).

### 5. Rerun / follow-up candidates and weaknesses

- AM MVP lead over full context (+.0113) is inside the CI; full context was AM-own 1 pass in the figure vs 10 vLLM passes later (L902, L930–931). Engine gap: Right Patient .8375 (harness) vs .8617 (vLLM) (L921).
- **Unfair corpus baselines:** KV arms start from AM's closed-form cache (.7575/.5575), input arms from ≈no-context (.3625) (L907); lr 1e-3 tuned for rows only; all arms lr_reader 0 (LoRA-enabled reruns L_G2n/L_U2n/L_rows/L_deepS queued) (L934, L941). Input arms not plateaued at 1x (C_rows still rising) (L910).
- deepS u2221 lost to preemption; C_deepS continued from u2000; R_deepS rerun separately (L900–901, L919). C_G used 5e-4 vs others 1e-3 (L893). G2n ctx0 s548 never evaluated (L878).
- v2 data lower quality (eval-shape agreement .769–.788) and v2s3 never delivered (raw on ORCD .../datagen_v2/out/raw_v2s3/); v2s2 (19,916) unused (L894, L904).
- Cartridges number read off AM Fig. 3 (±0.02) and never run; token-budget comparison ≈65% confident interpretation (L836, L888, L913). AM baselines on ORCD (B1–B6) failed without running (L853).
- TQA: M1 vs RO not significant at 5 seeds (p .105); rows' effect on DeepMath unresolved (rows hurt at 6–8: M1 vs M1-RR +4/−18, p .004) (L746, L750, L763). Summary arm only from untuned 9B with old prompt (L771).
- Compaction: H8F (2.2x) no better than H8a; panel (c) timing from 27B frozen-100 vs 9B eval-200 panels; truncation baseline missing on eval-200 (L754, L758, L762).
- Llama corpus: letter-prior skew, KV at 2e-5 trailed rows/deltas; channels compared at unequal effective LR (L796, L808, L826).
- Retrieval: hc dev200 evals capped at 4,096 (cap-dominated, uninformative); start pair 6459 is a mid-run checkpoint; possible sampling mismatch (presence 1.5) between baseline and teacher reads; strict parser penalizes bare answers (top400 5/200) (L927, L954–959).

## STATE.md extract C — lines 959–1231 (Oct 20 16:40Z → Nov 9)

Conventions: S3 = s3://general-bucket-aln/activation-context-data/ (verified: `rq_runs/`, `sec45_20261026/runs/` exist there). $F = CSAIL rq work root; $CLAUDE_JOB_DIR/tmp = old container tmp (gone unless archived). "§4.5" in the log = paper §3.5 (training ablations, Fig 7). dev200 = HotpotQA dev 200; eval500 = disjoint 500-question eval set (L1223).

### 1. Final / headline results

**Retrieval Q/A (Fig 3; Qwen3.5-2B encoder → Qwen3.5-9B reader, 1/10, BM25 calls of 20, question-first, thinking off)**
- Terminal figure `fig_rq_frontier_mvp` (eval500, every point a 3-pass mean): AC (Arm G) top-100 (5×20) **52.87** / top-200 (10×20) **54.07**; Text 9B top-20 43.93 / top-100 49.2 / top-200 52.67; Gold Reference 67.73; No Context 23.07. User: "Retrieval has reached mvp." (L1048, L1045, L1047)
- Arm G single-pass eval500: 1×20 49.4, 5×20 52.4, 10×20 54.8, 20×20 53.6 (vs text top400 53.47, 3 passes) (L1035, L1037, L1038, L1040). dev200 Arm G 1/2/5/10/20×20 = 53.5 / 54.5 / 57.5 / 60.0 / 60.0 vs text top20 50.5, top100 53.5, top200 51.5, top400 54.0 (L1039, L1038, L1023).
- Text eval500 thinking-off baselines (3 passes, cap 12,288, strict/lenient): none 23.07, gold 67.73/67.80, top20 43.93, top400 53.47/55.20; thinking ON: none 25.4, gold 73.07, top20 49.53, top400 59.0/59.13 (L959). Text top100 49.2 (EM 44.47), top200 52.67 (EM 47.8) eval500 (L1026). Copies `IB/TMP/PAPER_RETRIEVAL_QA/text_baselines_ec2/` (L959).
- Reader-token counts (dev200): text top20 2,402 / top100 11,977 / top200 24,316 / top400 49,197; AC 1×20 331 / 2×20 574 / 5×20 1,303 / 10×20 2,555 / 20×20 5,079; gold text 337; none 81 (tok_counts.py next to make_figs.py) (L1032).
- Figure sources: `IB/ARTIFACTS/PAPER_CORPUS_RETRIEVAL_QA/figs/{results.json,make_figs.py}` (frontier_paper) → PAPER_FIGURES `fig_rq_frontier_{mvp,dev200}` (L1032, L1040, L1047). Arm G evals: S3 `rq_runs/20261021/evals_ec2/trackA_armG/`; Arm G ckpt `rq_runs/20261021/trackA_real/armG_final/` (2.77 GB, no optimizer) (L1027, L1041).
- **Ideal recall (golds injected, eval500, 3 passes; paper Fig 3(a))**: Text 9B k=20/100/200 = 66.33 / 64.07 / 63.80 vs AC Arm G 1×20i/5×20i/10×20i = **67.13 / 63.67 / 62.87**; tokens text 2,402 / 11,996 / 24,350 vs AC 331 / 1,306 / 2,559. S3 `rq_runs/20261027_ideal_recall/` (25 files); fig source `IB/ARTIFACTS/PAPER_WRITING/fig3_ideal_final/make_fig_retrieval_ideal.py` (L1108, L1110, L1111, L1112).
- **Select-then-answer baseline** (2B decodes selections, 9B reads; eval500, 3 passes): v1 sel100 38.6 / sel200 41.7; 2B ranked 39.13 / 42.67; 9B self-select 46.13 / 47.73; 2B uncapped 47.13 / 49.13 (5.0k/9.6k tokens); summarize sum100 29.67 / sum200 30.87. All below AC 52.9/54.1. S3 `rq_runs/20261027_select_baseline/` (L1122–1131, L1145, L1161–1165).
- **Latency (Fig 3(b), batch 1, k=200, sel200r exact replay)**: text 1,545.5 / select 1,002.2 / AC 808.0 ms TTFT; TPOT 17.7/16.2/16.2; k=100 text 734 / select 513 / AC 386 → "1.2x faster than selection", 1.9x vs text. S3 `rq_runs/20261027_systems_timing/v3/`; summarizer `IB/TMP/PAPER_SYSTEMS/summarize.py` (L1184). §3.5 TTFT (4B setup, IB/TMP/PAPER_SYSTEMS/results.json): text 81, AC 0.6B 48, 1.7B 65, 4B 115 ms, select 172 ms — **4B encoder slower than text** (L1205).
- Reranker probe (eval500 both-gold recall): BM25 @20/100/200 52.0/65.2/72.0, @300 75.2, @500 79.6, @1000 83.4; bge-reranker-v2-m3 top200→20 70.6; Qwen3-VL-Reranker-2B 70.4. `IB/TMP/RERANKER_PROBE/` (L1101, L1102).

**§3.5 training ablations v4 (Qwen3-4B-Instruct-2507, HotpotQA dev200, top20 + both golds injected; final acc)** — data `IB/ARTIFACTS/PAPER_WRITING/sec45_v4/data/`, figs `sec45_v4/figs/` (make_sec45_v4.py), runs S3 `sec45_20261026/runs/<run>/` (L1204, L1100)
- qa-after-recon T_{in,res,kv}_v4: 68.5 / 68.5 / 68.5 (best mid-run in 70.5@3600, kv 69.5@3200) (L1203).
- qa-from-scratch S_in_v4 21.0, S_res_v4 21.5, S_kv_v4 67.0 (L1194, L1196).
- Frozen reader: S_in_v4fr 18.5 (L1196; L1192 says 19.5 — conflict), S_res_v4fr 22.0, **S_kv_v4fr 71.5** (unreported, highest), T_in_v4fr 19.5 (from stalled R), T_res_v4fr 62.0, T_kv_v4fr 66.0 (L1195, L1197, L1201, L1202).
- Smaller encoders (Input Emb., linear-skip head): T_in_e06ls (0.6B) 60.0, T_in_e17ls (1.7B) 66.0 (L1199, L1200).
- **Text + reader-LoRA (T_text_lora)**: u0 53.5 / u1000 72.5 / u2000 72.5 / u3000 71.5 / u4000 70.5 — beats every AC T arm ("FRAMING ISSUE", L1193) (L1198). Untuned 4B text top20i 53.0 lenient (strict 27.0), gold 67.5, none 19.0 (L1107).
- v3 (superseded, archived on S3, not in figure): S_in 21.0, S_res 22.0, S_kv 63.5, T_in 64.0, T_res 44.0, T_kv 69.0 (L1076–1083, L1084).
- §3.5 twin select baselines (4B, dev200 top20i, lenient): sel20i 59.0, sum20i 61.0 (strict 43.5/58.0) (L1146).

**Other paper facts settled in range**: abstract 81.5% / ~100× (corpus; earlier 81.8%/43× was the 2.3M end-of-epoch point) (L1064, L1065); compaction timing 7.42 s vs 1.34 s = 5.5× (was 5.7×) (L1216); multi-agent always 3 solver runs, synthesizer 3 or 5 samples (L1052, L1055); eval500 BM25 both-gold recall @20/100/200 52.0/65.2/72.0 (L1052); DeepMath difficulty 1–10, test range 6–9.5 (L1057). Corpus point: Input+Residuals u2221 (R_deepS rerun) 0.58; MVP corpus figure capped at u2,221, only shared update counts drawn (L961, L968, L1006).

### 2. Training protocols / hyperparameters

**Retrieval chain (→ Arm G)**
- Pair `dense_pow1_tgt48_256_lc3i` (CSAIL /data/scratch/ngom/activation/long_recon_csail): from pairs/lc3i (8,300 updates at 1/10), 4,000 updates cosine 2e-5/2e-5, train mix passage 9,600 / lookup 11,200 / span 11,200; Qwen3.5-2B → Qwen3.5-9B, interleaved, intervention_frequency −1, 184 modules, reader LoRA r128 (L969, L972).
- Run A (SFT): from dense pair, 1/10, native context + question-first + interleaved, 20k items, held 200, enc 5e-5 / reader 2e-5 constant + 250-update cooldown, 2×H200, 2,500 updates (6,361 s); target "Final answer: <answer>" (thinking off) (L972, L973, L980). Ckpt $F/sync/sftd_dense_r10b/.../epoch_002.
- Track A stage 1: KL to same-context references (base 9B reading the student's N×20 list as text, golds injected, thinking off, correct+stopped kept; 24,096/30,000 kept), from Run A epoch_002, --items 20000, enc 5e-5 / reader 2e-5 constant, warm-up 50, 2×H200 (L986, L990, L991). Stage 2: N=1 half / N=2–5 half mix, from tA_s1 epoch_003, warm-up 20, cooldown 300; item count changed 26,000 → 22,000 (L993, L995) but final job 2744629 ran **2,000 updates** (L1000) — undocumented change. Final tA_s2 epoch_004 (S3 `rq_runs/20261021/trackA/tA_s2_epoch004/`) (L1001, L1002).
- **Arm G (paper model)**: from tA_s2 epoch_004; KL toward gold references (base 9B reading golds alone, thinking off; 5,841/7,000 real + 163/200 held kept); student input BM25 top-400 as 20×20, no injected golds; questions hotpot_train_native rest[70000:77000]; 4,000 items / 500 updates (8 items/update), enc 5e-5 / reader 2e-5, warm-up 20, cooldown 100, held 64, ckpt 250; 2× RTX PRO 6000 DDP mb2 ≈36 s/update (L1007, L1008, L1010, L1012, L1013). Arm S (same-context 400-passage refs, 4,086/6,125 kept) same recipe, worse (dev 20×20 52.0) (L1015, L1047).
- **Undocumented / ad hoc**: encoder LR 5e-5 throughout vs header rule "rates 2e-5 / 2e-5 (never 5e-5…)" (L97); rq_train defaulted to cosine though canon is constant + cooldown (user, L965); Arm G's 500-update count chosen ad hoc; LR checks (625 updates, 1 GPU) showed enc 2e-5/1e-4, reader 1e-5/5e-5 all within ≈.15 KL → "hyperparameters are not the limit" (L1000); a resumed run saves epoch_K+1 (L996).
- Eval protocol: dev200 1 pass / eval500 3 passes (sampled), question-first, thinking off, 1/10, max_new 12,288 = text cap (user: "don't cap us at 4096, if baseline is not capped either", pre-range); lenient scoring = finished reply w/o answer line scores last non-empty line if ≤10 words (L959); text arms vLLM bf16 KV; text arms presence penalty 1.5 vs AC none (handoff).
- Run B (from scratch, enc 1e-4 cosine / reader 2e-5): failed — encoder starvation, normal = swapped p_exact .2377/.2378 (L982, L983). Track B (reconstruction from scratch on MuSiQue/2Wiki/MS MARCO 15,200 q each) only opened the channel after a 750-update gold-only warm-up (agent's addition) then 1×20; endpoint ≈2–5 pts under Track A (L989, L995, L998, L1026).

**§3.5 v4 (paper Fig 7)**
- Reader Qwen3-4B-Instruct-2507 + LoRA r128; encoder = 4B copy with own adapter; modes Input Embeddings / Hidden Residuals every 4th layer / All-Layer KVs; ratio 1/10; R = FineWeb-Edu (Wikipedia URLs removed) reconstruction; task = HotpotQA both golds injected in top-20 BM25 (L1068).
- v4: R 2,000 updates (16,000 passages), S/T 4,000 (32,000 questions); lr 2e-5/2e-5, 8 examples/update, warm-up 50, cosine, no-context penalty 1.0, rank 128, R ckpt 200 mb2, T ckpt/eval 400 mb4 (L1084, L1121); penalty 1.0 on recon per user (all earlier recon had it) (L1071). Frozen reader = reader lr exactly 0, no penalty, never 1e-12 (L1098).
- **Arm differences**: small encoders use encoder lr 1e-4 + linear-skip (least-squares) row head while 4B arms use 2e-5 + identity skip (L1141, L1170, L1174); text+LoRA has **no no-context penalty** (unlike AC), lr 2e-5, cosine to 0.1×, same 32,000 items in same order, held NLL on dev200 (the eval set) (`IB/TMP/PAPER_SEC45/text_lora/text_lora_train.py` docstring; L1187); AC gen cap 128 vs text 2,048 (handoff); frozen-reader Input Emb. R stalls at held 2.30 at both enc 2e-5 and 1e-4 (L1160, L1200).
- v3 sizes (P0-derived): R 1,524×8 = 12,192 passages, S/T 1,992×8 = 15,936; v3 R truncated at step 1143 (75%) after CSAIL crash and used as R final (L1073, L1075).
- Kit: v5 resumable bundle `sec45_code_v5.tar.gz` (continuity test PASS: resume within run-to-run noise), v6 adds SEC45_ENCODER, v7 adds SEC45_LINEAR_SKIP (S3 `sec45_20261026/code/`) (L1087, L1097, L1121, L1169).
- Appendix framing: hyperparameters "designed and searched by an autonomous coding agent", "values reported as used, not exhaustively tuned"; per-phase final-run settings only (L1217, L1219).

### 3. Durable lessons

- Setup cost: per-part side-token counting was CPU-bound (47 min setup on 9,897 items); fixed with memoized + batched token counts (prefetch_side_token_counts) (L960).
- CSAIL `sbatch --export=NONE` → "user env retrieval failed" hold; use `--export=NIL` but then set PATH explicitly (g++/ld for TileLang kernels) (L962, L964). `--export=NIL,SEC45_ALLOW_INVALID_INIT=1` got jobs cancelled by root (L1075).
- pgrep/pkill -f patterns matched their own shell three times → look up PIDs in one call, kill by explicit PID in another (L987, L1001, L1046). `$A` expanded locally in double-quoted remote command broke a sweep (L987).
- New EC2 sky nodes may get old image uv-de16520b51ef997e without bm25s → run `uv sync --frozen` in ~/sky_workdir before launching; use ~/sky_workdir/.venv/bin/python under sky exec (peft not found via uv run) (L1102, L1187, L1188).
- Spot g7e terminated without notice (eu-central-1a "no-capacity"); keep checkpoints off-node (L995). RTX PRO 6000: fp8 KV needs FlashInfer → use bf16 KV (`--kv-cache-dtype auto`) (L1008, L1011). ORCD vLLM: `MODULEPATH: unbound variable` under set -u; VLLM_USE_FLASHINFER_SAMPLER=0 + cuda/13.0.1 (L1011, L1013).
- ORCD mit_normal_gpu 2 GPUs/user, 6 h; 2-GPU H200 start estimated days out; preemptable RTX 1-GPU starts at once (L1006, L1012). CSAIL GH200 often held by one 4-node job (L1086); CSAIL /data/scratch/ngom over quota → use /data/scratch-fast/ngom (L1071, L1072); CSAIL/ORCD have no S3 creds — container relay (user: never forbade creds there, L1085) (L1070).
- Speeds: 2×H200 SFT ≈2.0–2.4 s/update; 20×20 KL ≈18–20 s (H200) vs ≈36 s (2×RTX) (L973, L1013); eval batch 100 on GH200 OOMs (L1006); long sampled replies to the 12,288 cap stall eval batches → shard evals across GPUs, seed with done rows (L1004, L1044).
- Thinking off + "Final answer:" target: only 3 prefix tokens; held_nll diluted ≈2× → use answer_nll / p_exact (kit_s13) (L975). rq_diag swapped/q_blind/no_parts variants detect a dead channel (L981, L983).
- From-scratch encoders starve (grad norm collapse ≈40×) unless warmed up on gold-only reconstruction (L982, L997, L998). Smaller encoder with random-FFN row head: reader ignores rows (held stuck 2.276) — linear-skip head fixes it (L1158, L1171).
- Paper build: Tectonic fell back to Latin Modern (no bold) → use pdfLaTeX/latexmk via TinyTeX (`IB/ARTIFACTS/PAPER_WRITING/build_pdf.sh`) (L1060). transformers 5 apply_chat_template(tokenize=True) returns a non-list (L1107).
- Container disk filled twice (100%); leaked VS Code helpers (944, ≈16 GB RSS); archive at S3 `container_archive_20261024/jobtmp_6f1c1373/` (L1051, L1193).

### 4. User decisions and preferences

- "Stop the hard run. Go with dense checkpoint. 1/10th is good. Use 2xgpus instead." (L971). Canon schedule is constant + cooldown, not cosine (L965). Golds injected: yes (L985).
- "Arm G seems to be good enough for our MVP"; "As long as we don't go below text, this is completely fine" (L1032, L1036); frontier figure: avoid log scale, only top 200, no inset, no ours-before-KL; AC top-20 point removed (L1033, L1042).
- Retrieval: no stronger-retriever probe ("redundant"); ideal-recall framing: "to show complementarity with more sophisticated retrieval we run ideal recall experiments; ACs still improve retrieval workloads even with rerankers + dense embeddings" (L1105, L1106); AC "complementary to reranking, not in competition" (L1103).
- Frozen-reader arms lr exactly 0; frozen Input Emb. at encoder 1e-4 "for fairness" (L1098, L1188). Linear-skip start accepted: "not a fundamental change, so it won't affect the story of the paper" (L1174).
- "prioritize spot" = create spot first; don't shut spot nodes; tear down idle on-demand only after S3 (L1185).
- Writing: keep their voice, prefer simplicity (L1225); report body length excluding references (L1183); presence penalty disclosed in appendix only; test-set-independence claim dropped (false for corpus/compaction/multi-agent) (L1216, L1218); declined Fig 5 caption / compaction-timing provenance / No Context intro fixes (L1227). Select→text baseline "KEEP for now, but a good candidate for removal" (header L80).
- **Never resolved**: Fig 7c framing given text+LoRA ≥ AC (L1193, L1203; paper shows "Text (+ LoRA)", L1206); abstract corpus 81.8% vs §4.2 81.5% resolved to 81.5 (L1065); stopping the 30-min cron asked (L1214).

### 5. Rerun / follow-up candidates and weaknesses

- **Text + reader-LoRA on main workloads** (only exists for §3.5; it beat AC 70.5–72.5 vs 68.5) — and match the no-context penalty / held-set choice when doing it (L1198, text_lora_train.py).
- **S_kv_v4fr 71.5** (frozen reader, KVs, from scratch, 1 seed, n=200) above every reported §3.5 number (L1201).
- All §3.5 numbers are 1 seed, dev200 (±3.5 noise at n=200, L1000); v4 picks final not best (in 70.5@3600) (L1203).
- Arm G dev200 +4 over text (60.0 vs 54.0, sign-test p .185) shrank to 53.6 vs 53.47 on eval500 (L1028, L1035). Ideal recall: AC below text at k=100/200 (63.67 vs 64.07; 62.87 vs 63.80) (L1111).
- Retrieval encoder lr 5e-5 and Track A stage-2 length (26k→22k items→2,000 updates) inconsistent/undocumented (L993, L995, L1000); Arm G training is 3 stacked stages (Run A SFT → KL stage 1 → stage 2 → gold KL) developed ad hoc — a clean single recipe is a rerun target (L1219).
- Small-encoder arms differ from 4B arms in lr and head init (L1170); frozen-reader Input Emb. R stalled so T_in_v4fr is effectively from scratch (L1160). R_in_v4fre4 S3 copy overwritten by node l run (held 2.305 vs CSAIL 2.294) (L1204).
- Latency: 4B encoder slower than text (115 vs 81 ms) — §3.5 claim must be qualified (L1205); systems timing used select prompts from a replay only for k=200/100 main setup (L1178, L1184).
- Arm S (same-context refs) underperformed; Track B ≈2–5 pts below Track A (consistent direction, not significant) (L1026, L1047).
- Appendix audit mismatches reported but left: multi-agent targets 27B only, AC retrieval decode without presence penalty (L1215, L1216).

