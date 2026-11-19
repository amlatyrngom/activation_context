# Math synthesis (§3.4) clean recipe: runs log (from 2026-11-17)

The recipe (CLEAN_RECIPES_PLAN §2, §4 and the 2026-11-17 math decision), one linear chain, nothing picked on the test set:

1. **Reconstruction.** Start from the shared FineWeb pair `R_res_rq9fr` (2B encoder → frozen Qwen3.5-9B reader, Hidden Residuals),
   then one pass of **long-context reconstruction on the math training runs** (`M_lcrec`), as compaction did (`compaction/stage1/LCREC.md`).
2. **Stage 1, KL**, one model per ratio (`M_s1_r10`, `M_s1_r5`): reader frozen, encoder 5e-5, warm-up 50, cosine, batch 8, one pass.
   Student = the frozen 9B reading the AC parts of the three solver runs (+ the question and results); teacher = the untuned 9B reading
   the three runs as text (the teacher test's "9B, full runs" arm, same prompt); targets = the teacher's own sampled syntheses.
3. **Stage 2, plain cross-entropy** (`M_s2_r10`, `M_s2_r5`) on the teacher's shortest correct syntheses (of k = 3 per problem, the
   shortest correct one plus any within 1.3×), from the stage-1 final, same trainer settings, one pass.
4. **Hedge** (reader LoRA 2e-5) only if the frozen arm clearly fails on dev; its fair baseline is a text reader LoRA on the same targets.

Code: `math/recipe/`. Data (local): `IB/TMP/PRE_REBUTTAL/math_recipe/`. S3: `pre_rebuttal_20261109/math/recipe/`. Every number below
was read from the file named next to it.

## Choices (and why)

| Choice | What | Why |
|---|---|---|
| Ratios | **1/10** (priority) and **1/5**, one model each | PAPER_CANON §3.4: the paper's M1 used ratio 0.1 only, so 1/10 is the paper's setting; 1/5 is the user's second default. |
| What the encoder reads | each solver run as **the same text the teacher reads** (`render_run`, 40k rule), as one user message: `ac_part([{"role": "user", "content": run_text}], AC, r)` | The student's prompt is then the teacher's prompt with each run's text replaced by its part (asserted per item), and the form is the FineWeb warm-up's passage form. rows = clamp(ceil(encoder tokens × r), 2, 4,096) per run (the paper's "ratio 0.1"). |
| Student prompt | the teacher-test full-runs prompt with each run replaced by its part (same instruction text, same "Final answer: <result>" lines) | Teacher and student differ only in how the runs are represented. |
| Long-context reconstruction | one item per training run (1,500 × 3), r ∈ {1/10, 1/5} drawn per run (seeded), warm-up reconstruction prompt, target = the run text, one pass; held panel = 50 **dev** runs × 2 sizes | Runs are long (rendered: median ≈2.6k, max ≈39k tokens per run) vs FineWeb's 1–2k; compaction showed the FineWeb-only start fails on long inputs. One shared phase for both ratios, as compaction's per-ratio runs share `R_cmp_lcrec`. Held panel from dev so no training material is held out. |
| Teacher samples | 1,500 train problems × seeds 0, 1, 2, plain vLLM, the teacher-test settings (T 1.0, top-p 0.95, top-k 20, min-p 0, presence 1.5, max_tokens 64,000, max_model_len 160,000, bf16 KV); token ids recorded | One generation serves stage 1 (every stopped sample) and stage 2 (selection). |
| Trainer limits | items whose teacher view (prompt + sample) passes 72,000 tokens are dropped and counted | The trainer's guard (compaction's too); 17 of 1,500 training prompts alone pass 72k. |
| Dev evaluation | **one code path for every arm**: the harness's residual vLLM engine and an Agent per request (no tools, one turn); text arms send no part; text prompts asserted token-equal to the teacher test's chat text | The user's fairness rule (same decoding, caps, parser, code path). Scorer = the teacher-test scorer (lenient parse, math-verify). |
| Dev proxy | dev problems' seed-0 teacher synthesis, teacher-forced: NLL under AC (r10 / r5) vs full runs vs results-only; gap closed = (NLL_results − NLL_AC) / (NLL_results − NLL_full) | The cheap in-training panel, like compaction's dev-75 gap closed. |

## Splits (`math_recipe/data/splits.json`, `build_data.py`)

- Train: 1,500 problems (1,000 `tqa_m1_train_v2` at ≥ 8.5 + 500 `tqa_train_solve_v1` DeepMath at 6–9), seeds 0–2 each; k = 0/1/2/3:
  213 / 106 / 237 / 944; single solver 75.82 %, any of 3 85.80 %; full prompt tokens median 11,378, p90 38,949, max 102,333
  (17 over 72k).
- Dev: `tqa_dev_solve_v1`, 200 problems (difficulty 6–9), seeds 0–2; k = 27 / 12 / 28 / 133; single solver 77.83 %, any of 3
  86.50 %; full prompt tokens median 17,325, max 76,799 (4 over 72k).
- Test: dmext 300 + dmhard 300 (task ids and question texts read for the overlap check only).
- Overlap, asserted: 0 shared task ids and 0 shared normalised question texts for every pair of train / dev / dmext / dmhard; 0
  duplicate questions inside a split. The 200 teacher-test problems (all in train) give the same full-prompt chat text (sha256) as
  the teacher test's `prompts.jsonl`: 200 / 200.

## Log

- 2026-11-17 ~21:30Z setup on CPU: splits + prompts (`build_data.py`), teacher sampler (`synth_gen.py`), packet builders
  (`build_packets.py`: lcrec / stage1 / stage2 / panel), proxy (`math_proxy_eval.py`), dev eval (`math_eval.py`, `make_eval_inputs.py`),
  scorer (`score_math.py`), cluster job (`math_job.sh`, `math_gh200_job.sh`), staging (`math_stage.sh`), bundles `math_code_v1/v2`.
- CSAIL at 21:25Z: quanta-h200-1 7/7 GPUs and quanta-gh200-[1-4] 4/4 held by our group's quanta-main jobs (no preemptible job to
  take). ORCD in maintenance until ~02:00Z 2026-11-18; jobs submitted there wait for it.
- Submitted on ORCD (pending until the maintenance ends): `mgen_tr0` 29462890 / `mgen_tr1` 29462916 (teacher samples, train, two
  halves), `mgen_dev` 29462917 (dev, seed 0: the proxy panel's targets), `meval_base_full` 29463530 / `meval_base_results` 29463532
  (dev text baselines through math_eval.py, seeds 0–2).
- 21:42Z the first ORCD jobs started early (maintenance over) and the two train generation halves failed in vLLM's start-up: the
  FlashInfer top-k/top-p sampler JIT-compiles with nvcc, which the node lacks (`logs/mgen_tr0_29462890.out`: "Could not find nvcc").
  Fix (`math_job.sh`): `VLLM_USE_FLASHINFER_SAMPLER=0` for every generation on every cluster (vLLM's PyTorch sampler, same sampling
  parameters; all dev arms and the teacher samples share it) + `module load cuda/13.0.1` on ORCD, the 2026-11-09 ORCD precedent.
  The three jobs that had just started with the old script (29462917, 29463530, 29463532) were cancelled by id and resubmitted.
- Long-context reconstruction packet `m_lcrec_v1` (`packets/m_lcrec_v1/build.json`): 4,483 items of 4,495 requested (5 runs left
  out: the run text holds `</think>`; 12 dropped: student history over 72,000 tokens), r = 0.1 for 2,307 / 0.2 for 2,176, target
  tokens median 2,935 / p90 13,501 / max 35,529, 89 parts clamped at 4,096 rows, **560 updates per pass**; held panel 100 items (50 dev
  runs × r10, r5). Backed up to S3 `math/recipe/packets/m_lcrec_v1/`.
- 21:45–21:47Z submitted on ORCD (mit_preemptable, 1 RTX PRO 6000 each, --requeue, resumable): `M_lcrec` 29464051 (reconstruction,
  sft, reader LR 0, panel every 50, recovery slot every 100); teacher samples `mgen_tr0..3` 29464426 / 29464428 / 29464429 /
  29464436 (train, 4 shards × seeds 0–2), `mgen_dev` 29464437 (dev seed 0, the proxy targets); dev text baselines
  `meval_base_full` 29464438, `meval_base_results` 29464439 (math_eval.py, seeds 0–2). Relay to S3: `math_relay.sh orcd` (every 20 min).
- 22:04Z ORCD started `M_lcrec` and `mgen_tr0..2` (per-user cap 4 GPUs: tr3, dev and the baselines wait). Coordinator: queue the
  long-pole jobs on CSAIL quanta-main as second copies, first to start wins (`math_arbiter.sh`; in-job claim of the run folder /
  gen shard per cluster; separate file systems and S3 prefixes per cluster). CSAIL lesson: `sbatch --export=NIL,VAR=...` jobs are
  held ("user env retrieval failed requeued held") or cancelled by root (test 3099870); plain `--export=NIL` works, so the job
  scripts take `MATH_*=value` as leading arguments. Held copies 3099708 / 3099732 / 3099854 cancelled by id; tr0..2 CSAIL copies
  cancelled (ORCD copies running).
- 22:38Z the GH200 copy of `M_lcrec` (3099930, quanta-gh200-2, fresh aarch64 venv) started training; ORCD's copy (started 22:04Z)
  was at 101 / 560 at the same s/update, so ORCD won and 3099930 was cancelled at 6 / 560.
- `M_lcrec` panel (`runs/M_lcrec/logs/train_29464051.log`, held nll per token r10 / r5): u0 0.2401 / 0.3170, u51 0.0222 / 0.0069,
  u101 0.0212 / 0.0058 (token accuracy 0.9391 → 0.9954).
- 23:10Z–00:28Z teacher samples done (`gen_in/teacher_train/`, 4 × 1,125 = 4,500, all finish stop; `gen_in/teacher_dev/`, 200, seed 0).
  Dev proxy panel `m_dev_panel_v1` (`panel.json`): 196 problems × {r10, r5, none} (4 dropped: teacher view over 72k tokens).
- 00:46Z proxy reference (`proxyref_in/proxy_ref_mdev.json`, `R_res_rq9fr_final`, 588 items): NLL per token full 0.3895, results-only
  0.5623, AC r10 0.5912, AC r5 0.6283 → gap closed r10 −0.167, r5 −0.382 (the FineWeb start is worse than results-only).
- 00:37Z stage-1 packets (`packets/m_s1_r{10,5}_v1/build.json`): 4,431 items of 4,500 each (69 dropped: student history over 72,000
  tokens), 1,478 problems, target tokens median 3,332 / p90 6,889 / max 27,032, **553 updates**; r5: 389 parts clamped at 4,096 rows.
  (Session note: the host-side watcher did not wake this agent; ~4.5 h idle until 05:02Z.)
- 02:05Z `M_lcrec` done (ORCD 29464051; 560 updates, 25.57 s effective per update, peak 63.3 GB). Held panel (100 dev runs) nll per token:

| update | r10 | r5 | all | token accuracy |
|---|---|---|---|---|
| 0 | 0.2401 | 0.3170 | 0.2785 | 0.9391 |
| 101 | 0.0212 | 0.0058 | 0.0135 | 0.9954 |
| 301 | 0.0129 | 0.0023 | 0.0076 | 0.9974 |
| 560 | 0.0074 | 0.0010 | 0.0042 | 0.9986 |

  Final = `init/R_math_lcrec_final` (host copy sha256-checked against ORCD; S3 `math/recipe/init/`), staged on ORCD and CSAIL.
- Dev text baselines (`scores/dev_baselines.json`, math_eval.py, 200 problems × seeds 0–2, the untuned 9B):

| arm | solved % | per seed | generated tokens median / mean | length stops |
|---|---|---|---|---|
| results-only | 81.83 | 81.5 / 82.0 / 82.0 | 13,983 / 16,112 | 3 |
| full runs | 82.33 | 83.0 / 81.5 / 82.5 | 3,021 / 3,480 | 0 |
| single solver | 77.83 | | | |
| majority of 3 | 81.00 | | | |
| any of 3 | 86.50 | | | |

  full − results +0.50 (95 % CI −1.83 to +2.67, sign test p 0.69).
- 05:1xZ stage 1 submitted from `R_math_lcrec_final` (KL, reader frozen, encoder 5e-5, warm-up 50, cosine, batch 8, one pass = 553
  updates; dev panel every 70 updates, recovery slot every 50): ORCD `M_s1_r10` 29502600, `M_s1_r5` 29502601; CSAIL quanta-main
  H200 + GH200 copies (arbiter: first to start wins). `mproxy_lcrec` 29502603 = the gap closed of `M_lcrec` final.
- Stage 1 started 05:20Z (`M_s1_r10`, ORCD 29502600) and 05:27Z (`M_s1_r5`, ORCD 29502601); CSAIL copies cancelled by the arbiter.
  `M_lcrec` final dev proxy (`runs/M_lcrec/proxy/final.json`, math_proxy_eval): NLL per token r10 0.7136 / r5 0.7465, gap closed
  **−0.876 / −1.066** (FineWeb start: −0.167 / −0.382): reconstruction on the runs alone hurt the synthesis proxy; the stage-1 u0
  panels give the same numbers. Stage-1 panels (dev 196, NLL per token; gap closed = (0.5623 − NLL) / (0.5623 − 0.3895)):

| run | update | AC NLL | KL | token accuracy | gap closed |
|---|---|---|---|---|---|
| M_s1_r10 | 0 | 0.7137 | 0.2780 | 0.7919 | −0.876 |
| M_s1_r10 | 71 | 0.4384 | 0.0424 | 0.8492 | 0.717 |
| M_s1_r5 | 0 | 0.7465 | 0.3068 | 0.7867 | −1.066 |
- Stage-2 packets (`packets/m_s2_r{10,5}_v1/build.json`; built on CPU from the same 4,500 teacher samples, before stage 1 ends) match
  the coordinator's stage-2 spec: plain CE on the selected teacher continuations, reader frozen, the stage-1 trainer settings, one pass;
  per training problem only correct samples (finish stop, lenient parse, math-verify = the teacher-test scorer), the shortest plus every
  other within 1.3× its generated tokens; problems without a correct sample dropped; only the 1,500 training problems are read (dev and
  test excluded by the asserted split disjointness). One addition: the trainer's 72,000-token guard drops 26 selected items (14 problems).
  Counts (same for r10 and r5): 3,661 correct samples; 1,265 problems with a correct sample (kept 1 / 2 / 3: 535 / 470 / 260) → 2,255
  selected → **2,229 items, 1,251 problems, 278 updates**, target tokens median **2,800** (p90 5,496, max 22,783). Staged on ORCD and
  CSAIL; S3 `math/recipe/packets/`. Each ratio's stage 2 starts from its own stage-1 final.
- 07:32Z `M_s1_r5` was preempted on ORCD and resumed (requeue) from its update-50 recovery slot (`sync/M_s1_r5.resumes.jsonl`).
- 08:4xZ `M_s1_r5` (29502601) cancelled after two preemptions that both resumed from its u50 slot. Each resume first recomputes an
  unlogged ~18 min baseline panel, then the u71 panel, so ~67 min between preemptions never reached the u100 slot. It was restarted
  fresh as **`M_s1_r5b`** (ORCD 29519172 + CSAIL copies): identical except a recovery slot every 25 updates. Checked against
  `sec45_train.py` (coordinator's question): a resume refuses any argument change except `--seconds` / `--resume-if-possible`, so the
  slot interval cannot change on a resume. There is no launch flag to skip the resume baseline panel (`resume_baseline`, always
  computed and not logged). `--held-panel N` exists but is fixed at the first attempt like every other argument. Both would be code
  changes, so neither was made. Every new run (stage 2, the evals) uses a slot every 25 updates from its first attempt.
- 09:35Z `M_s1_r5b` (29519172) was preempted ~50 min after its start, before its first slot (u0 panel ≈ 18 min + 25 updates), so it
  restarted at 0; cancelled. **`M_s1_r5c`** (ORCD 29522328 + CSAIL H200/GH200 copies): the same training, recovery slot every 20,
  in-training panel = `m_dev_panel64_v1` (64 of the 196 dev panel problems, `random.Random("20261117:panel64")`; ≈ 6 min). Launch
  inputs only, no code change. Its final proxy uses all 196 (`m_dev_panel_v1`), like r10.
- `M_s1_r10` panels: u141 NLL 0.4297 / KL 0.0352 / gap closed 0.767; u211 0.4272 / 0.0323 / 0.782.
- 11:4xZ chain queued on ORCD with `--dependency=afterok` (each stage-1 job ends with its final proxy on all 196 dev problems):
  `M_s1_r10` 29502600 → dev eval `meval_s1_r10` 29531213 (ac_r10, seeds 0–2) + stage 2 `M_s2_r10` 29531284 (sft, 278 updates, slot every
  20, in-training panel m_dev_panel64_v1) → `meval_s2_r10`; `M_s1_r5c` 29522328 → `meval_s1_r5` 29531292 + `M_s2_r5` 29531295 →
  `meval_s2_r5`. Stage 2 reads `init/M_s1_r{10,5c}_final` = links to the stage-1 run's `ckpt/final` (only read after afterok).
  Panels: r10 u281 gap closed 0.797, u351 0.801 (196); r5c u71 0.782 (64-problem subset vs its own references
  `scores/panel64_reference.json`: results-only 0.5659, full 0.3776).
- 15:27Z **`M_s1_r10` done** (ORCD 29502600; 553 updates, 64.1 s effective / 42.5 s median per update, peak 60.4 GB; no preemption).
  Final dev proxy (`runs/M_s1_r10/proxy/final.json`, all 196): NLL per token 0.4213, KL 0.0275, **gap closed 0.816** (FineWeb start
  −0.167, `M_lcrec` −0.876). Panels (196): u71 0.717, u141 0.767, u211 0.782, u281 0.797, u351 0.801, u421 0.810, u491 0.815,
  u553 0.816. `M_s1_r5c` (64 subset): u141 0.824, u211 0.846, u281 0.853.
- 15:39Z `M_s2_r10` started (ORCD 29531284, afterok). The dev eval `meval_s1_r10` (29531213) failed: CUDA OOM in the AC encode. The
  harness lowers the engine share by 0.1 for the AC side model (0.9 → 0.8 = 76 GB of 95), and the 2B encoder needed more than the
  remaining ~18 GB. Fix: `math_eval.py --gpu-memory-utilization` (bundle `math_code_v3`, `MATH_GPU_UTIL=0.75` → 0.65 effective). That is
  the engine's memory share only (KV-cache room); decoding, caps, parser and code path are unchanged from the text baselines (which ran
  at 0.8 effective). All four dev evals resubmitted on v3: `meval_s1_r10` 29547300, `meval_s1_r5` 29547301 (afterok r5c),
  `meval_s2_r10` 29547302 (afterok 29531284), `meval_s2_r5` 29547303 (afterok 29531295); the old 29531292/29531298/29531302 cancelled.
- 16:10Z the v3 retry of `meval_s1_r10` (29547300) aborted ("malloc(): unsorted double linked list corrupted"). The log shows 7
  concurrent "side_base - Moving model to cuda": the 96 request threads all triggered the harness's lazy first load of the 2B encoder at
  once. That race was also the real cause of the first OOM (18.4 GB = several encoder copies). Fix (`math_eval.py`, bundle
  `math_code_v4`): an AC arm runs its first request alone, then the thread pool. Once loaded, `model_to_device` returns early, so later
  requests do not reload. All dev evals resubmitted on v4 (`MATH_GPU_UTIL=0.75` kept): `meval_s1_r10` ORCD 29550751 (running 16:46Z) +
  CSAIL H200 copy 3117360 (`init/M_s1_r10_final` relayed ORCD → host → CSAIL, sha256-checked; arbiter: ORCD started first),
  `meval_s1_r5` 29550752, `meval_s2_r10` 29550754, `meval_s2_r5` 29550755 (afterok as before); v3 29547301–03 cancelled.
- Panels: `M_s1_r5c` u351 (64) NLL 0.4035, KL 0.0227, gap closed 0.862. `M_s2_r10` (stage 2, sft; 64-subset panel scored on the
  teacher's seed-0 dev syntheses) u0 NLL 0.4123 (gap 0.818), u70 0.4015 (gap 0.875).
- 17:21–17:23Z **ORCD scratch quota full** (per-user ≈1.0 TB, invisible to df; ROOT_CANON "Quotas you cannot see"): `M_s1_r5c` died
  writing its u400 slot, `M_s2_r10` its u140 slot, `meval_s1_r10` 29552238 at vLLM's compile-cache write; the afterok dependents
  were dropped. Cleanup (rm of my own items): every per-job JIT cache under `math1/cache/`, the abandoned `runs/M_s1_r5`,
  `runs/M_s1_r5b` (slots on S3 via the relay). Usage job 29554981 (`orcd_usage_job.sh`, CPU sbatch): scratch total **796 GB**
  (activation 731, hf 41, venvs 9) → ≈200 GB headroom; published slots intact: `M_s1_r5c` u380 and `M_s2_r10` u120 (record =
  directory, every .pt / .safetensors loads). From now on the jobs keep their JIT/compile caches on node-local `/tmp` (removed at
  exit), training included. Resumed with identical arguments: `M_s1_r5c` 29555707, `M_s2_r10` 29555708; `M_s2_r5` 29555709
  (afterok r5c); evals (v4, util 0.75, concurrency 32 for r10 / 24 for r5): `meval_s1_r10` 29555711 (+ CSAIL H200 copy 3117645),
  `meval_s1_r5` 29555721, `meval_s2_r10` 29555727, `meval_s2_r5` 29555736. Earlier eval tries: 29550751 OOMed after 29 rows at
  concurrency 96: the AC rows and deltas of in-flight requests sit on the GPU, hence the concurrency cap.
- 17:58Z `meval_s1_r10` 29555711 OOMed again in the AC encode after 4 more rows (33 in the file; the encoder process held 28.8 GiB
  allocated, the engine 62.29 GiB). 18:28Z all evals were resubmitted on v4 with `MATH_GPU_UTIL=0.55` (0.45 effective) and concurrency 16
  (r10) / 12 (r5): `meval_s1_r10` 29559462 (CSAIL H200 copy 3118211 cancelled by the arbiter, ORCD started first), `meval_s1_r5`
  29559463, `meval_s2_r10` 29559464, `meval_s2_r5` 29559465 (afterok as before). By 19:00Z 29559462 had 337/600 rows and no OOM.
- 18:40Z **leak check** (the coordinator suspected a leak in the encoder process). `math_eval.py` (bundle `math_code_v5`) logs this
  process's CUDA memory after every finished request to `<out>/mem.jsonl`: allocated, reserved and the peak since the last record.
  `--mem-census` adds a gc census of the live CUDA tensors. New knobs: `--encode-batch-tokens` (the AC model's
  `encode_batch_max_tokens`, default 65,536) and `--empty-cache-every`. Two ORCD runs on `M_s1_r10` final, 10 dev problems x 3 seeds,
  max_tokens 1,024, util 0.55, concurrency 16 (`eval/memtest_r10*`, not part of any result):
  - default (29559833): allocated after request 1 5.697 GiB, after request 30 5.890 GiB. In between it rose to 9.4 GiB at most and
    fell back. The census shows the extra was the fp32 rows and deltas of the last encode batch (e.g. (2459, 4096) float32 x 9),
    which the encode-queue thread references until its next batch. Peak 22.17 GiB (one encode batch); reserved grew to 24.72 GiB
    and stayed there (caching allocator).
  - encode budget 32,000 + empty_cache after every request (29559834): allocated 5.697 → 5.721 GiB; at most 14.7 GiB in between;
    peak 18.13 GiB; reserved 15.64 GiB at most, back to 5.93 GiB.
  - Verdict: **no per-request growth** (+0.19 GiB over 30 requests in the default run, ≈6 MiB per request; +0.02 GiB with the fix).
    Memory returns to the baseline once the encodes are done. The OOMs came from per-encode-batch peaks: a padded multi-input batch
    up to 65,536 side tokens, plus the previous batch's GPU outputs still held, plus allocator caching. At 0.65 effective
    utilization this exceeded the ~25 GB the encoder had; at 0.45 effective it fits. The running and queued evals keep v4 at 0.55,
    so every AC arm uses the same settings.
- 19:19Z **dev eval, stage 1 r10** (`meval_s1_r10` 29559462, v4, util 0.55, concurrency 16; 600/600 rows, every finish "stop").
  Scored with `score_math.py` (math-verify, lenient; `scores/dev_s1_r10.json`, and `scores/dev_s1_r10_vs_full.json` with reference full):

  | arm (200 dev problems x 3 seeds) | lenient acc | gen tokens median / mean | prompt tokens max |
  |---|---|---|---|
  | 9B results-only | 81.83% | 13,983 / 16,112 | 496 |
  | 9B full runs (teacher view) | 82.33% | 3,021 / 3,480 | 76,799 |
  | **AC r10, stage 1 (`M_s1_r10`)** | **83.00%** (seeds 83.0 / 83.5 / 82.5) | 3,274 / 3,629 | 7,919 |
  | single solver / majority / any-of-3 | 77.83 / 81.00 / 86.50% | | |

  AC r10 − results +1.17 pts (95% CI −1.00 to +3.33; 18 wins / 11 losses, p 0.26). AC r10 − full +0.67 pts (CI −1.17 to +2.67; 17 / 13,
  p 0.58). AC r10 vs majority 28 / 16 (p 0.096). Gap closed (single → any) 0.596 (full 0.519, results 0.462). The AC arm matches the
  full-runs teacher at about a tenth of its prompt and keeps the teacher's short syntheses (median 3,274 vs results-only 13,983 tokens).
  None of the differences is significant. The frozen arm does not fail, so no reader-LoRA hedge.
- ~20:00Z **`M_s2_r10` done** (ORCD 29555708, 278/278 updates; the resumed segment ran 158 updates in 7,055 s). Final dev proxy
  (`runs/M_s2_r10/proxy/final.json`, all 196): NLL per token 0.4041, KL 0.0486, **gap closed 0.915** (stage 1 `M_s1_r10` 0.816;
  results-only 0.5623, full 0.3895). Its dev eval `meval_s2_r10` 29559464 started on afterok (321/600 rows at 20:35Z).
- 20:49Z **dev eval, stage 2 r10** (`meval_s2_r10` 29559464, v4, util 0.55, concurrency 16; 600/600 rows, all "stop").
  `scores/dev_r10.json`, `dev_r10_vs_full.json` and `dev_r10_s2_vs_s1.json`:

  | arm | lenient acc | gen tokens median / mean |
  |---|---|---|
  | 9B results-only | 81.83% | 13,983 / 16,112 |
  | 9B full runs | 82.33% | 3,021 / 3,480 |
  | AC r10 stage 1 | 83.00% | 3,274 / 3,629 |
  | **AC r10 stage 2** (`M_s2_r10`) | **81.83%** (80.5 / 82.0 / 83.0) | 2,742 / 3,048 |

  Stage 2 − results 0.00 pts (CI −2.50 to +2.67; 15 / 15). Stage 2 − full −0.50 pts (CI −2.33 to +1.50; 11 / 14, p 0.69).
  Stage 2 − stage 1 −1.17 pts (CI −3.00 to +0.67; 9 / 16, p 0.23). Stage 2 shortens the syntheses (median 2,742 vs 3,274 tokens)
  without a significant accuracy change. Its dev proxy improved (0.915 vs 0.816), but dev accuracy did not.
- 21:30Z **goal update** (coordinator, from the user): raise the solve rate on hard problems without a reader LoRA. MVP: AC clearly
  above results-only at ≥9. Two parallel lines:
  - **dev_hard** (`make_dev_hard.py`): `tqa_math_dev_hard_v1`, 180 fresh DeepMath problems (d8.5 80, d9 96, d9.5 4; seed 20261118;
    sample sha256 224ba6de…). 0 shared ids and question texts with train / dev / dmext / dmhard / the teacher test (asserted). Also
    excluded: every DeepMath sample.json, at3, `excluded_tasks_m1`, and 1,456 hard ids scanned in PAPER_TRAJ_QA and the math folders.
    No unused hard problem had solver runs (all 1,063 train ≥8.5 problems are in train), so the 3 × 9B solver runs are being
    generated (`kit/run_tqa_solve_v2.py`, the dmhard setup). They need the Podman sandbox: none on ORCD (probe 29570145), so they run
    on the coordinator's Sky node dvA as sky job 19 (`devhard_launch.sh` / `devhard_solve_node.sh`; S3 mirror every 5 min, resumable).
  - **stage 2 hard8**: 5 more teacher samples (seeds 3–7, same 9B / prompt / sampler / caps) for the 1,063 training problems at ≥8.5
    (`data/tasks_train_ge85.json`: d8.5 791, d9 261, d9.5 10, d10 1). ORCD `mgen_h8_0..3` 29573151 / 29573157 / 29573162 /
    29573164 (bundle `math_code_v6`, nice 1000, two at a time: the r5 chain goes first). Packet `m_s2_hard8_v1` per ratio
    (`build_packets.py stage2 --gen-extra`): ≥8.5 problems select over all 8 samples, the others keep their 3.
- ~21:30Z **`M_s1_r5c` done** (553/553; the resumed segment ran 173 updates in 12,820 s). Final dev proxy (all 196): NLL per token
  0.4117, KL 0.0193, **gap closed 0.871** (r10 stage 1: 0.816). On afterok: `M_s2_r5` 29555709 and `meval_s1_r5` 29559463 started
  ~21:55Z.
- 21:36Z **dev_hard solver runs done** on dvA (sky job 19: 540/540 rows in ~35 min; S3 `math/dev_hard/ROLLOUTS/`). Downloaded into
  `dev_hard/dl_solver/` and built with `build_dev_hard_data.py`: `data/{problems,prompts,eval_inputs}_devhard.jsonl`
  (`devhard_summary.json`). The 0-overlap asserts were re-run on the solver rows. 180 problems, k = 0/1/2/3: 28 / 20 / 28 / 104; single
  solver 71.85%, any-of-3 84.44%. d8.5 (80): single 74.58%, any 85.00%. ≥9 (100): single 69.67%, any 84.00%. Full prompt median 8,786
  tokens, max 89,821.
- 22:00Z dev_hard evals on ORCD (bundle v6, nice 100; seeds 0–2): math_eval `mevh_results` 29574772 and `mevh_full` 29574776
  (R_res_rq9fr pair, as the dev baselines), `mevh_s1_r10` 29574777, `mevh_s2_r10` 29574780, `mevh_s1_r5` 29574856, `mevh_s2_r5`
  29574844 (afterok `M_s2_r5`). AC arms at util 0.55, concurrency 16 (r10) / 12 (r5). Teacher-test path (`synth_gen.py`, plain
  vLLM): `mgenh_tg27` 29574830 (unsloth/Qwen3.8-27B-NVFP4, full) and `mgenh_tg9` 29574828 (9B, full), so 27B − 9B is paired
  within one code path.
- 22:38Z **dev eval, stage 1 r5** (`meval_s1_r5` 29559463, 600/600, all "stop"; `scores/dev_s1_r5.json`, `dev_s1_r5_vs_full.json`):
  **82.67%** (83.0 / 82.5 / 82.5), gen tokens median 3,112 / mean 3,527, prompt max 12,581. r5 − results +0.83 pts (CI −1.17 to
  +3.00; 14 / 9, p 0.40). r5 − full +0.33 pts (CI −1.50 to +2.33; 13 / 11, p 0.84). vs majority 24 / 14 (p 0.14). Gap closed (single →
  any) 0.558. Like r10 stage 1 (83.00), it matches the full-runs teacher on dev.
- ~22:50Z `M_s2_r5` (29555709) was preempted at about u56 and requeued (PENDING, Resources); it resumes from its latest slot.
- 01:30Z **dev_hard results (v1: every arm except 27B and r5 stage 2)**. `devhard_report.py`; per-band score files are `scores/devhard_{all,d8.5,ge9}_vs_{results,full,tg9_full}.json`; the tables are in `scores/devhard_report_v1.md`. Accuracy % (generated tokens median):

  | arm | all (180) | d8.5 (80) | ≥9 (100) | gen median |
  |---|---|---|---|---|
  | single / majority / any-of-3 | 71.85 / 73.33 / 84.44 | 74.58 / 76.25 / 85.00 | 69.67 / 71.00 / 84.00 | |
  | 9B results-only | 76.11 | 80.83 | 72.33 | 18,741 |
  | 9B full runs (math_eval) | 77.78 | 80.83 | 75.33 | 3,384 |
  | 9B full runs (teacher-test path) | 77.41 | 79.58 | 75.67 | 3,496 |
  | AC r10 stage 1 | 76.85 | 80.83 | 73.67 | 3,184 |
  | AC r10 stage 2 | 76.11 | 80.42 | 72.67 | 1,871 |
  | AC r5 stage 1 | 77.59 | 81.25 | 74.67 | 3,209 |

  ≥9, paired diffs (95% CI; W/L): full − results +3.00 (−1.33, +7.67; 21/12, p 0.16). AC r10 s1 − results +1.33 (−2.67, +5.67). AC r10 s2 − results +0.33. AC r5 s1 − results +2.33 (−1.33, +6.00; 17/10, p 0.25). AC r10 s1 − full −1.67 (−5.33, +2.00). AC r5 s1 − full −0.67 (−3.67, +2.00). No difference is significant. On dev_hard ≥9 the AC arms sit between results-only and full runs. Stage 2 (3-sample selection) shortens the syntheses (r10: median 1,871 vs 3,184) and loses about 1 pt at ≥9. The MVP (AC clearly above results-only at ≥9) is not met. 27B full (29589575, ninja fix) and AC r5 s2 are pending. The first two 27B tries failed: HF offline with the model not cached (fetched by 29587304, 22 GB), then ninja missing from PATH.
- ~02:40Z **dev_hard 27B full** (`mgenh_tg27` 29589575, unsloth/Qwen3.8-27B-NVFP4, teacher-test path, 540/540, all "stop";
  `scores/devhard_report_v2.md`): all 79.07% (gen median 2,413), d8.5 82.50%, **≥9 76.33%** (gen median 2,850). Paired vs 9B full
  on the same path (tg9_full): all +1.67 (CI −2.22, +5.56; 37/28, p 0.32), d8.5 +2.92 (−2.50, +8.75; p 0.21), **≥9 +0.67 (−4.67,
  +5.67; 22/20, p 0.88)**. vs results-only: all +2.96 (−0.56, +6.67; 37/21, p 0.048), ≥9 +4.00 (−0.33, +9.00; 23/11, p 0.06).
  Coordinator's bar for 27B RFT targets (27B − 9B full at ≥9: at least +3 with p < 0.1): **not met** (+0.67, p 0.88), so no 27B
  target generation. The 27B is not better than the 9B teacher on dev_hard ≥9; both beat results-only by 3–4 pts there.
- ~03:20Z **`M_s2_r5` done** (278/278; preempted once at about u56 and resumed). Final dev proxy (196): NLL per token 0.3943, KL 0.0408, **gap
  closed 0.972** (r5 stage 1: 0.871; r10 stage 2: 0.915). Its evals `meval_s2_r5` 29559465 (dev) and `mevh_s2_r5` 29574844
  (dev_hard) are now eligible. hard8 teacher samples: 3,942 / 5,315 (shards 0 and 1 done; 2 running; 3 preempted at 543, requeued).
- 04:14Z **dev eval, stage 2 r5** (`meval_s2_r5` 29559465, 600/600, all "stop"; `scores/dev_r5.json`, `dev_r5_vs_full.json`,
  `dev_r5_s2_vs_s1.json`): **81.67%** (81.0 / 82.5 / 81.5), gen median 2,667 / mean 2,918. s2 − results −0.17 (CI −2.33, +2.00; 11 / 12).
  s2 − full −0.67 (−2.50, +1.00; 10 / 14, p 0.54). s2 − s1 −1.00 (−2.33, +0.33; 6 / 12, p 0.24). Same pattern as r10: stage 2
  (3-sample selection) shortens the syntheses and costs about 1 pt (n.s.), although its dev proxy rose to 0.972.
- ~06:10Z **hard8 packets** (all 5,315 seed 3–7 samples landed, sha-checked, S3 `math/recipe/gen_in/teacher_train_s37/`). Built on the host
  with `build_packets.py stage2 --gen-extra` (`packets/<name>/build.json`; SHA256SUMS; S3 `math/recipe/packets/`; staged on ORCD
  `data/packets/`):

  | packet | items (requested) | problems | updates / pass | target median / mean |
  |---|---|---|---|---|
  | `m_s2_r10_v1` (3 samples, baseline) | 2,229 (2,255) | 1,251 | 278 | 2,800 / 3,168.0 |
  | `m_s2_hard8_r10_v1` (shortest rule over 8 samples at ≥8.5) | 3,187 (3,220) | 1,275 | 398 | 2,633 / 3,008.1 |
  | `m_s2_hard8rnd9_r10_v1` (size-matched random correct at ≥9) | 3,187 (3,220) | 1,275 | 398 | 2,786 / 3,226.7 |

  The 33 dropped items have a student history over 72k tokens. Problems with ≥1 correct sample, 3 → 8 samples: ≥9 216 → 225 of 272;
  d8.5 667 → 682 of 791; <8.5 382 of 437 (no extra samples). Random variant (`--random-min-difficulty 9`: k = size of the
  within-1.3× set, drawn without replacement from all correct samples, seeded by task id), ≥9: 225 problems, 608 items, target median
  3,915.5 / mean 4,272.8 vs the shortest rule's 2,656.5 / 3,123.8 on the same problems. `m_s2_hard8_r5_v1` (r 0.2): same 3,187 items
  and target stats. Training (ORCD, bundle v6, init `M_s1_r10_final`, sft, slot every 20): `M_s2h_r10` 29616157 and `M_s2hr_r10`
  29616158; the r5 runs follow after both.
- ~06:30Z **dev_hard AC r5 stage 2** (`mevh_s2_r5` 29574844, 540/540; `scores/devhard_report_v3.md`, every arm): all 75.93% (gen
  median 1,976), d8.5 79.17%, **≥9 73.33%** (gen median 1,512). ≥9 paired: − results +1.00 (CI −2.33, +4.67; 15/12, p 0.70), − full
  −2.00 (−6.33, +2.00; 11/17, p 0.34). r5 stage 1 at ≥9: 74.67. As for r10, stage 2 (3-sample selection) roughly halves the generated
  tokens and loses about 1 pt at ≥9 (n.s.).
- 11:24Z / 11:29Z **`M_s2h_r10` and `M_s2hr_r10` done** (398/398 each, about 4 h 50 min; peak 57.1 GB). u398 held panel (64): NLL per
  token 0.3948 / 0.3942 (u0 0.4123). Final dev proxy (all 196): **gap closed 0.915 (shortest rule) / 0.918 (random at ≥9)**, KL
  0.0511 / 0.0511; 3-sample stage 2 `M_s2_r10`: 0.915, KL 0.0522. Evals running / queued: dev `meval_s2h_r10` 29620426,
  `meval_s2hr_r10` 29620428; dev_hard `mevh_s2h_r10` 29620427, `mevh_s2hr_r10` 29620433. r5 (from `M_s1_r5c_final`, nice 200 so the
  r10 evals start first): `M_s2h_r5` 29633624, `M_s2hr_r5` 29633627; evals afterok: dev 29633628 / 29633639, dev_hard 29633637 /
  29633640 (util 0.55, conc 12). A first r5 submission (29633582–29633609) passed `--nice` after the job script; cancelled by id.
- 12:35Z **hard8 r10 evals** (dev: `scores/dev_r10_s2h_vs_s1.json`, `dev_r10_s2hr_vs_s2h.json`; dev_hard: `scores/devhard_report_v4_r10_hard8.md`,
  `devhard_report.py` with `DEVHARD_REFS=results,full,ac_r10_s1`, which rewrites `scores/devhard_<band>_vs_{results,full}.json` with
  these arms; the v3 tables stay in `devhard_report_v3.md`). Accuracy % (gen median); diffs vs r10 stage 1 (95 % CI; W/L, p):

  | arm | dev (200) | dev_hard all (180) | d8.5 (80) | ≥9 (100) | ≥9 − s1 |
  |---|---|---|---|---|---|
  | results-only | 81.83 (13,983) | 76.11 | 80.83 | 72.33 (19,824) | −1.33 (−5.67, +2.67) |
  | 9B full | 82.33 (3,021) | 77.78 | 80.83 | 75.33 (3,349) | +1.67 (−2.00, +5.67) |
  | AC r10 s1 | 83.00 (3,274) | 76.85 | 80.83 | 73.67 (2,947) | — |
  | AC r10 s2 (3 samples) | 81.83 (2,742) | 76.11 | 80.42 | 72.67 (1,461) | −1.00 (−4.67, +2.33) |
  | AC r10 s2 hard8 shortest | 82.17 (2,644) | 76.67 | 80.00 | 74.00 (1,461) | +0.33 (−4.33, +4.67; 16/15) |
  | AC r10 s2 hard8 random ≥9 | 81.83 (2,699) | 77.41 | 81.67 | 74.00 (1,629) | +0.33 (−3.00, +3.67; 12/11) |

  dev − s1: shortest −0.83 (−2.50, +0.83; 8/13, p 0.38), random −1.17 (−2.83, +0.33; 6/13, p 0.17); random − shortest on dev −0.33
  (−1.67, +1.00; p 0.81). ≥9 − results-only: +1.67 for both (shortest −3.33, +6.67; random −2.67, +6.00; p 0.49 / 0.46). hard8 recovers
  the ~1 pt that the 3-sample stage 2 lost at ≥9 (72.67 → 74.00, back to stage-1 level) while keeping its shorter syntheses (gen median
  1,461 / 1,629 vs s1 2,947), but neither variant beats stage 1 or the full-runs teacher; random vs shortest makes no measurable
  difference. MVP (AC clearly above results-only at ≥9) still not met.
- Coordinator decisions (2026-11-19): the math recipe candidate is recon → stage 1 → stage 2 hard8 (shortest rule); random-at-≥9 is
  dropped (no gain). r5 hard8 (both variants) finishes as queued.
- 12:45Z **rationalization (STaR-style) step** (`recipe/build_rat.py`; `data/rat_select.json`). Problems: train, difficulty ≥ 8.5, ≤ 1
  correct of the 8 ordinary teacher samples: **170** (d8.5 118, d9 52; 0 correct 156, 1 correct 14; solver k 0/1/2: 142/18/10); 79
  have binary golds (Yes 45, True 26, No 8). Prompt: the full-runs prompt with one block before the final rule that states the gold
  answer and asks for a self-contained derivation that never refers to it (`RAT_BLOCK`); prompt tokens median 12,347, max 77,981
  (`data/prompts_rat_v1.jsonl`, sha256 e03eeb0e…, S3 `math/recipe/data/`). Generation `mgen_rat` 29637374 (ORCD, bundle v6, the
  9B with the same sampler / caps, seeds 100–103: 680 samples). Kept = stopped, gold reached (teacher-test scorer), leak filter
  passed (`LEAK`, case-insensitive regex on the whole reply, thinking included), then the shortest rule per problem; binary golds are
  counted but left out of the packet (coordinator). Leak filter on the 8,504 ordinary teacher samples of all ≥ 8.5 train problems:
  675 flagged (7.9 %; "expected answer" 423, "ground truth" 127), so it is conservative. The training items keep the ordinary
  student / teacher views (no gold answer); `build_packets.py stage2 --gen-rat` adds them to the hard8 shortest selection.
- 14:02Z **rationalization samples** (`mgen_rat` 29637374, 680/680 in 1 h 12 min; `gen_in/rat_v1`, sha-checked, S3). `build_rat.py filter
  --exclude-binary` (`gen_in/rat_v1_kept_nb/filter.json`): drops wrong answer 444, leak 199, no stop 1; passing 36 samples on 24
  problems: binary 30 on 20 (23 after the shortest rule; out of the packet), **non-binary 6 on 4 → 5 kept items** (d8.5 2, d9 3;
  generated median 1,914). Leak hits on correct samples: "known: " 167, "(given/known/…/expected) answer" 162, "self-contained" 109,
  "ground truth" 86, others ≤ 54; the same filter flags 165 of the 1,360 ordinary samples of these problems. Even with the gold
  stated, 279 of 364 non-binary samples miss it, 219 of them answering Yes/No/True/False: in 69 of the 91 non-binary-gold problems the
  ordinary teacher samples answer yes/no in ≥ 5 of 8 (gold 'f(x) = x', 'e^{O(N)}', '0' for "determine whether Ext^1 = 0" …). The
  ≥ 8.5 "never solved" set is dominated by yes/no questions with object golds (label / format mismatch). dev_hard: 9 of 180 problems
  have a non-binary gold with ≥ 2 of 3 solver answers yes/no (8 of the 28 k = 0 problems). 5 items add < 1 update to hard8: training
  not recommended (sent to the coordinator with options). Packet `m_s2_hard8rat_r10_v1` built on CPU in case.
- Coordinator: rationalization not trained (option i); `m_s2_hard8rat_r10_v1` stays built, unused (3,191 items incl. the 5
  rationalized, 398 updates; SHA256SUMS, S3). Benchmark scoring unchanged (the label question goes to the user).
- 14:22Z **dev_hard sensitivity: label-mismatch problems excluded** (existing outputs only; `scores/labelmismatch.json`: excluded =
  gold not in {Yes, No, True, False} and ≥ 2 of 3 solver answers in that set → 9 of 180 dev_hard problems (d8.5 4, d9 5; 8 with
  k = 0); the same rule finds **13 of 200 dev problems** (all k = 0; d6 3, d6.5 1, d7 1, d7.5 2, d8 5, d9 1)). Tables:
  `scores/devhard_report_v5.md` (standard, every arm, refs results / full / tg9_full / ac_r10_s1) and
  `scores/devhard_report_v5_sensitivity_labelmismatch_excluded.md` (score files `devhard_lmx_*`). ≥ 9, standard (100) → sensitivity
  (95): results 72.33 → 76.14, 9B full 75.33 → 78.95, tg9 full 75.67 → 79.30, 27B 76.33 → 80.35, AC r10 s1 73.67 → 76.49, r10 s2
  72.67 → 75.79, r10 hard8 74.00 → 77.19, r10 hard8 random 74.00 → 76.84, r5 s1 74.67 → 77.89, r5 s2 73.33 → 76.84. The gaps barely
  move: full − results +3.00 → +2.81 (p 0.16 → 0.22), hard8 − results +1.67 → +1.05, r5 s1 − results +2.33 → +1.75; still nothing
  significant.
