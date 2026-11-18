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
