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
