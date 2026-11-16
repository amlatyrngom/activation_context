# Clean recipes for compaction (§3.3) and math synthesis (§3.4): plan (2026-11-15, draft for the user)

Goal (user, 2026-11-15): the bigger issue is not the missing text + LoRA baseline but our "weirdly arbitrary"
training histories. Every reported AC should come from **one linear recipe with clearly named phases**, like
§3.1 now: reconstruction → stage 1 → stage 2, final checkpoint, test set read once. Removing the reader LoRA
(frozen reader) also removes the need for a text + LoRA baseline, because every arm then reads with the same
untuned model.

- Compaction: frozen reader only (user: "compaction is just representing information more densely").
- Math: frozen reader first, plus some runs with the reader LoRA on, in case the frozen version does not work.

## 1. How the submitted numbers were trained (the history we are replacing)

Sources: `IB/CANON/PAPER_CANON.md` (§3.3, §3.4, "Settings that differ"), `IB/ARTIFACTS/HANDOFF_EC2/STATE_DIGEST_20261109.md`.

### Compaction (§3.3, model H8a: 2B encoder → 9B reader + reader LoRA, Input Embeddings)

A chain of six trainings, each starting from the previous one's checkpoint:

1. A reconstruction pair (log only: no recipe survives).
2. 1,000 updates of long-context reconstruction.
3. "Arm A": 300 runs, 300 updates.
4. H0: + 608 runs whose prefixes were written by the 27B model.
5. H5: + 156 runs written by the 9B (prefixes too short vs the eval, median 7.8k vs 20k).
6. H8a: rejection-sampling fine-tuning (RFT) from H5: for 3,393 prefixes, the untuned 9B continued 3 times
   with the full prefix and the shortest correct continuation (plus up to 2 within 1.3×) became the target;
   6,957 items, 870 updates, 2e-5 / 2e-5, cosine.

Along the way: H3, H5x2, H6, H6L, H6fromH5, H7S, H7M, R5/R10/R20 and H8-full were tried. Inits were
inconsistent (H5/H6 from H0, H7/H8 from H5). **H8a was chosen over the others on eval-200 itself**, seeds
2–4: the same seeds the paper reports. Training ratios were drawn log-uniformly in [1/20, 1/5] (never
exactly the reported 1/10). Only the AC arm had a trained reader.

### Math synthesis (§3.4, model M1: 2B encoder → 9B reader + reader LoRA)

1. A reconstruction phase that the appendix states but nothing backs.
2. `own_ro`: 1,000 updates of blind question answering from rows (MuSiQue 4,556 + DeepMath 3,444 items).
3. A last-mile SFT on 27B syntheses (`synth_sft_v1`, 2 epochs). The final save failed on quota, so update
   245 of 326 became "the model". It was chosen on the set it was reported on.
4. Round 2: arm A (162 updates), then arm B, the union `r2_bu` (9,976 items, 1,234 updates, cut by an 18 h
   time limit).
5. M1 from B: 4,209 items (DeepMath 3,804 + 405 MuSiQue "replay"), 527 updates, 2e-5 / 2e-5, cosine. No
   `recipe.json` survives.

Targets were 27B syntheses kept when correct. The grounding filter was loosened mid-way (reject rate
54–65 % → 10–18 %). The "Results + LoRA" baseline is M1's reader adapter run without rows, not a LoRA trained
on text. AC and text arms ran through different code paths with different generation caps.

### What §3.1 does now (the template)

One pair per phase, each starting from the previous phase's final checkpoint, nothing picked by evaluation:

| Phase | Data | Updates × batch | Loss | Notes |
|---|---|---|---|---|
| Reconstruction | FineWeb (generic text) | 2,000 × 8 | reconstruction | reader frozen, encoder 5e-5 |
| Stage 1 | 44,800 HotpotQA questions, golds in top 20 | 5,600 × 8, one pass | KL to the same reader reading only the golds | reader frozen |
| Stage 2 | the next 19,200 questions, injected 100 / 200 passages | 2,400 × 8, one pass | same KL | reader frozen |

All choices were made on dev1000; the reported numbers are the final checkpoint on test1000, read once.

## 2. The recipe all three workloads share

1. **Reconstruction.** Generic FineWeb, 2,000 × 8, frozen reader, encoder 5e-5, warm-up 50. **One pair per
   write mode, shared across workloads** (all three use the 2B encoder and the 9B reader).
2. **Task distillation.** One pass over the workload's training material, a fixed number of updates, KL to
   a **privileged teacher**: the same frozen reader shown the full text that the AC compresses (§3.1: the
   gold passages). Same trainer settings as §3.1 (encoder 5e-5, reader LR 0, warm-up 50, batch 8).
3. **A second stage only if the reported setting differs from stage 1's data**, as §3.1's injected-k stage
   does. It gets the same explanation and the same loss.
4. **Selection rules.** A dev set separate from the test set for every choice. The final checkpoint is
   reported, never a peak. Test read once with the evaluation settings fixed in advance (same decoding, caps,
   parser and code path for every arm).

## 3. Compaction (frozen reader)

**Constraint found today.** Compaction is evaluated by agent runs in vLLM, and the vLLM path only takes
Input Embeddings (input rows). Residual interventions exist only in the HF path (`activation/harness/
hf_utils.py`, `loaded_model.py`). §3.1's frozen-reader win used Hidden Residuals, and frozen reader + input
rows is the combination we know least about.

Steps:

- **C0. Inventory.** Locate the 9B training runs and prefixes (H7/H8 material, task-level exclusion of
  eval-200 ∪ dev-75 ∪ frozen-100) on S3 / CSAIL / nodes, and the dev-75 set (`prefixes_dev9_75`).
- **C1. Reconstruction pair, Input Embeddings, frozen reader** (`R_in_rq9fr`: as §3.1's but write mode `in`).
- **C2. Cheap probe before any agent run.** Teacher-forced KL / next-token loss of the continuation on
  held-out prefixes: frozen + input rows vs untuned no-context vs full prefix. If frozen input rows barely move
  off no-context, decide between (a) adding residual writes to the vLLM path (engineering) and (b) running
  compaction evaluation through HF (slow at 50k tokens). Neither starts without this evidence.
- **C3. Stage 1 = distillation on the training prefixes.** Teacher: the untuned 9B with the full prefix.
  Student: the same frozen 9B with the AC rows of the prefix. Loss: KL on the continuation tokens of the
  run's own continuation. One pass; one model for both reported ratios (ratio drawn from {1/5, 1/10}).
  This replaces RFT best-of-3 and the six-step chain.
- **C4. Dev check on dev-75** (3 seeds): solved and continuation tokens vs full context, uncapped summary and
  no context. Hill-climb only here, and only on 2–3-point differences.
- **C5. Test: eval-200, seeds 2–4, 50k budget**, read once. Arms: full context, no context, uncapped summary
  (already run: 69.17 % solved, 21.2k tokens), AC 5× and 10×. All arms use the untuned 9B, so the comparison
  is fair by construction.

## 4. Math synthesis (frozen reader first, reader LoRA as a hedge)

- **M0. Splits.** dmext (≥ 6) and dmhard (≥ 8.5) were both used for choices, so they cannot also be a dev
  set. Carve a new dev set from DeepMath (disjoint from training and both test sets, 0 overlap checked), run
  the 9B solvers on it (3 trajectories per problem, 40k cap) and keep dmext + dmhard as the test.
- **M1. Teacher ceiling on dev.** The untuned 9B reading the three full solver trajectories as text (the
  privileged view). This decides the teacher:
  - If its accuracy is clearly above results-only, use KL to it (same recipe as §3.1 and compaction).
  - If not, distill the 27B syntheses kept when correct (SFT), as before but in one phase.
- **M2. Training material.** DeepMath only (drop the MuSiQue replay: the paper reports DeepMath only). One
  fixed packet, one grounding filter, written down.
- **M3. Two arms, same recipe and data, differing only in the reader:** frozen reader (main), and reader LoRA
  on (hedge). If the LoRA arm is the one that works, it needs its fair baseline: a text reader LoRA trained on
  the same targets with results-only input (not M1's adapter run without rows).
- **M4. One evaluation path for every arm**: same harness, generation cap, decoding, `math-verify` parser,
  3 runs per problem on both test buckets.

## 5. Order and resources

1. C0 + M0 (inventories and splits) in parallel; C1 on one GPU.
2. C2 probe → C3 compaction stage 1 → C4 dev → C5 test.
3. M1 teacher ceiling as soon as the dev solvers finish → M3 both arms → M4 test.

Compaction runs first: it is the simpler case and its result decides whether the frozen recipe carries over
to input rows at all. GPUs: ORCD and EC2 spot as for §3.1, one job per GPU, checkpoints to S3.

## 6. Decisions for the user

1. **Compaction loss:** KL to the full-prefix teacher on the original continuations (recommended: same
   explanation as §3.1) vs keeping RFT (shortest correct of 3) as one phase.
2. **If frozen + input rows fails the C2 probe:** add residual writes to vLLM vs evaluate compaction through HF.
3. **Math teacher:** decided by M1 (KL to the 9B reading full trajectories vs SFT on 27B syntheses); confirm
   the rule.
4. **Math data:** DeepMath only, dropping the MuSiQue replay (recommended).

## 7. User answers and new findings (2026-11-15, second round)

- **Math data:** DeepMath only; the MuSiQue replay is dropped (user).
- **Math teacher:** measure both before deciding (user): the untuned 9B and the 27B, each reading the three 9B solver
  trajectories in full as text, scored like the synthesis arms. The 27B is a candidate teacher because the synthesis gap
  (any-of-3 vs single solver) is what the AC should close.
- **Synthesis repeats:** the paper read 3 synthesis runs per problem on the 6–8 bucket (dmext) and 5 on the ≥ 8.5 bucket
  (dmhard) (PAPER_CANON §3.4). The rerun uses one count for every bucket and arm.
- **Write mode vs vLLM:** frozen reader + input rows alone is not acceptable (user: it consistently leaves too much on the
  table). The agent path needs Hidden Residuals at serving time. Findings:
  - HF path: the residual write is one per-request scatter-add at the input of the named reader layers
    (`activation/harness/hf_utils.py`, `install_intervention_hooks`), so it is easy to express anywhere.
  - vLLM 0.28: prompt embeddings already travel per request and per position, are written into a runner buffer whose width
    is a config value (`ModelConfig.get_inputs_embeds_size`), and are hashed for prefix caching. A width check pins them to
    the hidden size (`vllm/renderers/embed_utils.py`). Qwen3-VL's "deepstack" already adds extra per-position vectors to the
    residual stream at chosen decoder layers (`vllm/model_executor/models/qwen3_vl.py`); Qwen3.5's text model does not.
  - Options and ratings are in the chat reply of 2026-11-15; recommended: residual writes inside vLLM (custom model class
    + widened prompt embeddings), time-boxed, with an equivalence test against the HF hooks.

## 8. Started 2026-11-15 ~21:30Z (user: go; CSAIL reopened; LR not hill-climbed)

- Starting pair for both workloads: §3.1's `R_res_rq9fr` (FineWeb reconstruction, 2,000 × 8, frozen reader, 2B → 9B, Hidden
  Residuals; S3 `pre_rebuttal_20261109/sec35/runs/R_res_rq9fr/`), described as one shared warm-up.
- Learning rate is not hill-climbed (user; evidence: §3.1 stage 2 at 1/4 LR ended level, §3.5 encoder 1e-4 trailed 2e-5):
  encoder 5e-5, reader frozen (math LoRA arm: reader 2e-5 as before).
- Parallel streams: vLLM residual prototype (agent, own worktree, 1 EC2 spot GPU); compaction data inventory (agent);
  math data inventory (agent); CSAIL check (done: `CSAIL_AUDIT_20261115.md`; new CSAIL jobs use a fresh environment).
- Priority runs (math teacher tests, math frozen/LoRA arms, compaction main variant) go on ORCD / EC2 first; extra compaction
  hill-climbing variants (loss, continuation span, data amount) fill CSAIL when it has room.
- 2026-11-15 ~22:30Z compaction inventory done (`compaction/DATA_INVENTORY.md`): 2,676 training-side tasks with one 9B run each;
  1,261 qualify under the eval cut rule (≈160 updates at batch 8; ≈5,000 items with extra 16k/24k cuts). Data-starved, so two
  more 9B seeds of the same training tasks are being generated (agent, ≤2 EC2 spot GPUs, S3 `compaction/pool_seeds/`). Stage-1
  pipeline (material, KL packet from `R_res_rq9fr` with frozen reader, dev-75 "gap closed" proxy, smoke on ORCD) is being set
  up by a second agent. The old H5–H8 material is not reused (selected samples; part evaluation-side).
- 2026-11-15 ~23:00Z math inventory done (`math/DATA_INVENTORY.md`): 1,500 training problems with three 9B runs (1,000 at >= 8.5 in
  `tqa_m1_train_v2`), 0 overlap with dmext/dmhard; three runs together: median ≈9.0k tokens, max 92.4k. The 27B
  (`unsloth/Qwen3.8-27B-NVFP4`) shares the 9B's vocabulary, so KL from the 27B is possible. Teacher test started (agent, 1 EC2
  spot GPU): 9B full trajectories vs 27B full trajectories vs 9B results-only, 200 training problems >= 8.5, 3 repeats.
- 2026-11-16 vLLM Hidden Residuals merged to main (bf266d5; `activation/vllm_residuals/`, README + tables in
  `vllm_res/`). Vs HF hooks on T_res_rq9fr_kI_KL, 16 dev1000 items: first token 16/16, greedy identical 13/16 compiled
  (15/16 eager); stock rows-only vLLM vs HF was 10/16. Prefix cache hits/misses correctly. Throughput at 32 × 4,000-row
  requests: prefill 20,880 vs 22,781 prompt tok/s rows-only. Cost to watch for compaction evals: each 4,000-row request is
  304 MB (9× rows-only) and is resent every agent turn, so 40 agents per GPU may need more host RAM or fewer agents; an
  engine-side cache of packed rows would remove it. Nodes need `uv sync` to install the plugin entry point.
- 2026-11-16 stage-1 pipeline ready (`compaction/stage1/`; packets on S3 `compaction/stage1/packets/`). Dev-75 proxy (token-weighted,
  497,993 target tokens): untuned full prefix NLL 0.5057, no context 0.6449; start pair `R_res_rq9fr` gap closed −0.493 (1/10) /
  −0.717 (1/5), i.e. worse than no context; after 20 smoke updates +0.078 / +0.076. Smoke: 100 s/update incl. teacher passes,
  57 GB peak. Launched on ORCD: `S1_res_kl_c20` (main: 20k cut, 1,261 items, 157 updates; job 29242839) and
  `S1_res_kl_c162024` (variant: 16k+20k+24k cuts, 3,796 items, 474 updates; job 29242840, time limit raised to 24 h).
- 2026-11-16 correction (user): items never repeat across splits, repositories are shared (ROOT_CANON). Stage-1 packets so far were training-side only (repository-level), and "part evaluation-side" was a wrong reason to drop the old material. Next packet: both sides minus the 322 tasks (+206 runs at seed 0), plus the new seeds; eval-side tasks also get new seeds. Old-model dev-75 NLL is the quick reference.
- 2026-11-16 pool seeds: `compaction/pool_seed_node.sh` runs seeds 1-2 of the 9B pools as spot sky jobs: training side `pc_pool9_tr_s{1,2}_{h0,h1}` (cpool1/cpool2, training order 0-1337 / 1338-2675) and evaluation order 600-1009 `pc_pool9_ev_s{1,2}` (cpool3; asserts none of the 322 excluded tasks); S3 `pre_rebuttal_20261109/compaction/pool_seeds/ROLLOUTS/<cache>/` (DONE marker when complete); bf16 KV (seed 0: fp8).
- 2026-11-16 ~01:40Z both-sides packet `s1kl_both_c20_v1` (build_stage1.sh now `--sides training evaluation`): seed 0, 1,467 runs = items (1,261 training side + 206 evaluation side from `pc_pool9_h6_eval`; 0 of the 322 excluded tasks, checked against `kit/exclude_h7.json`), 184 updates. Launched on ORCD as `S1_res_kl_both_c20` (job 29254883), same recipe as `S1_res_kl_c20`: the difference between the two is only the evaluation-side repositories.
- 2026-11-16 old model H8a on the dev-75 proxy (`compaction/stage1/OLD_H8A_DEV75.md`): rows only (as the paper served it) NLL 0.5216 (1/10) / 0.5110 (1/5), gap closed 0.886 / 0.962; deep (rows + deltas, as trained: H8a has trained delta heads at 3..31) 0.5456 / 0.5414, gap 0.713 / 0.744; vs new runs at update 41 ≈0.630 (gap ≈0.10). H8a has a trained reader LoRA (its own full-prefix teacher NLL 0.4780 < 0.5057), so not an encoder-only comparison.
- 2026-11-16 H8a reader vs encoder split (rows only, OLD_H8A_DEV75.md): H8a reader without rows 0.6418 (vs 0.6449 untuned); H8a encoder + untuned frozen 9B 0.5917 / 0.5758 NLL (1/10, 1/5), gap 0.382 / 0.496, KL 0.0833 / 0.0689 (comparable; new runs at update 81: KL ≈0.112, gap ≈0.11).
- 2026-11-16 ~04:30Z math teacher test done (`math/TEACHER_TEST.md`, 992941c). 200 hard training problems × 3 repeats, paired vs the 9B reading the three results only (77.67): 9B full runs 77.50 (−0.17 [−3.00, +2.50]); 27B full runs 78.83 (+1.17 [−1.83, +4.17], p 0.41); 27B results-only 78.00. No arm is significantly better; on mixed problems (k=1–2, n=53) the 9B full-runs teacher trails by 4.4 (it re-solves 5× less). Single solver 70.17, majority 74.33, any-of-3 82.00. Teacher choice is the user's (pending).
- 2026-11-16 ~07:10Z compaction stage 1, dev-75 NLL r10/r5 (gap closed): `S1_res_kl_both_c20` (seed 0, both sides, 183 updates from R_res_rq9fr, reader frozen) u81 0.6271/0.6274 (0.128/0.126), u121 0.6085/0.6073 (0.261/0.270), u161 0.5765/0.5638 (0.491/0.583), final u183 0.5739/0.5603 (0.510/0.608), KL 0.0627/0.0505. Training-side only: `S1_res_kl_c20` final u157 0.6260/0.6262 (0.136/0.134); cuts variant u241 0.6238/0.6243 (flat). Old H8a encoder + frozen untuned reader 0.382/0.496 (KL 0.0833/0.0689), H8a with its reader LoRA 0.886/0.962. Order is shuffled (ac_model_training.py), so the late jump is a learning transition; one run per arm, so eval-side repositories vs run-to-run timing is not yet separated (S1_lcrec100_kl_both_c20 on the same packet is a second sample). Next: all-seeds both-sides packet.
- 2026-11-16 ~07:40Z long-context reconstruction works: `S1_lcrec100_kl_both_c20` (stage 1 on s1kl_both_c20_v1 from R_cmp_lcrec u100, reader frozen) at u41: NLL 0.5785/0.5654, KL 0.0669/0.0551, gap closed 0.477/0.571, vs 0.111/0.106 for the R_res_rq9fr start at u41 (0.510/0.607 at its final u183). Compaction recipe = R_res_rq9fr (FineWeb recon) → long-context recon on the training prefixes (one pass, `R_cmp_lcrec`, 241 updates) → KL stage 1. Cuts variant S1_res_kl_c162024 cancelled at ~u250 (old start, training side only, flat). Larger run (all seeds, both sides, from R_cmp_lcrec final) next.
- 2026-11-16 ~08:00Z long-context reconstruction phase written up (`compaction/stage1/LCREC.md`): material `lcrec_v1` (`build_lcrec.py`; S3 `compaction/stage1/packets/lcrec_v1/`) = 1,935 runs (1,654 training side + 281 `pc_pool9_h6_eval`; 473 trajectory-capped; 0 of the 322), one item per run, encoder input = the stage-1 AC part, target = that view as text, r ∈ {1/10, 1/5}; held panel 50 training-side runs × 2 sizes. `R_cmp_lcrec` (ORCD 29263023): 241 updates, reader LR 0 (adapter hash unchanged), held recon NLL 0.561/0.595 → 0.031/0.010 at u51; its u100 dev-75 proxy alone −0.696/−0.681 (reconstruction does not by itself help continuation). `cmp_stage1_job.sh` gained `CMP_INIT` (start pair under cmp1/init/, default R_res_rq9fr_final) and `CMP_LOSS` / `CMP_HELD` / `CMP_KEEP_STEPS`; `cmp_ckpt_proxy_job.sh RUN STEP` proxies one checkpoint on its own GPU.
- 2026-11-16 ~08:45Z `S1_lcrec100_kl_both_c20` u81 (sync/held.jsonl): NLL 0.5696/0.5553, KL 0.0586/0.0455, gap closed 0.541/0.644 -- already past the R_res_rq9fr-start run's final (u183: 0.510/0.608) and past the H8a encoder with the same frozen untuned reader (0.382/0.496; KL 0.0833/0.0689); H8a with its own reader LoRA stays ahead (0.886/0.962). Larger all-seeds run from R_cmp_lcrec final launches once tr_s2 caches and the recon final are staged.
- 2026-11-16 ~09:20Z `S1_lcrec100_kl_both_c20` u121 (held.jsonl): NLL 0.5638/0.5495, KL 0.0531/0.0407, gap closed 0.582/0.685 (R_res_rq9fr start at u121: 0.261/0.270).
- 2026-11-16 09:50Z LARGER RUN launched: `S1_lcrec_kl_both_c20_s012` = compaction stage 1 (KL, reader frozen, encoder 5e-5, warm-up 50, cosine, batch 8, one pass = 558 updates) from `R_cmp_lcrec_final` (long-context recon, 241/241 updates) on packet `s1kl_both_c20_s012_v1` (all 13 pool caches: seed 0 + seeds 1-2, training side + evaluation order 600-1009; 4,470 items = 4,470 runs over 2,145 tasks, seeds 0/1/2 = 1,467/1,500/1,503, r 0.1/0.2 = 2,199/2,271; 0 of the 322, checked independently against kit/exclude_h7.json; items.pkl sha256 0f2df9bb..., S3 compaction/stage1/packets/). ORCD job 29301026 (18:50 limit, before the 2026-11-17 00:00 EST monthly maintenance) + resume job 29301045 (afterany, --resume-if-possible from the recovery slot). Dev-75 held report every 40 updates.
