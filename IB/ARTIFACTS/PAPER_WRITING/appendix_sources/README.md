# Appendix hyperparameter sources

This folder holds the files behind every value in `/workspace/paper/appendix.tex` (Appendix A, Hyperparameters). It was built 2026-10-28 at git HEAD `258fce6`.

- Copies were made with `cp -n` and keep the source name, prefixed by origin when two names would collide. `04_multiagent/_copied_sources.json` maps each copy in that folder to its source.
- `*.excerpt.txt` files hold numbered lines from a source. Each starts with `# source`, `# lines`, `# extracted` and `# git commit` headers.
- `*.config.json`, `*.summary.json` and `*.agent_config.json` files are extracts from `.pt` checkpoints or large jsonl files. Each records its `_source` and `_extraction`.
- `*.log_only.excerpt.txt` files hold the exact `IB/STATE.md` or `PLAN.md` lines for a value that no config, script or code backs. These values are marked **log only** in the tables below.
- No weights, optimizer states or large data were copied, and a scan found no secrets.

Paths in the tables are relative to each section's folder.

## 00_shared: "Shared across all experiments"

| Claim | File | Key / line |
|---|---|---|
| AdamW β (0.9, 0.95), ε 1e-8, weight decay 0 | `trainer_defaults.json`, `trainer_config_defaults.excerpt.txt`, `lr_schedule_optimizer.excerpt.txt` | `defaults.adam_betas/adam_eps/weight_decay`; agent_training_utils 448-465 |
| One clip of 1.0 over the encoder and reader adapter | `trainer_clip.excerpt.txt` | ac_model_training.py l.921; `max_grad_norm` |
| Linear warm-up, then the schedule; cosine decays to 0.1× | `lr_schedule_optimizer.excerpt.txt` | `lr_factor` l.504-545; `lr_final_fraction` 0.1 |
| LoRA on q,k,v,o and gate,up,down; α = 2r | `register_lora.excerpt.txt`, plus any `*adapter_config.json` | module_manager l.39, 89-92, 109; `target_modules`, `lora_alpha` |
| No-context penalty only while the reader adapter trains; p = 0.25, scaled by 1/0.25 | `trainer_config_defaults.excerpt.txt` | l.111-112, 229-231 |
| Thinking off by default (Qwen3.5) | `default_call_kwargs.excerpt.txt` | vllm_wrapper l.226; per-run override in each `*agent_config.json` |
| LoRA dropout inactive during AC training | `trainer_dropout_eval_mode.excerpt.txt` | l.667-670 (`disable_dropout`) |

`ac_model_training.py` has uncommitted changes. Line numbers refer to the working tree; `trainer_defaults.json` also gives `line_at_HEAD`, and every default is identical at HEAD.

## 01_retrieval: Retrieval Q/A, Arm G `real_gold`, epoch_005

| Row | File(s) | Key / line |
|---|---|---|
| Reader / encoder (9B r128, 2B r128) | `armG__real_gold__LORAS__rag_reader__epoch_005__adapter_config.json`, `armG__…side_lora__adapter_config.json`, `ec2_fresh_rq_pair_ac__ac_modules.pt.config.json` | `r`, `base_model_name_or_path`; `model_ids`, `side_lora_rank` |
| Write mode (in + residual every 4th layer) | `ec2_fresh_rq_pair_ac__ac_modules.pt.config.json`, `ac_model_rows_and_writes.excerpt.txt`, `ac_model_utils_intervention_layers.excerpt.txt` | `interventions.layers`; ac_model.py 111-136 |
| Compression (0.1, rows 2–4096, k/20 calls, question first) | same config; `armG__real_gold__real_gold.run.json`; `kit_rq_train.excerpt.txt`; `kit_rq_common.excerpt.txt` | `min/max_view_rows`; `ratio`, `per_call`; rq_train l.171 |
| Reconstruction | `retrieval_phase_chain.log_only.excerpt.txt`, `xfer_csail_fast_rq__sft_dense2.sbatch` (comment l.5-6) | **log only** (STATE l.969) |
| Task training: SFT | `xfer_csail_fast_rq__sft_dense2.sbatch` | `--loss sft` |
| Task training: KL to text refs (intermediate stage) | `xfer_csail_fast_rq__tA_{train,teach}.sbatch`, `retrieval_phase_chain.log_only.excerpt.txt` | **log only** for the stage settings (STATE l.991, 996, 1001) |
| Task training: final gold-only KL, top-32 | `armG__real_gold__real_gold.run.json`, `c1__run_teach_real.sh`, `trainer_schedule_kl.excerpt.txt` | `loss`, `teacher_view`; `teacher_cache_top_k=32` |
| Task training: ref sampling, kept only when correct | `kit_rq_common.excerpt.txt`, `kit_rq_refs.excerpt.txt`, `c1__real_gold.jsonl.summary.json` | l.38; l.109, 121; `by_stage` (5,841 of 7,000 kept) |
| Optimization | `armG__real_gold__real_gold.run.json`, `armG__run_train_real.sh`, `trainer_schedule_kl.excerpt.txt` | `epu`, `planned_updates`, `encoder_lr`, `reader_lr`, `warmup`, `schedule`, `cooldown_updates`, `penalty_weight` |
| Evaluation | `m1__build_questions.py`, `m1__eval500.questions.summary.json`, `c3__passes_e500.sh`, `*eval500.scores.json`, `kit_rq_eval.excerpt.txt`, `kit_rq_text_eval*.excerpt.txt`, `select_rq_select_eval2.excerpt.txt` | `val[:500]`; `passes`; rq_eval l.96; text_eval l.47, 90 |
| Retrieval (bm25s, 509,315 paragraphs) | `m1_rq_data2.excerpt.txt`, `c1__bm25_top400.py`, `m1__build_questions.py` | l.56-61; assert 509315 |
| Ideal recall | `kit_rq_common.excerpt.txt`, `pr__run_ac2_ideal_recall.excerpt.txt`, `pr__run_text_ideal_recall.sh` | l.91-107; `--inject-golds` |
| Select → text | `select_rq_select_eval2.excerpt.txt`, `pr_select__run_select2.sh`, `pr_select__eval500.scores.json` | l.176-178, 90 |
| Results | `PAPER_CORPUS_RETRIEVAL_QA_figs__results.json` | |

## 02_corpus: Corpus compression, run `G2n_all20_lr1e3`

| Row | File(s) | Key / line |
|---|---|---|
| Model (4B, frozen reader) | `G2n_all20_lr1e3.run.json`, `G2n_all20_lr1e3.log.excerpt.txt` | `args.lr_reader` "0.0"; log l.14 |
| Corpus (20 patients, 4×5, 400 questions) | `G2n_all20_lr1e3.log.excerpt.txt`, `E_G2n_all20_lr1e3_final.summary.json` | l.1, 10; `views.am.n` |
| AC (KV at 0.1; AM fit; trained keys, biases and values) | `G2n_all20_lr1e3.run.json`, `t0.1_fastest_a0.AM_fit_log.excerpt.txt`, `am_run.sh`, `am_train.excerpt.txt` | `args.table`, `kept_ratio`; fit-log l.6-23, 260 |
| Loss (KL top-32 to the one-patient view) | `am_train.excerpt.txt`, `trainer_kl_clip.excerpt.txt` | l.2-8; trainer l.113 |
| Training data (17,764 pairs; written and answered by the 4B; ~519 tokens per pair) | `G2n_all20_lr1e3.run.json`, `selfstudy_v1_all.stats.json`, `selfstudy_v1_all.finalize.log`, `gen_selfstudy.excerpt.txt`, `PAPER_AM__results.json` | `samples`; `tokens.per_pair_question_answer` |
| Optimization (1e-3, 25 warm-up, cosine, 8 per update, 1 epoch, seed 0, no clip) | `G2n_all20_lr1e3.run.json`, `G2n__partial_epoch.json`, `trainer_kl_clip.excerpt.txt` | `lr_special`, `warmup`, `schedule`, `epu`, `seed`; `steps` 2221; l.921 |
| Optimization (1,037 tokens per update) | `PAPER_AM__results.json` | `tokens.note` |
| Evaluation | `E_G2n_*.summary.json`, `am_eval.excerpt.txt`, `am_lh.excerpt.txt` | `sampling`, `max_new`, `passes` |
| Baselines (LC, gold and AM non-uniform, one pass each) | `PAPER_AM__results.json`, `baselines_cartridges.log_only.excerpt.txt` | **log only** (STATE l.849, 857, 867) |
| Baselines (Cartridges as reported by AM) | `baselines_cartridges.log_only.excerpt.txt` | **log only** (STATE l.888) |

## 03_compaction: model H8a, eval set `eval9c_200`

The H8a pair is not stored locally, so the adapter and AC configs come from its starting pair `h5_final`.

| Row | File(s) | Key / line |
|---|---|---|
| Tasks | `split.summary.json`, `prefixes_eval9c_200__index.json`, `cut_prefixes.excerpt.txt`, `submit_h8full.sh`, `h8a_material_pc{a,b}.summary.json` | `lower/upper/keep_first`; `--exclude-tasks` |
| Agent | `h8a_eval_rollouts.agent_config.json`, `pc_pool9_eval_v3.agent_config.json` | `sampling_params`, `enable_thinking`, `max_turns` |
| Prefix | `cut_prefixes.excerpt.txt`, `split.summary.json` | l.49-57; `eval9c_200_prefix_tokens` |
| AC (2B r128 / reader r128, 4,000 or 2,000 rows, Input Embeddings) | `h5_final_*adapter_config.json`, `h5_final_ac_modules.pt.config.json`, `h8a_eval_ac_rows.summary.json`, `run9h_unit.sh`, `queue_a1_h8a_eval.log.excerpt.txt` | `r`; `keep_tokens`, `num_ac_rows`; `--rows-only` |
| Reconstruction | `reconstruction_lineage.log_only.excerpt.txt`, `h5_final_ac_modules.pt.config.json` | **log only** (PLAN l.142-144, 247, 260, 269); `checkpoint_path` |
| Task training | `export_h7_material.excerpt.txt`, `h8a_material_pc{a,b}.summary.json`, `prepare_agent_continuation_h8.excerpt.txt` | `samples_per_prefix`, `keep`; `--r-range` |
| Optimization (8 per update, 2e-5 / 2e-5, warm-up 50, cosine) | `prepare_agent_continuation_h8.excerpt.txt`, `00_shared/trainer_defaults.json` | l.48, 141, 213-214 (builder defaults; no H8a recipe stored locally) |
| Optimization (870 updates, one epoch) | `h8a_training_run.log_only.excerpt.txt` | **log only** (STATE l.697, 717) |
| Evaluation | `run9h_unit.sh`, `h8a_eval_rollouts.agent_config.json`, `lca_bug_localization.scoring.excerpt.txt`, `RESULTS_SO_FAR.excerpt.txt`, `make_fig_compaction_v4.py` | `--budget 50000`; `per_run[*].seed`; scoring l.3, 253 |
| Text compaction | `ac_parts.summary_prompt.excerpt.txt`, `agent_tools.compaction_tool.excerpt.txt`, `text_summary_arm_rollouts.agent_config.json` | l.23; `tools.compact` |
| Compaction time | `frozen100_h0_fig.json`, `pc_timing_v1_*__{resume,summary}.json`, `make_fig_compaction.timing.excerpt.txt` | `compact_seconds`, `ac_encode_seconds`; `agents_per_gpu` |
| Results | `fig_compaction_mvp.json` | `arms` |

## 04_multiagent: AC pair M1 `tqa_m1_v2`

| Row | File(s) | Key / line |
|---|---|---|
| Test sets | `dmext__sample.json`, `dmext__buckets.txt`, `dmhard__sample.json`, `draws_check.json`, `make_tqa_m1_train_v2.excerpt.txt` | `difficulty`; `overlap_check` 0; `norm()` l.71-79 |
| Solvers | `solver_dmhard.agent_config.json`, `run_tqa_solve_v2.excerpt.txt` | `sampling_params`, `max_turns`, `absolute_trajectory_cap`; `--seeds 3` |
| AC | `tqa_m1_v2__{reader,side_lora}__adapter_config.json`, `tqa_m1_v2__ac_modules.pt.config.json`, `material.json`, `synth_ac_M1_dmhard_s1.agent_config.json` | `r`; `ratio`; `compression_target` |
| Reconstruction | none | **missing** |
| Task training (QA, then synthesis rejection-sampling fine-tuning, 27B targets, keep rule) | `traj_qa_plan_*.log_only.excerpt.txt`, `state_qa_lineage.log_only.excerpt.txt`, `m1v2_gen.sh`, `tqa_r2_generate.excerpt.txt`, `tqa_r2_keep.excerpt.txt`, `keep_report.json` | **log only** for the QA phase; gen l.21; keep l.4-9, 124-133 |
| Task training (difficulty ≥ 8.5) | `m1_train_v2__sample.json` | `min_task_difficulty` |
| Optimization (8 per update, 2e-5, warm-up 50, cosine, 1 epoch) | `prepare_synth_sft_v3.excerpt.txt`, `build_m1_cpu.sbatch` | l.44, 49, 133-135 |
| Optimization (527 updates) | `m1_training_updates.log_only.excerpt.txt` | **log only** (STATE l.717) |
| Synthesis | `run_synth1call_v2.excerpt.txt`, `sumres_dmh_s0__synth1call.json`, `run_synth_ac.excerpt.txt`, `run_synth_ac_norows.py`, `synth_results_lora_M1_dmhard_s0.agent_config.json` | `max_tokens`, `sampling` |
| Evaluation | `score_synth1call.excerpt.txt`, `tqa_unified_table_sumres.excerpt.txt`, `tqa_unified_sumres.json`, `dmhard__tables_M1_5seed.txt`, `tables_unified_tb_sumres.txt` | l.37; `layout` |

## 05_ablations: Section 3.5

`runs_overview.json` gives each run's settings in one view. `runs/<RUN>/` holds each run's `run.json`, its stats summary, `final.summary.json` and adapter configs where they exist. `runs_context_no_ls/` holds the encoder-size runs without least-squares init, for context only.

| Row | File(s) | Key / line |
|---|---|---|
| Reader (4B r128; LR 0 for frozen-reader arms; one at encoder LR 1e-4) | `runs/*/*.run.json`, `runs/R_*_v4*/reader__adapter_config.json`, `runs/*_v4fr/`, `runs/R_in_v4fre4/` | `QA4B_RANK`; `args.reader_lr` "0.0"; `args.encoder_lr` 0.0001 |
| Encoder (default and 0.6B/1.7B with linear skip) | `runs/*/side_lora__adapter_config.json`, `runs/{R,T}_in_e{06,17}ls/`, `excerpts/sec45_common.excerpt.txt` | `SEC45_ENCODER`, `SEC45_LINEAR_SKIP`; l.69-76 |
| Write modes | `runs/*/*.run.json` (`ac_fields`), `excerpts/sec45_common.excerpt.txt` | `intervention_layers`, `kv_transfer`; l.9-14, 32-36 |
| Compression | `excerpts/sec45_common.excerpt.txt` | l.36-38, 126, 137-138 |
| Reconstruction (FineWeb-Edu, 2,000×8) | `runs/R_*`, `data/fineweb_recon_v1.stats.json` | `steps`, `epu` |
| Task training (4,000×8) | `runs/{S,T}_*`, `excerpts/sec45_train.excerpt.txt` | `steps`, `epu`, `train_items` |
| Optimization | `runs/*/*.run.json`, `*.stats.summary.json`, `excerpts/ac_model_training.penalty.excerpt.txt` | `warmup`, `schedule`, `seed`, `penalty_weight`; `learning_rates_last_over_peak` |
| Text + reader LoRA | `runs/T_text_lora/T_text_lora.run.json`, `excerpts/text_lora_train.excerpt.txt` | `lora`, `args`, `penalty` null |
| Evaluation | `runs/*/final.summary.json`, `data/hotpot_tasks_v1.stats.json`, `excerpts/{build_hotpot_tasks,sec45_eval,job_template,text_lora_eval,text4b_sec45}.excerpt.txt` | `n` 200; job_template l.155 |
| Time to first token | `ttft/results_sec35_ttft.extract.json`, `ttft/PAPER_SYSTEMS_README.excerpt.txt` | `_meta.gpu`; l.60 |

No row in this section is log only.

## Not found or not local

- **Retrieval:**
  - `tmp/m1/rq_data2.py` and `build_questions.py` were not at the given paths. The copies came from `IB/TMP/PAPER_RETRIEVAL_QA/m1/`.
  - Arm G's final `ac_modules.pt` is on S3, so the config comes from the dense starting pair.
  - The SFT and Track A runs have no `run.json`.
- **Corpus:**
  - No local AM log or JSON exists for the long-context, gold-context and no-context baselines.
  - `teacher_v1_all.pt` is on EC2.
- **Compaction:**
  - The H8a pair (`pairs/h8a_final`) configs, `packets/h8a_from_h5/recipe.json` and the H8a training log are not local; they are on CSAIL.
- **Multi-agent:**
  - The M1 packet's `recipe.json`, `prepared.json` and `synth_sft_build.json` are missing.
  - The results-only `synth1call.json` for dmhard and dmext is missing; the extres run (`resonly_r2ext_s0__synth1call.json`) stands in.
  - Nothing backs the Reconstruction row.
- **Ablations:**
  - Adapter configs are missing for these runs (no local checkpoint): T_{in,res,kv}_v4, S_{in,res,kv}_v4, S_in_v4fr, R/T_in_e06ls, R/T_in_e17ls and T_in_v4fre4. Their ranks are recorded in each `run.json`.
  - T_in_v4fre4 has no final eval because it was cancelled.
