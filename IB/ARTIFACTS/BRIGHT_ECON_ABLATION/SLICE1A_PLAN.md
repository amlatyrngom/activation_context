# Slice 1a — BRIGHT economics ablation: probes for the overnight budget (agent source of truth)

Source: the user's uncommitted diff in `/source` (2026-09-05): `@AI` notes in `dataset.py`,
`dataset_manager.py`, `dataset_study.py`, `runtime_config.py`, `retrieval_ac.py`,
`retrieval_model.py`, `retrieval_trainer.py`, `cloud/sky.py`; new stubs
`common/data_syncing.py`, `dataset/dataset_caching.py`, `dataset/dataset_study_prompts.py`,
`bench/bright_econ_ablation/{bench_programmatic_fitting,bench_description_fitting,bench_qa_fitting}.py`;
the bench scripts moved to `bench/agent_probes/`.

Deliverable of slice 1a (user, chat): the probe runs that decide what fits overnight today or early
tomorrow in about 6 hours. Probes must be short.

## Facts gathered

### BRIGHT economics (computed on the host from `xlangai/BRIGHT`, 2026-09-05)

| quantity | value |
| --- | --- |
| documents | 50,220 |
| characters | 19.8M (mean 394 per document, max 39,672) |
| chunks at 4,096 chars | 50,811 (one per document except 591 long ones) |
| chunks at 1,024 chars | 61,592 |
| labeled queries | 103, mean 740 chars |
| gold documents per query | mean 7.8, max 85 |

So "num_samples ~4x the corpus size" means ~200k programmatic examples; each chunk is on average
~100 tokens plus instruction and AC rows.

### Measured rates to extrapolate from (all one RTX PRO 6000)

| stage | rate | source |
| --- | --- | --- |
| training, Qwen3-0.6B, LoRA r128 + AC V=8, checkpointing, batch 128, ~8 candidates/example | 15.0k real tokens/s, 604 s per 10k examples | `IB/TMP/BASELINE_EVAL/run/training_stats.json` |
| training, Qwen3-0.6B, 2 candidates/example, checkpointing | 9.1k real tokens/s, 0.85 s per 32 examples | `IB/TMP/BATCHING/microbench/` |
| generation, RedHatAI/Qwen3.5-4B-FP8-dynamic, max_num_seqs 256–512, 512 max tokens | 3.9–4.0k output tokens/s, ~204k QA per GPU-hour | `IB/TMP/CHEAP_SYNTHETIC/combined_results.json` |
| generation, Qwen3.8-27B (dense, MTP) | ~512 output tokens/s | `IB/STATE.md` (multi-GPU test) |
| labeling, Qwen3.6-35B-A3B (MoE) | 28.4k prompt tokens/s | `IB/STATE.md` |

Qwen3-4B is 6.7x the parameters of Qwen3-0.6B (d_model 2560 vs 1024, 36 vs 28 layers); per-token
training FLOPs scale with parameters, so first-guess training throughput is 2–3k real tokens/s.
These guesses set the probe shapes only; the probes replace them.

### Code paths that exist

- Study: `DatasetStudyGenerator.generate_examples_qa` (chunk sampling by seeded shuffle, one JSON
  QA per chunk, `DataOrigin.SYNTHETIC`), `generate_examples_labels` (BM25 top-k pool, JSON
  positives/negatives, `_apply_labels`, `_inherit_labels`), prompts in `dataset_study.py`.
- Selection: `DatasetManager.select_training_data(dataset_id, num_samples, synthetic_only, oracle_labeled_only, val_ratio, max_reporting_size, force_partition, seed)`.
- Training: `RetrievalTrainer.train(config, model, reporter, training_data, reporting_data, validation_data)` with the references (`retrieval_baseline.py`) scored on the validation batches, epoch-0 points, in-batch metrics.
- Sky: `exec` (upload the mirror, run with `--secret-file`), `watch` (rsync a remote folder under `~/activation_artifacts` to `IB/TMP/...` every 15 s; `-a` is recursive so subfolders come along; `--until-file` looks at the top level only), `download`.
- BRIGHT loader: golds as `positive_doc_ids`, `excluded_doc_ids`, split TEST, no hard negatives.

## Interpretation of the notes

| note | reading |
| --- | --- |
| `DataOrigin.SYNTHETIC_QA / SYNTHETIC_DESCRIPTION / PROGRAMMATIC`, `programmatic_retrieval_examples` | three synthetic origins; programmatic examples live in their own dict because 200k of them should not mix with labeled ones |
| `origins: list[str] | None` on `label_study_examples` and `select_training_data`, "propagate" | replace `synthetic_only` by an origin filter everywhere (`_select_examples_to_label`, `select_training_data`, the test) |
| `select_testing_data(dataset_id, num_samples=None)` | the native test-split examples (BRIGHT: all 103 queries) with inherited chunk labels, for `eval` |
| `synthesize_study_examples_description`, `generate_examples_descriptions`, `make_study_description_prompt`, `make_label_prompt(for_qa)` | a second study kind: a search-friendly description per chunk becomes the query (no reference answer); the labeler judges the BM25 pool for a description ("Description" instead of "Question / Reference Answer") |
| `generate_programmatic_examples`: "chunk is its own positive, in-batch negatives only", `programmatic_min/max_chars` 128–1024 | query = a span of the chunk of 128–1024 chars (decision 1), positive = the chunk, no hard negatives |
| `num_prefix_tokens=16`, "P+V", "handle prefix and view here", `d_ac_model 1024`, `num_ac_layers 8` | the AC model emits P prefix rows before the tokens and V view rows after EOS; new default width and depth |
| `DOCUMENT_INSTRUCTION` | documents get an instruction prefix too |
| `validation_data` "remove from train, move to eval", `eval(model, reporter, eval_data)` | training reports on the reporting batch only; `eval` scores a validation or test set with the references |
| `dataset_caching.py` | JSONL cache of generation and labeling outputs keyed by caching id, dataset, model, kind, seed; prefix reuse across sample counts |
| `data_syncing.py`, `exec --sync`, "replace watch" | `IB/TMP/SYNC/` mirrored to the node before, during and after a job; `resolve_path` maps a sync-relative path to the local or remote root |
| bench stubs | one script per mode, `--probe`, datetime-stamped reports under `IB/TMP/SYNC/BRIGHT_ECON_ABLATION/(PROBE_)REPORTS/`, cache under `.../CACHE/`, Qwen3-4B base, in-batch nDCG on the test queries vs the baseline embedder |
| "2 rounds overnight: bare techniques, then programmatic fit + (cached) training" | Round 1: each mode alone on its own node. Round 2: programmatic pretraining followed by description or QA training in the same process (no checkpointing yet), study outputs read from the cache |

## Round 2 (decisions integrated)

All six recommendations accepted (`SLICE1A_PLAN.interactions.json`). Notes: probe both labelers
(A3B and the 4B generator) on the same 512 examples, report labels/s and agreement; `eval` scores
the whole test set as one batch when it has at most 128 queries and names the baseline under that
policy; orphan nodes: five clusters torn down, `ac-fp4-probe` and the untracked `ac-test` instance
need the user's commands (classifier refused), the untagged t3.2xlarge left alone. M1–M4 Implementing.

## Plan

See `SLICE1A_PLAN.tressoir.md` for the projection (executive summary, decisions, milestone cards).
Everything human-relevant is there; this file adds the agent-side detail below.

### Milestone order and the minimum path to the probes

1. M1 data model and study kinds (dataset, manager, study generator, prompts file).
2. M3 model and trainer (P+V, document instruction, `eval`, train without validation).
3. M4 benches with `--probe` (shared module, three thin scripts, timing file).
4. M2 sync and cache (needed by the probes only for the report path and the cache write; the
   probes can run with M2's cache and `resolve_path` but without the `exec --sync` loop, using
   `watch` as today; `--sync` lands before Round 1).
5. M5 the probe runs (deliverable) and the budget table.

### Estimation formula (implemented in the bench, printed at the end of a probe)

For n study chunks in one mode, epochs e:

    T(n) = t_gen_load + n / r_gen + t_label_load + n / r_label + t_base_load + e * n * c / r_train + t_eval

with c the measured real tokens per example (queries + candidates), r_train in real tokens/s;
programmatic has no generation or labeling terms. The probe measures every t and r and prints the
largest n with T(n) <= 5.5 h (30 min safety), plus T(n) for n in {25k, 50k, 100k, 200k}.

### Risks

- Qwen3-4B training memory: LoRA r128 on 36 layers of width 2560 is ~4x the 0.6B adapter; the
  sizing formula handles it (measured constants are per-token, model-derived). The probe's real
  batch guards the run.
- The 27B generator would take ~4 h for 50k descriptions; the 4B FP8 generator ~15 min. This is
  decision 2.
- Labeling pools on economics are small (top-10 chunks of ~400 chars ≈ 1.2k tokens), so labeling
  should run near 15–20 labels/s; the probe confirms.
- In-batch nDCG@10 on BRIGHT test queries is meaningful (golds are true positives, unlike
  MS-MARCO's sibling passages), but 103 queries in batches of 50 gives ±0.05 noise.

## Round 3 (probes done, 2026-09-05/06)

M1–M4 implemented and in Review, M5 Completed. Handoff: `SLICE1A_ROUND.tressoir.md` (header from
`IB/TMP/BRIGHT_ECON_ABLATION/round_header.md`, 25 cards = exact deltas vs `/source`, staged copies
under `BRIGHT_ECON_ABLATION/activation/`; generator `IB/TMP/BRIGHT_ECON_ABLATION/make_round_doc.py`).

### Probe results (net of loads; `recompute.py` over the three `probe_timing.json`)

| mode | chunks | study | training | eval after (rank-1 / MRR@10 / nDCG@10) |
| --- | --- | --- | --- | --- |
| programmatic | 2,048 | — | 2,007 ex, batch 128, 13.3 s/step, 2,341 tok/s, 248 tok/ex, 0.106 s/ex-epoch, 41.6 GB | 0.44 / 0.51 / 0.48 |
| description | 512 | 155 descriptions (70 % no bridge), gen 4.5 s net; A3B 0.089 s/label, 4B 0.056 s/label, Jaccard 0.75/0.75 | 145 ex, batch 20, 3,020 tok/s, 2,389 tok/ex, 0.791 s/ex-epoch, 48.8 GB | 0.54 / 0.62 / 0.60 |
| question | 512 | 512 questions, gen 3.6 s net; A3B 0.066, 4B 0.043 s/label, Jaccard 0.87/0.84 | 502 ex, batch 23, 2,950 tok/s, 1,751 tok/ex, 0.593 s/ex-epoch, 44.5 GB | 0.59 / 0.65 / 0.60 |

Baseline Qwen3-Embedding-4B 0.61 / 0.67 / 0.64 (loss 3.90); frozen base 0.01 / 0.03 / 0.02; untrained
model 0.02 / 0.03 / 0.02. Loads: generator ~277 s cold, A3B NVFP4 ~685 s, base ~41 s, embedder ~48 s;
the 4B's second load as labeler 53 s. Eval pass ~75 s net. Fixed costs: 268 s programmatic, 1,222 s
with the A3B, ~589 s with the 4B labeler, ~340 s for a fully cached Round 2.

### Budget (5.5 h) and commands

Per chunk (4B labeler): programmatic 0.106 s; description 0.00875 + 0.303 × (0.056 + 0.791) = 0.265 s;
question 0.007 + 0.043 + 0.593 = 0.643 s. Round 1: 150k spans (4.5 h), 50,811 chunks description
(3.9 h; 15.4k examples), 25k questions (4.6 h). Round 2: `--pretrain-programmatic 50000` + corpus
descriptions (5.0 h); `--pretrain-programmatic 25000` + 25k questions (5.0 h). Commands in the round
doc (`--label-model-ids RedHatAI/Qwen3.5-4B-FP8-dynamic`; drop it for the A3B, +20 min per mode).

### Drifts

- `StageTimer` net-of-load seconds and load-as-fixed-cost extrapolation were fixed after the probes
  ran (their files carry the old stage totals; `recompute.py` gives the corrected tables).
- `eval` scores ≤128 queries as one batch (user note), returns `{name: (loss, metrics)}`.
- Description cache rows exist for every chunk (null query for no-bridge chunks).
- The reporting hold-out (2 %) also applies to study sets (502 of 512 trained).

### Open

- User applies the cards, runs Round 1 (three nodes) tonight or tomorrow morning, Round 2 after.
- Orphans: `uv run sky teardown ac-fp4-probe`; terminate i-0f794f54eae4b1c41 (user's commands).
- Follow-ups: chunk junk filter (numbers/URLs), length-sorted forwards (51 % padding in programmatic).

## Round 4 (closing, 2026-09-06)

Round 1 ran (`IB/TMP/SYNC/BRIGHT_ECON_ABLATION/REPORTS/*_20260906_1520/`): description 0.61/0.67/0.63,
question 0.58/0.65/0.59, programmatic 0.39/0.45/0.39 vs reference 0.61/0.67/0.64. Curves with
`--report-on-test --reporting-fraction 0.05`: n2700 (968 ex) and n5000 (1,832 ex) plateau 0.53–0.57
after ~150 examples; n50811 reaches 0.62–0.64 at 20–30 % (stopped). User: saturated benchmark, move
on; Round 2 not run. Code since round 3: seed-per-pass generation, `--study-only`, context-length
guard in labeling, per-kind label cache key, unique report folders, `--report-on-test`. Incidents
and the cache inventory are in the round doc. All nodes torn down.
