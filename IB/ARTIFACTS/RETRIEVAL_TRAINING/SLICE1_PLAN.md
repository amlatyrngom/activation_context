# Slice 1 plan — activation-context retrieval training (agent source, implemented)

Agent-facing source of truth. The human projection is `SLICE1_PLAN.tressoir.md` (cards first,
test card first; intent record condensed at the end). Version plan: v1 intent (closed, all nine
decisions accepted), v2 interface + exact e2e test (this), v3 defaults, reporter mock, stats.

## Research notes (what the prototype and the tree say)

Prototype under `/source/activation/retrieval/` (only in `/source`; `/workspace` predates it):

- `retrieval_ac.py` — `StandardRetrievalACModel(harness, base_model_name, ac_model_name, lora_rank=64,
  d_ac_model=None → d_model/4, num_view_tokens=8, num_ac_layers=None → "1/4th of main model's d_model"
  (typo: read as 1/4 of the main model's layer count))`. Docstring architecture: byte embedding table
  256×d_ac → positional encoding → windowed self-attention + pooling (8-byte windows, stride 4, pooled
  to one vector per window, FFN mult 4) → standard bidirectional transformer with `num_ac_layers` →
  "query with up-projection to the main d_model with the number of view tokens" (read as
  Perceiver-style learned queries, `num_view_tokens` of them, cross-attending over the AC output,
  then a linear d_ac → d_model). Output rows are *appended* to the main model's input embeddings.
  `checkpoint` / `load_from_checkpoint` explicitly not yet. "Should accept bog-standard training
  hyperparams if these happen to differ from lora" → per-module optimizer groups.
- `retrieval_model.py` — `RetrievalModel(harness, base_model_name, d_embedding_result, ac_name|None,
  lora_name|None)` ("None = no lora (no full fine-tuning either)"), `embed_batch(queries) -> Tensor`.
  No separate document-embedding method.
- `retrieval_trainer.py` — completion criteria for slice 1: (a) independent review marks the default
  configs standard (lrs, clipping, warmup…), (b) a 1000-train / 50-report run passes and is displayed
  in a nice report the reviewer also reads, (c) loss trend as expected. `RetrievalTrainingConfig`
  (lrs, schedules, clipping, warmup 10 %, epochs; batches: "oom is a risk, handle worst cases well per
  gpu; possibly not a knob"). `RetrievalTrainingStats` (timings, losses, reporting trend).
  `RetrievalTrainer(harness).train(training_config, retrieval_model, retrieval_reporter,
  training_data, reporting_data) -> RetrievalTrainingStats`. Slice 2 = multi-vector indexing and
  retrieval; slice 3 = dataset studying.
- `retrieval_reporter.py` — live HTML file; every 10 % of the training data compute the reporting
  loss and plot it. Sketch: `__init__(path, title, description)`,
  `initialize_(table|line_plot|bar_plot)(name, title, description, metadata)`, `add_data_point(name,
  data)`. "Feel free to improve if insufficiently general." Goals: simple, general, nice to view.
  Mocks requested in planning (details version).
- `retrieval_dense_index.py` — dense retrieval moves here later, must support multi-vector MaxSim;
  skip in slice 1, do not prototype.
- `tests/test_basic_retrieval_training.py` — the key test; sketch: `harness.register_lora(name,
  rank=128)`, `StandardRetrievalACModel(...)`, `harness.register_retrieval_ac_model(name, ac)`,
  `RetrievalModel(...)`, `RetrievalTrainer(...)`, `dataset_manager.get_training_data(dataset_id,
  num_samples, synthethic_only)` → `(training_data, val_data)`, `reporting_data = sample(val, 10,
  seed)`, `RetrievalReporter(...)`, `trainer.train(model, training, reporting,
  report_folder="IB/TMP/<...>/<test_name>/")`, `print(stats.summarize())`, eval skipped. Comments:
  "From now on, our tests will use RedHatAI/Qwen3.5-9B-FP8-dynamic or AxionML/Qwen3.5-9B-NVFP4 for
  faster loads. Keep the existing recommended configs." and "if this runs without oracle using small
  nq/ms-marco, that would be even better."
- `harness/module_manager.py` — `ModuleManager.register_retrieval_ac(ac_name, model_name, ac_model)`,
  `register_lora(lora_name, model_name, rank=128)`; not wired into `HarnessRuntime`.
  `harness/lora.py` is an empty placeholder. `ac_model/passthrough.py` has `ACModelConfig(lora_rank=16,
  d_model_ratio=1/4)` and says "An AC Model has two components: A Lora" — the LoRA is described as part
  of the AC model there, but as a separate named module in `RetrievalModel`.
- `DatasetManager.select_training_data(dataset_id, num_samples, synthetic_only=False,
  oracle_labeled_only=True, val_ratio=0.1, max_reporting_size=0.01, force_partition=True) ->
  (train, val, reporting)` is a stub with a docstring: minimum 10 train / 10 val / 1 reporting.
  The test sketch calls a different name (`get_training_data`) returning two lists.
- Harness facts: `HarnessRuntime.loaded_models[model_name]` → `LoadedModel` (lazy
  `AutoModelForCausalLM`, `model.eval()`, device moves, engine per GPU with `enable_lora=True,
  max_loras=4, max_lora_rank=harness_config.max_lora_rank`). `ModelDescription` gives `d_model`,
  layer descriptions (count), eos token. `readout_embedding` supports EOS / EOT / mean pooling.
  vLLM engine kwargs already assume LoRA adapters in a loadable (PEFT) format.
- Prior M3 staging (IB-only, never adopted): `activation/training/{contrastive_data, encoder, losses,
  train_retriever}.py`. Reusable pieces: loss form A over a flat candidate list with `owner` /
  `is_positive` and same-document masking (canon), `bm25_pseudo` hard negatives, epoch reshuffle
  sampler, warmup + linear decay loop. Canon's "full fine-tune of Qwen3-0.6B, no LoRA" is superseded
  by the prototype (LoRA + AC, frozen base); canon gets updated when slice 1 lands.
- Environment: `peft` not installed; `compressed-tensors` 0.17 present (vLLM dependency);
  transformers 5.15, torch 2.13, vllm 0.28. HF cache has `Qwen/Qwen3-0.6B` (d_model 1024, 28 layers),
  `Qwen/Qwen3.5-0.8B`, `Qwen3-Embedding-0.6B/4B`, MS-MARCO and BRIGHT datasets. This container has no
  GPU, 8 cores, 30 GB RAM. NQ loader exists (`dataset/loaders/nq.py`) with `max_examples`,
  `max_corpus_documents`.

## Accepted intent decisions (closed)

1. Base `Qwen/Qwen3-0.6B` bf16 for the test, no engine; 9B FP8/NVFP4 for oracle tests only.
2. `peft`; LoRA separate harness module; `ac_model/` deleted by user; dependencies welcome.
3. `embed_batch -> [B, V, d_out]` from appended view positions; V = 1 EOS without AC; MaxSim.
4. Symmetric document path.
5. `select_training_data` three lists; inherit rule (one chunk per positive doc) for unlabeled
   positives; BM25 pseudo negatives outside positive docs, carried on the training example, not
   written to `hard_negative_chunk_ids`; no-positive examples dropped + counted; in-batch on top.
6. Method A: physical = logical; checkpointing on base + AC; static token-budget sizing from GPU
   memory + model shape; token packing with per-batch padding; token + byte caps; worst-case probe
   before step 1; single GPU. User: isolate memory logic in reviewable helpers; probes in utils or
   private tests. GradCache dropped. Envelope: `IB/TMP/RETRIEVAL_SLICE1/memory_envelope.py`
   (0.6B ckpt 131 MB/seq → 107/276/501/613 seqs on 24/48/80/96 GB; 8B ckpt 599 MB/seq →
   0/36/85/110; trainables 16 B/param = 5.7 GB worst).
7. Plotly.js, vendored beside the report (the VS Code webview blocks remote scripts).
8. Qwen3-Embedding / Sentence-Transformers LoRA contrastive recipe; values v3, reviewer-audited.
9. Harness owns one PEFT-wrapped base per model name with named adapters; engine/training
   exclusive ("whatever is cleanest").

## Interface (version 2.1) — mirrors the projection cards exactly

Review-pass 1 folded in: CPU test at 20 examples; no token/byte caps, the harness
`doc_embedding_input_limit_chars` (256 in the test) bounds tokens and AC bytes via
`safe_truncate_embedding_chunk`; `batch_size` in examples (8 on CPU, None = recommended);
registration on `harness.module_manager` only (no runtime delegates); `LoraConfig`,
`get_lora_config`, `get_model_with_lora`; no `lora.py` (PEFT config builder in `hf_utils`);
no `input_embedding_table` (reuse the frozen `embedding_layer` copy; LoRA never touches it);
AC/retrieval attrs as `self.x`; no `RetrievalTrainingExample` (trainer reads
`LabeledRetrievalQAExample`); no inherit-rule move (selection calls the study generator's existing
`_inherit_labels` with an empty pool); `retrieval_memory.py` folded into `retrieval_batching.py`
without a class; no `example_cost`; no doc ids in the batch (same-document masking reads the doc id
from `doc:chunk_num` and the example's `positive_doc_ids`). LoRA default targets: the seven block
projections (q,k,v,o,gate,up,down), never lm_head (answers "what is all-linear").

Review-pass 2 (v2.2): no BM25 pseudo negatives and no `bm25_top_k`. MS-MARCO carries native
hard negatives at document level (`hard_negative_doc_ids`, the unselected passages); NQ, BRIGHT,
SciFact, SciQ carry none. `select_training_data(oracle_labeled_only=False)` only inherits what the
dataset has via the study generator's `_inherit_labels` with an empty pool (one chunk per positive
and per hard-negative document); datasets without negatives train on in-batch negatives until
the oracle pass. The test uses `MsMarcoDataset.load(max_examples=20, max_corpus_documents=256)`.

Also accepted live: `batch_size` in examples sized by the heuristic, nothing enforced. LoRA
defaults alpha = 2 × rank, dropout 0.05 (common conventions, reviewer-audited). The loss and
rank-1 metric are `RetrievalTrainer` methods; no `retrieval_loss.py`; no hard-negative cap
(every labeled chunk is used).

Open question (projection decision): (1) batch shape — user's sketch was cut off; proposal
`RetrievalBatch(examples, candidates, num_positives)`.

- M1 test: harness `doc_chunk_size_chars=1024`, `doc_embedding_input_limit_chars=256`;
  `MsMarcoDataset.load(max_examples=20, max_corpus_documents=256)`; `select_training_data(num_samples=20,
  oracle_labeled_only=False, val_ratio=0.5, max_reporting_size=4)` → 10/10/≤4, positives + hard
  negatives present, `oracle_labeled` False, disjoint ids; `module_manager.register_lora(LORA,
  BASE, rank=32)`; `ac_model = StandardRetrievalACModel(harness, BASE, num_view_tokens=4)`;
  `module_manager.register_retrieval_ac(AC, BASE, ac_model)`; `RetrievalModel(harness, BASE,
  d_embedding_result=256, ac_name, lora_name)`; `RetrievalTrainingConfig(epochs=4, batch_size=8,
  gradient_checkpointing=False)`; `RetrievalReporter(folder,
  title, description)`; `RetrievalTrainer(harness).train(config, model, reporter, train, report,
  val)`; asserts epochs 4, steps 8, finite, ≥4 reporting points, last < first, 4 validation
  losses, report files exist.
- M2 harness: `LoraConfig(lora_name, model_name, rank, alpha=2r, dropout=0.05, target_modules=[7
  projections])`; `ModuleManager(harness)`: `register_lora(lora_name, model_name, rank=128,
  alpha=None, dropout=0.05, target_modules=None)`, `register_retrieval_ac(ac_name, model_name,
  ac_model)`, `get_lora_config`, `get_retrieval_ac`, `get_model_with_lora(name) -> PeftModel`
  (load base on TARGET, wrap once, add adapter by name, set active), `lora_parameters(name)`.
  `HarnessRuntime.module_manager`. `LoadedModel.peft_model`, `decoder_forward(inputs_embeds,
  attention_mask) -> [B,S,d]`. `hf_utils.make_peft_lora_config(lora_config)`. Non-interface:
  `ModelDescription.d_ff`; `model_to_device(FREE)` asserts no adapters; reload `.eval()` unchanged
  (retrieval model toggles; HF checkpointing needs training mode; non-reentrant variant for frozen
  embedding inputs); `pyproject.toml` + `peft`.
- M3: `StandardRetrievalACModel(harness, base_model_name, d_ac_model=None, num_view_tokens=8,
  num_ac_layers=None, dropout=0.0)` with `self.harness/base_model_name/d_model/d_ac_model/
  num_view_tokens/num_ac_layers/input_limit_chars`; `forward(texts) -> [B,V,d_model]`;
  `encode_bytes(texts)`; checkpoint methods NotImplementedError. `RetrievalModel(nn.Module)(harness,
  base_model_name, d_embedding_result|None, ac_name|None, lora_name|None, query_instruction)` with
  `self.ac_model/lora_name/num_vectors/d_embedding_result/head/input_limit_chars`;
  `embed_batch(texts, is_query) -> [B,V,d_out]`; `trainable_parameter_groups()`;
  `ensure_resident()`; `similarity(q, c)` MaxSim. Layout tokens + EOS + V view rows, left padded.
  Loss and rank-1 live on the trainer (M5).
- M4: `select_training_data(dataset_id, num_samples, synthetic_only=False,
  oracle_labeled_only=True, val_ratio=0.1, max_reporting_size=50, force_partition=True, seed=0)
  -> three lists of LabeledRetrievalQAExample`; unlabeled examples inherit chunk labels from their
  document-level positives and native negatives (study generator `_inherit_labels`, empty pool);
  dropped-no-positive counted. Non-interface: `DatasetStats.training_select_num_dropped_no_positive`.
- M5: `retrieval_batching.py`: `RetrievalBatch(examples, candidates: list[list[chunk]],
  num_positives: list[int])`; `make_batches(examples, dataset_index, batch_size, rng)` (shuffle, group, every
  labeled chunk); `fixed_batch(examples, dataset_index)`; `recommended_batch_size(harness,
  retrieval_model, gradient_checkpointing, headroom_fraction, device) -> int` (envelope arithmetic:
  usable = total − weights − embedding copy − 16 B × trainable − 1 GB − headroom; example = 17 seqs ×
  chunk_chars/4 tokens × per-token cost; prints); `probe_batch_size(step_fn, harness, model,
  batch_size, attempts=3)` (synthetic worst-case step, halve on OOM, skipped on CPU).
  `RetrievalTrainingConfig(epochs=1, batch_size=None,
  lora/ac/head lrs*, weight_decay*, warmup_fraction=0.1, schedule, max_grad_norm=1.0,
  temperature*, gradient_checkpointing=True, memory_headroom_fraction=0.1,
  reporting_fraction=0.1, seed=0)`; `RetrievalTrainingStats(..., batch_size, probe_attempts,
  step_losses, step_learning_rates, step_batch_shapes, reporting_losses, reporting_rank1,
  validation_losses, batch_sizing; summarize())`; `RetrievalTrainer(harness).train(config, model,
  reporter, training_data, reporting_data, validation_data=None)`, `contrastive_loss(model, batch,
  temperature)` (form A, MaxSim/τ, same-doc masking from the chunk-id doc prefix +
  `positive_doc_ids`), `in_batch_rank1(model, batch)`.
- M6 reporter unchanged from v2. M7 CPU test + SkyPilot 1000/50 (v3 details). M8 review.

## Version 3 (details) — projected in full in the tressoir cards M5/M6/M7

- Scale target: 10k–100k queries (50k typical); 1000/50 is the acceptance run only. At 50k, batch
  64: ~780 steps/epoch, reporting every 78 steps, validation 5k examples ≈ 80 forward-only
  batches/epoch, curves with a few thousand points.
- `retrieval_batching.py` mechanical optimizations (pure functions, private tests):
  `embed_in_length_groups(model, texts, is_query, tolerance=0.15, min_group=8)` (sort by length,
  groups within 15 %, forward each padded to its own longest, restore order; same graph/loss);
  `flatten_candidates(batch)` (dedup chunks within a batch); `recommended_batch_size(harness,
  model, training_data, dataset_index, gradient_checkpointing, headroom_fraction, device, cap=128)
  -> (size, arithmetic str)` on the observed longest text and largest candidate count;
  `probe_batch_size(step_fn, harness, model, batch_size, longest_text, max_candidates, attempts=3)`;
  `fixed_batches(examples, dataset_index, batch_size|None, seed)` (reporting: one batch;
  validation: seeded groups drawn once). No bucketed shuffling (length grouping inside the step
  already removes padding; composition stays random).
- Defaults (reviewer-audited): epochs 1; batch_size None (cap 128); lora lr 1e-4; ac lr 5e-4;
  head lr 1e-3; wd 0.01; AdamW (0.9, 0.999), eps 1e-8; warmup 0.1; cosine to zero; clip 1.0;
  τ 0.02 fixed (GTE-family; E5 0.01, Contriever 0.05); checkpointing on; headroom 0.1;
  reporting 0.1; LoRA r128 α256 dropout 0.05 on q,k,v,o,gate,up,down; AC d/4, L/4, V=8;
  d_embedding_result None (= d_model).
- Stats `summarize()` (flat dict for scale extrapolation): counts (num_examples, num_epochs,
  num_steps, batch_size, probe_attempts, total_candidates, avg_candidates_per_example,
  avg_tokens_per_example real); time (total_train_time, train_time_per_epoch, avg_step_time,
  total_reporting_time, total_validation_time, reporting_time_fraction); throughput
  (examples_per_s, candidates_per_s, tokens_per_s real, padded_tokens_per_s,
  avg_padding_fraction, seconds_per_10k_examples); memory (peak_memory_gb, batch_sizing);
  loss trend (first/last/min step loss, first/last/min reporting loss, last_reporting_rank1,
  validation_losses). Raw lists on the object: step_losses, step_learning_rates,
  step_batch_shapes (examples, candidates, real tokens, padded tokens), step_times.
- Reporter: mock `SLICE1_REPORT_MOCK.tressoir.html` + `.json` generated by
  `IB/TMP/RETRIEVAL_SLICE1/report_mock.py` (renderer prototype). Written for the extension's real
  pipeline (read from `~/.vscode-server/extensions/tressoir.tressoir-artifacts-0.1.5/dist`): the
  artifact markup is injected via innerHTML into the shell after load, scripts re-created with the
  nonce, `tressoir:render` fired; file changes morph the DOM (SCRIPT elements never updated) and
  fire it again; theme changes set `data-theme-kind` on the root. Hence: render on
  `tressoir:render` (plain browser: immediately + timed reload; no meta refresh, which could
  navigate the webview away), data in a hidden `<pre id="report-data">` not a JSON script tag,
  full rebuild per event with `Plotly.purge` of connected plots, MutationObserver on the theme
  attribute, no double render at start (a second render purged plots before Plotly's deferred
  `plotly_react` emit, giving `e.emit is not a function`). Vendored Plotly basic 2.35.2 in
  `vendor/plotly/` (its one `new Function` is try/caught, fine under the no-eval CSP). One
  column, per-series x arrays, smoothed step loss. Assets ship in
  `activation/retrieval/report_assets/` and are copied beside `report.tressoir.html` on first
  render. Verified in headless Chromium (Playwright in the scratchpad) as a plain file and inside a
  replay of the extension shell with its real `notebook-webview.js` and state/update/theme
  messages (`IB/TMP/RETRIEVAL_SLICE1/webview_replay.js`, `webview_shell.html`, screenshots
  `webview_{dark,light}.png`, `mock_{light,dark}.png`): plots draw, morph updates, theme recolors,
  no errors. The user's first report (no plots in VS Code) had two causes: Plotly from a CDN
  (blocked by the webview CSP) and, in the first `.tressoir.html` rewrite, waiting for
  `DOMContentLoaded`, which never fires for injected markup.
- M7 acceptance run: MS-MARCO train 1000 queries, 5 % val, 50 reporting, 4 epochs,
  batch_size None, 4096-char chunks, one RTX PRO 6000 via SkyPilot; report downloaded to
  `IB/TMP/RETRIEVAL_SLICE1/`; teardown.
- Open: user approval of v3 (→ implementation M2→M6, CPU test, GPU run, review) and the batch
  shape (sketch cut off; proposal `RetrievalBatch(examples, candidates, num_positives)`).

## Boundaries

- Source changes land in `/workspace/activation/{retrieval,harness,dataset,tests}` and are handed
  off through diff cards; `/source` is read-only.
- Disposable checks in `IB/TMP/RETRIEVAL_SLICE1/`.
- New dependency: `peft`. Plotly.js vendored as package data (no network at view time).

## Implementation record (M1–M8 done; handoff `SLICE1_ROUND.tressoir.md`)

- Landed in the workspace (diff cards in `SLICE1_ROUND.tressoir.md`, staged copies under
  `IB/ARTIFACTS/RETRIEVAL_TRAINING/activation/`): `harness/module_manager.py` (new), `hf_utils.py`
  (`d_ff`, `make_peft_lora_config`), `loaded_model.py` (`decoder_forward`, PEFT-aware free assert),
  `runtime.py`, `model_config.py`, `harness/__init__.py`; `retrieval/{retrieval_ac,retrieval_model,
  retrieval_batching,retrieval_trainer,retrieval_reporter,__init__}.py` + `report_assets/`;
  `dataset/dataset_manager.py` (`select_training_data`), `dataset/dataset.py`;
  `tests/test_basic_retrieval_training.py`; `bench/retrieval_training_bench.py` (new);
  `pyproject.toml` + `uv.lock` (`peft>=0.20.0`).
- Deviations from the cards: adapter names sanitized (`LoraConfig.adapter_name`); `DatasetIndexes`
  dict per dataset id; `reporter.finish()` and `smoothing_window`; the test shape shrank to fp32,
  batch 2, 128 chars, 32 rows / 20 steps after the local run at the plan's shape peaked at 29.5 GB
  on the 30 GB agent host (OOM-killed; heavy runs are Sky-only from now on); the 1000-query run used
  `batch_size=32` explicitly; the bench loads `int(n * 1.25) + 8` rows to survive the drop of
  unlabeled queries; the old IB-only `activation/training/` package was removed from the staging
  folder by the generator (only its `losses.py` restored to
  `IB/TMP/RETRIEVAL_TRAINING/old_training_package/`).
- Review (M8) changed the model: MaxSim is the **mean** over query views (the sum over V=8 ran at an
  effective τ 0.0025: run 1 started at loss 45.8 and rose to 5.4 at flat rank-1); AC rows enter the
  base at token-embedding RMS; no weight decay on 1-D parameters; probe cleanup after the except;
  seeded construction; validation rank-1 recorded; validation batches of the reporting size; bench
  default 2 epochs.
- Results on one RTX PRO 6000: test `1 passed in 35.61s`; run 2 (950 / 50 / 50, batch 32, 2 epochs,
  60 steps) reporting loss 5.99 → 1.85 → 2.00, validation rank-1 0.34 → 0.38, 4.1 s/step, 7.0k real
  tokens/s, 25.7 % padding, peak 8.5 GB, 1288 s per 10k examples (≈ 1.8 h per 50k-query epoch at
  batch 32). Report `IB/TMP/RETRIEVAL_SLICE1/gpu_run/report.tressoir.html`. Open for the 50k run:
  τ 0.05 if the loss keeps drifting up at improving rank-1 (MS-MARCO false negatives); batch from
  the heuristic (64+ fits).

## User review round 2 (notes in the staged tree, applied)

- LoRA ownership moved from `LoadedModel` to `ModuleManager`: one PEFT wrapper per base
  (`peft_models[model_name]`), adapters by name; `decoder_forward(..., lora_name)` selects the
  adapter per pass through `lora_context` (None disables all); `free_lora(model_name, lora_name,
  checkpoint_path=None)` with the checkpoint assert; base free refused while `has_loras`. Base
  weights present once, adapters add only their A/B matrices. `get_base_model()` was correct.
- `register_retrieval_ac` typed to `StandardRetrievalACModel` (no base class, per the user).
- AC dropout fixed at 0.1 in the layers, none on the output; knob removed. Window attention in the
  byte pooling stays at 0 (SDPA's dropout path refuses the flattened B × windows batch > 65535).
- Log axes label ticks as 1e-4 instead of SI prefixes; the test asserts the step-loss drop and a
  reporting-loss dip instead of last < first on 4 held-out queries.
- In-batch MRR@10 and nDCG@10 next to rank-1 (reporting and validation); corpus-level metrics
  wait for slice 2.
- Reporter split: `common/reporting.py` (`HtmlReporter`, assets in `common/report_assets/`) and
  `retrieval_reporter.py` (`RetrievalReporter` layout + `report_*` events); no reporting code in
  the trainer.
- `retrieval_dense_index.py` untouched.
- Reports are self-contained after the html-skill update: embedded data, pinned HTTPS assets
  (Tressoir linear v0.1.7 on jsDelivr, CodeMirror 5.65.16 on cdnjs, Plotly basic 2.35.2 on
  cdn.plot.ly), no `report_assets/` copied beside the page.
- Node `ac-slice1b`: test `1 passed in 37.10s` (LoRA lifecycle: per-pass adapter selection differs from the plain base, `free_lora` then base free; `sky_e2e_test_run6.log`); run 3 (950 / 50 / 50, batch 32, 2 epochs): 950 / 50 / 50, batch 32, 2 epochs, 60 steps: reporting loss 5.38 → 1.89 → 2.19; validation in-batch rank-1 0.34 → 0.42, MRR@10 0.55 → 0.59, nDCG@10 0.66 → 0.68; 4.0 s/step, 7.2k real tokens/s, peak 8.5 GB, 1258 s per 10k examples (`sky_acceptance_run5.log`, report `IB/TMP/RETRIEVAL_SLICE1/gpu_run3/`). An identical run one commit earlier (`sky_acceptance_run4.log`, before the pooling-attention dropout was removed and the metrics renamed) gave rank-1 0.24 → 0.28, so run-to-run spread at this size is ±0.1 in-batch rank-1 and the dropout-versus-no-dropout gap (run 2: 0.34 → 0.38) is within it.
