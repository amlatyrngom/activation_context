# Activation context, slice 3a: the handoff

Round 2 (2026-09-09). Round 1 implemented and CPU-validated M0–M5, the three key tests of M8 and the M9 driver. This round ran everything on a node before the application, as asked: the three tests pass; the trainer is profiled per phase against the card's measured GEMM ceiling; two throughput levers landed (the fla kernel configs in the AC trainer, including the 0.8B's kernels, and a cached top-k teacher for the later epochs); the cap on one input moved from 52k to 72k teacher tokens with 104k measured; and the validity runs A (compaction), B (trajectory QA) and C (RAG QA) run for two epochs each. While A trained, a review pass over both trainers for training practice (the user's ask: nothing training-unfriendly outside the new parts, agent training included) fixed four things, listed under "Review pass" below; its cards are part of this handoff and B and C run on that code. The cards at the end are the complete diffs against `/source`; the staged tree under `slice3/activation/` is what they produce.

## Application handoff (for your workspace agent)

Order of work, in your checkout of `/source` (HEAD `1cfa058` plus your uncommitted slice-3 stubs, which the cards below replace):

1. **The deletions first** (block below): the retrieval package, its bench and its test leave the tree; their archive is `IB/ARTIFACTS/RETRIEVAL_TRAINING/archive_slice3/`.
2. **Copy the implementation, the tests, the probe and the kernel configs**: `rsync -a IB/ARTIFACTS/AGENT_ROLLOUTS/slice3/activation/ activation/` (45 files: `ac_model/` (6), `agent_training/` (`agent_trainer.py`, `agent_training_config.py`, `agent_training_utils.py`, `fla_cache.py`) + the 10 JSON kernel configs under `agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/` (merged: every earlier entry kept, the 0.8B's and the new length buckets added), `bench/agent_probes/ac_training_bench.py` + `ac_training_profile.py`, `dataset/` (6), `dataset/loaders/` (10), `harness/` (3), `tests/` (3)); copy the staged `pyproject.toml` (`faiss-cpu` removed) and `uv.lock`. The cards are the fallback if a local file has drifted.
3. **CPU checks**: `uv run python -m compileall -q activation`; `uv run pytest activation/tests/test_basic_dataset_loading.py` (4 passed here in about 2.5 min with the network); the scratch checks are reproducible from `IB/TMP/SLICE3_CHECKS/` (`check_fixtures.py`, `study_cache_smoke.py`, `optimizer_smoke.py`, `ac_smoke.py`, `ac_train_smoke.py`, `ac_perf_smoke.py`; the last three need Qwen3.5-0.8B and about 12 GB of RAM; run with `MALLOC_ARENA_MAX=2`).
4. **Node** (what ran on `ac-fp4-probe`, for the record; run A used the pre-review code, B and C the reviewed code with its defaults, `--warmup-updates 10` and the stratified split; the runs' final numbers land in this document's next update, nothing is needed from you on a node):

```bash
uv run sky exec --sync ac-fp4-probe -- uv run pytest activation/tests/test_basic_agent_ac_training.py --gpu --slow -s -x
uv run sky exec --sync ac-fp4-probe -- uv run python -m activation.bench.agent_probes.ac_training_bench --kind compaction --items 2000 --held 200 --generate-only
TRITON_PRINT_AUTOTUNING=1 uv run sky exec --sync ac-fp4-probe -- uv run python -m activation.bench.agent_probes.ac_training_profile --items-cache AC_ITEMS/compaction_0.jsonl --steps 3 --synthetic 72000:2,104000:2 --teacher-cache-top-k 64 --dump-fla-configs --tag profile_r1
uv run sky exec --sync ac-fp4-probe -- uv run python -m activation.bench.agent_probes.ac_training_profile --items-cache AC_ITEMS/compaction_0.jsonl --steps 3 --synthetic 72000:2,104000:2 --teacher-cache-top-k 128 --tag profile_r2
uv run sky exec --sync ac-fp4-probe -- uv run python -m activation.bench.agent_probes.ac_training_bench --kind compaction --items 2000 --held 200 --epochs 2 --teacher-cache-top-k 128 --tag compaction_r1
uv run sky exec --sync ac-fp4-probe -- uv run python -m activation.bench.agent_probes.ac_training_bench --kind traj_qa --items 2000 --held 200 --epochs 2 --teacher-cache-top-k 128 --tag traj_qa_r1
uv run sky exec --sync ac-fp4-probe -- uv run python -m activation.bench.agent_probes.ac_training_bench --kind rag_qa --items 2000 --held 200 --epochs 2 --teacher-cache-top-k 128 --tag rag_qa_r1
```

   Outputs under `IB/TMP/SYNC/`: `AC_TRAINING_TEST/{compaction,traj_qa,rag_qa}/` (test pages), `AC_BENCH/profile/profile_r{1,2}.json`, `AC_BENCH/{compaction,traj_qa,rag_qa}_r1/{report.tressoir.html,summary.json}`, `AC_ITEMS/*.jsonl` (the item caches, 300 MB for compaction), `AC_MODELS/ac_bench/` and `LORAS/ac_target/` (checkpoints per round), `STUDY/` (the 27B's questions). Logs: `IB/TMP/AGENT_ROLLOUTS/ac_*.log`.

Things that look wrong but are not: the first steps print gradient norms in the tens before clipping (rows at embedding scale, Adam-normalized); the side LoRA's A matrices have zero gradient at step 1 (B starts at zero); `EncodeQueue` statistics stay at zero in training mode; the head scales print once per process (`Head scales initialized: side 0.0194, target 0.0128`); an item's `rows` in the profile can read 8192 (a depth-2 item encodes its child and its parent, 4096 each); `sky exec --sync` can wedge on a dropped rsync session (it happened once here, the node's sshd stopped answering for about ten minutes; kill the local `rsync`/`ssh` pair and the wrapper returns).

## Review pass (training practice, both trainers)

Read end to end during run A: `ac_model_training.py`, `ac_model.py`, `ac_model_utils.py`, the bench, and slice 2's `agent_trainer.py`, `agent_training_config.py`, `agent_training_utils.py`, `agent_training_selection.py`. Outside the deliberately new parts (the rows, the mixer, the coarsened KL) the question was whether everything follows ordinary practice. Four things did not:

| finding | where | fix |
| --- | --- | --- |
| A fresh AdamW every `train()` call: the moments and the step counters reset at every round boundary, so each round (each 50-update chunk in the bench, each 4-update round in agent training) opened with bias-corrected first steps, i.e. sign-like updates at the full learning rate | both trainers | `persistent_optimizer` / `optimizer_state_to` in `agent_training_utils.py`: one AdamW per AC model (AC trainer) or per adapter (agent trainer), created on first use, reused across rounds, its moments parked on the host between rounds (the bases leave the GPU between rounds); rebuilt with a printed note only when the parameter objects change (an adapter re-injected) |
| No learning-rate warm-up | both trainers | `warmup_updates` (linear, counted across rounds: `set_learning_rates`); the AC bench defaults to 10 updates, the agent trainer's default stays 0 so its tests and the rollout loop are unchanged until the loop sets it |
| A mid-round eval restored `model.train()` and with it every dropout module (harmless for Qwen3.5, whose dropout rates are 0, but wrong in general) | AC trainer `_evaluate` | `disable_dropout` after `train()`; the eval also reports how many items it dropped for length |
| The bench held out the head of the item list after one seeded shuffle: mixed in expectation (run A's 200 held items were 95 Open-SWE / 105 Nemotron), not by construction, and with no guard against a training item cut from a held-out item's trajectory (the compaction generator reuses documents once it has cycled through them) or asking a held-out item's study question | AC bench | `split_items`: held-out in proportion to the sources, then training items whose origin (the trajectory of a compaction item, the study question of a QA item) is not held out (`item_origin`); the counts print at start |

Looked at and left alone: the loss normalization (per-example mean over positions, mean over the examples of a step; the agent trainer's per-sequence mean of the clipped surrogate), gradient clipping before the step, the fp32 head logits, `weight_decay` 0 on adapters, the betas, the seeds, the checkpointing switches, the frozen embeddings, the no-grad teacher, the placement (models to host RAM between rounds), the packing and the pi_old fill in agent training. Left for slice 3a1 (the user's list): the eval cadence inside `train()` (the bench splits an epoch into `train()` calls to get held evals every 50 updates, which also moves the models and writes a checkpoint per chunk), the round / chunk / epoch naming in the reports, the sync of large folders.

Run A had already trained for an epoch on the earlier code when the pass finished, so it completed on that code; runs B and C picked up the reviewed tree (the wrapper uploads the working copy at launch). The change to the AC trainer is exercised by B and C on the node; the agent trainer's change is CPU-checked (`optimizer_smoke.py`, the helpers on toy parameters) and syntax-checked, and its node test (`test_basic_agent_training.py --gpu --slow`) is queued after C.

## Node results

### The three tests (M8)

`3 passed in 823 s` on `ac-fp4-probe` (RTX PRO 6000, 96 GB). Held KL(teacher ‖ student) on 8 held items, before and after one round of 5 updates on 40 items:

| test | untrained AC | no context | after one round | agreement before → after |
| --- | --- | --- | --- | --- |
| compaction (Open-SWE + Nemotron tir) | 0.579 | 0.574 | 0.525 | 0.765 → 0.779 |
| trajectory QA (S1 + Open-SWE, 27B questions) | 1.055 | 0.936 | 0.917 | 0.674 → 0.675 |
| RAG QA (BRIGHT biology) | 0.897 | 0.869 | 0.849 | 0.747 → 0.733 |

The untrained model is worse than an empty context (the rows are noise at start); one round already beats the empty context on every kind. Nothing more should be read into 40 items.

### The profile (M9 phase 0)

`ac_training_profile` times one training example per phase on real compaction items bucketed by teacher length, plus synthetic depth-2 items at 72k and 104k (a real trajectory's messages cycled), and counts the FLOPs of each phase from the token counts and the model shapes (dense 2·params·tokens plus the full-attention layers' T² term). The ceiling is the card's measured bf16 GEMM rate, **410 TFLOPS** (8192³), not the spec sheet. Two passes, r1 and r2, before and after the kernel-config dump; they are the same within noise, and `TRITON_PRINT_AUTOTUNING=1` printed no benchmarking line in either: fla's fuzzy matching served the 0.8B's kernels from the 4B's checked-in entries, and the dump only made those matches explicit entries.

| teacher tokens (bucket) | n | mean teacher tokens | s / example r1 → r2 | examples / h (r2) | teacher tokens / h (r2) | achieved TFLOPS (r2) | of ceiling | peak GB | cached-teacher gap (k = 128) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0k-8k | 3 | 4,901 | 0.98 → 0.97 | 3,694 | 18.1M | 73.1 | 18% | 26.6 | 0.1% |
| 8k-16k | 3 | 15,106 | 1.77 → 1.77 | 2,037 | 30.8M | 120.3 | 29% | 46.7 | 0.2% |
| 16k-32k | 3 | 23,140 | 2.50 → 2.50 | 1,441 | 33.4M | 124.5 | 30% | 46.8 | 0.4% |
| 32k-48k | 3 | 35,666 | 3.82 → 3.80 | 948 | 33.8M | 129.4 | 32% | 51.3 | 0.3% |
| 48k-64k | 2 | 50,038 | 5.60 → 5.56 | 647 | 32.4M | 133.7 | 33% | 60.7 | 0.3% |
| synthetic 72k depth 2 | 2 | 74,918 | 8.50 → 8.46 | 426 | 31.9M | 143.7 | 35% | 68.6 | 0.4% |
| synthetic 104k depth 2 | 2 | 107,814 | 12.67 → 12.65 | 285 | 30.7M | 156.5 | 38% | 73.5 | 0.4% |

Where the time goes (r2, shares of one example): teacher forward 64%, backward 23%, ac encode 7%, student forward 5%, head chunks 0% (items ≥ 8k).

| phase | share of time | TFLOPS (range over buckets) | of ceiling | peak GB (max) |
| --- | --- | --- | --- | --- |
| teacher forward | 64% | 139–184 | 34–45% | 25.4 |
| ac encode | 8% | 61–93 | 15–23% | 32.9 |
| student forward | 5% | 136–144 | 33–35% | 73.3 |
| head chunks | 0% | 70–128 | 17–31% | 73.5 |
| backward | 23% | 83–94 | 20–23% | 73.4 |

Inside the AC encode: embed 2%, mixer 6%, pooling 7%, side_decoder 85%, heads 1% (items ≥ 8k and the synthetic ones included).

What the numbers say:

- **The teacher forward is the example.** It is 46–69% of the time at every length and runs at 34–45% of the GEMM ceiling on HF eager with the fla kernels (about 12k tokens/s, linear in length: the T² term of the 8 full-attention layers is not what limits it below 104k). The student forward runs at the same efficiency; the backward (student recompute and gradients, side decoder, modules) at 20–23%; the AC encode at 17–22%, of which the side decoder is 70% and the mixer plus pooling about 10%. The head chunks are 0.01 s per example: nothing to gain there, and the checkpointing threshold does not apply (the student sequence is under the card's recompute-free limit).
- **Lever 1, the fla configs, is 2×.** The AC trainer did not load the checked-in kernel configs (only the agent trainer did), so every new length bucket re-benchmarked the l2norm kernels; with `configure_fla_cache` at placement the test round went from 4,681 to 9,714 tokens/s (steps 84 / 53 / 26 / 22 / 36 s → 19 / 25 / 19 / 22 / 22 s). The 0.8B needed no tuning of its own: fuzzy matching resolves its kernels from the 4B's entries (no autotuning line in either profile pass); the merged JSONs in the cards carry those resolutions explicitly, which costs nothing and pins them.
- **Lever 2, the cached teacher, makes the later epochs cost the non-teacher share.** With `teacher_cache_top_k` the teacher's top-k log-probs per completion position stand in from an item's second pass on; the KL becomes the KL of the k+1-way coarsened distributions, measured against the exact one on the same student states: at k = 64 the gap is 0.1–1.0% of the KL (mean 0.6%) on real items, at k = 128 0.1–0.6% (mean 0.3%). Under the "about 1%" criterion it is on for the validity runs (k = 128; the cache is about 1 GB of CPU memory for 2,000 items). An epoch with the cache costs 36% of a live-teacher epoch.
- **The next 2× is not in this design's Python.** At 35–45% of the GEMM ceiling on a single sequence, the teacher forward is where HF eager tops out; a prefill engine would do it faster (vLLM's prompt log-probs are exactly the cached top-k), which is a slice-3b-sized change (the engine holds the adapter and would fill the cache once per epoch). Overlapping the next teacher with the current backward on a second stream would not help a saturated card. Small items (< 8k) run at 14–25% of the ceiling because the per-layer launch overhead dominates; they are 3% of the compaction mix by time.

### The caps

`max_example_tokens` moved from 52k to 72k teacher tokens: depth 2 at the default 32k compaction threshold (66k with the task and the completion) and depth 1 up to a 50k threshold, both of which the old cap dropped. 104k (depth 2 at 50k) was measured, not tuned for. The AC side never exceeds the threshold plus 4k rows, so the mixer and the side decoder needed no change; the teacher side runs without gradient and grows linearly.

| teacher tokens | peak GB (r2) | s / example | fits the 96 GB card with the engine asleep |
| --- | --- | --- | --- |
| 35,666 (32k-48k) | 51.3 | 3.8 | yes |
| 50,038 (48k-64k) | 60.7 | 5.6 | yes |
| 74,918 (synthetic 72k depth 2) | 68.6 | 8.5 | yes |
| 107,814 (synthetic 104k depth 2) | 73.5 | 12.7 | yes |

The generator's total bound for a compaction item is 72k (`max_total_tokens`); the threshold range stays 8k–32k by default and 8k–24k in the bench. Nothing about the caps is tuned for anything past 72k.

### What one node hour buys

From the r2 profile (one example at a time, live teacher), and from the runs (the whole loop with evals and checkpoints):

| what | per node hour |
| --- | --- |
| examples of 0k-8k (4,901 teacher tokens on average), live teacher | 3,694 examples, 18M in-context tokens |
| examples of 8k-16k (15,106 teacher tokens on average), live teacher | 2,037 examples, 31M in-context tokens |
| examples of 16k-32k (23,140 teacher tokens on average), live teacher | 1,441 examples, 33M in-context tokens |
| examples of 32k-48k (35,666 teacher tokens on average), live teacher | 948 examples, 34M in-context tokens |
| examples of 48k-64k (50,038 teacher tokens on average), live teacher | 647 examples, 32M in-context tokens |
| examples of synthetic 72k depth 2 (74,918 teacher tokens on average), live teacher | 426 examples, 32M in-context tokens |
| examples of synthetic 104k depth 2 (107,814 teacher tokens on average), live teacher | 285 examples, 31M in-context tokens |

Rule of thumb for sizing: a live-teacher epoch costs about 30M in-context tokens per hour whatever the length mix above 8k; a cached-teacher epoch about 2.5–3× that; evals cost one live forward pair per held item.

### The validity runs (M9)

Side Qwen3.5-0.8B (LoRA 128), target Qwen3.5-4B (LoRA 64), 2,000 items and 200 held per run, `examples_per_update` 16, two epochs, held evals every 50 updates in epoch 1 and per epoch, the cached teacher (k = 128) from the second pass of an item on, the greedy secondary metric on 100 held items per epoch. References on the held set: `no_context` (the parts removed), `recent_text` (each part replaced by verbatim text of the same token budget as its rows), `untrained_ac`, and `in_context` (the teacher, KL 0 by construction). The read-out the plan asked for: held KL below `recent_text` within epoch 1 and still descending at the end of epoch 2.

**compaction_r1** (compaction, 2000 training / 200 held items, 2 epochs, in progress when this document was built: 1 of 2 epochs done)

| reference / eval | held KL | agreement |
| --- | --- | --- |
| no_context | 0.6517 | 0.753 |
| recent_text | 0.3710 | 0.818 |
| untrained_ac | 0.6482 | 0.754 |
| epoch 1, after 50 updates | 0.4590 | 0.797 |
| epoch 1, after 100 updates | 0.4407 | 0.802 |
| epoch 1, after 125 updates | 0.4360 | 0.804 |

Secondary metric (greedy 64 tokens on 100 held items): agreement of the first tool call with the teacher's greedy continuation, over the items where the teacher calls a tool (share `teacher_calls`); `name` = same tool, `exact` = same tool and arguments, `args` = token overlap of the arguments when the tool matches:

| after epoch | teacher calls | student name / exact / args | no_context name / exact / args | recent_text name / exact / args |
| --- | --- | --- | --- | --- |
| 0 | 0.30 | 0.03 / 0.00 / 0.00 | 0.43 / 0.03 / 0.04 | 0.03 / 0.00 / 0.00 |
| 1 | 0.21 | 0.29 / 0.00 / 0.02 | 0.29 / 0.00 / 0.05 | 0.29 / 0.00 / 0.06 |

Read-out so far: below `recent_text` never; still descending at the end (the run continues; the final numbers land in the next update of this document).

**traj_qa_r1**: pending (the run had not started when this document was built).

**rag_qa_r1**: pending (the run had not started when this document was built).

## Validation (CPU, this container)

| check | what ran | result |
| --- | --- | --- |
| dataset model and bm25 index | `pytest activation/tests/test_basic_dataset_loading.py` | 4 passed (bm25 basics; trajectory chunking; BRIGHT × 2, NQ, MS MARCO, SciFact, SciQ at 20 examples; Open-SWE, Nemotron tir, S1 at 2 streamed rows each) |
| reformatting | `check_fixtures.py` on saved rows (`IB/TMP/SLICE3_FIXTURES/`) | Open-SWE 40 / 68 messages, 24 / 44 `shell` calls; Nemotron cot 2 messages of pure reasoning (100k / 13k chars); tir 28 messages, 13 `python` calls; S1 16 / 20 messages, 7 / 9 `search` / `visit` calls, five tool definitions; AIME 13-gram self-hit true, Nemotron sample no hit |
| study cache | `study_cache_smoke.py` (fake engine) | 6 questions = 6 engine calls; identical rerun 0 calls; 9 requested = 3 new calls; trajectory variant samples trajectory chunks; labels cached by example id (3 calls, then 0) |
| AC model | `ac_smoke.py` (0.8B side and target) | 1,095 tokens → 64 rows in 5.8 s; cache hit; nested part = 3 passes; batched vs single ≤ 0.009 at RMS 0.019; queue batches 4 requests; gradients into 46 / 46 module tensors and 96 / 192 LoRA tensors; save 219 MB + side LoRA, load, version 1 |
| trainer | `ac_train_smoke.py` (fabricated items of the three kinds) | 7 fabricated items (3 compaction at depth 1–2, 2 trajectory QA, 2 RAG QA; completions 32 tokens; thresholds 120–200 tokens; `gradient_checkpointing_min_tokens=0`, `encode_batch_max_tokens=4096`, `warmup_updates=3`), two rounds of 4 updates with Qwen3.5-0.8B as both side and target on CPU, rerun on the reviewed trainer. Held-out eval KL(teacher‖student) went from 3.24 (agreement 0.238; compaction 1.86, traj QA 4.36, RAG QA 4.19) untrained to 2.14 (agreement 0.408) after 8 updates; per-step training KL 2.13, 2.11, 4.51, 2.86 | 3.32, 2.03, 1.94, 1.16 (single-item steps, so the per-kind spread dominates the trend: round-2 per-kind KL compaction 1.15, traj QA 3.60, RAG QA 3.65 vs 1.83 / 4.21 / 3.96 in round 1). After the two rounds one AdamW exists for `ac_dev`, its step counters at 8 (both rounds counted), its moments on the host, the update clock at 8; the warm-up factors 1/3, 2/3, 1 land on both parameter groups; the eval reports 0 dropped items. The bench's `split_items` on the 7 items holds out 2 compaction + 1 RAG QA (in proportion to the two fabricated sources), trains on the other 4, and drops none for a shared origin. Checkpoints `AC_MODELS/ac_dev/round_000`, `round_001` + `latest` (ac_modules.pt 219 MB + side LoRA) and `LORAS/ac_target/round_00N` written and reloaded. Steps took about 20–60 s (about 21 tokens/s), ~7 GB RSS. |
| optimizer helpers | `optimizer_smoke.py` (toy parameters) | two rounds of three updates keep one AdamW (step counters at 6, moments on the host after the park); the warm-up factors 1/4, 2/4, 3/4, 1 land on both groups' rates; changed parameter objects rebuild the optimizer with the note; `warmup_updates=0` leaves the configured rate |
| teacher cache and profile | `ac_perf_smoke.py` (toy head; 4 fabricated items; the probe on them) | coarsened KL ≤ exact for k = 1, 5 and equal at k = vocabulary; two rounds with k = 32: 0 then 4 cache hits, finite losses, exact eval after; the probe runs its buckets, a synthetic depth-2 item within 5% of its target length, the cached gap and the encode phases on CPU |

## Milestone completion reports

### M0 — completion report

**What landed.** `dataset.py`: `origin` is a string (`NATIVE` / `EXTERNAL` / `SYNTHETIC` constants), `DatasetDocument.trajectory` + `trajectory_kwargs` + `modality`, `DatasetDocumentChunk.chunk_messages`, `LabeledRetrievalQAExample` → `DatasetQAExample` (`labeled_qa_examples`), `DatasetStats` reduced to load / chunk / bm25 / study counters (`num_decontaminated`, `study_num_junk_skipped`, `study_num_cached`, `study_num_labels_cached` added). `dataset_index.py` is bm25-only (dense paths, faiss and the embedding knobs gone; `HarnessRuntimeConfig.doc_embedding_*` removed). `dataset_utils.py` renders messages for the index (`render_message`: `role: text` plus `[call name(args)]` lines). `dataset_manager.py` keeps `register_dataset`, `build_bm25_indexes` and the two study wrappers. Five loaders follow the rename. `retrieval/`, `retrieval_training_bench.py`, `test_basic_retrieval_training.py` and the pre-slice-3 `dataset_index.py` / `module_manager.py` are archived under `IB/ARTIFACTS/RETRIEVAL_TRAINING/archive_slice3/` (README records the HEAD); `faiss-cpu` left `pyproject.toml` and `uv.lock` (`uv lock`: 215 packages resolved, faiss-cpu 1.15.0 removed).

**Drifts.** The bm25 loading test lost its dense variant as planned and gained `test_trajectory_chunking_bm25` and a live `test_trajectory_dataset_loading` (M1); `DatasetStats.summarize()` was rewritten rather than trimmed (the old one referenced the removed fields). `README.md` in the workspace was already modified before this slice and is not part of the handoff.

**Validation.** `uv run pytest activation/tests/test_basic_dataset_loading.py` (CPU, network): 4 passed (bm25 basics, trajectory chunking, five public datasets at 20 examples / 100 documents, the three trajectory loaders at 2 rows).

### M1 — completion report

**What landed.** `loaders/trajectory_utils.py`: the shared intermediate (`{role, content, reasoning, tool_calls:[{name, arguments}]}` + tool definitions) and `reformat_trajectory` into our dialect (source system prompt dropped, `bash(command)` → `shell(script)`, `stateful_python_code_exec(code)` → `python(code)`, reasoning kept as text ahead of the reply, structured `tool_calls` with `call_NNNN` ids, one `tool` message per result, JSON `{"returncode","output"}` flattened with an `[exit code N]` line, definitions for every mapped tool the trajectory calls); `parity_split` (one document in ten is TEST by hash); `ngram_windows` for decontamination. Three loaders on it: `OpenSweTracesDataset.load(harness, max_examples, config="v1.2", split="minisweagent", resolved_only=False, min_chars=0)`, `NemotronMathDataset.load(harness, max_examples, subset="tir", excluded_problems=None, min_chars=0, min_tool_calls=0, hf_dataset=..., data_files=None)` with the 13-word-window check (`stats.num_decontaminated`), `S1DeepResearchDataset.load(harness, max_examples, language="en", min_chars=0)`; `loaders/aime.py` (`load_aime_problems`, `aime_problem_statements`; `math-ai/aime24` stores the answer as `solution` = `\boxed{...}`, `aime25` as `answer`; 30 + 30 rows confirmed). Trajectory chunking in `dataset_index._trajectory_pieces`: whole messages packed to the chunk size, no overlap, an over-long message split by its own text into one-message slices (the calls ride on the last slice), chunk text = the rendering.

**Drifts.** (1) Nemotron's single `train` split is ordered cot-then-tir and the boundary (row 285,516) lies inside shard 7 of 12 (row counts read from the parquet footers), so each subset streams only its shards (`cot` 0–7, `tir` 7–11); a `min_tool_calls` filter was added because some `tir` rows never call the tool. (2) S1's system prompt mentions a literal empty `<tools></tools>` before the real block; the parser takes the first non-empty block (five tools: `search`, `visit`, `PythonInterpreter`, `google_scholar`, `parse_file`; they keep their names). (3) Open-SWE trajectories end with a call whose result never arrives (the submit call); the reformatter keeps it. (4) HF ids for AIME confirmed from the datasets-server, not on a node.

**Validation.** Fixture rows saved under `IB/TMP/SLICE3_FIXTURES/` (three Open-SWE, four Nemotron cot, two tir, three S1, two AIME each) and checked by `IB/TMP/SLICE3_CHECKS/check_fixtures.py`: every row reformats with dict arguments, mapped tool names, one tool message per call (minus the trailing submit), no system role; AIME self-hit true, Nemotron sample no hit. Live: `test_trajectory_dataset_loading` streams two rows of each source (Open-SWE 17 s, Nemotron tir 19 s, S1 19 s cold) and checks modality, origin, tools, calls, chunk round trip.

### M2 — completion report

**What landed.** `dataset_caching.py`: `StudyCacheKey(caching_id, dataset_id, model_id, kind, seed, label_model_id)` → one JSONL per key under `STUDY/`, `StudyCache.read / append / read_by_id` (a torn last line is dropped). `dataset_study.py` rewritten: `has_valuable_question` schema (junk chunks skipped and cached as junk rows), a trajectory prompt variant, `generate_examples_qa(num_samples, base_seed=0, study_context=None, caching_id=None, modality=None)` with cached rows served first (checked by chunk id against the seeded sampler), engine seed `base_seed + batch index`, examples returned (origin SYNTHETIC, positive chunk = the source chunk) and nothing written into the dataset; `generate_examples_labels(examples, caching_id=None)` labels exactly the given examples, cached by example id. `dataset_manager.synthesize_study_examples_qa(dataset_id, num_samples, base_seed=None, study_context=None, caching_id=None, modality=None)` and `label_study_examples(dataset_id, examples, caching_id=None)`. `test_basic_dataset_study.py` follows the API (synthetic examples labeled first, then the native ones; count assertions allow junk).

**Drifts.** The label cache is keyed by the label model id with seed 0 (labels do not depend on the sampling seed). Parse failures are not cached (a later pass retries). The `study_context` argument of the old dataset object is gone (it was never set).

**Validation.** `IB/TMP/SLICE3_CHECKS/study_cache_smoke.py` (fake engine on a fabricated dataset): 6 questions cost 6 engine calls, the second identical request costs none, a request for 9 reuses the 6 and generates 3, the trajectory variant samples trajectory chunks only, labels are cached and served by example id (3 label calls, 3 cached on the rerun). The node test (`--gpu`) is not rerun here.

### M3 — completion report

**What landed.** `ac_model/ac_model_utils.py`: `tokenize_with_parts` (sentinel per part in the chat template, split, tokenize the pieces, pad-id runs, spans; a part without a user message gets an empty one because Qwen's template refuses a conversation without a user query), `BlockedLocalAttentionLayer` (blocks of `mixer_window` rows attending to themselves and both neighbours through SDPA on the reshaped tensor; padded queries attend everything in their window so no row is fully masked), `WindowedPooling` (the retrieval byte pooling on rows, window/stride from the config), `RowHead` (FFN ×4, RMS-normalized × learned scale), `RowCache` (CPU bf16, LRU by bytes), `EncodeQueue` (one worker, 5 ms drain, length-sorted batches, futures). `ac_model/ac_model.py`: `ActivationContextModelConfig` (the planned fields plus `mixer_heads=8`, `side_lora_rank=128`, `encode_batch_max_tokens=65_536`, `side_gradient_checkpointing=True`), `ActivationContextModules` (fp32: mixer, pooling, marker kind/index, recursive adapter, target head, a `scales_initialized` buffer), `ActivationContextModel` with `encode / encode_batch / encode_async / part_view_rows / num_view_rows / set_mode / prepare / release / save / load / trainable_parameters` and `ActivationContextModelStats`. `ModuleManager.register_ac_model / get_ac_model / ac_models`; `LoadedModel.engine_to_device` subtracts `ac_engine_memory_reservation` (0.1) from `gpu_memory_utilization` when an AC model is registered before the engine builds; `HarnessRuntimeConfig.ac_row_cache_bytes` (4 GB).

**Drifts.** (1) The side model's own input embedding table is used directly (the side base is resident whenever the AC model runs), so no `embedding_layer_to_device` copy for the side; the target's copy is used only to initialize the head scale when the target is not loaded. (2) The head scales initialize lazily (first `prepare`) from the destination embedding RMS and are saved in the checkpoint. (3) `part_view_rows` was added so the trainer can size placeholder runs before encoding (V depends on the side tokenization, not the target's). (4) Routing a part to a different AC model is asserted against, not implemented. (5) `load` re-injects the side adapter from the checkpoint folder through `free_lora` + `checkpoint_path`.

**Validation.** `IB/TMP/SLICE3_CHECKS/ac_smoke.py` on CPU (Qwen3.5-0.8B as side and target): a 1,095-token part → [64, 1024] rows in 5.8 s; cache hit on the second call; a nested part costs three network passes (child, parent, and the earlier top-level); batched vs single rows differ by ≤ 0.009 at RMS 0.019 (bf16 decoder noise; the mixer runs fp32 on CPU); four queued requests form one batch; in training mode gradients reach all 46 module tensors and 96 of 192 side-LoRA tensors (the B matrices; A's gradient is zero at B = 0); save → `AC_MODELS/ac_dev/round_000/{ac_modules.pt (219 MB), side_lora/}` + `latest`, load restores a mutated scale, version 1. CPU runs need `sys.modules["fla"] = None` before importing transformers (see the round document).

### M4 — completion report

**What landed.** `ac_model/ac_model_study.py`: `ActivationContextStudyGenerator(harness, ac_model_name)` with `generate_compaction_samples(dataset_id, num_samples, seed=0, depth_range=(1, 2), threshold_range_tokens=(8192, 32768), ratios_range=(1/8, 1/16), completion_max_tokens=512, max_total_tokens=72_000, min_trajectory_tokens=1024)`, `generate_trajectory_qa_samples(dataset_id, num_samples, depth_range, trajectory_tokens_range, distractors_range, ratios_range, seed, caching_id)`, `generate_rag_qa_samples(dataset_id, num_samples, depth_range, distractors_token_range, ratios_range, seed, caching_id)`; `ac_part(messages, ac_name, ratio)`; the compaction instructions constant; items carry `info` (depth, thresholds, ratio, gold position, `gold_chunk_text`).

**Drifts.** No teacher-sampled fallback: every item's completion is real text (the plan's fallback was for cuts without a continuation; the generator moves or drops such cuts instead). Items carry `completion_text` / `completion_complete` / `teacher_partial_text` / `tools` / `info` instead of a `completion` message list. Round 2: the total bound of a compaction item moved from 48k to 72k tokens so a depth-2 item at the default 32k threshold is no longer cut short; the threshold range itself is unchanged (the bench generates at 8k–24k per level).

**Validation.** `IB/TMP/SLICE3_CHECKS/ac_train_smoke.py` on fabricated trajectories and passages (3 compaction items at depths 1–2, 2 traj_qa, 2 rag_qa, every item with parts, spans and a completion). Node: 2,200 compaction items generated from Open-SWE v1.2 and Nemotron tir (≥ 40k chars) in about 4 minutes, cached at `AC_ITEMS/compaction_0.jsonl` (300 MB); the test generators produced 48 items per kind with real 27B study questions for the QA kinds.

### M5 — completion report

**What landed.** `ac_model/ac_model_training.py`: `ActivationContextTrainingItem`, `ActivationContextTrainingConfig` (the planned fields plus `examples_per_update`, `fp32_head_matmul`, `teacher_cache_top_k`), `ActivationContextTrainingStats` (per-step KL / agreement / drift / grad norm / seconds / tokens, per-kind lists, per-example records with `seconds_by_bucket()`, teacher-cache counters), `ActivationContextTrainer.build_example / train / eval`. Teacher ids = template(in-context prefix, generation prompt) + partial text + completion (+ eot when complete); student ids = `tokenize_with_parts(target tokenizer, ac_prefix, [V per part])` + the same completion; the rows come from `encode_batch` in training mode; loss = mean full-vocabulary KL(teacher ‖ student) over the completion positions, the head one chunk at a time under `torch.utils.checkpoint`; AdamW in two groups (5e-4 for the AC modules and the side LoRA, 2e-5 for the target adapter); `examples_per_update` or `updates_per_round`; the drift term when `drift_term_weight > 0`; checkpoints (`AC_MODELS/<name>/round_<n>` + the target LoRA) per round; models back to host RAM after the round. Round 2 additions: (1) `configure_fla_cache(device)` at placement, as the agent trainer does, so fla's Triton kernels load the checked-in configs instead of re-tuning per length bucket; (2) `max_example_tokens` 52k → 72k; (3) `teacher_cache_top_k` (default 0 = exact teacher every pass): after an item's first pass its teacher top-k log-probs and the log of the remainder mass are kept per completion position on the CPU (`_teacher_top_k`), and later passes skip the teacher forward and minimize the KL of the k+1-way coarsened distributions (`_chunked_kl(..., cached=...)`: a lower bound of the exact KL, the same loss in every epoch; `eval` is always exact); (4) per-example records (kind, teacher / student tokens, seconds, cached) for the throughput read-out. Review pass (during run A): (5) the optimizer persists across rounds (one AdamW per AC model, `persistent_optimizer`; moments parked on the host between rounds) instead of a fresh AdamW per `train()` call, which reset Adam's moments and bias correction at every chunk boundary; (6) `warmup_updates` (linear, on a cross-round update clock); (7) a mid-round `_evaluate` no longer re-enables dropout modules when it restores training mode, and it counts the items it drops for length.

**Drifts.** The reporting items are evaluated with `eval` after the round (not inside the step loop). Placeholder runs are sized by `part_view_rows` and asserted against the encoded rows. The cached teacher is stale by design (the adapter as of the item's last exact pass); with the target adapter at 2e-5 that is the approximation the user accepted, measured below.

**Validation.** CPU (`IB/TMP/SLICE3_CHECKS/ac_train_smoke.py`, `ac_perf_smoke.py`): 7 fabricated items, two rounds, held KL 3.20 → 2.19; on a toy head the coarsened KL is ≤ the exact one for k = 1, 5 and equal at k = vocabulary; two rounds with `teacher_cache_top_k=32` on 4 items: 0 then 4 cache hits, finite losses, exact `eval` afterwards. Node: the three tests (M8) and the profile (M9) below; the cached-teacher gap is 0.1–1.0% of the KL at k = 64 on real items.

### M8 — completion report

**What landed.** `activation/tests/test_basic_agent_ac_training.py`: `test_ac_compaction` (Open-SWE v1.2 ≥ 40k chars + Nemotron tir ≥ 40k chars with ≥ 1 call, 24 + 24 items), `test_ac_trajectory_qa` (S1 + Open-SWE, 48 study questions through the 27B QA engine, cached under `ac_training_test`), `test_ac_rag_qa` (BRIGHT biology, 48 questions); side `qwen3.5-0.8b`, target `qwen3.5-4b`, AC `ac_dev`, target LoRA `ac_target` rank 64, side LoRA `ac_side` rank 128; each test: held KL of the untrained model and of the no-context reference, one round of `examples_per_update=8` on 40 items, asserts finite KL, checkpoints, and held KL below the no-context reference; teardown frees the adapters before the bases.

**Drifts.** The `recent_text` reference and the secondary metrics live in the M9 driver, not in the tests. The first node run failed in teardown only (freeing a base with adapters still injected); fixed by `free_lora` on both adapters first.

**Validation.** Node `ac-fp4-probe` (RTX PRO 6000, 96 GB), 2026-09-09: `3 passed in 823 s`. Held KL untrained / no-context / after one round: compaction 0.579 / 0.574 / 0.525 (agreement 0.765 → 0.779); trajectory QA 1.055 / 0.936 / 0.917 (0.674 → 0.675); RAG QA 0.897 / 0.869 / 0.849 (0.747 → 0.733). Pages under `IB/TMP/SYNC/AC_TRAINING_TEST/{compaction,traj_qa,rag_qa}/`, log `IB/TMP/AGENT_ROLLOUTS/ac_tests_run3.log`. The compaction round ran at 9,714 tokens/s with the fla configs against 4,681 in the first run without them (steps 84 / 53 / 26 / 22 / 36 s → 19 / 25 / 19 / 22 / 22 s).

### M9 — completion report

**What landed.** `activation/bench/agent_probes/ac_training_bench.py` (`--kind {compaction,traj_qa,rag_qa,mixed} --items 2000 --held 200 --epochs 2 --examples-per-update 16 --eval-every 50 --side --target --qa-model --ratio-range --threshold-range 8192,24576 --max-total-tokens 72000 --max-example-tokens --teacher-cache-top-k 0 --side-lora-rank --target-lora-rank --lr-ac --lr-target --secondary-items 100 --secondary-tokens 64 --seed --tag --generate-only`): items generated once per kind and cached at `AC_ITEMS/<kind>_<seed>.jsonl`, references `no_context`, `recent_text`, `untrained_ac`, epochs with held evals, the greedy secondary metric, `summary.json` with a `throughput` block (examples and teacher tokens per hour, by length bucket, live vs cached teacher). Review pass: `split_items` holds out in exact proportion to the sources (the head slice used before was mixed only by the seeded shuffle) and drops training items that share their origin (trajectory, study question) with a held-out item; `--warmup-updates 10`. `activation/bench/agent_probes/ac_training_profile.py` (in the tree now): the card's bf16 GEMM ceiling, examples bucketed by teacher length up to 72k plus synthetic depth-2 compaction items at requested lengths (a real trajectory's messages cycled, measured and rescaled to within 5%), per phase time / peak memory / counted FLOPs / achieved TFLOPS, the exact-vs-cached-teacher KL gap, an fla kernel-cache counter and `--dump-fla-configs`.

**Drifts.** The secondary metric defaults to 100 held items. The `in_context` reference is not computed (it is the teacher). The probe is a tree file, not IB-only (the user asked for the performance work to be part of the 3a handoff). The plan's phase-0 lever list was reordered by measurement: loading the fla configs in the AC trainer and the teacher cache are the two that matter (the 0.8B needed no tuning of its own: fuzzy matching serves it from the 4B's entries); head chunks and the checkpointing threshold have nothing to give (the head is 0.01 s per example, the student sequence is below the card's recompute threshold); child batching would touch at most the 7–11% that the encode costs and was not done.

**Validation.** Node, 2026-09-09: the profile before and after the 0.8B's kernel configs, the cached-teacher gap, the caps, and the validity runs are reported in `SLICE3_ROUND.tressoir.md` under "Node results" and summarized in the M9 node results below.


## The deletions (run first, in your checkout)

`activation/retrieval/`, `activation/bench/agent_probes/retrieval_training_bench.py`, `activation/tests/test_basic_retrieval_training.py` · archived under `IB/ARTIFACTS/RETRIEVAL_TRAINING/archive_slice3/`

```bash
rm -r activation/retrieval
rm activation/bench/agent_probes/retrieval_training_bench.py activation/tests/test_basic_retrieval_training.py
```

The cards below are exact deltas against `/source` (HEAD `1cfa058` plus your uncommitted stubs: `activation/ac_model/*` notes, the empty `dataset_caching.py`, the `test_basic_agent_ac_training.py` docstring); the copies under `slice3/` are byte-for-byte what the cards produce. `uv.lock` is supplied alongside (regenerated by `uv lock` after removing `faiss-cpu`; nothing else changed). The kernel-config cards are JSON: apply them by copying the staged files.

## Diffs to apply

At build time `/source` already matched the staged tree for 44 of the 46 files (the application has started); the cards still to apply are `activation/bench/agent_probes/ac_training_bench.py`, `activation/bench/agent_probes/ac_training_profile.py`. Cards of applied files say so and carry no diff.

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/ac_model/__init__.py</span>
    <span class="card-oneliner">Package exports: model, config, stats, trainer, items, study generator, reporter.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/ac_model/__init__.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/ac_model/ac_model.py</span>
    <span class="card-oneliner">The AC model: config, modules, encode / encode_batch / encode_async, part_view_rows, modes, save / load; optional encode phase timings.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/ac_model/ac_model.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/ac_model/ac_model_reporter.py</span>
    <span class="card-oneliner">HTML report of AC training: KL with reference lines, agreement, throughput, tables, completions.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/ac_model/ac_model_reporter.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/ac_model/ac_model_study.py</span>
    <span class="card-oneliner">Item generators: compaction (nested cuts, total ≤ 72k), trajectory QA (window + distractor parts), RAG QA (one part of passages).</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/ac_model/ac_model_study.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/ac_model/ac_model_training.py</span>
    <span class="card-oneliner">Items, config, stats and the trainer: teacher/student sequences, chunked KL (exact or against the cached top-k teacher), fla configs, 72k cap, per-example records.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/ac_model/ac_model_training.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/ac_model/ac_model_utils.py</span>
    <span class="card-oneliner">tokenize_with_parts, blocked local attention, windowed pooling, row head, row cache, encode queue.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/ac_model/ac_model_utils.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/agent_training/agent_trainer.py</span>
    <span class="card-oneliner">Review pass: the adapter's AdamW persists across rounds (moments parked on the host between rounds), linear warm-up, the update clock.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/agent_training/agent_trainer.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/agent_training/agent_training_config.py</span>
    <span class="card-oneliner">Review pass: `warmup_updates` (0 by default).</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/agent_training/agent_training_config.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/agent_training/agent_training_utils.py</span>
    <span class="card-oneliner">Review pass: `persistent_optimizer`, `optimizer_state_to`, `set_learning_rates` shared by both trainers.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/agent_training/agent_training_utils.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/agent_training/fla_cache.py</span>
    <span class="card-oneliner">live_autotuners tolerates a missing fla (CPU checks).</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/agent_training/fla_cache.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_bwd_kernel_dqkwg.json</span>
    <span class="card-oneliner">fla kernel config for this GPU type: earlier entries kept, the 0.8B's and the new length buckets merged in.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_bwd_kernel_dqkwg.json`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_bwd_kernel_dv_local.json</span>
    <span class="card-oneliner">fla kernel config for this GPU type: earlier entries kept, the 0.8B's and the new length buckets merged in.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_bwd_kernel_dv_local.json`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_fwd_kernel_o.json</span>
    <span class="card-oneliner">fla kernel config for this GPU type: earlier entries kept, the 0.8B's and the new length buckets merged in.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_fwd_kernel_o.json`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64.json</span>
    <span class="card-oneliner">fla kernel config for this GPU type: earlier entries kept, the 0.8B's and the new length buckets merged in.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64.json`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_gated_delta_rule_fwd_kernel_h_blockdim64.json</span>
    <span class="card-oneliner">fla kernel config for this GPU type: earlier entries kept, the 0.8B's and the new length buckets merged in.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_gated_delta_rule_fwd_kernel_h_blockdim64.json`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_gated_delta_rule_fwd_kkt_solve_kernel.json</span>
    <span class="card-oneliner">fla kernel config for this GPU type: earlier entries kept, the 0.8B's and the new length buckets merged in.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_gated_delta_rule_fwd_kkt_solve_kernel.json`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_local_cumsum_scalar_kernel.json</span>
    <span class="card-oneliner">fla kernel config for this GPU type: earlier entries kept, the 0.8B's and the new length buckets merged in.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/chunk_local_cumsum_scalar_kernel.json`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/l2norm_bwd_kernel.json</span>
    <span class="card-oneliner">fla kernel config for this GPU type: earlier entries kept, the 0.8B's and the new length buckets merged in.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/l2norm_bwd_kernel.json`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/l2norm_fwd_kernel.json</span>
    <span class="card-oneliner">fla kernel config for this GPU type: earlier entries kept, the 0.8B's and the new length buckets merged in.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/l2norm_fwd_kernel.json`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/recompute_w_u_fwd_kernel.json</span>
    <span class="card-oneliner">fla kernel config for this GPU type: earlier entries kept, the 0.8B's and the new length buckets merged in.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/agent_training/fla_configs/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/recompute_w_u_fwd_kernel.json`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/bench/agent_probes/ac_training_bench.py</span>
    <span class="card-oneliner">M9 driver: cached items, references, epochs with held evals, greedy secondary metric, throughput summary; generation and cache flags.</span>
    <span class="card-badge">Diff</span>
  </summary>

New file for `activation/bench/agent_probes/ac_training_bench.py` (416 lines):

```diff
diff --git ab/activation/bench/agent_probes/ac_training_bench.py b/activation/bench/agent_probes/ac_training_bench.py
new file mode 100644
index 0000000..62292d3
--- /dev/null
+++ b/activation/bench/agent_probes/ac_training_bench.py
@@ -0,0 +1,410 @@
+"""
+AC training validity run: items of one kind (compaction, traj_qa, rag_qa; or the three caches mixed),
+three forward-only reference points on the held-out set, N epochs of training with a held-out KL
+after every epoch (and every `--eval-every` updates in epoch 1), a secondary metric (QA answer
+accuracy or compaction next-action agreement) from greedy HF decoding, a live page and a summary
+JSON under `AC_BENCH/<tag>/`. Items are generated once (the study questions need the QA engine)
+and cached under `AC_ITEMS/<kind>_<seed>.jsonl`; the engine is asleep for the whole training.
+
+  uv run sky exec --sync <node> -- uv run python -m activation.bench.agent_probes.ac_training_bench --kind compaction --items 2000 --held 200 --epochs 2 --tag compaction_r1
+
+References (same target adapter as the teacher, i.e. the base at run start):
+  no_context   the AC prefix with every part removed (task + instructions only)
+  recent_text  every part replaced by verbatim text of the same token budget as its rows (the last V
+               tokens of a compacted span; the gold chunk's first V tokens for the QA kinds)
+  untrained_ac the AC prefix through the untrained AC model
+  in_context   the teacher itself (KL 0, agreement 1, by construction)
+"""
+import argparse
+import dataclasses
+import json
+import random
+import re
+import time
+from dataclasses import replace
+from pathlib import Path
+
+import torch
+
+from activation.ac_model import (
+    ActivationContextModelConfig,
+    ActivationContextStudyGenerator,
+    ActivationContextTrainer,
+    ActivationContextTrainingConfig,
+    ActivationContextTrainingItem,
+    ActivationContextTrainingReporter,
+    ActivationContextTrainingStats,
+)
+from activation.ac_model.ac_model_utils import direct_parts
+from activation.agent.agent_utils import ModelDialect
+from activation.common.data_syncing import resolve_path
+from activation.dataset.dataset_utils import render_messages
+from activation.dataset.loaders import BrightDataset, NemotronMathDataset, OpenSweTracesDataset, S1DeepResearchDataset, aime_problem_statements
+from activation.harness import FREE_DEVICE, SOURCE_DEVICE, SUPPORTS_FP4, HarnessRuntime, HarnessRuntimeConfig, ModelConfig
+
+KINDS = ("compaction", "traj_qa", "rag_qa")
+SIDE_NAME, TARGET_NAME, AC_NAME, SIDE_LORA, TARGET_LORA = "side", "target", "ac_bench", "ac_side", "ac_target"
+DEFAULT_RATIOS = {"compaction": (1 / 16, 1 / 8), "traj_qa": (1 / 32, 1 / 16), "rag_qa": (1 / 32, 1 / 16)}
+
+
+# ------------------------------------------------------------------------------------------------ items
+def item_cache_path(kind: str, seed: int) -> Path:
+    return resolve_path(f"AC_ITEMS/{kind}_{seed}.jsonl")
+
+
+def write_items(path: Path, items: list[ActivationContextTrainingItem]) -> None:
+    with open(path, "w") as handle:
+        for item in items:
+            handle.write(json.dumps(dataclasses.asdict(item), ensure_ascii=False) + "\n")
+
+
+def read_items(path: Path) -> list[ActivationContextTrainingItem]:
+    with open(path) as handle:
+        return [ActivationContextTrainingItem(**json.loads(line)) for line in handle if line.strip()]
+
+
+def generate_items(harness: HarnessRuntime, kind: str, total: int, seed: int, ratios: tuple[float, float], qa_model: str | None,
+                   thresholds: tuple[int, int] = (8192, 24576), max_total_tokens: int = 72_000) -> list[ActivationContextTrainingItem]:
+    generator = ActivationContextStudyGenerator(harness, AC_NAME)
+    half = -(-total // 2)
+    if kind == "compaction":
+        swe = OpenSweTracesDataset.load(harness, max_examples=half + 50, min_chars=40_000)
+        math = NemotronMathDataset.load(harness, max_examples=half + 50, subset="tir", excluded_problems=aime_problem_statements(), min_chars=40_000, min_tool_calls=1)
+        items = (generator.generate_compaction_samples(swe.dataset_id, half, seed=seed, threshold_range_tokens=thresholds, ratios_range=ratios, max_total_tokens=max_total_tokens)
+                 + generator.generate_compaction_samples(math.dataset_id, half, seed=seed, threshold_range_tokens=thresholds, ratios_range=ratios, max_total_tokens=max_total_tokens))
+    elif kind == "traj_qa":
+        s1 = S1DeepResearchDataset.load(harness, max_examples=half + 50)
+        swe = OpenSweTracesDataset.load(harness, max_examples=half + 50)
+        harness.dataset_manager.build_bm25_indexes()
+        items = (generator.generate_trajectory_qa_samples(s1.dataset_id, half, seed=seed, ratios_range=ratios, caching_id=f"ac_bench_{seed}")
+                 + generator.generate_trajectory_qa_samples(swe.dataset_id, half, seed=seed, ratios_range=ratios, caching_id=f"ac_bench_{seed}"))
+    elif kind == "rag_qa":
+        items = []
+        for domain in ("biology", "economics"):
+            bright = BrightDataset.load(harness, max_examples=None, domain=domain, max_corpus_documents=20_000)
+            harness.dataset_manager.build_bm25_indexes()
+            items += generator.generate_rag_qa_samples(bright.dataset_id, half, seed=seed, ratios_range=ratios, caching_id=f"ac_bench_{seed}")
+    else:
+        raise ValueError(kind)
+    if qa_model and qa_model in harness.loaded_models:
+        harness.loaded_models[qa_model].engine_to_device(FREE_DEVICE)
+    random.Random(seed).shuffle(items)
+    return items
+
+
+# ------------------------------------------------------------------------------------------------ references
+def text_only(messages: list[dict]) -> list[dict]:
+    out = []
+    for message in messages:
+        content = message.get("content")
+        if isinstance(content, list):
+            message = dict(message, content="".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"))
+        out.append(message)
+    return out
+
+
+def split_items(items: list[ActivationContextTrainingItem], held: int, train: int) -> tuple[list, list, int]:
+    """
+    Held-out items in exact proportion to the sources (the generated list is shuffled once by seed, so a
+    head slice is mixed only in expectation), then training items among the rest whose origin (the
+    trajectory of a compaction item, the study question of a QA item) is not held out. Returns
+    (held, train, dropped for sharing an origin).
+    """
+    by_source: dict[str, list] = {}
+    for item in items:
+        by_source.setdefault(item.dataset_id, []).append(item)
+    held_items: list = []
+    for index, (source, source_items) in enumerate(by_source.items()):
+        quota = round(held * len(source_items) / len(items)) if index < len(by_source) - 1 else held - len(held_items)
+        held_items += source_items[:quota]
+    held_ids = {item.item_id for item in held_items}
+    held_origins = {item_origin(item) for item in held_items}
+    train_items, shared = [], 0
+    for item in items:
+        if item.item_id in held_ids:
+            continue
+        if item_origin(item) in held_origins:
+            shared += 1
+            continue
+        train_items.append(item)
+    return held_items, train_items[:train], shared
+
+
+def item_origin(item: ActivationContextTrainingItem) -> str:
+    """What a held-out item must not share with a training item: its trajectory (compaction) or its study question (QA kinds)."""
+    if item.kind == "compaction":
+        return f"doc:{item.doc_ids[0] if item.doc_ids else item.item_id}"
+    return f"question:{item.info.get('example_id', item.item_id)}"
+
+
+def no_context_item(item: ActivationContextTrainingItem) -> ActivationContextTrainingItem:
+    return replace(item, item_id=item.item_id + ":no_context", ac_prefix=text_only(item.ac_prefix))
+
+
+def recent_text_item(item: ActivationContextTrainingItem, ac_model, tokenizer) -> ActivationContextTrainingItem:
+    """
+    Every part becomes verbatim text of V tokens (V = the rows it would have become): the tail of the
+    compacted span for compaction; for the QA kinds the gold chunk's first V tokens in the gold part's
+    place (rag_qa has one part; traj_qa's distractor parts contribute nothing).
+    """
+    gold_position = item.info.get("gold_position", 0)
+    gold_text = item.info.get("gold_chunk_text") or ""
+    messages = []
+    part_index = 0
+    for message in item.ac_prefix:
+        content = message.get("content")
+        if isinstance(content, list):
+            pieces = []
+            for part in content:
+                if isinstance(part, dict) and part.get("type") == "activation_context":
+                    rows = ac_model.part_view_rows(part["messages"], part.get("compression_target"))
+                    if item.kind == "compaction":
+                        ids = tokenizer.encode(render_messages(part["messages"]), add_special_tokens=False)
+                        pieces.append(tokenizer.decode(ids[-rows:]))
+                    elif item.kind == "rag_qa" or part_index == gold_position:
+                        pieces.append(tokenizer.decode(tokenizer.encode(gold_text, add_special_tokens=False)[:rows]))
+                    part_index += 1
+                elif isinstance(part, dict):
+                    pieces.append(part.get("text", ""))
+            message = dict(message, content="".join(pieces))
+        messages.append(message)
+    return replace(item, item_id=item.item_id + ":recent_text", ac_prefix=messages)
+
+
+# ------------------------------------------------------------------------------------------------ secondary metric
+def normalize_answer(text: str) -> str:
+    return re.sub(r"[^0-9a-z]+", " ", text.lower()).strip()
+
+
+def greedy_text(harness, ac_model, trainer, item: ActivationContextTrainingItem, side: str, max_new_tokens: int) -> str:
+    """Greedy HF decoding (KV cache, the target adapter, rows for the student) of the teacher ('teacher') or the student's prefix."""
+    example = trainer.build_example(ac_model, item)
+    target = ac_model.target
+    base = target.model
+    embedding = base.get_input_embeddings()
+    device = next(base.parameters()).device
+    lora = ac_model.config.target_model_lora_name
+    if side == "teacher":
+        ids = torch.tensor(example.teacher_ids[:len(example.teacher_ids) - example.num_completion], device=device)
+        embeds = embedding(ids)
+    else:
+        prefix_len = len(example.student_ids) - example.num_completion
+        ids = torch.tensor(example.student_ids[:prefix_len], device=device)
+        embeds = embedding(ids)
+        if example.part_requests:
+            rows = ac_model.encode_batch(example.part_requests)
+            pieces, cursor = [], 0
+            for (start, end), part_rows in zip(example.spans, rows):
+                pieces.extend([embeds[cursor:start], part_rows.to(embeds.dtype)])
+                cursor = end
+            pieces.append(embeds[cursor:])
+            embeds = torch.cat(pieces, dim=0)
+    peft_model = harness.module_manager.ensure_lora(lora)
+    with harness.module_manager.lora_context(target.model_config.model_name, lora), torch.inference_mode():
+        out = peft_model.generate(inputs_embeds=embeds[None], max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
+                                  pad_token_id=target.tokenizer.pad_token_id or target.tokenizer.eos_token_id)
+    return target.tokenizer.decode(out[0], skip_special_tokens=True)
+
+
+def first_call(dialect: ModelDialect, text: str) -> tuple[str, str] | None:
+    """(tool name, normalized argument text) of the first call in a generated text, or None."""
+    _, calls = dialect.parse(text)
+    if not calls:
+        return None
+    call = calls[0]
+    arguments = call["arguments"] if isinstance(call["arguments"], dict) else {"raw": call["arguments"]}
+    return call["name"], " ".join(normalize_answer(str(value)) for _, value in sorted(arguments.items()))
+
+
+def call_agreement(reference: tuple[str, str] | None, candidate: tuple[str, str] | None) -> dict[str, float]:
+    """Against the teacher's first call: made a call at all, same tool name, same arguments, and the token overlap (Jaccard) of the arguments."""
+    if reference is None:
+        return {}
+    if candidate is None:
+        return {"call": 0.0, "name": 0.0, "exact": 0.0, "args_overlap": 0.0}
+    same_name = float(candidate[0] == reference[0])
+    a, b = set(reference[1].split()), set(candidate[1].split())
+    overlap = len(a & b) / len(a | b) if (a | b) else 1.0
+    return {"call": 1.0, "name": same_name, "exact": float(same_name and candidate[1] == reference[1]), "args_overlap": same_name * overlap}
+
+
+def secondary_metric(harness, ac_model, trainer, items, variants: dict[str, list[ActivationContextTrainingItem]], max_new_tokens: int) -> dict:
+    """
+    QA: answer accuracy (normalized contains) per variant and the teacher. Compaction: agreement of each variant's first
+    tool call with the teacher's greedy continuation over the items where the teacher calls a tool (`<variant>_name`,
+    `<variant>_exact`, `<variant>_args_overlap`, `<variant>_call`; `teacher_calls` is that share of items).
+    """
+    dialect = ModelDialect.for_tokenizer(ac_model.target.tokenizer)
+    results: dict[str, float] = {}
+    counted = 0
+    samples = []
+    started = time.time()
+    for index, item in enumerate(items):
+        teacher_text = greedy_text(harness, ac_model, trainer, item, "teacher", max_new_tokens)
+        texts = {name: greedy_text(harness, ac_model, trainer, variant_items[index], "student", max_new_tokens) for name, variant_items in variants.items()}
+        if item.kind == "compaction":
+            reference = first_call(dialect, teacher_text)
+            results["teacher_calls"] = results.get("teacher_calls", 0.0) + float(reference is not None)
+            if reference is not None:
+                counted += 1
+                for name, text in texts.items():
+                    for key, value in call_agreement(reference, first_call(dialect, text)).items():
+                        results[f"{name}_{key}"] = results.get(f"{name}_{key}", 0.0) + value
+        else:
+            counted += 1
+            gold = normalize_answer(item.completion_text)
+            results["teacher"] = results.get("teacher", 0.0) + float(gold in normalize_answer(teacher_text))
+            for name, text in texts.items():
+                results[name] = results.get(name, 0.0) + float(gold in normalize_answer(text))
+        if len(samples) < 6:
+            samples.append({"item_id": item.item_id, "kind": item.kind, "reference": item.completion_text[:200], "teacher": teacher_text[:200],
+                            "student": texts.get("student", "")[:200]})
+    print(f"secondary metric on {len(items)} items in {time.time() - started:.0f}s", flush=True)
+    out = {"items": len(items), "counted": counted}
+    for key, value in results.items():
+        out[key] = value / max(1, len(items)) if key == "teacher_calls" else value / max(1, counted)
+    return {**out, "samples": samples}
+
+
+def throughput_summary(records: list[dict]) -> dict:
+    """Examples per hour and in-context (teacher) tokens per hour over the whole run, by teacher-length bucket and by whether the teacher was cached."""
+    out = {}
+    for label, selected in (("all", records), ("teacher_live", [r for r in records if not r["teacher_cached"]]), ("teacher_cached", [r for r in records if r["teacher_cached"]])):
+        seconds = sum(record["seconds"] for record in selected)
+        if not selected or seconds <= 0:
+            continue
+        stats = ActivationContextTrainingStats(example_records=selected)
+        out[label] = {"examples": len(selected), "examples_per_hour": round(3600 * len(selected) / seconds),
+                      "teacher_tokens_per_hour": round(3600 * sum(record["teacher_tokens"] for record in selected) / seconds),
+                      "mean_teacher_tokens": round(sum(record["teacher_tokens"] for record in selected) / len(selected)), "by_bucket": stats.seconds_by_bucket()}
+    return out
+
+
+# ------------------------------------------------------------------------------------------------ main
+def main() -> None:
+    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
+    parser.add_argument("--kind", choices=KINDS + ("mixed",), required=True)
+    parser.add_argument("--items", type=int, default=2000, help="training items (mixed: per kind)")
+    parser.add_argument("--held", type=int, default=200, help="held-out items (mixed: per kind)")
+    parser.add_argument("--epochs", type=int, default=2)
+    parser.add_argument("--examples-per-update", type=int, default=16)
+    parser.add_argument("--eval-every", type=int, default=50, help="held-out eval every this many updates in epoch 1 (0: per epoch only)")
+    parser.add_argument("--side", default="Qwen/Qwen3.5-0.8B")
+    parser.add_argument("--target", default="Qwen/Qwen3.5-4B")
+    parser.add_argument("--qa-model", default="unsloth/Qwen3.8-27B-NVFP4" if SUPPORTS_FP4 else "Qwen/Qwen3.8-27B-FP8", help="study question engine (QA kinds, when the item cache is missing)")
+    parser.add_argument("--ratio-range", default=None, help="compression ratios low,high (default per kind)")
+    parser.add_argument("--threshold-range", default="8192,24576", help="compaction segment thresholds low,high in tokens (item generation only)")
+    parser.add_argument("--max-total-tokens", type=int, default=72_000, help="compaction items: bound on the segments' total (item generation only)")
+    parser.add_argument("--max-example-tokens", type=int, default=None, help="trainer cap on the teacher sequence (default: the trainer's)")
+    parser.add_argument("--warmup-updates", type=int, default=10, help="linear learning-rate warm-up over this many updates (0: none)")
+    parser.add_argument("--teacher-cache-top-k", type=int, default=0, help="> 0: cached top-k teacher after an item's first pass (see ActivationContextTrainingConfig)")
+    parser.add_argument("--side-lora-rank", type=int, default=128)
+    parser.add_argument("--target-lora-rank", type=int, default=64)
+    parser.add_argument("--lr-ac", type=float, default=5e-4)
+    parser.add_argument("--lr-target", type=float, default=2e-5)
+    parser.add_argument("--secondary-items", type=int, default=100, help="held items for the greedy secondary metric (0: off)")
+    parser.add_argument("--secondary-tokens", type=int, default=64)
+    parser.add_argument("--seed", type=int, default=0)
+    parser.add_argument("--tag", default=None)
+    parser.add_argument("--generate-only", action="store_true", help="generate and cache the items, then stop (the profile probe reads the cache)")
+    args = parser.parse_args()
+    tag = args.tag or f"{args.kind}_s{args.seed}"
+    kinds = list(KINDS) if args.kind == "mixed" else [args.kind]
+    ratios = tuple(float(value) for value in args.ratio_range.split(",")) if args.ratio_range else None
+
+    # Which caches are missing decides whether the QA engine is needed.
+    missing = [kind for kind in kinds if not item_cache_path(kind, args.seed).exists()]
+    needs_qa = any(kind in ("traj_qa", "rag_qa") for kind in missing)
+    model_configs = {SIDE_NAME: ModelConfig(SIDE_NAME, args.side), TARGET_NAME: ModelConfig(TARGET_NAME, args.target)}
+    if needs_qa:
+        model_configs[args.qa_model] = ModelConfig(args.qa_model, args.qa_model)
+    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs=model_configs, dataset_study_qa_model_name=args.qa_model if needs_qa else None))
+    harness.module_manager.register_lora(TARGET_LORA, TARGET_NAME, rank=args.target_lora_rank)
+    ac_model = harness.module_manager.register_ac_model(ActivationContextModelConfig(AC_NAME, SIDE_NAME, SIDE_LORA, TARGET_NAME, TARGET_LORA, side_lora_rank=args.side_lora_rank))
+    train_items, held_items = [], []
+    for kind in kinds:
+        path = item_cache_path(kind, args.seed)
+        if path.exists():
+            items = read_items(path)
+            print(f"{kind}: {len(items)} items from {path}")
+        else:
+            items = generate_items(harness, kind, args.items + args.held, args.seed, ratios or DEFAULT_RATIOS[kind], args.qa_model if needs_qa else None,
+                                   thresholds=tuple(int(value) for value in args.threshold_range.split(",")), max_total_tokens=args.max_total_tokens)
+            write_items(path, items)
+            print(f"{kind}: {len(items)} items generated and cached at {path}")
+        assert len(items) > args.held, f"{kind}: {len(items)} items, {args.held} held"
+        held, train, shared = split_items(items, args.held, args.items)
+        held_items += held
+        train_items += train
+        print(f"{kind}: {len(held)} held-out, {len(train)} training ({shared} items sharing a document with a held-out item dropped)", flush=True)
+    random.Random(args.seed).shuffle(train_items)
+    print(f"{len(train_items)} training items, {len(held_items)} held-out", flush=True)
+    if args.generate_only:
+        return
+
+    report_folder = resolve_path(f"AC_BENCH/{tag}")
+    reporter = ActivationContextTrainingReporter(str(report_folder), f"AC training: {tag}", f"{args.kind}, {len(train_items)} items, {args.epochs} epochs, side {args.side}, target {args.target}")
+    training_config = ActivationContextTrainingConfig(examples_per_update=args.examples_per_update, learning_rate_ac=args.lr_ac, learning_rate_target_lora=args.lr_target,
+                                                      seed=args.seed, teacher_cache_top_k=args.teacher_cache_top_k, warmup_updates=args.warmup_updates)
+    if args.max_example_tokens:
+        training_config.max_example_tokens = args.max_example_tokens
+    trainer = ActivationContextTrainer(harness, training_config)
+    tokenizer = ac_model.target.tokenizer
+    variants = {"no_context": [no_context_item(item) for item in held_items],
+                "recent_text": [recent_text_item(item, ac_model, tokenizer) for item in held_items]}
+    summary = {"tag": tag, "kind": args.kind, "args": vars(args), "train_items": len(train_items), "held_items": len(held_items), "references": {}, "epochs": [], "evals": []}
+    started = time.time()
+    for name, items in list(variants.items()) + [("untrained_ac", held_items)]:
+        result = trainer.eval(AC_NAME, items, release=False)
+        summary["references"][name] = {"kl": result["kl"], "agreement": result["agreement"], "by_kind": result["by_kind"]}
+        reporter.report_reference(name, result["kl"])
+        reporter.report_eval(0, name, result)
+        print(f"reference {name}: kl {result['kl']:.4f} agreement {result['agreement']:.3f}", flush=True)
+    summary["references"]["in_context"] = {"kl": 0.0, "agreement": 1.0}
+    secondary_items = held_items[:args.secondary_items] if args.secondary_items else []
+    if secondary_items:
+        secondary = secondary_metric(harness, ac_model, trainer, secondary_items, {"student": secondary_items, **{name: items[:len(secondary_items)] for name, items in variants.items()}}, args.secondary_tokens)
+        summary["evals"].append({"epoch": 0, "secondary": {key: value for key, value in secondary.items() if key != "samples"}})
+        reporter.report_completions(secondary["samples"])
+        print(f"secondary before training: { {k: round(v, 3) for k, v in secondary.items() if isinstance(v, float)} }", flush=True)
+    (report_folder / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
+
+    updates = 0
+    example_records: list[dict] = []
+    for epoch in range(args.epochs):
+        chunk = args.eval_every * args.examples_per_update if (epoch == 0 and args.eval_every) else len(train_items)
+        for start in range(0, len(train_items), chunk):
+            stats = trainer.train(AC_NAME, train_items[start:start + chunk], reporting_data=held_items, reporter=reporter)
+            updates += stats.steps
+            round_summary = stats.summarize()
+            summary["evals"].append({"epoch": epoch + 1, "updates": updates, "held": {"kl": stats.reporting["kl"], "agreement": stats.reporting["agreement"], "by_kind": stats.reporting["by_kind"]},
+                                     "train_kl_last": stats.kl[-1], "tokens_per_second": round_summary["tokens_per_second"], "peak_memory_gb": round_summary["peak_memory_gb"],
+                                     "seconds_by_bucket": round_summary["seconds_by_bucket"], "teacher_cache_hits": round_summary["teacher_cache_hits"],
+                                     "dropped_too_long": stats.dropped_too_long, "train_seconds": round(sum(stats.step_seconds), 1)})
+            example_records.extend(stats.example_records)
+            (report_folder / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
+        epoch_summary = {"epoch": epoch + 1, "updates": updates, "held_kl": summary["evals"][-1]["held"]["kl"], "held_agreement": summary["evals"][-1]["held"]["agreement"],
+                         "checkpoint": stats.checkpoint_path, "target_lora_checkpoint": stats.target_lora_checkpoint_path}
+        if secondary_items:
+            trainer.eval(AC_NAME, held_items[:1], release=False)                       # places the models again
+            secondary = secondary_metric(harness, ac_model, trainer, secondary_items, {"student": secondary_items, **{name: items[:len(secondary_items)] for name, items in variants.items()}}, args.secondary_tokens)
+            epoch_summary["secondary"] = {key: value for key, value in secondary.items() if key != "samples"}
+            reporter.report_completions(secondary["samples"])
+            ac_model.release()
+            ac_model.target.model_to_device(SOURCE_DEVICE)
+            ac_model.side.model_to_device(SOURCE_DEVICE)
+        summary["epochs"].append(epoch_summary)
+        (report_folder / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
+        print(f"epoch {epoch + 1}: {json.dumps(epoch_summary, default=str)}", flush=True)
+    held_kls = [entry["held"]["kl"] for entry in summary["evals"] if "held" in entry]
+    summary["final_held_kl"] = held_kls[-1] if held_kls else None
+    summary["best_held_kl"] = min(held_kls) if held_kls else None
+    summary["duration_s"] = time.time() - started
+    summary["throughput"] = throughput_summary(example_records)
+    (report_folder / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
+    reporter.finish()
+    print(json.dumps({key: value for key, value in summary.items() if key not in ("evals", "args")}, indent=1, default=str))
+
+
+if __name__ == "__main__":
+    main()
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/bench/agent_probes/ac_training_profile.py</span>
    <span class="card-oneliner">The profile probe: GEMM ceiling, per-phase time / memory / FLOPs, synthetic lengths, cached-teacher gap, fla config dump.</span>
    <span class="card-badge">Diff</span>
  </summary>

New file for `activation/bench/agent_probes/ac_training_profile.py` (407 lines):

```diff
diff --git ab/activation/bench/agent_probes/ac_training_profile.py b/activation/bench/agent_probes/ac_training_profile.py
new file mode 100644
index 0000000..6ef7bb7
--- /dev/null
+++ b/activation/bench/agent_probes/ac_training_profile.py
@@ -0,0 +1,401 @@
+"""
+AC training profile: what one training example costs, per phase, against the card's measured GEMM
+ceiling. Items come from a bench item cache (`AC_ITEMS/<kind>_<seed>.jsonl`) bucketed by teacher
+length, plus synthetic compaction items lengthened to a target (a real trajectory's messages cycled,
+split into `depth` nested segments) for the lengths the cache lacks. For every example it times
+the teacher forward, the AC encode (embedding, mixer, pooling, side decoder, heads), the student
+forward, the head chunks and the backward, records the peak memory of each, counts the FLOPs of
+each from the token counts and the model shapes (dense 2·params·tokens, plus the full-attention
+layers' T² term; the linear-attention layers' chunked scans are omitted, well under 1% at these
+lengths) and reports the achieved TFLOPS next to the ceiling. Also: the exact-vs-cached-teacher KL
+gap for `--teacher-cache-top-k`, the fla kernel cache growth (real benchmarking shows up as Triton's
+autotuning lines under TRITON_PRINT_AUTOTUNING=1), and an optional dump of the resolved configs (`--dump-fla-configs`, into the tree's folder for this GPU plus a copy under
+`AC_BENCH/fla_configs/` that the sync brings back).
+
+  TRITON_PRINT_AUTOTUNING=1 uv run python -m activation.bench.agent_probes.ac_training_profile \\
+      --items-cache AC_ITEMS/compaction_0.jsonl --steps 3 --synthetic 72000:2,104000:2 --teacher-cache-top-k 64 --dump-fla-configs
+"""
+import argparse
+import json
+import shutil
+import time
+from pathlib import Path
+
+import torch
+
+from activation.ac_model import ActivationContextModelConfig, ActivationContextTrainer, ActivationContextTrainingConfig, ActivationContextTrainingItem
+from activation.ac_model.ac_model import TRAINING
+from activation.ac_model.ac_model_study import COMPACTION_INSTRUCTIONS, ac_part
+from activation.agent_training.agent_training_utils import checkpointing_min_tokens, disable_dropout, set_checkpointing
+from activation.agent_training.fla_cache import configure_fla_cache, dump_fla_configs, gpu_config_dir, live_autotuners
+from activation.common.data_syncing import resolve_path
+from activation.dataset.dataset_utils import message_text
+from activation.harness import SOURCE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig
+
+SIDE_NAME, TARGET_NAME, SIDE_LORA, TARGET_LORA = "side", "target", "ac_side", "ac_target"
+DEFAULT_BUCKET_EDGES = "8192,16384,32768,49152,65536,73728"
+
+
+# ------------------------------------------------------------------------------------------------ device helpers
+def sync(device):
+    if device.type == "cuda":
+        torch.cuda.synchronize(device)
+
+
+def peak(device) -> float:
+    return torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
+
+
+def reset_peak(device):
+    if device.type == "cuda":
+        torch.cuda.reset_peak_memory_stats(device)
+
+
+def gemm_ceiling_tflops(device, size: int = 8192, repeats: int = 10) -> float:
+    """bf16 GEMM throughput of the card (size³ matmuls): the ceiling every phase is measured against."""
+    if device.type != "cuda":
+        return 0.0
+    a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
+    b = torch.randn(size, size, device=device, dtype=torch.bfloat16)
+    for _ in range(3):
+        a @ b
+    sync(device)
+    started = time.time()
+    for _ in range(repeats):
+        a @ b
+    sync(device)
+    seconds = time.time() - started
+    del a, b
+    return repeats * 2 * size**3 / seconds / 1e12
+
+
+# ------------------------------------------------------------------------------------------------ FLOP model
+def text_config(model):
+    return model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
+
+
+def decoder_flops(model):
+    """tokens -> forward FLOPs of the decoder (no head): 2·params·T plus the full-attention layers' causal T² term."""
+    params = sum(p.numel() for name, p in model.named_parameters() if "embed_tokens" not in name and "lm_head" not in name and "lora" not in name.lower())
+    config = text_config(model)
+    layer_types = list(getattr(config, "layer_types", None) or [])
+    full_layers = layer_types.count("full_attention") if layer_types else int(config.num_hidden_layers)
+    attention_dim = int(config.num_attention_heads) * int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
+
+    def flops(tokens: int) -> float:
+        return 2.0 * params * tokens + full_layers * 2.0 * tokens * tokens * attention_dim
+    flops.params = params  # type: ignore[attr-defined]
+    return flops
+
+
+def module_params(module) -> int:
+    return sum(p.numel() for p in module.parameters())
+
+
+def items_ac_name(items: list[ActivationContextTrainingItem], default: str = "ac_profile") -> str:
+    """The AC model name the items' parts are addressed to (parts of another model's name are refused by the encoder)."""
+    for item in items:
+        for message in item.ac_prefix:
+            for part in (message.get("content") if isinstance(message.get("content"), list) else []):
+                if isinstance(part, dict) and part.get("type") == "activation_context" and part.get("ac_name"):
+                    return str(part["ac_name"])
+    return default
+
+
+# ------------------------------------------------------------------------------------------------ synthetic items
+def synthetic_item(item: ActivationContextTrainingItem, tokenizer, target_tokens: int, depth: int, ac_name: str, scale: float = 1.0) -> ActivationContextTrainingItem:
+    """
+    A compaction item of about `target_tokens` teacher tokens: the item's in-context messages after the task
+    are cycled until the budget is met, then split into `depth` equal segments nested as the generator nests them.
+    The per-message estimate ignores the chat template's rendering of calls, so `scale` (target / measured,
+    see `build_synthetic`) corrects the budget on a second pass.
+    """
+    prefix = list(item.in_context_prefix)
+    system = [message for message in prefix[:1] if message.get("role") == "system"]
+    rest = prefix[len(system):]
+    assert rest and rest[0].get("role") == "user", "the first message after the system prompt is the task"
+    task, tail = rest[0], rest[1:]
+    assert tail, "an item with messages after the task"
+
+    def tokens(message: dict) -> int:
+        text = message_text(message)
+        for call in message.get("tool_calls") or []:
+            text += json.dumps(call.get("function", call).get("arguments", {}))
+        return len(tokenizer.encode(text, add_special_tokens=False)) + 8
+
+    budget = (target_tokens - tokens(task) - len(tokenizer.encode(item.completion_text, add_special_tokens=False)) - 64) * scale
+    body: list[dict] = []
+    total = 0
+    cursor = 0
+    while total < budget:
+        message = tail[cursor % len(tail)]
+        body.append(message)
+        total += tokens(message)
+        cursor += 1
+    # Segments of equal token share, cut at message boundaries.
+    segments: list[list[dict]] = []
+    share = total / depth
+    current: list[dict] = []
+    current_tokens = 0
+    for message in body:
+        current.append(message)
+        current_tokens += tokens(message)
+        if current_tokens >= share and len(segments) < depth - 1:
+            segments.append(current)
+            current, current_tokens = [], 0
+    if current:
+        segments.append(current)
+    ratio = float(item.info.get("ratio") or 1 / 16)
+    nested = None
+    for segment in segments:
+        inner = [{"role": "user", "content": [nested]}] if nested is not None else [task]
+        nested = ac_part(inner + segment, ac_name, ratio)
+    ac_user = {"role": "user", "content": [nested, {"type": "text", "text": "\n\n" + COMPACTION_INSTRUCTIONS}]}
+    return ActivationContextTrainingItem(
+        item_id=f"synthetic:{target_tokens}:{depth}:{item.item_id}", kind="compaction", in_context_prefix=system + [task] + body,
+        ac_prefix=system + [task, ac_user], completion_text=item.completion_text, completion_complete=item.completion_complete,
+        teacher_partial_text="", tools=item.tools, dataset_id=item.dataset_id, doc_ids=list(item.doc_ids),
+        info={"depth": len(segments), "ratio": ratio, "synthetic_target": target_tokens})
+
+
+def build_synthetic(trainer, ac_model, item: ActivationContextTrainingItem, target_tokens: int, depth: int, ac_name: str):
+    """The example of a synthetic item within 5% of the target: one build to measure the template overhead, one corrected."""
+    example = trainer.build_example(ac_model, synthetic_item(item, ac_model.target.tokenizer, target_tokens, depth, ac_name))
+    measured = len(example.teacher_ids)
+    if abs(measured - target_tokens) > 0.05 * target_tokens:
+        example = trainer.build_example(ac_model, synthetic_item(item, ac_model.target.tokenizer, target_tokens, depth, ac_name, scale=target_tokens / measured))
+    return example
+
+
+# ------------------------------------------------------------------------------------------------ main
+def main() -> None:
+    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
+    parser.add_argument("--items-cache", default="AC_ITEMS/compaction_0.jsonl", help="under the synced folder")
+    parser.add_argument("--side", default="Qwen/Qwen3.5-0.8B")
+    parser.add_argument("--target", default="Qwen/Qwen3.5-4B")
+    parser.add_argument("--steps", type=int, default=3, help="examples per bucket")
+    parser.add_argument("--bucket-edges", default=DEFAULT_BUCKET_EDGES, help="teacher-length bucket upper edges")
+    parser.add_argument("--synthetic", default="", help="synthetic compaction items as teacher_tokens:depth, comma separated (e.g. 72000:2,104000:2)")
+    parser.add_argument("--synthetic-steps", type=int, default=2, help="examples per synthetic length (different source items)")
+    parser.add_argument("--max-example-tokens", type=int, default=None, help="cap on the teacher sequence (default: the trainer's; synthetic items are exempt)")
+    parser.add_argument("--logits-chunk", type=int, default=2048)
+    parser.add_argument("--no-checkpointing", action="store_true")
+    parser.add_argument("--checkpointing-min-tokens", type=int, default=None)
+    parser.add_argument("--teacher-cache-top-k", type=int, default=0, help="> 0: also compute the cached-teacher KL and report the gap to the exact KL")
+    parser.add_argument("--dump-fla-configs", action="store_true", help="write the fla kernel configs tuned during this pass")
+    parser.add_argument("--tag", default="profile")
+    args = parser.parse_args()
+
+    with open(resolve_path(args.items_cache, create=False)) as handle:
+        items = [ActivationContextTrainingItem(**json.loads(line)) for line in handle if line.strip()]
+    ac_name = items_ac_name(items)
+    print(f"{len(items)} items from {args.items_cache} (AC model {ac_name!r})")
+    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={SIDE_NAME: ModelConfig(SIDE_NAME, args.side), TARGET_NAME: ModelConfig(TARGET_NAME, args.target)}))
+    harness.module_manager.register_lora(TARGET_LORA, TARGET_NAME, rank=64)
+    ac_model = harness.module_manager.register_ac_model(ActivationContextModelConfig(ac_name, SIDE_NAME, SIDE_LORA, TARGET_NAME, TARGET_LORA))
+    config = ActivationContextTrainingConfig(logits_chunk_tokens=args.logits_chunk, gradient_checkpointing=not args.no_checkpointing,
+                                             gradient_checkpointing_min_tokens=args.checkpointing_min_tokens)
+    if args.max_example_tokens:
+        config.max_example_tokens = args.max_example_tokens
+    trainer = ActivationContextTrainer(harness, config)
+
+    # Placement as in train().
+    harness.module_manager.ensure_lora(TARGET_LORA)
+    device = ac_model.prepare()
+    fla_dir = configure_fla_cache(device)
+    print(f"fla configs: {fla_dir if fla_dir else 'none for this GPU (kernels autotune per length bucket)'}")
+    autotune_entries_before = sum(len(tuner.cache) for tuner in live_autotuners())
+    ac_model.set_mode(TRAINING)
+    ac_model.profile_phases = True
+    target, side = ac_model.target.model, ac_model.side.model
+    for base in (target, side):
+        base.train()
+        disable_dropout(base)
+    min_tokens = checkpointing_min_tokens(config, device)
+    set_checkpointing(target, config, min_tokens, min_tokens)
+    set_checkpointing(side, config, min_tokens, min_tokens)
+    embedding = target.get_input_embeddings()
+    head = target.get_output_embeddings()
+    base_dtype = next(target.parameters()).dtype
+    parameters = ac_model.trainable_parameters() + harness.module_manager.lora_parameters(TARGET_LORA)
+    target_flops, side_flops = decoder_flops(target), decoder_flops(side)
+    d_target, vocab = int(text_config(target).hidden_size), int(head.weight.shape[0])
+    head_flops_per_position = 2.0 * d_target * vocab
+    mixer_params = module_params(ac_model.modules.mixer) + module_params(ac_model.modules.pooling)
+    head_module_params = module_params(ac_model.modules.target_head) + module_params(ac_model.modules.recursive_adapter)
+    ceiling = gemm_ceiling_tflops(device)
+    print(f"GEMM ceiling {ceiling:.0f} TFLOPS (bf16 8192³); target {target_flops.params / 1e9:.2f}B decoder params, side {side_flops.params / 1e9:.2f}B, "
+          f"vocab {vocab}, d {d_target}", flush=True)
+
+    # Examples: the cache bucketed by teacher length (built lazily: tokenizing thousands of items takes minutes), then synthetic lengths.
+    edges = [int(value) for value in args.bucket_edges.split(",")]
+    buckets = [(0 if index == 0 else edges[index - 1], edge) for index, edge in enumerate(edges)]
+    examples_by_bucket: dict[tuple[int, int], list] = {bucket: [] for bucket in buckets}
+    for item in items:
+        if all(len(examples) >= args.steps for examples in examples_by_bucket.values()):
+            break
+        example = trainer.build_example(ac_model, item)
+        if len(example.teacher_ids) > config.max_example_tokens:
+            continue
+        for bucket in buckets:
+            if bucket[0] <= len(example.teacher_ids) < bucket[1] and len(examples_by_bucket[bucket]) < args.steps:
+                examples_by_bucket[bucket].append(example)
+    labels = {bucket: f"{bucket[0] // 1024}k-{bucket[1] // 1024}k" for bucket in buckets}
+    if args.synthetic:
+        sources = [item for item in items if item.kind == "compaction" and not item.teacher_partial_text][:args.synthetic_steps]
+        for spec in args.synthetic.split(","):
+            target_tokens, depth = (int(value) for value in spec.split(":"))
+            label = f"synthetic {target_tokens // 1000}k depth {depth}"
+            examples_by_bucket[(target_tokens, depth)] = [build_synthetic(trainer, ac_model, item, target_tokens, depth, ac_name) for item in sources]
+            labels[(target_tokens, depth)] = label
+
+    rows_out = []
+    for bucket, examples in examples_by_bucket.items():
+        label = labels[bucket]
+        if not examples:
+            print(f"bucket {label}: no items")
+            continue
+        for example in examples:
+            timings, peaks, flops = {}, {}, {}
+            num = example.num_completion
+            teacher_tokens, student_tokens = len(example.teacher_ids), len(example.student_ids)
+            for parameter in parameters:
+                parameter.grad = None
+            side_tokens_before, rows_before = ac_model.stats.side_tokens, ac_model.stats.view_rows
+            phases_before = dict(ac_model.stats.phase_seconds)
+            try:
+                # Teacher forward.
+                teacher_ids = torch.tensor(example.teacher_ids, device=device)
+                reset_peak(device); sync(device); started = time.time()
+                with torch.no_grad():
+                    teacher_states = ac_model.target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=TARGET_LORA)[0][-num - 1:-1]
+                sync(device); timings["teacher_forward"] = time.time() - started; peaks["teacher_forward"] = peak(device)
+                flops["teacher_forward"] = target_flops(teacher_tokens)
+                # Teacher top-k extraction (the cache write) when asked.
+                cached = None
+                if args.teacher_cache_top_k:
+                    reset_peak(device); sync(device); started = time.time()
+                    cached = trainer._teacher_top_k(teacher_states, head, args.teacher_cache_top_k)
+                    sync(device); timings["teacher_topk"] = time.time() - started; peaks["teacher_topk"] = peak(device)
+                    flops["teacher_topk"] = head_flops_per_position * num
+                # AC encode (grad).
+                reset_peak(device); sync(device); started = time.time()
+                rows = ac_model.encode_batch(example.part_requests)
+                sync(device); timings["ac_encode"] = time.time() - started; peaks["ac_encode"] = peak(device)
+                side_tokens = ac_model.stats.side_tokens - side_tokens_before
+                view_rows = ac_model.stats.view_rows - rows_before
+                side_sequence = side_tokens // ac_model.modules.pooling.stride + view_rows           # windows + marker rows through the side decoder
+                flops["ac_encode"] = 2.0 * mixer_params * side_tokens + side_flops(side_sequence) + 2.0 * head_module_params * view_rows
+                # Student forward.
+                reset_peak(device); sync(device); started = time.time()
+                student_ids = torch.tensor(example.student_ids, device=device)
+                with torch.no_grad():
+                    student_embeds = embedding(student_ids)
+                checkpointed = set_checkpointing(target, config, student_tokens, min_tokens)
+                pieces, cursor = [], 0
+                for (start, end), part_rows in zip(example.spans, rows):
+                    pieces.extend([student_embeds[cursor:start], part_rows.to(base_dtype)])
+                    cursor = end
+                pieces.append(student_embeds[cursor:])
+                student_embeds = torch.cat(pieces, dim=0)
+                student_states = ac_model.target.decoder_forward(student_embeds[None], None, lora_name=TARGET_LORA)[0][-num - 1:-1]
+                sync(device); timings["student_forward"] = time.time() - started; peaks["student_forward"] = peak(device)
+                flops["student_forward"] = target_flops(student_tokens)
+                # Head chunks (KL), exact.
+                reset_peak(device); sync(device); started = time.time()
+                kl, agreement = trainer._chunked_kl(teacher_states, student_states, head)
+                sync(device); timings["head_chunks"] = time.time() - started; peaks["head_chunks"] = peak(device)
+                flops["head_chunks"] = 2.0 * head_flops_per_position * num
+                kl_cached_gap = None
+                if cached is not None:
+                    with torch.no_grad():
+                        kl_cached, _ = trainer._chunked_kl(None, student_states.detach(), head, cached=cached)
+                    kl_cached_gap = float(kl) - float(kl_cached)
+                # Backward (student + side + AC modules).
+                reset_peak(device); sync(device); started = time.time()
+                kl.backward()
+                sync(device); timings["backward"] = time.time() - started; peaks["backward"] = peak(device)
+                side_checkpointed = config.gradient_checkpointing and ac_model.config.side_gradient_checkpointing
+                flops["backward"] = (target_flops(student_tokens) * (2 if checkpointed else 1)             # recompute + activation gradients
+                                     + side_flops(side_sequence) * (2 if side_checkpointed else 1)
+                                     + 4.0 * mixer_params * side_tokens + 4.0 * head_module_params * view_rows   # trainable: activation + weight gradients
+                                     + 2.0 * head_flops_per_position * num)                              # student head: recompute + gradients
+            except torch.OutOfMemoryError as error:
+                print(json.dumps({"bucket": label, "item": example.item.item_id, "teacher_tokens": teacher_tokens, "student_tokens": student_tokens,
+                                  "oom_in": next((name for name in ("backward", "head_chunks", "student_forward", "ac_encode", "teacher_topk", "teacher_forward") if name not in timings), "?"),
+                                  "timings_s": {key: round(value, 3) for key, value in timings.items()}, "peaks_gb": {key: round(value, 2) for key, value in peaks.items()},
+                                  "error": str(error)[:200]}), flush=True)
+                rows_out.append({"bucket": label, "item": example.item.item_id, "teacher_tokens": teacher_tokens, "student_tokens": student_tokens, "oom": True,
+                                 "timings_s": {key: round(value, 3) for key, value in timings.items()}, "peaks_gb": {key: round(value, 2) for key, value in peaks.items()}})
+                for parameter in parameters:
+                    parameter.grad = None
+                if device.type == "cuda":
+                    torch.cuda.empty_cache()
+                continue
+            total = sum(timings.values())
+            total_flops = sum(flops.values())
+            encode_phases = {key: round(value - phases_before.get(key, 0.0), 3) for key, value in ac_model.stats.phase_seconds.items()}
+            row = {"bucket": label, "item": example.item.item_id, "kind": example.item.kind, "depth": example.item.info.get("depth"),
+                   "teacher_tokens": teacher_tokens, "student_tokens": student_tokens, "side_tokens": side_tokens, "side_sequence": side_sequence,
+                   "parts": len(example.part_requests), "rows": view_rows, "completion": num, "kl": round(float(kl), 4), "agreement": round(agreement, 4),
+                   "kl_cached_gap": None if kl_cached_gap is None else round(kl_cached_gap, 5),
+                   "timings_s": {key: round(value, 3) for key, value in timings.items()}, "encode_phases_s": encode_phases,
+                   "peaks_gb": {key: round(value, 2) for key, value in peaks.items()},
+                   "tflops": {key: round(flops[key] / max(timings[key], 1e-6) / 1e12, 1) for key in timings},
+                   "total_s": round(total, 3), "total_tflop": round(total_flops / 1e12, 1), "achieved_tflops": round(total_flops / total / 1e12, 1),
+                   "roofline_s": round(total_flops / max(ceiling, 1e-6) / 1e12, 3) if ceiling else None,
+                   "checkpointed": bool(checkpointed), "tokens_per_s": round((teacher_tokens + student_tokens) / total)}
+            rows_out.append(row)
+            print(json.dumps(row), flush=True)
+            del teacher_states, student_states, rows, kl, cached
+            for parameter in parameters:
+                parameter.grad = None
+            if device.type == "cuda":
+                torch.cuda.empty_cache()
+
+    # Aggregates.
+    autotune_entries_after = sum(len(tuner.cache) for tuner in live_autotuners())
+    measured = [row for row in rows_out if not row.get("oom")]
+    by_bucket = {}
+    for row in measured:
+        by_bucket.setdefault(row["bucket"], []).append(row)
+    table = []
+    for label, rows in by_bucket.items():
+        mean_total = sum(row["total_s"] for row in rows) / len(rows)
+        mean_teacher = sum(row["teacher_tokens"] for row in rows) / len(rows)
+        table.append({"bucket": label, "examples": len(rows), "mean_teacher_tokens": round(mean_teacher), "mean_s": round(mean_total, 2),
+                      "examples_per_hour": round(3600 / mean_total), "teacher_tokens_per_hour": round(3600 * mean_teacher / mean_total),
+                      "achieved_tflops": round(sum(row["total_tflop"] for row in rows) / sum(row["total_s"] for row in rows), 1),
+                      "roofline_fraction": round(sum(row["roofline_s"] or 0 for row in rows) / sum(row["total_s"] for row in rows), 3) if ceiling else None,
+                      "phase_share": {key: round(sum(row["timings_s"][key] for row in rows) / sum(row["total_s"] for row in rows), 3) for key in rows[0]["timings_s"]},
+                      "peak_gb": round(max(max(row["peaks_gb"].values()) for row in rows), 1),
+                      "kl_cached_gap": round(sum(row["kl_cached_gap"] for row in rows) / len(rows), 5) if all(row.get("kl_cached_gap") is not None for row in rows) else None})
+    print("\nbucket | n | teacher tok | s/example | examples/h | teacher tok/h | TFLOPS | of ceiling | peak GB | shares")
+    for entry in table:
+        shares = " ".join(f"{key[:7]} {value:.2f}" for key, value in entry["phase_share"].items())
+        print(f"{entry['bucket']} | {entry['examples']} | {entry['mean_teacher_tokens']} | {entry['mean_s']} | {entry['examples_per_hour']} | {entry['teacher_tokens_per_hour']} | "
+              f"{entry['achieved_tflops']} | {entry['roofline_fraction']} | {entry['peak_gb']} | {shares}")
+    print(f"fla kernel cache entries: {autotune_entries_before} before, {autotune_entries_after} after (entries also appear when a config is matched from the files; "
+          f"a real benchmark prints 'Triton autotuning for function ...' under TRITON_PRINT_AUTOTUNING=1, so grep the log for that)")
+    print("AC model stats:", ac_model.stats.summarize())
+    written = {}
+    if args.dump_fla_configs and device.type == "cuda":
+        folder = gpu_config_dir(device)
+        if folder is not None:
+            written = dump_fla_configs(folder)
+            copy = resolve_path("AC_BENCH/fla_configs") / folder.name
+            shutil.copytree(folder, copy, dirs_exist_ok=True)
+            print(f"fla configs written: {written} -> {folder} (copy under {copy})")
+    ac_model.profile_phases = False
+    ac_model.release()
+    ac_model.target.model_to_device(SOURCE_DEVICE)
+    ac_model.side.model_to_device(SOURCE_DEVICE)
+    out = resolve_path("AC_BENCH/profile") / f"{args.tag}.json"
+    out.write_text(json.dumps({"args": vars(args), "gemm_ceiling_tflops": round(ceiling, 1), "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
+                               "target_params": target_flops.params, "side_params": side_flops.params, "autotune_entries": [autotune_entries_before, autotune_entries_after],
+                               "fla_configs_written": written, "table": table, "rows": rows_out}, indent=1))
+    print("written", out)
+
+
+if __name__ == "__main__":
+    main()
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/dataset.py</span>
    <span class="card-oneliner">Origin as string, trajectory documents, chunk messages, the example rename, the reduced stats.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/dataset.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/dataset_caching.py</span>
    <span class="card-oneliner">Study cache: one JSONL per (caching id, dataset, model, kind, seed) under STUDY/.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/dataset_caching.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/dataset_index.py</span>
    <span class="card-oneliner">bm25-only index; trajectory chunking at message boundaries.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/dataset_index.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/dataset_manager.py</span>
    <span class="card-oneliner">register / build_bm25_indexes / synthesize_study_examples_qa / label_study_examples.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/dataset_manager.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/dataset_study.py</span>
    <span class="card-oneliner">Study generator: has_valuable_question, trajectory prompt, cached questions and labels, examples returned not stored.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/dataset_study.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/dataset_utils.py</span>
    <span class="card-oneliner">Message rendering for the index (render_message / render_messages / document_text).</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/dataset_utils.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/loaders/__init__.py</span>
    <span class="card-oneliner">Exports of the three trajectory loaders and the AIME helper.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/loaders/__init__.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/loaders/aime.py</span>
    <span class="card-oneliner">AIME 2024 / 2025 statements for decontamination.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/loaders/aime.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/loaders/bright.py</span>
    <span class="card-oneliner">Rename follow-up (DatasetQAExample, origin NATIVE, labeled_qa_examples).</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/loaders/bright.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/loaders/msmarco.py</span>
    <span class="card-oneliner">Rename follow-up.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/loaders/msmarco.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/loaders/nemotron_math.py</span>
    <span class="card-oneliner">Nemotron-SFT-Math-v4 loader (cot / tir, shard selection, n-gram decontamination, min_tool_calls).</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/loaders/nemotron_math.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/loaders/nq.py</span>
    <span class="card-oneliner">Rename follow-up.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/loaders/nq.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/loaders/open_swe_traces.py</span>
    <span class="card-oneliner">Open-SWE-Traces loader (config / split, resolved_only, min_chars).</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/loaders/open_swe_traces.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/loaders/s1_deep_research.py</span>
    <span class="card-oneliner">S1-DeepResearch-15k loader (Hermes blocks, <tools> in the system prompt).</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/loaders/s1_deep_research.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/loaders/scifact.py</span>
    <span class="card-oneliner">Rename follow-up.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/loaders/scifact.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/loaders/sciq.py</span>
    <span class="card-oneliner">Rename follow-up.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/loaders/sciq.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/loaders/trajectory_utils.py</span>
    <span class="card-oneliner">Shared intermediate, reformat_trajectory into our dialect, parity_split, ngram_windows.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/dataset/loaders/trajectory_utils.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/harness/loaded_model.py</span>
    <span class="card-oneliner">The engine leaves ac_engine_memory_reservation of the GPU when an AC model is registered before it builds.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/harness/loaded_model.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/harness/module_manager.py</span>
    <span class="card-oneliner">register_ac_model / get_ac_model / ac_models.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/harness/module_manager.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/harness/runtime_config.py</span>
    <span class="card-oneliner">ac_engine_memory_reservation, ac_row_cache_bytes; the doc_embedding_* knobs removed.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/harness/runtime_config.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/tests/test_basic_agent_ac_training.py</span>
    <span class="card-oneliner">The three slice-3a key tests (compaction, trajectory QA, RAG QA); adapters freed before the bases.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/tests/test_basic_agent_ac_training.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/tests/test_basic_dataset_loading.py</span>
    <span class="card-oneliner">bm25-only loading test, trajectory chunking, live trajectory loaders.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/tests/test_basic_dataset_loading.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/tests/test_basic_dataset_study.py</span>
    <span class="card-oneliner">The study test on the new API.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/activation/tests/test_basic_dataset_study.py`): nothing to apply. The staged file stays the reference.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">pyproject.toml</span>
    <span class="card-oneliner">faiss-cpu removed.</span>
    <span class="card-badge">Applied</span>
  </summary>

Already in `/source` when this document was built (byte-identical to the staged copy `slice3/pyproject.toml`): nothing to apply. The staged file stays the reference.

</details>

