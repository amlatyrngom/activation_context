# Activation context, slice 3 — agent-facing source of truth

Projection: `SLICE3_PLAN.tressoir.md` (same folder). Interactions: `SLICE3_PLAN.interactions.json`.
Version 2.2, 2026-09-09. Status: decisions integrated; 3a milestones (M0–M5, the tests of M8) in Planning with moderate diffs; 3b (M6–M7, the 3b probes) TBD until 3a passes on a node. No product code changed.

## What this plan is built from

The user's working tree at `/source` (HEAD `1cfa058`) carries the slice-3 prototype as an uncommitted diff:

| path | state | what the notes say |
| --- | --- | --- |
| `activation/ac_model/ac_model.py` | new, stubs + docstring | the one canonical AC model: config, message format, cache, batching, tentative architecture, base-model choice, pooler ideas, causality question, "give me an html visualization" |
| `activation/ac_model/ac_model_training.py` | new, stubs | KL self-distillation items, teacher question, "AC fixed during agent training", isolation/continuation workflow |
| `activation/ac_model/ac_model_study.py` | new, stubs | three sample generators: compaction, trajectory QA, RAG QA |
| `activation/ac_model/ac_model_utils.py`, `__init__.py` | new, empty | low-level optimizations/utilities |
| `activation/ac_model/agent_ac_model.py` | deleted | the older retrieval-era note; superseded |
| `activation/agent/agent_config.py` | modified | `ac_model_name`; soft compaction threshold; `absolute_trajectory_cap`; `ac_compaction_ratio` |
| `activation/agent/agent_tools.py` | modified | `CompactionTool` docstring and registration |
| `activation/dataset/dataset.py` | modified | origin as string; `DataModality`; trajectory documents; `DatasetQAExample`; drop `LabeledMessagesExample`, `scope`, `study_context`; simplify stats |
| `activation/dataset/dataset_index.py` | modified | archive retrieval + faiss; keep bm25; trajectory chunking |
| `activation/dataset/dataset_manager.py` | modified | remove `build_dense_indexes`, `select_training_data` |
| `activation/dataset/dataset_study.py` | modified | caching, junk filter, trajectory QA, context as parameters, no write-back, per-batch seeds, labels take explicit examples |
| `activation/dataset/dataset_caching.py` | new, "@AI: Adapt me." | the staged `IB/ARTIFACTS/BRIGHT_ECON_ABLATION/activation/dataset/dataset_caching.py` |
| `activation/harness/module_manager.py` | modified | remove `register_retrieval_ac` and related |
| `activation/tests/test_basic_agent_ac_training.py` | new, docstring | compaction, trajectory QA, RAG QA tests with HTML reports; an optimization probe |
| `VLDB_NOTES.md` | untracked | LAkeQA, OTT-QA, Hybrid-QA: later, not this slice |

Prior decisions respected: token-segment training, sleep-mode GPU sharing, LoRA save/exchange/branch, group-mean selection with subagents inheriting credit, one engine per GPU at `gpu_memory_utilization` 0.9, pair-programming diffs with minimal churn, tests opt-in (this slice's one main-tree test file is named by the user), probes IB-only, `enable_prompt_embeds` in the recommended engine kwargs when the AC path lands.

## Facts verified

- vLLM 0.28 mixed prompts: `EmbedsPrompt(prompt_embeds, prompt_token_ids, prompt_is_token_ids)`. The tensor is full length (`_build_mixed_prompt_embeds` in `vllm/renderers/hf.py`, zeros at token positions; `input_processor.py` forwards it as is). In the worker (`gpu_model_runner.py` ~3650–3672) the engine embeds the token-id positions itself on the GPU (`embed_input_ids` on `where(is_token_ids, ids, 0)`) and keeps the shipped rows only where the mask is False. So the rows ride along as CPU tensors and the GPU lookup for ordinary tokens is not bypassed and costs nothing; the cost is the IPC of the full-length tensor (~5 KB/token on the 4B). Shipping rows for non-AC tokens would save nothing. Prefix-cache block hashes add a SHA-256 of the block's rows only for requests that carry `prompt_embeds`, so AC requests share blocks among themselves (zero blocks hash equal) but never with token-only requests.
- Qwen3.5 dense: 3 gated-delta layers per full-attention layer (`full_attention_interval` 4, conv kernel 4). 0.8B: d_model 1024, 24 layers, vocab 248,320; 4B: 2560, 32 layers. Cached locally: 0.8B, 4B. Causality removal is not a flag.
- Datasets named by the user (all public, non-gated):
  - `nvidia/Open-SWE-Traces`, config `v1.2`, split `minisweagent` (Qwen3.8-27B traces, 107k rows; v1.1 adds `openhands`, `sweagent`). Row: `instance_id, repo, license, language, trajectory_id, messages[{role, content, reasoning_content, think, tool_calls[{id, type, function{name, arguments(JSON string)}}]}], tools[JSON strings], resolved (1/0/-1), metadata{category, reference_patch, model_patch}, hf_dataset_name`. One tool `bash(command)`; tool messages are JSON `{"returncode", "output"}`; 23–401 messages per trajectory; sample row 36k chars. CC-BY-4.0.
  - `nvidia/Nemotron-SFT-Math-v4` (545,431 rows: `cot` 285,516 and `tir` 259,915; DeepSeek-V4-Pro generator). Row: `uuid, problem, expected_answer, messages[{role, content, reasoning_content, tool_calls, name, tool_call_id}], tools, source (AoPS / StackExchange), license, subset, ...`. `cot`: one user + one assistant whose `reasoning_content` holds the long solution (sample 100k chars ≈ 25k tokens) and `content` the boxed answer. `tir`: tool `stateful_python_code_exec(code)`, assistant turns with empty `content`, reasoning in `reasoning_content`, tool results plain text. Decontamination: the v4, v3 and source `Nemotron-Math-v2` cards only say "decontaminated to avoid overlap with public benchmarks", naming no benchmark. The AIME 2024/2025 check must be done locally.
  - `ScienceOne-AI/S1-DeepResearch-15k` (15,000 rows, `data.jsonl`, Apache-2.0). Row: `meta{id, question, answer, language, task_type}, messages[{role, content}]` with a proper `system / user / assistant / tool` sequence; tool definitions (`search`, `visit`) inline in the system prompt in Qwen Hermes style; assistant content has `<think>` blocks and `<tool_call>` JSON blocks; tool messages hold the search results; the final answer is in `<answer>` tags. English and Chinese.
  - `artillerywu/DeepResearch-9K`: rows `question, difficulty, "search trajectory"[{role, content}], "final answer"`; the trajectory alternates the user question (repeated verbatim) and assistant `<think>` + `<tool_call>` text; the tool observations are absent (the assistant refers to results that are not in the row). Unusable for an AC channel that must carry observations; recommended: S1 instead.
- Local trajectory sources in harness style: `IB/TMP/SYNC/ROLLOUTS/*/rollouts.jsonl` (~2.7 GB DAPO rollouts). Not selected by the user as a source; usable for CPU fixtures.
- `HtmlReporter` API (`common/reporting.py`): `initialize_line_plot / initialize_table / set_text / set_trajectories / add_data_point / set_status / render / finish`.

## Decisions (interactions file, 2026-09-08)

| # | question | accepted | consequence |
| --- | --- | --- | --- |
| 1 | teacher and trainable set | **snapshot of the current policy**, with the user's note: "need not necessarily be branched (generally not branch, since some later rollout should probably use it)" | teacher = the target base with the **same** target adapter, no grad, computed just in time per micro-batch; no `branch_lora`, no second adapter name; the adapter that AC training updates is the one later rollouts use after `exchange_lora`. AC training updates side LoRA + poolers + markers + heads + target LoRA. Within-round co-drift is bounded by one pass and the low LoRA rate; an in-memory frozen copy of the adapter is a later flag if the reporter shows the teacher moving. Drift term: flag, off. |
| 2 | pooler | **attention mixer**: two bidirectional sliding-window attention layers → windowed attention pooling (16/4) → markers | M3 |
| 3 | slice shape | **two halves** | 3a = M0–M5 + the three tests of M8 + the validity runs of M9; 3b = M6–M7 + the 3b probes |
| 4 | payload | **accept and measure**; user's question answered above (rows do not bypass the engine's lookup; shipping non-AC rows saves nothing) | the payload probe opens 3b |
| 5 | trajectory sources | **public agentic sets**, named: Open-SWE-Traces v1.2 (coding), Nemotron-SFT-Math-v4 (math; confirm no AIME 2024/2025 leak, else fall back to an earlier version), DeepResearch-9K or S1-DeepResearch-15k (general search; "not sure which is best"); "best if reformatted so the compression bias is in our favor" | M1: three loaders on one reformatter into our dialect; S1 chosen over DeepResearch-9K (the latter has no tool observations); a local n-gram decontamination against AIME 2024/2025 in the math loader with the fallback documented |
| 6 | archive | **IB and drop faiss-cpu** | M0 |
| 7 | models | **side 0.8B, target 4B on nodes; 0.8B/0.8B on CPU** | M3, M8 |
| 8 | assumptions | **adjust one**: the completion row, with two questions | see below; the other twelve rows are accepted |

### The three questions in the feedback, answered

- **Payload (decision 4):** the rows are CPU tensors that travel to the engine core with the request; the engine still performs its own GPU embedding lookup for every token-id position and overwrites nothing there. The AC rows do not replace work vLLM does anyway; they add the transfer. Sending full embeddings for non-AC tokens would only add bytes. What the probe measures is the transfer and the per-block hashing.
- **Truncation and the completion (decision 8):** no fake turn. When the cut lands mid-assistant-turn, the completion is the remainder of that same turn (as one assistant message), and both prefixes end at the cut with the generation prompt open; the AC prefix carries the compaction instruction before it. When the cut lands at a turn boundary, the completion is the next assistant turn. Either way the completion is capped at `completion_max_tokens` (512) and the teacher-forced tokens are identical in both branches.
- **The 512-token teacher-sampled fallback:** produced at item-generation time by one batched engine call on the in-context prefix (`engine_chat_many`, greedy, 512 max tokens), stored on the item as its completion. The trainer never samples; every trainer pass is an encode-only, teacher-forced forward. In this slice the fallback is used only for items without a natural continuation (RAG QA and trajectory QA use the study gold answer, compaction uses the trajectory).

### Assumptions accepted (recorded)

origin as string with constants; the rename everywhere; `generate_examples_labels(examples, caching_id=None)`; `base_seed + batch_index` engine seeds; `CompactionTool` without constructor kwargs; explicit tool + soft threshold + absolute cap (`finish_reason="trajectory_cap"`); two encode modes; cache key = (part JSON, ratio, recursive flag, AC version) → rows after the head; AC training updates the target LoRA (decision 1); no exchange for the AC model; AC parts inside any segment; the memory-reservation knob; run-level restart after compaction; the user's config field names kept.

## Design

### Reformatting into our dialect (M1)

`loaders/trajectory_utils.py::reformat_trajectory(messages, tools, *, tool_map, reasoning="keep") -> tuple[list[dict], dict]` returns messages in our shape and `trajectory_kwargs`:

- system: dropped from the messages; a short domain system prompt goes in `trajectory_kwargs["system_prompt"]` (the study and AC generators template with our tool list, so the source system prompt never reaches the model).
- user: the task text as is.
- assistant: `content` = reasoning text (when `reasoning="keep"`: `reasoning_content`, or the `<think>` body of the content, plain, no tags) followed by the visible content; `tool_calls` in our structured form `[{"id", "name", "arguments": dict}]` from either structured `tool_calls` or Hermes `<tool_call>` JSON blocks parsed with `agent_utils.parse_tool_calls`.
- tool: one message per call in order, `{"role": "tool", "content": text}`; Open-SWE's JSON `{"returncode", "output"}` becomes the output text plus an `exit code N` line when non-zero.
- tool names: `tool_map` renames `bash(command)` → `shell(script)` and `stateful_python_code_exec(code)` → `python(code)`; unknown tools keep their names and definitions (`search`, `visit`) and their definitions ride in `trajectory_kwargs["tools"]` so the dialect can render them.
- final answers: `\boxed{}` and `<answer>` stay in the text; `trajectory_kwargs["answer"]` records `expected_answer` / `meta.answer` / `resolved`.

Loaders (`loaders/open_swe_traces.py`, `nemotron_math.py`, `s1_deep_research.py`), all streaming with `take_to_budget`, `min_chars` filter, seed-parity split, `DatasetDocument(modality=TRAJECTORY, origin=EXTERNAL)`:

- `OpenSweTracesDataset.load(harness, max_examples, config="v1.2", split="minisweagent", resolved_only=False, min_chars=0)`.
- `NemotronMathDataset.load(harness, max_examples, subset="tir" | "cot" | None, excluded_problems: list[str] | None, min_chars=0)`; `excluded_problems` are benchmark statements; a problem sharing any 13-word window with one of them is dropped and counted (`stats.num_decontaminated`). The test passes the AIME 2024/2025 statements (`loaders/aime.py`, HF ids to be confirmed on the node: `math-ai/aime24`, `math-ai/aime25`; the DeepScaleR loader of slice 2 M4 is the alternative). Fallback if the leak count is large: an earlier version by `hf_dataset` parameter.
- `S1DeepResearchDataset.load(harness, max_examples, language="en", min_chars=0)`.

### Chunking trajectories (M1)

`dataset_utils.render_message(message) -> str` and `render_messages`. `DatasetIndex._trajectory_pieces(document)`: pack whole messages while the rendering fits `doc_chunk_size_chars`; a single over-long message splits by characters (its piece's `chunk_messages` is that message with the slice as content); `chunk_start` is the offset in the full rendering; `chunk_text` the rendering of `chunk_messages`. bm25 indexes `chunk_text`.

### Study (M2)

`StudyCache` as staged with kinds `qa`, `traj_qa`, `labels`. `generate_examples_qa(num_samples, base_seed=0, study_context=None, caching_id=None, modality=None)`; `generate_examples_labels(examples, caching_id=None)`; schema with `has_valuable_question`; trajectory prompt variant.

### AC model (M3)

(Implemented; see the completion reports below for the drifts.)

`ActivationContextModelConfig{ac_model_name, base_side_model_name, base_side_model_lora_name, target_model_name, target_model_lora_name, default_compression_ratio=1/16, input_pooling_stride=4, input_pooling_window=16, mixer_layers=2, mixer_window=256, min_view_rows=4, max_view_rows=4096, checkpoint_path=None}`.

Modules (all on the side model's width d_s; d_t the target's): `mixer` = `mixer_layers × BandedAttentionLayer` (pre-norm MHA with a boolean band mask |i−j| ≤ window, FFN ×4); `pooling` = `WindowedPooling(window, stride)` (the retrieval `WindowedBytePooling` on rows); `markers` = `Embedding(2, d_s)` (top-level / recursive) + `Embedding(max_view_rows, d_s)` index rows; `recursive_adapter` = FFN d_s→4d_s→d_s, RMS-normalized to the side embedding RMS × learned scale; `target_head` = FFN d_s→4d_s→d_t, RMS-normalized × learned scale. The side LoRA (rank 128, registered by `register_ac_model` if absent) trains through `decoder_forward(lora_name=base_side_model_lora_name)`.

Forward of one part: children first (cached, recursive) → adapter → `tokenize_with_parts(side tokenizer, part messages)` (template to text with one sentinel string per child, split, tokenize the pieces, insert V_child pad ids per child; returns ids and spans) → side input embeddings (`embedding_layer_to_device`, frozen copy) with child rows written over the pad positions → mixer → pooling → concat markers (V = clamp(ceil(tokens × ratio), min, max)) → side decoder with LoRA (causal, right-padded batches with an attention mask) → rows at marker positions → head. Rollout mode: no grad, `RowCache`, `EncodeQueue`. Training mode: grad, no cache, caller-batched.

`RowCache`: sha256(canonical JSON of the part, ratio, is_recursive, version) → CPU bf16 rows; LRU by bytes (4 GB). `EncodeQueue`: one worker thread, drains every 5 ms, length-sorted no-grad batches under a token budget, futures.

Checkpoints: `AC_MODELS/<name>/round_NNN` (+ `latest`) holding `ac_modules.pt` (mixer, pooling, markers, adapter, head, scales) and the side LoRA folder from `save_lora`; `version` = number of saves. `ModuleManager.register_ac_model(config)` / `get_ac_model(name)` / `ac_models`. `HarnessRuntimeConfig.ac_engine_memory_reservation = 0.1` lowers `gpu_memory_utilization` when an AC model is registered before the engine is built.

M3 implementation requirement (from the M9 profiling discussion): the mixer's banded attention is blocked local attention — blocks of `mixer_window` rows, each attending to itself and its two neighbours through SDPA on the reshaped tensor — never a materialized N×N band mask (1 GB of booleans at 32k).

### AC study (M4)

`ActivationContextTrainingItem{item_id, kind, in_context_prefix: list[dict], ac_prefix: list[dict], completion: list[dict], weight=1.0, dataset_id, doc_ids}`; `completion` is always filled at generation time (real continuation or the batched teacher sample). Generators on `ActivationContextStudyGenerator(harness, ac_model_name)` (the AC model name fixes ratios and the target tokenizer):

- `generate_compaction_samples(dataset_id, num_samples, seed=0, depth_range=(1, 2), threshold_range_tokens=(8192, 32768), ratios_range=(1/8, 1/16), completion_max_tokens=512)`: per sample one trajectory document; token offsets of every message under the target template; depth d and thresholds T_1..T_d; cuts at cumulative T; in-context prefix = system + user + messages up to the last cut (mid-turn allowed); AC prefix = system + user + one user message with the nested chain (`AC_d{ AC_{d-1}{…}, messages of segment d }`) followed by `COMPACTION_INSTRUCTIONS`; completion = the remainder of the cut turn or the next assistant turn, capped.
- `generate_trajectory_qa_samples(dataset_id, num_samples, depth_range=(1, 2), trajectory_tokens_range=(8192, 32768), distractors_range=(0, 2), ratios_range=(1/16, 1/32), seed=0)`: questions from M2 on trajectory chunks (`traj_qa`); the window around the source chunk grown to the token budget; distractors = bm25 top documents for the question with the source document excluded; in-context prefix = the window + question; AC prefix = AC(distractor 1), …, AC(window), question (each with the task text first); completion = the gold answer.
- `generate_rag_qa_samples(dataset_id, num_samples, depth_range=(1, 2), distractors_token_range=(8192, 32768), ratios_range=(1/16, 1/32), seed=0)`: questions from M2 on text chunks; in-context = gold chunk + question; AC prefix = one AC(gold + bm25 distractors shuffled to the token budget) + question; completion = the gold answer.

### AC trainer (M5)

`ActivationContextTrainingConfig{learning_rate_ac=5e-4, learning_rate_target_lora=2e-5, weight_decay=0.0, adam_betas=(0.9, 0.95), adam_eps=1e-8, max_grad_norm=1.0, updates_per_round=4, max_example_tokens=52_000, drift_term_weight=0.0, logits_chunk_tokens=2048, gradient_checkpointing=True, gradient_checkpointing_min_tokens=None, seed=0, checkpoint_every_round=True}` (rates to be reviewed as the retrieval defaults were).

`train(ac_model_name, training_data, reporting_data=(), reporter=None) -> ActivationContextTrainingStats`: engine to sleep; target base + target LoRA (`ensure_lora`) and side base + side LoRA on the GPU; AC modules on the GPU; teacher = target with the same adapter under `torch.no_grad()`; per item: `teacher_ids = template(in_context_prefix, generation prompt) + encode(completion text) + eot`, `student_ids/spans = tokenize_with_parts(target tokenizer, ac_prefix) + the same completion tokens`; teacher hidden → chunked fp32 log-softmax at completion positions; student `inputs_embeds` with rows (grad) → same; loss = mean KL(teacher ‖ student) over completion positions (+ `drift_term_weight` × KL(teacher ‖ student-in-context) when > 0); AdamW two groups; one example per micro-batch; `updates_per_round` steps; checkpoint AC + `save_lora(target)`; model to host RAM. Stats: kl per step, per kind, top-1 agreement, completion tokens, tokens/s, peak memory, checkpoint paths. `eval(ac_model_name, items)`: KL and agreement, no grad. Reporter: `HtmlReporter` with kl/agreement/throughput plots, a per-kind table, and a few teacher-vs-student completions as text.

### Validity runs (M9, closes 3a)

What the three tests give: held-out KL before/after two rounds of four updates on ~96 items per kind (mini-sample, minutes). That is a smoke signal (finite loss, KL moves, checkpoints written, timings), not evidence for the idea. M9 is the evidence: the profile probe, one bench driver, three node runs of two epochs, three reference points in every run.

Driver: `activation/bench/agent_probes/ac_training_bench.py` (in the tree like `dapo_bench.py`; pattern: harness → data → items (cached) → references → epochs → report + summary JSON). Arguments: `--kind {compaction,traj_qa,rag_qa,mixed}`, `--items 2000`, `--held 200`, `--epochs 2`, `--updates-per-epoch` (derived: items / examples-per-update, default 16 examples per update), `--side Qwen/Qwen3.5-0.8B`, `--target Qwen/Qwen3.5-4B`, `--ratio-range`, `--tag`, `--items-cache` (`AC_ITEMS/<kind>_<seed>.jsonl` under the synced folder: items are generated once with the engine awake — the study questions need the 27B — then the engine sleeps for the whole training; the mixed run reads the three caches). Phases on one node: load data → study questions (QA kinds) → build items → write cache → references (forward-only) → train N epochs with a held-out eval after every epoch (and every 50 updates in epoch 1) → save → report. Command shape:

```
uv run sky exec --sync <node> -- uv run python -m activation.bench.agent_probes.ac_training_bench --kind compaction --items 2000 --held 200 --epochs 2 --tag compaction_r1
```

Reference points (forward-only, computed on the held-out set before training, plotted as horizontal lines on every KL/agreement chart; all with the same adapter as the teacher, i.e. the target base at run start):

| reference | student input | reads as |
| --- | --- | --- |
| `no_context` | the AC prefix with the AC part removed (task + instruction, or the question alone) | the ceiling the channel must beat |
| `recent_text` | the AC part replaced by verbatim text of the same token budget as its rows: the last V tokens of the compacted span (compaction), the V tokens around the gold chunk (QA) | the same budget spent on text instead of rows; the bar for "compression is worth it" |
| `untrained_ac` | the AC prefix with the untrained AC model (step 0) | the starting point of the curve |
| `in_context` | the teacher itself | 0 by construction; agreement 1 |

Secondary metric, greedy decoding on the held-out set with HF forwards (no engine; LoRA + rows through `decoder_forward`, no KV cache, 64 new tokens, ≤ 200 items): QA kinds report answer accuracy (exact/contains match against `gold_answers`, the study labels' rule) for the student, `no_context`, `recent_text` and the teacher; compaction reports **next-action agreement**: the first tool call (name + arguments after whitespace normalization) of the student's greedy continuation equals the teacher's greedy first call. Both are reported per epoch; they cost a few minutes per eval.

Phase 0, before run A: the AC training profile probe, in the tree (`activation/bench/agent_probes/ac_training_profile.py`, round 2 moved it out of IB at the user's request): cached compaction items bucketed by teacher length (8k … 72k) plus synthetic depth-2 items at 72k and 104k, per phase time / peak memory / counted FLOPs / achieved TFLOPS against the card's measured bf16 GEMM ceiling, the exact-vs-cached-teacher KL gap, an fla autotune counter and `--dump-fla-configs`. Budget: about 10 minutes on a node per pass.

Levers the probe ranks, in the order I expect them to matter: (1) the teacher forward is roughly half the FLOPs of an item and identical across epochs — cache its top-k log-probs per completion position (k = 64 plus the remainder mass; exact KL needs the full row, so this is an approximation reported next to the exact number on the held-out set) so epoch 2 skips it; (2) the mixer's banded attention must not materialize an N×N mask (a 32k boolean mask is 1 GB): implement it as blocked local attention (blocks of `mixer_window`, each attending to itself and its neighbours, the same reshape as the pooling), which is a design requirement of M3, not an optimization; (3) the head chunk size and the checkpointing threshold for the student side, as in slice 2; (4) the encode batch: with one example per micro-batch the side model sees one part at a time, so a depth-2 item's child and parent encodes serialize — batch the children of one item; (5) fla configs for the 0.8B if the probe shows autotuning.

The three runs (side 0.8B, target 4B, one RTX 6000 Pro each; two epochs, enough for a signal before larger runs; time estimates from the slice-2 numbers — 4B forward-only ≈ 12k tok/s, forward+backward ≈ 4k tok/s — replaced by the profile's real numbers):

| run | items (train / held) | data | item parameters | est. per item | est. run (2 epochs) |
| --- | --- | --- | --- | --- | --- |
| A `compaction` | 2,000 / 200 | Open-SWE v1.2 (min 40k chars) and Nemotron-Math `tir` (min 40k chars), 50/50 | depth 1–2, thresholds 8k–24k per level (total ≤ 48k so depth 2 fits the 52k cap), ratio 1/8–1/16, completion ≤ 512 | ~3 s | ~3.5 h |
| B `traj_qa` | 2,000 / 200 | S1-DeepResearch-15k and Open-SWE, 50/50; `traj_qa` study questions on trajectory chunks | window 8k–32k, 0–2 distractor trajectories, ratio 1/16–1/32 | ~2.5 s | ~3 h |
| C `rag_qa` | 2,000 / 200 | BRIGHT biology + BRIGHT economics; `qa` study questions on text chunks | distractors to 8k–32k, ratio 1/16–1/32, depth 1–2 | ~2.5 s | ~3 h |

Run order: profile, then A, B, C independently (interpretable, each gives its own KL trend and references). The mixed run (D: the three item caches interleaved, 6,000/600) is deferred until A–C show a signal; the driver already accepts `--kind mixed`, and all runs share seed 0 and the item caches, so the held-out sets stay identical when D is run later. Study cost for B and C: ~4,400 questions + labels on the 27B, ~10–15 min per run, cached under `STUDY/`.

Read-out (what "the idea is valid" means at this scale): on every kind the held-out KL curve drops below `recent_text` within epoch 1 and is still descending at the end of epoch 2 (no early plateau near `no_context`); agreement rises correspondingly; QA answer accuracy of the student sits above `recent_text` and approaches the teacher; compaction next-action agreement is clearly above `no_context`. A curve that beats `no_context` but not `recent_text` says the channel carries something but compression is not yet paying: the first ablations are then ratio (1/8) and the 2B side model. Reports under `AC_BENCH/<tag>/`; summary JSON next to the page with the final and best held-out numbers per kind and the reference values.

Not in M9: the mixed run D (deferred), rollouts with AC parts (3b), the payload probe (3b), ratio/depth/side-model sweeps (after the three runs, same driver with `--ratio-range` / `--side`).

### The formatting note (interactions file, 2026-09-08 late)

"You might need dataset-specific reformattings; the formats look fairly different, but maybe there is a shared universal." Both, and the plan is now explicit about it: each loader owns a `_normalize_row` that handles its source's quirks and yields one shared intermediate (a list of `{role, content, reasoning, tool_calls: [{name, arguments}], tool_results: [text]}` plus a tool-definition list); `reformat_trajectory` works on that intermediate only. The quirks per source: Open-SWE — structured `tool_calls`, tool results as JSON `{"returncode","output"}`, tool definitions as JSON strings; Nemotron-Math — structured `tool_calls`, reasoning in `reasoning_content`, `tools` a list (`cot` rows have no tools); S1 — Hermes `<tool_call>` blocks inside the assistant text, `<think>` blocks, tool definitions in the system prompt's `<tools>` block, results in `tool` messages. The shared step then maps tool names/arguments, drops the source system prompt, and renders reasoning as text.

### Slice 3b (M6, M7) — TBD, shaped by the payload probe

As in v1: AC parts in segments; mixed-prompt submission; `enable_prompt_embeds`; compaction tool with the nudge and cap; subagent exchange; `items_for` always includes main-named children; the agent trainer overwrites placeholder rows with the fixed AC model; version mismatch is an error.

## Validation plan

CPU: chunking, rename and reformatting on fixtures (a few rows of each dataset saved under `IB/TMP/`); study cache round trips with a fake engine; AC forward on 0.8B/0.8B (shapes, V counts, cache hits, queue batching, save/load, version bump); one trainer step on fabricated items (finite loss, adapter updated, KL decreasing over a few steps). Node 3a: the three tests (side 0.8B, target 4B; data: Open-SWE v1.2, Nemotron-Math tir, S1, BRIGHT biology); the forward profile probe; then M9: the profile probe, and the three validity runs (A compaction, B traj_qa, C rag_qa; two epochs each) with the reference points and the read-out above; D mixed deferred. Node 3b: payload probe; rollout with subagent exchange and a forced compaction; one fixed-AC agent-training round.

## Completion reports (slice 3a implemented, 2026-09-09)

### M0

**What landed.** `dataset.py`: `origin` is a string (`NATIVE` / `EXTERNAL` / `SYNTHETIC` constants), `DatasetDocument.trajectory` + `trajectory_kwargs` + `modality`, `DatasetDocumentChunk.chunk_messages`, `LabeledRetrievalQAExample` → `DatasetQAExample` (`labeled_qa_examples`), `DatasetStats` reduced to load / chunk / bm25 / study counters (`num_decontaminated`, `study_num_junk_skipped`, `study_num_cached`, `study_num_labels_cached` added). `dataset_index.py` is bm25-only (dense paths, faiss and the embedding knobs gone; `HarnessRuntimeConfig.doc_embedding_*` removed). `dataset_utils.py` renders messages for the index (`render_message`: `role: text` plus `[call name(args)]` lines). `dataset_manager.py` keeps `register_dataset`, `build_bm25_indexes` and the two study wrappers. Five loaders follow the rename. `retrieval/`, `retrieval_training_bench.py`, `test_basic_retrieval_training.py` and the pre-slice-3 `dataset_index.py` / `module_manager.py` are archived under `IB/ARTIFACTS/RETRIEVAL_TRAINING/archive_slice3/` (README records the HEAD); `faiss-cpu` left `pyproject.toml` and `uv.lock` (`uv lock`: 215 packages resolved, faiss-cpu 1.15.0 removed).

**Drifts.** The bm25 loading test lost its dense variant as planned and gained `test_trajectory_chunking_bm25` and a live `test_trajectory_dataset_loading` (M1); `DatasetStats.summarize()` was rewritten rather than trimmed (the old one referenced the removed fields). `README.md` in the workspace was already modified before this slice and is not part of the handoff.

**Validation.** `uv run pytest activation/tests/test_basic_dataset_loading.py` (CPU, network): 4 passed (bm25 basics, trajectory chunking, five public datasets at 20 examples / 100 documents, the three trajectory loaders at 2 rows).

### M1

**What landed.** `loaders/trajectory_utils.py`: the shared intermediate (`{role, content, reasoning, tool_calls:[{name, arguments}]}` + tool definitions) and `reformat_trajectory` into our dialect (source system prompt dropped, `bash(command)` → `shell(script)`, `stateful_python_code_exec(code)` → `python(code)`, reasoning kept as text ahead of the reply, structured `tool_calls` with `call_NNNN` ids, one `tool` message per result, JSON `{"returncode","output"}` flattened with an `[exit code N]` line, definitions for every mapped tool the trajectory calls); `parity_split` (one document in ten is TEST by hash); `ngram_windows` for decontamination. Three loaders on it: `OpenSweTracesDataset.load(harness, max_examples, config="v1.2", split="minisweagent", resolved_only=False, min_chars=0)`, `NemotronMathDataset.load(harness, max_examples, subset="tir", excluded_problems=None, min_chars=0, min_tool_calls=0, hf_dataset=..., data_files=None)` with the 13-word-window check (`stats.num_decontaminated`), `S1DeepResearchDataset.load(harness, max_examples, language="en", min_chars=0)`; `loaders/aime.py` (`load_aime_problems`, `aime_problem_statements`; `math-ai/aime24` stores the answer as `solution` = `\boxed{...}`, `aime25` as `answer`; 30 + 30 rows confirmed). Trajectory chunking in `dataset_index._trajectory_pieces`: whole messages packed to the chunk size, no overlap, an over-long message split by its own text into one-message slices (the calls ride on the last slice), chunk text = the rendering.

**Drifts.** (1) Nemotron's single `train` split is ordered cot-then-tir and the boundary (row 285,516) lies inside shard 7 of 12 (row counts read from the parquet footers), so each subset streams only its shards (`cot` 0–7, `tir` 7–11); a `min_tool_calls` filter was added because some `tir` rows never call the tool. (2) S1's system prompt mentions a literal empty `<tools></tools>` before the real block; the parser takes the first non-empty block (five tools: `search`, `visit`, `PythonInterpreter`, `google_scholar`, `parse_file`; they keep their names). (3) Open-SWE trajectories end with a call whose result never arrives (the submit call); the reformatter keeps it. (4) HF ids for AIME confirmed from the datasets-server, not on a node.

**Validation.** Fixture rows saved under `IB/TMP/SLICE3_FIXTURES/` (three Open-SWE, four Nemotron cot, two tir, three S1, two AIME each) and checked by `IB/TMP/SLICE3_CHECKS/check_fixtures.py`: every row reformats with dict arguments, mapped tool names, one tool message per call (minus the trailing submit), no system role; AIME self-hit true, Nemotron sample no hit. Live: `test_trajectory_dataset_loading` streams two rows of each source (Open-SWE 17 s, Nemotron tir 19 s, S1 19 s cold) and checks modality, origin, tools, calls, chunk round trip.

### M2

**What landed.** `dataset_caching.py`: `StudyCacheKey(caching_id, dataset_id, model_id, kind, seed, label_model_id)` → one JSONL per key under `STUDY/`, `StudyCache.read / append / read_by_id` (a torn last line is dropped). `dataset_study.py` rewritten: `has_valuable_question` schema (junk chunks skipped and cached as junk rows), a trajectory prompt variant, `generate_examples_qa(num_samples, base_seed=0, study_context=None, caching_id=None, modality=None)` with cached rows served first (checked by chunk id against the seeded sampler), engine seed `base_seed + batch index`, examples returned (origin SYNTHETIC, positive chunk = the source chunk) and nothing written into the dataset; `generate_examples_labels(examples, caching_id=None)` labels exactly the given examples, cached by example id. `dataset_manager.synthesize_study_examples_qa(dataset_id, num_samples, base_seed=None, study_context=None, caching_id=None, modality=None)` and `label_study_examples(dataset_id, examples, caching_id=None)`. `test_basic_dataset_study.py` follows the API (synthetic examples labeled first, then the native ones; count assertions allow junk).

**Drifts.** The label cache is keyed by the label model id with seed 0 (labels do not depend on the sampling seed). Parse failures are not cached (a later pass retries). The `study_context` argument of the old dataset object is gone (it was never set).

**Validation.** `IB/TMP/SLICE3_CHECKS/study_cache_smoke.py` (fake engine on a fabricated dataset): 6 questions cost 6 engine calls, the second identical request costs none, a request for 9 reuses the 6 and generates 3, the trajectory variant samples trajectory chunks only, labels are cached and served by example id (3 label calls, 3 cached on the rerun). The node test (`--gpu`) is not rerun here.

### M3

**What landed.** `ac_model/ac_model_utils.py`: `tokenize_with_parts` (sentinel per part in the chat template, split, tokenize the pieces, pad-id runs, spans; a part without a user message gets an empty one because Qwen's template refuses a conversation without a user query), `BlockedLocalAttentionLayer` (blocks of `mixer_window` rows attending to themselves and both neighbours through SDPA on the reshaped tensor; padded queries attend everything in their window so no row is fully masked), `WindowedPooling` (the retrieval byte pooling on rows, window/stride from the config), `RowHead` (FFN ×4, RMS-normalized × learned scale), `RowCache` (CPU bf16, LRU by bytes), `EncodeQueue` (one worker, 5 ms drain, length-sorted batches, futures). `ac_model/ac_model.py`: `ActivationContextModelConfig` (the planned fields plus `mixer_heads=8`, `side_lora_rank=128`, `encode_batch_max_tokens=65_536`, `side_gradient_checkpointing=True`), `ActivationContextModules` (fp32: mixer, pooling, marker kind/index, recursive adapter, target head, a `scales_initialized` buffer), `ActivationContextModel` with `encode / encode_batch / encode_async / part_view_rows / num_view_rows / set_mode / prepare / release / save / load / trainable_parameters` and `ActivationContextModelStats`. `ModuleManager.register_ac_model / get_ac_model / ac_models`; `LoadedModel.engine_to_device` subtracts `ac_engine_memory_reservation` (0.1) from `gpu_memory_utilization` when an AC model is registered before the engine builds; `HarnessRuntimeConfig.ac_row_cache_bytes` (4 GB).

**Drifts.** (1) The side model's own input embedding table is used directly (the side base is resident whenever the AC model runs), so no `embedding_layer_to_device` copy for the side; the target's copy is used only to initialize the head scale when the target is not loaded. (2) The head scales initialize lazily (first `prepare`) from the destination embedding RMS and are saved in the checkpoint. (3) `part_view_rows` was added so the trainer can size placeholder runs before encoding (V depends on the side tokenization, not the target's). (4) Routing a part to a different AC model is asserted against, not implemented. (5) `load` re-injects the side adapter from the checkpoint folder through `free_lora` + `checkpoint_path`.

**Validation.** `IB/TMP/SLICE3_CHECKS/ac_smoke.py` on CPU (Qwen3.5-0.8B as side and target): a 1,095-token part → [64, 1024] rows in 5.8 s; cache hit on the second call; a nested part costs three network passes (child, parent, and the earlier top-level); batched vs single rows differ by ≤ 0.009 at RMS 0.019 (bf16 decoder noise; the mixer runs fp32 on CPU); four queued requests form one batch; in training mode gradients reach all 46 module tensors and 96 of 192 side-LoRA tensors (the B matrices; A's gradient is zero at B = 0); save → `AC_MODELS/ac_dev/round_000/{ac_modules.pt (219 MB), side_lora/}` + `latest`, load restores a mutated scale, version 1. CPU runs need `sys.modules["fla"] = None` before importing transformers (see the round document).

### M4

**What landed.** `ac_model/ac_model_study.py`: `ActivationContextStudyGenerator(harness, ac_model_name)` with `generate_compaction_samples(dataset_id, num_samples, seed=0, depth_range=(1, 2), threshold_range_tokens=(8192, 32768), ratios_range=(1/8, 1/16), completion_max_tokens=512, max_total_tokens=72_000, min_trajectory_tokens=1024)`, `generate_trajectory_qa_samples(dataset_id, num_samples, depth_range, trajectory_tokens_range, distractors_range, ratios_range, seed, caching_id)`, `generate_rag_qa_samples(dataset_id, num_samples, depth_range, distractors_token_range, ratios_range, seed, caching_id)`; `ac_part(messages, ac_name, ratio)`; the compaction instructions constant; items carry `info` (depth, thresholds, ratio, gold position, `gold_chunk_text`).

**Drifts.** No teacher-sampled fallback: every item's completion is real text (the plan's fallback was for cuts without a continuation; the generator moves or drops such cuts instead). Items carry `completion_text` / `completion_complete` / `teacher_partial_text` / `tools` / `info` instead of a `completion` message list. Round 2: the total bound of a compaction item moved from 48k to 72k tokens so a depth-2 item at the default 32k threshold is no longer cut short; the threshold range itself is unchanged (the bench generates at 8k–24k per level).

**Validation.** `IB/TMP/SLICE3_CHECKS/ac_train_smoke.py` on fabricated trajectories and passages (3 compaction items at depths 1–2, 2 traj_qa, 2 rag_qa, every item with parts, spans and a completion). Node: 2,200 compaction items generated from Open-SWE v1.2 and Nemotron tir (≥ 40k chars) in about 4 minutes, cached at `AC_ITEMS/compaction_0.jsonl` (300 MB); the test generators produced 48 items per kind with real 27B study questions for the QA kinds.

### M5

**What landed.** `ac_model/ac_model_training.py`: `ActivationContextTrainingItem`, `ActivationContextTrainingConfig` (the planned fields plus `examples_per_update`, `fp32_head_matmul`, `teacher_cache_top_k`), `ActivationContextTrainingStats` (per-step KL / agreement / drift / grad norm / seconds / tokens, per-kind lists, per-example records with `seconds_by_bucket()`, teacher-cache counters), `ActivationContextTrainer.build_example / train / eval`. Teacher ids = template(in-context prefix, generation prompt) + partial text + completion (+ eot when complete); student ids = `tokenize_with_parts(target tokenizer, ac_prefix, [V per part])` + the same completion; the rows come from `encode_batch` in training mode; loss = mean full-vocabulary KL(teacher ‖ student) over the completion positions, the head one chunk at a time under `torch.utils.checkpoint`; AdamW in two groups (5e-4 for the AC modules and the side LoRA, 2e-5 for the target adapter); `examples_per_update` or `updates_per_round`; the drift term when `drift_term_weight > 0`; checkpoints (`AC_MODELS/<name>/round_<n>` + the target LoRA) per round; models back to host RAM after the round. Round 2 additions: (1) `configure_fla_cache(device)` at placement, as the agent trainer does, so fla's Triton kernels load the checked-in configs instead of re-tuning per length bucket; (2) `max_example_tokens` 52k → 72k; (3) `teacher_cache_top_k` (default 0 = exact teacher every pass): after an item's first pass its teacher top-k log-probs and the log of the remainder mass are kept per completion position on the CPU (`_teacher_top_k`), and later passes skip the teacher forward and minimize the KL of the k+1-way coarsened distributions (`_chunked_kl(..., cached=...)`: a lower bound of the exact KL, the same loss in every epoch; `eval` is always exact); (4) per-example records (kind, teacher / student tokens, seconds, cached) for the throughput read-out. Review pass (during run A): (5) the optimizer persists across rounds (one AdamW per AC model, `persistent_optimizer`; moments parked on the host between rounds) instead of a fresh AdamW per `train()` call, which reset Adam's moments and bias correction at every chunk boundary; (6) `warmup_updates` (linear, on a cross-round update clock); (7) a mid-round `_evaluate` no longer re-enables dropout modules when it restores training mode, and it counts the items it drops for length.

**Drifts.** The reporting items are evaluated with `eval` after the round (not inside the step loop). Placeholder runs are sized by `part_view_rows` and asserted against the encoded rows. The cached teacher is stale by design (the adapter as of the item's last exact pass); with the target adapter at 2e-5 that is the approximation the user accepted, measured below.

**Validation.** CPU (`IB/TMP/SLICE3_CHECKS/ac_train_smoke.py`, `ac_perf_smoke.py`): 7 fabricated items, two rounds, held KL 3.20 → 2.19; on a toy head the coarsened KL is ≤ the exact one for k = 1, 5 and equal at k = vocabulary; two rounds with `teacher_cache_top_k=32` on 4 items: 0 then 4 cache hits, finite losses, exact `eval` afterwards. Node: the three tests (M8) and the profile (M9) below; the cached-teacher gap is 0.1–1.0% of the KL at k = 64 on real items.

### M8

**What landed.** `activation/tests/test_basic_agent_ac_training.py`: `test_ac_compaction` (Open-SWE v1.2 ≥ 40k chars + Nemotron tir ≥ 40k chars with ≥ 1 call, 24 + 24 items), `test_ac_trajectory_qa` (S1 + Open-SWE, 48 study questions through the 27B QA engine, cached under `ac_training_test`), `test_ac_rag_qa` (BRIGHT biology, 48 questions); side `qwen3.5-0.8b`, target `qwen3.5-4b`, AC `ac_dev`, target LoRA `ac_target` rank 64, side LoRA `ac_side` rank 128; each test: held KL of the untrained model and of the no-context reference, one round of `examples_per_update=8` on 40 items, asserts finite KL, checkpoints, and held KL below the no-context reference; teardown frees the adapters before the bases.

**Drifts.** The `recent_text` reference and the secondary metrics live in the M9 driver, not in the tests. The first node run failed in teardown only (freeing a base with adapters still injected); fixed by `free_lora` on both adapters first.

**Validation.** Node `ac-fp4-probe` (RTX PRO 6000, 96 GB), 2026-09-09: `3 passed in 823 s`. Held KL untrained / no-context / after one round: compaction 0.579 / 0.574 / 0.525 (agreement 0.765 → 0.779); trajectory QA 1.055 / 0.936 / 0.917 (0.674 → 0.675); RAG QA 0.897 / 0.869 / 0.849 (0.747 → 0.733). Pages under `IB/TMP/SYNC/AC_TRAINING_TEST/{compaction,traj_qa,rag_qa}/`, log `IB/TMP/AGENT_ROLLOUTS/ac_tests_run3.log`. The compaction round ran at 9,714 tokens/s with the fla configs against 4,681 in the first run without them (steps 84 / 53 / 26 / 22 / 36 s → 19 / 25 / 19 / 22 / 22 s).

### M9

**What landed.** `activation/bench/agent_probes/ac_training_bench.py` (`--kind {compaction,traj_qa,rag_qa,mixed} --items 2000 --held 200 --epochs 2 --examples-per-update 16 --eval-every 50 --side --target --qa-model --ratio-range --threshold-range 8192,24576 --max-total-tokens 72000 --max-example-tokens --teacher-cache-top-k 0 --side-lora-rank --target-lora-rank --lr-ac --lr-target --secondary-items 100 --secondary-tokens 64 --seed --tag --generate-only`): items generated once per kind and cached at `AC_ITEMS/<kind>_<seed>.jsonl`, references `no_context`, `recent_text`, `untrained_ac`, epochs with held evals, the greedy secondary metric, `summary.json` with a `throughput` block (examples and teacher tokens per hour, by length bucket, live vs cached teacher). Review pass: `split_items` holds out in exact proportion to the sources (the head slice used before was mixed only by the seeded shuffle) and drops training items that share their origin (trajectory, study question) with a held-out item; `--warmup-updates 10`. `activation/bench/agent_probes/ac_training_profile.py` (in the tree now): the card's bf16 GEMM ceiling, examples bucketed by teacher length up to 72k plus synthetic depth-2 compaction items at requested lengths (a real trajectory's messages cycled, measured and rescaled to within 5%), per phase time / peak memory / counted FLOPs / achieved TFLOPS, the exact-vs-cached-teacher KL gap, an fla kernel-cache counter and `--dump-fla-configs`.

**Drifts.** The secondary metric defaults to 100 held items. The `in_context` reference is not computed (it is the teacher). The probe is a tree file, not IB-only (the user asked for the performance work to be part of the 3a handoff). The plan's phase-0 lever list was reordered by measurement: loading the fla configs in the AC trainer and the teacher cache are the two that matter (the 0.8B needed no tuning of its own: fuzzy matching serves it from the 4B's entries); head chunks and the checkpointing threshold have nothing to give (the head is 0.01 s per example, the student sequence is below the card's recompute threshold); child batching would touch at most the 7–11% that the encode costs and was not done.

**Validation.** Node, 2026-09-09: the profile before and after the 0.8B's kernel configs, the cached-teacher gap, the caps, and the validity runs are reported in `SLICE3_ROUND.tressoir.md` under "Node results" and summarized in the M9 node results below.

**Node results.**

**Node results (2026-09-09, `ac-fp4-probe`, RTX PRO 6000 96 GB).** Tests: 3 passed in 823 s (held KL untrained / no-context / after one round: compaction 0.579 / 0.574 / 0.525, trajectory QA 1.055 / 0.936 / 0.917, RAG QA 0.897 / 0.869 / 0.849). Profile (r2, with the kernel configs): GEMM ceiling 410 TFLOPS; the teacher forward is the largest phase (46–69% of an example, 34–45% of the ceiling), backward 20–23%, AC encode 17–22%, head chunks 0.01 s; the fla configs loaded by the AC trainer doubled the test round (4,681 → 9,714 tokens/s; the 0.8B's kernels resolve from the 4B's entries by fuzzy matching, no tuning needed); the cached teacher's gap is 0.1–1.0% of the KL at k = 64 and 0.1–0.6% (mean 0.3%) at k = 128, so it is on (k = 128) for the runs and a cached epoch costs 36% of a live one; 72k depth-2 items peak at 68.6 GB and 104k at 73.5 GB.

| teacher tokens (bucket) | n | mean teacher tokens | s / example r1 → r2 | examples / h (r2) | teacher tokens / h (r2) | achieved TFLOPS (r2) | of ceiling | peak GB | cached-teacher gap (k = 128) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0k-8k | 3 | 4,901 | 0.98 → 0.97 | 3,694 | 18.1M | 73.1 | 18% | 26.6 | 0.1% |
| 8k-16k | 3 | 15,106 | 1.77 → 1.77 | 2,037 | 30.8M | 120.3 | 29% | 46.7 | 0.2% |
| 16k-32k | 3 | 23,140 | 2.50 → 2.50 | 1,441 | 33.4M | 124.5 | 30% | 46.8 | 0.4% |
| 32k-48k | 3 | 35,666 | 3.82 → 3.80 | 948 | 33.8M | 129.4 | 32% | 51.3 | 0.3% |
| 48k-64k | 2 | 50,038 | 5.60 → 5.56 | 647 | 32.4M | 133.7 | 33% | 60.7 | 0.3% |
| synthetic 72k depth 2 | 2 | 74,918 | 8.50 → 8.46 | 426 | 31.9M | 143.7 | 35% | 68.6 | 0.4% |
| synthetic 104k depth 2 | 2 | 107,814 | 12.67 → 12.65 | 285 | 30.7M | 156.5 | 38% | 73.5 | 0.4% |

Validity runs:

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

## Status log

- 2026-09-09 v5: review pass during run A (user: re-review for training practice, agent training included): AdamW state persisted across rounds in both trainers, linear warm-up, dropout kept off after a mid-round eval, the bench's held-out split stratified by source with no shared origin; run A on the pre-review code, B and C on the reviewed code; slice 3a1 (sync, reporting, eval cadence in the trainer) noted before 3a is checked in.
- 2026-09-09 v4: node round before the application (user: profiling and the larger runs first, as part of the 3a handoff): three tests passed; profile r1/r2 against the measured GEMM ceiling; fla configs in the AC trainer (2×) and the 0.8B's kernel configs checked in; cached top-k teacher (`teacher_cache_top_k`, gap ≤ 1%); caps 52k → 72k (104k measured); runs A/B/C for two epochs; handoff round 2 with 42 cards; M4/M5/M8/M9 completion reports updated.
- 2026-09-09 v3: slice 3a implemented (M0–M5, M8, M9 driver + IB-only profile probe) and CPU-validated; every 3a card in Review with its completion report; handoff `SLICE3_ROUND.tressoir.md`, staged `slice3/`, explainer `SLICE3_AC_MODEL_EXPLAINER.tressoir.html`; node runs pending (three tests, profile, runs A/B/C).
- 2026-09-09 v2.2: M9 reduced to three runs of two epochs (D deferred); phase 0 profile probe added with the ranked levers (teacher top-k cache, blocked local attention, head chunks/checkpointing, child batching, fla configs for the 0.8B); blocked local attention recorded as an M3 requirement.
- 2026-09-09 v2.1: M9 validity runs specified (bench driver, four runs, references `no_context` / `recent_text` / `untrained_ac`, secondary metrics, read-out) in answer to the check-in question; the formatting note answered (per-loader normalization to a shared intermediate, one common reformatter).
- 2026-09-08 v2: decisions 1–8 integrated from the interactions file; the three questions answered (payload does not bypass the engine's lookup; mid-turn cut → remainder of the turn as completion, no fake turn; teacher-sampled fallback via one batched engine call at generation time); dataset schemas verified (S1 over DeepResearch-9K, which lacks tool observations; Nemotron-Math cards name no benchmark → local n-gram check); 3a milestones moved to Planning with moderate diffs; 3b TBD until 3a passes on a node.
- 2026-09-08 v1: intent pass; reviewer pass integrated.
