# Deep activation intervention MVP — implementation and review source

Implement additive per-layer AC inputs for Qwen3.5 in HF training and a pinned vLLM fork, preserving the existing input-embedding channel and exact request replay. This is a proposed implementation plan, not an implementation. The companion [human projection](PLAN.tressoir.md) leads with the local interfaces. This source leads with engine correctness and names every intended code seam.

## Baseline and completed repository preparation

- Local product snapshot: `0a77891` (53 existing product/dependency paths committed without changing their contents). Parent was `a839760`. The user's existing code, including prototypes and the preexisting DeepScaleR deletion, was retained. This snapshot is not a new quality or test certification.
- Submodule registration: `13f19d9`; `.gitmodules` points `IB/REPOS/vllm` at `https://github.com/amlatyrngom/vllm.git`.
- GitHub fork exists and branch `ac-interventions-v0.28.0` was pushed using the existing authenticated account. Its starting commit is `2cf0a6915ce544dc493a0990f2ea38d81601128a`, the upstream `v0.28.0` tag. No upstream pull request or root-repository push was made.
- Reviewed prerequisite source: [co-design handoff](../AGENT_AC_PRETRAINING/codesign/README.md). Its `changes.patch` passes `git apply --check` against the committed local product. It has not been applied here. Current product still rejects AC-bearing AgentTrainer examples and lacks row publication/shared checkpoint binding; STATE describes newer work elsewhere.
- POC evidence cut: reconstruction ledger through 2026-09-16 13:38 UTC and the four-arm RAG report. Ongoing runs are not interrupted, restarted or source-synced by this task. Experimental product tags named by those artifacts are absent in this checkout. Never fabricate their availability or copy an unverified remote worktree over the local baseline.
- Scope of this turn is fork + local commits + reviewed plan. Product implementation follows approval of this pair. TASK.md is untouched.

## Accepted contract

The three user-visible AC config fields are exactly:

`ActivationContextModelConfig`

```python
intervention_frequency: int = 4
add_last_layer_intervention: bool = True
prefer_attention_interventions: bool = True
```

For L decoder blocks and n=frequency: n=0 returns no new layers irrespective of the other flags. For n>0, nominal interior indices are `floor(L*i/(n+1))` for i=1..n. Reject negative/noninteger frequency. Remove layer 0; optionally snap each interior point to the nearest full-attention block within distance 2 (earlier index breaks ties, candidates exclude 0); add exactly L-1 when requested; sort and deduplicate, excluding 0. Thus shallow models may have fewer distinct outputs. Store the actual resolved indices, target topology fingerprint and head dimensions in checkpoints. n=4 gives four fifth-depth points plus the last block before deduplication, not quarter-depth spacing. Full/linear attention both support injection; full attention is a placement preference, not a backend requirement. L-1 means before the final decoder block, not after final normalization.

Only additive input deltas ship. At each selected prompt row: `h_logical := h_logical + delta`, before block input normalization. Scales are multiplied into delta by the producer. All output heads target the same M memory positions. No KV/state carrier, module transfer, target-state-conditioned producer, separate attention bank, token skipping, architecture-wide transformer heads, or new compression policy is added. Input embeddings remain the layer-0 carrier. Recursive children retain their existing side-input adapter with no deep interventions into the side reader.

The first supported engine is V1 GPUModelRunner, dense Qwen3.5 text generation (including its conditional-generation wrapper used for text), TP=PP=1, independent replicas, synchronous scheduling, eager execution, BF16 weights/activations, BF16 then FP8 KV cache, LoRA, chunked prefill and hybrid prefix caching in the supported align mode. Reject intervention requests on unsupported architectures, runner-v2, speculative/streaming inputs, multimodal data, context/sequence/tensor/pipeline parallelism, async scheduling, unsupported hybrid-cache modes, or compilation/CUDA graphs. These are explicit initial limits, not silently ignored payloads. Existing input-only requests retain existing support. Validate the configured experimental engine envelope at initialization/admission; do not silently switch a running engine's options.

Graph/compile support and distributed execution are later milestones outside MVP acceptance. No new attention or GDN kernel is required by the selected eager implementation. The FP8 release check uses BF16 model inputs and the engine's existing KV quantization path; it does not imply arbitrary quantized-weight model support.

## Shared payload and ownership

`activation/harness/hf_utils.py · Interventions` (new typed dictionary; no runtime vLLM import)

```python
class Interventions(TypedDict):
    positions: list[int]                 # sorted unique sequence rows; HF uses prepared physical-row offsets
    layer_inputs: dict[int, Tensor]     # each [M, target_hidden_size]
```

Each logical example owns one optional payload. Single-request generation accepts `Interventions | None`; batched offline generation and pre-collation examples carry one payload/None per logical example. HF `decoder_forward` instead accepts one prepared payload/None per PHYSICAL row of inputs_embeds, with length equal to B; collation alone converts logical coordinates to physical coordinates. No broadcasting one payload across different prompts. The public dictionary is backend-neutral; engine normalization creates a dedicated typed internal struct. Positions are sequence rows, not RoPE/MRoPE coordinates. At vLLM admission they are unpadded prompt rows. At the HF forward boundary they are offsets within the corresponding physical tensor row after collation has applied padding/packing offsets exactly once; the type is shared, but this coordinate conversion is explicit. Every tensor has exactly M rows, floating supported dtype, target width, finite values and validated layer IDs. Empty positions/maps normalize to None; inconsistent emptiness is rejected. Reject duplicate positions rather than permit nondeterministic repeated scatter writes. Do not inspect numerical finiteness in every GPU layer forward; validate once at the ingestion/boundary appropriate to the backend.

`activation/ac_model/ac_model_utils.py · ACOutput` (new internal dataclass)

```python
@dataclass
class ACOutput:
    input_embeds: Tensor
    layer_inputs: dict[int, Tensor] = field(default_factory=dict)
```

Top-level encoding returns [M,d_target] embeddings and per-layer deltas. Recursive encoding returns [M,d_side] embeddings with an empty layer map. One return shape avoids hidden union handling in queues/cache. Existing row count is `output.input_embeds.shape[0]`. The renderer maps each part's rows to the prompt spans and concatenates the per-layer tensors in position order. A missing layer in one part means no contribution for that part; construct zero rows for that layer if combining parts with different layer sets. The request union map retains the same M positions for every layer. AC config is normally uniform per reader, but rendering must not assume all stored historical payloads have the same layer set.

HF keeps differentiable tensors on the execution device; `.to(dtype)` remains in the graph. Serving/recording produces detached, contiguous CPU copies cast once to target activation dtype, with scaled deltas frozen at that boundary. Cached raw encoder output must not be unconditionally BF16-cast before this boundary. Engine admission takes immutable ownership/copies; callers cannot mutate data whose digest has already been used. An entire request's tensors remain reconstructible until request completion/cancellation; GPU staging includes only currently scheduled rows.

No tensor copies belong in JSON or logs. Records contain relative content-addressed references; reports contain shapes/bytes/layers/scales/provenance only.

## vLLM source-backed implementation specification

Source references in this section point to the actual pinned submodule. Implementation updates must rebase these anchors if the pin changes, and repeat both reviews. It is insufficient to add a keyword only at the public generate call.

### V1 — public API to engine IPC

| Actual source seam | Current behavior | Planned change |
|---|---|---|
| [entrypoints/llm.py:418](../../REPOS/vllm/vllm/entrypoints/llm.py#L418), [offline_utils.py:290/523/552](../../REPOS/vllm/vllm/entrypoints/offline_utils.py#L290) | Offline `generate` normalizes and submits prompt batches | Add keyword-only batch-aligned interventions and forward through `_add_completion_requests` → `_render_and_add_requests` → `_add_request` (and `_run_completion` if used), preserving rendered order and sampling-child association; length mismatch is an error |
| [v1/engine/async_llm.py:283,424,550](../../REPOS/vllm/vllm/v1/engine/async_llm.py#L283) | `add_request`, `_add_request`, `generate` build/submit core requests | Add optional per-request interventions, propagate all paths including parallel-sampling children; reject streaming-input interventions |
| [v1/engine/llm_engine.py:218](../../REPOS/vllm/vllm/v1/engine/llm_engine.py#L218) | Synchronous `add_request` processes input | Forward interventions to the common input processor |
| [v1/engine/input_processor.py:input processing](../../REPOS/vllm/vllm/v1/engine/input_processor.py#L247) | Normalizes decoder inputs, LoRA and lengths and builds EngineCoreRequest | Validate against the rendered prompt length/model/envelope, canonicalize owned CPU tensors and attach typed payload; reject truncation of an intervention-bearing prompt unless an explicit aligned remapping is implemented |
| `vllm/inputs/interventions.py` (new), `inputs/__init__.py` | No intervention schema | Export typed public shape and small internal normalized carrier/validation as appropriate; no change to EmbedsPrompt fields |
| [v1/engine/__init__.py:100](../../REPOS/vllm/vllm/v1/engine/__init__.py#L100) | EngineCoreRequest is an array-like msgspec struct with optional prompt tensors | Append an optional typed intervention field without reordering existing fields |
| [v1/serial_utils.py](../../REPOS/vllm/vllm/v1/serial_utils.py) | Typed tensor encode/decode hooks support prompt tensor IPC | Extend only if existing hooks cannot roundtrip the typed nested struct; prove dict integer keys and all tensor bytes/dtypes survive both IPC legs |

Use typed fields (e.g. `positions: list[int]`, `layer_inputs: dict[int, torch.Tensor]`), not `Any` that leaves tensor Ext objects undecoded. `EmbedsPrompt` and chat rendering stay unchanged. This MVP is the Python API used by the harness; OpenAI HTTP schema/content parts are out of scope. Pure token prompts may also carry interventions at valid rows, useful for calibration and parity tests.

### V2 — scheduler state, replay and cache identity

| Actual source seam | Planned change |
|---|---|
| [v1/request.py:Request](../../REPOS/vllm/vllm/v1/request.py) | Retain full normalized payload and lazy per-block intervention digests; `from_engine_core_request` forwards it |
| [v1/core/sched/output.py:NewRequestData](../../REPOS/vllm/vllm/v1/core/sched/output.py#L31) | Add optional typed payload and populate `from_request`; log only metadata |
| [v1/worker/gpu_input_batch.py:CachedRequestState](../../REPOS/vllm/vllm/v1/worker/gpu_input_batch.py#L34) | Retain full payload in request-ID-owned state; avoid a second dictionary keyed by changing batch slots |
| [v1/worker/gpu_model_runner.py:_update_states](../../REPOS/vllm/vllm/v1/worker/gpu_model_runner.py#L1241), new/resumed/replaced/finished branches | Initialize/replace the payload on admission or new prompt, preserve during preemption and cached resume, release on actual finish/abort. No automatic fallback to dummy token IDs when a payload is unavailable |
| [v1/core/kv_cache_utils.py:513–568](../../REPOS/vllm/vllm/v1/core/kv_cache_utils.py#L513) | Add `_gen_interventions_extra_hash_keys` alongside the existing prompt-embedding helper; cache per-block digests on Request |

For a block [start,end), hash only intervention rows in that interval: schema/operation version, target-layer IDs in sorted order, absolute positions, tensor shape/dtype and effective bytes. Existing chained parent hashes invalidate every subsequent block; prior unaffected blocks remain reusable when a later tool span is appended. Empty-intervention blocks add no delta key. Separately fix mixed-input identity in the existing embedding hash path: every mixed prompt block must hash its `prompt_is_token_ids` slice, including blocks before the first delta and requests with no interventions. Equal token IDs/embedding bytes with a different token-vs-embedding mask execute differently. Token-only requests receive no new mask/delta key. This narrowly scoped mask correction preserves output behavior while intentionally changing mixed-input cache identity. All hybrid groups inherit the conservative request token-block identity; do not claim lower-layer-only reuse. Model config/engine identity and existing LoRA identity remain part of the surrounding cache contract. `_gen_lora_extra_hash_keys` uses `lora_name`; the local manager's version-suffixed names must remain intact when adapters reload.

The correctness skeleton may disable prefix caching initially. Completion requires real per-block hashing and hybrid cache hit/miss/recompute proof, not a whole-request salt presented as equivalent performance. Whole-request salt is a documented diagnostic fallback only.

No scheduler algorithm change: interventions affect values for already scheduled token rows, not token counts or readiness. Audit `scheduler.py` preemption and resume paths and add changes only if source-level tracing reveals a payload-lifetime hole. Preserve prompt lengths, chunk budgets and hybrid alignment. Request cancellation and payload load failure must release memory without creating reusable partial cache state.

### V3 — packed worker rows to Qwen3.5 residuals

The pinned runner's `_prepare_inputs` computes `positions_np = num_computed_tokens + query_pos` (~2043), then slices prompt embeds at `start_pos` (~2108). Reuse this request/offset information, not `positions` after conversion to MRoPE.

For request r in a step with base offset b, c already-computed tokens and q scheduled rows, select intervention positions p in [c,c+q). Local flat row is `b + p - c`. Gather only those delta rows. Combine requests into per-layer `(flat_indices, deltas)` tensors on the device. No full `[num_layers, prompt_length, hidden]` staging allocation. Profiling accounts for configured per-step intervention staging; runtime payload bytes have an explicit budget/check against host and GPU staging capacity. Ensure no hidden full-prompt GPU copy on every decode step.

`v1/worker/gpu_model_runner.py · _preprocess/_model_forward/_dummy_run`

```python
# proposed schematic, omitted surrounding runner logic unchanged
model_kwargs["layer_input_deltas"] = scheduled_layer_deltas
```

Every `_preprocess` branch used by mixed prompt embeddings and Qwen3.5 conditional generation must forward this kwarg; `_model_forward` already forwards model kwargs. Dummy/profile calls use None/empty input with valid shapes and include staging-memory accounting. Decode-only steps with no matching original prompt positions pass no deltas. Buffers cannot retain a prior request's rows after a batch changes.

[Qwen3.5 ForCausalLMBase.forward](../../REPOS/vllm/vllm/model_executor/models/qwen3_5.py#L363) currently accepts kwargs but discards them when calling `self.model`. [Qwen3.5 ForConditionalGeneration.forward](../../REPOS/vllm/vllm/model_executor/models/qwen3_5.py#L538) directly calls `self.language_model.model` and also drops kwargs. Explicitly forward `layer_input_deltas` in both wrappers. Dense text Qwen3.5 uses the inherited [Qwen3NextModel.forward](../../REPOS/vllm/vllm/model_executor/models/qwen3_next.py#L619) loop; add its optional argument and select deltas using absolute layer_idx.

`model_executor/models/qwen3_next.py · Qwen3NextModel.forward()`

```python
# immediately before the selected decoder block call
if layer_input_deltas is not None and layer_idx in layer_input_deltas:
    indices, delta = layer_input_deltas[layer_idx]
    hidden_states = hidden_states.index_add(0, indices, delta)
hidden_states, residual = layer(
    positions=positions, hidden_states=hidden_states, residual=residual
)
```

This adds to one summand of the fused residual. The [decoder block](../../REPOS/vllm/vllm/model_executor/models/qwen3_next.py#L487) then combines hidden_states with residual through its existing input norm; when residual=None it captures the already-modified hidden state. Both attention and residual/MLP paths therefore see the intervention. Injection into only the normalized branch would be wrong. Leave its attention/GDN implementation and KV/convolution/recurrent updates unchanged. The inherited loop change must be a no-op for other models with no deltas; unsupported request architectures fail at admission. Sequence-parallel sharding is excluded from this first implementation, so indices refer to full hidden rows.

CUDA graphs/torch.compile are explicitly disabled for intervention engines in this MVP, allowing sparse dynamic gathers/index_add without claiming graph support. Later graph support would use configured layer sets/stable buffers and needs its own review; it must not happen accidentally via the model decorator/compiler path.

### V4 — installation and deployability

The submodule gitlink alone does not change the installed vLLM package. Chosen release binding: add `[tool.uv.sources] vllm = {git = "https://github.com/amlatyrngom/vllm.git", rev = "<full implementation commit>"}` to root pyproject.toml, regenerate uv.lock, and require its locked Git SHA to equal the submodule gitlink. The placeholder is replaced by a real pushed commit during implementation; no moving branch is allowed. Development can use an editable submodule build, but the clean acceptance environment must install the locked immutable source. Fork commits used for tests/builds are named in provenance.

Actual build seams: `IB/REPOS/vllm/pyproject.toml` pins build torch==2.13.0 and supports Python 3.14; setup.py supports `VLLM_USE_PRECOMPILED`, `VLLM_PRECOMPILED_WHEEL_LOCATION`, `VLLM_PRECOMPILED_WHEEL_COMMIT` and `VLLM_VERSION_OVERRIDE`. For this Python-only patch, reuse only the exact upstream 0.28.0 compatible platform wheel's native extensions, with its URL/digest verified against the existing lock; set an explicit local verified wheel location so setup never fetches moving upstream HEAD. Verify ABI/torch/CUDA compatibility in the GPU build; fail rather than silently substitute a different native wheel. If precompiled reuse fails, build this pinned source normally and retain a hashed wheel as build evidence.

`activation/cloud/Dockerfile` currently copies only pyproject/lock before `uv sync --frozen --no-install-project`; add Git/build prerequisites and the explicit pinned build environment there. `activation/cloud/sky.py:exec` resyncs uploaded dependencies to the lock, so it must inherit the same verified build settings or use the already built matching cache. `.skyignore` currently excludes all IB: keep that privacy boundary; release installs the public immutable fork from its locked Git source rather than uploading IB. A narrowly scoped local development sync can be added later if needed, never broad `IB/` inclusion. Image/source revision checks assert imported capability plus the installed Git/build SHA, not merely version string 0.28.0. This deployability gate must be exercised in a clean environment before release.

Read `IB/REPOS/vllm/AGENTS.md` before implementation; use uv/venv Python, focused existing test suites, lint and model-affecting evaluation. No upstream PR is part of this scope. Fork commits should include appropriate assistance attribution, without fabricating the user's sign-off.

## Local source changes and end-to-end flow

M0 first applies/reconciles the reviewed co-design handoff; all references to its helpers below are explicitly AFTER that prerequisite. Product code remains in the real checkout during implementation, not duplicated in TMP. Snapshot/tag before long checks. New probes and detailed benchmark scripts stay in IB; extend only approved existing mainline examples/tests as necessary.

### M0 — reconcile the actual training baseline

Inventory committed product, handoff manifests and any concurrently delivered newer product before applying anything. Verify source hashes and `git apply --check`; apply the accepted co-design patch, preserving user prototypes and unrelated changes. If newer co-design code arrives first, prove it satisfies the same contracts and skip duplicate application. The patch provides captured rows, fixed-row AgentTrainer, whole-history KL/SFT, shared-model checkpoint adapter context, and serialization helpers. Run its focused CPU regression checks before changing carrier semantics. Its recorded GPU proofs remain historical evidence and do not certify interventions.

Do not silently import the whole live reconstruction trainer. Read the latest completed POC ledger at implementation freeze. If a promising POC initialization/recipe is selected as the quality baseline, recover its immutable source by manifest or reimplement the small specified behavior and verify it; record the resulting local SHA before comparing carriers. Missing remote source is not permission to claim a POC recipe was reproduced. The intervention implementation itself is based on the available co-design source and can proceed independently of unverified recipe ports.

### M1 — output heads, types, config and migration

Files: `activation/ac_model/ac_model.py`, `ac_model_utils.py`, exports in `activation/ac_model/__init__.py` only if needed; shared payload/position helpers in `activation/harness/hf_utils.py`.

- Resolve layer indices against `target.config.get_text_config().layer_types` / existing canonical layer descriptions; do not assume every fourth block from the model name.
- Extend `ActivationContextModules` with a ModuleDict of independent DeltaHeads from the shared pre-target-head summary hidden states. Keep existing `target_head` and `recursive_adapter` weights/semantics, including any verified POC initialization already present at M0.
- Concrete small default head: `Linear(d_side,r) -> SiLU -> Linear(r,d_target)`, where `r=max(1,min(d_side,d_target)//4)`. Parameters, RMS normalization and learned scalar are FP32. The dimension rule is internal/versioned metadata, not another required user knob. Independent heads, shared encoder, no extra attention.
- `delta_l = alpha_l * unit_rows(head_l(z))`. Estimate destination residual RMS once on a small fixed training-only calibration sample with the chosen reader adapter and no deep deltas; initialize alpha to 0.01 times that RMS. This is a conservative starting hypothesis, not an empirically optimal value. Keep raw head weights nonzero so every branch learns immediately. Calibrate every resolved layer in one target pass; record sample identity/statistics. Lifecycle: add `ActivationContextModel.calibrate_interventions(training_prompts)` after loading the intended reader adapter and before the first deep encode/rollout or optimizer construction. ACTrainer invokes it once on a deterministic training-only sample if needed. A fresh rollout-only deep model requires an explicit calibration call or a calibrated checkpoint; `prepare()` fails clearly while uncalibrated. Input-only mode has no calibration requirement. A deep checkpoint restores calibration metadata and learned scales without silently recalibrating. Never calibrate on held evaluation material or repeat silently on checkpoint load. Optional future tuning uses observed delta/residual ratios, not an unsupported hardcoded fast scale LR.
- `encode`, `encode_async`, `encode_batch`, `_encode_requests`, `_encode_prepared_batch`, queue futures and RowCache return/retain ACOutput. Training never hits detached rollout cache; cache bytes sum all tensors, and head/scale changes invalidate producer version. Remove the current unconditional BF16 cast in RowCache in favor of preserving stored dtype and performing one explicit serving cast.
- Save resolved topology, head width/parameters/scales/calibration provenance; bump architecture version. A v2 checkpoint with absent fields loads explicitly input-only (`frequency=0`), preserving identical embeddings. Opting a legacy checkpoint into deep mode is an explicit migration that initializes only new heads/scales and resets incompatible optimizer state; never substitute default frequency=4 during legacy deserialization. Native deep checkpoint resume requires matching target topology/heads; optimizer restoration covers every new parameter with stable group names.

At d_side=d_target=2560 the five 640-bottleneck heads contain about 16.4M weights, versus about 262M for five copies of the current 4x RowHead. The FP32 parameter+gradient+two-Adam-moment lower bound is ~262 MB decimal versus ~4.2 GB; excludes biases, activations, allocator overhead and any master copies. Verify actual parameter/memory accounting after construction.

### M2 — HF forward, generation and gradient correctness

Files: `activation/harness/hf_utils.py`, `loaded_model.py`; existing `module_manager.py` checkpoint/adapter ownership only if required by M0 integration.

- `LoadedModel.decoder_forward(..., interventions=None)` accepts a list of B prepared payloads/None for inputs_embeds of shape [B,S,D]; it checks len==B and positions<S and does not apply offsets again. Collation owns logical-to-physical mapping: ordinary padded batches have one payload per physical row, while packed E-example batches with B=1 merge all E payloads into one map with segment offsets already added. Within each physical row, union positions are sorted, and a layer absent at a position receives zero contribution. Preserve masks, position_ids, sequence boundaries and chosen adapter.
- hf_utils installs idempotent, persistent decoder-layer wrappers with no parameters/state_dict key changes. Layer kwargs carry the prepared intervention for this forward; wrappers consume the private argument before calling original attention/GDN code. Injection occurs before the original HF forward executes `residual = hidden_states`. Use functional scatter/index_add, not detached copies or mutation of shared hidden tensors.
- Compose with the co-design `checkpoint_adapter_scope`: non-reentrant checkpoints capture immutable per-forward payload and adapter context. A teacher/side call made before student backward cannot replace the pending reader's payload/adapter. Missing interventions is a no-op. Frozen bases still propagate gradients into live deltas. Reject unsupported HF architectures instead of silently dropping kwargs.
- Add `hf_utils.generate_with_interventions(...)` for the direct HF sample generation path. Its initial supported B-row batch has one prepared payload per physical input row, identical to decoder_forward; packed independent-example generation is not supported. It keeps full prompt-relative payload and derives current sequence offsets from the actual generation input slicing/cache position, not reset positional encodings. Initial support is greedy or single-sequence sampling (num_beams=1, num_return_sequences=1), with explicit rejection of unsupported expansion/beam paths. Apply at prefill and any full prompt recomputation, never again to cached prompt states. Scope generation bindings until generation completes and restore in finally; forward wrappers stay installed. Validate HF's model-kwargs validation/prepare_inputs_for_generation propagation on pinned Transformers. A no-cache reference generation provides a parity oracle.
- Local `LoadedModel` generation convenience methods and AC sampled-completion paths route through this helper whenever interventions are supplied; ordinary HF generation is unchanged. No decoder-only training helper is presented as covering generation automatically.

### M3 — agents, records, cache and frozen replay

Files: `activation/agent/agent.py`, `agent_config.py`, `agent_utils.py`, `rollout_caching.py`, `rollout_manager.py`, `rollout_reporter.py`; `activation/common/ac_parts.py` only if shared span assembly requires it. Extend the established helpers rather than create a second store.

- `_encode_parts` receives ACOutput; `_begin_segment` and `_append_step` cast/capture complete bundles. `prepare_request` assembles original token IDs/full mixed embeds/mask plus all interventions for the current causal segment. `_submit` and `submit_probe` forward that payload. Append, tool output, compaction, parent/subagent exchange and simulated/resumed requests retain exact span offsets. A newly encoded future part never changes prior captured bundles.
- Extend the co-design `capture_ac_spans`, `load_ac_rows`, `serialize_ac_rows`, `deserialize_ac_rows`, `release_ac_rows`, and report filters. Each span retains `start`, `length`, input rows and an optional integer-layer map of effective delta tensors. Serialized form carries explicit payload schema version and per-layer relative paths, dtype/shape/content digest plus target/producer provenance. Normalize JSON layer keys to ints on load.
- Publish all content-addressed tensors atomically before the trajectory record, reuse identical files, preserve lazy references, and release in-memory aliases only after durable JSONL publication. Existing file-safe path confinement and digest validation apply to every new tensor. Missing or corrupt advertised delta files are hard errors, not input-only fallback. Old records with no intervention schema mean empty deltas.
- Resume uses saved effective deltas and saved input rows, not today's encoder. A new request sends the complete saved payload again. Payload caches account for all heads/scales; rollout records include target topology and producer version but bytes are authoritative. Stale LoRA/output identities miss cache via the existing adapter version mechanism.
- Step-relative positions become segment-relative during history assembly, then batch/packed positions during collation. Reasoning/token rewrites must map spans and all deltas together; reject a rewrite that cuts an intervention span unless an explicit tested row-slice transform exists. Do not silently drop rows/targets. Segment resets remain separate causal examples.

### M4 — public serving interfaces and fork integration

Files: `activation/harness/loaded_model.py`, `vllm_wrapper.py`, runtime execution/config capability checks, `activation/cloud/sky.py`, `pyproject.toml`/`uv.lock`/image build inputs as required for the pinned installation.

`LoadedModel.engine_submit_tokens()` and `VLLMWrapper.submit()`/`_Replica.submit()` gain optional interventions. `_Replica._collect()` forwards it to AsyncLLM.generate with the same EmbedsPrompt. `engine_chat_many`/wrapper `chat` may accept a batch-aligned payload only after prompts are rendered and positions known; never guess raw-text offsets. Existing callers supplying none stay identical. Check intervention capability once when the engine starts and report imported fork identity; no silent stock-wheel fallback. Preserve independent replica assignment and full payload ownership through the future's completion.

Implement V1–V4 above. First land a BF16 eager skeleton with prefix caching disabled, then enable hashed hybrid prefix caching and FP8 KV after their separate correctness gates. Do not call the skeleton the completed serving MVP before those gates.

### M5 — differentiable AC training, AgentTrainer and inspection

Files: `activation/ac_model/ac_model_training.py`, `ac_model_study.py`, `ac_model_reporter.py`; `activation/agent_training/agent_training_utils.py`, `agent_trainer.py`; execution-profile/accounting code only where new head memory affects real limits.

- ACTrainer builds full-history student embeddings and intervention tensors from live ACOutput. All KL/SFT training, eval and current-reader reference paths intended to read AC use the same payload assembly. Full-text teacher/no-context controls receive no interventions. Future assistant targets must not enter the producer's source span.
- `_sample_completions` currently calls `peft_model.generate` directly; replace the student call with the HF intervention-aware helper and supply effective payload. The teacher call is input-only. `ac_model_study` vLLM submission also sends deltas. Sampling and teacher-forced evaluation must not disagree about which carrier the reader saw.
- AgentTrainer uses captured bundles only. Extend its `TrainingExample`, `Collated.ac_spans`, `training_inputs_embeds` companion intervention assembly, `validate_fixed_rows`, reference/current-policy logprob forward and evaluation. Old-policy/reference and train passes consume exactly the same frozen tensors; only reader adapter parameters update. Keep existing assistant loss masks/clipping objective.
- Padded and packed collation transforms intervention coordinates with the same segment offsets as input rows and emits B physical-row payloads; a packed B=1 batch merges all logical examples before decoder_forward, preserving the existing separate causal masks. Preserve Qwen3.5 GDN and full-attention segment boundaries. Padding/spacer rows receive no delta and no loss. Delta heads get gradient only from examples whose AC they produced; frozen recorded payloads get none.
- Persistent optimizer groups include all trainable new heads/scales, with no duplicate parameters for shared bases. Retain explicit optimizer reset/resume behavior. Report per-head and encoder/reader gradient norms, preclip norms, delta RMS/residual-calibration ratio, scale trajectory, payload bytes and encode/forward latency. Do not inherit a stale checkpoint's optimizer hyperparameters contrary to the resumed config.
- Preserve meaningful existing input-head checkpoints. A POC skip initialization is a promising fresh-run baseline, not evidence to replace each deep head with an identity. Recover its verified code before claiming compatibility. No mandatory fast scale learning-rate group or universal frozen-reader schedule is introduced by this MVP.

### M6 — validation and bounded quality comparison

Keep detailed probe code in IB/TMP or the task artifact bench; extend focused existing vLLM suites under its contribution instructions. Main project additions to tests require the normal plan approval of those specific files; prefer extending the co-design examples already approved rather than inventing broad test scaffolding.

| Gate | Smallest evidence that can disprove correctness | Required outcome |
|---|---|---|
| Config/legacy | Boundary/shallow model schedules, snap ties, n=0, last index, checkpoint migration | No accidental layer0/duplicate, zero disables everything, old state stays input-only |
| Types/IPC | Actual serial_utils roundtrip and EngineCoreRequest/scheduler reconstruction | Every layer key, position, shape/dtype/byte preserved; bad payload rejected |
| HF derivatives | Tiny Qwen3.5 full+linear interventions: plain vs non-reentrant; two forwards/different adapters+payloads before backward | Outputs/gradients agree within declared dtype tolerance; all expected heads/encoder/reader receive gradients; no stale payload |
| Collation/causality | Individual vs unequal padded/packed examples, adjacent AC spans, intervention at final block; E=2 packed into B=1 with collator offsets [0,8] maps logical positions [2] and [1] to one payload at [2,9] | No cross-example/preceding-token influence; masks and target alignment preserved |
| Capture/replay | Serialize/load/resume/compaction/cancellation with whole bundles and missing/corrupt files | Frozen bytes identical; missing advertised tensors fail; no re-encoding |
| HF generation | Cached vs no-cache single-sequence generation with same interventions, first generation step vs teacher-forced logits | Only original prompt rows are intervened; no repeated cached-state addition |
| vLLM BF16 | Same model/adapter/embeds/deltas in HF reference and vLLM; n=0 and zero-delta controls | Greedy/logprob parity with documented tolerances; quantify kernel baseline difference separately |
| Scheduler | Mixed requests, reorder/condense, chunk cuts through AC spans, forced real preemption+recompute, cancellation | Same logits/continuation as uninterrupted reference; retained full payload; no memory leak/stale rows |
| Cache/FP8 | Identical payload hit; changed row/layer/position/LoRA or mixed token/embedding mask miss (including before first delta); append-only reuse of earlier blocks; hybrid align block lengths | Correct results and observed hash/cache behavior; FP8 compared against matching FP8 baseline, not promised bitwise HF equality |
| Deployment | Clean env/image imports fork SHA; installed package paths/capability; locked immutable fork source installed | Stock vLLM cannot masquerade as fork; lock/gitlink/installed SHA agree; reproducible install without credentials in image |
| Learning | Paired input-only vs deep on fixed clean train/held split with correct, shuffled and absent content controls | Report clean held quality and causal content sensitivity, not merely training loss; no guaranteed improvement threshold invented |

Before declaring a training comparison, choose and freeze one source-verified recipe for BOTH carriers; initialize shared encoder/input head/reader identically and add only deep heads. The default intervention config adds up to five delta tensors: payload becomes up to 6x input-only at equal M and dtype. Report both equal-position comparison and payload/parameter/memory costs; a later equal-byte comparison is separate, not implicit fairness. Track fixed-base/no-context gold NLL alongside teacher KL to expose reader shortcuts; when evaluating a no-context condition remove every delta as well as the input AC content and remap positions. Shuffled-content control swaps whole bundles only within matching row-count, target width and layer-set buckets, retaining the recipient span positions. If buckets are insufficient, explicitly re-render and realign a separate control; never truncate/pad donor information silently. Report control coverage and do not compare a differently selected subset without labeling it.

Do not announce learning failure before the control recipe has had a credible opportunity to transition. POC tiny/short runs missed learning onsets; comparisons must share data, schedule, updates and stop criteria. Long cloud campaigns are not launched by approval of this plan alone unless a budget is explicitly included later. Mechanics gates use the smallest practical GPU fixtures and don't touch active POC jobs.

## Training findings informing the plan

| Evidence | What we take into this MVP | What remains unproven |
|---|---|---|
| [RAG four-arm ablation](../AGENT_AC_PRETRAINING/rag_ac_poc/ABLATION_RESULTS.md): held AC gold NLL .584 vs no-context .586 | Require absent/shuffled-content and fixed-base controls; reader LoRA remains allowed | Extra heads alone solve the shortcut |
| [Reconstruction ledger](../AGENT_AC_PRETRAINING/ac_recon/LEDGER.md), 11:20–13:30: skip+gold recipe learns sooner, including trainable 1/4 reader | Preserve useful input channel; match source-verified initializations across comparisons | Skip-alone effect, optimal head architecture; earlier 'trainable reader fails at 1/4' is superseded |
| Ledger scale observations and gate LR arms | Calibrate residual-relative scales and measure their learning | A universal 3e-3 scale LR; identity skip for deep heads |
| Ledger 13:30 MD .800 vs MK 1.581 at 512 updates | Match and report schedule/warmup; optional verified schedule recipe | Warmup vs cosine causality, a universal optimum; supersedes unique-data-only explanation |
| [Review 2](../AGENT_AC_PRETRAINING/ac_recon/CODE_REVIEW_2_fresh.md) and clean-panel audit | Fixed deduplicated split; label original/clean results | Old headline numbers are clean or final |
| Ledger SR resume test | New head/scalar optimizer state and schedule replay must be tested | Old resume tests certify new intervention payloads |

The [reviewers' initial training report](TRAINING_SOURCE_FINDINGS.md) identifies which mechanics exist in co-design source and which POC product snapshots are missing. Review evidence freezes with this plan; updates during ongoing experiments are integrated at implementation baseline selection rather than silently changing the reviewed design.

## Review, milestone order and approval boundary

Order: M0 source reconciliation → M1 types/heads → M2 HF → M3 records → M4 engine → M5 all training/eval callers → M6 acceptance. M2/M3/V1 preparation may proceed independently after the shared contract is fixed, but the combined release requires all gates. The two requested reviewers assess correctness and training friendliness independently, then the solver classifies each finding as genuine/cheap-nit/reviewer-overkill/bad-review and updates both documents. Substantial corrections receive another pass. Final reports and disposition are linked beside the human plan.

This plan deliberately names the non-vLLM interfaces and source prerequisite in the human projection. The detailed engine table, exact request/worker/model path, hashing rule, support exclusions and validation matrix here remain reviewer-facing authority. Both documents must agree before approval. Both independent final reviews pass with no unresolved findings: [correctness](CORRECTNESS_REVIEW.md), [training friendliness](TRAINING_REVIEW.md). [Review disposition and completed checks](REVIEW.md) records the five resolved findings and the [source manifest](SOURCE_MANIFEST.json). No planned GPU check is reported as already run.
