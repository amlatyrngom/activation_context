# Deep AC inputs for Qwen3.5

Add per-layer residual deltas to the existing AC embedding channel, with one payload shared by HF training, vLLM serving and recorded agent replay. This plan leads with the local interface changes; the [agent implementation source](PLAN.md) gives the detailed vLLM source map. Product implementation has not started.

## Executive Summary

The AC encoder runs once and produces its existing input embeddings plus small independent FFN heads for selected reader layers. Each head's learned scale is already included in its delta. The reader adds those rows before the selected block's input normalization. The reader LoRA remains the only reader-side learned adapter; no module is transferred with a request.

| Boundary | Interface / ownership |
|---|---|
| AC encoder → renderer | `ACOutput(input_embeds, layer_inputs)`; the renderer attaches the existing memory-span positions |
| Renderer → vLLM | Existing `generate` + `EmbedsPrompt`, plus `interventions=` |
| Renderer → HF | Existing `decoder_forward`, plus one prepared `interventions` payload per physical tensor row; collation maps logical spans, implementation stays in `hf_utils.py` |
| Agent → record/cache | Complete effective embeddings and deltas, stored as immutable tensors with relative references |
| AC training | Re-encode raw content with gradients into encoder, heads/scales and reader LoRA |
| Agent training | Replay captured tensors exactly; train only the reader LoRA |

The user-facing configuration is settled:

`activation/ac_model/ac_model.py · ActivationContextModelConfig`

```python
intervention_frequency: int = 4  # Number of interior points; 0 disables all interventions.
add_last_layer_intervention: bool = True
prefer_attention_interventions: bool = True
```

For n>0, interior positions are `floor(L*i/(n+1))`. Snap to a full-attention block within two indices when requested, choosing the earlier one on a tie; add exactly `L-1` if requested; exclude 0, sort and deduplicate. Save the resolved indices. The default means four fifth-depth points plus the last block before deduplication: up to five deltas per memory row. Layer 0 remains the existing embedding channel. Zero disables all new outputs, including the last-layer flag.

**Repository preparation is complete.** [GitHub fork](https://github.com/amlatyrngom/vllm/tree/ac-interventions-v0.28.0) created and branch pushed; [submodule](../../REPOS/vllm/) pinned to upstream v0.28.0 commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`. Local product snapshot: `0a77891`; submodule registration: `13f19d9`. No root-repository push or upstream PR was made.

**One prerequisite is real:** this checkout contains older product code than the ongoing POC documents describe. The reviewed [co-design handoff](../AGENT_AC_PRETRAINING/codesign/README.md) passes `git apply --check` here and supplies recorded-row replay, full-history KL/SFT and checkpoint adapter binding. M0 integrates it deliberately before extending those mechanisms. Experimental reconstruction source tags are absent locally; their reports inform this plan but are not treated as installed code.

**MVP runtime:** dense Qwen3.5 text, V1 GPU runner, one GPU per independent replica, synchronous scheduling, eager execution, BF16 activations, LoRA, chunked prefill, hybrid prefix caching in align mode, and BF16/FP8 KV cache. Graphs/compilation, parallel sharding, runner-v2, speculation, streaming inputs, multimodal requests and unvalidated model/cache modes are explicitly unsupported for intervention requests initially. Normal input-only behavior remains unchanged. The fork must actually be installed; a submodule alone does not replace the stock package.

## Requested Decisions

Accepted: additive deltas; learned producer scales; reader LoRA; shared encoder with independent small heads; unchanged recursive side-input path; complete immutable payload for replay; the three config fields above. No architecture choice is reopened here.

<article class="decision" data-tressoir-decision data-decision-state="unresolved" aria-labelledby="deep-mvp-approval">
  <header class="decision-header"><div>
    <h3 class="decision-title" id="deep-mvp-approval">Proceed with this implementation scope?</h3>
    <p class="decision-context">Approve the source-reconciliation prerequisite, interface changes and correctness gates below. Long quality campaigns need a separately stated budget.</p>
  </div><span class="decision-state" data-decision-indicator role="status" aria-live="polite">Unresolved</span></header>
  <fieldset class="decision-options">
    <legend class="visually-hidden">Implementation decision</legend>
    <label class="decision-option"><input type="checkbox" data-tressoir-input="deep_mvp.plan.approve"><span><strong>Proceed with M0–M6</strong><small>Implement the reviewed payload, HF and vLLM paths, records and focused validation.</small></span></label>
    <label class="decision-option"><input type="checkbox" data-tressoir-input="deep_mvp.plan.revise"><span><strong>Revise the plan first</strong><small>Specify the changes below; the fork and local snapshot are already prepared.</small></span></label>
  </fieldset>
  <div class="field decision-feedback"><label for="deep-mvp-response">Free Response</label><textarea id="deep-mvp-response" rows="2" data-tressoir-input="deep_mvp.plan.feedback" data-tressoir-autogrow="2:6" placeholder="Adjust scope, interfaces or validation…"></textarea></div>
</article>

## Milestones

<details class="card" data-tressoir-markdown>
<summary><span class="card-title">M0 — Establish the actual training baseline</span><span class="card-oneliner">Integrate the reviewed prerequisites without disturbing ongoing experiments.</span><span class="card-badge">Planning</span></summary>

#### Planning Overview

Current local AgentTrainer rejects AC-bearing examples and records lack the newer captured-tensor helpers. Apply/reconcile the already reviewed co-design handoff against its verified hashes, preserving prototypes and any newer incoming changes. Do not copy an evolving remote POC worktree over the local source.

#### Planned Changes

`IB/ARTIFACTS/AGENT_AC_PRETRAINING/codesign/changes.patch`

```text
Verified co-design prerequisite:
  agent_utils: captured tensors, relative publication, lazy loading and release
  agent_training_utils: fixed-row training and padded/packed collation
  ac_model_training: whole-history KL/SFT
  hf_utils / loaded_model: checkpoint-bound adapter context
```

The exact existing delta is in [CODESIGN_ROUND](../AGENT_AC_PRETRAINING/CODESIGN_ROUND.tressoir.md); this plan does not recreate it. Run its focused CPU regressions after integration. Prior GPU evidence remains historical, not proof of the new interventions.

If we select a POC initialization or optimizer recipe for the quality comparison, first recover its immutable source or explicitly reimplement and test the small behavior. Record a new baseline SHA. The intervention work can proceed without pretending every live POC feature is present.

</details>

<details class="card" data-tressoir-markdown>
<summary><span class="card-title">M1 — AC emits a complete payload</span><span class="card-oneliner">Keep the input head; add small independent delta heads and explicit checkpoint migration.</span><span class="card-badge">Planning</span></summary>

#### Planning Overview

All heads read the shared summary states before `target_head`. No extra transformer runs. Each intervention head has a bottleneck `r=max(1,min(d_side,d_target)//4)`, FP32 parameters and RMS/scale arithmetic, and a separate learned scalar. Calibrate initial delta magnitude to 1% of each destination layer's residual RMS on a small fixed training-only sample; this is a starting hypothesis, with diagnostics, not a proven optimum. Do not recalibrate on load. `calibrate_interventions(training_prompts)` runs after reader-adapter loading and before the first deep encode or optimizer construction; ACTrainer can invoke it once on its fixed training sample. Rollout-only use needs a calibrated checkpoint or explicit calibration; an uncalibrated deep model fails clearly.

#### Planned Changes

`activation/ac_model/ac_model_utils.py · ACOutput / DeltaHead`

```python
@dataclass
class ACOutput:
    input_embeds: Tensor
    layer_inputs: dict[int, Tensor] = field(default_factory=dict)

# DeltaHead: Linear(d_side, r) -> SiLU -> Linear(r, d_target)
# output = learned_scale * unit_rows(raw_output), with FP32 arithmetic
```

`activation/ac_model/ac_model.py · _encode_prepared_batch()`

```python
z = summary_hidden_states
output = ACOutput(
    input_embeds=modules.target_head(z),
    layer_inputs={int(layer): head(z)
                  for layer, head in modules.delta_heads.items()},
)
```

These are proposed interface excerpts; surrounding existing encode logic is omitted. `encode`, `encode_batch`, `encode_async`, the queue and RowCache carry ACOutput. Recursive results use `recursive_adapter(z)` as input_embeds and an empty layer map. Cache byte counts include every tensor; producer version changes on any head/scale update. Preserve dtype in the encoder cache and cast once at the serving boundary.

Five copies of today's 4× RowHead would be unnecessarily large. At side/target width 2560, the proposed bottlenecks add about 16.4M weights versus about 262M; actual runtime memory must still be measured.

Save the resolved layers, target topology, head width, scales and calibration provenance. Old checkpoints/records without intervention fields mean input-only, never the new default of 4. Explicit legacy-to-deep migration initializes only new heads/scales and resets incompatible optimizer state. Existing input-head weights remain intact. Fresh-run POC skip initialization is a separately verified baseline choice; it is not blindly applied to deep heads.

</details>

<details class="card" data-tressoir-markdown>
<summary><span class="card-title">M2 — The same intervention reaches HF forwards and samples</span><span class="card-oneliner">Keep backend mechanics in hf_utils.py, including checkpoint recomputation.</span><span class="card-badge">Planning</span></summary>

#### Planning Overview

One logical request/example owns one payload. Before collation, positions are sorted unique indices in the unpadded sequence, independent of RoPE positions. vLLM consumes those prompt indices. HF collation applies padding/packing offsets once and emits one payload per physical [S,D] row; a packed [1,S,D] batch merges all logical examples into one payload. decoder_forward checks list length equals physical B and never applies offsets again. Multiple memory spans concatenate in that order. Every delta tensor has the same M rows and target width. `None` means no intervention.

#### Planned Changes

`activation/harness/hf_utils.py · Interventions`

```python
class Interventions(TypedDict):
    positions: list[int]  # prompt indices; prepared physical-row offsets at HF boundary
    layer_inputs: dict[int, Tensor]  # [M, target_hidden_size], scales included
```

`activation/harness/loaded_model.py · decoder_forward()`

```python
hidden = reader.decoder_forward(
    inputs_embeds=embeds,
    attention_mask=mask,
    position_ids=positions,
    lora_name=reader_lora,
    interventions=[payload],  # one per physical row; collation already mapped offsets
)
```

Persistent, parameter-free layer wrappers in hf_utils consume per-forward arguments and add deltas before the original layer captures its residual. They compose with the co-design non-reentrant checkpoint/adapter context. Two pending forwards must retain their own payload and adapter through backward; no mutable global current-payload field, temporary removed hook, or detached training copy.

`activation/harness/hf_utils.py · generate_with_interventions()`

```python
output = generate_with_interventions(
    peft_model,
    inputs_embeds=embeds,
    interventions=[payload],
    **generation_kwargs,
)
```

HF sampled completions currently call `peft_model.generate` directly, so this helper is required in addition to decoder_forward. It retains full prompt payloads, maps current cache/input offsets, and intervenes only on executing original prompt rows. Its payload list matches physical input batch rows; independent packed-example generation is not supported. Initially support greedy/single-sequence sampling; reject beam expansion and other unvalidated paths. Cached and no-cache generation must agree. The helper preserves HF output conventions and cleans up its generation scope on failure.

</details>

<details class="card" data-tressoir-markdown>
<summary><span class="card-title">M3 — Agent steps, serialization and caches retain every delta</span><span class="card-oneliner">A resumed request reads exactly what the original request read.</span><span class="card-badge">Planning</span></summary>

#### Planning Overview

The agent captures the complete effective bundle when it renders a part: detached CPU tensors in reader activation dtype, with scales already multiplied in. Recomputing after KV eviction or resuming a trajectory uses those tensors, not a newer producer.

#### Planned Changes

`activation/agent/agent.py · prepare_request()`

```python
# Proposed return extends today's token/embed/mask tuple.
token_ids, embeds, mask, interventions = agent.prepare_request()
```

`activation/agent/agent_config.py · TrajectoryStep.ac_spans / AgentRunResult.prompt_ac_spans`

```python
# In-memory span; start is relative to the existing step or prompt token list.
{"start": start, "length": M, "rows": input_rows,
 "layer_inputs": {layer: effective_delta}}
```

Extend the existing co-design helpers in `agent_utils.py`: capture, load, serialize, deserialize, release and report filtering. Publish per-layer tensors to the same content-addressed relative-file store; persist a payload schema version, layer/dtype/shape/hash and producer/target provenance. Normalize JSON layer keys to integers. Missing/corrupt advertised tensors fail explicitly. Old row-only records retain empty layer maps.

`rollout_caching.py`/`rollout_manager.py` publish all tensor files before JSONL and release live aliases only after publication; `rollout_reporter.py` exposes metadata, not tensor copies. Append, compaction, synthetic runs, parent/subagent exchange and resume keep the same span-offset rules. Reasoning/token rewrites transform embeddings and deltas together or fail if they cut an unsupported span. Segment resets remain separate causal sequences.

The renderer supplies the complete payload for every generation request. The engine retains it through completion, stages only currently executing rows, and applies a delta once to a freshly computed layer input. This is replay equivalence; repeatedly adding a delta to an existing state would be incorrect.

</details>

<details class="card" data-tressoir-markdown>
<summary><span class="card-title">M4 — Generate accepts interventions in the pinned vLLM fork</span><span class="card-oneliner">Small public change; explicit engine transport, model forwarding and cache work underneath.</span><span class="card-badge">Planning</span></summary>

#### Planning Overview

Keep EmbedsPrompt unchanged. The local wrappers forward an optional request payload; vLLM owns conversion to packed scheduled rows. The [agent-facing V1–V4 specification](PLAN.md#vllm-source-backed-implementation-specification) gives actual file/symbol references and intended changes.

#### Planned Changes

`activation/harness/vllm_wrapper.py · _Replica._collect()`

```python
async for output in engine.generate(
    prompt=prompt,  # current EmbedsPrompt or token prompt
    sampling_params=sampling_params,
    request_id=request_id,
    lora_request=lora_request,
    interventions=payload,
):
    ...
```

`LoadedModel.engine_submit_tokens`, `VLLMWrapper.submit`, `_Replica.submit` and batch chat submission propagate the same argument. Batch entry points require one payload per rendered prompt; raw-text offsets are not inferred. Stock vLLM must fail the capability check, not silently ignore the argument.

| Engine change | Why it is necessary |
|---|---|
| Async/offline generate → input processor → typed EngineCoreRequest | Preserve tensors and integer layer keys across IPC; validate unsupported configurations before execution |
| Request → scheduler NewRequestData → worker CachedRequestState | Retain complete immutable payload through preemption, resume, batch reorder and cancellation |
| Worker scheduling map | Select prompt rows in the current chunk and gather sparse per-layer GPU data, using sequence offsets rather than MRoPE |
| Both Qwen3.5 forward wrappers + inherited Qwen3Next loop | Existing wrappers discard kwargs; add to one summand of the fused residual before normalization |
| Per-block cache hashes | Include effective deltas, layer/position/schema/dtype/shape and the mixed token/embedding mask; changed content invalidates affected blocks and descendants, while unchanged earlier blocks remain reusable |
| Fork install/build identity | Pin uv’s Git source to the submodule SHA; verify native-wheel compatibility, imported capability/source and cloud resync |

The dependency lock uses the fork’s full immutable Git commit, matching the submodule. Docker and cloud sync must install that source with explicitly pinned compatible native extensions; the existing exclusion of IB from cloud uploads remains. A clean-install check proves the stock wheel cannot be used accidentally. Mixed-prompt block hashes also include the token/embedding mask before the first intervention and on input-only mixed prompts, fixing an existing identity gap; token-only cache keys stay unchanged.

No token scheduling algorithm or attention/GDN kernel is redesigned. Start with BF16 eager/no-prefix-cache correctness, then complete the required hybrid prefix-cache and FP8 KV checks. The initial release excludes graphs/compile and sharded execution explicitly. Full engine details and failure/lifetime rules are part of the reviewed source, not hidden implementation discretion.

</details>

<details class="card" data-tressoir-markdown>
<summary><span class="card-title">M5 — Training, evaluation and agent replay consume the same bundle</span><span class="card-oneliner">Every model caller is covered, including HF sampled completions.</span><span class="card-badge">Planning</span></summary>

#### Planning Overview

ACTrainer differentiably re-encodes raw parts; AgentTrainer replays captured tensors and updates only reader LoRA. Full-text teacher and no-context reference paths receive no interventions. The same effective payload reaches training, evaluation, sampled completions and serving.

#### Planned Changes

`activation/ac_model/ac_model_training.py · _example_loss() / _sample_completions()`

```python
outputs = ac_model.encode_batch(example.part_requests)
student_embeds, payload = assemble_student_inputs(example, outputs)
student_hidden = target.decoder_forward(
    student_embeds[None], None,
    lora_name=reader_lora, interventions=[payload],
)
# Student samples use generate_with_interventions with the same payload.
```

`activation/agent_training/agent_training_utils.py · Collated / training_inputs_embeds()`

```python
# Existing collated AC spans carry embeddings AND frozen delta references.
inputs_embeds = training_inputs_embeds(model, batch)
interventions = training_interventions(model, batch)
# Both reference-policy and current-policy forwards receive these values.
```

These are schematic interface excerpts; assembly uses the same span transforms rather than duplicating them. Update `agent_trainer.py` reference/train/eval calls and `ac_model_study.py` serving calls. Padded/packed collation maps all deltas once with existing segment offsets and emits one payload per physical row (merging logical examples when packed) and preserves Qwen3.5 GDN/full-attention masks. Spacer/padding rows have no intervention.

Optimizer groups include every new head/scale with stable names and FP32 storage; preserve explicit reset/resume and no duplicate shared-base parameters. Record head/scale/encoder/reader gradient norms, delta RMS relative to calibration, scale trajectories, payload bytes and encode/forward latency in the existing AC reporter. Do not silently reset a trained input head or adopt experimental optimizer knobs as universal defaults.

</details>

<details class="card" data-tressoir-markdown>
<summary><span class="card-title">M6 — Prove replay, gradients and actual content use</span><span class="card-oneliner">Correctness first; quality measured against a matched input-only baseline.</span><span class="card-badge">Planning</span></summary>

#### Planning Overview

A working API is insufficient: the tests must expose forgotten payloads, incorrect checkpoint gradients, cache collisions and a reader that ignores the content. Detailed probe code stays in IB. Reuse focused vLLM tests and the existing co-design mainline examples; broad test scaffolding and long campaigns are not implicit deliverables.

#### Planned Changes

| Gate | Required comparison |
|---|---|
| Config and migration | Zero mode, shallow topology, snap ties, deduplication, old checkpoint/input-only behavior |
| IPC and records | Nested tensor/int-key roundtrip; serialization/load/resume byte equality; corrupt/missing files |
| HF derivatives | Direct vs checkpointed, two different pending payloads/adapters, full and linear blocks, every head's gradients |
| Packed/padded causality | Individual vs unequal batches, segment isolation, no effects on preceding tokens; two examples packed at offsets [0,8] map positions [2] and [1] to one payload [2,9] |
| HF generation | Cached vs no-cache; no reinjection into already-cached prompt states |
| vLLM execution | HF/reference logits vs vLLM with matching model/adapter/inputs; mixed requests, chunk cuts, actual preemption/recompute and cancellation |
| Cache and FP8 | Identical-payload hits, changed payload/layer/position/LoRA/mixed-mask misses, append-only earlier-prefix reuse, real hybrid block geometry |
| Packaging | Clean environment imports the pinned fork, not a stock 0.28.0 wheel |
| Learning | Input-only vs deep on clean held data, correct/shuffled/absent content, matched recipe and initial shared weights |

Declare numeric tolerances by dtype and baseline kernel differences before assertions. FP8 is compared with its corresponding cache baseline, not a claim of bitwise BF16/HF equality. Test the complete release envelope once; later changes rerun affected gates.

A quality comparison freezes a source-verified recipe for both arms, uses the same memory positions, initialization, data, schedule and updates, and reports payload/memory/parameter costs. Default deep payload is up to 6× input-only bytes. No-context controls remove all deltas as well as embeddings; shuffled controls swap complete bundles only between matching row counts, target widths and layer sets, keeping recipient positions. Report coverage; any different-length control must explicitly rerender and realign, never silently truncate or pad donor information. Keep teacher KL alongside fixed-base/no-context gold NLL so reader drift cannot masquerade as compression. A long experiment budget is separate from this plan's mechanics checks.

</details>

## What the ongoing POCs change about this plan

| Observed evidence | Design consequence |
|---|---|
| [RAG ablation](../AGENT_AC_PRETRAINING/rag_ac_poc/ABLATION_RESULTS.md): held AC .584 versus no-context .586 gold NLL | Require content-removal/shuffle controls; falling training loss is not enough |
| [Reconstruction ledger](../AGENT_AC_PRETRAINING/ac_recon/LEDGER.md): combined skip+gold recipe learns sooner, including a trainable 1/4 reader | Preserve the useful input channel; don't claim trainable readers inherently fail at that ratio or skip alone explains the gain |
| Scale learning and gate-LR results vary by recipe | Small residual-relative initialization and diagnostics; no assumed universal fast scale LR |
| MD warmup50+cosine .800 vs matched MK 1.581 after 512 updates | Match schedules and allow enough learning time; warmup and decay effects are still confounded |
| Held tool-window near-duplicates were measured | Fixed clean splits and explicitly labeled clean results |
| Optimizer resume has been tested on the POC | Extend state/provenance to every new head/scale; repeat the proof for interventions |

These observations were frozen through the ledger's 13:38 UTC entry. Several runs were still active; they are not final universal recipe conclusions. Recover an exact POC source snapshot before claiming to reproduce its recipe. More recent findings can inform baseline selection without silently altering the agreed carrier architecture.

## Review and validation status

**Both independent reviews pass, with no unresolved findings:** [correctness](CORRECTNESS_REVIEW.md) and [training friendliness](TRAINING_REVIEW.md). The revised plan fixes mixed token/embed cache identity and packed-batch coordinates, and clarifies calibration ownership, shape-compatible shuffle controls and the offline submission path. See the [finding disposition and validation record](REVIEW.md).

Completed checks: fork ownership/branch push, exact v0.28.0 pin, local-code/submodule commits, co-design patch applicability, source/projection agreement and artifact validation. The [source manifest](SOURCE_MANIFEST.json) records inspected baselines. No intervention implementation or GPU/training validation has run in this planning turn. The next decision is approval of M0–M6 above.
