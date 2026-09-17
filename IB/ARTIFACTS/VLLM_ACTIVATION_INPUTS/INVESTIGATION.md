# Experimental KV and layer activation inputs for vLLM

Historical investigation. The [reviewed MVP plan](PLAN.tressoir.md) supersedes its proposed implementation scope and design choices.

Both carriers are feasible additions to our existing mixed token/embedding path. The first experiment should establish their meaning with normal scheduling; skipping target computation is a separate optimization. This investigation uses the project's pinned vLLM 0.28.0 source and a tiny CPU semantic probe, as of 2026-09-14. No vLLM fork or GPU implementation has been made.

## What each carrier changes

| Carrier | Side model supplies | Target reads it at | First implementation |
|---|---|---|---|
| Existing input embeds | One vector per memory position | Input to decoder layer 0 | Already integrated |
| `type: "kvs"` | K and V for each selected attention layer and memory position | Attention, bypassing the target's K/V projection for those values | Replace computed K/V before the normal attention/cache-write path |
| `layer_inputs: dict[int, Tensor]` | An activation per selected layer and position | Incoming residual stream, before that layer's normalization | Add or replace rows at an explicit layer boundary |

These are target-space representations. The side model may have a different architecture, but it needs trained heads that emit the target's layer widths, KV head geometry, scale and basis. Copying the side model's native KVs is not generally meaningful. Selecting a layer selects the injection location: the resulting information can propagate into later layers.

The existing public embedding interface is documented in [vLLM's pinned prompt-embedding guide](https://docs.vllm.ai/en/v0.28.0/features/prompt_embeds/). Our harness already sends full-length embedding tensors, token IDs and a token/embedding mask; the proposed carriers need additional transport and execution support.

## A concrete provisional input contract

Keep the current sequence and embedded-span placement. Use explicit prompt positions alongside sparse per-layer tensors. Layer IDs are zero-based, absolute decoder-block indices. The following is a proposed experiment schema, not an API that vLLM accepts today.

`ExperimentalPrompt()` (proposed construction)

```python
{
    "prompt_token_ids": ...,       # full rendered sequence, including placeholders
    "prompt_embeds": ...,          # optional existing embedding carrier
    "prompt_is_token_ids": ...,
    "activation_spans": [
        {
            "type": "kvs",
            "positions": [120, 121, 122],
            "kvs": {layer: (K, V)},  # each [3, target_num_kv_heads, head_dim]
            "key_space": "post_norm_pre_rope",
        },
        {
            "type": "layer_inputs",
            "positions": [150, 151, 152],
            "layer_inputs": {layer: A},  # each [3, target_hidden_size]
            "operation": "add",         # provisional; "replace" is also feasible
        },
    ],
}
```

The dictionary alone does not specify which token rows a tensor targets. A surrounding span makes the requested `dict[int, activation tensor]` unambiguous and avoids shipping full prompt-sized tensors at every layer. Reject overlapping writes to the same layer/position in the first experiment. Initially use the two carriers separately to avoid an implicit ordering between them.

For KV keys, choose one documented boundary. `post_norm_pre_rope` means the producer emits the target's normalized, unrotated key representation; the target applies its positional rotation once at the actual prompt positions. Values are after the value projection, before cache quantization. For a cache-replay diagnostic, accept an explicitly separate post-RoPE format bound to the original positions. Never silently apply normalization or rotary position encoding twice. Model families without key normalization need their corresponding boundary defined explicitly.

A selected-layer KV overlay still runs the base embedding/hidden path at those positions; omitted layers compute normally. This is a useful way to put new information into only some attention layers. A pure KV memory whose placeholders have no meaningful hidden path requires every relevant attention layer to receive its memory KVs, or a separate per-layer masking design. Zero-valued KVs are not an absent memory: their keys still participate in the softmax denominator.

For layer inputs, the proposed first behavior is addition, `h_l[p] += A_l[p]`, before input normalization. Replacement means `h_l[p] = A_l[p]`. The user preference remains open. Missing layers are untouched; prompt-scoped inputs are not reapplied on each generated token. Recomputed prompt rows must receive the same injection after preemption.

## Direct KVs: separate reading from compute skipping

**First: substitute K/V at ordinary scheduled positions.** Render memory spans where embedded spans sit today. Execute their placeholder rows, replace K/V at selected layers, and let the existing attention call write the replacement values into the paged cache. Ordinary causal attention makes earlier text unable to see a future memory span; later text can read it. This preserves batching and sequence accounting and should need no new attention kernel for the initial Qwen3/FlashAttention path. It does not save the placeholder rows' target forward computation; it tests the representation.

In [Qwen3](../../TMP/INPUT_EMBEDS_RESEARCH/vllm-0.28.0-src/vllm/model_executor/models/qwen3.py), the source order is projection → Q/K normalization → RoPE → `self.attn(q, k, v)`. Substitute at the chosen representation boundary before that attention call. Writing external cache values beforehand while leaving normal K/V writes active would overwrite them. The pinned [FlashAttention backend](../../TMP/INPUT_EMBEDS_RESEARCH/vllm-0.28.0-src/vllm/v1/attention/backends/flash_attn.py) scatters using `slot_mapping`; we should reuse that packing/quantization path rather than expose its physical layout to the producer.

**Second: load a complete leading prefix and skip its forward pass.** vLLM already has external KV transfer machinery; see [disaggregated prefill](https://docs.vllm.ai/en/v0.28.0/features/disagg_prefill/). The local [connector interface](../../TMP/INPUT_EMBEDS_RESEARCH/vllm-0.28.0-src/vllm/distributed/kv_transfer/kv_connector/v1/base.py) reports the largest loadable contiguous prefix, receives allocated blocks, and coordinates loading. An external connector can be loaded through `kv_connector_module_path`, while request metadata can carry an immutable payload handle through `kv_transfer_params`. This makes a prefix-only prototype plausible without a core fork, although its packing and lifecycle still need implementation and GPU verification.

Only count positions as computed when all required state is available. Preserve at least one normal suffix/query token to produce the next-token logits: K/V alone do not supply the final residual activation or logits. If normal text precedes a KV span, the whole earlier prefix must already be computed/available before the prefix count can advance through it. Block alignment, partial blocks, preemption and reload need explicit tests; never replace a missing external payload by silently processing dummy IDs.

**Third: skip interleaved memory spans.** The current scheduler predominantly tracks a contiguous computed prefix. Efficiently skipping holes inside a new prefill requires splitting scheduling steps around spans or separating query positions from cache positions, with corresponding attention metadata changes. Selected-layer-only memory also prevents claiming that an entire token position has been computed. This is the larger fork, and is unnecessary for the first representation experiment.

A separate attention memory bank is another valid design. It needs explicit visibility, position and per-layer memory-length rules, plus backend work. It should be evaluated separately from the inline span contract above.

## Layer inputs: a smaller model change with the same transport work

The insertion point is the decoder loop or layer entry. Define it in terms of the logical residual stream, not a framework variable name. In Qwen3's fused execution, the incoming state is `hidden_states + residual` when a residual exists. Addition can modify one summand before the fused add/norm. Replacement must replace the combined state at the selected positions, for example by setting their hidden rows to the supplied activation and their residual rows to zero. Preserve other rows and the fused representation.

The lower layers still run: their token states are needed by ordinary text queries and by layers without an override. Supplying a later-layer activation does not by itself permit skipping the earlier layers or creating new sequence rows only at that depth.

Python hooks may establish behavior in an eager reference. The maintained fork should expose the input explicitly, with a fixed configured layer set, request-specific masks and stable buffers for compilation and CUDA graphs. A plain Python dictionary of fresh GPU allocations on every forward is not the intended optimized implementation.

## The Qwen3.5 boundary

Our current reader family mixes full attention with gated delta-net linear attention. The pinned [linear-attention implementation](../../TMP/INPUT_EMBEDS_RESEARCH/vllm-0.28.0-src/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py) keeps both convolution state and recurrent state in its linear-attention cache. Those are not ordinary per-token K/V pairs.

- Layer-input injection can affect either kind of layer through its incoming hidden representation, provided the chosen boundary is implemented consistently in the HF training model and vLLM.
- K/V overlays can target the full-attention layers while the recurrent layers execute the specified base placeholder/embedding path. This is a partial intervention, not complete replacement of memory state.
- Skipping a complete hybrid prefix needs convolution and recurrent state as well as attention KVs. A future typed `layer_states` carrier would describe that more honestly than stretching `kvs` to cover everything.

Use a small conventional full-attention Qwen3 target to establish KV correctness first. Use a separate Qwen3.5 experiment to test the current reader's layer-input path. An apparent limitation in one carrier should not be mistaken for a limitation in the other.

## Where the fork would change

All vLLM paths below are relative to the retained [vLLM 0.28.0 source](../../TMP/INPUT_EMBEDS_RESEARCH/vllm-0.28.0-src/). These are inspected extension seams, not implemented diffs.

| Area | Files / symbols | Required work |
|---|---|---|
| Prompt schema and normalization | `inputs/llm.py`, `inputs/engine.py`, `inputs/preprocess.py`; renderer paths if public chat support is added | Accept the carriers, validate layer/position/shape contracts and retain existing embeds |
| Engine transport | `v1/engine/input_processor.py`, `v1/engine/__init__.py::EngineCoreRequest`, `v1/request.py::Request` | Carry tensors or immutable handles into the engine process; preserve replay information |
| Scheduling output | `v1/core/sched/output.py::NewRequestData` | Deliver injection metadata with new/resumed requests |
| Worker batching | `v1/worker/gpu_input_batch.py`, `v1/worker/gpu_model_runner.py` | Map absolute prompt positions to packed scheduled rows; slice chunked prefills; survive swaps/removal/resume; provision buffers |
| Model execution | `model_executor/models/qwen3.py`, `qwen2.py` inherited loop; later `qwen3_5.py` / `qwen3_next.py` | KV substitution and correct residual-boundary injection; explicit model input or forward context |
| Cache identity | `v1/core/kv_cache_utils.py` or existing `cache_salt` | Include payload bytes/identity, positions, layer map, operation and representation format alongside existing target/LoRA identity |
| Prefix optimization | `distributed/kv_transfer/kv_connector/v1/base.py`, external connector implementation; scheduler only if needed | Allocate/load complete state, synchronize readiness, manage failure/replay and cleanup |
| Project integration | `activation/common/ac_parts.py`, `activation/ac_model/`, `activation/agent/agent_utils.py`, `activation/harness/loaded_model.py`, `activation/harness/vllm_wrapper.py` | Encode spans, add output heads and training reader support, carry payloads, version checkpoints and record actual tensors/provenance |

The existing embedding-byte hash does not automatically cover either new carrier. Start with prefix caching disabled for the experimental engine. A digest in the existing request cache salt is a conservative first caching implementation; it invalidates the whole request prefix when any payload changes. Finer per-block hashes can restore reuse of unaffected earlier blocks later.

Keep payloads immutable and alive through request completion and possible recomputation. Process-local global dictionaries alone are insufficient for the engine's worker processes. Begin with CPU tensor transport analogous to embeds; a connector can instead use a payload handle with an explicit lifetime. Add GPU IPC/zero-copy only after measuring transport cost. Initial validation must reject incompatible model geometry, out-of-range positions/layers, bad dtype/shape and unsupported cache formats before a GPU write.

## Training and cost

Train the side heads through a differentiable HF/PyTorch reader that has the same injection locations. Use functional tensor replacement/addition; do not attempt to backpropagate through vLLM's serving cache. The current KL/SFT machinery can supply the objective once the reader forward accepts the new carrier. Frozen reader weights still allow gradients into injected activations. A reader LoRA update changes the target computation and must be reflected in serving/cache identity and the paired training setup.

Capacity comes with bandwidth cost. For M memory positions, L selected conventional attention layers, Hkv KV heads, head dimension d and b bytes/value:

- Embedding payload: `M * D * b` bytes.
- KV payload: `2 * M * L * Hkv * d * b` bytes, when K/V dimensions are equal.
- Layer-input payload: `M * L * D * b` bytes.

Illustratively, D=4096, L=32, Hkv=8 and d=128 makes all-layer KVs 16 times larger than input embeddings at the same position count and dtype. Four residual-injection layers are four times larger. These are arithmetic examples, not measurements for a project checkpoint. Direct KVs do not reduce the final target KV-cache size at equal memory-position count. Compression to fewer positions could do so; avoiding prefill work is a separate benefit of the optimized path.

A single dense producer head for all layers can itself become large. Start with a small selected layer set or factorized/shared trunk with per-layer projections. Compare carriers under both equal position counts and equal payload bytes; otherwise richer carriers receive more capacity unnoticed.

## Experimental sequence and exit criteria

1. **Semantic reference — completed here.** Tiny random causal GQA model with rotary positions: replace an internal span's K/V at all layers, check ordinary-position outputs, check prefix-cache replay, no-op/add/replace behavior and gradients.
2. **First vLLM implementation.** Pin 0.28.0; one full-attention architecture/backend, one GPU, eager execution, BF16 cache, no prefix caching or speculative decoding. Thread sparse carriers through the request path; implement KV overlays and residual injection independently. This is a bounded Python-level fork with shared plumbing, not a CUDA-kernel project initially.
3. **Execution correctness.** Replay target-produced payloads and compare suffix logits against the reference with dtype-appropriate tolerances. Mix requests with and without payloads, change batching order, split a span across prefill chunks, force preemption, resume and cancel requests. Ensure no injection leaks between requests and no payload disappears on resume. Test prompt-end memory with a normal readout suffix. Exclude placeholder IDs from token-history penalties or disable those penalties for the initial parity test.
4. **Learning signal.** Use the same examples and target budget for embeds, selected-layer activations and KVs; record held-out loss, quality, payload bytes, producer latency and target time. The first prototype should answer whether richer carriers help, without claiming prefill savings it does not implement.
5. **Performance only after correctness.** Add prefix loading/skipping, then caching, graph/compile support, FP8 cache handling and distributed ranks as separate checks. Tensor parallelism requires target KV-head sharding/replication; pipeline parallelism requires absolute layer mapping. Treat hybrid complete-state skipping and interleaved-hole scheduling as distinct later work.

Effort assessment: the layer hook and KV substitution are small relative to the request/batch lifecycle work. A narrow fork is a days-to-weeks engineering task depending on parity failures and GPU access; robust multi-backend, hybrid, distributed and arbitrary-span skipping support is a substantially larger project. This is a scope estimate, not a measured schedule.

## Evidence and current limits

[Probe source](../../TMP/VLLM_ACTIVATION_INPUTS/semantics_probe.py) and [results](../../TMP/VLLM_ACTIVATION_INPUTS/semantics_result.json) are disposable experiment evidence. The CPU float64 probe found:

| Check | Maximum absolute error / result |
|---|---:|
| Replace internal-span KVs at all layers; compare nonmemory rows | 0 |
| Full prefix KVs; compare suffix execution | 0 |
| Zero residual addition | 0 |
| Replace with original incoming residual rows | 0 |
| Add into a split residual summand | 2.22e-16 |
| Incorrect replacement that retains the old residual | 2.28, demonstrating the bug |
| Gradient into every supplied K and V tensor | Nonzero at all three layers |

This establishes toy-model semantics only. It does not validate vLLM GPU kernels, BF16/FP8 tolerances, scheduler lifecycle, Qwen3.5 hybrid state or learned task quality. The initial experiment intentionally retains normal token scheduling. No product files, dependency pins or cloud resources were changed. The projected document is checked structurally; custom-editor visual inspection is unavailable in this session.

The next implementation should retain inline spans and sparse layer maps, with explicit add/replace and key-space contracts. Those are proposed experimental choices, not promoted project invariants.
