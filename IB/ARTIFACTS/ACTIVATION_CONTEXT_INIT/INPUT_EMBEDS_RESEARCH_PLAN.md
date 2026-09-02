# Message schemas and vLLM input embeddings — research plan

Status: complete. Full bounded investigation completed on 2026-08-30; no product code was changed.

Final report: `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/INPUT_EMBEDS_REPORT.tressoir.html`. The live and local evidence is copied alongside it; the experiment cluster was torn down and final SkyPilot status was empty.

## Objective

Produce a source-backed and experimentally verified answer to two related questions:

1. For each chat-model family in project scope, does the model/template/vLLM path accept OpenAI-style typed text content, or must `message["content"]` be a direct string? If typed content fails, identify exactly which layer rejects it and whether vLLM surfaces the error without our adapter.
2. How does pinned vLLM 0.28.0 accept input embeddings, especially when only the user text should be supplied as embeddings inside a chat prompt? Define the role of chat templates, special tokens, token IDs or “equivalent text,” device placement, dtype/shape constraints, and computational cost.

The final deliverable will be `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/INPUT_EMBEDS_REPORT.tressoir.html`. It will distinguish documented behavior, pinned-source behavior, model-template behavior, live-validated behavior, and inference.

## Current baseline

- The project pins vLLM 0.28.0 and Transformers 5.15.1.
- `LoadedEngineModel.chat()` deep-copies messages, calls `adapt_message_format()`, then calls `vllm.LLM.chat()`.
- `adapt_message_format()` currently leaves multimodal-model messages unchanged and converts one typed text part to a direct string for non-multimodal models.
- Because adaptation occurs before vLLM, the current successful Qwen test does not answer whether vLLM itself would reject the typed representation.
- The engine module states an intent to support both ordinary completion-style messages and messages with input embeddings, but no embedding ingress exists yet.
- Target families named by current source are Qwen 3/3.5, Gemma 3/4, Nemotron 3, and Qwen 3 embedding models. Current concrete chat fixtures are:
  - `Qwen/Qwen3.5-0.8B`
  - `Qwen/Qwen3-1.7B`
  - `google/gemma-3-1b-it`
  - `google/gemma-4-E2B-it`
  - the official 4B Nemotron 3 representative selected during the source audit
- Qwen embedding models are not chat-generation targets. They are in scope only to prevent confusion between sentence/readout embeddings and token-level input embeddings consumed by a causal LM.

## Key distinctions the research must preserve

### Message schema versus chat template

A typed message may be accepted by vLLM’s schema layer yet rejected by the model’s Jinja chat template, tokenizer/processor normalization, or downstream prompt parser. Conversely, vLLM may normalize typed text before the template sees it. The report must name the actual boundary instead of attributing every failure to “the model.”

### Full prompt embeddings versus a user-content span

There are at least two materially different experiments:

- **Full-prompt embeddings:** render the entire chat prompt, including system/user delimiters and the assistant generation prefix, then provide one embedding row per resulting prompt position.
- **Hybrid prompt:** retain ordinary chat/template structure while replacing only the `Say: 'Hello, World!'` span with embeddings.

The plan must establish whether vLLM natively supports the hybrid form, requires callers to assemble a complete embedding sequence, or expects a placeholder/equivalent-text mechanism through multimodal input plumbing.

### Token embeddings versus semantic embeddings

A causal model’s input expects vectors in its own input-embedding space, with matching hidden width, dtype, order, and positional length. A pooled sentence embedding from Qwen Embedding or another model may not be interchangeable. The report must prove or disprove each proposed path and avoid calling unrelated vector representations equivalent.

### Special-token ownership

The experiment must identify exactly one owner for:

- system/user/assistant control tokens;
- beginning/end-of-sequence or turn delimiters;
- the assistant generation prefix;
- tokenization of the user string;
- positional length represented by embedding rows.

Every working example must show whether `apply_chat_template(tokenize=True)` is used directly or whether rendered text is tokenized with special-token insertion disabled to avoid duplication.

## Target compatibility matrix

The investigation will cover behavior classes and concrete representatives:

| Family | Typed-content check | Input-embedding relevance | Planned representative |
| --- | --- | --- | --- |
| Qwen 3.5 | required | primary live anchor | `Qwen/Qwen3.5-0.8B` |
| Qwen 3 | required | compare template/API behavior | `Qwen/Qwen3-1.7B` |
| Gemma 3 | required | text/multimodal processor boundary | `google/gemma-3-1b-it` |
| Gemma 4 | required | text/multimodal processor boundary | `google/gemma-4-E2B-it` |
| Nemotron 3 | required | compare non-Qwen/Gemma template behavior | smallest practical supported fixture |
| Qwen 3 Embedding | not a chat target | distinguish pooled output from LM token embeddings | current 0.6B/4B fixtures as needed |

If a model is gated, unavailable, too large for the bounded live run, or unsupported by pinned dependencies, the report will mark that limitation explicitly and use tokenizer/config/source evidence rather than silently substituting another family.

## Evidence policy

- Use official vLLM 0.28 documentation and the installed/pinned vLLM 0.28 source as the API authority.
- Use official Transformers 5.15 documentation/source and each model repository’s tokenizer, processor, config, and chat-template files as model-format authority.
- Record exact versions, public types, shapes, and exception provenance.
- Keep exploratory scripts, logs, token dumps, and benchmark output under `IB/TMP/INPUT_EMBEDS_RESEARCH/`.
- Do not edit engine/harness product code during research.
- Do not promote a workaround as a recommendation until it survives a live vLLM probe.
- Label measured cost separately from reasoned cost.

## Milestone 1 — Message-format compatibility

Status: Complete.

### Work

1. Trace pinned `LLM.chat()` from public type hints through message parsing, chat-template application, and prompt construction.
2. For every target representative, inspect tokenizer/processor chat-template expectations.
3. Exercise the smallest non-generation path possible with both forms:
   - `{"role": "user", "content": "Say: ..."}`
   - `{"role": "user", "content": [{"type": "text", "text": "Say: ..."}]}`
4. Capture:
   - accepted/rejected;
   - normalized representation;
   - rendered prompt;
   - token IDs;
   - exception class/message and rejecting layer.
5. Repeat with system plus user messages, because some templates treat system content differently.
6. Determine whether current `is_multimodal` routing is a sufficient proxy for message-shape support or whether message-format capability needs its own model metadata.

### Planned experiment surface

`IB/TMP/INPUT_EMBEDS_RESEARCH/message_matrix.py`

```python
for model in target_models:
    for shape in ("content-string", "typed-text-part"):
        result = trace_chat_preprocessing(
            model=model,
            messages=fixture(shape),
            add_generation_prompt=True,
        )
        record(model, shape, result.stage, result.rendered, result.token_ids)
```

### Exit criteria

A per-family matrix states whether adaptation is necessary, whether vLLM throws without it, and which component owns the restriction.

## Milestone 2 — Pinned vLLM embedding contract

Status: Complete.

### Work

1. Trace all public/offline vLLM 0.28 prompt types and entry points involving `prompt_embeds`, `inputs_embeds`, multimodal placeholders, or equivalent carriers.
2. Determine for each viable path:
   - whether it is accepted by `LLM.chat()`, `LLM.generate()`, or only a lower-level API;
   - required tensor rank, batch representation, hidden width, dtype, device, and length;
   - whether prompt text or prompt token IDs must accompany embeddings;
   - what “equivalent text” or placeholder data does operationally;
   - how positions, prefix caching, scheduling, LoRA, and sampling interact;
   - whether arbitrary hybrid text/embedding spans are supported.
3. Trace where embedding tensors are copied, cast, or consumed and whether vLLM can derive embeddings from token IDs internally without exposing/duplicating the model.
4. Identify any experimental flags or known limitations in the pinned version.

### Planned evidence artifact

`IB/TMP/INPUT_EMBEDS_RESEARCH/vllm_contract.json`

```json
{
  "version": "0.28.0",
  "entry_points": [],
  "accepted_prompt_types": [],
  "required_companion_fields": [],
  "shape_dtype_device_contract": {},
  "hybrid_span_support": null,
  "limitations": []
}
```

### Exit criteria

The report can show one exact public call shape for every supported mechanism and explicitly mark unsupported mechanisms.

## Milestone 3 — Controlled embedding experiments

Status: Complete.

### Baselines

For `Qwen/Qwen3.5-0.8B`:

1. Render the canonical system/user chat prompt with the assistant generation prefix.
2. Record the canonical prompt token IDs and special-token positions.
3. Run deterministic generation through the ordinary token path.
4. Obtain the model-space token embedding sequence using the least duplicative supported mechanism discovered in Milestone 2.
5. Run the same prompt through vLLM’s embedding path.
6. Compare:
   - accepted input and output;
   - prompt length/accounting;
   - deterministic generated text or token prefix;
   - errors/limitations;
   - memory and latency.

### Hybrid user-span experiment

Test the exact user text `Say: 'Hello, World!'` as embeddings while preserving the system/user/assistant template structure:

1. Locate the user-content token span in the canonical token sequence.
2. Establish whether vLLM accepts an explicit hybrid structure.
3. If it only accepts a full embedding tensor, assemble the complete prompt embedding sequence while replacing only that span’s rows.
4. Test whether any companion token IDs/text are semantic inputs, bookkeeping, cache keys, placeholder alignment, or unnecessary.
5. Verify that special/control tokens appear exactly once.
6. Run negative controls for wrong width, wrong length, wrong dtype/device, pooled sentence embeddings, and double-added special tokens.

### Cross-family validation

- Run preprocessing/template probes for every family.
- Consolidate live checks into the fewest runs possible: use Qwen 3.5 as the embedding anchor and add another family only when source/preprocessing evidence reveals a distinct runtime path.
- Do not download multiple large models merely to repeat identical code paths; document the evidence used to group behavior classes.

### Performance protocol

Measure separately:

- CPU tokenization/chat-template time;
- CPU embedding lookup, if supported and meaningful;
- GPU embedding lookup;
- host-to-device transfer for externally constructed embeddings;
- vLLM prefill and generation;
- peak GPU memory;
- incremental memory from any duplicate Transformers model required only to construct embeddings.

Use warmup, CUDA synchronization, repeated short and longer prompts, and label every table cell as measured or inferred.

### Cloud lifecycle

- Reuse the approved SkyPilot wrapper and baked image.
- Prefer one fresh L40S and reuse it for all probes; burst only when it materially shortens the blocked investigation.
- Keep live logs attached.
- Write raw artifacts under `~/activation_artifacts/input-embeds-research` and download them into `IB/TMP`.
- Teardown any cluster created specifically for this investigation after artifacts are retrieved.
- If a stopped cluster cannot restart in its zone, report the capacity boundary rather than silently deleting retained state.

### Exit criteria

At least one real full-prompt embedding call and one real user-span result are demonstrated, or the report contains pinned-source plus runtime evidence that the requested form is unsupported.

## Milestone 4 — Tressoir HTML report

Status: Complete.

### Deliverable

Create `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/INPUT_EMBEDS_REPORT.tressoir.html` using the Tressoir HTML skill after experiments complete.

The report will include:

- a concise direct answer to every user question;
- target-model compatibility matrix;
- typed-message error-provenance matrix;
- vLLM embedding API/type/shape table;
- visual pipeline from typed message to chat template, token IDs, embedding rows, prefill, and generation;
- a special-token/span alignment view for `Say: 'Hello, World!'`;
- working minimal code for supported paths;
- rejected/non-equivalent approaches and why;
- CPU/GPU compute, transfer, memory, and latency findings;
- implications for current `Engine.chat()` and a recommended future engine API;
- source links and version scope;
- clear labels for measured, documented, inferred, and unverified conclusions.

No product patch will be included unless the user separately requests implementation after reviewing the report.

## Non-goals

- Implementing the final engine input-embedding API in this pass.
- Designing concurrency or request batching.
- LoRA implementation beyond documenting whether it constrains the embedding path.
- Treating pooled semantic embeddings as token embeddings without proof.
- Exhaustively benchmarking every model size.
- Changing current message adaptation before compatibility evidence exists.

## Original requested decision

Approve either:

1. **Full bounded investigation (recommended):** pinned source/docs, all-family preprocessing matrix, one bounded L40S experiment per distinct behavior class, performance measurements, teardown, and the HTML report.
2. **Source/local investigation only:** complete API/template analysis and local preprocessing matrix, but mark vLLM GPU behavior and costs unverified.
## Accepted execution decision

- Run the full bounded investigation.
- Omit Qwen 3.6; Qwen 3.5 is sufficient for that branch.
- Use the 4B Nemotron 3 representative.
- Consolidate answers into very few runs that reuse one setup cluster.
- Prioritize speed; launch a fresh cluster rather than waiting on an existing stopped cluster.
- Temporary burst capacity is allowed, but every experiment-specific cluster must be torn down.

