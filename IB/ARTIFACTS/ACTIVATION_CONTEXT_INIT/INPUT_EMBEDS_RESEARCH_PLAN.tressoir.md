# Message schemas and vLLM input embeddings — experiment plan

This plan will determine which target chat models require direct string content, how pinned vLLM handles typed message parts, and how the exact `Say: 'Hello, World!'` span can—or cannot—enter generation as embeddings.

> **Completed 2026-08-30.** The live-validated report is [`INPUT_EMBEDS_REPORT.tressoir.html`](./INPUT_EMBEDS_REPORT.tressoir.html). Qwen 3.5 direct text, typed text, full-prompt embeddings, and a seven-row hybrid user span produced identical greedy output; the experiment cluster was torn down.

## Executive Summary

The investigation separates three layers that otherwise produce misleading answers:

| Layer | Question | Evidence |
| --- | --- | --- |
| Message schema | Does typed `content: [{type: "text", ...}]` survive vLLM parsing? | pinned vLLM types/source and exact exceptions |
| Chat template | Does this model’s tokenizer/processor accept and render that representation? | per-family template probes and token IDs |
| Embedding ingress | Can vLLM consume full or partial prompt embeddings, and what companion text/IDs are required? | pinned API/source plus live GPU calls |

Current code converts typed text parts to strings for non-multimodal models before `LLM.chat()`. The successful Qwen test therefore proves our adaptation works; it does **not** prove vLLM would throw without it. The first milestone will isolate that error boundary for Qwen 3/3.5, Gemma 3/4, and Nemotron 3.

The embedding work will distinguish full-prompt embeddings from replacing only the user-content span. It will also distinguish a causal model’s token-level input vectors from pooled semantic embeddings produced by an embedding model.

`InputEmbeddingFlow()`

```text
typed messages
    │
    ▼
model chat template ──► canonical prompt token IDs
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
          ordinary token path       model-space embedding rows
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                       vLLM prefill/generation
```

Every experiment will give one component ownership of system/user/assistant delimiters, end-of-turn tokens, and the assistant generation prefix. This avoids the common mistake of applying a chat template and then adding special tokens a second time.

The final deliverable will be `INPUT_EMBEDS_REPORT.tressoir.html` in this folder. It will contain the compatibility matrix, working calls, span alignment, performance findings, and a recommendation for the future engine API. No product code will change during this research pass.

## Requested Decisions

**Accepted execution scope — 2026-08-30**

- Full bounded investigation, including live GPU validation and teardown.
- Omit Qwen 3.6; Qwen 3.5 covers that branch.
- Use the 4B Nemotron 3 representative.
- Consolidate work into very few runs on one reusable cluster.
- Prioritize speed: launch a fresh cluster instead of waiting on a stopped one; temporary burst is allowed if all experiment clusters are torn down.

## Milestones

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M1 — Message compatibility matrix</span>
    <span class="card-oneliner">Identify exactly where typed text content is accepted, normalized, or rejected.</span>
    <span class="card-badge">Complete</span>
  </summary>

#### Planning Overview

Trace `vllm.LLM.chat()` through pinned vLLM 0.28 parsing and each target model’s tokenizer/processor chat template. Probe both direct-string and typed-text-part messages without generation first, recording rendered prompts, token IDs, normalization, and exact exception provenance.

The matrix covers:

- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3-1.7B`
- `google/gemma-3-1b-it`
- `google/gemma-4-E2B-it`
- the smallest practical Nemotron 3 representative

Qwen 3 embedding models are not chat targets; they appear only where needed to contrast pooled embeddings with causal-LM input embeddings.

#### Planned Changes

`IB/TMP/INPUT_EMBEDS_RESEARCH/message_matrix.py`

```python
for model in target_models:
    for content_shape in ("string", "typed-text-part"):
        trace = preprocess_chat(
            model=model,
            messages=fixture(content_shape),
            add_generation_prompt=True,
        )
        record(
            model=model,
            accepted=trace.accepted,
            rejecting_layer=trace.rejecting_layer,
            rendered_prompt=trace.rendered_prompt,
            token_ids=trace.token_ids,
        )
```

The current `adapt_message_format()` remains untouched. The report will answer whether adaptation is actually required, whether vLLM throws without it, and whether multimodality is the right capability boundary.

#### Validation

Repeat each form with both user-only and system-plus-user messages. Mark gated, unavailable, or pinned-version-incompatible models rather than silently replacing them.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M2 — vLLM embedding contract</span>
    <span class="card-oneliner">Resolve public input types, companion fields, shapes, devices, and hybrid-span support.</span>
    <span class="card-badge">Complete</span>
  </summary>

#### Planning Overview

Audit official vLLM 0.28 documentation and installed/pinned source for `prompt_embeds`, `inputs_embeds`, multimodal placeholders, and every accepted offline prompt type.

For each mechanism, answer:

- Is it accepted by `LLM.chat()`, `LLM.generate()`, or only a lower-level API?
- What tensor rank, hidden width, length, dtype, and device are required?
- Must token IDs or text accompany the embeddings?
- If “equivalent text” exists, is it semantic input, position/length bookkeeping, cache identity, or placeholder alignment?
- Does vLLM support mixed text and embedding spans, or must the caller assemble one complete embedding sequence?
- Where are embeddings copied or cast?
- What limitations apply to LoRA, prefix caching, scheduling, and multimodal inputs?

#### Planned Changes

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

The source audit will use only official documentation, pinned package source, and official model artifacts as authorities.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M3 — Controlled token-versus-embedding experiments</span>
    <span class="card-oneliner">Run the same chat through ordinary tokens, full prompt embeddings, and a user-span replacement in one consolidated probe.</span>
    <span class="card-badge">Complete</span>
  </summary>

#### Planning Overview

Use `Qwen/Qwen3.5-0.8B` as the primary live anchor. First render one canonical system/user prompt with `add_generation_prompt=True` and record every special/control token. Then run:

1. deterministic ordinary-token generation;
2. the same complete prompt as model-space input embeddings;
3. `Say: 'Hello, World!'` as the replaced embedding span while retaining the surrounding chat structure.

If vLLM accepts only a full embedding tensor, the hybrid experiment will assemble the complete prompt sequence and replace only the user span’s rows. If the API provides a real placeholder/equivalent-text mechanism, the experiment will prove what the placeholder controls.

Negative controls will intentionally supply:

- the wrong hidden width;
- mismatched row/token length;
- wrong dtype or device;
- pooled Qwen sentence embeddings;
- special tokens added twice.

#### Planned Changes

`IB/TMP/INPUT_EMBEDS_RESEARCH/embed_probe.py`

```python
prompt_ids = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)
baseline = generate_from_token_ids(prompt_ids)

prompt_embeds = obtain_model_space_embeddings(prompt_ids)
embedded = generate_from_prompt_embeddings(
    prompt_embeds=prompt_embeds,
    companion_ids_or_text=contract_if_required,
)

user_span = locate_exact_user_span(prompt_ids, "Say: 'Hello, World!'")
hybrid = replace_span_and_generate(prompt_ids, user_span)
```

#### Performance protocol

Measure and separate:

| Stage | CPU/GPU question |
| --- | --- |
| chat template and tokenization | CPU time |
| embedding lookup | CPU versus GPU viability and latency |
| externally built tensor transfer | host-to-device time and bytes |
| vLLM prefill | GPU latency |
| generation | GPU latency |
| duplicate model, if required | incremental CPU/GPU memory |

Use warmup, CUDA synchronization, repeated short and longer prompts, and labels that distinguish measured values from inference.

#### Cloud boundary

Use the existing SkyPilot wrapper and prefer one fresh L40S reused for all probes. Keep logs attached, store raw results under `~/activation_artifacts/input-embeds-research`, download them into `IB/TMP`, and tear down any cluster created specifically for this work.

A stopped cluster cannot relocate its disk to another availability zone. Capacity failure will be reported rather than resolved by silently deleting retained state.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M4 — Interactive HTML report</span>
    <span class="card-oneliner">Turn the source evidence and experiments into direct, navigable answers.</span>
    <span class="card-badge">Complete</span>
  </summary>

#### Planning Overview

After the experiments, create:

[`INPUT_EMBEDS_REPORT.tressoir.html`](./INPUT_EMBEDS_REPORT.tressoir.html)

The report will include:

- a direct answer to every question in the request;
- target-family message compatibility and error-provenance matrices;
- a vLLM API/type/shape table;
- a visual message → template → token IDs → embedding rows → prefill pipeline;
- a row-by-row special-token and user-span alignment for `Say: 'Hello, World!'`;
- minimal working code for supported paths;
- rejected or non-equivalent approaches;
- CPU/GPU time, transfer, and memory results;
- implications and a recommended future `Engine` API;
- source links, pinned-version scope, and evidence labels.

#### Planned Changes

`INPUT_EMBEDS_REPORT.tressoir.html · report structure`

```diff-html
+<section id="direct-answers">...</section>
+<section id="model-matrix">...</section>
+<section id="embedding-contract">...</section>
+<section id="special-token-alignment">...</section>
+<section id="performance">...</section>
+<section id="engine-recommendation">...</section>
```

No engine/harness product patch will be made unless separately requested after report review.

</details>

## Acceptance Criteria

- Every target chat family has a typed-content verdict and named rejecting/normalizing layer.
- The report states whether vLLM itself throws when current adaptation is bypassed.
- The pinned vLLM embedding call contract is shown exactly, including companion text/token requirements.
- Full-prompt and user-span embedding cases are separately answered.
- Special/control tokens have one explicit owner and are never double-added.
- Token-level model embeddings are clearly separated from pooled semantic embeddings.
- CPU/GPU work, tensor transfer, latency, and duplicate-memory costs are measured where approved and otherwise labeled as inferred.
- At least one real embedding-path generation succeeds, or runtime evidence demonstrates that the requested form is unsupported.
- Any experiment-specific cloud resources are torn down after artifacts are retrieved.

