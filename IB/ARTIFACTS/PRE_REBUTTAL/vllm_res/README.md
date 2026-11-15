# Hidden Residuals in vLLM 0.28 (Qwen3.5-9B reader) — 2026-11-15

AC models whose reader receives per-layer residual deltas ("deep AC inputs", the `res` write mode) can now be served by vLLM
in agent rollouts. No vLLM fork: an out-of-tree model class registered through a vLLM general-plugin entry point.

## How it works

- **Packing.** Each AC row travels as one prompt-embedding row `[row, Δ_L1, …, Δ_Lk]` ((1 + k)·d wide, layers ascending);
  token positions get `[token embedding, 0, …, 0]` (the model pads its own token embeddings). For the 9B reader with deltas at
  every full-attention layer (3, 7, …, 31): k = 8, width 9 × 4096 = 36,864.
- **Model.** `activation/vllm_residuals/qwen3_5.py`: `Qwen3_5ResForConditionalGeneration` = upstream Qwen3.5 with (a) token
  embeddings zero-padded to the wide row and (b) a decoder forward that splits the row and adds each Δ to the input of its layer.
  vLLM passes `(hidden_states, residual)` and fuses the add into the next norm, so HF's "layer input" is `hidden_states + residual`;
  the Δ is added onto `residual` (onto `hidden_states` at layer 0). A zero Δ is an exact no-op, so decode steps (zero deltas) and
  CUDA graphs are unaffected. The decoder keeps upstream's `support_torch_compile`.
- **Engine.** `activation.vllm_residuals.engine_overrides(d, layers)` is the engine's `hf_overrides`: the architecture name plus
  `text_config.projection_size = (1 + k)·d` (vLLM sizes `GPUModelRunner.inputs_embeds` with `get_inputs_embeds_size()`, which reads
  it) and `text_config.ac_residual_layers`. The in-process `EmbedsPrompt` path has no width check (only the serialized/API path in
  `renderers/embed_utils.py` does), so nothing in vLLM had to be relaxed.
- **Registration.** `pyproject.toml` entry point `vllm.general_plugins: ac_residuals = activation.vllm_residuals:register`, loaded by
  vLLM in every process (front end, engine core, workers). It also maps the new architecture to Qwen3.5's config hook (mamba state
  dtype from the checkpoint).
- **Prefix cache.** vLLM hashes every block's prompt-embedding bytes, so the deltas are part of the cache key with no change.

## How a rollout enables it

Nothing new to set: register the deep AC model on the reader as today (`intervention_frequency` / `intervention_layers`).
`LoadedModel.engine_to_device` adds the `hf_overrides` when an AC model with intervention layers targets that loaded model;
`ActivationContextModel.encode_async` then resolves to the whole `ACOutput` for a deep model, and `Agent._encode_parts` packs rows and
deltas (`pack_rows`); `Agent.prepare_request` builds the request at the packed width. Rows-only AC models are untouched.
The environment needs the project re-installed once so the entry point exists (`uv sync`; the image's `uv sync --frozen` in
`~/sky_workdir` does it). RTX PRO 6000: bf16 KV cache (`kv_cache_dtype="auto"`).

## Validation (one RTX PRO 6000, `scripts/`, outputs in `results/`)

Pair `T_res_rq9fr_kI_KL` final (frozen reader: reader LoRA max |B| = 0.0, so vLLM serves the untuned Qwen/Qwen3.5-9B; encoder
Qwen/Qwen3.5-2B). 16 dev1000 questions built exactly as `sec45_eval.py` (k = 20 ×5, 100 ×6, 200 ×5; prompts 284–2,857 tokens with
191–2,731 rows). HF reference = `install_intervention_hooks` via `generate_with_interventions`, one item at a time. vLLM: one request
at a time, prefix cache reset before each, greedy, up to 64 new tokens; log-prob differences over vLLM's top-20 tokens at the first
generated position.

| Comparison | max \|Δ logprob\| | mean | top-1 agree | greedy exact |
|---|---|---|---|---|
| vLLM residual model (compiled) vs HF hooks, rows + deltas | 0.495 | 0.071 | 16/16 | 13/16 |
| vLLM residual model (eager) vs HF hooks, rows + deltas | 0.370 | 0.068 | 16/16 | 15/16 |
| upstream vLLM (compiled) vs HF, rows only (the existing engine gap) | 1.431 | 0.109 | 16/16 | 10/16 |
| upstream vLLM (eager) vs HF, rows only | 0.602 | 0.103 | 16/16 | 9/16 |
| scale: HF with vs without deltas (top-20 shift per item) | 2.99–8.26 | — | — | 0/16 same |

Zero-delta pack vs upstream rows-only: **eager 16/16 bit-exact** (max diff 0.0, greedy 16/16). Compiled engines: 0/16 bit-exact
(max 0.997, greedy 11/16) — the same size as compiling the unchanged upstream model (upstream compiled vs eager: 0/16 bit-exact,
max 0.758, greedy 11/16): the extra adds change Inductor's fusion (the compiled engines run the native RMSNorm), not the values.

Prefix cache (k = 200 item: 2,857 tokens, 2,731 rows, first row at position 66; block 528, mamba cache mode `align`), cached tokens:
cold 0 · repeat 2,640 · all deltas ×1.01 → 0 · deltas changed from position 2,114 → 0 · original again 2,640 · deltas changed only
after the 2,640 boundary → 2,640 · deltas changed only inside block [2,112, 2,640) → 2,112 · rows changed, deltas kept → 0.
(Hybrid Qwen3.5 can only hit where a recurrent-state snapshot exists, so a hit is the stored boundary or an earlier stored one.)

Throughput: 32 concurrent requests (`LLM.generate`), each 4,126 tokens with 4,000 distinct rows, compiled engine, prefix cache reset:

| | rows only (upstream) | rows + deltas (residual) |
|---|---|---|
| payload per request | 33,800,192 B (33.8 MB) | 304,201,728 B (304.2 MB) |
| prefill only (max_tokens 1) | 5.80 s, 22,781 prompt tok/s | 6.32 s, 20,880 prompt tok/s |
| end to end (256 tokens, ignore_eos) | 10.37 s, 790 gen tok/s | 11.31 s, 724 gen tok/s |
| CPU SHA-256 of one payload | 0.018 s | 0.160 s |
| CPU copy of one payload's frames | 0.004 s | 0.144 s |

Harness path (`scripts/agent_smoke.py`, `results/agent_smoke.txt`): `HarnessRuntime` + `Agent` with the pair; the reader engine built
by `LoadedModel.engine_to_device` (LoRA slots on, sleep mode on) got the overrides and the residual architecture
(inputs_embeds_size 36,864); one part of a k = 200 question's passages: prompt 3,208 tokens, 1,977 rows, request tensor
(3,208, 36,864) bf16 = 225.6 MiB; submitted twice: cached 0 → 3,168, identical reply. (The reply was wrong for that question: the
smoke's prompt layout is not the trained question-first layout; it checks mechanics, not accuracy.) A second run gave the same
output and left the GPU at 0 MiB after exit.

## Limits and risks

- **Host memory and CPU scale with the 9× payload.** Every turn re-sends the full-length tensor (zeros at token positions included):
  304 MB per 4,000-row request, held by the agent, the front end and the engine core. 32 concurrent such agents ≈ 10 GB per copy;
  128 would not fit a 62 GB g7e.2xlarge. The engine core SHA-256-hashes all bytes of every request (≈ 0.16 s per 4,000-row request).
  Cutting this needs vLLM to accept narrow rows at token positions or sparse deltas (not done).
- Compiled residual engines are not bit-identical to upstream even with zero deltas (compile-level noise, see above).
- Qwen3.5 dense reader only, pipeline parallel 1. A trained reader LoRA together with deltas is untested in vLLM (the validated
  pair's reader LoRA is zero; the smoke engine had LoRA slots but sent no adapter).
- Recorded trajectories of deep-model agents keep the packed rows (`ac_spans`, exact resume; 9× the bytes). Consumers that feed
  recorded rows to an HF reader would have to unpack them first (none was exercised here).
- The GPU runner's prompt-embedding buffer grows with the width: max_num_batched_tokens × (1 + k)·d × 2 B (16,384 × 36,864 × 2 B
  = 1.2 GB on the GPU plus the same pinned on the host, computed from the config, not measured separately).
- The first engine start with the new architecture compiles from scratch (load 92 s here vs 20 s for a warm upstream engine).
