# Multi-GPU study engine: independent replicas (agent-facing source)

Human projection: `PLAN.tressoir.md` in this folder (complete surface; keep both in agreement).
Interaction file: `PLAN.interactions.json` (no live keys; the first-pass decisions are integrated below).
Evidence: `IB/TMP/MULTI_GPU/` (`dp_probe.py`, `probe_async_*.log`, `moe_probe/*.log`, sky logs).

## Intent

`sky setup --gpu-count N` gives N copies of the study engine on one node, driven from the existing
single process. No scheduler on our side: split a batch by world size, send, reassemble in order.
Same mechanism for dense and MoE. One-line swap at the engine construction site; the rest is config.

## Facts established (2026-09-03)

### vLLM 0.28 behavior (read from `.venv/lib/python3.14/site-packages/vllm`)

- `entrypoints/llm.py:284-296`: `LLM(data_parallel_size>1)` raises for single-process use.
- `v1/engine/core_client.py:134-138`: async engine uses `DPLBAsyncMPClient` (internal LB) unless
  `data_parallel_external_lb`; `get_core_engine_for_request` (line ~1471) honors
  `request.data_parallel_rank`, else least-loaded.
- `v1/engine/core.py:1306-1312`: `if data_parallel and model_config.is_moe: DPEngineCoreProc`
  else "Non-MoE DP ranks are completely independent, so treat like DP=1". `DPEngineCoreProc.__init__`
  asserts `is_moe` (line 1977). `is_moe` = `get_num_experts() > 0` (`config/model.py:2089`), no override.
- DP lockstep: `_has_global_unfinished_reqs` all-reduce every 32 steps (`core.py:2216`), dummy
  batches when a rank idles (`core.py:2172-2183`), DP metadata / padding each step.
- MoE all-to-all kernels only with expert parallel (`fused_moe/config.py:1056`); EP off by default.
- `AsyncLLM.generate(..., data_parallel_rank=None, session_id=None, ...)`; `session_id` belongs to
  resumable streaming sessions (`v1/request.py`, scheduler `_update_request_as_session`), not to routing.
- `AsyncEngineArgs` accepts the study engine kwargs with `data_parallel_size=2` (checked locally).
- Env: `VLLM_DP_*` only for the offline SPMD path; no switch disables the DP sync.

### Probe results (2× RTX PRO 6000, g7e.12xlarge; 512 study prompts of ~880 tokens, 128 label prompts of ~6.8k tokens)

| leg | labels prompt tok/s (s) | study output tok/s (s) | notes |
| --- | --- | --- | --- |
| dense NVFP4+MTP async dp1 s64 | 11,754 (73.9) | 914 (52.0) | baseline |
| dense async dp2 s64 | 23,007 (37.7) | 1,777 (26.8) | 1.98× / 1.99× |
| dense async dp2 s128 | 23,273 (37.3) | 1,818 (26.2) | s128 neutral |
| dense async dp2 s64, 64 prompts | 24,242 | 1,403 | under-filled replicas |
| dense async dp2, 64 prompts pinned rank 0 | — | 853 | steering works |
| MoE `nvidia/Qwen3.6-35B-A3B-NVFP4` async dp1 noMTP s64 | 20,078 (43.2) | 1,227 (46.8) | baseline |
| MoE async dp2 MTP s64 | 30,541 (28.4) | 2,311 (23.5) | MTP taxes prefill, helps decode |
| MoE async dp2 noMTP s64 | 35,620 (24.4) | 1,804 (31.4) | lockstep costs ~half a GPU |
| MoE async dp2 noMTP s128 | 34,331 (25.3) | 2,504 (22.7) | +28% decode |
| **MoE replicas dp2 noMTP s64** | **68,702 (12.6)** | **2,918 (19.5)** | 2× vLLM DP, >2× dp1 |

Engine loads: dense cold with JIT 440 s (2 replicas in parallel via async DP), warm 51 s; MoE cold
506 s, warm ~150 s; replicas mode sequential: replica 0 at 161 s, replica 1 at 207 s.

Earlier single-GPU study test (dense, sync `LLM.chat`, batch 8): ~480 output tok/s study,
11.2k prompt tok/s labels; first study batch carries ~50 s of Triton JIT during inference.

### Derived planning rates

Per hour, 1×/2× RTX PRO 6000: labels (MoE) ~18k/~36k; generation dense+MTP 140-token answers
~24k/~48k; generation MoE+MTP ~45k/~90k (uncalibrated in replicas mode); generate+label with
dense generator ~10k/~21k; 12 h ≈ 125k/250k labeled.

## Decisions

Accepted (from chat, 2026-09-03):

- Independent engines, no scheduler, dense and MoE alike.
- No world-size knob: every visible GPU; restrict with `CUDA_VISIBLE_DEVICES`.
- `ENGINE_MAX_NUM_SEQS = 128` tentatively; M1 gate includes an L40S FP8 startup check.
- Batch constants in the wrapper module: `ENGINE_CONCURRENCY = WORLD_SIZE × 128` (labels),
  `RECOMMENDED_BATCH_SIZE = 4 × ENGINE_CONCURRENCY` (study). Study knobs become optional and
  default to them.
- Labeler: MoE A3B NVFP4 (FP8 without FP4 support), MTP off. Generator keeps MTP on.
- Label top-k 10, pool 32,768 chars (from the labeling round).

Accepted (from the projection's interaction file, 2026-09-03):

- Generator: dense 27B (`unsloth/Qwen3.8-27B-NVFP4`, FP8 `Qwen/Qwen3.8-27B-FP8`), MTP on.
- `VLLMWrapper.recommended_engine_kwargs(model_id) -> dict` (user asked for a static
  "recommended construction kwargs" method). Known ids: the two 27B ids → base formula + MTP(2);
  the two A3B ids → base formula. Base = `max_model_len 20000, limit_mm_per_prompt {image 0,
  video 0}, kv_cache_dtype fp8, gpu_memory_utilization 0.9, enable_lora False`. Unknown id → base
  (no speculative decoding) plus a printed note; never raise. `max_num_seqs` is not part of the
  formula (the constructor's `setdefault` supplies `ENGINE_MAX_NUM_SEQS`).
- Applied automatically: `LoadedModel.engine_to_device` builds `engine_kwargs` as harness defaults
  → `recommended_engine_kwargs(model_id)` → `model_config.engine_kwargs` (explicit wins). Known
  models therefore need no `engine_kwargs` in their `ModelConfig`.
- No `recommended_chat_kwargs`: sampling/structured-output settings are per pass
  (`dataset_study_chat_kwargs`, label pass), not per model; `enable_thinking=False` is already the
  `engine_chat_many` default; Qwen3.6 and Qwen3.8 publish the same non-thinking sampling.
- GPU study test loads both models and asserts the residency sequence from `harness_stats`:
  `model_loading_times` names == `[STUDY, LABEL]`, `model_freeing_times` names == `[STUDY, LABEL]`
  (generator freed by the swap inside the label pass, labeler freed by the test's `finally`).
  Holds because the study test never loads the HF model or embedding layer (bm25 index only).

## Milestones

### M0 — Probes (Completed)

See facts above. Private probe stays in `IB/TMP`; the temporary copy under `activation/bench/` was
removed after each run. Nodes `ac-dp-probe` (eu-north-1) and `ac-moe-probe` (us-east-1) torn down.

### M1 — VLLMWrapper (Review)

Files: new `activation/harness/vllm_wrapper.py`; `loaded_model.py` (import, field type, construct,
shutdown); `vllm_utils.py` (remove `best_effort_shutdown_vllm`); `harness/__init__.py` (exports);
`tests/test_basic_dataset_study.py` (drop the whole `STUDY_ENGINE_KWARGS` block; the wrapper's 27B
formula is that block minus `max_num_seqs`).

Wrapper contract: `VLLMWrapper(**engine_kwargs)`; `.chat(conversations, **chat_kwargs)` same
signature/order as `vllm.LLM.chat`; `.shutdown()`; `@staticmethod recommended_engine_kwargs(model_id)`
per the accepted decision (four known ids, base formula fallback with a print).
`loaded_model.py`: merge order harness defaults → recommendation → `model_config.engine_kwargs`. `engine_kwargs.setdefault("max_num_seqs",
ENGINE_MAX_NUM_SEQS)` with a print on override. Sequential construction with `CUDA_VISIBLE_DEVICES`
set per replica and restored in `finally`. Strided shards; thread per replica (daemon); errors
collected and re-raised after join; empty shard skips the call; `WORLD_SIZE == 1` calls through.

Gate: study test 1× RTX (parity), 2× RTX (≈2× on `study_output_tokens_per_s` and
`study_label_prompt_tokens_per_s`), L40S FP8 27B startup at `max_num_seqs` 128 + one small batch.
Also calibrate MoE+MTP in replicas mode (fills the uncalibrated cell) if the node is up anyway.

### M2 — Label model + derived batch sizes (Review)

`runtime_config.py`: `dataset_study_batch_size: int|None = None`,
`dataset_study_label_batch_size: int|None = None`, `dataset_study_label_model_name: str|None = None`.
`loaded_model.py`: `ensure_engine_loaded()` frees every other loaded model's engine first
("one engine resident at a time" lives in the harness). `dataset_study.py`: batch defaults from the
constants (module-level constant import only; no runtime harness objects), label pass resolves
`label_model_name`. Test: register `LABEL_MODEL_NAME` (`nvidia/Qwen3.6-35B-A3B-NVFP4`, FP8
`Qwen/Qwen3.6-35B-A3B-FP8`) with a bare `ModelConfig` (formula from the wrapper), set
`dataset_study_label_model_name`, drop the explicit batch-size knobs, replace the single-model
`finally` with a loop freeing every loaded model's engine, then assert the load/free name sequences
`[STUDY, LABEL]` / `[STUDY, LABEL]`. Cold fresh-node runtime ≈ 25 min (two JIT compiles).

Cost note: one engine swap per label call (few min warm, ≤10 min cold).

### M3 — Docs and canon (Review)

SkyPilot README (`--gpu-count` shapes: g7e.12xlarge 2×, g7e.24xlarge 4×, g6e.12xlarge 4× L40S;
capacity scarcer); `ROOT_CANON.md` "Engine replica decisions"; `STATE.md` milestone results; retire
the old 1-hour budget table.

### M4 — Session stickiness (TBD)

`chat(conversations, session_ids=None, ...)`; shard by `hash(session_id) % WORLD_SIZE`. Measure
prefill tokens per turn on a synthetic 8-turn rollout with/without ids. Candidate later: vLLM
resumable streaming sessions.

## Implementation report (2026-09-03)

M1–M3 applied in the workspace; cards in `MULTI_GPU_ENGINE_ROUND.tressoir.md`, staged tree refreshed.
Drifts: `vllm_utils.py` deleted rather than left empty (single importer); `VLLMWrapper.chat` splits
by `len(self.replicas)` instead of the module constant; the test's unused `EMBEDDING_MODEL_ID`
untouched. CPU validation passed (import/constants, formula table, shard logic with fake replicas,
bm25 loading tests `2 passed in 20.51s`, private contract check). GPU gates (2026-09-04, both nodes
torn down): 2× RTX cold `1 passed 1574s`; 1× warm `118.8s` (study 510.8 tok/s, labels 30.3k prompt tok/s);
2× warm `206.1s` (537.3 tok/s, 49.5k prompt tok/s: 1.63× labels, study latency-bound at 8 prompts per
replica); L40S FP8 `1 passed 1198s` with 27B FP8 at `max_num_seqs` 128. Residency asserts held everywhere.
MoE+MTP replicas calibration not done (needs the probe, not the test).
After the user's adaptation (qa renames, recommended chat kwargs applied in `engine_chat_many`, required
label-model knob): 2× cold `1 passed 1617.87s`, 1× warm `118.71s`; residency held; logs `final_*.log`.

## Validation plan summary

- CPU: loading tests unaffected (no engine). Private contract check still passes (stubs
  `ensure_engine_loaded`).
- GPU: as per M1/M2 gates on fresh nodes; tear down after.
- Report per milestone: what landed, drifts, focused diffs, validation.
