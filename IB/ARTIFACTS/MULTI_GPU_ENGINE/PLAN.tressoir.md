# Multi-GPU study engine: independent replicas

`sky setup --gpu-count N` should give N copies of the study engine on one node, driven from the
existing single process with no new way to run anything. Two probe rounds on 2× RTX PRO 6000 settled
the mechanism: one independent vLLM engine per GPU, a batch split in N and sent, results reassembled
in order. It scales linearly for the dense 27B and beats vLLM's own data parallel 2× for the
mixture-of-experts labeler. This plan turns that into a `VLLMWrapper` behind a one-line swap, adds a
separate label model, derives batch sizes from the node, and reserves a later milestone for session
stickiness in multi-step rollouts. Both decisions are accepted and recorded below; the plan is ready to implement.

## Executive Summary

### Goal

On a fresh node with N GPUs, `synthesize_study_examples` and `label_study_examples` run N× faster
with nothing configured beyond `--gpu-count N`, and the same code path runs unchanged on one GPU.
Expected overnight yield on 2× RTX PRO 6000: about 250k labeled questions (dense generator, MoE
labeler), against 50k today.

### Approach

| Piece | What lands | Where |
| --- | --- | --- |
| Replicas | `VLLMWrapper`: one `vllm.LLM` per visible GPU, strided batch split, thread per replica, in-order results, `chat` + `shutdown` surface | new `activation/harness/vllm_wrapper.py`, one-line swap in `loaded_model.py` |
| Constants | `WORLD_SIZE` from visible GPUs, `ENGINE_MAX_NUM_SEQS = 128`, `ENGINE_CONCURRENCY`, `RECOMMENDED_BATCH_SIZE` (4×) | same file, exported from `activation.harness` |
| Model defaults | `VLLMWrapper.recommended_engine_kwargs(model_id)`: the validated engine formula per known model (dense 27B with MTP, A3B MoE without), applied by the harness under any explicit `engine_kwargs` | same file, merge in `loaded_model.py` |
| Label model | `dataset_study_label_model_name` (MoE, MTP off) separate from the generator (dense, MTP on); one engine resident at a time, the harness swaps | `runtime_config.py`, `dataset_study.py`, `loaded_model.py` |
| Batch defaults | study and label batch knobs become optional and default to the constants | `runtime_config.py`, `dataset_study.py` |
| Gates | study test on 1× and 2× RTX PRO 6000; L40S FP8 startup check at `max_num_seqs` 128 | `tests/test_basic_dataset_study.py`, SkyPilot |
| Later | session → replica pinning for rollout KV reuse | M4, design only |

No change to `activation/cloud/sky.py`: `setup --gpu-count N` and `exec --gpus TYPE:N` already
request N accelerators, and the probes ran through the wrapper unchanged. The 2-GPU shape is
`g7e.12xlarge` (2× RTX PRO 6000, $8.29/h on demand), then `g7e.24xlarge` (4×).

### How a batch flows

`activation/harness/vllm_wrapper.py · VLLMWrapper.chat()`

```
DatasetStudyGenerator.engine_chat_many(conversations)          # unchanged
        │
        ▼
LoadedModel.vllm_model.chat(conversations, sampling_params, …)  # same call as today
        │  VLLMWrapper
        ├── shard 0 = conversations[0::N] ──thread──▶ vllm.LLM #0 (GPU 0, own scheduler, own KV cache)
        ├── shard 1 = conversations[1::N] ──thread──▶ vllm.LLM #1 (GPU 1)
        └── …
        ▼
results reassembled in input order → EngineChatOutput list       # unchanged
```

Each `vllm.LLM` still spawns its own engine-core process, so nothing on our side launches processes
or sets per-process environment beyond `CUDA_VISIBLE_DEVICES` during construction. There is no
scheduler: a shard that finishes early idles until the others do, which the 4× backlog amortizes.

### Evidence from the probes

All on 2× RTX PRO 6000 (`g7e.12xlarge`), 512 study prompts (4k-char chunk) and 128 label prompts
(32k-char pool ≈ 6.8k tokens each), warm engines. Logs: `IB/TMP/MULTI_GPU/`.

| Model, mechanism | Labels: prompt tok/s | Study: output tok/s |
| --- | --- | --- |
| Dense 27B NVFP4 + MTP, one replica | 11.8k | 914 |
| Dense 27B NVFP4 + MTP, vLLM data parallel ×2 | 23.0k | 1,777 |
| Dense 27B NVFP4 + MTP, vLLM data parallel ×2, `max_num_seqs` 128 | 23.3k | 1,818 |
| MoE 35B-A3B NVFP4, no MTP, one replica | 20.1k | 1,227 |
| MoE, no MTP, vLLM data parallel ×2 | 35.6k | 1,804 |
| MoE, no MTP, vLLM data parallel ×2, `max_num_seqs` 128 | 34.3k | 2,504 |
| MoE + MTP, vLLM data parallel ×2 | 30.5k | 2,311 |
| **MoE, no MTP, two independent engines (this plan)** | **68.7k** | **2,918** |

Readings: dense scales linearly under vLLM's data parallel because vLLM treats non-MoE ranks as
independent; MoE does not, because vLLM hard-codes a lockstep engine for any model with experts
(explainer below), and independent engines recover the full 2×. MTP taxes MoE prefill (labeler: off)
and helps decode (generator: on). `max_num_seqs` 128 is neutral for the dense model and +28% decode for
the MoE. Four-times oversubscription lifted dense generation 27% over an exact-fit batch.

### Expected rates after this plan

| Per hour | 1× RTX PRO 6000 | 2× RTX PRO 6000 |
| --- | --- | --- |
| Labeling, MoE, near worst-case pool | ~18,000 | ~36,000 |
| Generation, dense + MTP, 140-token answers | ~24,000 | ~48,000 |
| Generation, MoE + MTP (uncalibrated in replicas mode) | ~45,000 | ~90,000 |
| Generate + label, dense generator | ~10,000 | ~21,000 |

### Boundaries

- Single node only. Multi-node replicas would need a remote router; out of scope.
- Every engine claims `gpu_memory_utilization` of every visible GPU. The embedding model cannot share
  a GPU with a resident engine, and two engines cannot be resident together. The harness frees the
  other engines when one is placed (M2).
- Tensor, pipeline, and expert parallel are not used. Each replica holds the whole model; every model
  in scope fits one 96 GB card (and the FP8 variants fit a 45 GB L40S).
- LoRA per replica is untouched: `chat_kwargs` pass through, and LoRA routing remains unsupported as today.

## Requested Decisions

None open. Both questions from the first pass were answered in the projection on 2026-09-03 and are
folded into the milestones:

- **Generator stays the dense 27B with MTP.** The MoE is the labeler only. Switching later is a
  model-name change because the wrapper now carries each model's engine formula (M1).
- **The GPU study test loads both models** and asserts the whole sequence: load the generator,
  generate, free it, load the labeler, label, free it (M2).

### Accepted decisions

- **Mechanism: independent engines, no scheduler.** One `vllm.LLM` per visible GPU; a batch is split
  by world size and sent; the harness never queues or rebalances. Applies to dense and MoE alike.
- **No world-size knob.** Every visible GPU gets a replica; restrict with `CUDA_VISIBLE_DEVICES` on the
  process if a GPU must be kept free.
- **`ENGINE_MAX_NUM_SEQS = 128`, tentative.** Neutral for dense, +28% MoE decode, expected to matter
  on bigger cards. The M1 gate includes an L40S FP8 startup check so the hybrid state cache at 128 slots
  is proven on a 45 GB card.
- **Batch constants live with the wrapper.** `ENGINE_CONCURRENCY = WORLD_SIZE × 128` for prefill-bound
  passes (labels), `RECOMMENDED_BATCH_SIZE = 4 × ENGINE_CONCURRENCY` for decode-heavy passes (study).
- **Labeler: MoE `nvidia/Qwen3.6-35B-A3B-NVFP4` (FP8 `Qwen/Qwen3.6-35B-A3B-FP8` without FP4), MTP off.**
- **Generator: dense `unsloth/Qwen3.8-27B-NVFP4` (FP8 `Qwen/Qwen3.8-27B-FP8` without FP4), MTP on.**
- **Engine formulas live on the wrapper.** `VLLMWrapper.recommended_engine_kwargs(model_id)` returns
  the validated kwargs for the four known ids; unknown ids get the base formula without speculative
  decoding and a printed note. `LoadedModel.engine_to_device` merges harness defaults, then the
  recommendation, then the model config's explicit `engine_kwargs`, so callers pass nothing for known
  models and can still override anything. No `recommended_chat_kwargs`: sampling and structured
  outputs are per pass, not per model, and thinking is already disabled by default in `engine_chat_many`.
- **The study test exercises the engine swap end to end**: generator load → generate → free →
  labeler load → label → free, asserted from the harness stats.
- **Label top-k stays 10; pool budget stays 32,768 chars.**

## Explainers

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">Kinds of parallelism in vLLM, and why replicas</span>
    <span class="card-oneliner">Tensor, pipeline, expert, and data parallel each solve a different problem; ours is throughput of a model that already fits.</span>
    <span class="card-badge">Explainer</span>
  </summary>

| Kind | Splits | Use when | Cost |
| --- | --- | --- | --- |
| Tensor parallel (TP) | every layer's weights across GPUs | one model does not fit one GPU | all-reduce every layer; GPUs must step together |
| Pipeline parallel (PP) | layers across GPUs | same, with weaker interconnect | bubbles between stages |
| Expert parallel (EP) | MoE experts across GPUs | experts alone do not fit | all-to-all in every MoE layer |
| Data parallel (DP) | requests across full copies | the model fits, you want throughput | none in principle; see next explainer for vLLM's MoE caveat |
| **Independent replicas (this plan)** | requests across full copies, no shared group | same as DP | tail effect within a batch |

Every model in scope fits one card, so only the last two rows apply. vLLM's data parallel is the last
row for dense models and something else for MoE, which is why the plan owns the replicas.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">vLLM's process model and the MoE lockstep</span>
    <span class="card-oneliner">Why the sync `LLM` refuses data parallel, why the async one couples MoE replicas, and why neither matters once we own the replicas.</span>
    <span class="card-badge">Explainer</span>
  </summary>

A `vllm.LLM` is a thin frontend in our process plus an engine-core subprocess that vLLM spawns
itself (the Sky wrapper forces the spawn start method). The frontend tokenizes and detokenizes; the
core schedules and runs the GPU. Data parallel in vLLM means several cores behind one frontend.

- The sync `LLM` class raises on `data_parallel_size > 1` ("not supported for single-process usage and
  may hang"). Only the async engine, which the API server uses, launches and load-balances several cores.
- For a model without experts, vLLM's engine selector says "Non-MoE DP ranks are completely
  independent, so treat like DP=1". That is why the dense 27B scaled linearly.
- For any model with experts, the selector hard-codes a lockstep engine (`DPEngineCoreProc`, asserted
  MoE-only). Ranks exchange token counts and pad to the largest batch every step, an idle rank runs
  dummy forward passes, and they all-reduce a finish flag. The detection reads the architecture's
  expert count, so there is no flag to turn it off. On our workload it cost the MoE half its second GPU.

Owning the replicas sidesteps all of it: each `vllm.LLM` is data-parallel size one, so vLLM never
enters the coupled path, and the one-GPU behavior is bit-for-bit today's.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">Memory: what can and cannot OOM</span>
    <span class="card-oneliner">Batch and backlog sizes are throughput knobs; GPU memory is decided once at engine startup.</span>
    <span class="card-badge">Explainer</span>
  </summary>

At startup each engine measures free memory on its GPU, runs one profiling pass at the largest step
it will ever schedule (`max_num_batched_tokens` = 16k tokens across `max_num_seqs` sequences), and
carves the remainder of `gpu_memory_utilization` into a fixed KV-block pool. Nothing on the GPU grows
with load afterwards: requests take blocks from the pool, wait when it is full, and get preempted and
recomputed under pressure. On the RTX PRO 6000 the pool held 898k tokens; 128 prompts of 1k tokens
use a seventh of it.

What can fail, all at startup and fail-fast:

| Knob | Failure |
| --- | --- |
| `gpu_memory_utilization` on a GPU that already holds something (embedding model, another engine) | refused: less than the fraction is free. This is the real risk in the harness and why M2 frees other engines before placing one. |
| `max_model_len` larger than one request's share of the pool | refused with the limit named |
| `max_num_batched_tokens` / `max_num_seqs` far too large | profiling peak exceeds the budget; for hybrid models the per-slot linear-attention state grows with `max_num_seqs`, which is why 64 was called "hybrid-safe" and 128 gets an L40S check |

Host RAM is the other side: the first-run kernel compile fans out compiler processes (bounded by the
`MAX_JOBS` default in `runtime_config.py`), and the frontend holds every prompt and answer of a call,
which is megabytes at our sizes.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">Continuous batching and the 4× backlog</span>
    <span class="card-oneliner">Why a call should carry more than the engine can run at once, and why labeling wants less of it than generation.</span>
    <span class="card-badge">Explainer</span>
  </summary>

An engine admits up to `max_num_seqs` sequences and, whenever one finishes, pulls the next from its
queue in the same step. With a backlog, slots never idle until the very end of the call. With an
exact-fit batch the call drains: answers differ in length, so the last long ones run nearly alone.
A backlog of about four times the cap amortizes that tail; measured +27% on dense generation.

Labeling is prefill-bound: the throughput follows `max_num_batched_tokens` (16k tokens per step, about
two worst-case pools), and concurrency barely matters. So it gets `ENGINE_CONCURRENCY`, generation gets
`RECOMMENDED_BATCH_SIZE`. Costs of a big call: results return only when the whole call finishes, and a
crash loses the call. Both are fine offline at a few minutes per call.

Why labeling is slower than generation at all: a label prompt carries about 6.8k tokens (eight 4k-char
chunks) against about 0.9k for a study prompt, at the same prefill rate. Prefix caching does not help
because pools are different chunk sets in different orders.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">Session and KV stickiness for multi-step rollouts (M4 preview)</span>
    <span class="card-oneliner">How a growing agent conversation keeps hitting the replica that already holds its prefix.</span>
    <span class="card-badge">Explainer</span>
  </summary>

Each replica has its own KV cache and prefix cache. A rollout is a conversation whose prompt grows by
one turn per step. If step t+1 lands on the replica that served step t, the shared prefix is found in
that replica's prefix cache and only the new tokens are prefilled. If it lands elsewhere, the whole
conversation is re-prefilled, a cost that grows with every turn.

The strided split in `VLLMWrapper.chat` is positional, so a session's turns would scatter. M4 adds the
smallest possible affinity without a scheduler: an optional `session_ids` argument to `chat`, and a
stable hash of the session id picks the replica (`hash(session_id) % WORLD_SIZE`) instead of the
position. Same code shape, same thread per replica, same in-order return. Load can drift between
replicas when sessions are uneven; that is accepted for a first version and measurable from the
per-replica shard sizes.

Two vLLM features are relevant later and deliberately not used now:

- `data_parallel_rank` on the async engine's `generate` steers a request to a rank, but only on the
  vLLM-managed path, which this plan leaves for the MoE reason above.
- vLLM 0.28 has resumable streaming sessions (`session_id`, `StreamingInput`): a request stays paused in
  the scheduler holding its KV blocks between turns. Stronger than prefix caching but pins memory per
  open session and is a new API; evaluate once rollouts exist.

</details>

## Milestones

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M0 — Probes</span>
    <span class="card-oneliner">Mechanism and model choices measured on 2× RTX PRO 6000.</span>
    <span class="card-badge">Completed</span>
  </summary>

#### What landed

`IB/TMP/MULTI_GPU/dp_probe.py` (private, not product code) with two modes: `async` (vLLM async engine
with `data_parallel_size`) and `replicas` (N sync engines pinned through `CUDA_VISIBLE_DEVICES`, thread
per engine). Nine legs across two fresh `g7e.12xlarge` nodes, both torn down. Results are the evidence
table in the Executive Summary; raw logs in `IB/TMP/MULTI_GPU/probe_async_*.log` and
`IB/TMP/MULTI_GPU/moe_probe/`.

#### Drifts and unplanned steps

- The first plan for the async path was to keep it for dense models. The MoE results made one mechanism
  for both the simpler and faster choice, so the async path is evidence only.
- Legs were chained into one detached remote job because a local `sky exec` cannot outlive ten minutes.
- The replicas leg ran only with MTP off and `max_num_seqs` 64; the MoE-as-generator figure is
  therefore extrapolated, and M1's gate calibrates it.

#### Validation

Every leg exited 0 with parseable JSON under structured outputs, MTP where enabled, fp8 KV cache, and
`max_model_len` 20k. Steering by rank on the async path worked (all-to-rank-0 ran 1.6× slower than
balanced).

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M1 — VLLMWrapper behind the one-line swap</span>
    <span class="card-oneliner">Every visible GPU gets an engine; the harness call path does not change.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `activation/harness/vllm_wrapper.py` (constants, `recommended_engine_kwargs`, sequential
pinned construction, strided `chat`, `shutdown`); `loaded_model.py` builds engine kwargs as harness defaults →
recommendation → explicit config and constructs `VLLMWrapper`; `vllm_utils.py` deleted; `harness/__init__.py`
exports the wrapper and constants; the study test lost `STUDY_ENGINE_KWARGS`. Cards:
`MULTI_GPU_ENGINE_ROUND.tressoir.md`.

**Drifts.** `vllm_utils.py` is deleted instead of emptied (its only importer was `loaded_model.py`). `chat`
shards by `len(self.replicas)` rather than `WORLD_SIZE` so an explicit `CUDA_VISIBLE_DEVICES` list can never
disagree with the split.

**Validation.** CPU only: import and constants (`WORLD_SIZE 1`, concurrency 128, batch 512); formula table for
the four known ids plus an unknown id (note printed, no raise); shard/reassemble with fake replicas (order,
round-robin owners, error re-raise, empty batch); `test_basic_dataset_loading -k bm25` `2 passed in 20.51s`.
GPU gate (2026-09-04): 2× RTX cold `1 passed in 1574s` (two replicas per model, replica 1 loads warm);
1× warm `118.8s`, study 510.8 tok/s, labels 30.3k prompt tok/s; 2× warm `206.1s`, study 537.3 tok/s, labels
49.5k prompt tok/s (1.63×; 1.04× on study, which at 8 prompts per replica is latency-bound, so the 2× from
M0's 512/128-prompt probe cannot show at this test size); L40S FP8 `1 passed in 1198s`, 27B FP8 up at
`max_num_seqs` 128 with MTP, A3B FP8 after the swap. Logs `IB/TMP/MULTI_GPU/gate_*.log`; nodes torn down. Re-run on the user's adapted tree (qa renames, chat
recommendations, required label knob): 2× cold `1 passed in 1617.87s`, 1× warm `118.71s`, residency held
(`final_*.log`).

#### Planning Overview

New `activation/harness/vllm_wrapper.py` owns the replicas, the four constants, and the per-model
engine formulas. `LoadedModel` constructs a `VLLMWrapper` where it constructed a `vllm.LLM`, merges
the recommended kwargs for the model id under the model config's explicit ones, and shuts down through
the wrapper.
`engine_chat_many` is untouched: it already calls `.chat(conversations, sampling_params=…,
use_tqdm=False, **kwargs)` and reads token ids off each result. `vllm_utils.py` loses its shutdown helper.
The study test drops its whole `STUDY_ENGINE_KWARGS` block: the wrapper's formula for the dense 27B is
exactly that block minus `max_num_seqs`, so passing nothing yields the same engine.

Edge cases: replicas construct sequentially because `CUDA_VISIBLE_DEVICES` is process-global (the
first pays the kernel JIT, later ones load warm in about a minute); a shard error is re-raised after
all threads join; an empty shard skips the engine call; on a CPU host `WORLD_SIZE` is 1 and the engine
fails exactly as it does today.

Gate: study test on 1× RTX PRO 6000 (parity with the current stats), on 2× (expect about 2× tokens/s in
`study_output_tokens_per_s` and `study_label_prompt_tokens_per_s`), and an L40S startup smoke with the
FP8 27B at `max_num_seqs` 128 (engine placed, one small batch answered).

#### Planned Changes

`activation/harness/vllm_wrapper.py · new file`

```python
WORLD_SIZE = max(1, torch.cuda.device_count())
ENGINE_MAX_NUM_SEQS = 128
ENGINE_CONCURRENCY = WORLD_SIZE * ENGINE_MAX_NUM_SEQS
RECOMMENDED_BATCH_SIZE = 4 * ENGINE_CONCURRENCY


class VLLMWrapper:
    def __init__(self, **engine_kwargs):
        engine_kwargs = dict(engine_kwargs)
        if engine_kwargs.setdefault("max_num_seqs", ENGINE_MAX_NUM_SEQS) != ENGINE_MAX_NUM_SEQS:
            print(f"Engine max_num_seqs={engine_kwargs['max_num_seqs']} overrides {ENGINE_MAX_NUM_SEQS}; batch constants assume the latter.")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = visible.split(",") if visible else [str(i) for i in range(WORLD_SIZE)]
        self.replicas: list[vllm.LLM] = []
        try:
            for device in devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = device  # inherited by the spawned engine core
                self.replicas.append(vllm.LLM(**engine_kwargs))
        finally:
            if visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = visible

    @staticmethod
    def recommended_engine_kwargs(model_id: str) -> dict:
        """The validated engine formula for a known model; base formula (no speculative decoding) otherwise."""
        base = {
            "max_model_len": 20_000,
            "limit_mm_per_prompt": {"image": 0, "video": 0},
            "kv_cache_dtype": "fp8",
            "gpu_memory_utilization": 0.9,
            "enable_lora": False,
        }
        mtp = {"speculative_config": {"method": "mtp", "num_speculative_tokens": 2}}
        known = {
            # Dense generator: MTP helps decode.
            "unsloth/Qwen3.8-27B-NVFP4": base | mtp,
            "Qwen/Qwen3.8-27B-FP8": base | mtp,
            # A3B MoE labeler: MTP taxes prefill, and labeling is prefill-bound.
            "nvidia/Qwen3.6-35B-A3B-NVFP4": base,
            "Qwen/Qwen3.6-35B-A3B-FP8": base,
        }
        if model_id not in known:
            print(f"{model_id} - No recommended engine kwargs; using the base formula without speculative decoding.")
        return dict(known.get(model_id, base))

    def chat(self, conversations, **chat_kwargs) -> list[vllm.RequestOutput]:
        if WORLD_SIZE == 1:
            return self.replicas[0].chat(conversations, **chat_kwargs)
        shards = [conversations[rank::WORLD_SIZE] for rank in range(WORLD_SIZE)]
        results, errors = [None] * WORLD_SIZE, [None] * WORLD_SIZE
        def work(rank):
            try:
                results[rank] = self.replicas[rank].chat(shards[rank], **chat_kwargs) if shards[rank] else []
            except BaseException as error:
                errors[rank] = error
        threads = [threading.Thread(target=work, args=(rank,), daemon=True) for rank in range(WORLD_SIZE)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        for error in errors:
            if error is not None:
                raise error
        return [results[i % WORLD_SIZE][i // WORLD_SIZE] for i in range(len(conversations))]

    def shutdown(self):
        for replica in self.replicas:
            try:
                replica.llm_engine.engine_core.shutdown()
            except Exception:
                pass
        self.replicas = []
```

`activation/harness/loaded_model.py · imports, __init__, engine_to_device()`

```diff-python
-from .vllm_utils import (
-    best_effort_shutdown_vllm,
-)
+from .vllm_wrapper import VLLMWrapper
⋯
-        self.vllm_model: vllm.LLM|None = None
+        self.vllm_model: VLLMWrapper|None = None
⋯
             engine_kwargs = dict(
                 model=self.model_config.model_id,
                 max_loras=4,
                 max_lora_rank=self.harness.harness_config.max_lora_rank,
                 enable_lora=True,
                 dtype=self.model_config.dtype or "auto",
             )
+            # Harness defaults < recommended formula for this model < explicit model-config kwargs.
+            engine_kwargs.update(VLLMWrapper.recommended_engine_kwargs(self.model_config.model_id))
             engine_kwargs.update(self.model_config.engine_kwargs)
-            self.vllm_model = vllm.LLM(**engine_kwargs)
+            self.vllm_model = VLLMWrapper(**engine_kwargs)
⋯
-            best_effort_shutdown_vllm(self.vllm_model)
+            self.vllm_model.shutdown()
```

`activation/harness/vllm_utils.py` — `best_effort_shutdown_vllm` removed (its only caller is gone).

`activation/harness/__init__.py · exports`

```diff-python
+from .vllm_wrapper import (
+    VLLMWrapper,
+    WORLD_SIZE,
+    ENGINE_MAX_NUM_SEQS,
+    ENGINE_CONCURRENCY,
+    RECOMMENDED_BATCH_SIZE,
+)
```

`activation/tests/test_basic_dataset_study.py · engine kwargs`

```diff-python
-# Our canonical formula.
-STUDY_ENGINE_KWARGS = {
-    "max_model_len": 20_000,
-    "max_num_seqs": 64,
-    "limit_mm_per_prompt": {"image": 0, "video": 0},
-    "kv_cache_dtype": "fp8",
-    "gpu_memory_utilization": 0.9,
-    "speculative_config": {"method": "mtp", "num_speculative_tokens": 2},
-    "enable_lora": False,
-}
⋯
-            STUDY_MODEL_NAME: ModelConfig(
-                STUDY_MODEL_NAME,
-                STUDY_MODEL_NAME,
-                engine_kwargs=STUDY_ENGINE_KWARGS,
-            ),
+            STUDY_MODEL_NAME: ModelConfig(STUDY_MODEL_NAME, STUDY_MODEL_NAME), # Formula comes from the wrapper.
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M2 — Separate label model and derived batch sizes</span>
    <span class="card-oneliner">The MoE labels, the dense model generates, the harness swaps engines, and batch knobs default to the node.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `runtime_config.py`: both study batch knobs `int|None = None` plus
`dataset_study_label_model_name`. `loaded_model.ensure_engine_loaded` frees every other loaded model's engine
first. `dataset_study.py` imports the two constants, defaults the batch sizes, resolves `label_model_name`, and
the label pass uses it. The study test registers both models with bare configs, sets the label model, drops
the batch knobs, frees every engine in `finally`, and asserts the `[STUDY, LABEL]` load and free sequences.

**Drifts.** None against the plan.

**Validation.** Private contract check (stubs `ensure_engine_loaded` and `engine_chat_many`) passes with the
new defaults. Residency asserts `[STUDY, LABEL]` / `[STUDY, LABEL]` held on all four GPU legs (2× cold, 1×
warm, 2× warm, L40S FP8); engine swap cost warm: ~40–70 s generator, ~55–105 s labeler (one vs two replicas).

#### Planning Overview

Two new behaviors. First, `dataset_study_label_model_name` names the labeler (default: the study
model, so today's single-model setup keeps working); the generator resolves it per pass. Second, the
two batch knobs become optional and default to the wrapper constants: study batches use
`RECOMMENDED_BATCH_SIZE`, label batches `ENGINE_CONCURRENCY`.

Because every engine claims its GPUs, `LoadedModel.ensure_engine_loaded()` first frees the engine of
every other loaded model. That rule ("one engine resident at a time") lives in the harness, so the
dataset package still imports nothing from it. The label pass therefore costs one engine swap per
call: a few minutes warm, up to ten cold on a fresh node.

The study test registers both models with no engine kwargs (the wrapper's formulas apply), keeps
the generator as the dense 27B, and asserts the whole residency sequence from the harness stats:
the generator loads and generates, labeling frees it and loads the labeler, and the `finally` frees
the labeler. Loading order and freeing order are each checked as name sequences, so an unexpected
extra load or a missing free fails the test.

#### Planned Changes

`activation/harness/runtime_config.py · HarnessRuntimeConfig`

```diff-python
     # Dataset Study
-    dataset_study_batch_size: int = 16
+    dataset_study_batch_size: int | None = None # None: RECOMMENDED_BATCH_SIZE (4 × node concurrency).
     dataset_study_model_name: str|None = None
+    dataset_study_label_model_name: str|None = None # None: the study model labels too.
⋯
-    dataset_study_label_batch_size: int = 16 # Label prompts carry the whole pool, so they get their own batch size.
+    dataset_study_label_batch_size: int | None = None # None: ENGINE_CONCURRENCY (labeling is prefill-bound).
```

`activation/harness/loaded_model.py · ensure_engine_loaded()`

```diff-python
     def ensure_engine_loaded(self):
+        # One engine resident at a time: every engine claims its share of every visible GPU.
+        for other in self.harness.loaded_models.values():
+            if other is not self:
+                other.engine_to_device(FREE_DEVICE)
         self.engine_to_device(TARGET_DEVICE)
```

`activation/dataset/dataset_study.py · DatasetStudyGenerator.__init__, generate_example_labels()`

```diff-python
+from ..harness.vllm_wrapper import ENGINE_CONCURRENCY, RECOMMENDED_BATCH_SIZE  # constants only, no runtime harness objects
⋯
-        self.batch_size = harness.harness_config.dataset_study_batch_size
-        self.label_batch_size = harness.harness_config.dataset_study_label_batch_size
+        self.batch_size = harness.harness_config.dataset_study_batch_size or RECOMMENDED_BATCH_SIZE
+        self.label_batch_size = harness.harness_config.dataset_study_label_batch_size or ENGINE_CONCURRENCY
+        self.label_model_name = harness.harness_config.dataset_study_label_model_name or self.study_model_name
⋯
-        loaded_model = self.harness.loaded_models[self.study_model_name]
+        loaded_model = self.harness.loaded_models[self.label_model_name]
         loaded_model.ensure_engine_loaded() # Keep the cold load out of the batch timings.
```

`activation/tests/test_basic_dataset_study.py · model configs and residency assertions`

```diff-python
+if SUPPORTS_FP4:
+    LABEL_MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
+else:
+    LABEL_MODEL_NAME = "Qwen/Qwen3.6-35B-A3B-FP8"
⋯
         model_configs={
             STUDY_MODEL_NAME: ModelConfig(STUDY_MODEL_NAME, STUDY_MODEL_NAME),
+            LABEL_MODEL_NAME: ModelConfig(LABEL_MODEL_NAME, LABEL_MODEL_NAME), # MoE formula: no MTP.
         },
         dataset_study_model_name=STUDY_MODEL_NAME,
+        dataset_study_label_model_name=LABEL_MODEL_NAME,
-        dataset_study_batch_size=STUDY_BATCH_SIZE,
-        dataset_study_label_batch_size=STUDY_LABEL_BATCH_SIZE,
⋯
-    study_model = harness.loaded_models[STUDY_MODEL_NAME]
     try:
         ... synthesize (generator resident) ...
         ... label synthetic, then native (labeler resident) ...
     finally:
-        study_model.engine_to_device(FREE_DEVICE)
+        for loaded_model in harness.loaded_models.values():
+            loaded_model.engine_to_device(FREE_DEVICE)
+    # Residency sequence: load generator → generate → free → load labeler → label → free.
+    loads = [name for name, _ in harness.harness_stats.model_loading_times]
+    frees = [name for name, _ in harness.harness_stats.model_freeing_times]
+    assert loads == [STUDY_MODEL_NAME, LABEL_MODEL_NAME], loads
+    assert frees == [STUDY_MODEL_NAME, LABEL_MODEL_NAME], frees
```

The residency assertion relies on the study test never loading the HF model or the embedding layer
(it builds bm25 only), so the harness stats hold engine events alone. Cold runtime on a fresh node is
about 25 minutes (two first-run kernel compiles); warm, a few minutes.



</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M3 — Docs, canon, and the 1-hour budget note</span>
    <span class="card-oneliner">Where the replica rules and the measured rates are written down.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** SkyPilot README: `--gpu-count` note (g7e.12xlarge 2×, g7e.24xlarge 4×, g6e.12xlarge 4× L40S,
scarcer capacity, one engine per visible GPU with no extra flag). `ROOT_CANON.md`: "Engine replica decisions"
(five entries: independent engines and why not vLLM DP, no world-size knob + one engine resident,
`ENGINE_MAX_NUM_SEQS` and the two batch constants, formulas in `recommended_engine_kwargs` with the merge
order, MoE labeler MTP off / dense generator MTP on). `STATE.md`: multi-GPU section rewritten as milestone
results; the old per-hour budget lines retired into `PLAN.md`'s "Derived planning rates".

**Drifts.** The canon section has five bullets rather than the four sketched: the formula/merge-order
decision was accepted in the interaction file after the sketch was written.

#### Planning Overview

- SkyPilot README: `--gpu-count 2` maps to `g7e.12xlarge` (2× RTX PRO 6000) and 4 to `g7e.24xlarge`;
  the L40S 4× shape is `g6e.12xlarge`; multi-GPU capacity is scarcer, so expect more region shopping.
- `IB/CANON/ROOT_CANON.md`, new "Engine replica decisions" section: one independent engine per visible
  GPU, no scheduler, no world-size knob, one engine resident at a time, `ENGINE_MAX_NUM_SEQS` 128, batch
  constants per pass, MoE labeler without MTP, vLLM data parallel avoided for MoE and why.
- `IB/STATE.md`: replace the probe summary with the milestone results; retire the old 1-hour budget table.

#### Planned Changes

`IB/CANON/ROOT_CANON.md · new section`

```diff-markdown
+## Engine replica decisions
+
+- **Decision — One independent vLLM engine per visible GPU, no scheduler.** `VLLMWrapper` splits a batch by
+  world size and reassembles in order. vLLM's own data parallel is avoided: it hard-codes a lockstep engine
+  for any MoE model, measured at half the throughput of independent engines.
+- **Convention — No world-size knob.** Restrict with `CUDA_VISIBLE_DEVICES`; one engine resident at a time.
+- **Decision — `ENGINE_MAX_NUM_SEQS = 128`; labels batch at node concurrency, study at 4×.**
+- **Decision — Labeler is the A3B MoE with MTP off; generator keeps MTP on.**
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M4 — Session stickiness for rollouts</span>
    <span class="card-oneliner">Hash a session id to a replica so multi-step conversations reuse their prefix cache.</span>
    <span class="card-badge">TBD</span>
  </summary>

#### Planning Overview

Depends on the rollout code existing. Shape from the explainer: `VLLMWrapper.chat(conversations,
session_ids=None, …)`; with session ids, shard by `hash(session_id) % WORLD_SIZE` instead of position.
No scheduler, no queue. Validation would measure prefill tokens per turn with and without session ids
on a synthetic 8-turn rollout. vLLM's resumable streaming sessions are the candidate if prefix caching
proves insufficient.

</details>
