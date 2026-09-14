# Rollout throughput tuning (slice 4a subplan)

Subplan of slice 4a for the teacher campaign's throughput: a common-sense, static tuning of the rollout stack, decided once from one probing phase and then held. The rollout manager gains an `autotune_id`: the first run under a new id runs 200 tasks per GPU under the defaults while recording engine and host metrics, hands them to an autotuner that returns one static configuration by simple rules, persists it under that id, and runs the remaining tasks under it; every later run with the same id applies the saved configuration, whatever the pool or node looks like. No adaptive routing, no search. Everything runs the oracle at reasoning effort medium, and every rollout of the tuning run is cached under the campaign's id, so it is campaign data, not a throwaway.

## Executive Summary

**Where we stand.** Four single-GPU probes of 30 tasks (`PROBE_COMPARE.tressoir.md`) showed: one 27B replica per GPU with 1.23M tokens of fp8 KV cache; prefix hits at 71 to 76% of prompt tokens with 12 to 16% of prompt tokens recomputed beyond the inherent block-alignment misses; no running-request preemption; wall clock set by the tail of one or two 80-turn runs while the engine idles at 76 generated tokens per second. The agent pool is a flat 64 threads whatever the GPU count, agents are pinned to replicas by a hash, and the engine's statistics are switched off, so none of this was measured in steady state. Thirty tasks never reach one.

**Approach.** Static tuning from evidence, in three steps.

| Step | What | Output |
| --- | --- | --- |
| T0 Instrumentation and static fixes | Engine statistics logger per replica, per-turn timing on the trajectory, host sampler; the fixes that need no tuning (per-GPU pool bound, balanced replica assignment, prefix warm-up, reporter cost, duration cap) | Metrics on every rollout run; a 2-GPU node fed properly |
| T1 Tuning config and autotuner | `RolloutTuningConfig`, `Autotuner.tune(metrics) -> config` by fixed rules, persistence keyed by the caller's `autotune_id`, `perform_single_rollouts(..., autotune_id=...)` running a probe phase on a new id and the saved configuration on a known one | One configuration per autotune id, reused by every later run with that id |
| T2 Tuning run | 2x RTX PRO 6000, suite tasks at medium effort: 200 per GPU under the defaults, then the same count under the tuned configuration, both measured | `TUNING_REPORT.tressoir.md` comparing the two phases; the cached rollouts join the campaign corpus |

**The knobs, ranked by expected effect** (the high poles). Only throughput knobs are tuned; knobs that change the data an agent produces stay fixed by decision.

| Knob | Today | Why it matters | Set how |
| --- | --- | --- | --- |
| Agents per GPU (pool bound = per GPU x replicas) | 64 total, not per GPU | Too few starves the engine, too many evicts idle prefixes and re-prefills 30k to 50k tokens | Rule from the probe: KV capacity over the 90th percentile live prefix, with headroom |
| Replica assignment | crc32 of the agent id | Unbalanced at tens of agents and blind to run length | Static round-robin at agent start, pinned for the agent's life (T0, no tuning) |
| Engine `max_num_batched_tokens` | vLLM default (8,192) | The workload is prefill-heavy; larger chunks raise prefill throughput at some decode latency | Rule: raise to 16,384 when the probe's prefill share is high |
| Engine `gpu_memory_utilization` | 0.9 | More KV headroom per GPU | Rule: 0.92 when the probe shows no host-side pressure; otherwise keep |
| Speculative decoding (MTP, 2 tokens) | on | Helps decode at low batch, taxes prefill at high batch | Probe records generation throughput; the rule keeps MTP unless the batch is saturated |
| Prefix warm-up | none: 30 simultaneous first prompts all missed | One request per replica caches the shared system prompt and tool list | T0, no tuning |
| Reporter cost | every step rewrites every running agent's trajectory file | Quadratic host work at 64 or more agents | T0: write only the stepping agent's file, render every 10 s |
| Duration cap | 1,800 s | Binds on long tasks under load (medium and xhigh hit it); it shapes the data | Decision D2: 3,600 s for the campaign, fixed |

**Data flow of an autotuned run** (`RolloutManager.perform_single_rollouts`):

```
perform_single_rollouts(configs, ..., autotune_id="campaign_27b_medium")
  |- saved TUNING/<autotune_id>.json ?  -> yes: apply it, run everything under it (pool or node changes do not re-probe)
  |- no: phase A = first 200 x gpu_count jobs under the defaults
  |        RolloutStatLogger (per replica)  -> kv usage, prefix hits, running/waiting, tokens per interval
  |        Agent turn timing (per step)     -> turn latency, tool time, prefix length
  |        HostSampler (5 s)                -> GPU utilization, CPU, memory
  |     ProbeMetrics.summarize(...)         -> percentiles the rules read
  |     Autotuner.tune(metrics, defaults)   -> RolloutTuningConfig (rollout_config_id)
  |     persist; apply: pool size now, engine kwargs through one engine reload if they changed
  |- phase B = the remaining jobs under the tuned configuration, same metrics
  '- report: Tuning section with phase A against phase B
```

**Boundaries.** The autotuner is a table of rules, not an optimizer: a handful of thresholds anyone can read and override. The tuned configuration is a static file named by the autotune id; a later run with the same id reuses it without a probe, and a new id is the way to re-tune (settled in chat: no force flag, no automatic keying by node). Behavior-shaping budgets (compaction threshold, trajectory cap, turns) are not tuned. Multi-node and dynamic routing are out of scope.

## Accepted decisions

- **Probe size: 200 tasks per GPU** (your proposal).
- **Scope: throughput knobs only.** Agents per GPU, batched tokens, memory utilization, speculative decoding. Budgets are fixed by decision: duration cap 3,600 s, compaction threshold 32k, cap 50k, 100 turns.
- **One engine reload between phases is acceptable.**
- **Node: 2x RTX PRO 6000 on AWS**, which validates the per-replica pool and the balanced assignment.
- **Reuse by `autotune_id`** (chat): the first run under an id probes and saves `TUNING/<autotune_id>.json`; every later run with that id applies the file as is, even if the pool, the GPU count or the node changed. A new id is a new tuning.

## Milestones

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">T0 — Instrumentation and static fixes</span>
    <span class="card-oneliner">Engine statistics per replica, per-turn timing, host sampler; per-GPU pool, balanced assignment, warm-up, reporter cost, duration cap.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `RolloutStatLogger` per replica (KV usage, prefix queries and hits, running and waiting requests, prompt and generated tokens per interval), `timing` on every `TrajectoryStep`, the host sampler, the per-GPU pool bound (`agents_per_gpu x world_size`), round-robin replica assignment pinned per agent, one warm-up request per replica, the reporter writing only the stepping agent's file, and the 3,600 s cap. Two fixes found by the first tuning run were added: the BM25 index of a searchable dataset is built once before the pool starts (`_build_indexes`) and guarded by a lock in `DatasetIndex` and `DatasetManager`, so 78 agents no longer build the same index at once; repositories are no longer indexed at all (`index_files_per_repo: 0`), because the seed-0 rows showed the code agents reaching the answer through `rg`, `find` and `cat` and the index only cost the build.

**Drifts.** The reporter now also keeps a per-benchmark summary table (`summary.json`: tasks, finish reasons, turns, generated tokens, compactions, subagents, tool calls, scores) because the campaign report needed it; the seed-0 run (`teacher_campaign_medium`, 804 rows) is kept but 67 LCA rows hit the cap while waiting on the index storm, so a filtered 737-row copy is the usable one.

**Validation.** `test_basic_rollout_tuning.py` (index lock, warm-up, assignment), `test_basic_teacher_study.py` (reporter summary), both green; the stat logger ran on `ac-4a-tune` and `ac-4a-tune2` (series in `node_tune2/tuning_v2/probe.json`).

#### Planning Overview (as planned)

Measure, and fix what needs no measurement. Four pieces.

1. **Engine statistics.** A `RolloutStatLogger` implementing vLLM's `StatLoggerBase` (0.28: `record(scheduler_stats, iteration_stats, ...)`) keeps a time series per replica: KV cache usage, prefix cache queries and hits, running and waiting requests, prompt and generated tokens per interval. Installed through the `stat_loggers` factory list when the replica constructs its `AsyncLLM`; `disable_log_stats` stays true so the console is not flooded. `VLLMWrapper.metrics()` returns the series of every replica.
2. **Per-turn timing on the trajectory.** `Agent._submit` records wall time from submission to output, prompt length, cached length and generated length; tool execution time per call is recorded on the tool step. Both go into `TrajectoryStep` as a small `timing` dict (serialized with the run, so cached rollouts carry them).
3. **Host sampler.** A thread in the rollout manager samples `nvidia-smi` (utilization, memory) per GPU, CPU utilization and live container count every 5 seconds while a pass runs.
4. **Static fixes.** Pool bound `agents_per_gpu x world_size` with a new runtime field `agent_max_concurrent_per_gpu = 40` (`agent_max_concurrent` becomes an explicit override); round-robin replica assignment at agent start, pinned for the agent's life (subagents inherit the parent's replica); one warm-up request per replica with the shared system prompt and tool list before the pool starts; the reporter writes only the stepping agent's trajectory file and renders at most every 10 seconds; the suite's `max_duration` becomes 3,600 s (D2).

Validation: CPU tests with the tiny fixture for the timing fields and their serialization, the round-robin assignment and the reporter's per-step write count; the stat logger is exercised on the node in T2 (its interface is vLLM-internal, so the test is the run).

#### Planned Changes

`activation/harness/vllm_wrapper.py · RolloutStatLogger, _Replica`

```diff
+class RolloutStatLogger(StatLoggerBase):
+    """Time series of the scheduler's view: kv usage, prefix hits, queue lengths, tokens per interval."""
+    def __init__(self, vllm_config, engine_index: int = 0):
+        self.samples: list[dict] = []
+    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
+        if scheduler_stats is None: return
+        self.samples.append({"t": time.time(), "kv_usage": scheduler_stats.kv_cache_usage,
+                             "running": scheduler_stats.num_running_reqs, "waiting": scheduler_stats.num_waiting_reqs,
+                             "prefix_queries": scheduler_stats.prefix_cache_stats.queries, "prefix_hits": scheduler_stats.prefix_cache_stats.hits,
+                             "prompt_tokens": iteration_stats.num_prompt_tokens if iteration_stats else 0,
+                             "generation_tokens": iteration_stats.num_generation_tokens if iteration_stats else 0})
+    def log_engine_initialized(self): pass
 class _Replica:
     def __init__(self, engine_kwargs: dict):
+        self.stats = RolloutStatLogger
         async def construct():
-            return AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs))
+            return AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs), stat_loggers=[self._logger_factory])
```

`activation/harness/vllm_wrapper.py · replica assignment`

```diff
-    @staticmethod
-    def replica_for(agent_id: str, world_size: int) -> int:
-        return zlib.crc32(agent_id.encode()) % world_size
+    def assign_replica(self) -> int:
+        """Round-robin at agent start; the agent keeps the replica for its life (prefix locality)."""
+        with self._assign_lock:
+            self._next_replica = (self._next_replica + 1) % self.world_size
+            return self._next_replica
```

`activation/agent/agent.py · _submit, _execute_tool_calls`

```diff
     def _submit(self, chat_kwargs=None):
+        started = time.time()
         output = self.loaded_model.engine_submit_tokens(prefix, ..., replica=self.replica)
+        self._pending_timing = {"turn_seconds": time.time() - started, "prompt_tokens": output.prompt_token_count,
+                                "cached_tokens": output.cached_prompt_token_count, "output_tokens": output.output_token_count}
         return output
```

`activation/harness/runtime_config.py · HarnessRuntimeConfig`

```diff
-    agent_max_concurrent: int = 64 # Thread pool bound for rollouts; the engine batches across them.
+    agent_max_concurrent_per_gpu: int = 40   # Rollout pool bound per engine replica (KV budget: ~1.2M tokens / ~25k live prefix, with headroom).
+    agent_max_concurrent: int | None = None  # Explicit total bound; None = per_gpu x replicas.
```

`activation/agent/rollout_reporter.py · report_agent_step`

```diff
     def report_agent_step(self, agent):
         with self.lock:
-            self._update_trajectories()
+            self._update_trajectory(agent)            # only this agent's item and file
```

Omitted here: the host sampler thread (about 40 lines in `rollout_manager.py`), the warm-up request, the suite's duration cap, the tests.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">T1 — Tuning config, autotuner, manager flag</span>
    <span class="card-oneliner">RolloutTuningConfig, rule-based Autotuner, a saved configuration per autotune id, perform_single_rollouts(autotune_id=...) in two phases.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `RolloutTuningConfig` (agents per GPU, batched tokens, memory utilization, speculative decoding; `save`/`load` under `TUNING/<autotune_id>.json`), the rule-based `Autotuner` (KV budget 0.8 x capacity over the p90 live prefix, x0.75 when prefixes are evicted, batched tokens 16,384 when the prefill share exceeds 0.7, utilization 0.92 without host pressure, speculative decoding off when the batch is saturated), and `perform_single_rollouts(..., autotune_id=...)`.

**Drifts from the plan, all from the first two tuning runs.**

- *Continuous queue instead of two static halves.* The probe is no longer the first N submitted jobs but the first N finished: one straggler among the first 400 delayed the v2 retune to the end of the run, so the tuned phase measured one task. Jobs keep flowing through a FIFO `_PoolLimiter` whose limit the tuner changes in place.
- *Order shuffled* (`random.Random(0)`) so the probe sees the campaign mix rather than the first benchmark's block.
- *Engine reload deferred.* `config.apply(harness, loaded_model, reload=False)` sets the pool size at once; engine kwargs are saved and applied by the next run under the same id (`tuning_report.reload_deferred`). A reload mid-run would drop every live prefix, and the 12 h run starts from the saved file anyway.
- *Wall-clock deadline* (`max_wall_seconds`): jobs acquiring the pool after the deadline are skipped, started jobs get `max_duration` shortened, and a rollout that ends by deadline is not written to the cache.

**Validation.** `test_basic_rollout_tuning.py` (rules, save/load, queue retune) and `test_basic_rollout_deadline.py` (3 tests) green; end to end on `ac-4a-tune2`.

#### Planning Overview (as planned)

`activation/agent/rollout_tuning.py` holds the configuration, the metrics summary and the rules.

- `RolloutTuningConfig`: `rollout_config_id` (hash of model id, GPU name, GPU count and the tuned values), `agents_per_gpu`, `max_num_batched_tokens`, `gpu_memory_utilization`, `speculative` (bool), plus `notes` (which rule fired, with the numbers). `apply(harness)` sets the pool bound and the engine kwargs, reloading the engine when they differ from the resident ones. Persisted as `TUNING/<autotune_id>.json` under the artifacts root (the file records the model id, GPU name and count it was tuned on, for the report); `RolloutTuningConfig.load(autotune_id)` returns it or None.
- `ProbeMetrics.summarize(stat series, host series, run results)`: p50 and p95 KV usage, prefix recompute share beyond the alignment bound, fraction of intervals with a waiting queue, prefill share of engine tokens, p90 live prefix, median turn latency, sandbox time share, GPU utilization p50, host CPU p95, tasks and generated tokens per hour per GPU.
- `Autotuner.tune(metrics, defaults) -> RolloutTuningConfig` by these rules, in order:

| Rule | Condition | Setting |
| --- | --- | --- |
| Concurrency from the KV budget | always | `agents_per_gpu = clamp(0.8 x kv_tokens / p90_live_prefix, 16, 96)` |
| Starved engine | p95 KV usage < 60% and sandbox share > 25% | `agents_per_gpu x 1.25` |
| Evicting | recompute share > 20% or waiting queue in > 10% of intervals | `agents_per_gpu x 0.75` |
| Prefill-bound | prefill share of engine tokens > 70% | `max_num_batched_tokens = 16_384` |
| KV headroom | host memory p95 < 80% and no CUDA OOM in the probe | `gpu_memory_utilization = 0.92` |
| Speculation | generation tokens per second per replica below 60% of the single-agent rate at p95 KV usage > 80% | `speculative = False` |

- `RolloutManager.perform_single_rollouts(..., autotune_id: str | None = None)`: with a saved file for the id, apply and run everything; else phase A on the first `probe_tasks_per_gpu x world_size` pending jobs, summarize, tune, persist under the id, apply, phase B on the rest. The rollout report gains a Tuning section: the config with its notes and the phase A against phase B metrics. Re-tuning is a new id.

Validation: CPU tests of the rules on synthetic metrics (each rule fires on the intended condition and is monotone), the config id and its persistence round trip, and a manager test with the tiny fixture and a fake stat series that runs two phases and applies the pool size.

#### Planned Changes

`activation/agent/rollout_tuning.py · RolloutTuningConfig, ProbeMetrics, Autotuner`

```diff
+@dataclass
+class RolloutTuningConfig:
+    model_id: str; gpu_name: str; gpu_count: int
+    agents_per_gpu: int = 40
+    max_num_batched_tokens: int = 8192
+    gpu_memory_utilization: float = 0.9
+    speculative: bool = True
+    notes: list[str] = field(default_factory=list)
+    @property
+    def rollout_config_id(self) -> str: ...            # sha256 of the fields above, 12 hex chars
+    def apply(self, harness) -> None: ...              # pool bound now; engine kwargs -> reload if changed
+    def save(self, autotune_id: str) -> Path: ...      # TUNING/<autotune_id>.json
+    @classmethod
+    def load(cls, autotune_id: str) -> "RolloutTuningConfig | None": ...
+
+class Autotuner:
+    def tune(self, metrics: ProbeMetrics, defaults: RolloutTuningConfig) -> RolloutTuningConfig:
+        config = replace(defaults, notes=[])
+        per_gpu = int(0.8 * metrics.kv_tokens / max(metrics.p90_live_prefix, 1))
+        config.agents_per_gpu = max(16, min(96, per_gpu)); config.notes.append(f"kv budget: {per_gpu} ...")
+        if metrics.p95_kv_usage < 0.6 and metrics.sandbox_share > 0.25: ...
+        ...
+        return config
```

`activation/agent/rollout_manager.py · perform_single_rollouts`

```diff
-    def perform_single_rollouts(self, configs, seed, caching_id, perform_scoring=False, reporter=None):
+    def perform_single_rollouts(self, configs, seed, caching_id, perform_scoring=False, reporter=None,
+                                autotune_id: str | None = None, probe_tasks_per_gpu: int = 200):
         ...
+        if autotune_id is not None:
+            saved = RolloutTuningConfig.load(autotune_id)
+            if saved is None:
+                probe, rest = pending[:probe_tasks_per_gpu * world_size], pending[probe_tasks_per_gpu * world_size:]
+                self._run_pending(probe, ...)                                   # phase A under the defaults, metrics on
+                config = Autotuner().tune(ProbeMetrics.summarize(self.harness.metrics(), self.host_samples, results), defaults)
+                config.save(autotune_id); pending = rest
+            (saved or config).apply(self.harness)
+        self._run_pending(pending, ...)                                          # phase B, or everything
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">T2 — Tuning run on 2x RTX PRO 6000</span>
    <span class="card-oneliner">200 tasks per GPU under the defaults, then the same under the tuned configuration; TUNING_REPORT with both phases.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** Two runs on 2x RTX PRO 6000 (g7e.12xlarge): `ac-4a-tune` (seed 0, 804 rows, autotune id `campaign_27b_medium_v1`, spoiled by the index storm) and `ac-4a-tune2` (seed 1, shuffled, 804 rows, id `campaign_27b_medium_v2`). `TUNING_REPORT.tressoir.md` reads the v2 probe.

**Measured under the defaults (seed 1, full mix, 40 agents per GPU).** Whole run 277 tasks per hour per GPU; pool-full plateau about 800 tasks per hour on the node; KV usage p95 84%; 59% of prompt tokens recomputed (prefix eviction); 45% of engine intervals with a waiting queue; GPU utilization 98.5%; prefill 95% of the batch; decode 28 tokens per second per replica; mean score 0.72.

**Tuned configuration saved as `campaign_27b_medium_v2`.** 33 agents per GPU, batched tokens 16,384, memory utilization 0.92, speculative decoding off. The tuned phase of that run measured one task (straggler, see T1), so the gain is not yet measured: the 12 h campaign run applies the file from its first task and its first hour is compared with the v2 defaults phase; if worse, it is relaunched under the defaults and the cache resumes it.

**Drifts.** The planned "same 400 under the tuned configuration" comparison moved into the campaign run; the seed-0 repair and a seed-2 remeasure were skipped because the prompt reorganization changed the cache keys.

#### Planning Overview (as planned)

The tuning run itself, after T0 and T1 land and your decisions are in: a 2x RTX PRO 6000 node (D4), the suite at medium effort with the campaign's caching id, a fresh `autotune_id`, 200 tasks per GPU in the probe phase and the same count after (800 tasks in all on two GPUs). Deliverable: `TUNING_REPORT.tressoir.md` with the chosen configuration and its notes, the metric series of both phases as small inline charts, and a table of phase A against phase B: tasks per hour per GPU, generated tokens per hour per GPU, prompt recompute share, KV usage p50 and p95, waiting-queue share, sandbox share, run duration p95, and the score per benchmark as a check that nothing shifted. Because every rollout is cached under the campaign id at medium effort, the 800 trajectories are the first 800 of the campaign.

</details>

## Status log

- 2026-09-10 v0.3.1: per-rollout duration cap back to 1,800 s (your call in chat); the campaign study and the canonical script default to it, `--max-duration` still overrides. The 3,600 s cap in the accepted decisions above applied to the tuning runs only.
- 2026-09-10 v0.3: T0, T1, T2 in Review. Second tuning run (`ac-4a-tune2`, seed 1, shuffled) measured the defaults on the full mix and saved `campaign_27b_medium_v2`; the manager became a continuous queue (probe on the first N finished, in-place pool resize, deferred engine reload, wall-clock deadline); BM25 builds locked and prebuilt; repository indexes dropped; `TUNING_REPORT.tressoir.md` published. The tuned-versus-default comparison is taken from the 12 h campaign run.
- 2026-09-10 v0.2: green light in chat with the recommended answers (200 per GPU, throughput knobs only, one reload allowed, 2x RTX PRO 6000); T0, T1 implemented, T2 running as `ac-4a-tune` (autotune id `campaign_27b_medium_v1`, caching id `teacher_campaign_medium`). Also landed with it: `think_end` on trajectory steps and the strip/keep rule in the training example builder.
- 2026-09-10 v0.1.1: reuse settled in chat: `autotune_id` replaces the boolean, same id reuses the saved configuration regardless of pool or node.
- 2026-09-10 v0.1: intent, knobs, data flow, five decisions; T0 and T1 planned with Moderate diffs, T2 an overview.
