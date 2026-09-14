"""
Single and grouped rollouts with varying seeds. Rollouts always occur through the engine: every agent
runs on its own thread and submits one request per turn; the engine batches across agents (no
synchronous batching of rollouts). One engine is resident at a time, as before, and raw HF models of
other names stay loaded next to it.

Slice 1 is full end-to-end rollouts plus caching (no rejection sampling). Slice 2 adds trajectory
harvesting and SFT.
"""
from __future__ import annotations

import random
import shutil
import subprocess
import threading
import time
import typing as t
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace

from .agent import Agent
from .agent_config import AgentConfig, AgentRunResult
from .rollout_caching import RedoPolicy, RolloutCache, config_key
from .rollout_tuning import Autotuner, ProbeMetrics, RolloutTuningConfig

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime
    from .rollout_reporter import RolloutReporter


class RolloutManager:
    """
    Wall-clock deadline (`max_wall_seconds` on perform_single_rollouts): measured from the start of `_run_pending`. Once
    it passes, no new job is started (a job that acquires the pool after the deadline is skipped and its result stays
    None), and every job that does start gets its `max_duration` clipped so it cannot run past the deadline: it runs
    under a copy of its config with `max_duration=min(max_duration, remaining)`, where remaining is the time left, at
    least one second (or the whole budget when that is shorter). A run cut by the clipped bound ends with finish_reason
    "deadline" instead of "max_duration". With a deadline set, perform_single_rollouts returns only the finished results,
    in job order; the grouped rollouts take no deadline.
    """

    def __init__(self, harness: "HarnessRuntime"):
        self.harness = harness
        self.tuning_report: dict | None = None   # the last pass's tuning: config, phases (see rollout_tuning.render_tuning_section)
        self.host_samples: list[dict] = []

    def pool_size(self, world_size: int) -> int:
        """Rollout threads: an explicit total bound, else agents per GPU times the engine's replicas."""
        config = self.harness.harness_config
        return int(config.agent_max_concurrent) if config.agent_max_concurrent else int(config.agent_max_concurrent_per_gpu) * max(1, world_size)

    def perform_single_rollouts(
        self,
        agent_configs: list[AgentConfig],
        seed: int = 0,
        caching_id: str | None = None,   # Used to save runs in a safe way.
        perform_scoring: bool = True,
        reporter: "RolloutReporter | None" = None,
        autotune_id: str | None = None,  # None: run under the current settings. An id: apply its saved tuning, or tune from a probe phase and save it.
        probe_tasks_per_gpu: int = 200,
        max_wall_seconds: float | None = None,   # Wall-clock deadline (see the class docstring); with it, only the finished results come back, in job order.
        redo: "RedoPolicy | None" = None,        # Cached rows to roll out again (their new rows supersede the old ones); `only`: nothing else runs.
        agents_per_gpu: int | None = None,       # Overrides the pool bound of the saved tuning (the engine arguments still apply).
    ) -> list[AgentRunResult]:
        results = self._run_jobs([(config, seed) for config in agent_configs], caching_id, perform_scoring, reporter,
                                 autotune_id=autotune_id, probe_tasks_per_gpu=probe_tasks_per_gpu, max_wall_seconds=max_wall_seconds,
                                 redo=redo, agents_per_gpu=agents_per_gpu)
        if max_wall_seconds is None and not (redo is not None and redo.only):
            return results
        return [result for result in results if result is not None]

    def perform_grouped_rollouts(
        self,
        agent_configs: list[AgentConfig],
        group_count: int,
        base_seed: int = 0,
        perform_scoring: bool = True,
        caching_id: str | None = None,   # Used to save runs: a group growing from 4 to 6 reuses the 4.
        reporter: "RolloutReporter | None" = None,
    ) -> list[list[AgentRunResult]]:
        jobs = [(config, base_seed + member) for config in agent_configs for member in range(group_count)]
        flat = self._run_jobs(jobs, caching_id, perform_scoring, reporter)
        return [flat[index:index + group_count] for index in range(0, len(flat), group_count)]

    # ----------------------------------------------------------------------------- internals
    def _run_jobs(self, jobs: list[tuple[AgentConfig, int]], caching_id, perform_scoring, reporter,
                  autotune_id: str | None = None, probe_tasks_per_gpu: int = 200, max_wall_seconds: float | None = None,
                  redo: "RedoPolicy | None" = None, agents_per_gpu: int | None = None) -> list[AgentRunResult]:
        cache = RolloutCache(caching_id, redo=redo)
        results: list[AgentRunResult | None] = []
        for config, seed in jobs:
            row = cache.get(config_key(config), seed)
            results.append(None if row is None else AgentRunResult.deserialize(row, self.harness, agent_config=config))
        pending = [index for index, result in enumerate(results) if result is None]
        skipped = 0
        if redo is not None and redo.only:                                       # the rows to redo alone; never-cached tasks stay None
            stale = [index for index in pending if cache.is_stale(config_key(jobs[index][0]), jobs[index][1])]
            skipped = len(pending) - len(stale)
            pending = stale
            print(f"RolloutManager - redo only: {len(pending)} cached rollouts to redo, {skipped} tasks without a cached row skipped.", flush=True)
        if reporter is not None:
            reporter.begin_rollouts(total=len(jobs) - skipped, cached=len(jobs) - skipped - len(pending))
        self.tuning_report = None
        if pending:
            model_names = {jobs[index][0].model_name for index in pending}
            assert len(model_names) == 1, f"one engine is resident at a time; configs name {sorted(map(str, model_names))}"
            loaded_model = self.harness.loaded_models[model_names.pop()]
            loaded_model.ensure_engine_loaded()                                  # once, before the threads
            world_size = loaded_model.vllm_model.world_size if loaded_model.vllm_model is not None else 1
            self._build_indexes(jobs, pending)                                   # once, before the threads
            self._warm_up(loaded_model, jobs[pending[0]][0], world_size)
            run = lambda indices, label, **kwargs: self._run_pending(indices, jobs, results, cache, perform_scoring, reporter, label,
                                                                     max_wall_seconds=max_wall_seconds, **kwargs)
            if autotune_id is None:
                run(pending, "rollouts")
            else:
                order = list(pending)
                random.Random(0).shuffle(order)      # every benchmark from the start: the probe sees the mix, and so does the host
                defaults = RolloutTuningConfig.defaults(loaded_model, self.harness)
                saved = RolloutTuningConfig.load(autotune_id)
                state: dict = {"config": saved, "reloaded": False, "deferred": False}
                if saved is not None:
                    state["reloaded"] = saved.apply(self.harness, loaded_model)
                    if agents_per_gpu is not None:                                   # the pool bound only; the engine keeps the saved arguments
                        self.harness.harness_config.agent_max_concurrent_per_gpu = int(agents_per_gpu)
                        self.harness.harness_config.agent_max_concurrent = None
                        print(f"RolloutManager - pool bound overridden: {agents_per_gpu} agents per GPU (saved tuning: {saved.agents_per_gpu}).", flush=True)
                    phases = run(order, "saved")
                else:
                    def tune(metrics: ProbeMetrics) -> int:
                        config = Autotuner().tune(metrics, defaults)
                        path = config.save(autotune_id)
                        print(f"RolloutManager - autotune {autotune_id}: {config.agents_per_gpu} agents per GPU; saved {path}", flush=True)
                        for note in config.notes:
                            print(f"RolloutManager -   {note}", flush=True)
                        state["config"] = config
                        state["deferred"] = bool(config.engine_kwargs_delta(loaded_model))   # agents are in flight: the next run reloads
                        config.apply(self.harness, loaded_model, reload=False)
                        return self.pool_size(world_size)
                    phases = run(order, "probe", probe_count=probe_tasks_per_gpu * world_size, on_probe=tune)
                config = state["config"]
                self.tuning_report = {"autotune_id": autotune_id, "from_saved": saved is not None, "reloaded": state["reloaded"],
                                      "reload_deferred": state["deferred"],
                                      "config": asdict(config) | {"rollout_config_id": config.rollout_config_id},
                                      "defaults": asdict(defaults), "phases": phases, "max_wall_seconds": max_wall_seconds}
        if reporter is not None:
            reporter.finish()
        return t.cast(list[AgentRunResult], results)

    def _run_pending(self, indices: list[int], jobs, results, cache, perform_scoring, reporter, label: str,
                     probe_count: int | None = None, on_probe: "t.Callable[[ProbeMetrics], int] | None" = None,
                     max_wall_seconds: float | None = None) -> list[dict]:
        """
        One continuously fed queue over `indices` in order, at most pool_size agents at a time. With `probe_count` and
        `on_probe`, the moment `probe_count` tasks have finished (whichever they are: waiting for the first submitted ones
        would let one straggler delay the tuning by an hour) the metrics so far go to
        `on_probe`, which returns the new pool bound; the queue never drains in between. Returns one ProbeMetrics
        dict per phase (`label`, then "tuned" once the bound changed). `max_wall_seconds` is the deadline of the class
        docstring, counted from here: jobs reaching the pool after it are skipped (their `results` entry stays None).
        """
        loaded_model = self.harness.loaded_models[jobs[indices[0]][0].model_name] if indices else None
        world_size = loaded_model.vllm_model.world_size if loaded_model is not None and loaded_model.vllm_model is not None else 1
        limiter = _PoolLimiter(self.pool_size(world_size))
        sampler = _HostSampler()
        sampler.start()
        started = time.time()
        deadline = started + max_wall_seconds if max_wall_seconds is not None else None
        counts = {"started": 0, "skipped": 0}
        counts_lock = threading.Lock()
        probe = set(indices[:probe_count]) if probe_count else set()
        finished: dict[int, tuple[float, AgentRunResult]] = {}
        marks: list[tuple[str, float]] = [(label, started)]
        phases: list[dict] = []

        def work(position: int, index: int) -> AgentRunResult | None:
            limiter.acquire(position)
            try:
                config, seed = jobs[index]
                if deadline is None:
                    return self._run_one(config, seed, perform_scoring, reporter)
                now = time.time()
                with counts_lock:
                    counts["skipped" if now >= deadline else "started"] += 1
                if now >= deadline:
                    return None
                remaining = max(deadline - now, min(1.0, max_wall_seconds))
                clipped = min(float(config.max_duration), remaining)
                run_config = replace(config, max_duration=clipped) if clipped < config.max_duration else config
                result = self._run_one(run_config, seed, perform_scoring, reporter)
                if run_config is not config:
                    if result.finish_reason == "max_duration":
                        result.finish_reason = "deadline"
                    if result.agent_config is run_config:
                        result.agent_config = config           # the clip is an execution detail: the result belongs to the job's own config
                return result
            finally:
                limiter.release()

        try:
            if indices:
                with ThreadPoolExecutor(max_workers=max(limiter.limit, Autotuner.max_agents * world_size)) as pool:
                    futures = {pool.submit(work, position, index): index for position, index in enumerate(indices)}
                    for future in as_completed(futures):
                        index = futures[future]
                        result = future.result()
                        if result is None:                     # skipped at the deadline
                            continue
                        results[index] = result
                        finished[index] = (time.time(), result)
                        if result.finish_reason != "deadline":        # a cut run is not a result: the next launch redoes it
                            cache.append(result)
                        if probe and on_probe is not None and len(finished) >= len(probe):   # the first N to finish: a straggler must not delay the tuning by its whole duration
                            now = time.time()
                            phases.append(self._phase_metrics(label, started, now, loaded_model, sampler.samples, finished))
                            limiter.set_limit(on_probe(ProbeMetrics(**phases[-1])))
                            marks.append(("tuned", now))
                            probe = set()
        finally:
            sampler.stop()
        end = time.time()
        for position, (phase_label, phase_start) in enumerate(marks[len(phases):], start=len(phases)):
            phase_end = marks[position + 1][1] if position + 1 < len(marks) else end
            phases.append(self._phase_metrics(phase_label, phase_start, phase_end, loaded_model, sampler.samples, finished))
        if deadline is not None and (counts["skipped"] or end >= deadline):
            print(f"RolloutManager - deadline reached after {end - started:.0f}s: {counts['started']} started, {counts['skipped']} skipped", flush=True)
        return phases

    def _phase_metrics(self, label: str, start: float, end: float, loaded_model, host_samples: list[dict], finished: dict) -> dict:
        """ProbeMetrics over the window: the engine and host samples taken in it and the runs that finished in it."""
        engine_metrics = [replica | {"samples": [s for s in replica["samples"] if start <= s["t"] < end]}
                          for replica in (loaded_model.engine_metrics() if loaded_model is not None else [])]
        host = [s for s in host_samples if start <= s["t"] < end]
        phase_results = [result for finished_at, result in finished.values() if start <= finished_at < end]
        self.host_samples = host_samples
        metrics = ProbeMetrics.summarize(label, engine_metrics, host, phase_results, end - start)
        print(f"RolloutManager - phase {label}: {len(phase_results)} runs in {end - start:.0f}s, {metrics.tasks_per_hour_per_gpu:.0f} tasks/h/GPU, "
              f"kv p95 {metrics.p95_kv_usage:.0%}, hits {metrics.prefix_hit_share:.0%}, sandbox {metrics.sandbox_share:.0%}", flush=True)
        return asdict(metrics)

    def _build_indexes(self, jobs, pending: list[int]) -> None:
        """
        The bm25 index of every corpus the pending tasks can search, built once before the threads: a 300k-chunk code
        corpus takes about a minute alone, and 80 threads each building their own on first search took over an hour.
        """
        manager = self.harness.dataset_manager
        dataset_ids = {jobs[index][0].dataset_task.dataset_id for index in pending if jobs[index][0].dataset_task is not None}
        for dataset_id in sorted(dataset_ids):
            loaded = manager.loaded_datasets.get(dataset_id)
            if loaded is not None and loaded.documents:
                manager._get_or_create_index(dataset_id).build_bm25_index()

    def _warm_up(self, loaded_model, config: AgentConfig, world_size: int) -> None:
        """One tiny request per replica with the first job's initial prompt: the shared system prompt and tool list get cached before the herd."""
        try:
            agent = Agent(self.harness, config)
            agent.begin()
            prefix = list(agent.prefix)
            agent.shutdown()
            chat_kwargs = dict(config.call_kwargs or {})
            chat_kwargs["sampling_params"] = dict(chat_kwargs.get("sampling_params") or {}) | {"max_tokens": 1}
            for replica in range(max(1, world_size)):
                loaded_model.engine_submit_tokens(prefix, seed=0, agent_id=f"warmup-{replica}", lora_name=config.lora_name,
                                                  chat_kwargs=chat_kwargs, replica=replica)
        except Exception as error:                                              # a warm-up must never fail a pass
            print(f"RolloutManager - warm-up skipped: {type(error).__name__}: {error}", flush=True)

    def _run_one(self, config: AgentConfig, seed: int, perform_scoring: bool, reporter) -> AgentRunResult:
        agent = Agent(self.harness, config, reporter=reporter, seed=seed)
        try:
            result = agent.run_program()                                    # the config's agentic program, or the model loop
            if perform_scoring and config.dataset_task is not None:
                agent.score()
                if reporter is not None:
                    reporter.report_agent_scored(agent)
        finally:
            agent.shutdown()
        return result


class _PoolLimiter:
    """
    At most `limit` agents run at once; the bound can change while the queue runs (waiting threads are re-checked).
    Slots are handed out strictly by queue position (0, 1, 2, ...), not to whichever waiter wakes first: the queue really
    runs in order, so the first `probe_count` jobs are the first to finish and the deadline skips the tail, not a random subset.
    """

    def __init__(self, limit: int):
        self.limit = max(1, int(limit))
        self.running = 0
        self._serving = 0                                       # the next position that may take a slot
        self._condition = threading.Condition()

    def acquire(self, position: int) -> None:
        with self._condition:
            while position != self._serving or self.running >= self.limit:
                self._condition.wait()
            self._serving += 1
            self.running += 1
            self._condition.notify_all()                        # the next position may be waiting behind this one

    def release(self) -> None:
        with self._condition:
            self.running -= 1
            self._condition.notify_all()

    def set_limit(self, limit: int) -> None:
        with self._condition:
            self.limit = max(1, int(limit))
            self._condition.notify_all()


class _HostSampler:
    """GPU utilization, host memory, host CPU and live sandbox count every `interval` seconds while a phase runs."""
    interval = 5.0

    def __init__(self):
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="rollout-host-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval + 1)

    def _loop(self) -> None:
        import psutil
        psutil.cpu_percent(interval=None)
        nvidia = shutil.which("nvidia-smi")
        podman = shutil.which("podman")
        while not self._stop.is_set():
            sample = {"t": time.time(), "host_cpu": psutil.cpu_percent(interval=None) / 100.0,
                      "host_memory": psutil.virtual_memory().percent / 100.0, "gpu_utilization": 0.0, "gpu_memory": 0.0, "containers": 0}
            if nvidia:
                try:
                    out = subprocess.run([nvidia, "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=4).stdout.strip().splitlines()
                    rows = [[float(x) for x in line.split(",")] for line in out if line.strip()]
                    if rows:
                        sample["gpu_utilization"] = sum(r[0] for r in rows) / len(rows) / 100.0
                        sample["gpu_memory"] = sum(r[1] for r in rows) / max(sum(r[2] for r in rows), 1.0)
                except Exception:
                    pass
            if podman:
                try:
                    out = subprocess.run([podman, "ps", "-q"], capture_output=True, text=True, timeout=4).stdout
                    sample["containers"] = len([line for line in out.splitlines() if line.strip()])
                except Exception:
                    pass
            self.samples.append(sample)
            self._stop.wait(self.interval)
