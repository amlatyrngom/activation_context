"""
Single and grouped rollouts with varying seeds. Rollouts always occur through the engine: every agent
runs on its own thread and submits one request per turn; the engine batches across agents (no
synchronous batching of rollouts). One engine is resident at a time, as before, and raw HF models of
other names stay loaded next to it.

Slice 1 is full end-to-end rollouts plus caching (no rejection sampling). Slice 2 adds trajectory
harvesting and SFT.
"""
from __future__ import annotations

import typing as t
from concurrent.futures import ThreadPoolExecutor, as_completed

from .agent import Agent
from .agent_config import AgentConfig, AgentRunResult
from .rollout_caching import RolloutCache, config_key

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime
    from .rollout_reporter import RolloutReporter


class RolloutManager:
    def __init__(self, harness: "HarnessRuntime"):
        self.harness = harness

    def perform_single_rollouts(
        self,
        agent_configs: list[AgentConfig],
        seed: int = 0,
        caching_id: str | None = None,   # Used to save runs in a safe way.
        perform_scoring: bool = True,
        reporter: "RolloutReporter | None" = None,
    ) -> list[AgentRunResult]:
        return self._run_jobs([(config, seed) for config in agent_configs], caching_id, perform_scoring, reporter)

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
    def _run_jobs(self, jobs: list[tuple[AgentConfig, int]], caching_id, perform_scoring, reporter) -> list[AgentRunResult]:
        cache = RolloutCache(caching_id)
        results: list[AgentRunResult | None] = []
        for config, seed in jobs:
            row = cache.get(config_key(config), seed)
            results.append(None if row is None else AgentRunResult.deserialize(row, self.harness, agent_config=config))
        pending = [index for index, result in enumerate(results) if result is None]
        if reporter is not None:
            reporter.begin_rollouts(total=len(jobs), cached=len(jobs) - len(pending))
        if pending:
            model_names = {jobs[index][0].model_name for index in pending}
            assert len(model_names) == 1, f"one engine is resident at a time; configs name {sorted(map(str, model_names))}"
            self.harness.loaded_models[model_names.pop()].ensure_engine_loaded()   # once, before the threads
            with ThreadPoolExecutor(max_workers=self.harness.harness_config.agent_max_concurrent) as pool:
                futures = {pool.submit(self._run_one, jobs[index][0], jobs[index][1], perform_scoring, reporter): index for index in pending}
                for future in as_completed(futures):
                    result = future.result()
                    results[futures[future]] = result
                    cache.append(result)
        if reporter is not None:
            reporter.finish()
        return t.cast(list[AgentRunResult], results)

    def _run_one(self, config: AgentConfig, seed: int, perform_scoring: bool, reporter) -> AgentRunResult:
        agent = Agent(self.harness, config, reporter=reporter, seed=seed)
        try:
            result = agent.run()
            if perform_scoring and config.dataset_task is not None:
                agent.score()
                if reporter is not None:
                    reporter.report_agent_scored(agent)
        finally:
            agent.shutdown()
        return result
