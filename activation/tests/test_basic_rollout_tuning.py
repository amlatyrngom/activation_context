"""
Small checks of the rollout tuning subplan (T0/T1) and the thinking-span handling: the rules fire on their conditions,
the configuration round-trips by autotune id, the pool bound follows the replicas, the agent records think_end, and
the training example builder strips or keeps the span.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from activation.agent import AgentConfig
from activation.agent import agent as agent_module
from activation.agent.rollout_manager import RolloutManager
from activation.agent.rollout_tuning import Autotuner, ProbeMetrics, RolloutTuningConfig, render_tuning_section
from activation.agent_training.agent_training_utils import build_example, think_replacement_ids
from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig

TINY = os.environ.get("TINY_QWEN35", "/workspace/IB/TMP/AGENT_ROLLOUTS/slice3_apply_20260909T103843Z/tiny_qwen35")


def _metrics(**overrides) -> ProbeMetrics:
    base = dict(label="probe", wall_seconds=600.0, runs=100, gpu_count=1, kv_tokens=1_230_000, p50_kv_usage=0.5, p95_kv_usage=0.7,
                waiting_share=0.0, preempted=0, prefill_share=0.5, prefix_recompute_share=0.1, prefix_hit_share=0.75,
                p50_live_prefix=15_000, p90_live_prefix=30_000, p50_turn_seconds=8.0, p90_turn_seconds=20.0, sandbox_share=0.1,
                p50_gpu_utilization=0.8, p95_host_memory=0.5, p95_host_cpu=0.4, tasks_per_hour_per_gpu=120.0,
                generated_tokens_per_hour_per_gpu=800_000.0, generation_tokens_per_second=200.0, p95_run_seconds=900.0, mean_score=0.7)
    return ProbeMetrics(**(base | overrides))


def _defaults() -> RolloutTuningConfig:
    return RolloutTuningConfig(model_id="m", gpu_name="g", gpu_count=1)


def test_rules_fire_on_their_conditions():
    tuner = Autotuner()
    quiet = tuner.tune(_metrics(p95_host_memory=0.9), _defaults())
    assert quiet.agents_per_gpu == 32 and quiet.max_num_batched_tokens is None and quiet.speculative and quiet.gpu_memory_utilization == 0.9   # 0.8 x 1.23M / 30k
    starved = tuner.tune(_metrics(p95_kv_usage=0.4, sandbox_share=0.4, p95_host_memory=0.9), _defaults())
    assert starved.agents_per_gpu == 40                                                                                   # 32 x 1.25
    evicting = tuner.tune(_metrics(prefix_recompute_share=0.3, p95_host_memory=0.9), _defaults())
    assert evicting.agents_per_gpu == 24                                                                                  # 32 x 0.75
    prefill = tuner.tune(_metrics(prefill_share=0.8), _defaults())
    assert prefill.max_num_batched_tokens == 16_384 and prefill.gpu_memory_utilization == 0.92                            # headroom rule too
    saturated = tuner.tune(_metrics(p95_kv_usage=0.9, generation_tokens_per_second=20.0), _defaults())
    assert saturated.speculative is False
    unknown = tuner.tune(_metrics(kv_tokens=None), _defaults())
    assert unknown.agents_per_gpu == 40 and unknown.notes[0].startswith("kv budget unknown")
    assert Autotuner.min_agents <= tuner.tune(_metrics(p90_live_prefix=200_000), _defaults()).agents_per_gpu
    assert tuner.tune(_metrics(p90_live_prefix=1_000), _defaults()).agents_per_gpu == Autotuner.max_agents


def test_config_round_trip_by_autotune_id(tmp_path, monkeypatch):
    monkeypatch.setenv("ACTIVATION_SYNC_ROOT", str(tmp_path))
    config = RolloutTuningConfig(model_id="m", gpu_name="g", gpu_count=2, agents_per_gpu=48, max_num_batched_tokens=16_384, notes=["x"])
    path = config.save("tune_a")
    assert path.name == "tune_a.json" and json.loads(path.read_text())["rollout_config_id"] == config.rollout_config_id
    assert RolloutTuningConfig.load("tune_a") == config and RolloutTuningConfig.load("tune_b") is None
    lines = render_tuning_section({"autotune_id": "tune_a", "config": json.loads(path.read_text()), "phases": [_metrics().__dict__, _metrics(label="tuned").__dict__]})
    assert any("48 agents per GPU" in line for line in lines) and any(line.startswith("| tasks_per_hour_per_gpu |") for line in lines)


@pytest.fixture(scope="module")
def harness():
    if "/" in TINY and TINY.startswith(("/", ".")) and not os.path.isdir(TINY):
        pytest.skip(f"tiny fixture missing: {TINY}")
    return HarnessRuntime(HarnessRuntimeConfig(model_configs={"tiny": ModelConfig("tiny", TINY)}))


def test_pool_size_follows_replicas(harness):
    manager = RolloutManager(harness)
    harness.harness_config.agent_max_concurrent_per_gpu = 40
    harness.harness_config.agent_max_concurrent = None
    assert manager.pool_size(1) == 40 and manager.pool_size(2) == 80
    harness.harness_config.agent_max_concurrent = 12
    assert manager.pool_size(2) == 12
    harness.harness_config.agent_max_concurrent = None


def test_think_end_recorded_and_stripped_or_kept(harness, monkeypatch):
    from dataclasses import dataclass
    from activation.agent.agent_utils import ModelDialect
    from activation.harness.loaded_model import EngineChatOutput

    tokenizer = harness.loaded_models["tiny"].tokenizer
    replacement = think_replacement_ids(tokenizer)
    think_end_id = replacement[-1]
    reasoning = tokenizer.encode("Let me think.", add_special_tokens=False)
    visible = tokenizer.encode("\n\nDone.", add_special_tokens=False)
    turn_tokens = reasoning + [think_end_id] + visible
    script = iter([("Done.", [{"id": "c1", "name": "submit_answer", "arguments": {"answer": "B"}}])])
    monkeypatch.setattr(agent_module.Agent, "_submit", lambda self, chat_kwargs=None: EngineChatOutput(
        text="", token_ids=list(turn_tokens), logprobs=[-0.5] * len(turn_tokens), prompt_token_count=len(self.prefix), output_token_count=len(turn_tokens)))
    monkeypatch.setattr(ModelDialect, "parse", lambda self, block, parameter_types=None: next(script))
    agent = agent_module.Agent(harness, AgentConfig(system_prompt="s", model_name="tiny", user_prompt="u", max_turns=3, record_sampling=True))
    results = agent.run()
    agent.shutdown()
    step = results.trajectory[0]
    assert step["role"] == "assistant" and step["think_end"] == len(reasoning) + 1 and step["timing"]["output_tokens"] == len(turn_tokens)
    assert results.trajectory[1]["timing"]["tool_seconds"] >= 0.0

    @dataclass
    class Item:
        run_results: object
        weight: float = 1.0
        ignore_logprobs: bool = False
        thinking: str = "keep"

    kept = build_example(Item(results), 0, replacement)
    prompt = len(results.prompt_token_ids)
    assert kept.token_ids[prompt:prompt + len(turn_tokens)] == turn_tokens and sum(kept.loss_mask[prompt:prompt + len(turn_tokens)]) == len(turn_tokens)
    stripped = build_example(Item(results, thinking="strip"), 0, replacement)
    assert stripped.token_ids[prompt:prompt + len(replacement) + len(visible)] == replacement + visible
    assert stripped.loss_mask[prompt:prompt + len(replacement)] == [False] * len(replacement)
    assert stripped.loss_mask[prompt + len(replacement):prompt + len(replacement) + len(visible)] == [True] * len(visible)
    assert stripped.old_logprobs[prompt + len(replacement)] == -0.5
    assert len(stripped.token_ids) == len(kept.token_ids) - len(reasoning) - 1 + len(replacement)
    with pytest.raises(ValueError):
        build_example(Item(results, thinking="strip"), 0, None)


def test_queue_keeps_running_and_retunes_mid_run(harness, monkeypatch):
    """The continuous queue: pool 2 until the first 4 submitted tasks have finished, then the tuner's answer (4) applies without a drain."""
    import threading
    import time
    from activation.agent.agent_config import AgentRunResult
    from activation.agent.rollout_manager import RolloutManager

    state = {"running": 0, "peak_before": 0, "peak_after": 0, "tuned": False}
    lock = threading.Lock()

    def fake_run_one(self, config, seed, perform_scoring, reporter):
        with lock:
            state["running"] += 1
            key = "peak_after" if state["tuned"] else "peak_before"
            state[key] = max(state[key], state["running"])
        time.sleep(0.03)
        with lock:
            state["running"] -= 1
        return AgentRunResult(agent_config=config, seed=seed, duration=0.03, finish_reason="submitted")

    monkeypatch.setattr(RolloutManager, "_run_one", fake_run_one)
    harness.harness_config.agent_max_concurrent = 2
    manager = RolloutManager(harness)
    jobs = [(AgentConfig(system_prompt="s", model_name="tiny", user_prompt=f"u{i}"), i) for i in range(16)]
    results: list = [None] * len(jobs)

    def on_probe(metrics: ProbeMetrics) -> int:
        assert metrics.label == "probe" and metrics.runs >= 4
        with lock:
            state["tuned"] = True
        harness.harness_config.agent_max_concurrent = 4
        return manager.pool_size(1)

    class NoCache:
        def append(self, result):
            pass

    phases = manager._run_pending(list(range(16)), jobs, results, NoCache(), False, None, "probe", probe_count=4, on_probe=on_probe)
    harness.harness_config.agent_max_concurrent = None
    assert all(result is not None for result in results)
    assert [phase["label"] for phase in phases] == ["probe", "tuned"] and phases[0]["runs"] + phases[1]["runs"] == 16
    assert state["peak_before"] <= 2 and state["peak_after"] == 4


def test_bm25_index_builds_once_under_concurrent_searchers(harness, monkeypatch):
    import threading
    from activation.dataset import dataset_index as index_module
    from activation.dataset.dataset import DatasetDocument, LoadedDataset
    from activation.dataset.dataset_utils import initialize_dataset_stats

    builds = []

    class SlowIndex:
        build_time = 0.0

        def __init__(self, texts):
            builds.append(len(texts))
            threading.Event().wait(0.05)

    monkeypatch.setattr(index_module, "BM25Index", SlowIndex)
    loaded = LoadedDataset(dataset_id="d", documents={f"d/{i}": DatasetDocument(doc_id=f"d/{i}", dataset_id="d", text=f"text {i}") for i in range(3)})
    loaded.stats = initialize_dataset_stats(loaded, load_time=0.0)
    index = index_module.DatasetIndex(harness, loaded)
    threads = [threading.Thread(target=index.build_bm25_index) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert builds == [len(index.chunks)] and index.bm25_index is not None
