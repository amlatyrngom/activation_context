"""
Slice 2 key tests: one training round on grouped rollouts, then two rounds with an exchange between
them. Small on purpose (16 problems x group 8 on Qwen3.5-4B, ~128 rollouts per phase); the asserts are
about mechanics (segments recorded, group-mean weights, step-1 ratio of 1, checkpoint, exchange seen by
the engine), the accuracies are printed.

  uv run pytest activation/tests/test_basic_agent_training.py --gpu --slow -s
"""
import json
import os
import random
from dataclasses import replace

import pytest

from activation.agent import AgentConfig, RolloutReporter
from activation.agent_training import AgentTrainer, AgentTrainingConfig, AgentTrainingReporter, group_mean_advantage
from activation.common.data_syncing import resolve_path
from activation.dataset.loaders import DapoMathDataset
from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig

MODEL_NAME, MODEL_ID, LORA = "qwen3.5-4b", "Qwen/Qwen3.5-4B", "dapo_group_mean"
PROBLEMS, GROUP = 16, 8

BASE = AgentConfig(
    system_prompt=("You are a careful problem solver working in a sandbox. Use the python or shell tools to compute "
                   "rather than guessing. When you are done, call submit_answer exactly once with the final answer."),
    model_name=MODEL_NAME, lora_name=LORA,
    call_kwargs={"sampling_params": {"max_tokens": 2048, "temperature": 0.7}},
    max_turns=8, max_tool_errors=3, max_duration=300,
)


def _harness() -> HarnessRuntime:
    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={MODEL_NAME: ModelConfig(MODEL_NAME, MODEL_ID)}, agent_max_concurrent=64))
    harness.module_manager.register_lora(LORA, MODEL_NAME, rank=64)          # zero-init B: round 0 samples are the base model's
    return harness


def _tasks(harness):
    dataset = DapoMathDataset.load(harness, max_examples=1000)
    harness.dataset_manager.register_dataset(dataset)
    tasks = list(dataset.scorable_tasks.values())
    random.Random(0).shuffle(tasks)
    return tasks[:PROBLEMS]


def _selection_function(group):
    return group_mean_advantage(group, model_name=MODEL_NAME, lora_name=LORA)


def _rollouts(harness, configs, round_index, base_seed, prefix="single"):
    reporter = RolloutReporter(str(resolve_path(f"AGENT_TRAINING_TEST/{prefix}_rollouts_round{round_index}")), f"training test ({prefix}), rollouts round {round_index}")
    return harness.rollout_manager.perform_grouped_rollouts(configs, group_count=GROUP, base_seed=base_seed, perform_scoring=True,
                                                            caching_id=f"agent_training_test_{prefix}_r{round_index}", reporter=reporter)


def _accuracy(groups):
    scores = [r.score for g in groups for r in g]
    return sum(scores) / len(scores)


def _release(harness):
    """Adapters off the base (their checkpoints are on disk), then the base and the engine off the GPU."""
    loaded = harness.loaded_models[MODEL_NAME]
    loaded.engine_to_device(FREE_DEVICE)
    harness.module_manager.free_lora(MODEL_NAME, LORA)
    loaded.model_to_device(FREE_DEVICE)


@pytest.mark.gpu
@pytest.mark.slow
def test_basic_agent_training():
    """Roll out, weight by group-mean advantage, train one round, exchange, roll out the same problems again with the adapter."""
    harness = _harness()
    tasks = _tasks(harness)
    configs = [replace(BASE, user_prompt=t.agent_prompt, dataset_task=t) for t in tasks]
    trainer = AgentTrainer(harness, AgentTrainingConfig(updates_per_round=2))
    try:
        groups = _rollouts(harness, configs, 0, base_seed=0)
        for r in (r for g in groups for r in g):
            assert r.prompt_token_ids and all(step["token_ids"] for step in r.trajectory), "segments not recorded"
            assert all(0 < len(step["logprobs"]) <= len(step["token_ids"]) for step in r.trajectory if step["role"] == "assistant")
            assert r.lora_name is None, "round 0 runs the base model (nothing exchanged yet)"
        selected = trainer.select_agent_runs(groups, selection_function=_selection_function)
        items = selected[(MODEL_NAME, LORA)]
        assert items, "no mixed group in 16 x 8 (would be very unlucky at ~75% accuracy)"
        assert any(item.weight > 0 for item in items) and any(item.weight < 0 for item in items)
        by_group = {}
        for item in items:
            by_group.setdefault(item.group_key, []).append(item.weight)
        assert all(abs(sum(weights)) < 1e-6 for weights in by_group.values()), "group-mean weights sum to zero per group"
        split = max(1, int(len(items) * 0.9))
        training_data, reporting_data = items[:split], items[split:]
        reporter = AgentTrainingReporter(str(resolve_path("AGENT_TRAINING_TEST/single_train_round0")), "training test (single), round 0")
        stats = trainer.train(MODEL_NAME, LORA, training_data, reporting_data, reporter)     # sleeps the engine
        print(json.dumps(stats.summarize(), indent=1))
        assert stats.steps == 2 and stats.examples > 0
        assert stats.examples + stats.dropped_too_long + stats.dropped_empty == len(training_data)
        assert 0.9 < stats.mean_ratio[0] < 1.1, stats.mean_ratio          # step 1: trainer and engine agree on pi_old (decision 3)
        assert stats.clip_fraction[0] < 0.15, stats.clip_fraction    # bf16 engine (fp8 KV cache) vs bf16 trainer: a few % of tokens land outside the clip
        assert stats.checkpoint_path and os.path.isdir(stats.checkpoint_path)
        harness.module_manager.exchange_lora(MODEL_NAME, LORA)
        again = _rollouts(harness, configs, 1, base_seed=1000)             # fresh seeds: not the cache, the adapter; wakes the engine
        assert all(r.lora_name == LORA for g in again for r in g)
        print(f"accuracy before {_accuracy(groups):.3f} -> after one round {_accuracy(again):.3f} (16 x 8; printed, not asserted)")
    finally:
        _release(harness)
    assert os.path.exists(os.path.join(reporter.report_folder, "report.tressoir.html"))


@pytest.mark.gpu
@pytest.mark.slow
def test_basic_agent_training_multi_rounds():
    """Two rounds: rollouts -> train -> exchange -> rollouts with the adapter -> train -> exchange -> rollouts."""
    harness = _harness()
    tasks = _tasks(harness)
    configs = [replace(BASE, user_prompt=t.agent_prompt, dataset_task=t) for t in tasks]
    trainer = AgentTrainer(harness, AgentTrainingConfig(updates_per_round=2))
    accuracies, ratios = [], []
    try:
        for round_index in range(2):
            groups = _rollouts(harness, configs, round_index, base_seed=10_000 * (round_index + 1), prefix="multi")
            accuracies.append(_accuracy(groups))
            if round_index > 0:
                assert all(r.lora_name == LORA for g in groups for r in g)
            items = trainer.select_agent_runs(groups, _selection_function)[(MODEL_NAME, LORA)]
            reporter = AgentTrainingReporter(str(resolve_path(f"AGENT_TRAINING_TEST/multi_train_round{round_index}")), f"training test (multi), round {round_index}")
            stats = trainer.train(MODEL_NAME, LORA, items, reporter=reporter)
            ratios.append(stats.mean_ratio[0])
            harness.module_manager.exchange_lora(MODEL_NAME, LORA)
        final = _rollouts(harness, configs, 2, base_seed=30_000, prefix="multi")
        accuracies.append(_accuracy(final))
    finally:
        _release(harness)
    print(f"accuracy by round {[round(a, 3) for a in accuracies]}; step-1 ratios {[round(r, 3) for r in ratios]}")
    assert all(0.9 < r < 1.1 for r in ratios), ratios                     # round 2's samples came from the exchanged adapter
    checkpoints = sorted(os.listdir(resolve_path(f"LORAS/{LORA}", create=False)))
    assert "round_000" in checkpoints and "round_001" in checkpoints and "latest" in checkpoints
