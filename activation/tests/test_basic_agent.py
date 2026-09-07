"""
Two end-to-end agent rollouts on a GPU node.

    uv run sky exec --sync <node> -- uv run pytest activation/tests/test_basic_agent.py --gpu --slow -s

The reporters write under the synced folder, so IB/TMP/SYNC/AGENT_TEST/{primes,dapo}/report.tressoir.html
morphs here every 15 s while the node runs.
"""
import json
import os
import random
from dataclasses import replace

import pytest

from activation.agent import AgentConfig, RolloutReporter
from activation.common.data_syncing import resolve_path
from activation.dataset.loaders import DapoMathDataset
from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig

MODEL_NAME = "qwen3.5-9b"
MODEL_ID = "Qwen/Qwen3.5-9B"

BASE_AGENT_CONFIG = AgentConfig(
    # Default tools only (shell, python, parallel_tool_call, submit_answer). Thinking is off by the harness default.
    system_prompt=(
        "You are a careful problem solver working in a sandbox. Use the python or shell tools to compute "
        "rather than guessing. When you are done, call submit_answer exactly once with the final answer."
    ),
    model_name=MODEL_NAME,
    call_kwargs={"sampling_params": {"max_tokens": 2048, "temperature": 0.7}},
    max_turns=10,
    max_tool_errors=3,
    max_duration=300,
)


def _make_harness() -> HarnessRuntime:
    return HarnessRuntime(HarnessRuntimeConfig(
        model_configs={MODEL_NAME: ModelConfig(MODEL_NAME, MODEL_ID)},
        agent_max_concurrent=8,
    ))


def _print_rollout(result) -> None:
    print(
        f"\n=== rollout seed={result.seed} finish={result.finish_reason} turns={result.num_turns} "
        f"tokens in/cached/out={result.num_input_tokens}/{result.num_cached_input_tokens}/{result.num_output_tokens} "
        f"duration={result.duration:.1f}s score={result.score:.2f} answer={result.answer!r}"
    )
    for step in result.trajectory:
        if step["role"] == "assistant":
            calls = ", ".join(f"{call['name']}({json.dumps(call['arguments'])[:80]})" for call in step["tool_calls"])
            print(f"  assistant: {step['content'][:200]!r} -> {calls}")
        else:
            for output in step["tool_call_results"]:
                print(f"  tool: {output[:200]!r}")


@pytest.mark.gpu
@pytest.mark.slow
def test_basic_agent_primes():
    """A self-contained task, no dataset: a group of 2 rollouts with the live report."""
    harness = _make_harness()
    config = replace(
        BASE_AGENT_CONFIG,
        user_prompt="Compute the sum of the 140th, 141st and 142nd prime numbers (2 is the 1st prime).",
    )
    reporter = RolloutReporter(
        str(resolve_path("AGENT_TEST/primes")),
        title="Agent test: primes",
        description=f"{MODEL_ID}, group of 2, {config.max_turns} turns max, default tools.",
    )
    try:
        groups = harness.rollout_manager.perform_grouped_rollouts(
            [config], group_count=2, base_seed=0, perform_scoring=False, reporter=reporter,
        )
    finally:
        harness.loaded_models[MODEL_NAME].engine_to_device(FREE_DEVICE)

    assert len(groups) == 1 and len(groups[0]) == 2
    for result in groups[0]:
        _print_rollout(result)
        assert result.finish_reason == "submitted", result.finish_reason
        assert any(
            call["name"] in ("python", "shell", "parallel_tool_call")
            for step in result.trajectory for call in step.get("tool_calls", [])
        ), "no tool was used"
        assert str(result.answer).replace(",", "").strip().lstrip("-").isdigit(), result.answer
    answers = [int(str(result.answer).replace(",", "").strip()) for result in groups[0]]
    print(f"answers {answers}; expected 2441 (809 + 811 + 821)")
    assert 2441 in answers, answers  # at least one of the two rollouts gets it (drop this line to print only)
    assert os.path.exists(os.path.join(reporter.report_folder, "report.tressoir.html"))
    assert os.path.exists(os.path.join(reporter.report_folder, "report_data.json"))
    print("\n=== harness stats ===")
    print(json.dumps(harness.harness_stats.summarize(), indent=2))


@pytest.mark.gpu
@pytest.mark.slow
def test_basic_agent_dapo():
    """Two DAPO-Math tasks, a group of 2 each, scored; then the same call served from the cache."""
    harness = _make_harness()
    dataset = DapoMathDataset.load(harness, max_examples=64)
    harness.dataset_manager.register_dataset(dataset)
    tasks = list(dataset.scorable_tasks.values())
    random.Random(0).shuffle(tasks)
    tasks = tasks[:2]
    configs = [replace(BASE_AGENT_CONFIG, user_prompt=task.agent_prompt, dataset_task=task) for task in tasks]
    reporter = RolloutReporter(
        str(resolve_path("AGENT_TEST/dapo")),
        title="Agent test: DAPO-Math",
        description=f"{MODEL_ID}, 2 tasks x group of 2, {BASE_AGENT_CONFIG.max_turns} turns max, default tools.",
    )
    try:
        groups = harness.rollout_manager.perform_grouped_rollouts(
            configs, group_count=2, base_seed=0, perform_scoring=True, caching_id="agent_test_dapo", reporter=reporter,
        )
        loads_before = len(harness.harness_stats.model_loading_times)
        # The same request again: every rollout comes from the cache, the engine is not touched.
        cached = harness.rollout_manager.perform_grouped_rollouts(
            configs, group_count=2, base_seed=0, perform_scoring=True, caching_id="agent_test_dapo",
        )
    finally:
        harness.loaded_models[MODEL_NAME].engine_to_device(FREE_DEVICE)

    assert len(groups) == 2 and all(len(group) == 2 for group in groups)
    for task, group in zip(tasks, groups):
        print(f"\n### task {task.task_id}: gold {task.gold_answer}\n{task.agent_prompt[:300]}")
        for result in group:
            _print_rollout(result)  # the problem need not be solved
            assert result.finish_reason in ("submitted", "max_turns", "max_tool_errors", "max_duration", "no_tool_call")
            assert result.score in (0.0, 1.0)
            assert result.agent_config.dataset_task.task_id == task.task_id
    print(f"scores {[[result.score for result in group] for group in groups]}")
    assert len(harness.harness_stats.model_loading_times) == loads_before
    assert [[r.answer for r in group] for group in cached] == [[r.answer for r in group] for group in groups]
    assert os.path.exists(os.path.join(reporter.report_folder, "report.tressoir.html"))
    print("\n=== harness stats ===")
    print(json.dumps(harness.harness_stats.summarize(), indent=2))
