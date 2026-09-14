"""
One end-to-end agentic-program rollout on a GPU node, in the style of test_basic_agent.py: a program written in this
file runs a solver subagent (run_subagent), injects its result ahead of the task (augment_context), and runs the main
agent to verify and submit. Then the same request from the cache. (The lenient-deserialization and runtime checks that
need no GPU live in activation/bench/agent_probes/agentic_program_probe.py.)

    uv run sky exec --sync <node> -- uv run pytest activation/tests/test_basic_agentic_program.py --gpu --slow -s
"""
import json
import os
from dataclasses import replace

import pytest

from activation.agent import Agent, AgentConfig, AgenticProgram, AgentRunResult, RolloutReporter
from activation.common.data_syncing import resolve_path
from activation.dataset import ANSWER_RULES, DatasetTask, DatasetTaskKind, DatasetTaskMetricsKind, bare_prompt
from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig

MODEL_NAME = "qwen3.5-9b"
MODEL_ID = "Qwen/Qwen3.5-9B"

# DeepMath-103K (zwhe99/DeepMath-103K, train shard 0), difficulty 2.0, topic Number Theory -> Factorization.
QUESTION = "How many trailing zeroes are there in 100!?"
TRAILING_ZEROES = DatasetTask(
    task_id="trailing-zeroes-100", dataset_id="deepmath___fixture",
    task_datum={"question": QUESTION, "final_answer": "24", "difficulty": 2.0},
    reference_metrics_kind=DatasetTaskMetricsKind.MATH_VERIFY, gold_answer="24",
    agent_prompt=bare_prompt(QUESTION, ANSWER_RULES["math"]), task_kind=DatasetTaskKind.MATH,
)

BASE_AGENT_CONFIG = AgentConfig(
    # Default tools only (shell, python, parallel_tool_call, submit_answer). Thinking is off by the harness default.
    system_prompt=(
        "You are a careful problem solver working in a sandbox. Use the python or shell tools to compute "
        "rather than guessing. When you are done, call submit_answer exactly once with the final answer."
    ),
    user_prompt=TRAILING_ZEROES.agent_prompt,
    dataset_task=TRAILING_ZEROES,
    model_name=MODEL_NAME,
    call_kwargs={"sampling_params": {"max_tokens": 2048, "temperature": 0.7}},
    max_turns=10,
    max_tool_errors=3,
    max_duration=300,
    absolute_trajectory_cap=40_000,                                # the 9B engine holds 52k tokens: cap + max_tokens must fit
)


class SolveThenVerifyProgram(AgenticProgram):
    """A solver subagent answers first; the main agent sees that answer ahead of the task, checks it and submits."""
    name = "solve_then_verify"

    def __init__(self, agent: Agent, solver_max_duration: float = 120.0):
        super().__init__(agent)
        self.solver_max_duration = solver_max_duration

    def execute(self) -> AgentRunResult:
        task = self.agent.agent_config.dataset_task
        solver = self.agent.run_subagent(task.agent_prompt, max_duration=self.solver_max_duration)   # a ToolCallResult
        self.agent.augment_context([solver, "Check the solver's answer above with the python tool before you submit, and submit the correct value."])
        return self.agent.run()


PROGRAM_CONFIG = replace(
    BASE_AGENT_CONFIG,
    agentic_program=(SolveThenVerifyProgram, {"solver_max_duration": 120.0}),
    metadata={"program": SolveThenVerifyProgram.name},
)


def _make_harness() -> HarnessRuntime:
    return HarnessRuntime(HarnessRuntimeConfig(
        model_configs={MODEL_NAME: ModelConfig(MODEL_NAME, MODEL_ID)},
        agent_max_concurrent=4,
    ))


def _print_rollout(label: str, result: AgentRunResult) -> None:
    print(
        f"\n=== {label} seed={result.seed} finish={result.finish_reason} turns={result.num_turns} "
        f"tokens in/cached/out={result.num_input_tokens}/{result.num_cached_input_tokens}/{result.num_output_tokens} "
        f"duration={result.duration:.1f}s score={result.score} answer={result.answer!r}"
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
def test_basic_agentic_program_solve_then_verify():
    """The program runs a solver, injects its answer, runs the main agent; scored; then the same call from the cache."""
    harness = _make_harness()
    reporter = RolloutReporter(
        str(resolve_path("AGENT_TEST/program")),
        title="Agent test: agentic program",
        description=f"{MODEL_ID}, solve-then-verify program on a DeepMath difficulty-2 problem, {PROGRAM_CONFIG.max_turns} turns max.",
    )
    try:
        results = harness.rollout_manager.perform_single_rollouts(
            [PROGRAM_CONFIG], seed=0, perform_scoring=True, caching_id="agent_test_program", reporter=reporter,
        )
        loads_before = len(harness.harness_stats.model_loading_times)
        cached = harness.rollout_manager.perform_single_rollouts(
            [PROGRAM_CONFIG], seed=0, perform_scoring=True, caching_id="agent_test_program",
        )
    finally:
        harness.loaded_models[MODEL_NAME].engine_to_device(FREE_DEVICE)

    assert len(results) == 1
    result = results[0]
    assert len(result.subagent_results) == 1, "the program runs exactly one solver"
    solver = result.subagent_results[0]
    _print_rollout("solver", solver)
    _print_rollout("main", result)

    # The solver: a plain run in the same sandbox, no program and no delegation tools, bounded to 120 s; the parent's
    # note is its injected input (the parent had no segment yet, so its part holds only the system message).
    assert solver.agent_config.agentic_program is None and "subagent" not in solver.agent_config.tools.keys() - {None}
    assert solver.agent_config.max_duration <= 120.0
    assert solver.finish_reason in ("submitted", "max_turns", "max_tool_errors", "max_duration", "no_tool_call")
    assert [item["tool"] for item in solver.injected_input] == ["parent"]
    assert solver.injected_input[0]["activation_content"][0]["kind"] == "subagent_prompt"

    # The main agent: the solver's result and the program's note are recorded as injected input, and their text blocks
    # sit ahead of the task text in the first user message; no part is rendered (no AC model on this config).
    assert [item["tool"] for item in result.injected_input] == ["subagent", "program"], result.injected_input
    returned = result.injected_input[0]
    assert returned["subagent_index"] == 0 and returned["output"].startswith("Subagent finished")
    part = returned["activation_content"][0]
    assert part["kind"] == "subagent_return" and part["messages"][0]["role"] == "system" and part["tools"]
    assert part["messages"][-1]["role"] in ("tool", "assistant") and "compression_target" not in part
    first_user = next(message for message in result.prompt_messages if message["role"] == "user")
    content = first_user["content"]
    assert isinstance(content, list) and len(content) == 3 and all(item["type"] == "text" for item in content), content
    assert content[0]["text"].startswith("[subagent]\nSubagent finished") and content[1]["text"].startswith("[program]\nCheck")
    assert content[-1]["text"].startswith(QUESTION)
    assert result.finish_reason == "submitted", result.finish_reason
    assert any(
        call["name"] in ("python", "shell", "parallel_tool_call")
        for step in result.trajectory for call in step.get("tool_calls", [])
    ), "the main agent did not check with a tool"
    assert result.score == 1.0, (result.answer, result.score)
    assert result.duration >= solver.duration, "run_program measures the whole program"

    # The cache: the same call replays the row without touching the engine; the live program class is restored.
    assert len(harness.harness_stats.model_loading_times) == loads_before
    assert cached[0].answer == result.answer and cached[0].score == result.score
    assert cached[0].agent_config.agentic_program[0] is SolveThenVerifyProgram
    assert len(cached[0].subagent_results) == 1 and cached[0].injected_input == result.injected_input
    assert os.path.exists(os.path.join(reporter.report_folder, "report.tressoir.html"))
    print("\n=== harness stats ===")
    print(json.dumps(harness.harness_stats.summarize(), indent=2))
