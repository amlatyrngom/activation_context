"""
CPU probe of the agentic-program plumbing (no engine, no sandbox): a record deserializes without the program class
present (MissingClass placeholder that errs on construction and re-serializes unchanged), the cache key separates the
programmed and plain variants, `__main__` classes serialize under the script's module name, and the runtime API
(augment_context ahead of the task, set_final_answer, run_program error handling) behaves on the tiny fixture.

    uv run python -m activation.bench.agent_probes.agentic_program_probe
"""
from __future__ import annotations

import os
from dataclasses import replace

from activation.agent import Agent, AgentConfig, AgenticProgram, AgentRunResult
from activation.agent.agent_config import _class_spec
from activation.agent.rollout_caching import config_key
from activation.common.utils import MissingClass, MissingClassError
from activation.dataset import ANSWER_RULES, DatasetTask, DatasetTaskKind, DatasetTaskMetricsKind, bare_prompt
from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig

TINY = os.environ.get("TINY_QWEN35", "/workspace/IB/TMP/AGENT_ROLLOUTS/slice3_apply_20260909T103843Z/tiny_qwen35")
QUESTION = "How many trailing zeroes are there in 100!?"
TASK = DatasetTask(
    task_id="trailing-zeroes-100", dataset_id="deepmath___fixture", task_datum={"question": QUESTION},
    reference_metrics_kind=DatasetTaskMetricsKind.MATH_VERIFY, gold_answer="24",
    agent_prompt=bare_prompt(QUESTION, ANSWER_RULES["math"]), task_kind=DatasetTaskKind.MATH,
)


class AnswerFromKwargs(AgenticProgram):
    """Finishes without a model turn: the answer comes from the config."""
    name = "answer_from_kwargs"

    def __init__(self, agent: Agent, answer: str = "24"):
        super().__init__(agent)
        self.answer = answer

    def execute(self) -> AgentRunResult:
        self.agent.augment_context([{"type": "text", "text": "Programmed context ahead of the task."}])
        self.agent.set_final_answer(self.answer)
        return self.agent.run_results


class Broken(AgenticProgram):
    def execute(self) -> AgentRunResult:
        raise RuntimeError("boom")


class WrongReturn(AgenticProgram):
    def execute(self) -> AgentRunResult:
        return AgentRunResult(agent_config=self.agent.agent_config)


def check_serde() -> None:
    config = AgentConfig(system_prompt="s", user_prompt=TASK.agent_prompt, dataset_task=TASK, model_name="tiny",
                         agentic_program=(AnswerFromKwargs, {"answer": "24"}), metadata={"program": AnswerFromKwargs.name})
    row = AgentRunResult(agent_config=config, seed=0, finish_reason="programmed", answer="24", score=1.0).serialize()
    spec = row["agent_config"]["agentic_program"]
    assert spec["class"] == f"{AnswerFromKwargs.__module__}:AnswerFromKwargs".replace("__main__", __spec__.name if __spec__ else "__main__"), spec
    assert spec["kwargs"] == {"answer": "24"}
    live = AgentRunResult.deserialize(row)
    live_class = live.agent_config.agentic_program[0]                                  # importable here: the real class (a second module
    assert not isinstance(live_class, MissingClass) and live_class.__qualname__ == "AnswerFromKwargs"   # object when run as __main__)

    row["agent_config"]["agentic_program"]["class"] = "activation.bench.agent_probes.nowhere:AnswerFromKwargs"
    row["agent_config"]["tools"] = {"custom": {"class": "activation.extensions.nowhere:CustomTool", "kwargs": {"k": 1}}}
    restored = AgentRunResult.deserialize(row)                                         # lenient by default
    placeholder, kwargs = restored.agent_config.agentic_program
    assert isinstance(placeholder, MissingClass) and kwargs == {"answer": "24"}, (placeholder, kwargs)
    tool_placeholder, tool_kwargs = restored.agent_config.tools["custom"]
    assert isinstance(tool_placeholder, MissingClass) and tool_kwargs == {"k": 1}
    try:
        placeholder(None)
        raise AssertionError("a MissingClass must refuse construction")
    except MissingClassError as error:
        assert "activation.bench.agent_probes.nowhere:AnswerFromKwargs" in str(error)
    assert _class_spec(placeholder, kwargs)["class"] == "activation.bench.agent_probes.nowhere:AnswerFromKwargs"   # unchanged on rewrite
    assert restored.serialize()["agent_config"]["tools"]["custom"]["class"] == "activation.extensions.nowhere:CustomTool"
    assert restored.answer == "24" and restored.score == 1.0
    try:
        AgentRunResult.deserialize(row, strict=True)
        raise AssertionError("strict deserialization must raise")
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass
    plain = replace(config, agentic_program=None, metadata={})
    assert config_key(config) != config_key(plain)
    labelled = replace(config, metadata={"template": "math/base/0", "program": "answer_from_kwargs"})
    assert config_key(labelled) == "deepmath___fixture/trailing-zeroes-100/math/base/0/answer_from_kwargs"
    assert config_key(replace(labelled, metadata={"template": "math/base/0"})) == "deepmath___fixture/trailing-zeroes-100/math/base/0"
    print("serde: placeholder, strict, key and __main__ mapping OK", flush=True)


def check_runtime(harness: HarnessRuntime) -> None:
    base = AgentConfig(system_prompt="s", user_prompt=TASK.agent_prompt, dataset_task=TASK, model_name="tiny", max_duration=30)
    agent = Agent(harness, replace(base, agentic_program=(AnswerFromKwargs, {"answer": "24"})), seed=0)
    result = agent.run_program()
    assert result is agent.run_results and result.finish_reason == "programmed" and result.answer == "24", result.finish_reason
    assert result.num_turns == 0 and result.duration >= 0
    assert agent.score() == 1.0
    agent.begin()                                                                       # tokenizer only: the recorded prompt
    first_user = next(message for message in agent.run_results.prompt_messages if message["role"] == "user")
    content = first_user["content"]
    assert isinstance(content, list) and content[0]["text"] == "Programmed context ahead of the task." and content[-1]["text"].startswith(QUESTION), content
    try:
        agent.augment_context([{"type": "text", "text": "late"}])
        raise AssertionError("augment_context after begin must be refused")
    except AssertionError as error:
        assert "before the first turn" in str(error)
    agent.shutdown()

    broken = Agent(harness, replace(base, agentic_program=(Broken, {})), seed=0)
    result = broken.run_program()
    assert result.finish_reason == "error" and "boom" in (result.score_feedback or ""), result.finish_reason
    broken.shutdown()

    wrong = Agent(harness, replace(base, agentic_program=(WrongReturn, {})), seed=0)
    result = wrong.run_program()
    assert result.finish_reason == "error" and "own run_results" in (result.score_feedback or "")
    wrong.shutdown()

    missing = Agent(harness, replace(base, agentic_program=(MissingClass("x.y", "Z", "ModuleNotFoundError"), {})), seed=0)
    result = missing.run_program()
    assert result.finish_reason == "error" and "x.y:Z is not importable" in (result.score_feedback or "")
    missing.shutdown()
    print("runtime: programmed finish, context ahead of the task, error handling OK", flush=True)


def main() -> int:
    check_serde()
    if not os.path.isdir(TINY):
        print(f"tiny fixture missing ({TINY}); runtime checks skipped", flush=True)
        return 0
    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={"tiny": ModelConfig("tiny", TINY)}))
    check_runtime(harness)
    print("AGENTIC PROGRAM PROBE PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
