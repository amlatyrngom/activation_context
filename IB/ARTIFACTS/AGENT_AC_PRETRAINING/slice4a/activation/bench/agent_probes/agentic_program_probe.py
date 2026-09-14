"""
CPU probe of the agentic-program plumbing (no engine, no sandbox): a record deserializes without the program class
present (MissingClass placeholder that errs on construction and re-serializes unchanged), the cache key separates the
programmed and plain variants, `__main__` classes serialize under the script's module name, the runtime API
(run_tool, augment_context ahead of the task, set_final_answer, run_program error handling) behaves on the tiny
fixture, and a base run's record converts to AC-bearing messages (activation_messages_of).

    uv run python -m activation.bench.agent_probes.agentic_program_probe
"""
from __future__ import annotations

import os
from dataclasses import replace

from activation.agent import Agent, AgentConfig, AgenticProgram, AgentRunResult
from activation.agent.agent_config import _class_spec
from activation.agent.agent_tools import AgentTool, ToolCallResult
from activation.agent.rollout_caching import config_key
from activation.agent_training.agent_training_utils import activation_messages_of
from activation.common.ac_parts import direct_parts
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
        self.agent.augment_context("Programmed context ahead of the task.")
        self.agent.set_final_answer(self.answer)
        return self.agent.run_results


class EchoTool(AgentTool):
    """A tool with no sandbox: returns its argument, so run_tool and truncation can be exercised on this host."""
    name = "echo"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    def execute(self, text: str = "") -> ToolCallResult:
        return ToolCallResult(output=text)


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
    assert agent.run_results.injected_input == [ToolCallResult(tool="program", output="Programmed context ahead of the task.").serialize()]
    agent.begin()                                                                       # tokenizer only: the recorded prompt
    first_user = next(message for message in agent.run_results.prompt_messages if message["role"] == "user")
    content = first_user["content"]
    assert isinstance(content, list) and content[0]["text"] == "[program]\nProgrammed context ahead of the task.\n\n", content
    assert content[-1]["text"].startswith(QUESTION), content
    try:
        agent.augment_context("late")
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


def check_activation_content(harness: HarnessRuntime) -> None:
    """run_tool through the model's path; activation content produced without an AC model, recorded, and rendered on conversion."""
    base = AgentConfig(system_prompt="s", user_prompt=TASK.agent_prompt, dataset_task=TASK, model_name="tiny", max_duration=30)
    agent = Agent(harness, base, seed=0)
    long_text = "0123456789" * 5_000                                                    # 50k chars: past the 20k visible limit
    result = agent.run_tool(EchoTool(harness, agent), text=long_text)
    assert result.tool == "echo" and "truncated" in result.output and len(result.output) < 21_000
    assert [part["kind"] for part in result.activation_content] == ["tool_output"]
    part = result.activation_content[0]
    assert part["messages"][1] == {"role": "tool", "content": long_text} and "compression_target" not in part
    context = part["messages"][0]["content"][0]                                       # before the first turn: the first prompt as it stands
    assert context["kind"] == "parent_context" and context["messages"][0] == {"role": "system", "content": "s"}
    assert context["messages"][1]["content"] == TASK.agent_prompt and context["tools"]
    short = agent.run_tool(EchoTool(harness, agent), text="short")
    assert short.output == "short" and short.activation_content == [] and short.tool == "echo"
    bad = agent.run_tool(EchoTool(harness, agent), text="x", nonsense=1)                 # an unknown extra argument (no alias target)
    assert bad.is_error and bad.output.startswith("Bad arguments for echo")
    agent.augment_context([result, "Check the echo."])
    before = agent.to_activation_context("subagent_prompt")                           # what a solver spawned now would receive
    first = before["messages"][1]["content"]
    assert [item.get("type") for item in first] == ["activation_context", "text", "text", "text"], [item.get("type") for item in first]
    assert first[0]["kind"] == "tool_output" and "ac_name" not in first[0] and first[1]["text"].startswith("[echo]\n")
    assert first[2]["text"].startswith("[program]\n") and first[3]["text"].startswith(QUESTION)
    agent.begin()
    run = agent.run_results
    assert [item["tool"] for item in run.injected_input] == ["echo", "program"]
    first_user = next(message for message in run.prompt_messages if message["role"] == "user")
    assert not direct_parts([first_user]) and first_user["content"][0]["text"].startswith("[echo]\n0123456789")   # nothing rendered: no AC model
    # A tool step recorded through the model's path, with a nested parent context now that the segment exists.
    tokenizer = agent.loaded_model.tokenizer
    call = {"id": "e1", "name": "echo", "arguments": {"text": "again"}}
    agent.record_assistant_turn("Echoing.", [call], tokenizer.encode("Echoing."), [])
    stepped = agent._finalize_result(call, ToolCallResult(output=long_text))
    agent.append_tool_results([call], [stepped])
    step = run.trajectory[-1]
    assert step["role"] == "tool" and not direct_parts(step["messages"]) and step["tool_results"][0]["tool"] == "echo"
    nested = step["tool_results"][0]["activation_content"][0]
    assert nested["kind"] == "tool_output" and nested["messages"][0]["role"] == "user"
    parent = nested["messages"][0]["content"][0]
    assert parent["kind"] == "parent_context" and parent["messages"][0]["role"] == "system" and parent["tools"]
    assert parent["messages"][-1]["role"] == "assistant"                                     # the segment through the echoing turn
    # The record survives serialization and converts to AC-bearing messages for a reader with an AC model.
    restored = AgentRunResult.deserialize(__import__("json").loads(__import__("json").dumps(run.serialize())))
    assert restored.injected_input == run.injected_input and restored.trajectory[-1]["tool_results"] == step["tool_results"]
    reader = replace(base, ac_model_name="ac_x")
    converted = activation_messages_of(restored, reader)
    parts = direct_parts(converted)
    assert [part["kind"] for part in parts] == ["tool_output", "tool_output"], [part["kind"] for part in parts]
    assert direct_parts(parts[0]["messages"])[0]["ac_name"] == "ac_x"                      # the nested pre-turn parent context resolved too
    assert all(part["ac_name"] == "ac_x" and part["compression_target"] == reader.ac_tool_output_ratio for part in parts)
    first = next(message for message in converted if message["role"] == "user")
    assert first["content"][0]["type"] == "activation_context" and first["content"][1]["text"].startswith("[echo]\n")
    assert first["content"][2]["text"].startswith("[program]\n") and first["content"][-1]["text"].startswith(QUESTION)
    tool_message = converted[-1]
    assert tool_message["role"] == "tool" and tool_message["content"][0]["type"] == "activation_context"
    inner = direct_parts(tool_message["content"][0]["messages"])[0]
    assert inner["kind"] == "parent_context" and inner["compression_target"] == reader.ac_subagent_ratio and inner["ac_name"] == "ac_x"
    assert activation_messages_of(restored, base) == restored.prompt_messages + [m for s in restored.trajectory for m in s["messages"]]
    assert restored.trajectory[-1]["messages"] == step["messages"]                             # the record itself is untouched
    agent.shutdown()
    print("activation content: run_tool, truncation part, nested parent context, record and conversion OK", flush=True)


def main() -> int:
    check_serde()
    if not os.path.isdir(TINY):
        print(f"tiny fixture missing ({TINY}); runtime checks skipped", flush=True)
        return 0
    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={"tiny": ModelConfig("tiny", TINY)}))
    check_runtime(harness)
    check_activation_content(harness)
    print("AGENTIC PROGRAM PROBE PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
