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
    request = part["messages"][0]                                                     # flat (09-26): the call, then the full output
    assert request["role"] == "user" and request["content"].startswith("Tool call: echo(") and len(request["content"]) < 2_100
    assert not direct_parts(part["messages"])
    short = agent.run_tool(EchoTool(harness, agent), text="short")
    assert short.output == "short" and short.activation_content == [] and short.tool == "echo"
    bad = agent.run_tool(EchoTool(harness, agent), text="x", nonsense=1)                 # an unknown extra argument (no alias target)
    assert bad.is_error and bad.output.startswith("Bad arguments for echo")
    agent.augment_context([result, "Check the echo."])
    before = agent.to_activation_context("subagent_prompt")                           # what a solver spawned now would receive: flat text
    assert before["messages"][0] == {"role": "system", "content": "s"} and before["tools"] and not direct_parts(before["messages"])
    first = before["messages"][1]["content"]
    assert isinstance(first, str) and first.startswith("[echo]\n") and "[program]\n" in first and first.endswith(TASK.agent_prompt), first[:200]
    agent.begin()
    run = agent.run_results
    assert [item["tool"] for item in run.injected_input] == ["echo", "program"]
    first_user = next(message for message in run.prompt_messages if message["role"] == "user")
    assert not direct_parts([first_user]) and first_user["content"][0]["text"].startswith("[echo]\n0123456789")   # nothing rendered: no AC model
    # A tool step recorded through the model's path: its part is flat (no parent context nested inside).
    tokenizer = agent.loaded_model.tokenizer
    call = {"id": "e1", "name": "echo", "arguments": {"text": "again"}}
    agent.record_assistant_turn("Echoing.", [call], tokenizer.encode("Echoing."), [])
    stepped = agent._finalize_result(call, ToolCallResult(output=long_text))
    agent.append_tool_results([call], [stepped])
    step = run.trajectory[-1]
    assert step["role"] == "tool" and not direct_parts(step["messages"]) and step["tool_results"][0]["tool"] == "echo"
    recorded = step["tool_results"][0]["activation_content"][0]
    assert recorded["kind"] == "tool_output" and recorded["messages"][0]["content"] == 'Tool call: echo({"text": "again"})'
    assert recorded["messages"][1] == {"role": "tool", "content": long_text} and not direct_parts(recorded["messages"])
    # The record survives serialization and converts to AC-bearing messages for a reader with an AC model.
    restored = AgentRunResult.deserialize(__import__("json").loads(__import__("json").dumps(run.serialize())))
    assert restored.injected_input == run.injected_input and restored.trajectory[-1]["tool_results"] == step["tool_results"]
    reader = replace(base, ac_model_name="ac_x")
    converted = activation_messages_of(restored, reader)
    parts = direct_parts(converted)
    assert [part["kind"] for part in parts] == ["tool_output", "tool_output"], [part["kind"] for part in parts]
    assert not any(direct_parts(part["messages"]) for part in parts)                        # depth 1
    assert all(part["ac_name"] == "ac_x" and part["compression_target"] == reader.ac_tool_output_ratio for part in parts)
    first = next(message for message in converted if message["role"] == "user")
    assert first["content"][0]["type"] == "activation_context" and first["content"][1]["text"].startswith("[echo]\n")
    assert first["content"][2]["text"].startswith("[program]\n") and first["content"][-1]["text"].startswith(QUESTION)
    tool_message = converted[-1]
    assert tool_message["role"] == "tool" and tool_message["content"][0]["type"] == "activation_context"
    assert not direct_parts(tool_message["content"][0]["messages"])
    assert activation_messages_of(restored, base) == restored.prompt_messages + [m for s in restored.trajectory for m in s["messages"]]
    assert restored.trajectory[-1]["messages"] == step["messages"]                             # the record itself is untouched
    agent.shutdown()
    print("activation content: run_tool, flat truncation part, record and conversion OK", flush=True)


def check_flattening(harness: HarnessRuntime) -> None:
    """09-26 flattening: summaries in order, subagent returns without the inherited parent, the token cap, the search part, the guards."""
    import activation.common.ac_parts as ac_parts
    from activation.agent import SemanticSearchTool
    from activation.dataset.dataset_utils import message_text
    base = AgentConfig(system_prompt="s", user_prompt=TASK.agent_prompt, dataset_task=TASK, model_name="tiny", max_duration=30)
    agent = Agent(harness, base, seed=0)
    agent.begin()
    tokenizer = agent.loaded_model.tokenizer
    turn = lambda text: agent.record_assistant_turn(text, [], tokenizer.encode(text), [])
    turn("FIRST-TURN")
    agent.compact("SUMMARY-ONE")
    turn("SECOND-TURN")
    agent.compact("SUMMARY-TWO")
    turn("THIRD-TURN")
    view = agent.flattened_view()
    text = "\n".join(message_text(message) for message in view)
    assert view[0]["role"] == "system" and not direct_parts(view)
    assert text.count("SUMMARY-ONE") == 1 and text.count("SUMMARY-TWO") == 1 and text.index("SUMMARY-ONE") < text.index("SUMMARY-TWO") < text.index("THIRD-TURN")
    assert "FIRST-TURN" not in text and "SECOND-TURN" not in text and text.count(QUESTION) == 1, text   # compacted turns live in the summaries only
    compaction = agent.to_activation_context("compaction")
    assert compaction["messages"] == view and compaction["tools"]

    # The token cap keeps the head and the latest whole messages, with a note of how many were omitted.
    saved = ac_parts.AC_FLAT_MAX_TOKENS
    try:
        for index in range(40):
            turn(f"FILLER-{index} " + "word " * 50)
        ac_parts.AC_FLAT_MAX_TOKENS = sum(len(tokenizer.encode(message_text(m), add_special_tokens=False)) for m in view[:2]) + 300
        capped = agent.flattened_view()
        capped_text = "\n".join(message_text(message) for message in capped)
        assert capped[:2] == view[:2] and "earlier messages omitted]" in capped_text and "FILLER-39" in capped_text and "FILLER-0 " not in capped_text
    finally:
        ac_parts.AC_FLAT_MAX_TOKENS = saved
    agent.shutdown()

    # A subagent's return leaves out the parent context it inherited; its prompt part is the parent's flattened view.
    parent = Agent(harness, base, seed=0)
    prompt_part = parent.to_activation_context("subagent_prompt")
    child = Agent(harness, replace(base, user_prompt="CHILD-TASK"), seed=1)
    child.augment_context(ToolCallResult(tool="parent", output="PARENT-INTRO", activation_content=[prompt_part]))
    child.begin()
    with_parent = "\n".join(message_text(message) for message in child.flattened_view())
    returned = child.to_activation_context("subagent_return")
    returned_text = "\n".join(message_text(message) for message in returned["messages"])
    assert "PARENT-INTRO" in with_parent and "PARENT-INTRO" not in returned_text and "CHILD-TASK" in returned_text
    assert not direct_parts(returned["messages"]) and QUESTION not in returned_text      # nothing of the parent's context
    # Teacher view of a compacted subagent (the training conversion): the task appears once, not again after its parent transcript.
    from activation.ac_model.ac_model_study import ActivationContextStudyGenerator
    from activation.common.ac_parts import COMPACTION_INSTRUCTIONS, activation_part as make_part
    child.record_assistant_turn("CHILD-TURN", [], tokenizer.encode("CHILD-TURN"), [])
    tree = child.to_activation_context("compaction")
    first_raw = child.first_user_messages(render=lambda parts: [dict(part) for part in parts])[-1]
    student = [child.run_results.prompt_messages[0], first_raw,
               {"role": "user", "content": [tree, {"type": "text", "text": "\n\n" + COMPACTION_INSTRUCTIONS}]}]
    teacher = ActivationContextStudyGenerator._expand_parts(object.__new__(ActivationContextStudyGenerator), student)
    teacher_text = "\n".join(message_text(message) for message in teacher)
    assert teacher_text.count("CHILD-TASK") == 1 and "Parent transcript:" in teacher_text and "CHILD-TURN" in teacher_text, [(m["role"], message_text(m)[:300]) for m in teacher]
    assert make_part("compaction", tree["messages"])["messages"] == tree["messages"]
    child.shutdown()
    parent.shutdown()

    # Search: the extra passages, then the query as the question (the trained contextual view), no parent context.
    class Chunk:
        def __init__(self, index):
            self.chunk_id = f"c{index}"
    class Index:
        def build_bm25_index(self):
            pass
        def bm25_query_many_frozen(self, queries, top_k):
            return [[Chunk(i) for i in range(top_k)]]
        def get_chunk_section(self, chunk):
            return f"passage {chunk.chunk_id}"
    manager = harness.dataset_manager
    manager.loaded_datasets["probe_corpus"], manager.dataset_indexes["probe_corpus"] = object(), Index()
    searcher = Agent(harness, base, seed=0)
    found = SemanticSearchTool(harness, searcher, dataset_id="probe_corpus", top_k=2).execute("where is it")
    search = found.activation_content[0]
    assert search["kind"] == "search" and not direct_parts(search["messages"]) and "[c0]" in found.output and "[c2]" not in found.output
    evidence = search["messages"][0]["content"]                                          # the trained RAG view: numbered passages
    assert search["messages"][0]["role"] == "user" and evidence.startswith("Passage 1:\npassage c2") and "passage c1" not in evidence
    assert search["messages"][-1] == {"role": "user", "content": "Question: where is it"}
    searcher.shutdown()
    del manager.loaded_datasets["probe_corpus"], manager.dataset_indexes["probe_corpus"]

    # Guards: no part inside a part, no oversized part.
    for messages in ([{"role": "user", "content": [prompt_part]}], [{"role": "user", "content": "x" * (ac_parts.AC_PART_MAX_CHARS + 1)}]):
        try:
            ac_parts.activation_part("probe", messages)
            raise AssertionError("guard did not fire")
        except ValueError:
            pass
    print("flattening: summaries in order, cap, subagent return, search part, guards OK", flush=True)


def main() -> int:
    check_serde()
    if not os.path.isdir(TINY):
        print(f"tiny fixture missing ({TINY}); runtime checks skipped", flush=True)
        return 0
    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={"tiny": ModelConfig("tiny", TINY)}))
    check_runtime(harness)
    check_activation_content(harness)
    check_flattening(harness)
    print("AGENTIC PROGRAM PROBE PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
