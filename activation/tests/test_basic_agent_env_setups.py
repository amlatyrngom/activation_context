"""
Slice 4a M0, CPU: env setups run once when an agent's env is created, the general subagent inherits the caller
without delegation, configs with setups and removed tools round-trip, and the default image recipe is stable.
The tiny Qwen3.5 fixture gives a tokenizer; a fake env stands in for podman.

    uv run pytest activation/tests/test_basic_agent_env_setups.py -s
"""
import os
import sys
from dataclasses import replace

import pytest

sys.modules.setdefault("fla", None)   # the tiny fixture never runs the FLA kernels

from activation.agent import AgentConfig, SubagentTool, SyntheticTurn, synthesize_agent
from activation.agent import agent as agent_module
from activation.agent.agent_env import CopyTreeSetup, WriteFilesSetup
from activation.agent.agent_tools import SUBAGENT_SYSTEM_NOTE, SUBAGENT_TASK_FORMAT
from activation.dataset import DatasetTask, DatasetTaskMetricsKind
from activation.dataset.environments import default_dockerfile, render_dockerfile
from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig

TINY = os.environ.get("TINY_QWEN35", "/workspace/IB/TMP/AGENT_ROLLOUTS/slice3_apply_20260909T103843Z/tiny_qwen35")


class FakeEnv:
    """In-memory stand-in for AgentEnv: records writes and copies; no podman."""
    instances: list["FakeEnv"] = []

    def __init__(self, dockerfile_path=None, env_args=None, default_image="", memory_limit_mb=None):
        self.files: dict[str, str] = {}
        self.copies: list[tuple[str, str]] = []
        self.dockerfile_path = dockerfile_path
        self.alive = True
        FakeEnv.instances.append(self)

    def write_file(self, path, content):
        self.files[path] = content
        return True

    def copy_in(self, local_path, env_path):
        self.copies.append((local_path, env_path))

    def run_shell(self, script, timeout=None):
        return "", True

    def run_python_code(self, code, timeout=None):
        return "", True

    def shutdown(self):
        self.alive = False


@pytest.fixture(scope="module")
def harness():
    if "/" in TINY and TINY.startswith(("/", ".")) and not os.path.isdir(TINY):
        pytest.skip(f"tiny fixture missing: {TINY}")           # a Hub id (TINY_QWEN35=Qwen/Qwen3.5-0.8B) is loaded instead on nodes
    return HarnessRuntime(HarnessRuntimeConfig(model_configs={"tiny": ModelConfig("tiny", TINY)}))


@pytest.fixture
def fake_env(monkeypatch):
    FakeEnv.instances.clear()
    monkeypatch.setattr(agent_module, "AgentEnv", FakeEnv)
    return FakeEnv


def _task() -> DatasetTask:
    return DatasetTask(task_id="q1", dataset_id="longdoc_fixture", task_datum={"article": "Once upon a time.", "checkout": "/tmp/nonexistent-checkout"},
                       reference_metrics_kind=DatasetTaskMetricsKind.EXACT_MATCH, gold_answer="B", agent_prompt="The document is at /workspace/document.txt.")


def test_env_setups_run_once_and_subagent_inherits(harness, fake_env):
    config = AgentConfig(
        system_prompt="You solve tasks.", model_name="tiny", user_prompt="Read the document and answer.", dataset_task=_task(),
        env_setups={
            "document": (WriteFilesSetup, {"files": {"/workspace/document.txt": {"datum_key": "article"}, "/workspace/note.txt": {"text": "hi"}}}),
            "repo": (CopyTreeSetup, {"datum_key": "checkout", "env_path": "/workspace/repo"}),
        },
        max_turns=6,
    )
    agent = synthesize_agent(harness, config, [SyntheticTurn("Let me delegate.", [("subagent", {"task": "TASK:\n- read the document"})], ["Subagent finished (simulated, 0 turns). Answer: (none)"])])
    assert "subagent" in agent.tools and isinstance(agent.tools["subagent"], SubagentTool)
    assert SUBAGENT_TASK_FORMAT in agent.tools["subagent"].tool_definition()["function"]["parameters"]["properties"]["task"]["description"]
    assert not fake_env.instances, "step mode must not create an env"
    env = agent.agent_env                                                   # first use: the env is created and prepared once
    assert env.files == {"/workspace/document.txt": "Once upon a time.", "/workspace/note.txt": "hi"}
    assert env.copies == [("/tmp/nonexistent-checkout", "/workspace/repo")]
    assert agent.agent_env is env and len(fake_env.instances) == 1
    # The general subagent inherits the caller: same model and task, the note, no delegation, no setups, the parent's env.
    child_config = agent.tools["subagent"]._inherit_general_config()
    assert child_config.model_name == "tiny" and child_config.dataset_task is config.dataset_task
    assert child_config.system_prompt.endswith(SUBAGENT_SYSTEM_NOTE) and child_config.env_setups == {}
    assert child_config.tools == {"subagent": None} and child_config.agent_name == "general_subagent"
    child = agent_module.Agent(harness, replace(child_config, user_prompt="do it"), agent_env=agent._env, parent_agent=agent)
    assert "subagent" not in child.tools and set(child.tools) == {"shell", "python", "parallel_tool_call", "submit_answer", "compact"}
    assert child.agent_env is env and len(fake_env.instances) == 1      # shared, not re-prepared
    child.shutdown()
    assert env.alive                                                       # the child does not own the env
    agent.shutdown()
    assert not env.alive


def test_setup_failure_shuts_the_env_down(harness, fake_env):
    config = AgentConfig(model_name="tiny", user_prompt="x", env_setups={"bad": (WriteFilesSetup, {"files": {"/x": {"datum_key": "missing"}}})})
    agent = agent_module.Agent(harness, config)
    with pytest.raises(ValueError, match="needs a dataset task"):
        agent.agent_env
    assert agent._env is None and len(fake_env.instances) == 1 and not fake_env.instances[0].alive


def test_config_round_trip_with_setups_and_removed_tool(harness):
    config = AgentConfig(model_name="tiny", user_prompt="x", dataset_task=_task(), tools={"subagent": None},
                         env_setups={"document": (WriteFilesSetup, {"files": {"/workspace/document.txt": {"datum_key": "article"}}})})
    data = config.serialize()
    assert data["tools"] == {"subagent": None}
    assert data["env_setups"] == {"document": {"class": "activation.agent.agent_env:WriteFilesSetup",
                                               "kwargs": {"files": {"/workspace/document.txt": {"datum_key": "article"}}}}}
    back = AgentConfig.deserialize(data, harness)
    assert back.tools == {"subagent": None} and back.env_setups == config.env_setups
    agent = agent_module.Agent(harness, back)
    assert "subagent" not in agent.tools and "compact" in agent.tools


def test_default_dockerfile_is_content_addressed(tmp_path, monkeypatch):
    monkeypatch.setenv("ACTIVATION_SYNC_ROOT", str(tmp_path))
    first = default_dockerfile()
    assert first == default_dockerfile() and open(first).read() == render_dockerfile()
    other = default_dockerfile(extra_lines=("RUN pip install torch",))
    assert other != first and "torch" in open(other).read()
    assert render_dockerfile().startswith("FROM docker.io/library/python:3.12-slim\n")


@pytest.mark.slow
def test_real_env_setups_with_podman(harness, tmp_path):
    """On a machine with podman: the default image builds (once, content-addressed), setups write and copy into a real sandbox."""
    from activation.agent.agent_env import podman_available
    if not podman_available():
        pytest.skip("podman not available")
    (tmp_path / "repo" / "pkg").mkdir(parents=True)
    (tmp_path / "repo" / "pkg" / "mod.py").write_text("VALUE = 7\n")
    task = DatasetTask(task_id="q1", dataset_id="fx", task_datum={"article": "Once upon a time.", "checkout": str(tmp_path / "repo")},
                       reference_metrics_kind=DatasetTaskMetricsKind.EXACT_MATCH, gold_answer="7")
    config = AgentConfig(
        model_name="tiny", user_prompt="x", dataset_task=task, env_dockerfile_path=default_dockerfile(),
        env_setups={"document": (WriteFilesSetup, {"files": {"/workspace/document.txt": {"datum_key": "article"}}}),
                    "repo": (CopyTreeSetup, {"datum_key": "checkout", "env_path": "/workspace/repo"})},
    )
    agent = agent_module.Agent(harness, config)
    try:
        output, ok = agent.agent_env.run_shell("cat /workspace/document.txt; ls /workspace/repo/pkg; python -c 'import numpy, scipy, pandas, sympy; print(numpy.__version__)'; rg --version | head -1")
        assert ok, output
        assert "Once upon a time." in output and "mod.py" in output and "ripgrep" in output, output
        output, ok = agent.agent_env.run_python_code("import sys; sys.path.insert(0, '/workspace/repo'); from pkg import mod; print(mod.VALUE)")
        assert ok and output.strip() == "7", output
    finally:
        agent.shutdown()


def test_max_turns_counts_every_compaction_segment(harness, fake_env, monkeypatch):
    """The turn budget is a total over the run: a compaction restarts the segment, not the counter."""
    from activation.agent.agent_utils import ModelDialect
    from activation.harness.loaded_model import EngineChatOutput

    config = AgentConfig(system_prompt="You solve tasks.", model_name="tiny", user_prompt="Look around.", dataset_task=_task(), max_turns=4)
    agent = agent_module.Agent(harness, config)
    def call(index, name, **arguments):
        return ("", [{"id": f"call_{index}", "name": name, "arguments": arguments}])
    script = [call(1, "shell", script="ls"), call(2, "compact", summary="nothing yet"), call(3, "shell", script="ls"),
              call(4, "shell", script="ls"), call(5, "shell", script="ls")]
    replies = iter(script)
    seeds = []

    def fake_submit(self, chat_kwargs=None):
        seeds.append(self.seed + self.run_results.total_turns())
        return EngineChatOutput(text="", token_ids=[1, 2, 3], logprobs=None, prompt_token_count=len(self.prefix), output_token_count=3, lora_name=None)

    monkeypatch.setattr(agent_module.Agent, "_submit", fake_submit)
    monkeypatch.setattr(ModelDialect, "parse", lambda self, block, parameter_types=None: next(replies))   # begin() installs the model's dialect
    original_after_append = agent._after_append

    def after_append():
        original_after_append()
        if agent.run_results.num_turns == 1 and not agent.run_results.compactions:
            agent.compaction_due = True                                       # force the demand after the first turn
    monkeypatch.setattr(agent, "_after_append", after_append)
    results = agent.run()
    assert results.finish_reason == "max_turns"
    assert len(results.compactions) == 1 and results.compactions[0].num_turns == 2 and results.num_turns == 2
    assert results.total_turns() == 4 and results.totals()["turns"] == 4
    assert seeds == sorted(set(seeds)), "the per-turn sampling seed never repeats across segments"
    agent.shutdown()


def test_argument_aliases_and_lone_unknown_key(harness):
    from activation.agent.agent_tools import PythonTool, SemanticSearchTool, ShellTool, SubmitAnswerTool
    agent = agent_module.Agent(harness, AgentConfig(system_prompt="s", model_name="tiny", user_prompt="u"))
    assert agent.tools["shell"].normalize_arguments({"command": "ls"}) == {"script": "ls"}
    assert agent.tools["shell"].normalize_arguments({"script": "ls", "command": "pwd"}) == {"script": "ls", "command": "pwd"}   # known names win
    assert agent.tools["python"].normalize_arguments({"src": "print(1)"}) == {"code": "print(1)"}                          # lone unknown -> lone missing
    assert agent.tools["python"].normalize_arguments({"src": "x", "other": "y"}) == {"src": "x", "other": "y"}             # ambiguous: untouched
    assert agent.tools["submit_answer"].normalize_arguments({"final_answer": "B"}) == {"answer": "B"}
    assert SemanticSearchTool.aliases["q"] == "query" and ShellTool.aliases["command"] == "script" and PythonTool.aliases["script"] == "code"
    assert isinstance(agent.tools["submit_answer"], SubmitAnswerTool)
    agent.shutdown()


def test_repeated_calls_are_not_run_and_empty_submit_is_refused(harness, fake_env, monkeypatch):
    from activation.agent.agent import REPEATED_CALL_NOTE
    from activation.agent.agent_utils import ModelDialect
    from activation.harness.loaded_model import EngineChatOutput

    def call(index, name, **arguments):
        return ("", [{"id": f"call_{index}", "name": name, "arguments": arguments}])
    script = iter([call(1, "shell", script="ls"), call(2, "shell", command="pwd"), call(3, "shell", script="pwd"),
                   call(4, "submit_answer"), call(5, "submit_answer", answer="  "), call(6, "submit_answer", final_answer="B")])
    monkeypatch.setattr(agent_module.Agent, "_submit", lambda self, chat_kwargs=None: EngineChatOutput(text="", token_ids=[1, 2], prompt_token_count=len(self.prefix), output_token_count=2))
    monkeypatch.setattr(ModelDialect, "parse", lambda self, block, parameter_types=None: next(script))
    agent = agent_module.Agent(harness, AgentConfig(system_prompt="s", model_name="tiny", user_prompt="u", dataset_task=_task(), max_turns=10, max_tool_errors=8))
    results = agent.run()
    outputs = [step["tool_call_results"][0] for step in results.trajectory if step["role"] == "tool"]
    assert outputs[0] == "(no output)" or "ls" in outputs[0] or outputs[0] == ""                        # the fake env ran it
    assert not outputs[1].startswith("Bad arguments") and not outputs[1].startswith("Not run"), "an aliased argument name runs the call"
    assert outputs[2] == REPEATED_CALL_NOTE.format(name="shell")                                       # identical after normalization: not run
    assert outputs[3].startswith("Nothing submitted") and outputs[4].startswith("Nothing submitted")
    assert results.finish_reason == "submitted" and results.answer == "B" and agent.errors == 3
    assert results.trajectory[2]["tool_calls"][0]["arguments"] == {"command": "pwd"}, "the model's original call stays in the trajectory"
    agent.shutdown()


def test_thinking_template_kwargs_reach_the_prompt(harness):
    """Thinking on: the generation prompt the model reads opens a think block (the tiny Qwen3.5 template honours enable_thinking)."""
    off = agent_module.Agent(harness, AgentConfig(system_prompt="s", model_name="tiny", user_prompt="u", call_kwargs={"chat_template_kwargs": {"enable_thinking": False}}))
    on = agent_module.Agent(harness, AgentConfig(system_prompt="s", model_name="tiny", user_prompt="u",
                                                 call_kwargs={"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}))
    off.begin(); on.begin()
    tokenizer = harness.loaded_models["tiny"].tokenizer
    assert tokenizer.decode(off.prefix).rstrip().endswith("</think>") and tokenizer.decode(on.prefix).rstrip().endswith("<think>")
    off.shutdown(); on.shutdown()
