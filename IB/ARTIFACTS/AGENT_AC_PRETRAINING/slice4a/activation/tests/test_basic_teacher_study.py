"""
The teacher study on CPU (tiny fixture, no engine, no sandbox): weighted, seeded sampling of benchmarks, teacher kinds
and templates into labelled configs; cache keys that separate templates; the rollout report's per-benchmark summary;
the oracle call kwargs; and the auto-offload of very long task texts.

    uv run pytest activation/tests/test_basic_teacher_study.py -s
"""
import json
import os
import random
import sys

import pytest

sys.modules.setdefault("fla", None)

from activation.agent import AgentConfig, SyntheticTurn, synthesize_agent
from activation.agent import agent as agent_module
from activation.agent.rollout_caching import config_key
from activation.agent.rollout_reporter import RolloutReporter
from activation.agent_training.agent_training_teacher_study import (
    BENCHMARKS, THINKING_SAMPLING, AgentTeacherKind, AgentTrainingStudy, oracle_call_kwargs, template_id,
)
from activation.agent_training.agent_training_teacher_study_prompts import META_AGENT_PROMPT_TEMPLATES, render_template
from activation.dataset import DatasetTask, DatasetTaskKind, DatasetTaskMetricsKind, LoadedDataset
from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig

TINY = os.environ.get("TINY_QWEN35", "/workspace/IB/TMP/AGENT_ROLLOUTS/slice3_apply_20260909T103843Z/tiny_qwen35")


@pytest.fixture(scope="module")
def harness():
    if "/" in TINY and TINY.startswith(("/", ".")) and not os.path.isdir(TINY):
        pytest.skip(f"tiny fixture missing: {TINY}")
    return HarnessRuntime(HarnessRuntimeConfig(model_configs={"tiny": ModelConfig("tiny", TINY)}))


def _task(task_id="t1", dataset_id="quality___fixture", kind=DatasetTaskKind.FILE_SEARCH):
    return DatasetTask(task_id=task_id, dataset_id=dataset_id, task_datum={"article": "x"}, reference_metrics_kind=DatasetTaskMetricsKind.EXACT_MATCH,
                       gold_answer="42", agent_prompt="What is six times seven?\n\nWhen you are done, call submit_answer with the number.", task_kind=kind,
                       env_setups={"document": {"class": "activation.agent.agent_env:WriteFilesSetup", "kwargs": {"files": {"/workspace/d.txt": {"datum_key": "article"}}}}})


def _study(harness, **kwargs):
    study = AgentTrainingStudy(harness, {"quality": 0.7, "dapo_math": 0.3}, model_name="tiny", env_dockerfile_path="/nonexistent/Dockerfile", **kwargs)
    study.datasets = {
        "quality": LoadedDataset(dataset_id="quality___fixture", scorable_tasks={f"q{i}": _task(f"q{i}") for i in range(30)}),
        "dapo_math": LoadedDataset(dataset_id="dapo_math___fixture", scorable_tasks={f"m{i}": _task(f"m{i}", "dapo_math___fixture", DatasetTaskKind.MATH) for i in range(6)}),
    }
    return study


def test_templates_cover_every_kind_and_render():
    for kind in DatasetTaskKind:
        if kind is DatasetTaskKind.CODE_IMPL:
            continue
        for teacher in AgentTeacherKind:
            templates = META_AGENT_PROMPT_TEMPLATES[(kind, teacher)]
            assert templates and all("{task}" in template for template in templates)
    rendered = render_template(META_AGENT_PROMPT_TEMPLATES[(DatasetTaskKind.MATH, AgentTeacherKind.PARALLEL_MULTI_AGENT)][0], "Compute 1+1.\n\nSubmit the number.")
    assert rendered.startswith("Compute 1+1.") and "subagent" in rendered and rendered.endswith("submit.")
    assert render_template("{task}", " bare ") == "bare"


def test_campaign_sampling_is_weighted_seeded_and_labelled(harness):
    study = _study(harness)
    configs = study.generate_rollout_campaign_tasks(24, seed=1)
    assert len(configs) == 24 and len({c.dataset_task.task_id for c in configs}) == 24                       # no repeats
    by_benchmark = {name: sum(c.metadata["benchmark"] == name for c in configs) for name in ("quality", "dapo_math")}
    assert by_benchmark["dapo_math"] <= 6 and by_benchmark["quality"] >= 12                                    # math exhausts at 6, the rest is quality
    assert study.generate_rollout_campaign_tasks(24, seed=1)[0].user_prompt == configs[0].user_prompt         # seeded
    kinds = {c.metadata["teacher_kind"] for c in configs}
    assert kinds <= {str(k) for k in AgentTeacherKind} and len(kinds) >= 2
    first = configs[0]
    assert first.metadata["task_kind"] in ("file_search", "math") and first.metadata["template"] == template_id(DatasetTaskKind(first.metadata["task_kind"]), AgentTeacherKind(first.metadata["teacher_kind"]), int(first.metadata["template"].split("/")[-1]))
    assert first.system_prompt and first.call_kwargs["chat_template_kwargs"]["reasoning_effort"] == "medium" and first.max_turns == 100
    quality = next(c for c in configs if c.metadata["benchmark"] == "quality")
    assert set(quality.tools) == {"semantic_search"} and "document" in quality.env_setups
    math = next((c for c in configs if c.metadata["benchmark"] == "dapo_math"), None)
    assert math is None or "semantic_search" not in math.tools
    # base templates keep the bare prompt; the cache key separates templates of the same task
    base = AgentTrainingStudy(harness, {"quality": 1.0}, {(DatasetTaskKind.FILE_SEARCH, AgentTeacherKind.BASE): 1.0}, model_name="tiny", env_dockerfile_path="x")
    base.datasets = {"quality": study.datasets["quality"]}
    plain = base.generate_rollout_campaign_tasks(1, seed=0)[0]
    assert plain.user_prompt == plain.dataset_task.agent_prompt and plain.metadata["teacher_kind"] == "base"
    guided = study.config_for_task("quality", plain.dataset_task, random.Random(0))
    assert config_key(plain) != config_key(guided) or plain.user_prompt == guided.user_prompt
    assert AgentConfig.deserialize(json.loads(json.dumps(guided.serialize())), harness).metadata == guided.metadata


def test_only_weighted_benchmarks_load(harness):
    study = _study(harness)
    assert set(study.dataset_choice_weights) == {"quality", "dapo_math"}
    with pytest.raises(KeyError):
        AgentTrainingStudy(harness, {"nope": 1.0})
    assert BENCHMARKS["lca_bug_localization"].searchable is False and BENCHMARKS["bright"].scored is False


def test_rollout_report_summarizes_per_benchmark_and_teacher_kind(harness, tmp_path):
    config = AgentConfig(system_prompt="Solve.", model_name="tiny", user_prompt="What is six times seven?", dataset_task=_task(), max_turns=8,
                         metadata={"benchmark": "quality", "teacher_kind": "parallel_multi_agent"})
    turns = [SyntheticTurn("Let me delegate.", [("parallel_tool_call", {"calls": [{"name": "python", "arguments": {"code": "print(42)"}}, {"name": "subagent", "arguments": {"task": "check"}}]})],
                           ["[python] 42\n\n[subagent] Subagent finished (simulated, 0 turns). Answer: (none)"]),
             SyntheticTurn("Searching.", [("semantic_search", {"query": "six times seven"})], ["No passage matched the query."]),
             SyntheticTurn("Done.", [("submit_answer", {"answer": "42"})], ["Answer submitted: 42"])]
    agent = synthesize_agent(harness, config, turns)
    results = agent.run_results
    results.finish_reason, results.answer, results.score, results.duration = "submitted", "42", 1.0, 12.5
    results.num_output_tokens, results.num_turns = 120, 3
    agent.finished = True
    reporter = RolloutReporter(str(tmp_path / "report"), title="t")
    reporter.begin_rollouts(total=1, cached=0)
    reporter.report_agent_start(agent)
    reporter.report_agent_finish(agent)
    reporter.report_agent_scored(agent)
    reporter.finish()
    rows = reporter.widgets["summary"]["rows"]
    assert rows == [{"benchmark": "quality", "teacher kind": "parallel_multi_agent", "tasks": 1, "finish reasons": "submitted 1", "turns mean": "3.0",
                     "generated mean": "120", "compactions": 0, "subagents": 1, "search": 1, "shell": 0, "python": 1, "score": "1.00"}]
    saved = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert saved[0]["benchmark"] == "quality" and saved[0]["scores"] == [1.0]


def test_oracle_call_kwargs_and_cache_keys(harness):
    assert oracle_call_kwargs(None) == {"sampling_params": {"max_tokens": 4096}}
    low = oracle_call_kwargs("low")
    assert low["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "low"} and low["sampling_params"] == THINKING_SAMPLING
    with pytest.raises(ValueError):
        oracle_call_kwargs("high")
    keys = {config_key(AgentConfig(system_prompt="s", model_name="tiny", user_prompt="p", dataset_task=_task(), call_kwargs=oracle_call_kwargs(r))) for r in (None, "low", "medium")}
    assert len(keys) == 3


def test_long_task_text_is_offloaded_to_the_sandbox(harness, monkeypatch):
    written = {}

    class FakeEnv:
        def write_file(self, path, content):
            written[path] = content
            return True

    long_prompt = "A" * 15_000 + "MIDDLE" + "B" * 15_000
    agent = agent_module.Agent(harness, AgentConfig(system_prompt="s", model_name="tiny", user_prompt=long_prompt, max_turns=1))
    agent._env = FakeEnv()
    messages = agent.first_user_messages()
    text = messages[-1]["content"]
    assert written[agent_module.OFFLOAD_PATH] == long_prompt and "MIDDLE" not in text
    assert text.startswith("A" * 100) and text.endswith("B" * 100) and agent_module.OFFLOAD_PATH in text and "10,006 characters omitted" in text
    assert agent.agent_config.user_prompt == long_prompt                                              # the config is untouched
    short = agent_module.Agent(harness, AgentConfig(system_prompt="s", model_name="tiny", user_prompt="short", max_turns=1))
    short._env = FakeEnv()
    assert short.first_user_messages()[-1]["content"] == "short" and len(written) == 1


def test_draw_record_replays_and_head_comes_first(harness, tmp_path):
    """A recorded draw replays entry by entry even when a benchmark's task set drifts; head entries lead the first draw."""
    study = _study(harness)
    record = tmp_path / "draw_tasks.jsonl"
    first = study.generate_rollout_campaign_tasks(20, seed=3, record=record)
    assert record.exists() and len(record.read_text().splitlines()) == 20
    drifted = _study(harness)                                                        # two quality tasks gone, one new
    drifted.datasets["quality"] = LoadedDataset(dataset_id="quality___fixture", scorable_tasks={f"q{i}": _task(f"q{i}") for i in range(2, 31)})
    replayed = drifted.generate_rollout_campaign_tasks(20, seed=3, record=record)
    gone = {c.dataset_task.task_id for c in first} & {"q0", "q1"}
    assert len(replayed) == 20 - len(gone)
    kept = [c for c in first if c.dataset_task.task_id not in gone]
    assert [(c.dataset_task.task_id, c.metadata["template"]) for c in replayed] == [(c.dataset_task.task_id, c.metadata["template"]) for c in kept]
    assert [c.user_prompt for c in replayed] == [c.user_prompt for c in kept]
    assert study.generate_rollout_campaign_tasks(20, seed=99, record=record)[0].user_prompt == first[0].user_prompt   # the record wins over the seed

    head = [study.draw_entry(c) for c in first[5:8]]
    fresh = _study(harness).generate_rollout_campaign_tasks(12, seed=0, record=tmp_path / "second.jsonl", head=head)
    assert [c.user_prompt for c in fresh[:3]] == [c.user_prompt for c in first[5:8]]
    ids = [(c.dataset_task.dataset_id, c.dataset_task.task_id) for c in fresh]
    assert len(set(ids)) == 12                                                       # head tasks are not drawn again
