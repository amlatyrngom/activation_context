"""
The redo policy: cached rows with listed finish reasons (and teacher kinds) read as absent and are rolled out again, the new
row superseding the old one at the next load; `only` skips the tasks without a cached row. And the draw guard of the
canonical script: a caching id keeps the seed, total, weights, reasoning and model it was first drawn with.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from activation.agent import AgentConfig
from activation.agent.agent_config import AgentRunResult
from activation.agent.rollout_caching import CACHE_FILENAME, RedoPolicy, RolloutCache, config_key
from activation.agent.rollout_manager import RolloutManager
from activation.common.data_syncing import SYNC_ROOT_ENV
from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig

TINY = os.environ.get("TINY_QWEN35", "/workspace/IB/TMP/AGENT_ROLLOUTS/slice3_apply_20260909T103843Z/tiny_qwen35")


def _config(i: int, kind: str) -> AgentConfig:
    return AgentConfig(system_prompt="s", model_name="tiny", user_prompt=f"u{i}", max_duration=600, metadata={"teacher_kind": kind})


def _row(config: AgentConfig, finish_reason: str, answer: str = "old") -> dict:
    result = AgentRunResult(agent_config=config, seed=0, duration=1.0, finish_reason=finish_reason, answer=answer)
    return result.serialize() | {"config_key": config_key(config)}


def test_redo_policy_reads_matching_rows_as_absent_and_new_rows_supersede(tmp_path, monkeypatch):
    monkeypatch.setenv(SYNC_ROOT_ENV, str(tmp_path))
    configs = [_config(0, "base"), _config(1, "parallel_multi_agent"), _config(2, "parallel_multi_agent"), _config(3, "sequential_multi_agent")]
    rows = [_row(configs[0], "max_duration"), _row(configs[1], "max_duration"), _row(configs[2], "submitted"), _row(configs[3], "max_turns")]
    folder = tmp_path / "ROLLOUTS" / "redo_test"
    folder.mkdir(parents=True)
    (folder / CACHE_FILENAME).write_text("".join(json.dumps(row) + "\n" for row in rows))

    plain = RolloutCache("redo_test")
    assert all(plain.get(config_key(c), 0) is not None for c in configs)

    policy = RedoPolicy(finish_reasons=frozenset({"max_duration", "max_turns"}), teacher_kinds=frozenset({"parallel_multi_agent", "sequential_multi_agent"}))
    cache = RolloutCache("redo_test", redo=policy)
    assert cache.get(config_key(configs[0]), 0) is not None                        # base: kind not listed, kept
    assert cache.get(config_key(configs[1]), 0) is None and cache.is_stale(config_key(configs[1]), 0)
    assert cache.get(config_key(configs[2]), 0) is not None                        # submitted, kept
    assert cache.get(config_key(configs[3]), 0) is None and cache.is_stale(config_key(configs[3]), 0)
    assert not cache.is_stale(config_key(_config(9, "base")), 0)                   # never cached: not stale either

    cache.append(AgentRunResult(agent_config=configs[1], seed=0, duration=2.0, finish_reason="submitted", answer="new"))
    reloaded = RolloutCache("redo_test", redo=policy)
    assert reloaded.get(config_key(configs[1]), 0)["answer"] == "new"               # last row per key wins; the old one stays in the file
    assert sum(1 for line in (folder / CACHE_FILENAME).read_text().splitlines() if line) == 5


@pytest.fixture(scope="module")
def harness():
    if not os.path.isdir(TINY):
        pytest.skip(f"tiny fixture missing: {TINY}")
    return HarnessRuntime(HarnessRuntimeConfig(model_configs={"tiny": ModelConfig("tiny", TINY)}))


def test_redo_only_runs_the_stale_rows_alone(harness, monkeypatch, tmp_path):
    monkeypatch.setenv(SYNC_ROOT_ENV, str(tmp_path))
    ran: list[str] = []

    def fake_run_one(self, config, seed, perform_scoring, reporter):
        ran.append(config.user_prompt)
        return AgentRunResult(agent_config=config, seed=seed, duration=0.1, finish_reason="submitted", answer="redone")

    loaded_model = harness.loaded_models["tiny"]
    monkeypatch.setattr(RolloutManager, "_run_one", fake_run_one)
    monkeypatch.setattr(RolloutManager, "_warm_up", lambda self, *args: None)
    monkeypatch.setattr(RolloutManager, "_build_indexes", lambda self, *args: None)
    monkeypatch.setattr(loaded_model, "ensure_engine_loaded", lambda: None)
    configs = [_config(i, "parallel_multi_agent") for i in range(4)]                # 0 cut, 1 fine, 2 cut, 3 never cached
    folder = tmp_path / "ROLLOUTS" / "redo_only"
    folder.mkdir(parents=True)
    (folder / CACHE_FILENAME).write_text("".join(json.dumps(row) + "\n" for row in
                                                 [_row(configs[0], "max_duration"), _row(configs[1], "submitted"), _row(configs[2], "max_turns")]))
    harness.harness_config.agent_max_concurrent = 2
    try:
        policy = RedoPolicy(finish_reasons=frozenset({"max_duration", "max_turns"}), only=True)
        results = RolloutManager(harness).perform_single_rollouts(configs, perform_scoring=False, caching_id="redo_only", redo=policy)
    finally:
        harness.harness_config.agent_max_concurrent = None
    assert sorted(ran) == ["u0", "u2"]
    assert [r.answer for r in results] == ["redone", "old", "redone"]               # task 3 skipped; cached row 1 replayed
    assert RolloutCache("redo_only").get(config_key(configs[0]), 0)["answer"] == "redone"


def test_draw_guard_records_then_refuses_a_changed_draw(tmp_path, monkeypatch):
    monkeypatch.setenv(SYNC_ROOT_ENV, str(tmp_path))
    from activation.bench.canonical_training.generate_agent_teacher_trajectories import check_draw
    draw = {"seed": 0, "total": 20000, "dataset_weights": {"musique": 0.2}, "teacher_weights": {"all:base": 0.2}, "reasoning": "medium", "model_id": "m"}
    path = check_draw("guard", draw)
    assert path.exists() and json.loads(path.read_text())["total"] == 20000
    check_draw("guard", dict(draw))                                                 # same draw: fine
    with pytest.raises(SystemExit, match="total"):
        check_draw("guard", draw | {"total": 30000})
    check_draw("guard", draw | {"total": 30000}, allow_change=True)                 # explicit override
    assert json.loads(path.read_text())["total"] == 20000                           # the record is never rewritten
