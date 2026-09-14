"""
The rollout wall-clock deadline (no job starts after it, started jobs get max_duration clipped, only finished results come
back) and the S3 client stub (environment-only settings, the expected S3 API calls) on a fake boto3.
"""
import os
import sys
import threading
import time
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from activation.agent import AgentConfig
from activation.agent.agent_config import AgentRunResult
from activation.agent.rollout_manager import RolloutManager
from activation.common import s3_client as s3_module
from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig

TINY = os.environ.get("TINY_QWEN35", "/workspace/IB/TMP/AGENT_ROLLOUTS/slice3_apply_20260909T103843Z/tiny_qwen35")


@pytest.fixture(scope="module")
def harness():
    if "/" in TINY and TINY.startswith(("/", ".")) and not os.path.isdir(TINY):
        pytest.skip(f"tiny fixture missing: {TINY}")
    return HarnessRuntime(HarnessRuntimeConfig(model_configs={"tiny": ModelConfig("tiny", TINY)}))


def test_deadline_skips_late_jobs_and_clips_started_ones(harness, monkeypatch):
    """12 jobs of 0.2 s on a pool of 2 under a 0.5 s deadline: about six start (each clipped to the time left), the rest are skipped."""
    seen: list[float] = []
    lock = threading.Lock()

    def fake_run_one(self, config, seed, perform_scoring, reporter):
        with lock:
            seen.append(config.max_duration)
        time.sleep(0.2)
        return AgentRunResult(agent_config=config, seed=seed, duration=0.2, finish_reason="submitted")

    loaded_model = harness.loaded_models["tiny"]
    monkeypatch.setattr(RolloutManager, "_run_one", fake_run_one)
    monkeypatch.setattr(RolloutManager, "_warm_up", lambda self, *args: None)          # no engine in this test
    monkeypatch.setattr(RolloutManager, "_build_indexes", lambda self, *args: None)
    monkeypatch.setattr(loaded_model, "ensure_engine_loaded", lambda: None)
    harness.harness_config.agent_max_concurrent = 2
    try:
        manager = RolloutManager(harness)
        configs = [AgentConfig(system_prompt="s", model_name="tiny", user_prompt=f"u{i}", max_duration=600) for i in range(12)]
        began = time.time()
        results = manager.perform_single_rollouts(configs, perform_scoring=False, max_wall_seconds=0.5)
        elapsed = time.time() - began
    finally:
        harness.harness_config.agent_max_concurrent = None
    assert elapsed < 3.0
    assert all(result is not None for result in results) and 0 < len(results) < 12   # only the finished results, fewer than the jobs
    assert len(results) == len(seen)
    assert seen and all(max_duration <= 0.5 for max_duration in seen)                  # every started job was clipped to the time left
    order = [next(i for i, config in enumerate(configs) if config is result.agent_config) for result in results]
    assert order == sorted(order)                                                         # in job order, under the jobs' own configs


def test_grouped_rollouts_unchanged_without_deadline(harness, monkeypatch):
    monkeypatch.setattr(RolloutManager, "_run_one", lambda self, config, seed, perform_scoring, reporter: AgentRunResult(
        agent_config=config, seed=seed, duration=0.0, finish_reason="submitted"))
    monkeypatch.setattr(RolloutManager, "_warm_up", lambda self, *args: None)
    monkeypatch.setattr(RolloutManager, "_build_indexes", lambda self, *args: None)
    monkeypatch.setattr(harness.loaded_models["tiny"], "ensure_engine_loaded", lambda: None)
    configs = [AgentConfig(system_prompt="s", model_name="tiny", user_prompt=f"u{i}") for i in range(2)]
    groups = RolloutManager(harness).perform_grouped_rollouts(configs, group_count=3, perform_scoring=False)
    assert [[r.seed for r in group] for group in groups] == [[0, 1, 2], [0, 1, 2]] and all(r is not None for g in groups for r in g)


class _FakeS3:
    def __init__(self, missing: set[str]):
        self.calls: list[tuple] = []
        self.missing = missing

    def upload_file(self, filename, bucket, key):
        self.calls.append(("upload_file", filename, bucket, key))

    def head_object(self, Bucket, Key):
        self.calls.append(("head_object", Bucket, Key))
        if Key in self.missing:
            error = Exception("not found")
            error.response = {"Error": {"Code": "404"}}
            raise error
        return {}


def test_s3_client_reads_env_and_calls_the_s3_api(monkeypatch, tmp_path):
    fake = _FakeS3(missing={"absent"})
    boto3 = types.ModuleType("boto3")
    boto3.client = lambda service, **kwargs: fake if service == "s3" else None
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setattr(s3_module, "_load_dotenv_once", lambda: None)   # the environment below is the whole configuration
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION", "ACTIVATION_S3_BUCKET"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="AWS_ACCESS_KEY_ID"):
        s3_module.S3Client()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("ACTIVATION_S3_BUCKET", "bucket-from-env")
    client = s3_module.S3Client()
    assert client.bucket == "bucket-from-env" and s3_module.DEFAULT_BUCKET == "activation-context-artifacts-bucket"
    local = tmp_path / "a.txt"
    local.write_text("x")
    client.upload_file(local, "folder/a.txt")
    assert client.exists("folder/a.txt") is True and client.exists("absent") is False
    assert fake.calls == [("upload_file", str(local), "bucket-from-env", "folder/a.txt"),
                          ("head_object", "bucket-from-env", "folder/a.txt"), ("head_object", "bucket-from-env", "absent")]
