"""Four real HotpotQA problems through rollout, AC self-distillation and AC rollout.

The custom program retrieves evidence, asks one real solver subagent, then lets the
main agent answer. No mocks, synthetic trajectories or gold answers enter the program.
Each loss mode runs one epoch with two optimizer updates. Scores are printed as
same-problem diagnostics; this small test does not assert a quality improvement.

    uv run pytest activation/tests/test_basic_ac_self_distillation.py --gpu --slow -s

Both loss modes use the paired-history, loss_kind and folder-local row APIs.
"""
import json
import math
import uuid
from pathlib import Path

import pytest
import torch

from activation.ac_model import (
    ActivationContextModelConfig,
    ActivationContextStudyGenerator,
    ActivationContextTrainer,
    ActivationContextTrainingConfig,
    ActivationContextTrainingReporter,
)
from activation.agent import AgentConfig, AgenticProgram, AgentRunResult, RolloutReporter, SemanticSearchTool
from activation.agent.agent_utils import load_ac_rows
from activation.common.data_syncing import resolve_path
from activation.dataset.loaders import HotpotQADataset
from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig

SIDE_NAME, SIDE_ID = "self_distill_side", "Qwen/Qwen3.5-0.8B"
READER_NAME, READER_ID = "self_distill_reader", "Qwen/Qwen3.5-4B"
PROBLEMS = 4


class RetrieveThenSolveProgram(AgenticProgram):
    """Actual retrieval and a solver precede the main agent's own recorded answer."""
    name = "retrieve_then_solve"

    def execute(self) -> AgentRunResult:
        question = self.agent.agent_config.dataset_task.task_datum["question"]
        evidence = self.agent.run_tool("semantic_search", query=question, top_k=2)
        if evidence.is_error:
            raise RuntimeError(evidence.output)
        solver = self.agent.run_subagent(
            f"Find evidence for this question using semantic_search, then submit a short answer: {question}",
            max_duration=60.0,
        )
        self.agent.augment_context([
            evidence,
            solver,
            "Use the retrieved evidence and check the solver's suggestion. You may search again. "
            "Submit only the short answer to the original question.",
        ])
        return self.agent.run()


def _harness(ac_name: str, side_lora: str, reader_lora: str) -> HarnessRuntime:
    harness = HarnessRuntime(HarnessRuntimeConfig(
        model_configs={
            SIDE_NAME: ModelConfig(SIDE_NAME, SIDE_ID),
            READER_NAME: ModelConfig(READER_NAME, READER_ID),
        },
        agent_max_concurrent=2,
    ))
    harness.module_manager.register_lora(reader_lora, READER_NAME, rank=16)
    harness.module_manager.register_ac_model(ActivationContextModelConfig(
        ac_name, SIDE_NAME, side_lora, READER_NAME, reader_lora,
        side_lora_rank=16,
    ))
    return harness


def _tasks(harness: HarnessRuntime):
    # Select by source difficulty and question length, never by model success.
    # The bounded source prefix supplies ordinary distractor documents as well.
    dataset = HotpotQADataset.load(harness, max_examples=128, split="train", seed=0)
    harness.dataset_manager.register_dataset(dataset)
    harness.dataset_manager.build_bm25_indexes()
    easy = [task for task in dataset.scorable_tasks.values() if task.task_datum.get("level") == "easy"]
    assert len(easy) >= PROBLEMS, f"expected four easy questions in the source prefix, found {len(easy)}"
    tasks = sorted(easy, key=lambda task: (len(task.task_datum["question"]), task.task_id))[:PROBLEMS]
    for task in tasks:
        print(f"HotpotQA {task.task_id}: {task.task_datum['question']}")
    return tasks


def _configs(tasks, ac_name: str | None, reader_lora: str | None) -> list[AgentConfig]:
    return [AgentConfig(
        agent_name=f"hotpot_{index}",
        system_prompt=(
            "Answer the question from the Wikipedia corpus using semantic_search. "
            "Check the evidence and call submit_answer with the short answer only."
        ),
        user_prompt=task.agent_prompt,
        dataset_task=task,
        model_name=READER_NAME,
        lora_name=reader_lora,
        ac_model_name=ac_name,
        tools={"semantic_search": (SemanticSearchTool, {"dataset_id": task.dataset_id, "max_chars": 1200}), "subagent": None},
        agentic_program=(RetrieveThenSolveProgram, {}),
        metadata={"program": RetrieveThenSolveProgram.name, "benchmark": "hotpotqa"},
        call_kwargs={"sampling_params": {"max_tokens": 1024, "temperature": 0.0}},
        max_turns=4,
        max_duration=120.0,
        max_tool_errors=2,
        compaction_threshold_tokens=16_384,
        absolute_trajectory_cap=32_000,
    ) for index, task in enumerate(tasks)]


def _rollouts(harness, configs, caching_id: str, report_folder: Path, seed: int) -> list[AgentRunResult]:
    reporter = RolloutReporter(str(report_folder), title=f"HotpotQA: {caching_id}")
    try:
        runs = harness.rollout_manager.perform_single_rollouts(
            configs, seed=seed, caching_id=caching_id, perform_scoring=True, reporter=reporter,
        )
    finally:
        reporter.finish()
    assert len(runs) == PROBLEMS
    for run in runs:
        assert run.finish_reason != "error", run.score_feedback
        assert len(run.subagent_results) == 1
        assert run.subagent_results[0].finish_reason != "error", run.subagent_results[0].score_feedback
        assert run.score is not None and math.isfinite(run.score)
        print(f"{run.agent_config.agent_name}: score={run.score:.3f}, answer={run.answer!r}, "
              f"finish={run.finish_reason}, rows={run.num_ac_rows}")
    return runs


def _segments(run: AgentRunResult):
    for segment in [*run.compactions, run]:
        yield segment
        for child in segment.subagent_results:
            yield from _segments(child)


def _spans(runs: list[AgentRunResult]):
    for root in runs:
        for segment in _segments(root):
            yield from segment.prompt_ac_spans
            for step in segment.trajectory:
                yield from step.get("ac_spans", [])


def _assert_coverage(runs, items) -> None:
    recorded = sum(
        len(step.get("token_ids") or [])
        for root in runs for segment in _segments(root) for step in segment.trajectory
        if step["role"] == "assistant"
    )
    selected = sum(len(ids) for item in items for ids in item.assistant_token_ids)
    assert selected == recorded > 0, (selected, recorded)
    assert all(item.weight == 1.0 for item in items)


def _assert_eval(summary, count: int) -> None:
    assert summary["items"] == count > 0
    assert summary["dropped_too_long"] == 0
    assert summary["loss"] is not None and math.isfinite(summary["loss"])


def _check_saved_rows(runs, caching_id: str, width: int) -> None:
    folder = resolve_path(f"ROLLOUTS/{caching_id}", create=False)
    assert (folder / "rollouts.jsonl").is_file()
    assert (folder / "saved_tensors").is_dir()
    spans = list(_spans(runs))
    assert spans, "the AC rollouts must produce actual captured rows"
    for span in spans:
        assert "rows" not in span, "successfully saved results must release their tensor fields"
        path = Path(span["row_path"])
        assert path.is_absolute() and path.is_file()
        assert path.parent == (folder / "saved_tensors").resolve()
    rows = load_ac_rows(spans[0])
    assert rows.device.type == "cpu" and not rows.requires_grad
    assert rows.shape == (spans[0]["length"], width)
    assert torch.isfinite(rows).all()


def _release(harness, ac_name: str) -> None:
    for loaded in harness.loaded_models.values():
        loaded.engine_to_device(FREE_DEVICE)
    ac_model = harness.module_manager.get_ac_model(ac_name)
    ac_model.release()
    for model_name, lora_name in (
        (READER_NAME, ac_model.config.target_model_lora_name),
        (SIDE_NAME, ac_model.config.base_side_model_lora_name),
    ):
        harness.module_manager.free_lora(model_name, lora_name)
    for loaded in harness.loaded_models.values():
        loaded.model_to_device(FREE_DEVICE)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("loss_kind", ["kl", "sft"])
def test_basic_ac_self_distillation(loss_kind: str) -> None:
    # A new namespace prevents cached rollouts/checkpoints from hiding this run.
    run_id = f"hotpot_ac_{loss_kind}_{uuid.uuid4().hex[:10]}"
    ac_name, side_lora, reader_lora = run_id, f"{run_id}_side", f"{run_id}_reader"
    folder = resolve_path(f"AC_SELF_DISTILLATION_TEST/{run_id}")
    harness = _harness(ac_name, side_lora, reader_lora)
    training_reporter = None
    try:
        tasks = _tasks(harness)
        before_configs = _configs(tasks, ac_name=None, reader_lora=None)
        before_runs = _rollouts(harness, before_configs, f"{run_id}_text", folder / "text_rollouts", seed=0)
        assert all(run.num_ac_rows == 0 for run in before_runs)
        assert all(len(run.injected_input) >= 2 for run in before_runs)

        config = ActivationContextTrainingConfig(
            loss_kind=loss_kind,
            num_epochs=1,
            updates_per_epoch=2,
            reporting_interval=1.0,
            completion_samples=1,
            completion_max_new_tokens=32,
            logits_chunk_tokens=128,
            seed=0,
        )
        study = ActivationContextStudyGenerator(harness, ac_name)
        # Explicitly include all four for this execution example in both modes.
        # The SFT default score_1 selection is tested privately; scores stay real.
        items = study.items_from_run_results(before_runs, loss_kind=loss_kind, selection="all")
        _assert_coverage(before_runs, items)
        assert len(items) >= PROBLEMS
        trainer = ActivationContextTrainer(harness, config)
        training_reporter = ActivationContextTrainingReporter(str(folder / "training"), f"HotpotQA self-distillation: {loss_kind}")
        before = trainer.eval(ac_name, items, reporter=training_reporter, reporting_name="before")
        _assert_eval(before, len(items))
        stats = trainer.train(ac_name, items, reporting_data=items, reporter=training_reporter)
        after = stats.reporting
        _assert_eval(after, len(items))
        assert stats.steps == 2 and len(stats.epochs) == 1
        assert stats.loss_kind == loss_kind and all(math.isfinite(value) for value in stats.loss)
        assert stats.checkpoint_path and (Path(stats.checkpoint_path) / "ac_modules.pt").is_file()
        assert stats.target_lora_checkpoint_path and Path(stats.target_lora_checkpoint_path).is_dir()
        if loss_kind == "sft":
            assert stats.teacher_tokens == 0 and all(row["teacher_tokens"] == 0 for row in after["per_item"])
            assert after["kl"] is None and after["agreement"] is None
        print(json.dumps(stats.summarize(), indent=2))

        ac_model = harness.module_manager.get_ac_model(ac_name)
        completed = ac_model.load_completed_epoch()
        assert resolve_path(completed["ac_checkpoint"], create=False).resolve() == Path(stats.checkpoint_path).resolve()
        reloaded = ActivationContextTrainer(harness, config).eval(ac_name, items)
        _assert_eval(reloaded, len(items))
        assert math.isclose(reloaded["loss"], after["loss"], rel_tol=5e-3, abs_tol=5e-3)
        harness.module_manager.exchange_lora(READER_NAME, reader_lora)

        ac_configs = _configs(tasks, ac_name=ac_name, reader_lora=reader_lora)
        ac_cache = f"{run_id}_ac"
        ac_runs = _rollouts(harness, ac_configs, ac_cache, folder / "ac_rollouts", seed=1)
        assert all(run.ac_model_name == ac_name and run.lora_name == reader_lora for run in ac_runs)
        assert all(len(run.prompt_ac_spans) >= 2 for run in ac_runs), "program should inject multiple actual AC parts"
        _check_saved_rows(ac_runs, ac_cache, ac_model.d_target)

        cached = harness.rollout_manager.perform_single_rollouts(ac_configs, seed=1, caching_id=ac_cache, perform_scoring=True)
        assert len(cached) == PROBLEMS
        assert [run.answer for run in cached] == [run.answer for run in ac_runs]
        _check_saved_rows(cached, ac_cache, ac_model.d_target)
        # Native AC histories also convert to differentiable training data.
        ac_items = study.items_from_run_results(cached, loss_kind=loss_kind, selection="all")
        _assert_coverage(cached, ac_items)

        summary = {
            "loss_kind": loss_kind,
            "task_ids": [task.task_id for task in tasks],
            "text_scores": [run.score for run in before_runs],
            "ac_scores": [run.score for run in ac_runs],
            "training_items": len(items),
            "ac_history_items": len(ac_items),
            "before_loss": before["loss"],
            "after_loss": after["loss"],
            "reloaded_loss": reloaded["loss"],
            "checkpoint": completed,
            "quality_scope": "same four training questions; diagnostic only",
        }
        (folder / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        if training_reporter is not None:
            training_reporter.finish()
        _release(harness, ac_name)
