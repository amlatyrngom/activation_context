"""
Slice 3a1 key tests: two AC training epochs per item kind on real data, small on purpose (side
Qwen3.5-0.8B, target Qwen3.5-4B, a few dozen items). The asserts are about mechanics (items built,
finite KL, matching evaluation items, checkpoints written); AC versus empty-context quality is a reported diagnostic,
not a dominance guarantee from a few held examples. The KL values are
printed and reported to HTML under AC_TRAINING_TEST/. The study questions (traj_qa, rag_qa) come
from the dataset study engine (dataset_study_qa_model_name), cached by caching_id.

  uv run pytest activation/tests/test_basic_agent_ac_training.py --gpu --slow -s
"""
import json
import math
import os

import pytest

from activation.ac_model import (
    ActivationContextModelConfig,
    ActivationContextStudyGenerator,
    ActivationContextTrainer,
    ActivationContextTrainingConfig,
    ActivationContextTrainingReporter,
    ActivationContextTrainingItem,
)
from activation.ac_model.ac_model_study import split_items
from activation.common.data_syncing import resolve_path
from activation.dataset.loaders import BrightDataset, NemotronMathDataset, OpenSweTracesDataset, S1DeepResearchDataset, aime_problem_statements
from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig

SIDE_NAME, SIDE_ID = "qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B"
TARGET_NAME, TARGET_ID = "qwen3.5-4b", "Qwen/Qwen3.5-4B"
AC_NAME, SIDE_LORA, TARGET_LORA = "ac_dev", "ac_side", "ac_target"
QA_MODEL_NAME = "RedHatAI/Qwen3.5-9B-FP8-dynamic"
ITEMS, HELD = 48, 8
CACHING_ID = "ac_training_test_3a1_9b"


def _harness(with_study: bool) -> HarnessRuntime:
    model_configs = {SIDE_NAME: ModelConfig(SIDE_NAME, SIDE_ID), TARGET_NAME: ModelConfig(TARGET_NAME, TARGET_ID)}
    if with_study:
        model_configs[QA_MODEL_NAME] = ModelConfig(QA_MODEL_NAME, QA_MODEL_NAME)
    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs=model_configs, dataset_study_qa_model_name=QA_MODEL_NAME if with_study else None))
    harness.module_manager.register_lora(TARGET_LORA, TARGET_NAME, rank=64)
    harness.module_manager.register_ac_model(ActivationContextModelConfig(AC_NAME, SIDE_NAME, SIDE_LORA, TARGET_NAME, TARGET_LORA))
    return harness


def _trainer(harness: HarnessRuntime) -> ActivationContextTrainer:
    return ActivationContextTrainer(harness, ActivationContextTrainingConfig(examples_per_update=8, num_epochs=2))


def _run(harness: HarnessRuntime, items: list[ActivationContextTrainingItem], name: str) -> None:
    """Two epochs, exact held/reference comparisons with matched adapters, and paired saves."""
    assert len(items) > HELD + 8, f"only {len(items)} items"
    held, train, dropped_shared = split_items(items, HELD, len(items))
    assert train and held
    print(f"{len(train)} train, {len(held)} held, {dropped_shared} shared-origin items excluded")
    references = ActivationContextStudyGenerator(harness, AC_NAME).transform_for_eval_reference(held, "no_context")
    trainer = _trainer(harness)
    reporter = ActivationContextTrainingReporter(str(resolve_path(f"AC_TRAINING_TEST/{name}")), f"AC training test: {name}")
    try:
        before = trainer.eval(AC_NAME, held, release=False, reporter=reporter, reference_name="untrained_ac")
        no_context = trainer.eval(AC_NAME, references, release=False, reporter=reporter, reporting_name="initial_no_context", reference_name="no_context")
        stats = trainer.train(AC_NAME, train, reporting_data=held, reporter=reporter)
        after = stats.reporting
        matched_no_context = trainer.eval(AC_NAME, references, reporter=reporter, reporting_name="final_no_context")
        assert len(stats.epochs) == 2
        assert all(len(epoch.samples) == min(4, after["items"]) for epoch in stats.epochs)
        print(f"\n=== {name}: held KL untrained {before['kl']:.4f}, no-context {no_context['kl']:.4f}, after two epochs {after['kl']:.4f} "
              f"(agreement {before['agreement']:.3f} -> {after['agreement']:.3f}) ===")
        print(json.dumps(stats.summarize(), indent=2))
        assert stats.steps >= 2 and all(value == value and value < float("inf") for value in stats.kl)
        assert stats.checkpoint_path and os.path.exists(os.path.join(stats.checkpoint_path, "ac_modules.pt"))
        assert stats.target_lora_checkpoint_path
        assert after["items"] == before["items"] == no_context["items"] == matched_no_context["items"] > 0
        assert all(math.isfinite(result["kl"]) for result in (before, no_context, after, matched_no_context))
        comparison = {"initial_ac": before, "initial_no_context": no_context,
                      "final_ac": after, "final_no_context": matched_no_context,
                      "ac_kl_advantage": matched_no_context["kl"] - after["kl"]}
        resolve_path(f"AC_TRAINING_TEST/{name}/comparison.json").write_text(json.dumps(comparison, indent=2))
        print(f"Matched-adapter AC KL advantage (positive is better): {comparison['ac_kl_advantage']:+.6f}")
    finally:
        reporter.finish()
        ac_model = harness.module_manager.get_ac_model(AC_NAME)
        ac_model.release()
        harness.module_manager.free_lora(ac_model.config.target_model_name, ac_model.config.target_model_lora_name)
        harness.module_manager.free_lora(ac_model.config.base_side_model_name, ac_model.config.base_side_model_lora_name)   # adapters first: a base with adapters cannot be freed
        for loaded in harness.loaded_models.values():
            loaded.engine_to_device(FREE_DEVICE)
            loaded.model_to_device(FREE_DEVICE)


@pytest.mark.gpu
@pytest.mark.slow
def test_ac_compaction() -> None:
    harness = _harness(with_study=False)
    swe = OpenSweTracesDataset.load(harness, max_examples=ITEMS, min_chars=40_000)
    math = NemotronMathDataset.load(harness, max_examples=ITEMS, subset="tir", excluded_problems=aime_problem_statements(), min_chars=40_000, min_tool_calls=1)
    generator = ActivationContextStudyGenerator(harness, AC_NAME)
    items = generator.generate_compaction_samples(swe.dataset_id, ITEMS // 2, seed=0) + generator.generate_compaction_samples(math.dataset_id, ITEMS // 2, seed=0)
    print(f"\n{len(items)} compaction items; depths {sorted({item.info['depth'] for item in items})}; mid-turn cuts {sum(bool(item.teacher_partial_text) for item in items)}")
    _run(harness, items, "compaction")


@pytest.mark.gpu
@pytest.mark.slow
def test_ac_trajectory_qa() -> None:
    harness = _harness(with_study=True)
    s1 = S1DeepResearchDataset.load(harness, max_examples=ITEMS)
    swe = OpenSweTracesDataset.load(harness, max_examples=ITEMS)
    harness.dataset_manager.build_bm25_indexes()
    generator = ActivationContextStudyGenerator(harness, AC_NAME)
    items = (generator.generate_trajectory_qa_samples(s1.dataset_id, ITEMS // 2, seed=0, caching_id=CACHING_ID)
             + generator.generate_trajectory_qa_samples(swe.dataset_id, ITEMS // 2, seed=0, caching_id=CACHING_ID))
    harness.loaded_models[QA_MODEL_NAME].engine_to_device(FREE_DEVICE)
    print(f"\n{len(items)} traj_qa items; distractors {sorted({item.info['distractors'] for item in items})}")
    for item in items[:3]:
        print(f"  [{item.item_id}] Q: {item.ac_prefix[-1]['content'][-1]['text'][-160:]!r} A: {item.completion_text[:80]!r}")
    _run(harness, items, "traj_qa")


@pytest.mark.gpu
@pytest.mark.slow
def test_ac_rag_qa() -> None:
    harness = _harness(with_study=True)
    bright = BrightDataset.load(harness, max_examples=100, domain="biology", max_corpus_documents=2000)
    harness.dataset_manager.build_bm25_indexes()
    generator = ActivationContextStudyGenerator(harness, AC_NAME)
    items = generator.generate_rag_qa_samples(bright.dataset_id, ITEMS, seed=0, caching_id=CACHING_ID)
    harness.loaded_models[QA_MODEL_NAME].engine_to_device(FREE_DEVICE)
    print(f"\n{len(items)} rag_qa items; passages per item {sorted({item.info['passages'] for item in items})}")
    for item in items[:3]:
        print(f"  [{item.item_id}] Q: {item.in_context_prefix[-1]['content'][-160:]!r} A: {item.completion_text[:80]!r}")
    _run(harness, items, "rag_qa")
