import json

import pytest

from activation.dataset.dataset import DataOrigin
from activation.dataset.loaders import BrightDataset
from activation.harness import (
    HarnessRuntime,
    HarnessRuntimeConfig,
    ModelConfig,
    FREE_DEVICE,
    SUPPORTS_FP4,
)

pytestmark = pytest.mark.gpu

STUDY_NUM_QUESTIONS = 16
if SUPPORTS_FP4:
    QA_MODEL_NAME = "unsloth/Qwen3.8-27B-NVFP4"
    LABEL_MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
else:
    QA_MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"
    LABEL_MODEL_NAME = "Qwen/Qwen3.6-35B-A3B-FP8"

EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
BRIGHT_DOMAIN = "biology"


def test_basic_dataset_study():
    harness_config = HarnessRuntimeConfig(
        model_configs={
            QA_MODEL_NAME: ModelConfig(QA_MODEL_NAME, QA_MODEL_NAME),
            LABEL_MODEL_NAME: ModelConfig(LABEL_MODEL_NAME, LABEL_MODEL_NAME),
        },
        dataset_study_qa_model_name=QA_MODEL_NAME,
        dataset_study_label_model_name=LABEL_MODEL_NAME,
        # Engine and chat kwargs come from VLLMWrapper's recommendations; batch sizes from the node.
    )
    harness = HarnessRuntime(harness_config)
    dataset = BrightDataset.load(
        harness,
        max_examples=20,
        domain=BRIGHT_DOMAIN,
        max_corpus_documents=100,
    )
    harness.dataset_manager.build_bm25_indexes()
    native = [
        example
        for example in dataset.labeled_retrieval_examples.values()
        if example.origin == DataOrigin.NATIVE
    ]
    native_doc_ids_before = {
        example.example_id: (list(example.positive_doc_ids or []), list(example.hard_negative_doc_ids or []))
        for example in native
    }
    try:
        harness.dataset_manager.synthesize_study_examples_qa(dataset.dataset_id, STUDY_NUM_QUESTIONS)
        synthetic = [
            example
            for example in dataset.labeled_retrieval_examples.values()
            if example.origin == DataOrigin.SYNTHETIC
        ]
        # Every sampled chunk yields exactly one question.
        assert len(synthetic) == STUDY_NUM_QUESTIONS
        for example in synthetic:
            assert example.gold_answers and example.gold_answers[0]
            assert example.positive_doc_ids and example.positive_chunk_ids
        print(f"\n=== {dataset.dataset_id}: {len(synthetic)} synthetic questions ===")
        for example in synthetic[:4]:
            print(f"\n[{example.example_id}] (doc {example.positive_doc_ids[0]})")
            print(f"  Q: {example.query}")
            print(f"  A: {example.gold_answers[0]}")

        # Labeling pass 1: the synthetic questions.
        labeled = harness.dataset_manager.label_study_examples(
            dataset.dataset_id, STUDY_NUM_QUESTIONS, synthetic_only=True,
        )
        assert labeled and all(example.origin == DataOrigin.SYNTHETIC for example in labeled)
        for example in labeled:
            assert example.oracle_labeled
            assert not set(example.positive_chunk_ids) & set(example.hard_negative_chunk_ids or [])
        assert any(example.hard_negative_chunk_ids for example in labeled)
        print(f"\n=== {len(labeled)} synthetic questions labeled ===")
        for example in labeled[:4]:
            print(f"\n[{example.example_id}]")
            print(f"  positives: {example.positive_chunk_ids}")
            print(f"  hard negatives: {example.hard_negative_chunk_ids}")

        # Labeling pass 2: the native examples. Doc-level ids are never touched; chunk-level
        # positives arrive from the oracle or by inheritance from the known positive documents.
        stats_before = dataset.stats.study_num_label_positives + dataset.stats.study_num_label_inherited_positives
        labeled_native = harness.dataset_manager.label_study_examples(
            dataset.dataset_id, len(native), synthetic_only=False,
        )
        assert labeled_native and all(example.origin == DataOrigin.NATIVE for example in labeled_native)
        for example in labeled_native:
            assert example.oracle_labeled
            assert (list(example.positive_doc_ids or []), list(example.hard_negative_doc_ids or [])) == native_doc_ids_before[example.example_id]
            assert not set(example.positive_chunk_ids or []) & set(example.hard_negative_chunk_ids or [])
        stats_after = dataset.stats.study_num_label_positives + dataset.stats.study_num_label_inherited_positives
        assert stats_after > stats_before
        print(f"\n=== {len(labeled_native)} native examples labeled ===")
        for example in labeled_native[:4]:
            print(f"\n[{example.example_id}]")
            print(f"  positives: {example.positive_chunk_ids}")
            print(f"  hard negatives: {example.hard_negative_chunk_ids}")
        print("\n=== dataset stats ===")
        print(json.dumps(dataset.stats.summarize(), indent=2))
    finally:
        for loaded_model in harness.loaded_models.values():
            loaded_model.engine_to_device(FREE_DEVICE)
    # Residency sequence: load generator → generate → free → load labeler → label → free.
    # Holds because this test never loads the HF model or the embedding layer (bm25 only).
    loads = [name for name, _ in harness.harness_stats.model_loading_times]
    frees = [name for name, _ in harness.harness_stats.model_freeing_times]
    assert loads == [QA_MODEL_NAME, LABEL_MODEL_NAME], loads
    assert frees == [QA_MODEL_NAME, LABEL_MODEL_NAME], frees

    print("\n=== harness stats ===")
    print(json.dumps(harness.harness_stats.summarize(), indent=2))