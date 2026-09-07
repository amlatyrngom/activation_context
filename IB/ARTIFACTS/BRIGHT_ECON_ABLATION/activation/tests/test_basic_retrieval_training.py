import json
import math
import os

import pytest
import torch

from activation.dataset.loaders import MsMarcoDataset
from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
from activation.harness.hf_utils import FREE_DEVICE
from activation.retrieval import (
    METRIC_NAMES,
    RetrievalModel,
    RetrievalReporter,
    RetrievalTrainer,
    RetrievalTrainingConfig,
    StandardRetrievalACModel,
)

pytestmark = pytest.mark.slow  # Real 0.6B weights; sized to stay under ~4 GB and a few minutes on CPU.

BASE_MODEL_NAME = "qwen3-0.6b"
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
LORA_NAME = "qwen3-0.6b-retrieval-lora"
AC_NAME = "qwen3-0.6b-retrieval-ac"
REPORT_FOLDER = "IB/TMP/RETRIEVAL_SLICE1/test_basic_retrieval_training"


def test_basic_retrieval_training():
    harness = HarnessRuntime(HarnessRuntimeConfig(
        # fp32 on CPU: no upcast copies, and bf16 matmul without AMX is emulated and slow.
        model_configs={BASE_MODEL_NAME: ModelConfig(BASE_MODEL_NAME, BASE_MODEL_ID, dtype=torch.float32)},
        doc_chunk_size_chars=1024,            # MS-MARCO passages are short; one chunk per document.
        doc_embedding_input_limit_chars=128,  # keeps every sequence to a few dozen tokens.
    ))
    # Rows without a selected passage yield no example, so a few extra rows guarantee 20 labeled queries.
    dataset = MsMarcoDataset.load(harness, max_examples=32, max_corpus_documents=256)
    harness.dataset_manager.build_bm25_indexes()

    # Native labels only: one chunk per positive and per hard-negative passage is inherited.
    training_data, validation_data, reporting_data = harness.dataset_manager.select_training_data(
        dataset.dataset_id, num_samples=20, oracle_labeled_only=False, val_ratio=0.5, max_reporting_size=4,
    )
    assert len(training_data) == 10 and len(validation_data) == 10 and 1 <= len(reporting_data) <= 4
    assert all(example.positive_chunk_ids and example.hard_negative_chunk_ids for example in training_data)
    assert not any(example.oracle_labeled for example in training_data)
    assert not {e.example_id for e in training_data} & {e.example_id for e in validation_data}

    harness.module_manager.register_lora(LORA_NAME, BASE_MODEL_NAME, rank=32)
    torch.manual_seed(0)                                                               # seeded AC and head init
    ac_model = StandardRetrievalACModel(harness, BASE_MODEL_NAME, d_ac_model=256, num_prefix_tokens=4, num_view_tokens=4, num_ac_layers=2)
    harness.module_manager.register_retrieval_ac(AC_NAME, BASE_MODEL_NAME, ac_model)
    retrieval_model = RetrievalModel(
        harness, BASE_MODEL_NAME, d_embedding_result=256, ac_name=AC_NAME, lora_name=LORA_NAME,
    )
    training_config = RetrievalTrainingConfig(
        epochs=4,
        batch_size=2,
        gradient_checkpointing=True,   # saved activations stay small even on CPU.
    )
    reporter = RetrievalReporter(
        REPORT_FOLDER, title="Basic retrieval training",
        description="Qwen3-0.6B + LoRA + AC on 20 MS-MARCO queries, CPU.",
    )
    trainer = RetrievalTrainer(harness)
    stats = trainer.train(training_config, retrieval_model, reporter, training_data, reporting_data)
    print(json.dumps(stats.summarize(), indent=2))
    # Held-out evaluation is a separate call: the model next to the frozen base, one batch (10 queries).
    results = trainer.eval(retrieval_model, reporter, validation_data, training_config, label="validation", progress=float(training_config.epochs))
    assert set(results) == {"model", "frozen base"} and all(set(metrics) == set(METRIC_NAMES) for _, metrics in results.values())

    assert stats.num_epochs == 4 and stats.num_steps == 20
    assert all(math.isfinite(loss) for _step, loss in stats.step_losses)
    assert len(stats.reporting_losses) >= 4  # at least one point per epoch.
    # The loss trend: 10 training queries over 4 epochs are learned (the mean step loss of the last
    # five steps is under half of the first five; AC dropout keeps it from collapsing further) and
    # the 4 held-out reporting queries improve at some point before overfitting sets in (their last
    # value is noise at this size).
    step_losses = [loss for _step, loss in stats.step_losses]
    assert sum(step_losses[-5:]) < 0.5 * sum(step_losses[:5]), step_losses
    reporting_losses = [loss for _progress, loss in stats.reporting_losses]
    assert min(reporting_losses) < reporting_losses[0], reporting_losses
    assert os.path.exists(os.path.join(REPORT_FOLDER, "report.tressoir.html"))
    assert os.path.exists(os.path.join(REPORT_FOLDER, "report_data.json"))

    # The adapter is selected per forward pass: with no LoRA name the plain base runs.
    loaded = harness.loaded_models[BASE_MODEL_NAME]
    ids = torch.tensor([[1, 2, 3, 4]], device=retrieval_model.device)
    with torch.no_grad():
        embeds = loaded.embedding_layer(ids)
        mask = torch.ones_like(ids)
        with_lora = loaded.decoder_forward(embeds, mask, lora_name=LORA_NAME)
        without_lora = loaded.decoder_forward(embeds, mask, lora_name=None)
    assert not torch.allclose(with_lora, without_lora), "The trained adapter should change the hidden states."

    # Drop the adapter (no checkpointing yet), then the base can be freed.
    harness.module_manager.free_lora(BASE_MODEL_NAME, LORA_NAME)
    assert not harness.module_manager.has_loras(BASE_MODEL_NAME)
    harness.loaded_models[BASE_MODEL_NAME].model_to_device(FREE_DEVICE)
