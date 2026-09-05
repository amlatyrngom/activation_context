
# @AI: From now on, our tests will use RedHatAI/Qwen3.5-9B-FP8-dynamic or AxionML/Qwen3.5-9B-NVFP4 for faster loads.
# Keep the existing recommended configs.
# Further, if this run without oracle using small nq/ms-marco, that would be even better.

# @AI: This is the key test that tells me what our slice1 interface will look like. Give me a full implementation of it.
# All else should be, at first, an interface.
def test_basic_retrieval_training():
    retrieval_training_config = ...
    harness = ...
    harness.register_lora("qwen-3-0.6b-retrieval-lora", rank=128)
    retrieval_ac_model = StandardRetrievalACModel(...)
    harness.register_retrieval_ac_model("qwen-3-0.6b-retrieval-ac-model", retrieval_ac_model)
    retrieval_model = RetrievalModel(...)
    retrieval_trainer = RetrievalTrainer(...)
    dataset_manager = harness.dataset_manager
    # dataset_manager.synthesize_study_examples_qa(...) # Avoid in this test if possible.
    # dataset_manager.generate_examples_labels(...)
    training_data, val_data = dataset_manager.get_training_data(
        dataset_id="id1",
        num_samples=10000,
        synthethic_only=False, # True for things we'll end up evaluating on.
    )
    reporting_data = some_sampling(val_data, 10, seed) # Small number to get an idea of the trend while training. To know when more data is redundant aside from epoch boundaries.
    # all_training_data.extend(...) # When merging many datasets.
    reporter = RetrievalReporter(...)
    training_stats = retrieval_trainer.train(
        retrieval_model,
        training_data,
        reporting_data,
        report_folder="IB/TMP/<SOME_FOLDER>/<test_name>/",
    )
    print(training_stats.summarize())
    # Skip eval for now: I think it actually requires building the multi-vector index.
