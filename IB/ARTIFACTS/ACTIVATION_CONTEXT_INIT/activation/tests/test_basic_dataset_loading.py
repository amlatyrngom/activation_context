import json

from activation.dataset import DatasetDocument, LoadedDataset
from activation.dataset.loaders import (
    BrightDataset,
    MsMarcoDataset,
    NqDataset,
    SciFactDataset,
    SciQDataset,
)
from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig, TARGET_DEVICE

# Last element is longer than the max atomic size.
# Second-to-last is shorter than the chunk size.
TEST_DOCUMENTS = [
    "Gravity pulls masses together.",
    "Photosynthesis makes sugar.",
    "Sound travels as a wave.",
    "Electrons have negative charge.",
    "DNA stores cell instructions.",
    "Volcanoes release molten rock.",
    "Magnets attract iron.",
    "Whales are mammals.",
    "Light bends through glass.",
    "Oxygen.", # Short
    "Atomic passage larger than sixteen characters.",
]

EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

def test_basic_dataset_loading():
    harness_config = HarnessRuntimeConfig(
        model_configs={
            EMBEDDING_MODEL_ID: ModelConfig(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_ID),
        },
        doc_chunk_size_chars=8,
        doc_chunk_overlap_chars=2,
        doc_chunk_max_atomic_size=16,
        doc_embedding_model_name=EMBEDDING_MODEL_ID,
    )
    harness = HarnessRuntime(harness_config)
    dataset_id: str = "test_basic"
    dataset = LoadedDataset(
        dataset_id=dataset_id,
        documents={
            str(i): DatasetDocument(
                doc_id=str(i),
                dataset_id=dataset_id,
                text=text,
                atomic = (i == len(TEST_DOCUMENTS) - 1), # For testing.
            )
            for i, text in enumerate(TEST_DOCUMENTS)}
    )
    harness.dataset_manager.register_dataset(dataset)
    assert dataset_id in harness.dataset_manager.loaded_datasets
    harness.dataset_manager.build_document_indexes()
    doc_index = harness.dataset_manager.document_indexes[dataset_id]
    # Some chunking assertions.
    num_docs = len(TEST_DOCUMENTS)
    assert doc_index.chunks[f"{num_docs-1}:0"][0] == TEST_DOCUMENTS[num_docs-1][:16]
    assert doc_index.chunks[f"{num_docs-1}:1"][0] == TEST_DOCUMENTS[num_docs-1][14:30]
    assert doc_index.chunks[f"{num_docs-2}:0"][0] == TEST_DOCUMENTS[num_docs-2]
    assert all(
        len(text) <= 8
        for text, document in doc_index.chunks.values()
        if not document.atomic
    )
    assert doc_index.faiss_index.ntotal == len(doc_index.chunks)
    # Some retrieval asserions.
    # Exact-mathch queries.
    queries = [doc_index.chunks["0:0"][0], doc_index.chunks["1:0"][0]]
    top_k = 5
    all_results = doc_index.query_many_frozen(queries, top_k)
    _, actual_doc0 = all_results[0][0]
    _, actual_doc1 = all_results[1][0]
    assert actual_doc0.doc_id == "0"
    assert actual_doc1.doc_id == "1"


def _elide(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def test_public_dataset_loading():
    # Test public dataset loading.
    BRIGHT_DOMAINS = ["biology", "economics"]
    TESTED_DATASETS = [
        "BRIGHT",
        "NQ",
        "MS_MARCO",
        "SCIFACT",
        "SCIQ",
    ]
    # Limit examples and embedding inputs for testing speed.
    MAX_EXAMPLES_PER_DATASET = 100 if TARGET_DEVICE == "cuda" else 20
    EMBEDDING_INPUT_LIMIT_CHARS = None if TARGET_DEVICE == "cuda" else 64
    TEST_BATCH_SIZE = 32
    harness_config = HarnessRuntimeConfig(
        model_configs={
            EMBEDDING_MODEL_ID: ModelConfig(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_ID),
        },
        doc_embedding_model_name=EMBEDDING_MODEL_ID,
        doc_embedding_batch_size=TEST_BATCH_SIZE,
        doc_embedding_input_limit_chars=EMBEDDING_INPUT_LIMIT_CHARS,
    )
    harness = HarnessRuntime(harness_config)
    loaders = {
        "BRIGHT": lambda: [
            BrightDataset.load(harness, MAX_EXAMPLES_PER_DATASET, domain=domain)
            for domain in BRIGHT_DOMAINS
        ],
        "NQ": lambda: [NqDataset.load(harness, MAX_EXAMPLES_PER_DATASET)],
        "MS_MARCO": lambda: [MsMarcoDataset.load(harness, MAX_EXAMPLES_PER_DATASET)],
        "SCIFACT": lambda: [SciFactDataset.load(harness, MAX_EXAMPLES_PER_DATASET)],
        "SCIQ": lambda: [SciQDataset.load(harness, MAX_EXAMPLES_PER_DATASET)],
    }
    loaded_datasets: list[LoadedDataset] = []
    for dataset_name in TESTED_DATASETS:
        loaded_datasets.extend(loaders[dataset_name]())
    harness.dataset_manager.build_document_indexes()
    for dataset in loaded_datasets:
        doc_index = harness.dataset_manager.document_indexes[dataset.dataset_id]
        # Size assertions.
        assert 0 < len(dataset.labeled_retrieval_examples) <= MAX_EXAMPLES_PER_DATASET
        # Every labeled example points at documents that actually loaded.
        for example in dataset.labeled_retrieval_examples.values():
            assert example.positive_doc_ids
            assert all(doc_id in dataset.documents for doc_id in example.positive_doc_ids)
        # Same-doc-embedding: querying with an exact stored chunk returns that chunk text first.
        probe_chunk_id, (probe_text, probe_doc) = next(iter(doc_index.chunks.items()))
        probe_results = doc_index.query_many_frozen([probe_text], top_k=1)
        top_chunk_id, _top_doc = probe_results[0][0]
        assert doc_index.chunks[top_chunk_id][0] == probe_text, f"exact-chunk probe failed for {probe_chunk_id}"
        # Exclusions are respected: excluding the probe's own document removes it.
        excluded_results = doc_index.query_many_frozen(
            [probe_text],
            top_k=5,
            excluded_doc_ids=[[probe_doc.doc_id]],
        )
        assert all(doc.doc_id != probe_doc.doc_id for _, doc in excluded_results[0])
        # Manual-inspection printout: 2 labeled queries with their top 5 chunks.
        inspected = list(dataset.labeled_retrieval_examples.values())[:2]
        all_results = doc_index.query_many_frozen(
            [example.query for example in inspected],
            top_k=5,
            excluded_doc_ids=[example.excluded_doc_ids for example in inspected],
        )
        print(f"\n=== {dataset.dataset_id} ===")
        for example, results in zip(inspected, all_results):
            print(f"\nquery [{example.example_id}]: {_elide(example.query, 250)}")
            for rank, (chunk_id, doc) in enumerate(results):
                gold = "GOLD " if doc.doc_id in example.positive_doc_ids else "     "
                chunk_text, _ = doc_index.chunks[chunk_id]
                print(f"  {rank + 1}. {gold}[{chunk_id}] {_elide(chunk_text, 250)}")
    print("\n=== dataset stats ===")
    for dataset in loaded_datasets:
        print(f"\n{dataset.dataset_id}")
        print(json.dumps(dataset.stats.summarize(), indent=2))