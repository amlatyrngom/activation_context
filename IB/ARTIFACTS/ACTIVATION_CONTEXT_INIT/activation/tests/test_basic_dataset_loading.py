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

# Last element spans several chunks at the test chunk size.
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
    "Longer passage spanning several chunks.",
]

EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

def _embedding_model_config(retrieval: str) -> dict:
    """The dense variant needs the embedding model; bm25 runs with no model at all."""
    if retrieval == "dense":
        return dict(
            model_configs={EMBEDDING_MODEL_ID: ModelConfig(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_ID)},
            doc_embedding_model_name=EMBEDDING_MODEL_ID,
        )
    return dict()


def _build_indexes(harness: HarnessRuntime, retrieval: str):
    if retrieval == "dense":
        harness.dataset_manager.build_dense_indexes()
    else:
        harness.dataset_manager.build_bm25_indexes()


def _query_many(doc_index, retrieval: str, *args, **kwargs):
    query_fn = doc_index.dense_query_many_frozen if retrieval == "dense" else doc_index.bm25_query_many_frozen
    return query_fn(*args, **kwargs)


def _query_docs_many(doc_index, retrieval: str, *args, **kwargs):
    query_fn = doc_index.dense_query_docs_many_frozen if retrieval == "dense" else doc_index.bm25_query_docs_many_frozen
    return query_fn(*args, **kwargs)


def test_basic_dataset_loading_bm25():
    _run_basic_dataset_loading("bm25")


def test_basic_dataset_loading_dense():
    _run_basic_dataset_loading("dense")


def _run_basic_dataset_loading(retrieval: str):
    harness_config = HarnessRuntimeConfig(
        doc_chunk_size_chars=8,
        doc_chunk_overlap_chars=2,
        **_embedding_model_config(retrieval),
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
            )
            for i, text in enumerate(TEST_DOCUMENTS)}
    )
    harness.dataset_manager.register_dataset(dataset)
    assert dataset_id in harness.dataset_manager.loaded_datasets
    _build_indexes(harness, retrieval)
    doc_index = harness.dataset_manager.dataset_indexes[dataset_id]
    # Some chunking assertions: every document chunks at size 8 with overlap 2 (step 6).
    num_docs = len(TEST_DOCUMENTS)
    long_text = TEST_DOCUMENTS[num_docs-1]
    assert doc_index.chunks[f"{num_docs-1}:0"].chunk_text == long_text[:8]
    assert doc_index.chunks[f"{num_docs-1}:1"].chunk_text == long_text[6:14]
    assert len(dataset.documents[str(num_docs-1)].chunks) == -(-(len(long_text) - 2) // 6)
    assert doc_index.chunks[f"{num_docs-2}:0"].chunk_text == TEST_DOCUMENTS[num_docs-2]
    assert all(len(chunk.chunk_text) <= 8 for chunk in doc_index.chunks.values())
    if retrieval == "dense":
        assert doc_index.dense_faiss_index.ntotal == len(doc_index.chunks)
    else:
        assert doc_index.bm25_index.num_documents == len(doc_index.chunks)
    # Some retrieval asserions.
    # Exact-mathch queries.
    queries = [doc_index.chunks["0:0"].chunk_text, doc_index.chunks["1:0"].chunk_text]
    top_k = 5
    all_results = _query_many(doc_index, retrieval, queries, top_k)
    assert all_results[0][0].doc_id == "0"
    assert all_results[1][0].doc_id == "1"
    # Exclusions are respected.
    excluded_results = _query_many(doc_index, retrieval, queries, top_k, excluded_doc_ids=[["0"], None])
    assert all(chunk.doc_id != "0" for chunk in excluded_results[0])
    # Document-level retrieval dedups chunks into documents.
    all_doc_results = _query_docs_many(doc_index, retrieval, queries, top_k)
    assert all_doc_results[0][0].doc_id == "0"
    assert len({doc.doc_id for doc in all_doc_results[0]}) == len(all_doc_results[0])
    if retrieval == "bm25":
        # A query sharing no term with the corpus returns nothing rather than arbitrary chunks.
        assert doc_index.bm25_query_many_frozen(["zzzz unmatched"], top_k) == [[]]
        # In-doc scoring: the second chunk of the multi-chunk document ranks first for its own text.
        long_doc_id = str(num_docs-1)
        second_chunk = doc_index.chunks[f"{long_doc_id}:1"]
        ranked = doc_index.bm25_rank_doc_chunks(second_chunk.chunk_text, [long_doc_id])[long_doc_id]
        assert ranked[0][0].chunk_id == second_chunk.chunk_id and ranked[0][1] > 0
        assert len(ranked) == len(dataset.documents[long_doc_id].chunks)
        unmatched = doc_index.bm25_rank_doc_chunks("zzzz unmatched", [long_doc_id])[long_doc_id]
        assert all(score == 0.0 for _, score in unmatched)


def _elide(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def test_public_dataset_loading_bm25():
    _run_public_dataset_loading("bm25")


def test_public_dataset_loading_dense():
    _run_public_dataset_loading("dense")


def _run_public_dataset_loading(retrieval: str):
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
    MAX_CORPUS_DOCUMENTS = 2000 if TARGET_DEVICE == "cuda" else 100
    EMBEDDING_INPUT_LIMIT_CHARS = None if TARGET_DEVICE == "cuda" else 64
    TEST_BATCH_SIZE = 32
    harness_config = HarnessRuntimeConfig(
        doc_embedding_batch_size=TEST_BATCH_SIZE,
        doc_embedding_input_limit_chars=EMBEDDING_INPUT_LIMIT_CHARS,
        **_embedding_model_config(retrieval),
    )
    harness = HarnessRuntime(harness_config)
    loaders = {
        "BRIGHT": lambda: [
            BrightDataset.load(harness, MAX_EXAMPLES_PER_DATASET, domain=domain,
                               max_corpus_documents=MAX_CORPUS_DOCUMENTS)
            for domain in BRIGHT_DOMAINS
        ],
        "NQ": lambda: [NqDataset.load(harness, MAX_EXAMPLES_PER_DATASET,
                                      max_corpus_documents=MAX_CORPUS_DOCUMENTS)],
        "MS_MARCO": lambda: [MsMarcoDataset.load(harness, MAX_EXAMPLES_PER_DATASET,
                                                 max_corpus_documents=MAX_CORPUS_DOCUMENTS)],
        "SCIFACT": lambda: [SciFactDataset.load(harness, MAX_EXAMPLES_PER_DATASET,
                                                max_corpus_documents=MAX_CORPUS_DOCUMENTS)],
        "SCIQ": lambda: [SciQDataset.load(harness, MAX_EXAMPLES_PER_DATASET,
                                          max_corpus_documents=MAX_CORPUS_DOCUMENTS)],
    }
    loaded_datasets: list[LoadedDataset] = []
    for dataset_name in TESTED_DATASETS:
        loaded_datasets.extend(loaders[dataset_name]())
    _build_indexes(harness, retrieval)
    for dataset in loaded_datasets:
        doc_index = harness.dataset_manager.dataset_indexes[dataset.dataset_id]
        # Size assertions.
        assert 0 < len(dataset.labeled_retrieval_examples) <= MAX_EXAMPLES_PER_DATASET
        # Every labeled example points at documents that actually loaded.
        for example in dataset.labeled_retrieval_examples.values():
            assert example.positive_doc_ids
            assert all(doc_id in dataset.documents for doc_id in example.positive_doc_ids)
        # Querying with an exact stored chunk returns that chunk text first (same embedding /
        # strongest lexical match).
        probe_chunk_id, probe_chunk = next(iter(doc_index.chunks.items()))
        probe_text = probe_chunk.chunk_text
        probe_results = _query_many(doc_index, retrieval, [probe_text], top_k=1)
        top_chunk = probe_results[0][0]
        assert top_chunk.chunk_text == probe_text, f"exact-chunk probe failed for {probe_chunk_id}"
        # Exclusions are respected: excluding the probe's own document removes it.
        excluded_results = _query_many(
            doc_index, retrieval,
            [probe_text],
            top_k=5,
            excluded_doc_ids=[[probe_chunk.doc_id]],
        )
        assert all(chunk.doc_id != probe_chunk.doc_id for chunk in excluded_results[0])
        # Manual-inspection printout: 2 labeled queries with their top 5 chunks.
        inspected = list(dataset.labeled_retrieval_examples.values())[:2]
        all_results = _query_many(
            doc_index, retrieval,
            [example.query for example in inspected],
            top_k=5,
            excluded_doc_ids=[example.excluded_doc_ids for example in inspected],
        )
        print(f"\n=== {dataset.dataset_id} [{retrieval}] ===")
        for example, results in zip(inspected, all_results):
            print(f"\nquery [{example.example_id}]: {_elide(example.query, 250)}")
            for rank, chunk in enumerate(results):
                gold = "GOLD " if chunk.doc_id in example.positive_doc_ids else "     "
                print(f"  {rank + 1}. {gold}[{chunk.chunk_id}] {_elide(chunk.chunk_text, 250)}")
    print("\n=== dataset stats ===")
    for dataset in loaded_datasets:
        print(f"\n{dataset.dataset_id}")
        print(json.dumps(dataset.stats.summarize(), indent=2))