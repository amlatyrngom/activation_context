import json

from activation.dataset import DatasetDocument, LoadedDataset
from activation.dataset.dataset import DataModality
from activation.dataset.loaders import (
    BrightDataset,
    MsMarcoDataset,
    NqDataset,
    SciFactDataset,
    SciQDataset,
)
from activation.harness import HarnessRuntime, HarnessRuntimeConfig, TARGET_DEVICE

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

# A small trajectory in our dialect: chunked at message boundaries.
TEST_TRAJECTORY = [
    {"role": "user", "content": "Count the files in /data and report."},
    {"role": "assistant", "content": "Listing first.", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "shell", "arguments": {"script": "ls /data | wc -l"}}}]},
    {"role": "tool", "content": "42"},
    {"role": "assistant", "content": "There are 42 files in /data."},
]


def test_basic_dataset_loading_bm25():
    harness_config = HarnessRuntimeConfig(
        doc_chunk_size_chars=8,
        doc_chunk_overlap_chars=2,
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
    harness.dataset_manager.build_bm25_indexes()
    doc_index = harness.dataset_manager.dataset_indexes[dataset_id]
    # Some chunking assertions: every document chunks at size 8 with overlap 2 (step 6).
    num_docs = len(TEST_DOCUMENTS)
    long_text = TEST_DOCUMENTS[num_docs-1]
    assert doc_index.chunks[f"{num_docs-1}:0"].chunk_text == long_text[:8]
    assert doc_index.chunks[f"{num_docs-1}:1"].chunk_text == long_text[6:14]
    assert len(dataset.documents[str(num_docs-1)].chunks) == -(-(len(long_text) - 2) // 6)
    assert doc_index.chunks[f"{num_docs-2}:0"].chunk_text == TEST_DOCUMENTS[num_docs-2]
    assert all(len(chunk.chunk_text) <= 8 for chunk in doc_index.chunks.values())
    assert doc_index.bm25_index.num_documents == len(doc_index.chunks)
    assert dataset.stats.num_chunks == len(doc_index.chunks)
    # Some retrieval asserions.
    # Exact-mathch queries.
    queries = [doc_index.chunks["0:0"].chunk_text, doc_index.chunks["1:0"].chunk_text]
    top_k = 5
    all_results = doc_index.bm25_query_many_frozen(queries, top_k)
    assert all_results[0][0].doc_id == "0"
    assert all_results[1][0].doc_id == "1"
    # Exclusions are respected.
    excluded_results = doc_index.bm25_query_many_frozen(queries, top_k, excluded_doc_ids=[["0"], None])
    assert all(chunk.doc_id != "0" for chunk in excluded_results[0])
    # Document-level retrieval dedups chunks into documents.
    all_doc_results = doc_index.bm25_query_docs_many_frozen(queries, top_k)
    assert all_doc_results[0][0].doc_id == "0"
    assert len({doc.doc_id for doc in all_doc_results[0]}) == len(all_doc_results[0])
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


def test_trajectory_chunking_bm25():
    """Trajectory documents chunk at message boundaries; an over-long message splits by characters; chunks keep their messages."""
    harness = HarnessRuntime(HarnessRuntimeConfig(doc_chunk_size_chars=120, doc_chunk_overlap_chars=8))
    dataset_id = "test_trajectory"
    long_message = {"role": "tool", "content": "x" * 300}
    documents = {
        "t0": DatasetDocument(doc_id="t0", dataset_id=dataset_id, trajectory=TEST_TRAJECTORY, modality=DataModality.TRAJECTORY),
        "t1": DatasetDocument(doc_id="t1", dataset_id=dataset_id, trajectory=TEST_TRAJECTORY[:1] + [long_message], modality=DataModality.TRAJECTORY),
    }
    dataset = LoadedDataset(dataset_id=dataset_id, documents=documents)
    harness.dataset_manager.register_dataset(dataset)
    harness.dataset_manager.build_bm25_indexes()
    doc_index = harness.dataset_manager.dataset_indexes[dataset_id]
    chunks_t0 = sorted(documents["t0"].chunks.values(), key=lambda chunk: chunk.chunk_start)
    assert len(chunks_t0) > 1
    assert [message for chunk in chunks_t0 for message in chunk.chunk_messages] == TEST_TRAJECTORY   # every message in exactly one chunk, in order
    assert all(len(chunk.chunk_text) <= 120 for chunk in chunks_t0)
    assert "[call shell(" in "".join(chunk.chunk_text for chunk in chunks_t0)                      # calls are rendered for bm25
    chunks_t1 = sorted(documents["t1"].chunks.values(), key=lambda chunk: chunk.chunk_start)
    assert any(len(chunk.chunk_messages) == 1 and chunk.chunk_messages[0]["role"] == "tool" and len(chunk.chunk_text) <= 120 for chunk in chunks_t1)
    assert dataset.stats.total_document_chars > 0
    results = doc_index.bm25_query_many_frozen(["files /data report"], top_k=3)
    assert results[0] and results[0][0].doc_id in documents


def _elide(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def test_public_dataset_loading_bm25():
    # Test public dataset loading.
    BRIGHT_DOMAINS = ["biology", "economics"]
    TESTED_DATASETS = [
        "BRIGHT",
        "NQ",
        "MS_MARCO",
        "SCIFACT",
        "SCIQ",
    ]
    # Limit examples for testing speed.
    MAX_EXAMPLES_PER_DATASET = 100 if TARGET_DEVICE == "cuda" else 20
    MAX_CORPUS_DOCUMENTS = 2000 if TARGET_DEVICE == "cuda" else 100
    harness = HarnessRuntime(HarnessRuntimeConfig())
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
    harness.dataset_manager.build_bm25_indexes()
    for dataset in loaded_datasets:
        doc_index = harness.dataset_manager.dataset_indexes[dataset.dataset_id]
        # Size assertions.
        assert 0 < len(dataset.labeled_qa_examples) <= MAX_EXAMPLES_PER_DATASET
        # Every labeled example points at documents that actually loaded.
        for example in dataset.labeled_qa_examples.values():
            assert example.positive_doc_ids
            assert all(doc_id in dataset.documents for doc_id in example.positive_doc_ids)
        # Querying with an exact stored chunk returns that chunk text first (strongest lexical match).
        probe_chunk_id, probe_chunk = next(iter(doc_index.chunks.items()))
        probe_text = probe_chunk.chunk_text
        probe_results = doc_index.bm25_query_many_frozen([probe_text], top_k=1)
        top_chunk = probe_results[0][0]
        assert top_chunk.chunk_text == probe_text, f"exact-chunk probe failed for {probe_chunk_id}"
        # Exclusions are respected: excluding the probe's own document removes it.
        excluded_results = doc_index.bm25_query_many_frozen(
            [probe_text],
            top_k=5,
            excluded_doc_ids=[[probe_chunk.doc_id]],
        )
        assert all(chunk.doc_id != probe_chunk.doc_id for chunk in excluded_results[0])
        # Manual-inspection printout: 2 labeled queries with their top 5 chunks.
        inspected = list(dataset.labeled_qa_examples.values())[:2]
        all_results = doc_index.bm25_query_many_frozen(
            [example.query for example in inspected],
            top_k=5,
            excluded_doc_ids=[example.excluded_doc_ids for example in inspected],
        )
        print(f"\n=== {dataset.dataset_id} [bm25] ===")
        for example, results in zip(inspected, all_results):
            print(f"\nquery [{example.example_id}]: {_elide(example.query, 250)}")
            for rank, chunk in enumerate(results):
                gold = "GOLD " if chunk.doc_id in example.positive_doc_ids else "     "
                print(f"  {rank + 1}. {gold}[{chunk.chunk_id}] {_elide(chunk.chunk_text, 250)}")
    print("\n=== dataset stats ===")
    for dataset in loaded_datasets:
        print(f"\n{dataset.dataset_id}")
        print(json.dumps(dataset.stats.summarize(), indent=2))


def test_trajectory_dataset_loading():
    """A few streamed rows of each trajectory source land in our dialect, chunk at message boundaries and index."""
    from activation.dataset.dataset import EXTERNAL
    from activation.dataset.loaders import NemotronMathDataset, OpenSweTracesDataset, S1DeepResearchDataset, aime_problem_statements

    harness = HarnessRuntime(HarnessRuntimeConfig(doc_chunk_size_chars=4000, doc_chunk_overlap_chars=0))
    aime = aime_problem_statements()
    assert len(aime) == 60
    datasets = [
        OpenSweTracesDataset.load(harness, max_examples=2),
        NemotronMathDataset.load(harness, max_examples=2, subset="tir", excluded_problems=aime, min_tool_calls=1),
        S1DeepResearchDataset.load(harness, max_examples=2),
    ]
    harness.dataset_manager.build_bm25_indexes()
    expected_tools = [{"shell"}, {"python"}, {"search", "visit"}]
    for dataset, tools in zip(datasets, expected_tools):
        assert len(dataset.documents) == 2, dataset.dataset_id
        doc_index = harness.dataset_manager.dataset_indexes[dataset.dataset_id]
        for document in dataset.documents.values():
            assert document.modality == DataModality.TRAJECTORY and document.origin == EXTERNAL
            assert document.trajectory[0]["role"] == "user" and document.trajectory_kwargs["system_prompt"]
            assert tools <= {tool["function"]["name"] for tool in document.trajectory_kwargs["tools"]}
            calls = [call for message in document.trajectory for call in message.get("tool_calls") or []]
            assert calls and all(isinstance(call["function"]["arguments"], dict) for call in calls)
            assert sum(message["role"] == "tool" for message in document.trajectory) >= len(calls) - 1  # the last call may end the trace
            chunks = sorted(document.chunks.values(), key=lambda chunk: chunk.chunk_start)
            flat = [message for chunk in chunks for message in chunk.chunk_messages]
            assert len(chunks) > 1 and len(flat) >= len(document.trajectory)
            # Every character of content lands in exactly one chunk (overlap 0), long messages as several slices of one role.
            assert "".join(message["content"] for message in flat) == "".join(message["content"] for message in document.trajectory)
            assert sum(len(message.get("tool_calls") or []) for message in flat) == len(calls)
        assert doc_index.bm25_index.num_documents == dataset.stats.num_chunks > 0
        print(f"\n{dataset.dataset_id}: {json.dumps(dataset.stats.summarize(), indent=2)}")
