# Retrieval training archive (slice 3, 2026-09-09)

Moved out of the source tree by slice 3 decision 6 (archive the retrieval code and the dense/faiss paths; keep bm25).
Source: the user's working tree at HEAD `1cfa058` plus the slice-3 prototype diff.

- `activation/retrieval/` — the retrieval trainer, batching, reporter, config, the byte-level `StandardRetrievalACModel`
  (`retrieval_ac.py`; its `WindowedBytePooling` is the ancestor of the AC model's `WindowedPooling`), and the
  `retrieval_dense_index.py` note.
- `activation/bench/agent_probes/retrieval_training_bench.py`, `activation/tests/test_basic_retrieval_training.py` — the bench and test.
- `activation/dataset/dataset_index.py` — the index before slice 3, with `build_dense_index`, `dense_query_many_frozen`,
  `dense_query_docs_many_frozen` and the faiss / embedding-model paths that were removed.
- `activation/module_manager_before_slice3.py` — the module manager with `register_retrieval_ac` / `get_retrieval_ac`.

Removed with it: the harness knobs `doc_embedding_model_name`, `doc_embedding_batch_size`, `doc_embedding_input_limit_chars`,
`DatasetManager.build_dense_indexes` / `select_training_data`, the dense `DatasetStats` fields, and the `faiss-cpu` dependency.
