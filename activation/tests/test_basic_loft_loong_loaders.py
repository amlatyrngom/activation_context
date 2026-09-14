"""
LOFT and Loong loaders against the real downloads (small: two LOFT rag zips, the Loong questions and doc.zip). The
materialized files land under the synced area (LOFT/, LOONG/) like the LCA checkouts, so a second load reuses them.
"""
import os

import pytest

from activation.dataset import DatasetTaskKind, DatasetTaskMetricsKind
from activation.dataset.loaders import LoftDataset, LoongDataset
from activation.dataset.loaders import loft, loong

COPY_TREE_SETUP = "activation.agent.agent_env:CopyTreeSetup"


def _assert_copy_tree(task, name: str, datum_key: str, env_path: str) -> str:
    setup = task.env_setups[name]
    assert setup["class"] == COPY_TREE_SETUP
    assert setup["kwargs"] == {"datum_key": datum_key, "env_path": env_path}
    local = task.task_datum[datum_key]
    assert os.path.isabs(local) and os.path.exists(local)
    return local


def _no_download(url, target, timeout=0):
    raise AssertionError(f"re-download attempted: {url}")


def test_loft_loader(monkeypatch):
    loaded = LoftDataset.load(None, 4, seed=1, names=("nq", "quest"), sizes=("128k",))
    assert loaded.dataset_id == "loft___names_nq+quest__sizes_128k__splits_test+dev__n_4__seed_1"
    assert len(loaded.scorable_tasks) == 4
    assert len(loaded.documents) == 883 + 328               # every passage of both 128k corpora
    assert loaded.stats.total_num_documents == len(loaded.documents)
    for task_id, task in loaded.scorable_tasks.items():
        name, size, qid = task_id.split("/")
        assert name in ("nq", "quest") and size == "128k" and task.task_datum["qid"] == qid
        assert task.task_datum["name"] == name and task.task_datum["size"] == size and task.task_datum["split"] in ("test", "dev")
        assert task.task_datum["gold_pids"] and all(f"{name}/{size}/{pid}" in loaded.documents for pid in task.task_datum["gold_pids"])
        assert task.reference_metrics_kind == DatasetTaskMetricsKind.F1 and task.task_kind == DatasetTaskKind.FILE_SEARCH
        assert task.gold_answer
        assert "/workspace/corpus.txt" in task.agent_prompt and "Question:" in task.agent_prompt
        rule = "every answer, separated by semicolons" if name == "quest" else "a short phrase, no explanation"
        assert rule in task.agent_prompt
        corpus_path = _assert_copy_tree(task, "corpus", "corpus_path", "/workspace/corpus.txt")
        assert corpus_path.endswith(f"/LOFT/{name}/128k/corpus.txt") and os.path.getsize(corpus_path) > 100_000
        with open(corpus_path, encoding="utf-8") as handle:
            head = handle.readline()
        assert head.startswith("### ") and " | " in head
        assert all(len(str(value)) < 2000 for value in task.task_datum.values())   # no document text in the datum
    # A second load reuses the zips and corpus.txt files: no download, same files.
    corpus_paths = {task.task_datum["corpus_path"] for task in loaded.scorable_tasks.values()}
    mtimes = {path: os.path.getmtime(path) for path in corpus_paths}
    monkeypatch.setattr(loft, "download_file", _no_download)
    again = LoftDataset.load(None, 4, seed=1, names=("nq", "quest"), sizes=("128k",))
    assert list(again.scorable_tasks) == list(loaded.scorable_tasks)
    assert {path: os.path.getmtime(path) for path in corpus_paths} == mtimes


def test_loong_loader(monkeypatch):
    loaded = LoongDataset.load(None, 3, seed=1)
    assert loaded.dataset_id == "loong___n_3__seed_1"
    assert len(loaded.scorable_tasks) == 3
    assert loaded.documents and loaded.stats.total_document_chars > 0
    for task_id, task in loaded.scorable_tasks.items():
        assert task.task_id == task_id
        assert task.reference_metrics_kind == DatasetTaskMetricsKind.UNSCORED and task.task_kind == DatasetTaskKind.FILE_SEARCH
        assert task.task_datum["type"] in ("paper", "financial") and task.task_datum["n_docs"] >= 2
        assert set(task.task_datum) == {"docs_path", "level", "set", "type", "n_docs", "length"}
        assert task.gold_answer
        assert task.agent_prompt.startswith("The documents are in /workspace/docs (") and "complete answer" in task.agent_prompt
        docs_path = _assert_copy_tree(task, "docs", "docs_path", "/workspace/docs")
        assert f"/LOONG/instances/{task_id}" in docs_path
        files = sorted(os.listdir(docs_path))
        assert len(files) == task.task_datum["n_docs"] and all(os.path.getsize(os.path.join(docs_path, f)) > 0 for f in files)
        assert all(f"loong/{name}" in loaded.documents for name in files)
        assert all(name in task.agent_prompt for name in files)
    # Second load: nothing downloaded again, the instance folders are reused.
    docs_paths = {task.task_datum["docs_path"] for task in loaded.scorable_tasks.values()}
    mtimes = {path: os.path.getmtime(path) for path in docs_paths}
    monkeypatch.setattr(loong, "download_file", _no_download)
    again = LoongDataset.load(None, 3, seed=1)
    assert list(again.scorable_tasks) == list(loaded.scorable_tasks)
    assert {path: os.path.getmtime(path) for path in docs_paths} == mtimes
