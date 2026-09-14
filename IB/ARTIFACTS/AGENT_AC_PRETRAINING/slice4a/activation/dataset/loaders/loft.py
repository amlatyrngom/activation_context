"""
LOFT text RAG tasks (`google-deepmind/loft`, the public `rag/<name>.zip` bundles): NQ, HotpotQA, MuSiQue, QAMPARI and
QuEST queries over a corpus of a few hundred to a few thousand Wikipedia passages (the 32k / 128k / 1m context sizes).
The zips are fetched once into LOFT/<name>/<size>/ under the synced area; per (name, size) the loader writes one
plain-text `corpus.txt` (a `### <pid> | <title>` line, the passage, a blank line, per passage) that the task's
CopyTreeSetup copies to /workspace/corpus.txt in the sandbox, so the task datum only names the path. Every passage
is also a corpus document so semantic_search can index it. Token F1 against the answers (single-answer sets: the
first answer is the gold and the rest aliases; QAMPARI and QuEST are multi-answer: the gold is the "; "-joined list).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import time
import typing as t
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from activation.common.data_syncing import resolve_path

from ..dataset import DatasetDocument, DatasetTask, DatasetTaskMetricsKind, LoadedDataset
from ..dataset_utils import initialize_dataset_stats, make_dataset_id
from ..dataset import ANSWER_RULES, DatasetTaskKind, bare_prompt

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime

LOFT_URL = "https://storage.googleapis.com/loft-bench/rag/{name}.zip"
LOFT_CACHE = "LOFT"
CORPUS_PATH = "/workspace/corpus.txt"
COPY_TREE_SETUP = "activation.agent.agent_env:CopyTreeSetup"
NAMES = ("nq", "hotpotqa", "musique", "qampari", "quest")
SIZES = ("32k", "128k", "1m")
MULTI_ANSWER_NAMES = {"qampari", "quest"}
DOWNLOAD_TIMEOUT = 600


def download_file(url: str, target: Path, timeout: int = DOWNLOAD_TIMEOUT) -> None:
    """`url` streamed to a temporary name next to `target`, then renamed: a partial download never sits at the target."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f".{target.name}.partial"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "activation-suite"}), timeout=timeout) as response:
            with open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle, length=1 << 20)
        partial.replace(target)
    except (urllib.error.URLError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"downloading {url} failed: {error}") from error


def ensure_loft(name: str) -> Path:
    """LOFT/<name>/ with its <size>/ folders, fetched and unzipped once (staged, then renamed into place)."""
    root = resolve_path(LOFT_CACHE, create=True)
    target = root / name
    if target.is_dir():
        return target
    archive = root / f".{name}.zip"
    staging = root / f".{name}.unzip"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        download_file(LOFT_URL.format(name=name), archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(staging)
        if not (staging / name).is_dir():
            raise RuntimeError(f"unexpected archive layout {sorted(entry.name for entry in staging.iterdir())}")
        (staging / name).rename(target)
    except (zipfile.BadZipFile, OSError, RuntimeError) as error:
        raise RuntimeError(f"fetching LOFT {name} failed: {error}") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        archive.unlink(missing_ok=True)
    return target


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ensure_corpus_text(folder: Path, passages: list[dict]) -> Path:
    """`corpus.txt` next to corpus.jsonl: one `### <pid> | <title>` block per passage, written to a temporary name then renamed."""
    target = folder / "corpus.txt"
    if target.is_file():
        return target
    partial = folder / ".corpus.txt.partial"
    with open(partial, "w", encoding="utf-8") as handle:
        for passage in passages:
            handle.write(f"### {passage['pid']} | {str(passage.get('title_text') or '').strip()}\n{str(passage.get('passage_text') or '').strip()}\n\n")
    partial.replace(target)
    return target


class LoftDataset:
    @classmethod
    def load(cls, harness: "HarnessRuntime", max_examples: int | None, seed: int = 0, names: t.Sequence[str] = ("nq", "hotpotqa", "musique", "qampari", "quest"),
             sizes: t.Sequence[str] = ("128k", "1m"), splits: t.Sequence[str] = ("test", "dev")) -> LoadedDataset:
        start = time.time()
        rng = random.Random(seed)
        unknown = [name for name in names if name not in NAMES] + [size for size in sizes if size not in SIZES]
        if unknown:
            raise ValueError(f"unknown LOFT names/sizes {unknown}; known names {NAMES}, sizes {SIZES}")
        dataset_id = make_dataset_id("loft", names="+".join(names), sizes="+".join(sizes), splits="+".join(splits), n=max_examples, seed=seed)
        documents: dict[str, DatasetDocument] = {}
        candidates: list[DatasetTask] = []
        for name in names:
            root = ensure_loft(name)
            for size in sizes:
                folder = root / size
                passages = read_jsonl(folder / "corpus.jsonl")
                corpus_path = ensure_corpus_text(folder, passages)
                n_passages, chars = len(passages), os.path.getsize(corpus_path)
                for passage in passages:
                    doc_id = f"{name}/{size}/{passage['pid']}"
                    documents.setdefault(doc_id, DatasetDocument(doc_id=doc_id, dataset_id=dataset_id,
                                                                 text=f"{str(passage.get('title_text') or '').strip()}\n{str(passage.get('passage_text') or '').strip()}"))
                multi = name in MULTI_ANSWER_NAMES
                rule = ANSWER_RULES["list"] if multi else ANSWER_RULES["f1"]
                for split in splits:
                    queries_path = folder / f"{split}_queries.jsonl"
                    if not queries_path.is_file():
                        continue                                                        # 32k bundles have no test split
                    for row in read_jsonl(queries_path):
                        answers = [str(answer) for answer in (row.get("answers") or [])]
                        if not answers:
                            continue
                        if multi:
                            gold, aliases = "; ".join(answers), []
                        else:
                            gold, aliases = answers[0], answers[1:]
                        task = (f"The passage collection is at {CORPUS_PATH} ({n_passages:,} passages, {chars:,} characters; each passage "
                                f"starts with a '### <id> | <title>' line). Find the answer in it.\n\nQuestion: {row['query_text']}")
                        candidates.append(DatasetTask(
                            task_id=f"{name}/{size}/{row['qid']}", dataset_id=dataset_id,
                            task_datum={"name": name, "size": size, "split": split, "qid": str(row["qid"]), "corpus_path": str(corpus_path),
                                        "gold_pids": [str(pid) for pid, _ in ((row.get("metadata") or {}).get("qrels") or [])]},
                            reference_metrics_kind=DatasetTaskMetricsKind.F1, gold_answer=gold, gold_answer_aliases=aliases,
                            agent_prompt=bare_prompt(task, rule), task_kind=DatasetTaskKind.FILE_SEARCH,
                            env_setups={"corpus": {"class": COPY_TREE_SETUP, "kwargs": {"datum_key": "corpus_path", "env_path": CORPUS_PATH}}},
                        ))
        rng.shuffle(candidates)
        if max_examples is not None:
            candidates = candidates[:max_examples]
        tasks = {task.task_id: task for task in candidates}
        loaded = LoadedDataset(dataset_id=dataset_id, documents=documents, scorable_tasks=tasks)
        loaded.stats = initialize_dataset_stats(loaded, load_time=time.time() - start)
        return loaded
