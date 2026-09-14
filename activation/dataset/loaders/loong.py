"""
Loong (`MozerWang/Loong`): long multi-document QA, the English part (arXiv papers: citation relations, and 10-K
financial statements: figures and comparisons across companies). Questions come from the repository's loong.jsonl,
documents from the doc.zip bundle (doc/paper/<arxiv id>.md, doc/financial/<year>-<company>-<suffix>.txt), both fetched
once into LOONG/ under the synced area. Per instance the loader materializes LOONG/instances/<id>/ with hard links (or
copies) of its documents under their original file names and the task's CopyTreeSetup copies that folder to
/workspace/docs in the sandbox; the task datum only names the path. Financial instances name companies, matched to the
financial file whose name contains them (all the exact `-<company>-` files when several do); an instance with an
unmatched company is skipped. Unscored: the answers are structured (json, lists, free text) and kept for reference.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import time
import typing as t
import zipfile
from pathlib import Path

from activation.common.data_syncing import resolve_path

from ..dataset import DatasetDocument, DatasetTask, DatasetTaskMetricsKind, LoadedDataset
from ..dataset_utils import initialize_dataset_stats, make_dataset_id
from ..dataset import ANSWER_RULES, DatasetTaskKind, bare_prompt
from .loft import download_file

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime

QUESTIONS_URL = "https://raw.githubusercontent.com/MozerWang/Loong/main/data/loong.jsonl"
DOCS_URL = "http://alibaba-research.oss-cn-beijing.aliyuncs.com/loong/doc.zip"
LOONG_CACHE = "LOONG"
DOCS_PATH = "/workspace/docs"
COPY_TREE_SETUP = "activation.agent.agent_env:CopyTreeSetup"
DOC_FOLDERS = ("paper", "financial")   # the English document kinds; doc/legal is Chinese and stays in the zip
MAX_DOC_CHARS = 2_000_000


def ensure_questions() -> Path:
    target = resolve_path(LOONG_CACHE, create=True) / "loong.jsonl"
    if not target.is_file():
        download_file(QUESTIONS_URL, target)
    return target


def ensure_docs() -> Path:
    """LOONG/doc/{paper,financial}/ unzipped once from doc.zip (staged, then renamed into place; __MACOSX and dotfiles skipped)."""
    root = resolve_path(LOONG_CACHE, create=True)
    target = root / "doc"
    if target.is_dir():
        return target
    archive = root / ".doc.zip"
    staging = root / ".doc.unzip"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        download_file(DOCS_URL, archive)
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                parts = Path(info.filename).parts
                if info.is_dir() or len(parts) != 3 or parts[0] != "doc" or parts[1] not in DOC_FOLDERS or parts[2].startswith("."):
                    continue
                bundle.extract(info, staging)
        if not (staging / "doc").is_dir():
            raise RuntimeError(f"unexpected archive layout {sorted(entry.name for entry in staging.iterdir())}")
        (staging / "doc").rename(target)
    except (zipfile.BadZipFile, OSError, RuntimeError) as error:
        raise RuntimeError(f"fetching Loong documents failed: {error}") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        archive.unlink(missing_ok=True)
    return target


def match_financial(company: str, files: list[str]) -> list[str]:
    """The financial file names holding `company` (case-insensitive): every exact `-<company>-` match, else every containing one."""
    needle = company.lower()
    exact = [name for name in files if f"-{needle}-" in name.lower()]
    return exact or [name for name in files if needle in name.lower()]


def resolve_documents(row: dict, docs_root: Path, financial_files: list[str]) -> list[Path] | None:
    """The instance's document files, or None when one of its names cannot be resolved."""
    paths: list[Path] = []
    if row["type"] == "financial":
        for company in row["doc"]:
            matches = match_financial(str(company), financial_files)
            if not matches:
                return None
            paths.extend(docs_root / "financial" / name for name in matches)
    else:
        for name in row["doc"]:
            path = docs_root / str(row["type"]) / str(name)
            if not path.is_file():
                return None
            paths.append(path)
    return list(dict.fromkeys(paths))


def ensure_instance(instance_id: str, paths: list[Path]) -> Path:
    """LOONG/instances/<id>/ with the documents under their file names (hard links, copies where linking fails), staged then renamed."""
    instances = resolve_path(f"{LOONG_CACHE}/instances", create=True)
    target = instances / instance_id
    if target.is_dir():
        return target
    staging = instances / f".{instance_id}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    for path in paths:
        try:
            os.link(path, staging / path.name)
        except OSError:
            shutil.copy2(path, staging / path.name)
    staging.rename(target)
    return target


class LoongDataset:
    @classmethod
    def load(cls, harness: "HarnessRuntime", max_examples: int | None, seed: int = 0, language: str = "en") -> LoadedDataset:
        start = time.time()
        rng = random.Random(seed)
        dataset_id = make_dataset_id("loong", n=max_examples, seed=seed)
        questions_path, docs_root = ensure_questions(), ensure_docs()
        financial_files = sorted(entry.name for entry in (docs_root / "financial").iterdir() if entry.is_file())
        rows, skipped = [], {}
        with open(questions_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("language") != language:
                    continue
                paths = resolve_documents(row, docs_root, financial_files)
                if paths is None:
                    skipped[row["type"]] = skipped.get(row["type"], 0) + 1
                    continue
                rows.append((row, paths))
        if skipped:
            print(f"{dataset_id} - skipped instances with unresolved documents: {skipped}", flush=True)
        rng.shuffle(rows)
        if max_examples is not None:
            rows = rows[:max_examples]
        documents: dict[str, DatasetDocument] = {}
        tasks: dict[str, DatasetTask] = {}
        for row, paths in rows:
            instance_id = str(row["id"])
            docs_path = ensure_instance(instance_id, paths)
            names = [path.name for path in paths]
            for path in paths:
                doc_id = f"loong/{path.name}"
                if doc_id not in documents:
                    documents[doc_id] = DatasetDocument(doc_id=doc_id, dataset_id=dataset_id, text=path.read_text(encoding="utf-8", errors="replace")[:MAX_DOC_CHARS])
            task = f"The documents are in {DOCS_PATH} ({len(names)} files: {', '.join(names)}). {str(row.get('instruction') or '').strip()}\n\n{str(row['question']).strip()}"
            tasks[instance_id] = DatasetTask(
                task_id=instance_id, dataset_id=dataset_id,
                task_datum={"docs_path": str(docs_path), "level": row.get("level"), "set": row.get("set"), "type": row["type"],
                            "n_docs": len(names), "length": row.get("length")},
                reference_metrics_kind=DatasetTaskMetricsKind.UNSCORED,
                gold_answer=str(row["answer"]),
                agent_prompt=bare_prompt(task, ANSWER_RULES["free"]), task_kind=DatasetTaskKind.FILE_SEARCH,
                env_setups={"docs": {"class": COPY_TREE_SETUP, "kwargs": {"datum_key": "docs_path", "env_path": DOCS_PATH}}},
            )
        loaded = LoadedDataset(dataset_id=dataset_id, documents=documents, scorable_tasks=tasks)
        loaded.stats = initialize_dataset_stats(loaded, load_time=time.time() - start)
        return loaded
