"""
LongCodeArena bug localization (`JetBrains-Research/lca-bug-localization`): an issue and the repository at the base
commit; the task is to name the file that must change. The loader fetches the base commit's tarball into a local cache
(REPO_CACHE/<owner>__<repo>/<sha>, no git history) and the task's CopyTreeSetup copies it to /workspace/repo in the
sandbox. Exact match on any changed file's path (the first is the gold, the rest aliases). The repository's text files
are corpus documents so semantic_search works over the code. `max_changed_files` and `max_repo_files` keep tasks
localizable and checkouts small.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import io
import os
import random
import shutil
import tarfile
import threading
import time
import urllib.error
import urllib.request
import typing as t
from pathlib import Path

from datasets import load_dataset

from activation.common.data_syncing import resolve_path

from ..dataset import DatasetDocument, DatasetTask, DatasetTaskMetricsKind, LoadedDataset
from ..dataset_utils import initialize_dataset_stats, make_dataset_id
from ..dataset import ANSWER_RULES, DatasetTaskKind, bare_prompt

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime

HF_DATASET = "JetBrains-Research/lca-bug-localization"
REPO_CACHE = "REPO_CACHE"
REPO_PATH = "/workspace/repo"
COPY_TREE_SETUP = "activation.agent.agent_env:CopyTreeSetup"
TEXT_SUFFIXES = {".py", ".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".java", ".kt", ".kts", ".gradle",
                 ".xml", ".sh", ".js", ".ts", ".html", ".css", ".c", ".h", ".cpp", ".hpp", ".go", ".rs", ".rb", ".sql"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".idea", ".mypy_cache"}
MAX_DOC_CHARS = 200_000



def _tar_filter(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
    """
    The standard data filter, except that links the filter rejects (absolute or escaping targets) are skipped
    instead of failing the whole checkout: the localization tasks only need the regular files.
    """
    try:
        return tarfile.data_filter(member, path)
    except tarfile.FilterError:
        if member.islnk() or member.issym():
            return None
        raise

def ensure_checkout(owner: str, repo: str, sha: str, timeout: int = 600) -> str:
    """
    The repository at `sha` under the cache, fetched once as GitHub's commit tarball (no git needed, no history);
    returns the local path. A partial download never leaves a marker, so a retry starts over.
    """
    target = resolve_path(f"{REPO_CACHE}/{owner}__{repo}", create=True) / sha
    with _CHECKOUT_LOCKS_GUARD:
        lock = _CHECKOUT_LOCKS.setdefault(str(target), threading.Lock())
    with lock:                                                   # rows sharing a base commit fetch it once, even in parallel
        return _ensure_checkout_locked(target, owner, repo, sha, timeout)


_CHECKOUT_LOCKS_GUARD = threading.Lock()
_CHECKOUT_LOCKS: dict[str, threading.Lock] = {}


def _ensure_checkout_locked(target: Path, owner: str, repo: str, sha: str, timeout: int) -> str:
    marker = target / ".activation_checkout_complete"
    if marker.exists():
        return str(target)
    if target.exists():
        shutil.rmtree(target)
    url = f"https://github.com/{owner}/{repo}/archive/{sha}.tar.gz"
    staging = target.parent / f".{sha}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "activation-suite"}), timeout=timeout) as response:
            with tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz") as archive:
                archive.extractall(staging, filter=_tar_filter)
        roots = [entry for entry in staging.iterdir() if entry.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"unexpected archive layout {sorted(entry.name for entry in staging.iterdir())}")
        roots[0].rename(target)
        marker.write_text(f"{url}\n")
    except (urllib.error.URLError, tarfile.TarError, OSError, RuntimeError) as error:
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError(f"fetching {owner}/{repo}@{sha[:10]} failed: {error}") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return str(target)


def iter_text_files(root: str, max_files: int) -> t.Iterator[tuple[str, str]]:
    """(relative path, text) of the repository's text files in sorted order, up to max_files (none when 0: the suite offers ripgrep, not a passage index, over code)."""
    if max_files <= 0:
        return
    count = 0
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.suffix.lower() not in TEXT_SUFFIXES or name.startswith(".activation_"):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if not text.strip():
                continue
            yield str(path.relative_to(root)), text[:MAX_DOC_CHARS]
            count += 1
            if count >= max_files:
                return


class LcaBugLocalizationDataset:
    @classmethod
    def load(cls, harness: "HarnessRuntime", max_examples: int | None, seed: int = 0, config: str = "py", split: str = "dev",
             max_changed_files: int = 3, max_repo_files: int = 3000, max_issue_chars: int = 6000, max_tasks_per_repo: int = 20,
             index_files_per_repo: int = 2000, fetch_workers: int = 8) -> LoadedDataset:
        start = time.time()
        rng = random.Random(seed)
        dataset_id = make_dataset_id("lca_bug_localization", config=config, split=split, n=max_examples, seed=seed)
        documents: dict[str, DatasetDocument] = {}
        tasks: dict[str, DatasetTask] = {}
        per_repo: dict[str, int] = {}                # the split is grouped by repository; the cap spreads tasks over many
        candidates: list[tuple[dict, list[str]]] = []
        margin = None if max_examples is None else max_examples + max(20, max_examples // 5)     # fetch failures and missing gold files drop rows
        for row in load_dataset(HF_DATASET, config, split=split, streaming=True):
            changed = [str(path) for path in ast.literal_eval(row["changed_files"])]
            if not changed or len(changed) > max_changed_files or int(row["repo_files_without_tests_count"]) > max_repo_files:
                continue
            if int(row.get("code_changed_files_count") or 0) == 0:
                continue
            repo = f"{row['repo_owner']}/{row['repo_name']}"
            if per_repo.get(repo, 0) >= max_tasks_per_repo:
                continue
            per_repo[repo] = per_repo.get(repo, 0) + 1
            candidates.append((row, changed))
            if margin is not None and len(candidates) >= margin:
                break
        # The checkouts are fetched in parallel (GitHub tarballs, one per base commit; cached across runs), in the row order.
        def fetch(item: tuple[dict, list[str]]) -> str | None:
            row = item[0]
            try:
                return ensure_checkout(str(row["repo_owner"]), str(row["repo_name"]), str(row["base_sha"]))
            except RuntimeError as error:
                print(f"{dataset_id} - skipping {row['text_id']}: {error}", flush=True)
                return None
        with ThreadPoolExecutor(max_workers=fetch_workers) as pool:
            checkouts = list(pool.map(fetch, candidates))
        for (row, changed), local in zip(candidates, checkouts):
            if local is None:
                continue
            if not any(os.path.exists(os.path.join(local, path)) for path in changed):
                continue                                                                        # the gold must exist at the base commit
            task_id = str(row["text_id"]).replace("/", "__")
            issue = f"Issue: {row['issue_title']}\n\n{str(row['issue_body'] or '')[:max_issue_chars]}"
            task = (f"The repository {row['repo_owner']}/{row['repo_name']} is checked out at {REPO_PATH} (the state before the fix). "
                    f"The issue below was fixed by changing one file. Find that file.\n\n{issue}")
            for relative, text in iter_text_files(local, index_files_per_repo):
                doc_id = f"{row['repo_owner']}/{row['repo_name']}@{str(row['base_sha'])[:8]}/{relative}"
                documents.setdefault(doc_id, DatasetDocument(doc_id=doc_id, dataset_id=dataset_id, text=f"{relative}\n{text}"))
            tasks[task_id] = DatasetTask(
                task_id=task_id, dataset_id=dataset_id,
                task_datum={"repo": f"{row['repo_owner']}/{row['repo_name']}", "base_sha": row["base_sha"], "issue_title": row["issue_title"],
                            "issue_body": row["issue_body"], "changed_files": changed, "checkout_path": local,
                            "issue_url": row.get("issue_url")},
                reference_metrics_kind=DatasetTaskMetricsKind.EXACT_MATCH, gold_answer=changed[0], gold_answer_aliases=changed[1:],
                agent_prompt=bare_prompt(task, ANSWER_RULES["path"]), task_kind=DatasetTaskKind.CODE_SEARCH,
                env_setups={"repo": {"class": COPY_TREE_SETUP, "kwargs": {"datum_key": "checkout_path", "env_path": REPO_PATH}}},
            )
            if max_examples is not None and len(tasks) >= max_examples:
                break
        loaded = LoadedDataset(dataset_id=dataset_id, documents=documents, scorable_tasks=tasks)
        loaded.stats = initialize_dataset_stats(loaded, load_time=time.time() - start)
        return loaded
