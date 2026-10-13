"""
LongCodeArena bug localization (`JetBrains-Research/lca-bug-localization`): an issue and the repository at the base
commit; the task is to name the file that must change. Exact match on any changed file's path (the first is the gold,
the rest aliases). `max_changed_files` and `max_repo_files` keep tasks localizable and checkouts small.

Without a passage index (`index_files_per_repo=0`, the suite's setting: agents use ripgrep) nothing is fetched at load:
each rollout's RepoTarballSetup downloads the base commit's GitHub tarball into node-local temp space, checks the gold
file exists, copies the tree to /workspace/repo in the sandbox and deletes the download. A repository that is gone
(or keeps failing to fetch) makes the task unavailable: the run ends before its first turn, recorded and skipped.
With an index, the loader fetches every checkout into a persistent cache (REPO_CACHE/<owner>__<repo>/<sha>) at load,
indexes its text files as corpus documents, drops tasks whose gold is absent, and CopyTreeSetup copies from the cache.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import os
import random
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import typing as t
from pathlib import Path

from datasets import load_dataset

from activation.agent.agent_env import AgentEnvSetup, TaskUnavailable
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
REPO_TARBALL_SETUP = f"{__name__}:RepoTarballSetup"
PERMANENT_HTTP = {404, 410, 451}                 # repository or commit gone / blocked: retrying cannot help
FETCH_ATTEMPTS, FETCH_BACKOFF_SECONDS = 4, 15
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
    url = tarball_url(owner, repo, sha)
    staging = target.parent / f".{sha}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        _extract_tarball(url, staging, timeout).rename(target)
        marker.write_text(f"{url}\n")
    except (urllib.error.URLError, tarfile.TarError, OSError, RuntimeError) as error:
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError(f"fetching {owner}/{repo}@{sha[:10]} failed: {error}") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return str(target)


def tarball_url(owner: str, repo: str, sha: str) -> str:
    return f"https://github.com/{owner}/{repo}/archive/{sha}.tar.gz"


def _extract_tarball(url: str, staging: Path, timeout: int) -> Path:
    """
    Downloads and extracts a GitHub commit tarball into `staging`; returns its single top-level directory. The archive is
    streamed to disk, never held in memory: some repositories are GBs, and a dozen concurrent in-memory downloads
    fragmented the rollout threads' heaps until the host OOM-killed the campaign.
    """
    archive_path = staging / "archive.tar.gz"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "activation-suite"}), timeout=timeout) as response:
        with open(archive_path, "wb") as sink:
            shutil.copyfileobj(response, sink, 1 << 20)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        archive.extractall(staging, filter=_tar_filter)
    archive_path.unlink()
    roots = [entry for entry in staging.iterdir() if entry.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"unexpected archive layout {sorted(entry.name for entry in staging.iterdir())}")
    return roots[0]


def fetch_tarball(owner: str, repo: str, sha: str, staging: Path, timeout: int = 600) -> Path:
    """
    The repository at `sha`, extracted under `staging` (no cache). A permanent HTTP failure (gone, blocked) raises
    TaskUnavailable at once; transient failures (timeouts, 5xx, torn downloads) retry with backoff, then raise it too.
    """
    url, last = tarball_url(owner, repo, sha), None
    for attempt in range(FETCH_ATTEMPTS):
        attempt_dir = staging / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True)
        try:
            return _extract_tarball(url, attempt_dir, timeout)
        except urllib.error.HTTPError as error:
            if error.code in PERMANENT_HTTP:
                raise TaskUnavailable(f"{owner}/{repo}@{sha[:10]}: HTTP {error.code}") from error
            last = error
        except (urllib.error.URLError, tarfile.TarError, OSError, RuntimeError, TimeoutError) as error:
            last = error
        shutil.rmtree(attempt_dir, ignore_errors=True)
        if attempt + 1 < FETCH_ATTEMPTS:
            time.sleep(FETCH_BACKOFF_SECONDS * 3 ** attempt * random.uniform(.5, 1.5))   # jittered: many rollouts start together
    raise TaskUnavailable(f"{owner}/{repo}@{sha[:10]}: fetch failed {FETCH_ATTEMPTS} times ({last})")


class RepoTarballSetup(AgentEnvSetup):
    """
    Per rollout: the task's repository at its base commit, downloaded into node-local temp space, checked to contain a
    gold file, copied into the sandbox at `env_path`, and deleted (the sandbox has its own copy; nothing persists on the host).
    """

    def __init__(self, harness, agent, env_path: str):
        super().__init__(harness, agent)
        self.env_path = env_path

    def setup(self, agent_env) -> None:
        datum = self.task_datum
        owner, repo = str(datum["repo"]).split("/", 1)
        sha = str(datum["base_sha"])
        with tempfile.TemporaryDirectory(prefix="lca-checkout-") as temporary:
            root = fetch_tarball(owner, repo, sha, Path(temporary))
            if not any((root / path).is_file() for path in datum["changed_files"]):
                raise TaskUnavailable(f"{owner}/{repo}@{sha[:10]}: no changed file exists at the base commit")
            agent_env.copy_in(str(root), self.env_path)


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
        lazy = index_files_per_repo <= 0                    # no passage index: rollouts fetch their own checkout (module docstring)
        # Eager fetching drops rows (fetch failures, absent gold) at load, so it reads a margin; lazy rows are all kept.
        margin = None if max_examples is None else max_examples if lazy else max_examples + max(20, max_examples // 5)
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
        if lazy:
            checkouts = [None] * len(candidates)
        else:
            with ThreadPoolExecutor(max_workers=fetch_workers) as pool:
                checkouts = list(pool.map(fetch, candidates))
        for (row, changed), local in zip(candidates, checkouts):
            if not lazy and local is None:
                continue
            if not lazy and not any(os.path.exists(os.path.join(local, path)) for path in changed):
                continue                                                                        # the gold must exist at the base commit
            task_id = str(row["text_id"]).replace("/", "__")
            issue = f"Issue: {row['issue_title']}\n\n{str(row['issue_body'] or '')[:max_issue_chars]}"
            task = (f"The repository {row['repo_owner']}/{row['repo_name']} is checked out at {REPO_PATH} (the state before the fix). "
                    f"The issue below was fixed by changing one file. Find that file.\n\n{issue}")
            for relative, text in (iter_text_files(local, index_files_per_repo) if not lazy else ()):
                doc_id = f"{row['repo_owner']}/{row['repo_name']}@{str(row['base_sha'])[:8]}/{relative}"
                documents.setdefault(doc_id, DatasetDocument(doc_id=doc_id, dataset_id=dataset_id, text=f"{relative}\n{text}"))
            tasks[task_id] = DatasetTask(
                task_id=task_id, dataset_id=dataset_id,
                task_datum={"repo": f"{row['repo_owner']}/{row['repo_name']}", "base_sha": row["base_sha"], "issue_title": row["issue_title"],
                            "issue_body": row["issue_body"], "changed_files": changed, "issue_url": row.get("issue_url"),
                            **({"tarball_url": tarball_url(str(row["repo_owner"]), str(row["repo_name"]), str(row["base_sha"]))}
                               if lazy else {"checkout_path": local})},
                reference_metrics_kind=DatasetTaskMetricsKind.EXACT_MATCH, gold_answer=changed[0], gold_answer_aliases=changed[1:],
                agent_prompt=bare_prompt(task, ANSWER_RULES["path"]), task_kind=DatasetTaskKind.CODE_SEARCH,
                env_setups={"repo": {"class": REPO_TARBALL_SETUP, "kwargs": {"env_path": REPO_PATH}} if lazy else
                            {"class": COPY_TREE_SETUP, "kwargs": {"datum_key": "checkout_path", "env_path": REPO_PATH}}},
            )
            if max_examples is not None and len(tasks) >= max_examples:
                break
        loaded = LoadedDataset(dataset_id=dataset_id, documents=documents, scorable_tasks=tasks)
        loaded.stats = initialize_dataset_stats(loaded, load_time=time.time() - start)
        return loaded
