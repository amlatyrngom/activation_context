"""Prepare reusable raw source JSONL, independent of synthesis cache identity.

Only raw document corpora are read. This is a bounded preparation script, not a
benchmark ingestion path. Source URLs/revisions and original document IDs stay
with every window; a downstream study can group windows by document again.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import tarfile
import urllib.request
import xml.etree.ElementTree as ET


PINNED = {
    "regulations": ("common-pile/regulations", "d1a63eef07bb893f834742f4738aeb1d545b7e01"),
    "caselaw": ("common-pile/caselaw_access_project", "3c2cb5080b3a16a04d8d8d07b28eaec7c1ba7a90"),
    "libretexts": ("common-pile/libretexts", "d4ce4b1edf95888b24992db3bbb2ae8e2bf6c22f"),
    "libreoffice": ("LibreOffice/help", "d811050d9789f52c01962a67ad210517aa64db3a"),
    "peps": ("python/peps", "b48713b1628027934bb322d2385343cf11101816"),
    "office_automation": ("MicrosoftDocs/VBA-Docs", "4a944c587afa362c5bc2490211f50736181cfc62"),
}


def clean(text):
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text.replace("\r\n", "\n")).strip()


def windows(text, size=9000, max_windows=5):
    """Bound per-document contribution, spreading selections over long texts."""
    text = clean(text)
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        part = text[start:end].strip()
        if len(part) >= 300:
            pieces.append((start, end, part))
        start = end + (end < len(text) and text[end] == "\n")
    if len(pieces) <= max_windows:
        return pieces
    indices = sorted({round(i * (len(pieces) - 1) / (max_windows - 1)) for i in range(max_windows)})
    return [pieces[i] for i in indices]


class Pool:
    def __init__(self, root, name):
        self.root, self.name = root, name
        self.tmp = root / (name + ".jsonl.partial")
        self.out = self.tmp.open("w")
        self.seen, self.docs = set(), set()
        self.counts = collections.Counter()
        self.total = 0
        self.samples = []

    def add(self, text, *, doc_id, domain, subdomain, kind, url, corpus, revision, title="", metadata=None, max_windows=5):
        for start, end, part in windows(text, max_windows=max_windows):
            digest = hashlib.sha256(part.encode()).hexdigest()
            if digest in self.seen:
                continue
            self.seen.add(digest)
            row = {"source_id": digest, "sha256": digest, "doc_id": doc_id,
                   "text": part, "source_kind": kind, "domain": domain, "subdomain": subdomain,
                   "title": title, "provenance": {"url": url, "corpus": corpus, "revision": revision,
                   "document_offsets_chars": [start, end], "metadata": metadata or {}}}
            self.out.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.docs.add(doc_id)
            self.counts[subdomain] += 1
            self.total += 1
            if len(self.samples) < 4:
                self.samples.append({k: row[k] for k in ("doc_id", "subdomain", "title", "provenance")} | {"text_preview": part[:500]})

    def finish(self):
        self.out.close()
        target = self.root / (self.name + ".jsonl")
        self.tmp.replace(target)
        digest = hashlib.sha256()
        with target.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        result = {"file": target.name, "rows": self.total, "documents": len(self.docs),
                  "by_subdomain": dict(self.counts), "bytes": target.stat().st_size,
                  "sha256": digest.hexdigest(), "samples": self.samples}
        (self.root / (self.name + ".manifest.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)
        return result


def stream_hf(key, filename):
    repo, rev = PINNED[key]
    url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{filename}"
    with urllib.request.urlopen(url, timeout=120) as response, gzip.GzipFile(fileobj=response) as compressed:
        for line in compressed:
            yield json.loads(line)


def legal(root):
    pool = Pool(root, "law")
    # Sample all seven files so the first agency/chronological block cannot dominate.
    for shard in range(7):
        kept = 0
        for row in stream_hf("regulations", f"v0/{shard:05d}_regulations.jsonl.gz"):
            if row.get("document_type", "").casefold() not in {"notice", "rule", "proposed rule"}:
                continue
            text = row.get("text", "")
            if len(text) < 900:
                continue
            lines = [line for line in text.splitlines() if line.strip()]
            if len(lines) >= 40 and sum(len(line.split()) <= 2 for line in lines) / len(lines) > .65:
                continue  # Obvious fragmented PDF extraction, not a legal-content audit.
            repo, rev = PINNED["regulations"]
            meta = row.get("metadata", {})
            pool.add(text, doc_id="regulations/" + row["id"], domain="law", subdomain="regulations",
                     kind="regulatory_document_window", url=meta.get("url", ""), corpus=repo, revision=rev,
                     title=row.get("title", ""), metadata={k: row.get(k) for k in ("agency", "document_type", "posted_date")}, max_windows=3)
            kept += 1
            if kept >= 1500:
                break
        print("regulations shard", shard, "docs", kept, "windows", pool.total, flush=True)
    # Multiple public-opinion shards, with bounded contribution from any opinion.
    for shard in (0, 7, 14, 21, 28, 35):
        kept = 0
        for row in stream_hf("caselaw", f"cap_{shard:05d}.jsonl.gz"):
            text = row.get("text", "")
            if len(text) < 900:
                continue
            repo, rev = PINNED["caselaw"]
            meta = row.get("metadata", {})
            pool.add(text, doc_id="caselaw/" + row["id"], domain="law", subdomain="judicial_opinions",
                     kind="judicial_opinion_window", url=meta.get("url", ""), corpus=repo, revision=rev,
                     title=text.strip().splitlines()[0][:300], metadata={"license": meta.get("license")}, max_windows=3)
            kept += 1
            if kept >= 1500:
                break
        print("caselaw shard", shard, "docs", kept, "windows", pool.total, flush=True)
    return [pool.finish()]


def libretexts(root):
    math = Pool(root, "math_libretexts")
    office = Pool(root, "office_business")
    # These are corpus-page filters, not benchmark-question matching.
    excluded = re.compile(r"(?:exercises?|homework|practice[_ +%-]*(?:problems|tests)|(?:answer|solution)[_ +%-]*(?:key|manual)|review[_ +%-]*(?:questions|problems))", re.I)
    office_topics = re.compile(r"(?:management|business_communication|organizational|accounting|human_resource|business_law|information_system|operations|project_management|leadership|workplace)", re.I)
    for row in stream_hf("libretexts", "00000_libretext.json.gz"):
        meta = row.get("metadata", {})
        url = meta.get("url") or row["id"]
        title = meta.get("title") or ""
        if excluded.search(url + " " + title) or "_New_Page" in url or any(part in url for part in ("/Front_Matter/", "/Back_Matter/")):
            continue
        text = row.get("text", "")
        if text.startswith("Skip to main content") and "Enter Reader Mode" in text:
            text = text.split("Enter Reader Mode", 1)[1].lstrip()
        text = "\n".join(line for line in text.splitlines() if "Map MindTouch.Deki.Logic.ExtensionProcessor" not in line)
        if len(text) < 900:
            continue
        if re.search(r"(?im)^(?:Exercise\b|(?:#+\s*)?Exercises?\s*:?$|Problem\s+\d+\b|Answers to\b)", text):
            continue  # Explanatory pages with embedded exercise sets are omitted too.
        if "math.libretexts.org/" in url or "stats.libretexts.org/" in url:
            pool, domain, subdomain = math, "math", "mathematical_exposition"
        elif "biz.libretexts.org/" in url and office_topics.search(url + " " + title):
            pool, domain, subdomain = office, "office", "business_management"
        else:
            continue
        repo, rev = PINNED["libretexts"]
        pool.add(text, doc_id=url, domain=domain, subdomain=subdomain, kind="textbook_exposition_window",
                 url=url, corpus=repo, revision=rev, title=title,
                 metadata={k: meta.get(k) for k in ("book_url", "author", "license")}, max_windows=4)
    return [math.finish(), office.finish()]


def tar_members(key):
    repo, rev = PINNED[key]
    url = f"https://codeload.github.com/{repo}/tar.gz/{rev}"
    with urllib.request.urlopen(url, timeout=120) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
        for member in archive:
            if member.isfile() and member.size < 2_000_000:
                yield member.name, archive.extractfile(member).read()


def libreoffice(root):
    pool = Pool(root, "office_libreoffice")
    repo, rev = PINNED["libreoffice"]
    errors = 0
    for filename, raw in tar_members("libreoffice"):
        # English authored help, not translations or toolchain fixtures.
        if "/source/text/" not in filename or not filename.endswith(".xhp"):
            continue
        relative = filename.split("/", 1)[1]
        if not any(f"/text/{app}/" in filename for app in ("scalc", "swriter", "simpress", "sdraw", "shared", "sbasic", "smath")):
            continue
        try:
            document = ET.fromstring(raw)
        except ET.ParseError:
            errors += 1
            continue
        # Preserve inline words and paragraph/table boundaries. Exclude help-system
        # metadata; links retain their visible labels, never their XML mechanics.
        body = document.find("body")
        if body is None:
            continue
        blocks = []
        for element in body.iter():
            if element.tag in {"paragraph", "h1", "h2", "h3", "note", "tip", "warning"}:
                block = "".join(element.itertext()).strip()
                if block:
                    blocks.append(block)
        text = "\n\n".join(blocks).replace("%PRODUCTNAME", "LibreOffice").replace("$[officename]", "LibreOffice")
        title_element = document.find("meta/topic/title")
        title = "" if title_element is None else "".join(title_element.itertext())
        app = relative.split("/")[2]
        pool.add(text, doc_id=f"github/{repo}/{relative}", domain="office", subdomain="libreoffice_" + app,
                 kind="office_application_manual", url=f"https://github.com/{repo}/blob/{rev}/{relative}",
                 corpus=repo, revision=rev, title=title, max_windows=4)
    print("LibreOffice XML parse failures", errors, flush=True)
    return [pool.finish()]


def peps(root):
    pool = Pool(root, "technical_peps")
    repo, rev = PINNED["peps"]
    for filename, raw in tar_members("peps"):
        relative = filename.split("/", 1)[1]
        if not re.fullmatch(r"peps/pep-\d+\.rst", relative):
            continue
        text = raw.decode("utf-8", errors="replace")
        match = re.search(r"^Title: (.*)$", text, re.M)
        pool.add(text, doc_id=f"github/{repo}/{relative}", domain="documents", subdomain="python_peps",
                 kind="technical_specification_window", url=f"https://github.com/{repo}/blob/{rev}/{relative}",
                 corpus=repo, revision=rev, title=match.group(1) if match else relative, max_windows=5)
    return [pool.finish()]


def office_automation(root):
    pool = Pool(root, "office_automation")
    repo, rev = PINNED["office_automation"]
    for filename, raw in tar_members("office_automation"):
        if not filename.endswith(".md") or "/api/" not in filename and "/concepts/" not in filename:
            continue
        relative = filename.split("/", 1)[1]
        # Keep named end-user Office applications, excluding unrelated language
        # reference and infrastructure (their source group would be code).
        app = next((app for app in ("excel", "word", "powerpoint", "outlook", "access", "project", "publisher", "visio")
                    if re.search(rf"(?:/|^){app}(?:\.|/)", relative, re.I)), None)
        if app is None:
            continue
        text = raw.decode("utf-8", errors="replace")
        text = re.sub(r"\A---\n.*?\n---\n", "", text, count=1, flags=re.S)
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
        # Strip repetitive footer but keep code examples and visible link labels.
        text = re.split(r"^## (?:Support and feedback|See also)\s*$", text, maxsplit=1, flags=re.M)[0]
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        title = next((line.lstrip("# ") for line in text.splitlines() if line.startswith("# ")), relative)
        pool.add(text, doc_id=f"github/{repo}/{relative}", domain="office", subdomain="office_automation_" + app,
                 kind="office_automation_documentation", url=f"https://github.com/{repo}/blob/{rev}/{relative}",
                 corpus=repo, revision=rev, title=title, max_windows=3)
    return [pool.finish()]


def repository_docs(root):
    pool = Pool(root, "technical_repository_docs")
    cache = Path("IB/TMP/SYNC/REPO_CACHE")
    for repo_dir in sorted(cache.iterdir()):
        if not repo_dir.is_dir():
            continue
        repo = repo_dir.name.replace("__", "/", 1)
        for checkout in sorted(repo_dir.iterdir()):
            if not checkout.is_dir() or not re.fullmatch(r"[a-f0-9]{40}", checkout.name):
                continue
            candidates = []
            for base, dirs, files in os.walk(checkout):
                dirs[:] = [name for name in dirs if not name.startswith(".") and name not in
                           {"node_modules", "vendor", "build", "dist", "target", "tests", "test", "benchmarks"}]
                for name in files:
                    path = Path(base) / name
                    relative = path.relative_to(checkout)
                    if path.suffix.lower() not in {".md", ".rst", ".mdx"} or path.is_symlink():
                        continue
                    if not (any(part.lower() in {"doc", "docs", "documentation", "guides"} for part in relative.parts[:-1])
                            or name.lower().startswith(("readme", "contributing"))):
                        continue
                    if not 900 <= path.stat().st_size <= 200000:
                        continue
                    candidates.append(path)
            random.Random(repo + checkout.name).shuffle(candidates)
            for path in candidates[:200]:
                relative = path.relative_to(checkout).as_posix()
                pool.add(path.read_text(errors="replace"), doc_id=f"github/{repo}/{relative}", domain="documents",
                         subdomain="repository_documentation", kind="technical_manual_window",
                         url=f"https://github.com/{repo}/blob/{checkout.name}/{relative}", corpus=repo,
                         revision=checkout.name, title=f"{repo}: {relative}", max_windows=3)
    return [pool.finish()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("IB/TMP/FIRST_TASK_AWARE_FINETUNING/expanded_sources"))
    parser.add_argument("--tasks", nargs="+", default=["law", "libretexts", "libreoffice", "peps", "office_automation", "repository_docs"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "PRESERVE").write_text("Reusable raw source pools with provenance, retained independently of synthesis cache keys.\n")
    handlers = {"law": legal, "libretexts": libretexts, "libreoffice": libreoffice, "peps": peps,
                "office_automation": office_automation, "repository_docs": repository_docs}
    results, errors = [], {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        jobs = {executor.submit(handlers[name], args.output): name for name in args.tasks}
        for future in concurrent.futures.as_completed(jobs):
            name = jobs[future]
            try:
                results.extend(future.result())
            except Exception as error:
                errors[name] = repr(error)
                print("FAILED", name, repr(error), flush=True)
    # Pool manifests make partial task reruns additive without modifying old pools.
    results = [json.loads(path.read_text()) for path in sorted(args.output.glob("*.manifest.json"))]
    result = {"schema_version": 1, "pinned_sources": PINNED, "pools": results, "errors": errors,
              "note": "Raw corpus material only, no benchmark questions/answers. Exact text windows and original document IDs are retained. Sources are dated snapshots, not current legal advice. Office material is text, not spreadsheet or GUI task execution."}
    (args.output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
