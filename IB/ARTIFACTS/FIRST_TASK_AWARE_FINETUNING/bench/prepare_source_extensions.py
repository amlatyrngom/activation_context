"""Publish immutable source-family extensions without changing the shipped bank."""
from __future__ import annotations
import argparse, collections, concurrent.futures, hashlib, io, json, os, random, re, tarfile, urllib.request
from urllib.parse import urlparse
from pathlib import Path
import pyarrow.parquet as pq
import ijson
from prepare_extra_sources import Pool, stream_hf, PINNED, windows

PINNED_EXT = {
    "financial": ("astr010/sec-10k-lsh-chunks", "02417379de1ce5ebd1f7abb1ecc75e3da31af684"),
    "bright": ("xlangai/BRIGHT", "3066d29c9651a576c8aba4832d249807b181ecae"),
    "mdn": ("mdn/content", "08a5a4313166a8f36f8c0c1320ac970423239b9a"),
    "python": ("python/cpython", "643013f58f703b39771fa35bbdba0ac762dd840a"),
    "kubernetes": ("kubernetes/website", "61274ef8907c17b00626fa7910861badf89f5213"),
    "open_web_math": ("open-web-math/open-web-math", "fde8ef8de2300f5e778f56261843dab89f230815"),
    "ms_marco": ("microsoft/ms_marco", "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a"),
}
SEED = 20260923
USED = set()
OLD_DOCS = {}


def pool(root, name):
    if (root / (name + ".jsonl")).exists():
        raise FileExistsError("Published extension is immutable: " + name)
    result = Pool(root, name)
    result.seen.update(USED)
    return result


def parquet_rows(key, filename):
    repo, rev = PINNED_EXT[key]
    url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{filename}"
    with urllib.request.urlopen(url, timeout=120) as response:
        raw = response.read()
    table = pq.ParquetFile(io.BytesIO(raw))
    for batch in table.iter_batches(batch_size=256):
        yield from batch.to_pylist()


def financial(root):
    result = pool(root, "financial__sec_filings_20260921_v1")
    repo, rev = PINNED_EXT["financial"]
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/revision/{rev}", timeout=60) as response:
        files = [r["rfilename"] for r in json.load(response)["siblings"] if r["rfilename"].endswith(".parquet")]
    random.Random(SEED).shuffle(files)
    companies = 0
    for filename in files:
        rows = list(parquet_rows("financial", filename))
        filings = collections.defaultdict(list)
        for row in rows:
            filings[(row['cik'], row['fiscal_year'])].append(row)
        selections = []
        for (_, year), chunks in sorted(filings.items()):
            chunks.sort(key=lambda row: row['chunk_id'])
            # Corpus blocks are ~500 characters, despite its token-count label.
            # Rejoin contiguous original chunks into meaningful source windows.
            for start in range(0, len(chunks), 20):
                block = chunks[start:start+20]
                if sum(bool(row.get('is_boilerplate')) for row in block) / len(block) > .7:
                    continue
                text = '\n'.join(row['text'] for row in block)
                if len(text) < 1500 or sum(c.isdigit() for c in text) < 25:
                    continue
                selections.append((block, text))
        random.Random(SEED + companies).shuffle(selections)
        kept = 0
        for block, text in selections:
            row = block[0]
            before = result.total
            result.add(f"Company ticker: {row['ticker']}. Annual filing, fiscal year {row['fiscal_year']}.\n\n" + text,
                       doc_id=f"sec10k/{row['cik']}/{row['fiscal_year']}", domain="financial", subdomain="sec_annual_filings",
                       kind="financial_filing_table" if any(r.get("is_table") for r in block) else "financial_filing_passage",
                       url=f"https://huggingface.co/datasets/{repo}/blob/{rev}/{filename}", corpus=repo, revision=rev,
                       title=f"{row['ticker']} annual filing {row['fiscal_year']}",
                       metadata={k: row.get(k) for k in ("ticker", "cik", "sic", "sic_description", "fiscal_year")} |
                       {"source_chunk_ids": [r['chunk_id'] for r in block], "has_table": any(r.get('is_table') for r in block)}, max_windows=2)
            kept += result.total - before
            if kept >= 300:
                break
        companies += 1
        print("financial companies", companies, "windows", result.total, flush=True)
        if result.total >= 18000:
            break
    return result.finish()


def open_web_exposition(root):
    result = pool(root, "math_textbook__web_exposition_20260921_v1")
    repo, rev = PINNED_EXT['open_web_math']
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/revision/{rev}", timeout=60) as response:
        files = [r['rfilename'] for r in json.load(response)['siblings'] if r['rfilename'].endswith('.parquet')]
    domains = {'mathworld.wolfram.com', 'planetmath.org', 'encyclopediaofmath.org', 'proofwiki.org', 'statlect.com',
               'probabilitycourse.com', 'online.stat.psu.edu', 'tutorial.math.lamar.edu', 'mathinsight.org',
               'openstax.org', 'mathcentre.ac.uk', 'mathsisfun.com', 'imomath.com'}
    per_host = collections.Counter()
    for filename in files[:12]:
        for row in parquet_rows('open_web_math', filename):
            url, text = row['url'], row['text']
            parsed = urlparse(url)
            host = parsed.hostname or ''
            allowed = any(host == domain or host.endswith('.' + domain) for domain in domains) or host.endswith('.edu')
            if not allowed or 'libretexts.org' in host or host in {'arxiv.org', 'export.arxiv.org'} or per_host[host] >= 1500:
                continue
            if re.search(r'homework|assign(?:ment)?|exam|quiz|olympiad|contest|problem[-_/ ]?set|solutions?', parsed.path, re.I):
                continue
            if len(text) < 1200 or re.search(r'(?im)^\s*(?:#+\s*)?(?:Exercises?|Homework|Problems?\s+\d+)\b', text):
                continue
            if len(re.findall(r'\\frac|\\sum|\\int|\\begin|\\alpha|\\beta|\\theta|\$[^\n$]+\$', text)) < 4:
                continue
            before = result.total
            result.add(text, doc_id=url, domain='math_textbook', subdomain='academic_math_web_exposition',
                       kind='mathematical_exposition_window', url=url, corpus=repo, revision=rev,
                       title=text.splitlines()[0][:200], metadata={'crawl_date': row.get('date'), 'host': host}, max_windows=5)
            per_host[host] += result.total - before
            if result.total >= 12000:
                break
        print('math exposition shard', filename, 'windows', result.total, flush=True)
        if result.total >= 12000:
            break
    return result.finish()


def technical_web(root):
    result = pool(root, "technical_web__bright_corpora_20260921_v1")
    repo, rev = PINNED_EXT["bright"]
    # Only raw documents. Never open examples, reasoning traces, AoPS/LeetCode
    # problem collections, or theoremqa_questions.
    for subject in ("robotics", "stackoverflow", "biology", "earth_science", "economics", "psychology", "sustainable_living", "pony"):
        # Uniform bounded reservoir, so a large corpus never becomes one giant
        # Python list while other preparation jobs are sharing the host.
        rows = []
        rng = random.Random(f"{SEED}:{subject}")
        for ordinal, row in enumerate(parquet_rows("bright", f"documents/{subject}-00000-of-00001.parquet")):
            if len(rows) < 5000:
                rows.append(row)
            else:
                index = rng.randrange(ordinal + 1)
                if index < len(rows):
                    rows[index] = row
        rng.shuffle(rows)
        before = result.total
        for row in rows:
            text = row.get("content") or row.get("text") or ""
            doc_id = f"bright/{subject}/{row['id']}"
            if doc_id in OLD_DOCS.get("technical_web", set()):
                continue
            result.add(text, doc_id=doc_id, domain="technical_web", subdomain="bright_" + subject,
                       kind="technical_post", url=f"https://huggingface.co/datasets/{repo}/blob/{rev}/documents/{subject}-00000-of-00001.parquet",
                       corpus=repo, revision=rev, title=str(row['id']), max_windows=2)
            if result.total - before >= 1800:
                break
        print("BRIGHT", subject, "new windows", result.total - before, flush=True)
    return result.finish()


def quantitative_exposition(root):
    result = pool(root, "math_textbook__quantitative_exposition_20260921_v1")
    repo, rev = PINNED["libretexts"]
    for row in stream_hf("libretexts", "00000_libretext.json.gz"):
        meta = row.get("metadata", {})
        url = meta.get("url") or row["id"]
        # New material is quantitative physics/engineering/statistical exposition,
        # tagged explicitly so its share can be changed independently later.
        if not any(domain in url for domain in ("phys.libretexts.org/", "eng.libretexts.org/", "chem.libretexts.org/")):
            continue
        title = meta.get("title") or ""
        if re.search(r"exercise|homework|practice[_ +%-]*(?:problems|tests)|(?:answer|solution)[_ +%-]*(?:key|manual)|review[_ +%-]*(?:questions|problems)|_New_Page|Front_Matter|Back_Matter", url + " " + title, re.I):
            continue
        text = row.get("text", "")
        if text.startswith("Skip to main content") and "Enter Reader Mode" in text:
            text = text.split("Enter Reader Mode", 1)[1].lstrip()
        text = "\n".join(line for line in text.splitlines() if "Map MindTouch.Deki.Logic.ExtensionProcessor" not in line)
        if len(text) < 1200 or re.search(r"(?im)^(?:Exercise\b|(?:#+\s*)?Exercises?\s*:?$|Problem\s+\d+\b|Answers to\b)", text):
            continue
        if len(re.findall(r"\\\(|\\\[|\\frac|\\sum|\\int|\\begin", text)) < 8:
            continue
        subject = "engineering" if "eng.libretexts" in url else "physics" if "phys.libretexts" in url else "quantitative_chemistry"
        result.add(text, doc_id=url, domain="math_textbook", subdomain="quantitative_" + subject,
                   kind="quantitative_textbook_exposition", url=url, corpus=repo, revision=rev, title=title,
                   metadata={k: meta.get(k) for k in ("book_url", "author", "license")}, max_windows=5)
        if result.total >= 16000:
            break
    return result.finish()


def manuals(root):
    result = pool(root, "documents__official_manuals_20260921_v1")
    for key in ("python", "kubernetes", "mdn"):
        repo, rev = PINNED_EXT[key]
        url = f"https://codeload.github.com/{repo}/tar.gz/{rev}"
        before = result.total
        with urllib.request.urlopen(url, timeout=120) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not 900 <= member.size <= 300000:
                    continue
                relative = member.name.split("/", 1)[1]
                eligible = (key == "python" and relative.startswith("Doc/") and relative.endswith(".rst")) or \
                           (key == "kubernetes" and relative.startswith("content/en/docs/") and relative.endswith(".md")) or \
                           (key == "mdn" and relative.startswith("files/en-us/") and relative.endswith("index.md"))
                if not eligible:
                    continue
                doc_id = f"github/{repo}/{relative}"
                if doc_id in OLD_DOCS.get("documents", set()):
                    continue
                text = archive.extractfile(member).read().decode("utf-8", errors="replace")
                text = re.sub(r"\A---\n.*?\n---\n", "", text, count=1, flags=re.S)
                result.add(text, doc_id=doc_id, domain="documents", subdomain="official_" + key + "_manual",
                           kind="technical_manual_window", url=f"https://github.com/{repo}/blob/{rev}/{relative}",
                           corpus=repo, revision=rev, title=relative, max_windows=5)
                if result.total >= 18000:
                    break
        print("manuals", key, "new windows", result.total - before, flush=True)
        if result.total >= 18000:
            break
    return result.finish()


def code(root):
    result = pool(root, "code__additional_cached_files_20260921_v1")
    cache = Path("IB/TMP/SYNC/REPO_CACHE")
    repos = sorted(cache.iterdir())
    random.Random(SEED).shuffle(repos)
    for repo_dir in repos:
        if not repo_dir.is_dir():
            continue
        files = []
        for base, dirs, names in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"node_modules", "vendor", "build", "dist", "target", "__pycache__"}]
            for name in names:
                path = Path(base) / name
                if path.suffix not in (".py", ".java", ".cpp", ".h", ".js", ".ts", ".go", ".rs", ".cs") or path.is_symlink():
                    continue
                if 900 <= path.stat().st_size <= 150000 and str(path.relative_to(cache)) not in OLD_DOCS.get("code", set()):
                    files.append(path)
        random.Random(f"{SEED}:{repo_dir.name}").shuffle(files)
        before = result.total
        for path in files:
            text = path.read_text(errors="replace")
            if any(len(line) > 500 for line in text.splitlines()):
                continue
            doc_id = str(path.relative_to(cache))
            parts = path.relative_to(repo_dir).parts
            repo, rev = repo_dir.name.replace("__", "/", 1), parts[0]
            relative = "/".join(parts[1:])
            result.add("File: " + path.name + "\n" + text, doc_id=doc_id, domain="code", subdomain="repository_source",
                       kind="code_window", url=f"https://github.com/{repo}/blob/{rev}/{relative}", corpus=repo, revision=rev,
                       title=f"{repo}: {relative}", max_windows=3)
            if result.total - before >= 400:
                break
        if result.total >= 24000:
            break
    return result.finish()


def office(root):
    result = pool(root, "office__business_workflows_20260921_v1")
    extra = root.parent / 'expanded_sources'
    groups = {}
    for kind, filename in {'automation': 'office_automation.jsonl', 'manuals': 'office_libreoffice.jsonl'}.items():
        with (extra / filename).open() as file:
            groups[kind] = [row for row in map(json.loads, file) if row['source_id'] not in USED]
    groups['business'] = []
    repo, rev = PINNED['libretexts']
    for row in stream_hf('libretexts', '00000_libretext.json.gz'):
        meta = row.get('metadata', {})
        url, title = meta.get('url') or row['id'], meta.get('title') or ''
        if 'biz.libretexts.org/' not in url or url in OLD_DOCS.get('office', set()):
            continue
        if re.search(r'exercise|homework|practice[_ +%-]*(?:problems|tests)|(?:answer|solution)[_ +%-]*(?:key|manual)|review[_ +%-]*(?:questions|problems)|_New_Page|Front_Matter|Back_Matter', url + ' ' + title, re.I):
            continue
        text = row.get('text', '')
        if text.startswith('Skip to main content') and 'Enter Reader Mode' in text:
            text = text.split('Enter Reader Mode', 1)[1].lstrip()
        text = '\n'.join(line for line in text.splitlines() if 'Map MindTouch.Deki.Logic.ExtensionProcessor' not in line)
        if re.search(r'(?im)^(?:Exercise\b|(?:#+\s*)?Exercises?\s*:?$|Problem\s+\d+\b|Answers to\b)', text):
            continue
        for start, end, part in windows(text, max_windows=4):
            sha = hashlib.sha256(part.encode()).hexdigest()
            if sha in USED:
                continue
            groups['business'].append({'source_id': sha, 'sha256': sha, 'doc_id': url, 'text': part,
                'domain': 'office', 'subdomain': 'business_commercial_operations', 'source_kind': 'textbook_exposition_window',
                'title': title, 'provenance': {'url': url, 'corpus': repo, 'revision': rev,
                'document_offsets_chars': [start, end], 'metadata': {k: meta.get(k) for k in ('book_url', 'author', 'license')}}})
    for kind, rows in groups.items():
        random.Random(f'{SEED}:office:{kind}').shuffle(rows)
    rounds = min(len(groups['automation']) // 6, len(groups['business']) // 3, len(groups['manuals']))
    offsets = collections.Counter()
    pattern = ('automation', 'business', 'automation', 'manuals', 'automation', 'business', 'automation', 'automation', 'business', 'automation')
    for _ in range(rounds):
        for kind in pattern:
            row = groups[kind][offsets[kind]]
            offsets[kind] += 1
            if row['source_id'] in result.seen:
                continue
            result.seen.add(row['source_id'])
            result.out.write(json.dumps(row, ensure_ascii=False) + '\n')
            result.docs.add(row['doc_id'])
            result.counts[row['subdomain']] += 1
            result.total += 1
    print('office available', {k: len(v) for k, v in groups.items()}, 'used', dict(offsets), flush=True)
    return result.finish()


def web(root):
    from huggingface_hub import HfFileSystem
    result = pool(root, 'web__msmarco_headroom_20260921_v1')
    repo, rev = PINNED_EXT['ms_marco']
    filename = 'v1.1/train-00000-of-00001.parquet'
    fs = HfFileSystem()
    ordinal = 0
    with fs.open(f'datasets/{repo}@{rev}/{filename}', 'rb') as file:
        parquet = pq.ParquetFile(file)
        # Project the single raw-text leaf at the Parquet reader. Native query,
        # answer and relevance fields never enter the materialization path.
        for batch in parquet.iter_batches(batch_size=512, columns=['passages.passage_text']):
            for row in batch.to_pylist():
                for passage, text in enumerate(row['passages']['passage_text']):
                    result.add(text, doc_id=f'msmarco:raw:{ordinal}:{passage}', domain='web',
                               subdomain='msmarco_raw_passage', kind='web_passage',
                               url=f'https://huggingface.co/datasets/{repo}/blob/{rev}/{filename}',
                               corpus=repo, revision=rev, title='',
                               metadata={'row': ordinal, 'passage': passage, 'split': 'train',
                                         'column': 'passages.passage_text'}, max_windows=2)
                    if result.total >= 40000:
                        break
                ordinal += 1
                if result.total >= 40000:
                    break
            print('web raw rows inspected', ordinal, 'new windows', result.total, flush=True)
            if result.total >= 40000:
                break
    return result.finish()


def main():
    parser = argparse.ArgumentParser()
    root = Path("IB/TMP/FIRST_TASK_AWARE_FINETUNING")
    parser.add_argument("--base", type=Path, default=root / "bank_sources_200k.json")
    parser.add_argument("--output", type=Path, default=root / "bank_source_extensions")
    parser.add_argument("--tasks", nargs="+", default=["financial", "technical_web", "math_textbook", "math_web", "documents", "code", "office"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "PRESERVE").write_text("Immutable reusable raw source extensions; independent of synthesis cache identity.\n")
    index_path = args.output / "base_source_index.json"
    global USED, OLD_DOCS
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        source_ids, doc_ids = set(), collections.defaultdict(set)
        with args.base.open("rb") as file:
            for prefix, event, value in ijson.parse(file):
                if event == "string" and prefix.startswith("pools."):
                    parts = prefix.split(".")
                    if len(parts) == 4 and parts[2] == "item":
                        if parts[3] == "source_id":
                            source_ids.add(value)
                        elif parts[3] == "doc_id":
                            doc_ids[parts[1]].add(value)
        index = {"source_ids": list(source_ids), "doc_ids": {family: list(ids) for family, ids in doc_ids.items()}}
        index_path.write_text(json.dumps(index))
        del source_ids, doc_ids
    USED = set(index["source_ids"])
    OLD_DOCS = {family: set(ids) for family, ids in index["doc_ids"].items()}
    del index
    for published in args.output.glob('*.jsonl'):
        family = published.name.split('__')[0]
        for line in published.open():
            row = json.loads(line)
            USED.add(row['source_id'])
            OLD_DOCS.setdefault(family, set()).add(row['doc_id'])
    handlers = {"financial": financial, "technical_web": technical_web, "math_textbook": quantitative_exposition,
                "math_web": open_web_exposition,
                "documents": manuals, "code": code, "office": office, "web": web}
    errors = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        jobs = {executor.submit(handlers[name], args.output): name for name in args.tasks}
        for future in concurrent.futures.as_completed(jobs):
            name = jobs[future]
            try:
                future.result()
            except Exception as error:
                errors[name] = repr(error)
                print("FAILED", name, repr(error), flush=True)
    result = {"pinned_sources": PINNED_EXT, "errors": errors,
              "pools": [json.loads(path.read_text()) for path in sorted(args.output.glob("*.manifest.json"))],
              "note": "Append-only source data. Quantitative textbook extension contains physics/engineering/chemistry exposition, explicitly tagged; raw financial corpus preserves tables. No native QA/reasoning fields were read."}
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
