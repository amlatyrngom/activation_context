"""Mixed training corpus for the AC reconstruction round: material_mix.json, the schema of material.json with seven source labels.

Sources (label: origin, target training tokens):
  swe, research : tool outputs of Open-SWE-Traces / S1-DeepResearch trajectories (as run_recon.prepare, windows up to 6,000 chars), 10.5M + 4.5M
  teacher       : tool-result messages of our own rollouts (teacher_campaign_12h/rollouts.jsonl, streamed), 7M; assistant text only when short
  code          : source files of repositories fetched by the LCA bug-localization loader (py and java), 7M
  docs          : Loong papers and 10-K reports (5M) and BRIGHT StackExchange posts (7 domains, 4M)
  math          : Nemotron-SFT-Math-v4 `cot` solutions (reasoning plus content), 5M
  web           : MS MARCO passages, whole or packed per query up to 2,000 chars, 5M
Every passage is a 400-6,000 character window cut at line ends (min 200), deduplicated online with the 50-char shingle rule of
dedup_material.py (a passage with > 50 % of its text in earlier kept passages is dropped; the shingle set is a Bloom filter so the
whole pool fits in memory). Held panels: 100 passages per label from documents (repositories for code, rollouts, trajectories,
queries) that no training passage uses, with <= 20 % shingle overlap with the training pool; wiki is copied from material.json.

Re-runnable: each source's kept passages land in <scratch>/<label>.train.jsonl (+ .held.jsonl, .meta.json) and are reused on the
next run, so a crash in one source does not redo the others. Delete a label's files to recollect it.
CPU only, ~1 h. Run: PYTHONPATH=. .venv/bin/python IB/ARTIFACTS/AGENT_AC_PRETRAINING/ac_recon/prepare_mix.py
"""
from __future__ import annotations
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time

import numpy as np

ROOT = Path(os.environ.get('AC_RECON_SOURCE_ROOT') or Path(__file__).resolve().parents[4])
sys.path.insert(0, str(ROOT))
sys.modules.setdefault('fla', None)                      # the harness imports transformers; fla (GPU kernels) is absent on CPU
os.environ.setdefault('ACTIVATION_SYNC_ROOT', str(ROOT / 'IB/TMP/SYNC'))   # loader caches (REPO_CACHE, LOONG) under IB/TMP/SYNC
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')

OUT = Path(os.environ.get('AC_RECON_ROOT', ROOT / 'IB/TMP/SYNC/AC_RECON')) / 'preparation'
SCRATCH = Path(os.environ.get('PREPARE_MIX_SCRATCH', '/tmp/claude-1000/-workspace/e10e46ef-5737-4008-8ce3-e531296b2a16/scratchpad/corpus'))
ROLLOUTS = ROOT / 'IB/TMP/AGENT_AC_PRETRAINING/node_camp/ROLLOUTS/teacher_campaign_12h/rollouts.jsonl'
MODEL = 'Qwen/Qwen3.5-4B'
SEED = 20260917
QUOTAS = {'swe': 10_500_000, 'research': 4_500_000, 'teacher': 7_000_000, 'code': 7_000_000, 'docs': 9_000_000, 'math': 5_000_000, 'web': 5_000_000}
LOONG_SHARE = 5_000_000                                  # of the docs quota; BRIGHT fills the rest
BRIGHT_DOMAINS = ('biology', 'earth_science', 'economics', 'psychology', 'robotics', 'stackoverflow', 'sustainable_living')
OVERSHOOT = 1.12                                         # collected beyond the quota, so shortfalls elsewhere can be filled
HELD_PER_SOURCE, HELD_CANDIDATES, HELD_PER_DOC = 100, 1500, 8
LO, HI, MIN_CHARS = 400, 6000, 200
K, STRIDE = 50, 8                                        # the shingle rule of dedup_material.py
CODE_REPOS = {'py': 60, 'java': 45}
CODE_TOKENS_PER_REPO = 110_000
MATH_DOCS, MATH_WINDOWS_PER_DOC = 1600, 4
LOONG_WINDOWS_PER_DOC = 6
WEB_QUERIES, WEB_PACK_CHARS = 9000, 2000
TEACHER_ASSISTANT_CHARS = 60_000_000                     # assistant text kept aside during the pass, used only when tool results fall short
if os.environ.get('PREPARE_MIX_SMOKE'):                  # a quick end-to-end check with tiny targets (scratch output, see main)
    QUOTAS = {label: 30_000 for label in QUOTAS}; LOONG_SHARE = 15_000; HELD_CANDIDATES = 60
    CODE_REPOS = {'py': 3, 'java': 2}; CODE_TOKENS_PER_REPO = 8_000; MATH_DOCS = 30; WEB_QUERIES = 150
    SCRATCH = SCRATCH / 'smoke'; OUT = SCRATCH / 'out'


# ------------------------------------------------------------------------------------------------------- shingle dedup
class ShingleFilter:
    """The 50-char shingle rule over a Bloom filter (2^31 bits, 3 hashes, ~256 MB): membership at every character position,
    insertion every STRIDE positions, like dedup_material.py's set of substrings. False positives ~1e-4 per lookup."""
    BITS = 1 << 31
    B = np.uint64(1099511628211)
    MIX = np.uint64(0x9E3779B97F4A7C15)

    def __init__(self):
        self.bits = np.zeros(self.BITS // 8, dtype=np.uint8)
        self.inserted = 0

    @classmethod
    def hashes(cls, text):
        codes = np.frombuffer(text.encode('utf-32-le'), dtype=np.uint32).astype(np.uint64)
        n = len(codes) - K + 1
        if n <= 0: return np.zeros(0, dtype=np.uint64)
        h = np.zeros(n, dtype=np.uint64)
        for j in range(K):
            h = h * cls.B + codes[j:j + n]
        return h

    def _indices(self, h):
        mask = np.uint64(self.BITS - 1)
        h2 = h * self.MIX
        return (h & mask, (h >> np.uint64(31)) & mask, (h2 >> np.uint64(29)) & mask)

    def _member(self, h):
        hit = np.ones(len(h), dtype=bool)
        for idx in self._indices(h):
            hit &= ((self.bits[idx >> np.uint64(3)] >> (idx & np.uint64(7)).astype(np.uint8)) & np.uint8(1)).astype(bool)
        return hit

    def coverage(self, text, h=None):
        """Fraction of the text's characters inside a 50-char shingle already in the filter."""
        h = self.hashes(text) if h is None else h
        if len(h) == 0 or self.inserted == 0: return 0.0
        hit = self._member(h)
        if not hit.any(): return 0.0
        positions = np.arange(len(h))
        last = np.maximum.accumulate(np.where(hit, positions, -K - 1))
        covered = int(((positions - last) < K).sum()) + max(0, min(K - 1, int(last[-1]) + K - len(h)))
        return covered / max(1, len(text))

    def insert(self, text, h=None):
        h = self.hashes(text) if h is None else h
        h = h[::STRIDE]
        if len(h) == 0: return
        for idx in self._indices(h):
            np.bitwise_or.at(self.bits, idx >> np.uint64(3), np.uint8(1) << (idx & np.uint64(7)).astype(np.uint8))
        self.inserted += len(h)

    def consider(self, text, threshold=0.5):
        """Keep (and insert) unless more than `threshold` of the text is already covered; returns (kept, coverage)."""
        h = self.hashes(text)
        cov = self.coverage(text, h)
        if cov > threshold: return False, cov
        self.insert(text, h)
        return True, cov


# ------------------------------------------------------------------------------------------------------- windows
def windows(text, rng, lo=LO, hi=HI):
    """run_recon.windows with hi=6000: whole when short, else consecutive slices of a random length in [lo, hi] cut at line ends."""
    text = text.strip()
    if len(text) < MIN_CHARS: return []
    if len(text) <= hi: return [text]
    out, cursor = [], 0
    while cursor < len(text):
        size = rng.randint(lo, hi)
        end = min(len(text), cursor + size)
        cut = text.rfind('\n', cursor + lo // 2, end) if end < len(text) else end
        if cut <= cursor: cut = end
        piece = text[cursor:cut].strip()
        if len(piece) >= MIN_CHARS: out.append(piece)
        cursor = cut
    return out


def held_key(key, every):
    return int(hashlib.sha1(str(key).encode()).hexdigest()[:8], 16) % every == 0


# ------------------------------------------------------------------------------------------------------- collector
class Collector:
    """Windows, tokenizes and deduplicates the documents of one label until the token target is reached; documents whose split key
    hashes into the held bucket (one in `held_every`) feed the held candidates instead and never the training pool."""

    def __init__(self, label, target_tokens, tokenizer, shingles, held_every, seed):
        self.label, self.target, self.tokenizer, self.shingles, self.held_every = label, target_tokens, tokenizer, shingles, held_every
        self.rng = random.Random(seed)
        self.train, self.held = [], []
        self.tokens = 0
        self.stats = collections.Counter()
        self.train_docs, self.held_docs = set(), set()
        self.exact = set()
        self.started = time.time()
        self.last_report = 0

    def done(self):
        return self.tokens >= self.target

    def count_tokens(self, texts):
        return [len(ids) for ids in self.tokenizer(texts, add_special_tokens=False, return_attention_mask=False)['input_ids']]

    def add_document(self, split_key, doc_id, texts, sub='', max_windows=None):
        """texts: the document's raw texts (tool outputs, a file, a solution); returns the training tokens added."""
        self.stats['docs_seen'] += 1
        if sub: self.stats[f'docs_seen:{sub}'] += 1
        held = held_key(split_key, self.held_every)
        if held and len(self.held) >= HELD_CANDIDATES: return 0
        pieces = [piece for text in texts if isinstance(text, str) for piece in windows(text, self.rng)]
        if not pieces: return 0
        if max_windows and len(pieces) > max_windows:
            pieces = self.rng.sample(pieces, max_windows)
        if held:
            counts = self.count_tokens(pieces)
            for piece, n in zip(pieces, counts):
                self.held.append({'text': piece, 'doc_id': doc_id, 'source': self.label, 'tokens': n})
            self.held_docs.add(split_key)
            self.stats['held_candidates'] += len(pieces)
            return 0
        self.stats['raw_passages'] += len(pieces)
        fresh = []
        for piece in pieces:
            key = hashlib.sha1(piece.encode()).hexdigest()
            if key in self.exact:
                self.stats['exact_duplicates'] += 1; continue
            self.exact.add(key); fresh.append(piece)
        kept = []
        for piece in fresh:
            ok, _ = self.shingles.consider(piece)
            if ok: kept.append(piece)
            else: self.stats['shingle_dropped'] += 1
        if not kept: return 0
        counts = self.count_tokens(kept)
        added = 0
        for piece, n in zip(kept, counts):
            self.train.append({'text': piece, 'doc_id': doc_id, 'source': self.label, 'tokens': n})
            added += n
        self.tokens += added
        self.train_docs.add(split_key)
        self.stats['train_tokens'] = self.tokens
        if sub: self.stats[f'train_tokens:{sub}'] += added; self.stats[f'docs_used:{sub}'] += 1
        if time.time() - self.last_report > 30:
            self.last_report = time.time()
            log(f'{self.label}: {self.tokens:,} / {self.target:,} tokens, {len(self.train):,} passages, {len(self.train_docs):,} docs, '
                f'{len(self.held):,} held candidates, {self.stats["shingle_dropped"]:,} shingle-dropped, {time.time() - self.started:.0f}s')
        return added

    def meta(self, **extra):
        m = dict(self.stats)
        m.update(label=self.label, target_tokens=self.target, train_passages=len(self.train), train_tokens=self.tokens,
                 train_docs=len(self.train_docs), held_docs=len(self.held_docs), held_candidates=len(self.held),
                 seconds=time.time() - self.started, **extra)
        return m


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


# ------------------------------------------------------------------------------------------------------- sources
def swe_documents(collector):
    from datasets import load_dataset
    from activation.dataset.loaders.open_swe_traces import normalize_open_swe_row
    for row in load_dataset('nvidia/Open-SWE-Traces', 'v1.2', split='minisweagent', streaming=True):
        normalized, _ = normalize_open_swe_row(row)          # the loader's row normalization, streamed row by row (memory)
        doc_id = str(row.get('trajectory_id') or row['instance_id'])
        collector.add_document(doc_id, doc_id, [m['content'] for m in normalized if m['role'] == 'tool' and m.get('content')])
        if collector.done(): break


def research_documents(collector):
    from datasets import load_dataset
    from activation.dataset.loaders.s1_deep_research import normalize_s1_row
    for index, row in enumerate(load_dataset('ScienceOne-AI/S1-DeepResearch-15k', split='train', streaming=True)):
        meta = row.get('meta') or {}
        if meta.get('language') != 'en':
            collector.stats['rows_other_language'] += 1; continue
        normalized, _ = normalize_s1_row(row)
        doc_id = str(meta.get('id') or index)
        collector.add_document(doc_id, doc_id, [m['content'] for m in normalized if m['role'] == 'tool' and m.get('content')])
        if collector.done(): break


def message_text(content):
    if isinstance(content, list):
        content = ''.join(piece.get('text', '') for piece in content if isinstance(piece, dict))
    return content if isinstance(content, str) else ''


def teacher_documents(collector):
    """Tool-result messages of every rollout (main trajectory and subagent trajectories); assistant texts are set aside and only
    added when the tool results fall short of the target."""
    assistant, assistant_chars = [], 0
    families = collections.Counter(); tools = collections.Counter()
    with open(ROLLOUTS, 'rb') as handle:
        for index, line in enumerate(handle):
            if not line.strip(): continue
            record = json.loads(line)
            doc_id = f"rollout:{index}:{record.get('config_key', '')}"
            families[str(record.get('config_key', '')).split('___')[0]] += 1
            outputs, texts = [], []

            def walk(rec):
                for message in rec.get('trajectory') or []:
                    role = message.get('role')
                    if role == 'tool':
                        outputs.append(message_text(message.get('content')))
                        for call in message.get('tool_calls') or []: tools[call.get('name')] += 1
                    elif role == 'assistant':
                        texts.append(message_text(message.get('content')).replace('</think>', '').strip())
                for sub in rec.get('subagent_results') or []:
                    if isinstance(sub, dict): walk(sub)
            walk(record)
            collector.add_document(doc_id, doc_id, outputs)
            if assistant_chars < TEACHER_ASSISTANT_CHARS:
                joined = '\n\n'.join(t for t in texts if t)
                assistant.append((doc_id, joined)); assistant_chars += len(joined)
            if collector.done(): break
    collector.stats['rollouts_read'] = index + 1
    if not collector.done():
        log(f'teacher: tool results reached {collector.tokens:,} tokens; adding assistant texts')
        for doc_id, joined in assistant:
            collector.add_document(doc_id, doc_id, [joined], sub='assistant_text')
            if collector.done(): break
    return {'task_families': dict(families.most_common()), 'tools': dict(tools.most_common(12))}


def code_file_ok(text):
    """Skip minified / generated / data files: any line over 500 chars, or over 30 % of the non-blank characters not word characters."""
    if any(len(line) > 500 for line in text.split('\n')): return False
    nonblank = len(text) - len(re.findall(r'\s', text))
    if nonblank == 0: return False
    return len(re.findall(r'\w', text)) / nonblank >= 0.70


def code_documents(collector, harness):
    from activation.dataset.loaders.lca_bug_localization import LcaBugLocalizationDataset, iter_text_files
    repos_used, bytes_used, files_skipped = {}, 0, 0
    for config, count in CODE_REPOS.items():
        loaded = LcaBugLocalizationDataset.load(harness, max_examples=count, config=config, max_tasks_per_repo=1,
                                                index_files_per_repo=0, fetch_workers=8)       # checkouts only; files are streamed below
        log(f'code: {config} loader returned {len(loaded.scorable_tasks)} repositories in {loaded.stats.initial_load_latency:.0f}s')
        for task in loaded.scorable_tasks.values():
            datum = task.task_datum
            repo, sha, local = datum['repo'], str(datum['base_sha'])[:8], datum['checkout_path']
            if repo in repos_used: continue
            start_tokens, files = collector.tokens, 0
            for relative, text in iter_text_files(local, 100_000):
                if not code_file_ok(text):
                    files_skipped += 1; continue
                files += 1
                collector.add_document(repo, f'{repo}@{sha}/{relative}', [f'{relative}\n{text}'], sub=config)
                if collector.tokens - start_tokens >= CODE_TOKENS_PER_REPO or collector.done(): break
            repos_used[repo] = {'config': config, 'files': files, 'tokens': collector.tokens - start_tokens, 'held': held_key(repo, collector.held_every)}
            bytes_used += sum(p.stat().st_size for p in Path(local).rglob('*') if p.is_file())
            if collector.done(): break
        if collector.done(): break
    return {'repositories': repos_used, 'repository_bytes_on_disk': bytes_used, 'files_skipped_by_filter': files_skipped,
            'held_repositories': sorted(r for r, v in repos_used.items() if v['held'])}


def docs_documents(collector, harness):
    from activation.dataset.loaders.loong import LoongDataset
    from activation.dataset.loaders.bright import BrightDataset
    info = {}
    loaded = LoongDataset.load(harness, max_examples=None)
    docs = sorted(loaded.documents.items()); collector.rng.shuffle(docs)
    info['loong_documents_available'] = len(docs); info['loong_chars_available'] = sum(len(d.text or '') for _, d in docs)
    loong_target = min(collector.target, int(LOONG_SHARE * OVERSHOOT))
    for doc_id, document in docs:
        kind = 'loong_paper' if doc_id.endswith('.md') else 'loong_financial'
        collector.add_document(doc_id, doc_id, [document.text or ''], sub=kind, max_windows=LOONG_WINDOWS_PER_DOC)
        if collector.tokens >= loong_target: break
    del loaded, docs
    info['loong_tokens'] = collector.tokens
    remaining = collector.target - collector.tokens
    per_domain = max(1, remaining // len(BRIGHT_DOMAINS))
    for domain in BRIGHT_DOMAINS:
        loaded = BrightDataset.load(harness, max_examples=1, domain=domain, max_corpus_documents=None)
        harness.dataset_manager.loaded_datasets.pop(loaded.dataset_id, None)      # the loader registers it; do not keep the corpus alive
        posts = collections.defaultdict(list)
        for doc_id, document in loaded.documents.items():
            stem = doc_id[:-4] if doc_id.endswith('.txt') else doc_id
            post, _, chunk = stem.rpartition('_')
            posts[post or stem].append((int(chunk) if chunk.isdigit() else 0, document.text or ''))
        del loaded
        keys = sorted(posts); collector.rng.shuffle(keys)
        info[f'bright_{domain}_posts_available'] = len(keys)
        start = collector.tokens
        for post in keys:
            chunks = [text for _, text in sorted(posts[post])]
            packed, current = [], ''
            for text in chunks:
                if current and len(current) + len(text) + 1 > HI:
                    packed.append(current); current = text
                else:
                    current = f'{current}\n{text}' if current else text
            if current: packed.append(current)
            doc_id = f'bright/{domain}/{post}'
            collector.add_document(doc_id, doc_id, packed, sub=f'bright_{domain}')
            if collector.tokens - start >= per_domain or collector.done(): break
        del posts
        if collector.done(): break
    return info


def math_documents(collector, harness):
    from activation.dataset.loaders.nemotron_math import NemotronMathDataset
    loaded = NemotronMathDataset.load(harness, max_examples=MATH_DOCS, subset='cot')
    harness.dataset_manager.loaded_datasets.pop(loaded.dataset_id, None)
    for doc_id, document in loaded.documents.items():
        parts = []
        for message in document.trajectory or []:
            if message.get('role') == 'assistant':
                parts.extend(p for p in (message.get('reasoning'), message.get('content')) if p)
        collector.add_document(doc_id, doc_id, ['\n\n'.join(parts)], max_windows=MATH_WINDOWS_PER_DOC)
        if collector.done(): break
    return {'math_documents_loaded': len(loaded.documents)}


def web_documents(collector, harness):
    from activation.dataset.loaders.msmarco import MsMarcoDataset
    loaded = MsMarcoDataset.load(harness, max_examples=WEB_QUERIES, max_corpus_documents=1)
    harness.dataset_manager.loaded_datasets.pop(loaded.dataset_id, None)
    queries = collections.defaultdict(list)
    for doc_id, document in loaded.documents.items():
        prefix, _, index = doc_id.rpartition(':')
        queries[prefix].append((int(index), document.text or ''))
    info = {'web_queries_loaded': len(queries), 'web_passages_loaded': len(loaded.documents)}
    del loaded
    for prefix in sorted(queries, key=lambda p: int(p.rpartition(':')[2])):
        packed, current = [], ''
        for _, text in sorted(queries[prefix]):
            text = text.strip()
            if current and len(current) + len(text) + 2 > WEB_PACK_CHARS:
                packed.append(current); current = text
            else:
                current = f'{current}\n\n{text}' if current else text
        if current: packed.append(current)
        collector.add_document(prefix, prefix, packed)
        if collector.done(): break
    return info


# ------------------------------------------------------------------------------------------------------- caching
def cache_paths(label):
    return SCRATCH / f'{label}.train.jsonl', SCRATCH / f'{label}.held.jsonl', SCRATCH / f'{label}.meta.json'


def write_jsonl(path, rows):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as handle:
        for row in rows: handle.write(json.dumps(row) + '\n')
    tmp.replace(path)


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect(label, target, tokenizer, shingles, harness, held_every, seed, producer):
    """The label's (train passages, held candidates, meta): from the scratch cache when present, else collected and cached."""
    train_path, held_path, meta_path = cache_paths(label)
    if meta_path.exists():
        train, held, meta = read_jsonl(train_path), read_jsonl(held_path), json.loads(meta_path.read_text())
        started = time.time()
        for passage in train: shingles.insert(passage['text'])
        log(f'{label}: reused {len(train):,} cached passages ({meta["train_tokens"]:,} tokens), shingles rebuilt in {time.time() - started:.0f}s')
        return train, held, dict(meta, cached=True)
    log(f'{label}: collecting up to {target:,} tokens')
    collector = Collector(label, target, tokenizer, shingles, held_every, seed)
    try:
        extra = producer(collector) or {}
    except Exception as error:                            # keep what was collected; the report says it fell short
        import traceback; traceback.print_exc()
        extra = {'error': f'{type(error).__name__}: {error}'}
    meta = collector.meta(**extra)
    write_jsonl(train_path, collector.train); write_jsonl(held_path, collector.held)
    meta_path.write_text(json.dumps(meta, indent=1, default=str))
    log(f'{label}: done, {collector.tokens:,} tokens in {len(collector.train):,} passages from {len(collector.train_docs):,} documents, '
        f'{len(collector.held):,} held candidates, {meta["seconds"]:.0f}s')
    return collector.train, collector.held, meta


# ------------------------------------------------------------------------------------------------------- assembly
def build_harness():
    import torch
    from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
    torch.set_num_threads(4); torch.manual_seed(SEED)
    return HarnessRuntime(HarnessRuntimeConfig(model_configs={'qwen4b': ModelConfig('qwen4b', MODEL, engine_kwargs={'max_model_len': 4096, 'gpu_memory_utilization': .75})},
                                               doc_chunk_size_chars=2048, doc_chunk_overlap_chars=0))


def take_quota(passages, quota):
    """Passages in kept order until the quota is reached; (taken, leftover)."""
    total = 0
    for index, passage in enumerate(passages):
        if total >= quota: return passages[:index], passages[index:]
        total += passage['tokens']
    return passages, []


def token_stats(passages):
    counts = [p['tokens'] for p in passages] or [0]
    return {'passages': len(passages), 'tokens': sum(counts), 'mean': statistics.mean(counts), 'median': statistics.median(counts), 'max': max(counts)}


def folder_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file()) if Path(path).exists() else 0


def main():
    started = time.time()
    SCRATCH.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    hub_cache = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub'
    repo_cache = Path(os.environ['ACTIVATION_SYNC_ROOT']) / 'REPO_CACHE'
    before = {'hub': folder_bytes(hub_cache), 'repo_cache': folder_bytes(repo_cache), 'loong': folder_bytes(Path(os.environ['ACTIVATION_SYNC_ROOT']) / 'LOONG')}
    log('building the harness and the tokenizer')
    harness = build_harness()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    shingles = ShingleFilter()
    producers = {
        'swe': (15, lambda c: swe_documents(c)),
        'research': (15, lambda c: research_documents(c)),
        'teacher': (20, lambda c: teacher_documents(c)),
        'code': (12, lambda c: code_documents(c, harness)),
        'docs': (20, lambda c: docs_documents(c, harness)),
        'math': (20, lambda c: math_documents(c, harness)),
        'web': (20, lambda c: web_documents(c, harness)),
    }
    pools, held_candidates, metas = {}, {}, {}
    for index, (label, (held_every, producer)) in enumerate(producers.items()):
        pools[label], held_candidates[label], metas[label] = collect(label, int(QUOTAS[label] * OVERSHOOT), tokenizer, shingles, harness, held_every, SEED + index, producer)
    after = {'hub': folder_bytes(hub_cache), 'repo_cache': folder_bytes(repo_cache), 'loong': folder_bytes(Path(os.environ['ACTIVATION_SYNC_ROOT']) / 'LOONG')}

    # quotas, then the shortfall filled from the other labels' leftovers in proportion to their quotas
    log('assembling the training pool')
    taken, leftover, shortfall = {}, {}, {}
    for label in QUOTAS:
        taken[label], leftover[label] = take_quota(pools[label], QUOTAS[label])
        shortfall[label] = max(0, QUOTAS[label] - sum(p['tokens'] for p in taken[label]))
    gap = sum(shortfall.values()); filled = collections.Counter()
    for _ in range(4):
        donors = [l for l in QUOTAS if leftover[l]]
        if gap <= 0 or not donors: break
        weight = sum(QUOTAS[l] for l in donors)
        for label in donors:
            share = int(gap * QUOTAS[label] / weight)
            extra, leftover[label] = take_quota(leftover[label], share)
            taken[label].extend(extra); got = sum(p['tokens'] for p in extra); filled[label] += got
        gap = sum(QUOTAS.values()) - sum(p['tokens'] for l in QUOTAS for p in taken[l])
    train = [p for label in QUOTAS for p in taken[label]]
    random.Random(SEED).shuffle(train)

    # held panels: documents outside the training pool, <= 20 % overlap with it, deduplicated among themselves
    log('building the held panels')
    train_docs = {p['doc_id'] for p in train}
    held, held_report = [], {}
    held_filter = ShingleFilter()
    rng = random.Random(SEED + 100)
    for label in QUOTAS:
        candidates = [c for c in held_candidates[label] if c['doc_id'] not in train_docs]
        rng.shuffle(candidates)
        chosen, per_doc, rejected_overlap, rejected_dup, deferred = [], collections.Counter(), 0, 0, []
        for cap in (HELD_PER_DOC, None):                  # a second pass without the per-document cap when the panel is short
            for candidate in (candidates if cap else deferred):
                if len(chosen) >= HELD_PER_SOURCE: break
                if cap and per_doc[candidate['doc_id']] >= cap:
                    deferred.append(candidate); continue
                if shingles.coverage(candidate['text']) > 0.2:
                    rejected_overlap += 1; continue
                ok, _ = held_filter.consider(candidate['text'])
                if not ok:
                    rejected_dup += 1; continue
                chosen.append(candidate); per_doc[candidate['doc_id']] += 1
        held.extend(chosen)
        held_report[label] = {'candidates': len(candidates), 'chosen': len(chosen), 'documents': len(per_doc), 'rejected_overlap_over_20pct': rejected_overlap, 'rejected_duplicate': rejected_dup,
                              'max_per_document': max(per_doc.values(), default=0)}
    wiki = json.loads((OUT / 'material.json').read_text())['wiki']

    by_source = {label: sum(p['source'] == label for p in train) for label in QUOTAS}
    tokens_by_source = {label: sum(p['tokens'] for p in train if p['source'] == label) for label in QUOTAS}
    stats = {label: token_stats([p for p in train if p['source'] == label]) for label in QUOTAS}
    held_stats = {label: token_stats([p for p in held if p['source'] == label]) for label in QUOTAS}
    seconds = (time.time() - started) + sum(m.get('seconds', 0) for m in metas.values() if m.get('cached'))   # this run plus the cached sources' collection
    material = {'train': train, 'held': held, 'wiki': wiki, 'preparation_seconds': seconds,
                'train_by_source': by_source, 'train_tokens_by_source': tokens_by_source,
                'token_stats': {'train_mean': statistics.mean(p['tokens'] for p in train), 'held_mean': statistics.mean(p['tokens'] for p in held),
                                'train_mean_by_source': {l: s['mean'] for l, s in stats.items()}, 'held_mean_by_source': {l: s['mean'] for l, s in held_stats.items()},
                                'train_tokens': sum(tokens_by_source.values())}}
    target = OUT / 'material_mix.json'; tmp = target.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(material)); tmp.replace(target)
    log(f'wrote {target} ({target.stat().st_size / 1e6:.0f} MB): {len(train):,} training passages, {sum(tokens_by_source.values()):,} tokens, {len(held)} held, {len(wiki)} wiki')

    # report
    lines = ['# material_mix.json — preparation report', '',
             f'Written {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())} by `prepare_mix.py`; this run took {time.time() - started:.0f} s '
             f'(collection time of every source, cached or not: {sum(m.get("seconds", 0) for m in metas.values()):.0f} s).', '',
             f'Training: {len(train):,} passages, {sum(tokens_by_source.values()):,} tokens (target {sum(QUOTAS.values()):,}); held: {len(held)} passages; wiki: {len(wiki)} (copied from material.json).',
             'Windows 400-6,000 characters cut at line ends (min 200); tokens counted with the Qwen3.5-4B tokenizer without special tokens; '
             'dedup: a passage with > 50 % of its characters inside 50-char shingles of earlier kept passages is dropped (Bloom filter, stride 8), whole pool, sources in the order below.', '',
             '## Training pool by source', '',
             '| source | quota | tokens | passages | documents | mean | median | max | raw passages | exact dups | shingle-dropped | held-out docs | gap fill | shortfall |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for label in QUOTAS:
        m, s = metas[label], stats[label]
        docs_used = len({p['doc_id'] for p in train if p['source'] == label})
        if label == 'code': docs_used = f"{docs_used} files / {len({p['doc_id'].split('@')[0] for p in train if p['source'] == label})} repos"
        lines.append(f"| {label} | {QUOTAS[label]:,} | {s['tokens']:,} | {s['passages']:,} | {docs_used} | {s['mean']:.0f} | {s['median']:.0f} | {s['max']} | "
                     f"{m.get('raw_passages', 0):,} | {m.get('exact_duplicates', 0):,} | {m.get('shingle_dropped', 0):,} | {m.get('held_docs', 0)} | {filled[label]:,} | {shortfall[label]:,} |")
    lines += ['', 'documents = distinct doc_ids in the final training pool (trajectories, rollouts, files, Loong documents / BRIGHT posts, solutions, MS MARCO queries). '
              'raw / exact dups / shingle-dropped count what the collection saw before trimming to the quota; gap fill = tokens taken beyond the quota to cover another source\'s shortfall.', '',
              '## Held panels', '', '| source | candidates | chosen | documents | max per document | rejected (> 20 % overlap) | rejected (duplicate) | mean tokens |', '|---|---|---|---|---|---|---|---|']
    for label in QUOTAS:
        h = held_report[label]
        lines.append(f"| {label} | {h['candidates']:,} | {h['chosen']} | {h['documents']} | {h['max_per_document']} | {h['rejected_overlap_over_20pct']} | {h['rejected_duplicate']} | {held_stats[label]['mean']:.0f} |")
    lines += ['', 'Held documents are chosen by a hash of the split key (one in 15 swe / research trajectories, one in 20 rollouts / documents / solutions / queries, one in 12 repositories) before any training passage is taken, '
              'so no held passage shares a document (or repository) with the training pool; at most 8 passages per document, more only when a panel would otherwise be short of 100.', '',
              '## Sources', '']
    m = metas['swe']; lines.append(f"- **swe**: Open-SWE-Traces v1.2 `minisweagent`, streamed in dataset order, {m.get('docs_seen', 0):,} trajectories read, {m.get('train_docs', 0):,} in training; tool outputs only. {m.get('seconds', 0):.0f} s.")
    m = metas['research']; lines.append(f"- **research**: S1-DeepResearch-15k, English rows ({m.get('rows_other_language', 0):,} other-language rows skipped), {m.get('docs_seen', 0):,} trajectories read, {m.get('train_docs', 0):,} in training; tool outputs only. {m.get('seconds', 0):.0f} s.")
    m = metas['teacher']
    lines.append(f"- **teacher**: `{ROLLOUTS.relative_to(ROOT)}`, {m.get('rollouts_read', 0):,} rollouts streamed; tool-result messages of the main and subagent trajectories"
                 f"{' plus assistant texts (' + str(m.get('train_tokens:assistant_text', 0)) + ' tokens) because tool results fell short' if m.get('train_tokens:assistant_text') else ''}. "
                 f"Task families: {m.get('task_families', {})}. Tools answered: {m.get('tools', {})}. Note: the semantic_search results are Wikipedia passages, likely familiar to the model. {m.get('seconds', 0):.0f} s.")
    m = metas['code']; repos = m.get('repositories', {})
    lines.append(f"- **code**: LCA bug localization checkouts, {sum(1 for r in repos.values() if r['config'] == 'py')} py + {sum(1 for r in repos.values() if r['config'] == 'java')} java repositories "
                 f"({len(m.get('held_repositories', []))} held out), one task per repository, at most {CODE_TOKENS_PER_REPO:,} tokens per repository, {m.get('files_skipped_by_filter', 0):,} files skipped by the minified/data filter, "
                 f"{m.get('repository_bytes_on_disk', 0) / 1e9:.2f} GB of checkouts on disk. {m.get('seconds', 0):.0f} s.")
    m = metas['docs']
    bright = {k.replace('docs_used:bright_', ''): v for k, v in m.items() if k.startswith('docs_used:bright_')}
    lines.append(f"- **docs**: Loong {m.get('docs_used:loong_paper', 0)} arXiv papers + {m.get('docs_used:loong_financial', 0)} financial reports (of {m.get('loong_documents_available', 0)} English documents, "
                 f"at most {LOONG_WINDOWS_PER_DOC} random windows each, {m.get('loong_tokens', 0):,} tokens); BRIGHT posts per domain {bright} (consecutive chunks of a post joined up to 6,000 chars). {m.get('seconds', 0):.0f} s.")
    m = metas['math']; lines.append(f"- **math**: Nemotron-SFT-Math-v4 `cot`, {m.get('math_documents_loaded', 0):,} solutions loaded (reasoning + content of the assistant turn), at most {MATH_WINDOWS_PER_DOC} random windows per solution, {m.get('train_docs', 0):,} in training. {m.get('seconds', 0):.0f} s.")
    m = metas['web']; lines.append(f"- **web**: MS MARCO v1.1 train, {m.get('web_queries_loaded', 0):,} queries ({m.get('web_passages_loaded', 0):,} passages) loaded, consecutive passages of a query packed up to {WEB_PACK_CHARS:,} chars, {m.get('train_docs', 0):,} queries in training. {m.get('seconds', 0):.0f} s.")
    lines += ['', '## Downloads', '',
              f"- Hugging Face hub cache: {before['hub'] / 1e9:.2f} -> {after['hub'] / 1e9:.2f} GB (streamed datasets leave little there).",
              f"- Repository checkouts (IB/TMP/SYNC/REPO_CACHE): {before['repo_cache'] / 1e9:.2f} -> {after['repo_cache'] / 1e9:.2f} GB.",
              f"- Loong documents (IB/TMP/SYNC/LOONG): {before['loong'] / 1e9:.2f} -> {after['loong'] / 1e9:.2f} GB.", '',
              '## Shortfalls', '']
    short = [l for l in QUOTAS if shortfall[l] > 0]
    errors = {l: metas[l]['error'] for l in QUOTAS if metas[l].get('error')}
    if not short and not errors: lines.append('Every source reached its quota.')
    for label in short:
        lines.append(f"- {label}: {shortfall[label]:,} tokens short of {QUOTAS[label]:,}" + (f" (collection error: {errors[label]})" if label in errors else ' (the source ran out of material)') + '.')
    if gap > 0: lines.append(f'- {gap:,} tokens could not be filled from the other sources.')
    elif short: lines.append('- The shortfall was filled from the other sources in proportion to their quotas (gap fill column above).')
    for label, error in errors.items():
        if label not in short: lines.append(f'- {label}: collection stopped with an error but the quota was met: {error}')
    (OUT / 'material_mix_report.md').write_text('\n'.join(lines) + '\n')
    log(f'wrote {OUT / "material_mix_report.md"}; total {time.time() - started:.0f}s')


if __name__ == '__main__':
    main()
