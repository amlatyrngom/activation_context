"""Natural long parents and coherent office-reference dossiers, without QA.

Each exact support span belongs to the longer evidence text. Parent/group IDs
permit document-level splits; a partial dossier is never called an intact book.
"""
from __future__ import annotations
import argparse, collections, concurrent.futures, hashlib, io, json, os, random, re, tarfile, threading, urllib.request
from pathlib import Path
from tokenizers import Tokenizer
from prepare_extra_sources import stream_hf, PINNED
from prepare_source_extensions import PINNED_EXT

SEED = 20260924
BANDS = {8192: (6000, 8100), 16384: (11000, 16000), 32768: (22000, 31500)}
DOMAIN_CAP = {'code': 85, 'documents': 55, 'legal': 50, 'office': 35, 'math': 50}
PAPER_REV = '901a9273a770e9d4138c5ddd91802f9c5c6cdc4b'


class Collector:
    def __init__(self, output):
        self.output = output
        self.path = output / 'candidates.jsonl.partial'
        tokenizer = next((Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots').glob('*/tokenizer.json'))
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.tokenizer_revision = tokenizer.parent.name
        self.counts, self.origins = collections.Counter(), collections.Counter()
        self.groups, self.seen = set(), set()
        self.lock = threading.Lock()
        if self.path.exists():
            with self.path.open() as previous:
                for line in previous:
                    row=json.loads(line)
                    self.counts[(row['domain'],row['target_tokens'])]+=1
                    self.origins[(row['provenance'].get('corpus',''),row['target_tokens'])]+=1
                    self.groups.add(row['doc_id']);self.seen.add(row['source_id'])
        self.file = self.path.open('a')

    def domain_full(self, domain):
        return all(self.counts[(domain, band)] >= DOMAIN_CAP[domain] for band in BANDS)

    def add(self, text, *, domain, doc_id, title, provenance, complete=True, members=None, kind='long_document', origin=None):
        if self.domain_full(domain) or doc_id in self.groups or not 16000 <= len(text) <= 350000:
            return False
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        n = len(encoded.ids)
        band = next((band for band, (lo, hi) in BANDS.items() if lo <= n <= hi), None)
        if band is None:
            return False
        digest = hashlib.sha256(text.encode()).hexdigest()
        origin = origin or provenance.get('corpus', '')
        with self.lock:
            if self.counts[(domain, band)] >= DOMAIN_CAP[domain] or doc_id in self.groups or digest in self.seen:
                return False
            if domain in {'code', 'documents'} and self.origins[(origin, band)] >= 5:
                return False
            sections = []
            at = 0
            offsets = encoded.offsets
            while at < len(text):
                token_at = next((i for i, (_, end) in enumerate(offsets) if end > at), len(offsets))
                token_end = min(token_at + 1500, len(offsets))
                end = len(text) if token_end == len(offsets) else offsets[token_end][0]
                # Keep natural line boundaries without changing a single byte.
                newline = text.rfind('\n', at + max(300, (end-at)//2), end)
                if newline > at and token_end < len(offsets):
                    end = newline + 1
                if end <= at:
                    end = min(len(text), at + 5000)
                part = text[at:end]
                tokens = len(self.tokenizer.encode(part, add_special_tokens=False).ids)
                sections.append({'id': f'{digest[:16]}:section:{len(sections):03d}', 'start_char': at, 'end_char': end,
                                 'tokens': tokens, 'title': next((line.strip()[:140] for line in part.splitlines() if line.strip()), title),
                                 'provenance': {'parent_doc_id': doc_id}})
                at = end
            usable = [i for i, section in enumerate(sections) if section['tokens'] >= 150]
            chosen = sorted({usable[round(f * (len(usable)-1))] for f in (.15, .5, .85)})
            if len(chosen) < 2:
                return False
            support_ids = [sections[i]['id'] for i in chosen]
            support_tokens = sum(sections[i]['tokens'] for i in chosen)
            assert support_tokens <= 6000
            row = {'schema_version': 1, 'source_id': digest, 'text': text, 'domain': domain, 'doc_id': doc_id,
                   'split_group_id': doc_id, 'source_kind': kind, 'source_tokens': n, 'target_tokens': band,
                   'complete_parent': complete, 'title': title, 'provenance': provenance,
                   'sections': sections, 'suggested_support_section_ids': support_ids,
                   'suggested_support_tokens': support_tokens}
            if members:
                row['parent_members'] = members
            self.file.write(json.dumps(row, ensure_ascii=False) + '\n')
            self.file.flush()
            self.counts[(domain, band)] += 1
            self.origins[(origin, band)] += 1
            self.groups.add(doc_id)
            self.seen.add(digest)
            if sum(self.counts.values()) % 25 == 0:
                print('candidates', sum(self.counts.values()), dict((f'{d}:{b}', c) for (d,b),c in sorted(self.counts.items())), flush=True)
            return True

    def finish(self):
        self.file.close()
        path = self.output / 'candidates.jsonl'
        self.path.replace(path)
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for part in iter(lambda: file.read(1024*1024), b''):
                digest.update(part)
        result = {'file': path.name, 'rows': sum(self.counts.values()), 'counts': {f'{d}:{b}': n for (d,b),n in sorted(self.counts.items())},
                  'bytes': path.stat().st_size, 'sha256': digest.hexdigest(), 'tokenizer_model': 'Qwen/Qwen3.5-2B',
                  'tokenizer_revision': self.tokenizer_revision, 'token_bands': BANDS,
                  'note': 'Raw evidence candidates only, no generated QA. Raw source token counts exclude encoder prompt wrapper. Intact documents are distinguished from selected coherent reference dossiers. Suggested support spans are exact substrings totaling<=6000 tokens. Split by split_group_id and parent member document identities.'}
        (self.output / 'manifest.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)


def cached_code_and_manuals(c):
    cache = Path('IB/TMP/SYNC/REPO_CACHE')
    repos = list(cache.iterdir()); random.Random(SEED).shuffle(repos)
    candidates = {'code': [], 'documents': []}
    for repo in repos:
        for base, dirs, files in os.walk(repo):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules','vendor','build','dist','target','__pycache__','third_party','generated'}]
            for name in files:
                path = Path(base) / name
                if path.suffix in {'.py','.java','.js','.ts','.cpp','.h','.go','.rs','.cs'}:
                    domain = 'code'
                elif path.suffix in {'.md','.rst','.mdx'} and any(x in str(path.relative_to(repo)).lower() for x in ('docs/','doc/','documentation/','readme')):
                    domain = 'documents'
                else:
                    continue
                if path.is_symlink() or not 20000 <= path.stat().st_size <= 280000:
                    continue
                candidates[domain].append(path)
    for domain, files in candidates.items():
        random.Random(f'{SEED}:{domain}').shuffle(files)
        for path in files:
            text = path.read_text(errors='replace')
            if domain == 'code' and any(len(line) > 700 for line in text.splitlines()):
                continue
            parts = path.relative_to(cache).parts
            repo, revision, relative = parts[0].replace('__','/',1), parts[1], '/'.join(parts[2:])
            c.add(text, domain=domain, doc_id=f'github/{repo}/{revision}/{relative}', title=f'{repo}: {relative}',
                  provenance={'url':f'https://github.com/{repo}/blob/{revision}/{relative}','corpus':repo,'revision':revision,'local_path':str(path)}, origin=repo)
            if c.domain_full(domain):
                break


def official_manuals(c):
    for key in ('python','kubernetes','mdn'):
        repo, revision = PINNED_EXT[key]
        with urllib.request.urlopen(f'https://codeload.github.com/{repo}/tar.gz/{revision}',timeout=120) as response, tarfile.open(fileobj=response,mode='r|gz') as archive:
            for member in archive:
                if not member.isfile() or not 24000 <= member.size <= 280000:
                    continue
                relative = member.name.split('/',1)[1]
                eligible=(key=='python' and relative.startswith('Doc/') and relative.endswith('.rst')) or (key=='kubernetes' and relative.startswith('content/en/docs/') and relative.endswith('.md')) or (key=='mdn' and relative.startswith('files/en-us/') and relative.endswith('index.md'))
                if not eligible:continue
                text=archive.extractfile(member).read().decode('utf-8',errors='replace')
                c.add(text,domain='documents',doc_id=f'github/{repo}/{revision}/{relative}',title=f'{repo}: {relative}',
                      provenance={'url':f'https://github.com/{repo}/blob/{revision}/{relative}','corpus':repo,'revision':revision},origin=repo)


def legal(c):
    for shard in (0,7,14,21,28,35):
        kept=0
        for row in stream_hf('caselaw',f'cap_{shard:05d}.jsonl.gz'):
            text=row.get('text','')
            if not 26000<=len(text)<=180000:continue
            repo,rev=PINNED['caselaw'];meta=row.get('metadata',{})
            if c.add(text,domain='legal',doc_id='caselaw/'+row['id'],title=text.strip().splitlines()[0][:250],
                     provenance={'url':meta.get('url',''),'corpus':repo,'revision':rev,'original_id':row['id']}):kept+=1
            if kept>=30 or c.domain_full('legal'):break
        if c.domain_full('legal'):break


def math(c):
    from compression import zstd
    url=f'https://huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/{PAPER_REV}/arxiv/train/arXiv_000.jsonl.zst'
    with urllib.request.urlopen(url,timeout=120) as response, zstd.ZstdFile(response) as reader:
        for ordinal,line in enumerate(io.TextIOWrapper(reader,encoding='utf-8')):
            row=json.loads(line);text=row['text']
            if not 24000<=len(text)<=180000 or sum(text.count(ch) for ch in ('=','\\','^','_'))<100:continue
            meta=row.get('meta',{})
            if isinstance(meta,str):
                try:meta=json.loads(meta)
                except ValueError:meta={'raw_metadata':meta}
            ident=str(meta.get('arxiv_id') or meta.get('id') or meta.get('url') or hashlib.sha256(text.encode()).hexdigest()[:24])
            c.add(text,domain='math',doc_id='proof-pile-2/arxiv/'+ident,title=text.splitlines()[0][:250],
                  provenance={'url':url,'corpus':'EleutherAI/proof-pile-2/arxiv','revision':PAPER_REV,'metadata':meta})
            if c.domain_full('math') or ordinal>=10000:break


def office(c):
    groups=collections.defaultdict(list)
    path=Path('IB/TMP/FIRST_TASK_AWARE_FINETUNING/expanded_sources/office_automation.jsonl')
    for line in path.open():
        row=json.loads(line)
        match=re.search(r'/(Excel|Word|PowerPoint|Outlook)\.([^.]+)',row['doc_id'],re.I)
        if match:groups[f'{match[1].lower()}.{match[2].lower()}'].append(row)
    keys=sorted(groups);random.Random(SEED).shuffle(keys)
    for key in keys:
        if 'office-reference/'+key in c.groups:continue
        rows=sorted(groups[key],key=lambda row:(row['doc_id'],row['provenance'].get('document_offsets_chars',[0])[0]))
        # One coherent object-reference packet per split group. Vary the natural
        # prefix budget without repeating an object across length bands.
        preferred=list(BANDS)[int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%3]
        text='';members=[]
        for row in rows:
            block=f'\n\n## {row["title"]}\n\n'+row['text']
            candidate=text+block
            if len(c.tokenizer.encode(candidate,add_special_tokens=False).ids)>BANDS[preferred][1]:break
            start=len(text);text=candidate
            members.append({'source_id':row['source_id'],'doc_id':row['doc_id'],'start_char':start,'end_char':len(text),'provenance':row['provenance']})
        c.add(text,domain='office',doc_id='office-reference/'+key,title='Microsoft Office '+key+' reference dossier',
              provenance={'corpus':'MicrosoftDocs/VBA-Docs','revision':PINNED['office_automation'][1],
                          'ordering':'alphabetical API reference within one application/object; selected prefix of available reference pages'},
              complete=False,members=members,kind='related_reference_dossier',origin=key.split('.')[0])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('IB/TMP/FIRST_TASK_AWARE_FINETUNING/long_evidence_sources'));args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'candidates.jsonl').exists():raise FileExistsError('Published candidate file is immutable; use a new output directory')
    (args.output/'PRESERVE').write_text('Reusable long-evidence source candidates and exact support spans.\n')
    c=Collector(args.output);errors={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        jobs={executor.submit(fn,c):fn.__name__ for fn in (cached_code_and_manuals,official_manuals,legal,math,office)}
        for future in concurrent.futures.as_completed(jobs):
            try:future.result()
            except Exception as error:errors[jobs[future]]=repr(error);print('FAILED',jobs[future],repr(error),flush=True)
    c.finish();(args.output/'preparation_errors.json').write_text(json.dumps(errors,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
