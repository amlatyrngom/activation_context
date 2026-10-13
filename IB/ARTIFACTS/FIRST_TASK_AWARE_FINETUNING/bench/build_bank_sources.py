"""Raw-only source preparation; deliberately never load upstream QA/task records."""
import json,random,hashlib,os,re,collections
from pathlib import Path
ROOT=Path('/workspace');OUT=ROOT/'IB/TMP/FIRST_TASK_AWARE_FINETUNING/bank_sources_100k.json'
rng=random.Random(20260922);pools={d:[] for d in ('code','numeric','web','documents')};seen=set()
def add(domain,doc_id,text,kind):
 text=text.strip()
 if len(text)<300:return
 key=hashlib.sha256(text.encode()).hexdigest()
 if key in seen:return
 seen.add(key);pools[domain].append(dict(doc_id=doc_id,text=text,source_id=key,source_kind=kind))
def windows(text,size):
 lines=text.splitlines(keepends=True);part=''
 for line in lines:
  if part and len(part)+len(line)>size:
   yield part;part=''
  if len(line)>size:
   for at in range(0,len(line),size):yield line[at:at+size]
  else:part+=line
 if part:yield part
raw=json.load(open(ROOT/'IB/TMP/SYNC/AC_RECON/preparation/material_mix.json'))
for row in raw['train']:
 if row['source']=='web':add('web',row['doc_id'],row['text'],'web_passage')
 elif row['source']=='docs' and row['doc_id'].startswith('bright/'):
  add('documents',row['doc_id'],row['text'],'technical_post')
# Full raw Loong documents, without opening loong.jsonl or instance question records.
for folder,domain in [('financial','numeric'),('paper','documents')]:
 files=sorted((ROOT/'IB/TMP/SYNC/LOONG/doc'/folder).iterdir());rng.shuffle(files)
 for path in files:
  if not path.is_file():continue
  text=path.read_text(errors='replace')
  for i,chunk in enumerate(windows(text,10000)):
   if domain=='numeric' and sum(c.isdigit() for c in chunk)<25:continue
   add(domain,'loong/'+path.name,f'Document: {path.name}\nWindow {i+1}:\n'+chunk,'document_window')
# Read repository files only, never benchmark issue descriptions, solutions, or task metadata.
repos=sorted((ROOT/'IB/TMP/SYNC/REPO_CACHE').iterdir());rng.shuffle(repos)
for repo in repos:
 candidates=[]
 for base,dirs,files in os.walk(repo):
  dirs[:]=[d for d in dirs if not d.startswith('.') and d not in ('node_modules','vendor','build','dist','target','__pycache__')]
  for name in files:
   p=Path(base)/name
   if p.suffix in ('.py','.java','.cpp','.h','.js','.ts') and not p.is_symlink() and 400<=p.stat().st_size<=150000:candidates.append(p)
 rng.shuffle(candidates);kept=0
 for p in candidates:
  text=p.read_text(errors='replace')
  if any(len(line)>500 for line in text.splitlines()):continue
  for i,chunk in enumerate(windows(text,10000)):
   add('code',str(p.relative_to(ROOT/'IB/TMP/SYNC/REPO_CACHE')),f'File: {p.name}\n'+chunk,'code_window');kept+=1
   if kept>=800:break
  if kept>=800:break
 if len(pools['code'])>=65000:break
# Add raw arXiv paper text, never a math benchmark's questions or solutions.
import urllib.request,io
from compression import zstd
paper_count=0
for shard in range(8):
 url=f'https://huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/main/arxiv/train/arXiv_{shard:03d}.jsonl.zst'
 with urllib.request.urlopen(url,timeout=90) as response:
  with zstd.ZstdFile(response) as reader:
   for line in io.TextIOWrapper(reader,encoding='utf-8'):
    paper=json.loads(line);text=paper['text'];meta=paper.get('meta',{})
    if isinstance(meta,str):
     try:meta=json.loads(meta)
     except ValueError:meta={'raw_meta':meta}
    paper_id=str(meta.get('arxiv_id') or meta.get('id') or meta.get('url') or hashlib.sha256(text.encode()).hexdigest()[:20])
    chunks=list(windows(text[:400000],9000));rng.shuffle(chunks)
    domain='numeric' if paper_count%3 else 'documents'
    for chunk in chunks[:12]:
     if domain=='numeric' and sum(chunk.count(c) for c in ('=','\\','^','_'))<20:continue
     add(domain,'proof-pile-2/arxiv/'+paper_id,chunk,'arxiv_paper_window')
    paper_count+=1
    if paper_count%500==0:print('arxiv',paper_count,{d:len(v) for d,v in pools.items()},flush=True)
    if len(pools['numeric'])>=55000 and len(pools['documents'])>=40000:break
 if len(pools['numeric'])>=55000 and len(pools['documents'])>=40000:break
# Read only raw MS MARCO passage columns, never query/answer/selection labels.
from datasets import load_dataset
web=load_dataset('microsoft/ms_marco','v1.1',split='train',streaming=True).select_columns(['passages'])
for ordinal,row in enumerate(web):
 for index,text in enumerate(row['passages']['passage_text']):
  add('web',f'msmarco:raw:{ordinal}:{index}',text,'web_passage')
 if len(pools['web'])>=55000:break
for rows in pools.values():rng.shuffle(rows)
OUT.with_suffix('.tmp').write_text(json.dumps({'seed':20260922,'pools':pools,'note':'Raw source fields only. 30/25/25/20 code/numeric/web/documents; wiki omitted from the expansion bank. No native questions, answers or solution traces.'}))
OUT.with_suffix('.tmp').replace(OUT)
print(json.dumps({'counts':{d:len(v) for d,v in pools.items()},'bytes':OUT.stat().st_size,'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest()},indent=2))
