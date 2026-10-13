"""Freeze available portable text, prepare one mixed pass, then train on 1/2/4 local GPUs.

Preparation is a separate CPU command. Training uses one public trainer call,
fixed initial-reader controls, the native paired checkpoint and unchanged global batch8.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import fcntl,hashlib,heapq,json,math,os,pickle,random,signal,time,unicodedata
from pathlib import Path

FAMILIES=('rag_qa','trajectory_qa','compaction','agent_teacher','agent_regular','reconstruction')   # every family the packer and reports know
# The mixed run (MIXED_RUN_PLAN.md): no reconstruction, the uncontextual Q/A views take its place.
TRAIN_FAMILIES=('rag_qa','trajectory_qa','compaction','agent_teacher','agent_regular')
QA_FAMILIES=('rag_qa','trajectory_qa')
SHARES=None   # natural mix: every example weighs 1; a {family: share} dict restores explicit loss shares
DROPPED_ROOT_FINISH=('max_tool_errors','no_tool_call')   # regular-agent runs whose segments do not train (broken behavior)
FOLDERS={'agent_regular':'agent_teacher_regular'}   # portable collection folder per family, when it differs
AGENT_FAMILIES=('agent_teacher','agent_regular')
SEED=20260925
MODES={'question':'contextual','parent':'recursive','summary':'uncontextual'}   # report labels (older packets carry 'parent')
VIEWS=('question','summary')   # depth 1: the mixed run trains no recursive parent view
RAG_VIEW=None   # forced encoder view of every training RAG item (None: the seeded 7/9 contextual draw)
RATIO=None      # one compression ratio for every Q/A, compaction and reconstruction item, train and held (None: sampled train ranges, held 1/16)
BARE_BLIND=False   # question-blind RAG encoders read the bare evidence (the reconstruction warm-up's input) instead of evidence + the summary instruction
OLD_TEXT_COMPACTION=("The conversation so far has been compacted into the activation context above. Continue the task "
 "from exactly where it left off, as if the full conversation were still in front of you.")

def panel_group(item):
 """Reporting group: task, then encoder view (RAG/trajectory Q/A), diagnostic or multi/flat (reconstruction), or agent task kind."""
 info=item.info;family=info.get('family');multi=bool(info.get('multi_target_count'))
 if family in ('rag_qa','trajectory_qa'):return family+'/'+('multi' if multi else MODES.get(info.get('encoder_conditioning'),'other'))
 if family=='compaction':return 'compaction/'+('multi' if multi else 'single')
 if family=='reconstruction':return 'reconstruction/'+('multi' if multi else info.get('diagnostic') or 'flat')
 if family=='agent_regular':return 'agent/'+str(info.get('task_kind'))
 if family=='agent_teacher':return 'agent/evidence_qa'
 return str(family)

REFERENCE_SUFFIXES=(':no_context',':full_context',':recent_text')   # transform_for_eval_reference renames its views

def base_id(item_id):
 for suffix in REFERENCE_SUFFIXES:
  if item_id.endswith(suffix):return item_id[:-len(suffix)]
 return item_id

def group_summary(rows,groups,overlap=frozenset()):
 """Per-group mean objective and token-weighted gold NLL over evaluation rows (reference views map to their item's group;
 ids outside `groups` are skipped). `clean_*` excludes items whose source text also occurs in the reconstruction corpus."""
 sums={}
 for row in rows:
  ident=base_id(row['item_id']);group=groups.get(ident)
  if group is None:continue
  for key in (group,)+((group+'|clean',) if ident not in overlap else ()):
   s=sums.setdefault(key,[0,0.,0.,0]);s[0]+=1;s[1]+=row['loss'];s[2]-=sum(row['target_logp']);s[3]+=len(row['target_logp'])
 out={}
 for key,(n,loss,nll,positions) in sorted(sums.items()):
  if key.endswith('|clean'):continue
  clean=sums.get(key+'|clean',[0,0.,0.,0])
  out[key]={'items':n,'loss':loss/n,'nll_token':nll/max(1,positions),'clean_items':clean[0],'clean_nll_token':clean[2]/clean[3] if clean[3] else None}
 return out

def kind_summary(rows):
 """Headline held loss and per-kind losses over the given evaluation rows (the trainer's summary shape)."""
 by={}
 for row in rows:by.setdefault(row['kind'],[]).append(row)
 def one(group):
  positions=sum(len(r['target_logp']) for r in group)
  return {'loss':sum(r['loss'] for r in group)/len(group),'gold_nll_token':-sum(sum(r['target_logp']) for r in group)/max(1,positions),'items':len(group)}
 return (one(rows) if rows else {'loss':None,'items':0}),{kind:one(group) for kind,group in by.items()}

def headline_summary(rows,diagnostic_ids):
 """Headline over the rows outside the held diagnostics (reference views map to their item); by-kind over all rows."""
 headline,_=kind_summary([r for r in rows if base_id(r['item_id']) not in diagnostic_ids])
 return headline,kind_summary(rows)[1]

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def write(path,value):
 from activation.common.reporting import _write_atomic
 _write_atomic(Path(path),json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def normalized(question):return ' '.join(unicodedata.normalize('NFC',question).lower().split())   # held/reserved question key

def row_identity(family,row):
 if family=='agent_teacher':return row['task']['id']
 if family=='agent_regular':return row['segment_id']
 if family=='reconstruction':return str(row.get('doc_id','passage'))+':'+digest(row['text'])
 return row['id']

def reserved_questions(snapshots):
 """The RAG bank's reserved questions (existing evaluation selections), normalized: they never train, in any family."""
 return {normalized(q) for q in snapshots['rag_qa']['controls']['reserved_questions']['data']['questions']}

def load_held(panel):
 """The drawn held identities of a committed task panel (its held.json, checked against panel.json)."""
 receipt=json.loads((panel/'panel.json').read_text())
 if file_sha(panel/'held.json')!=receipt.get('held_sha256'):raise ValueError('Task panel held.json changed or missing')
 value=json.loads((panel/'held.json').read_text())
 return {'rag_qa':set(map(normalized,value['rag_qa']['questions'])),'trajectory_qa':set(value['trajectory_qa']['ids'])},receipt

def select_rows(args,snapshots,held,subset=None,families=TRAIN_FAMILIES):
 """Seeded training rows per family: every usable row of the bank up to its limit, the held panel identities excluded.
 `subset` ({family: identities}) keeps only the listed rows (the M4 probe packet)."""
 from activation.dataset.dataset_caching import frozen_text_rows
 selected={};counts={};excluded=Counter()
 reserved=reserved_questions(snapshots)
 limits=dict(zip(TRAIN_FAMILIES,(args.qa_questions,args.qa_questions,args.summary_histories,args.teacher_histories,args.regular_segments)))
 for family in families:
  pool=[];available=Counter();seen=set();found=set()
  for row,location in frozen_text_rows(args.data/FOLDERS.get(family,family),snapshots[family]):
   ident=row_identity(family,row)
   if ident in seen:raise ValueError(f'Duplicate {family} identity: {ident}')
   seen.add(ident)
   if subset is not None and ident not in subset.get(family,()):continue
   split,reason=row_split(family,row,reserved,held)
   if split is None:
    excluded[reason]+=1;continue
   available[split]+=1
   if split=='held':found.add(normalized(row['question']) if family=='rag_qa' else ident)
   if split!='train':continue                         # held rows live in the task panel only
   score=int(digest((SEED,ident)),16);entry=(-score,ident,row,location)
   if len(pool)<limits[family]:heapq.heappush(pool,entry)
   elif score < -pool[0][0]:heapq.heapreplace(pool,entry)
  chosen=[(-score,row,location) for score,_,row,location in sorted(pool,reverse=True)]
  if not chosen and subset is None:raise ValueError('No available training rows: '+family)
  if family in QA_FAMILIES and subset is None and not held[family]<=found:
   raise ValueError(f'{family}: {len(held[family]-found)} of the {len(held[family])} drawn held rows are missing from the bank')
  selected[family]=chosen
  counts[family]={'available_train':available['train'],'held_rows':available['held'],'selected_train':len(chosen)}
 return selected,counts,dict(excluded)

def row_split(family,row,reserved,held):
 """(split, None) for a usable row; (None, reason) for an excluded one. The banks' own split fields and hash rules are
 ignored: 'held' is exactly the drawn panel identities (`held`: RAG question texts, trajectory Q/A row ids); every other
 usable row trains. Questions are held out, not histories: a held question's history may train through other questions."""
 if family=='rag_qa':
  question=normalized(row['question'])
  if question in reserved:return None,'rag_reserved_qa'
  return ('held' if question in held['rag_qa'] else 'train'),None
 if family=='agent_teacher':
  question=normalized(row['task']['question'])
  if question in reserved:return None,'teacher_reserved_qa'
  if question in held['rag_qa']:return None,'teacher_held_qa'   # a held RAG question never trains through evidence QA
  if not row['quality'].get('submitted') or row['run'].get('finish_reason')!='submitted':return None,'teacher_unfinished'
  return 'train',None           # no lexical-F1 filter: correct paraphrases are deliberately retained
 if family=='trajectory_qa':return ('held' if row_identity(family,row) in held['trajectory_qa'] else 'train'),None
 if family=='agent_regular' and row['root']['finish_reason'] in DROPPED_ROOT_FINISH:return None,'regular_failed_root'
 return 'train',None            # compaction, agent_regular: every usable row trains (the export's root split is ignored)

def conditioning(ident):
 # 70/20 contextual/uncontextual of all examples once 10% become multi-input groups: 7 of 9 singles see the question.
 value=int(digest((SEED,'view',ident))[:8],16)%9
 return 'question' if value<7 else 'summary'

def make_item(study,family,row,split,*,mode=None):
 """`mode` forces the Q/A encoder view (default: the seeded draw); reconstruction items are flat."""
 from activation.ac_model import ActivationContextTrainingItem
 from activation.ac_model.ac_model_utils import assistant_target_ids
 from activation.common.ac_parts import ac_part,COMPACTION_SUMMARY_PROMPT
 from activation.dataset.dataset import DatasetDocument
 ident=row_identity(family,row);rng=random.Random(f'{SEED}:{family}:{ident}')
 ratio=RATIO or (1/16 if split=='held' else rng.uniform(1/20,1/10))
 if family=='trajectory_qa':
  history=row['history'];mode=mode or conditioning(ident)
  document=DatasetDocument(ident,'portable_traj_qa',trajectory=history['messages'],trajectory_kwargs=history)
  messages=history['messages'];parent=None
  if mode=='parent':
   # Earlier trajectory goes through child rows; the outer encoder receives fresh later evidence.
   cut=max(1,len(messages)//2);parent=messages[:cut];messages=messages[cut:]
   if not messages:messages=[{'role':'user','content':'Use the inherited trajectory to answer the pending question.'}]
  item=study.trajectory_qa_item(document,messages,row['question'],row['answer'],
   item_id=ident,compression_ratio=ratio,depth=1,encoder_sees_question=mode!='summary',
   parent_messages=parent)
 elif family=='compaction':
  history=deepcopy(row['history']);messages=history['messages'];tools=history.get('tools')
  first=next((i for i,m in enumerate(messages) if m['role']=='assistant'),len(messages))
  frame=deepcopy(messages[:first]);part=ac_part(messages,study.ac_name,ratio,kind='compaction')
  if tools:part['tools']=deepcopy(tools)
  student=frame+[{'role':'user','content':[part,{'type':'text','text':'\n\n'+COMPACTION_SUMMARY_PROMPT}]}]
  full=deepcopy(messages)+[{'role':'user','content':COMPACTION_SUMMARY_PROMPT}]
  answer={'role':'assistant','content':row['summary']};kwargs={'enable_thinking':False}
  item=ActivationContextTrainingItem(ident,'compaction_summary',full+[deepcopy(answer)],student+[answer],len(full),len(student),
   [assistant_target_ids(study.tokenizer,answer,tools,kwargs)],tools=tools,info={'chat_template_kwargs':kwargs,'ratio':ratio})
 elif family=='agent_teacher':
  run=row['run'];tools=deepcopy(row['provenance']['tools']);kwargs={'enable_thinking':True}
  if run.get('ac_model_name') or run.get('compactions') or run.get('subagent_results'):raise ValueError('Expected plain single-agent teacher trajectory')
  messages=deepcopy(run['prompt_messages']);start=len(messages);targets=[]
  for step in run['trajectory']:
   messages.extend(deepcopy(step['messages']))
   if step['role']=='assistant':
    assistants=[m for m in step['messages'] if m['role']=='assistant']
    if len(assistants)!=1:raise ValueError('Assistant turn does not have exactly one message')
    targets.append(assistant_target_ids(study.tokenizer,assistants[0],tools,kwargs))
  item=ActivationContextTrainingItem(ident,'agent_sft',None,messages,0,start,targets,tools=tools,
   info={'chat_template_kwargs':kwargs,'text_only':True,'quality':row['quality']})
 elif family=='agent_regular':
  # One unrolled teacher segment (main run, compacted segment or subagent) exactly as the agent saw it: its recorded prompt
  # and step messages, text only; targets are its recorded turns. Never an AC part: the teachers ran without an AC model.
  from activation.common.ac_parts import COMPACTION_TEXT_INSTRUCTIONS,is_ac_part
  tools=deepcopy(row['tools']) or None;kwargs={'enable_thinking':True}
  messages=deepcopy(row['messages']);start=row['target_start']
  if any(is_ac_part(part) for m in messages if isinstance(m.get('content'),list) for part in m['content']):
   raise ValueError('Regular agent segment carries an AC part: '+ident)
  # The 12h campaign's text-only resumes still said "compacted into the activation context above": use the text-mode wording.
  for message in messages[:start]:
   if message['role']=='user' and isinstance(message.get('content'),str):
    message['content']=message['content'].replace(OLD_TEXT_COMPACTION,COMPACTION_TEXT_INSTRUCTIONS)
  targets=[assistant_target_ids(study.tokenizer,m,tools,kwargs) for m in messages[start:] if m['role']=='assistant']
  root=row['root']
  item=ActivationContextTrainingItem(ident,'agent_sft',None,messages,0,start,targets,tools=tools,
   info={'chat_template_kwargs':kwargs,'text_only':True,'task_kind':root['task_kind'],'benchmark':root['benchmark'],
    'role':row['segment']['role'],'path':row['path'],'root_finish_reason':root['finish_reason'],'root_score':root['score']})
 elif family=='reconstruction':
  item=study.generate_reconstruction_samples([row],ratio=RATIO or (1/16 if split=='held' else rng.uniform(1/30,1/10)),
   seed=int(digest(ident)[:8],16))[0]
  item.item_id=ident
 else:raise ValueError(family)
 item.info.update(family=family,source=row.get('source') if isinstance(row.get('source'),str) else row.get('source_family'),portable_id=ident)
 return item

def reader_tokens(builder,study,item,example=None):
 """The longer of the item's compressed and full reader histories (`example`: its compressed example, already built)."""
 example=example or builder.build_example(study.ac_model,item)
 lengths=[len(example.student_ids)]
 if item.in_context_messages is not None:
  full=replace(item,ac_messages=item.in_context_messages,ac_start=item.in_context_start)
  lengths.append(len(builder.build_example(study.ac_model,full).student_ids))
 return max(lengths)

def pack_multi_targets(study,items,*,fraction,split,max_reader_tokens=24_576,workers=1):
 """Groups of 2/3 short sources per Q/A family (see the receipt). With workers > 1 the candidate groups' reader lengths
 are measured ahead in one fork pool along the no-rejection path (a group is fully determined by its start and size) and
 the sequential walk below consumes them, so the packing is identical to the one-process walk. The look-ahead doubles
 while the path holds and restarts at one key per worker after a rejection shifts it, so a rejection-heavy tail costs
 about one parallel measurement per step, never a long plan re-measured per rejection."""
 global BUILD_CONTEXT
 from activation.ac_model import ActivationContextTrainer,ActivationContextTrainingConfig
 from activation.dataset.dataset_utils import message_text
 if not 0<=fraction<=.5:raise ValueError('Multi-target fraction must be between0and0.5')
 if not 0<max_reader_tokens<=72000:raise ValueError('Grouped reader budget must be within the native72000-token limit')
 builder=ActivationContextTrainer(study.harness,ActivationContextTrainingConfig(loss_kind='sft',max_example_tokens=1_000_000))
 families={}
 for family in FAMILIES:
  rows=[i for i in items if i.info['family']==family]
  if family not in AGENT_FAMILIES and fraction:
   random.Random(f'{SEED}:multi:{split}:{family}').shuffle(rows)
   # Multi-input familiarity uses shorter intact sources; long inputs remain singles.
   rows.sort(key=lambda item:sum(len(message_text(m)) for m in (item.in_context_messages or item.ac_messages)))
  families[family]=rows
 output=[];receipt={}
 BUILD_CONTEXT={'pack':(study,families,builder)}
 try:
  with ForkPool(workers) as pool:
   for family,rows in families.items():
    if family in AGENT_FAMILIES or not fraction:output.extend(rows);continue
    desired=round(len(rows)*fraction/(1+1.5*fraction));grouped=0;cursor=0;packed=[];rejected=0;measured={};window=workers
    def measure(start,count):
     nonlocal window
     if (start,count) not in measured and pool.parallel:
      # Measure ahead from here, assuming no rejection (its first key is this one; a 3-group's 2-group fallback follows it).
      measured.clear();keys=[];at,made=start,grouped
      while made<desired and len(rows)-at>=2 and len(keys)<window:
       size=min(2+made%2,len(rows)-at);keys.append((at,size))
       if size==3:keys.append((at,2))
       at+=size;made+=1
      window=min(16_384,2*window);step=-(-len(keys)//(4*workers))
      chunks=[(family,keys[i:i+step]) for i in range(0,len(keys),step)]
      for (_,chunk),values in zip(chunks,pool.map(pack_chunk,chunks)):
       measured.update(zip(chunk,values))
       if not pool.parallel:break   # the pool fell back to this process: measure only what the walk asks for
     # Measured ahead: the length only (the walk builds the groups it takes); unmeasured or failed: built here.
     length=measured.pop((start,count),None)
     return (None,length) if length is not None else combined_group(study,rows,builder,start,count)
    while grouped<desired and len(rows)-cursor>=2:
     count=min(2+grouped%2,len(rows)-cursor)
     combined,length=measure(cursor,count)
     if length>max_reader_tokens:
      window=workers
      if count==3:
       count=2;combined,length=measure(cursor,2)
      if length>max_reader_tokens:
       packed.append(rows[cursor]);cursor+=1;rejected+=1;continue
     measured.pop((cursor,2),None)
     packed.append(combined if combined is not None else multi_item(study,rows[cursor:cursor+count]));cursor+=count;grouped+=1
    packed.extend(rows[cursor:]);output.extend(packed)
    receipt[family]={'unique_sources':len(rows),'training_examples':len(packed),'multi_target_examples':grouped,
     'actual_fraction':grouped/len(packed) if packed else 0,'capacity_rejections_kept_single':rejected,
     'max_reader_tokens':max_reader_tokens,'selection':'shorter intact sources first; seeded ties'}
 finally:BUILD_CONTEXT=None
 return output,receipt

def multi_item(study,chosen):
 return study.multi_target_item(chosen,item_id='multi:'+digest([r.item_id for r in chosen]))

def combined_group(study,rows,builder,start,count):
 """(multi-input item, its reader length) of rows[start:start+count]."""
 combined=multi_item(study,rows[start:start+count])
 return combined,reader_tokens(builder,study,combined)

# Parallel preparation (fork): the parent sets BUILD_CONTEXT, a ForkPool forks the workers at its first map, and each
# chunk worker below reads the inherited state and returns its results in order. Every row is seeded by its own
# identity, so a row builds the same wherever it is built; a chunk's items and examples come back pickled together,
# keeping their shared objects.
BUILD_CONTEXT=None
CHUNK=64

class ForkPool:
 """map(function, tasks) yields function(task) for every task, in order, with at most 4 x workers tasks in flight. The
 workers fork at the first parallel map, so they see BUILD_CONTEXT as set then, and serve every later map of the block;
 one worker, or a single task, runs in this process. The parent's objects are frozen out of the cyclic collector
 first: a forked worker's collection would otherwise touch, and so copy, the parent's whole heap (tens of GB of rows).
 A worker that dies outright (OOM kill, crash) breaks the pool, seen by a result or by the next submit: results already
 finished are still yielded, then the pool is forked again from the first task not yet yielded; after a third break
 (several workers dying together count once) the block finishes in this process, as one process would build it (a
 crash one process meets recurs there)."""
 DEATHS=3
 def __init__(self,workers):self.workers=workers;self.parallel=workers>1;self.pool=None;self.deaths=0
 def __enter__(self):return self
 def __exit__(self,*exc):
  if self.pool is not None:
   import gc
   try:self.pool.shutdown(cancel_futures=True)
   finally:self.pool=None;gc.unfreeze()
 def map(self,function,tasks):
  import gc,multiprocessing,sys
  from collections import deque
  from concurrent.futures import ProcessPoolExecutor
  from concurrent.futures.process import BrokenProcessPool
  tasks=list(tasks);position=0                     # the first task not yet yielded
  while position<len(tasks):
   if not self.parallel or len(tasks)-position<=1:
    for task in tasks[position:]:yield function(task)
    return
   inflight=deque()
   try:
    if self.pool is None:
     gc.freeze();self.pool=ProcessPoolExecutor(max_workers=self.workers,mp_context=multiprocessing.get_context('fork'))
    submitted=position
    while True:
     while submitted<len(tasks) and len(inflight)<4*self.workers:
      inflight.append(self.pool.submit(function,tasks[submitted]));submitted+=1
     if not inflight:break
     result=inflight[0].result();inflight.popleft();position+=1
     yield result
   except BrokenProcessPool as error:
    # Finished results first: rebuilding them would also free their unpickled copies just before the next fork.
    while inflight and inflight[0].done() and not inflight[0].cancelled() and inflight[0].exception() is None:
     result=inflight.popleft().result();position+=1
     yield result
    self.deaths+=1;self.__exit__();self.parallel=self.deaths<self.DEATHS
    print(f'prepare: a worker died ({error}); death {self.deaths}: '+('forking the pool again' if self.parallel else 'finishing this stage in one process')
     +f' from task {position}/{len(tasks)}',file=sys.stderr,flush=True)
   finally:
    for future in inflight:future.cancel()

def attempt(scan,family,ident,build,constituents=None):
 """(value, None); under --scan a failing build gives (None, failure record) instead of raising."""
 if not scan:return build(),None
 try:return build(),None
 except Exception as error:
  return None,{'family':family,'id':ident,'error':f'{type(error).__name__}: {error}'[:600],**({'constituent_ids':constituents} if constituents else {})}

def pack_chunk(task):
 """Reader length per key, or None where the build fails: the look-ahead may measure groups the walk never takes, so a
 failure only counts (rebuilt, and raised, in the parent) if the walk reaches that group. Only lengths come back: the
 parent builds the groups it takes, so nothing measured is unpickled and freed there before the example workers fork."""
 family,keys=task;study,families,builder=BUILD_CONTEXT['pack'];out=[]
 for start,count in keys:
  try:out.append(combined_group(study,families[family],builder,start,count)[1])
  except Exception:out.append(None)
 return out

def rag_chunk(span):
 """Native matched-BM25 RAG items of the inherited entries[start:stop] against the parent's one index, then the
 mixture's view: per entry (item, failure). A failing chunk is rebuilt row by row under --scan, so only the failing rows
 drop (the one-process build recorded one '<material build>' failure and dropped every RAG row)."""
 build,extra,scan=BUILD_CONTEXT['rag_build'],BUILD_CONTEXT['rag_extra'],BUILD_CONTEXT['scan']
 entries=BUILD_CONTEXT['entries'][slice(*span)]
 def items_of(chunk):return build(rag_material([row for _,row,_ in chunk]),extra)
 if scan:
  try:built=[(item,None) for item in items_of(entries)]
  except Exception:
   built=[]
   for entry in entries:
    value,failure=attempt(True,'rag_qa',row_identity('rag_qa',entry[1]),lambda entry=entry:items_of([entry]))
    built.append((value[0] if value else None,failure))
 else:built=[(item,None) for item in items_of(entries)]
 out=[]
 for (item,failure),(_,row,_) in zip(built,entries,strict=True):
  if item is not None:item,failure=attempt(scan,'rag_qa',item.item_id,lambda item=item,row=row:condition_rag(item,row,'train'))
  out.append((item,failure))
 return out

def item_chunk(task):
 """(item, None) or (None, failure) per inherited row of rows[family][start:stop], with or without --scan: a chunk may
 run past a token budget's stop, so the parent decides, in row order, whether a failure counts (and raises)."""
 family,start,stop=task;study=BUILD_CONTEXT['study']
 return [attempt(True,family,row_identity(family,row),lambda row=row:make_item(study,family,row,'train')) for _,row,_ in BUILD_CONTEXT['rows'][family][start:stop]]

def budgeted_items(results,entries,budget,scan,failures,rebuild,progress=lambda done:None):
 """(item, location) per built row, in row order as one process builds them: a token budget stops at the same row
 (later rows' failures unrecorded); without --scan the first failing row before the stop raises its own error, rebuilt
 by `rebuild(row)` here (a worker's chunk may run past the stop, so its failures only count once reached)."""
 tokens=0;results=iter(results)
 for done,(_,row,location) in enumerate(entries,1):
  if budget is not None and tokens>=budget:return   # before waiting on the next result: nothing past the stop is awaited
  item,failure=next(results)
  progress(done)
  if failure and not scan:
   rebuild(row);raise RuntimeError(f'{failure["family"]} row {failure["id"]} failed in a worker only: {failure["error"]}')
  if failure:failures.append(failure)
  if item is None:continue
  tokens+=sum(map(len,item.assistant_token_ids))
  yield item,location

def checked_example(trainer,study,item):
 """The item's native training example, refused past the reader/encoder capacity, without targets, nested or (agent)
 carrying AC work."""
 from activation.ac_model.ac_model_utils import validate_part_capacity
 example=trainer.build_example(study.ac_model,item)
 length=reader_tokens(trainer,study,item,example)
 if length>72000:raise ValueError(f'full/compressed reader history of {length} tokens exceeds 72000')
 if not len(example.target_ids):raise ValueError('no target tokens (the token-mean loss would be undefined)')
 for request in example.part_requests:validate_part_capacity(study.ac_model,request,65536)
 if nested_parts(item):raise ValueError('an AC part nests another part (depth-1 rule)')
 if item.info['family'] in AGENT_FAMILIES and example.part_requests:raise ValueError('a text-only agent item carries AC work')
 return example

def example_chunk(indices):
 """Per item index: ((item, compact example, execution record), None) or (None, failure)."""
 from activation.ac_model.ac_model_utils import ac_execution_records,ac_execution_cost
 from run_trajectory_tasks import compact_example
 trainer,study,scan,output=BUILD_CONTEXT['trainer'],BUILD_CONTEXT['study'],BUILD_CONTEXT['scan'],BUILD_CONTEXT['output']
 out=[]
 for index in indices:
  item=output[index]
  example,failure=attempt(scan,item.info['family'],item.item_id,lambda item=item:checked_example(trainer,study,item),item.info.get('constituent_ids'))
  if example is None:out.append((None,failure));continue
  item.info['target_tokens']=len(example.target_ids);item.info['reader_tokens']=len(example.student_ids)
  example=compact_example(example)
  record=ac_execution_records(study.ac_model,[example])[0];example.execution_cost=ac_execution_cost(record)
  out.append(((item,example,record),None))
 return out

class StageClock:
 """Progress lines for the long preparation stages (at most one per 30 s, and at the end)."""
 def __init__(self,name,total):self.name,self.total,self.started,self.last=name,total,time.monotonic(),0.
 def __call__(self,done):
  now=time.monotonic()
  if done>=self.total or now-self.last>=30:
   self.last=now;rate=done/max(1e-9,now-self.started)
   print(f'prepare: {self.name} {done}/{self.total} ({now-self.started:.0f}s, {rate:.1f}/s)',flush=True)
 def seconds(self):return round(time.monotonic()-self.started,1)

def rag_answer(row):
 # NFC, as the reader tokenizer normalizes: a decomposed answer (e.g. 'p' + U+0307) would not round-trip (same target ids).
 return unicodedata.normalize('NFC',row['answer']).strip()

def rag_material(rows):
 return [{'domain':r['domain'],'original_doc_ids':r['source'].get('doc_ids',[]),
  'example':{'example_id':r['id'],'query':r['question'],'gold_answers':[rag_answer(r)],
   'study_source':{'units':[{'doc_id':u.get('doc_id',u.get('id')),'text':u['text']} for u in r['source']['units']],'request_id':r['provenance']['source_request_id']}}} for r in rows]

def rag_renders(study,entries,scan=False):
 """(kept, dropped ids): the RAG rows whose submit_answer target round-trips through the reader tokenizer, checked with
 material_items' own rendering and assertion, so one unrenderable answer is excluded (and counted) instead of aborting."""
 from activation.ac_model.ac_model_utils import assistant_target_ids
 from activation.dataset.loaders.trajectory_utils import submit_answer_definition
 from run_qa import answer_positions
 tools=[submit_answer_definition()];kept=[];dropped=[]
 for entry in entries:
  answer=rag_answer(entry[1])
  output=study.dialect.rendering.assistant_message('',[{'id':'answer','name':'submit_answer','arguments':{'answer':answer}}])
  try:answer_positions(study.tokenizer,assistant_target_ids(study.tokenizer,output,tools),answer);kept.append(entry)
  except (AssertionError,ValueError) if scan else AssertionError:dropped.append(row_identity('rag_qa',entry[1]))
 return kept,dropped

def condition_rag(item,row,split,mode=None):
 """Native matched-BM25 RAG item → the mixture's encoder view (seeded unless `mode` or RAG_VIEW is forced) and ratio (RATIO or sampled)."""
 from activation.ac_model.ac_model_utils import direct_parts,qa_encoder_messages
 from run_qa import AC
 mode=mode or RAG_VIEW or conditioning(item.item_id);rng=random.Random(f'{SEED}:rag:{item.item_id}');ratio=RATIO or (1/16 if split=='held' else rng.uniform(1/20,1/10))
 for part in direct_parts(item.ac_messages):
  evidence=[{'role':'user','content':item.info['passages_text']}]
  part['messages']=deepcopy(evidence) if BARE_BLIND and mode=='summary' else qa_encoder_messages(evidence,row['question'],AC,ratio,
   encoder_sees_question=mode!='summary',parent_messages=[] if mode=='parent' else None)
  part['compression_target']=ratio
 item.kind='rag_qa';item.info.update(family='rag_qa',encoder_conditioning=mode,ratio=ratio,source=row['source_family'],
  **({'encoder_input':'bare'} if BARE_BLIND and mode=='summary' else {}))
 return item

def step_seconds(record):
 """Planning estimate of one example's share of a 2×H200 update (fitted on the canonical mixed fixture and the 75M run)."""
 return .50+5.5e-5*(record['reader_tokens']+record['encoder_tokens']/4.5)

def prepare(args):
 """One mixed pass (MIXED_RUN_PLAN.md): every usable training row once, the task panel's held identities excluded.

 --scan builds every item and example exactly as below but records per-item failures and sizes in scan.json (and
 probe_ids.json: the heaviest rows per family plus a seeded sample) instead of committing pickles. --subset keeps only the
 listed rows, --exclude drops listed rows with a recorded reason, --probe orders the packet heaviest first without
 shuffling (the M4 probes). Everything else is the production path.
 """
 from activation.dataset.dataset_caching import freeze_text_collection
 from activation.ac_model import ActivationContextStudyGenerator,ActivationContextTrainer,ActivationContextTrainingConfig
 from run_qa import harness,AC,material_items
 args.output.mkdir(parents=True,exist_ok=True)
 if (args.output/'prepared.json').exists() or (args.output/'scan.json').exists():raise FileExistsError('Prepared run or scan already committed; use a new output')
 if args.task_panel is None:raise ValueError('prepare needs --task-panel: its held.json holds the run\'s only held-out rows')
 held,panel_receipt=load_held(args.task_panel)
 def listed(path):return {f:set(v) for f,v in json.loads(path.read_text())['ids'].items()} if path else None
 subset,exclude=listed(args.subset),listed(args.exclude)
 snapshots={}
 for family in TRAIN_FAMILIES:
  path=args.output/(family+'_snapshot.json')
  snapshots[family]=json.loads(path.read_text()) if path.exists() else freeze_text_collection(args.data/FOLDERS.get(family,family),path)
 if any(panel_receipt['snapshots'].get(f)!=digest(snapshots[f]) for f in QA_FAMILIES):raise ValueError('The task panel was drawn from different RAG/trajectory banks')
 rows,counts,excluded=select_rows(args,snapshots,held,subset)
 retrieval_rows=[r for _,r,_ in (rows if subset is None else select_rows(args,snapshots,held,families=('rag_qa',))[0])['rag_qa']]
 if exclude:
  for family in TRAIN_FAMILIES:
   before=len(rows[family]);rows[family]=[entry for entry in rows[family] if row_identity(family,entry[1]) not in exclude.get(family,())]
   excluded['listed_exclusion']=excluded.get('listed_exclusion',0)+before-len(rows[family])
 global BUILD_CONTEXT
 workers=args.workers or len(os.sched_getaffinity(0))
 if workers>1:
  import torch
  # The fork pool is for a CPU-only allocation (as run_trajectory_tasks' parallel preparation), set up before the first
  # tokenizer: no Rust or OpenMP thread pool is running when the workers fork, and none is oversubscribed after.
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('Parallel preparation needs CUDA_VISIBLE_DEVICES= (a CPU-only job); use --workers 1 otherwise')
  os.environ['TOKENIZERS_PARALLELISM']='false'
 pair=json.loads(args.initial_pair.read_text());h=harness(pair['ac_checkpoint'],pair['reader_checkpoint']);study=ActivationContextStudyGenerator(h,AC)
 if workers>1:torch.set_num_threads(1)   # after harness(), which sets its own count
 failures=[];stages={}
 def guarded(family,ident,build,constituents=None):
  """The production build; under --scan a failing row is recorded (and dropped) instead of aborting the preparation."""
  value,failure=attempt(args.scan,family,ident,build,constituents)
  if failure:failures.append(failure)
  return value
 output=[];locations={};views=Counter()
 # RAG: native matched BM25 evidence, question synthesis already frozen. The distractor pool is every usable training
 # RAG row (a subset packet included) and passage order is seeded per item; every chunk queries at the fetch depth of the
 # whole batch being built, so the chunks retrieve exactly as the one-batch build did.
 rows['rag_qa'],unrenderable=rag_renders(study,rows['rag_qa'],args.scan)
 if unrenderable:excluded['rag_answer_render']=len(unrenderable)
 rag_entries=rows['rag_qa']
 def rag_builder():
  build=material_items(h,{'train':[],'held':[],'retrieval':rag_material(retrieval_rows)},evidence_tokens=args.rag_evidence_tokens,max_passages=8,builder_only=True)[1]
  return build,build.extra_for(rag_material([r for _,r,_ in rag_entries]))
 built=guarded('rag_qa','<material build>',rag_builder) if rag_entries else None
 if built is not None:
  clock=StageClock('rag_qa items',len(rag_entries));done=0
  BUILD_CONTEXT={'rag_build':built[0],'rag_extra':built[1],'scan':args.scan,'entries':rag_entries}
  spans=[(i,min(i+CHUNK,len(rag_entries))) for i in range(0,len(rag_entries),CHUNK)]
  with ForkPool(workers) as pool:
   for span,results in zip(spans,pool.map(rag_chunk,spans)):
    chunk=rag_entries[slice(*span)]
    for (item,failure),(_,row,location) in zip(results,chunk,strict=True):
     if failure:failures.append(failure)
     if item is None:continue
     output.append(item);locations['rag_qa:'+item.item_id]=location;views['rag_qa:'+item.info['encoder_conditioning']]+=1
    done+=len(chunk);clock(done)
  stages['rag_qa_items']=clock.seconds()
 BUILD_CONTEXT={'study':study,'rows':rows}
 for family in TRAIN_FAMILIES[1:]:
  budget={'agent_teacher':args.teacher_tokens,'agent_regular':args.regular_tokens}.get(family)
  clock=StageClock(family+' items',len(rows[family]))
  chunks=[(family,i,min(i+CHUNK,len(rows[family]))) for i in range(0,len(rows[family]),CHUNK)]
  with ForkPool(workers) as pool:
   results=(pair for chunk_results in pool.map(item_chunk,chunks) for pair in chunk_results)
   rebuild=lambda row,family=family:make_item(study,family,row,'train')
   for item,location in budgeted_items(results,rows[family],budget,args.scan,failures,rebuild,clock):
    output.append(item);locations[family+':'+item.item_id]=location
    views[family+':'+item.info.get('encoder_conditioning','none')]+=1
   results.close()
  stages[family+'_items']=clock.seconds()
 BUILD_CONTEXT=None
 # The rows stay alive until the end (as in one process): freeing them before the packing/example forks would leave tens
 # of GB of fragmented free heap that every forked worker allocates into, materializing a private copy of each page it
 # lands on (full mix: 257 GB in packing, ~5 MB/s per example worker); kept, the workers allocate fresh contiguous memory.
 unique_sources=Counter(item.info['family'] for item in output)
 retained=output   # the packed sources too stay alive until the example workers are gone (same reason as the rows)
 clock=StageClock('multi-input packing',1)
 packed=guarded('multi','<packing>',lambda:pack_multi_targets(study,output,fraction=args.multi_target_fraction,split='train',workers=workers))
 output,multi_targets=packed or (output,{'error':'packing failed; see failures'});del packed   # the list is replaced by the kept items below
 clock(1);stages['packing']=clock.seconds()
 actual=Counter(item.info['family'] for item in output);total=len(output)
 for item in output:item.weight=SHARES[item.info['family']]*total/actual[item.info['family']] if SHARES else 1.0
 config=ActivationContextTrainingConfig(loss_kind='sft',learning_rate_ac=5e-5,learning_rate_target_lora=2e-5,examples_per_update=8,num_epochs=args.epochs,completion_samples=0,
  teacher_cache_block_examples=32,logits_chunk_tokens=64,gradient_checkpointing_min_tokens=0,seed=SEED,deterministic=bool(getattr(args,'deterministic',False)),
  schedule_total_updates=math.ceil(total/8)*args.epochs,checkpoint_every_updates=args.checkpoint_every,reporting_interval=args.reporting_interval,
  shuffle_examples=not args.probe)
 trainer=ActivationContextTrainer(h,config)
 examples=[];kept=[];records=[]
 # Freeze native token lists on CPU; GPU allocations reuse them on every attempt. Workers return each item with its
 # example (example.item is the item), so the kept items replace the parent's copies.
 clock=StageClock('examples',len(output));done=0
 BUILD_CONTEXT={'trainer':trainer,'study':study,'scan':args.scan,'output':output}
 chunks=[range(i,min(i+CHUNK,len(output))) for i in range(0,len(output),CHUNK)]
 with ForkPool(workers) as pool:
  for results in pool.map(example_chunk,chunks):
   for value,failure in results:
    if failure:failures.append(failure);continue
    item,example,record=value
    kept.append(item);examples.append(example);records.append(record)
   done+=len(results);clock(done)
 BUILD_CONTEXT=None;del retained;stages['examples']=clock.seconds()
 if len(kept)!=len(output):
  actual=Counter(item.info['family'] for item in kept);total=len(kept)
  config.schedule_total_updates=math.ceil(total/8)*args.epochs
 output=kept
 if args.probe:
  # Heaviest first (no shuffling): the first updates are the worst case for memory, update time and rank imbalance.
  cost={id(e.item):e.execution_cost for e in examples};output.sort(key=lambda item:-cost[id(item)])
 recipe={'seed':SEED,'epochs':args.epochs,'counts':counts,'actual_train':dict(actual),'unique_sources':dict(unique_sources),'multi_targets':multi_targets,'multi_target_fraction':args.multi_target_fraction,'excluded':excluded,'views':dict(views),
  'target_tokens':dict(Counter({family:sum(i.info['target_tokens'] for i in output if i.info['family']==family) for family in TRAIN_FAMILIES})),
  'loss_shares':SHARES or 'natural: every example weighs 1','family_weights':{f:SHARES[f]*total/actual[f] if SHARES else 1.0 for f in actual},'global_batch':8,
  'dropped_regular_root_finish':list(DROPPED_ROOT_FINISH),'conditioning':'7/9 contextual, 2/9 uncontextual among singles (70/20/10 with multi)',
  'held':{'rule':'the task panel\'s drawn held identities only; the banks\' own splits ignored; reserved RAG questions never train',
   'panel':str(args.task_panel.resolve()),'panel_sha256':panel_receipt['sha256'],'held_sha256':panel_receipt['held_sha256']},
  'subset':digest(sorted((f,sorted(v)) for f,v in subset.items())) if subset else None,
  'exclude':digest(sorted((f,sorted(v)) for f,v in exclude.items())) if exclude else None,'probe':bool(args.probe),
  'rag_evidence_tokens':args.rag_evidence_tokens,'rag_retrieval_rows':len(retrieval_rows),'rag_answer_render_excluded':unrenderable,'task_ratios':[1/20,1/10],
  'initial_pair':pair,'snapshots':{f:digest(s) for f,s in snapshots.items()},'training_config':asdict(config)}
 if args.scan:
  write_scan(args,study,recipe,output,examples,records,failures,preparation={'workers':workers,'stage_seconds':stages});return
 write(args.output/'recipe.json',recipe);write(args.output/'selection.json',locations)
 bundle={'train':output,'held':[]}
 temporary=args.output/'items.pkl.tmp'
 with temporary.open('wb') as stream:pickle.dump(bundle,stream,protocol=pickle.HIGHEST_PROTOCOL);stream.flush();os.fsync(stream.fileno())
 temporary.replace(args.output/'items.pkl')
 with (args.output/'items.pkl').open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
 temporary=args.output/'prepared_examples.pkl.tmp'
 with temporary.open('wb') as stream:pickle.dump({**bundle,'examples':examples},stream,protocol=pickle.HIGHEST_PROTOCOL);stream.flush();os.fsync(stream.fileno())
 temporary.replace(args.output/'prepared_examples.pkl')
 with (args.output/'prepared_examples.pkl').open('rb') as stream:prepared_checksum=hashlib.file_digest(stream,'sha256').hexdigest()
 write(args.output/'prepared.json',{'recipe_sha256':digest(recipe),'items_sha256':checksum,'items_bytes':(args.output/'items.pkl').stat().st_size,
  'prepared_examples_sha256':prepared_checksum,
  'validated_capacity':{'reader_tokens':72000,'encoder_tokens':65536},
  'training_examples':total,'held_examples':0,'one_pass':True,'preparation':{'workers':workers,'stage_seconds':stages},'quality_filter':'Finished submitted teacher runs; no lexical F1 threshold; synthetic targets are not judge-validated.'})
 print(json.dumps(recipe),flush=True)

def write_scan(args,study,recipe,items,examples,records,failures,preparation=None):
 """scan.json: sizes, failures, hours and memory for the whole mix; probe_ids.json: the rows the GPU probe stresses."""
 import statistics
 by={};family_of={id(e.item):e.item.info['family'] for e in examples}
 for example,record in zip(examples,records):by.setdefault(family_of[id(example.item)],[]).append((example,record))
 def spread(values):
  values=sorted(values);n=len(values)
  return {'sum':sum(values),'mean':round(statistics.fmean(values),1),'p50':values[n//2],'p90':values[min(n-1,n*9//10)],'p99':values[min(n-1,n*99//100)],'max':values[-1]} if values else {}
 families={};probe={};sample_bytes={}
 for family,pairs in sorted(by.items()):
  recs=[r for _,r in pairs]
  singles=[(e,r) for e,r in pairs if not e.item.info.get('multi_target_count')]
  rng=random.Random(f'{SEED}:scan_bytes:{family}');chosen=rng.sample(pairs,min(len(pairs),2000))
  per=len(pickle.dumps([(e.item,e) for e,_ in chosen],protocol=pickle.HIGHEST_PROTOCOL))/max(1,len(chosen))
  sample_bytes[family]=per*len(pairs)
  heavy=[e.item.item_id for e,_ in sorted(singles,key=lambda p:-p[0].execution_cost)[:args.probe_heaviest]]
  heavy+=[e.item.item_id for e,r in sorted(singles,key=lambda p:-p[1]['encoder_tokens'])[:max(1,args.probe_heaviest//4)]]
  heavy+=[e.item.item_id for e,r in sorted(singles,key=lambda p:-p[1]['reader_tokens'])[:max(1,args.probe_heaviest//4)]]
  seeded=[e.item.item_id for e,_ in random.Random(f'{SEED}:probe:{family}').sample(singles,min(len(singles),args.probe_sample))]
  probe[family]=list(dict.fromkeys(heavy+seeded))
  families[family]={'examples':len(pairs),'multi_examples':len(pairs)-len(singles),
   'reader_tokens':spread([r['reader_tokens'] for r in recs]),'encoder_tokens':spread([r['encoder_tokens'] for r in recs]),
   'target_tokens':spread([r['target_tokens'] for r in recs]),'penalty_tokens':spread([r['penalty_tokens'] for r in recs]),
   'estimated_hours_2xH200':round(sum(map(step_seconds,recs))/3600,2),
   'heaviest':[{'id':e.item.item_id,**{k:r[k] for k in ('reader_tokens','encoder_tokens','target_tokens','penalty_tokens')}}
    for e,r in sorted(pairs,key=lambda p:-p[0].execution_cost)[:5]]}
 prepared_bytes=sum(sample_bytes.values())
 write(args.output/'probe_ids.json',{'ids':probe,'rule':f'per family: the {args.probe_heaviest} costliest singles, the costliest by encoder and by reader tokens, and {args.probe_sample} seeded singles'})
 write(args.output/'scan.json',{'recipe_sha256':digest(recipe),'recipe':{k:recipe[k] for k in ('counts','actual_train','unique_sources','multi_targets','excluded','views','target_tokens','held','conditioning')},
  'failures':failures,'failure_counts':dict(Counter(f['family'] for f in failures)),'families':families,
  'training_examples':len(examples),'updates':math.ceil(len(examples)/8),
  'estimated_training_hours_2xH200':round(sum(map(step_seconds,records))/3600,2),
  'estimated_prepared_examples_bytes':int(prepared_bytes),
  'launch_ram_gib_2_gpus':math.ceil(2*(32*2**30+3*prepared_bytes)/2**30),
  'preparation':preparation,
  'note':'hours from the fitted step-time model (not measured); bytes from pickling a 2,000-example sample per family'})
 print(json.dumps({'scan':str(args.output/'scan.json'),'examples':len(examples),'failures':len(failures)}),flush=True)

def file_sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def dump(path,value):
 temporary=path.with_name(path.name+'.tmp')
 with temporary.open('wb') as stream:pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL);stream.flush();os.fsync(stream.fileno())
 temporary.replace(path)

def draw_held(args,snapshots,reserved):
 """The held questions: the first `per_group` usable RAG questions (one row per question text, reserved ones skipped) and
 trajectory Q/A rows in seeded order. The banks' own split fields and hash rules are ignored; these never train."""
 from activation.dataset.dataset_caching import frozen_text_rows
 held={}
 for family in QA_FAMILIES:
  def ranked():
   for row,_ in frozen_text_rows(args.data/family,snapshots[family]):
    if family=='rag_qa' and normalized(row['question']) in reserved:continue
    yield digest((SEED,'task_panel',row_identity(family,row))),row_identity(family,row),row
  rows=[];keys=set()
  for _,_,row in heapq.nsmallest(4*args.per_group,ranked(),key=lambda entry:(entry[0],entry[1])):
   key=normalized(row['question']) if family=='rag_qa' else row_identity(family,row)
   if key not in keys:keys.add(key);rows.append(row)
   if len(rows)==args.per_group:break
  if len(rows)<args.per_group:raise ValueError(f'{family}: only {len(rows)} usable held rows')
  held[family]=rows
 return held

def build_panel(args):
 """Evaluation-only task panel of the mixed run (MIXED_RUN_PLAN.md): the drawn held questions under both Q/A views.

 Fundamentals only: flat single RAG and trajectory Q/A, contextual and uncontextual, the same questions under each view.
 held.json records the drawn identities, the run's only held-out rows; prepare excludes exactly those. The other
 families show through the report's training loss by group and the fixed learning sample.
 """
 from activation.dataset.dataset_caching import freeze_text_collection,frozen_text_rows
 from activation.ac_model import ActivationContextStudyGenerator,ActivationContextTrainer,ActivationContextTrainingConfig
 from activation.ac_model.ac_model_utils import validate_part_capacity
 from run_qa import harness,AC,material_items
 from run_trajectory_tasks import compact_example
 out=args.output;out.mkdir(parents=True,exist_ok=True)
 if (out/'panel.json').exists():raise FileExistsError('Task panel already committed; use a new output')
 snapshots={}
 for family in QA_FAMILIES:
  path=out/(family+'_snapshot.json')
  snapshots[family]=json.loads(path.read_text()) if path.exists() else freeze_text_collection(args.data/family,path)
 reserved=reserved_questions(snapshots)
 held=draw_held(args,snapshots,reserved)
 write(out/'held.json',{'rule':'seeded draw over usable rows; RAG by normalized question text (reserved skipped), trajectory Q/A by row id; the banks\' own splits ignored',
  'rag_qa':{'ids':[row['id'] for row in held['rag_qa']],'questions':[normalized(row['question']) for row in held['rag_qa']]},
  'trajectory_qa':{'ids':[row_identity('trajectory_qa',row) for row in held['trajectory_qa']]}})
 # The training-scale BM25 distractor pool: every usable RAG row.
 retrieval=[row for row,_ in frozen_text_rows(args.data/'rag_qa',snapshots['rag_qa']) if normalized(row['question']) not in reserved]
 pair=json.loads(args.initial_pair.read_text());h=harness(pair['ac_checkpoint'],pair['reader_checkpoint']);study=ActivationContextStudyGenerator(h,AC)
 builder=ActivationContextTrainer(h,ActivationContextTrainingConfig(loss_kind='sft',max_example_tokens=1_000_000))
 panel=[]
 def add(group,items):
  for item in items:
   item.item_id=f'panel:{group}:{item.item_id}';item.info.update(panel_group=group,panel=True);panel.append(item)
 rows=held['rag_qa']
 _,_,base=material_items(h,{'train':rag_material(rows[:1]),'held':rag_material(rows),'retrieval':rag_material(retrieval)},
  evidence_tokens=args.rag_evidence_tokens,max_passages=8)
 del retrieval
 for mode in VIEWS:add('rag_qa/'+MODES[mode],[condition_rag(deepcopy(item),row,'held',mode) for item,row in zip(base,rows,strict=True)])
 for mode in VIEWS:add('trajectory_qa/'+MODES[mode],[make_item(study,'trajectory_qa',row,'held',mode=mode) for row in held['trajectory_qa']])
 if len({item.item_id for item in panel})!=len(panel):raise ValueError('Duplicate task-panel item ids')
 examples=[]
 for item in panel:
  example=builder.build_example(study.ac_model,item)
  if reader_tokens(builder,study,item)>72000:raise ValueError('Panel reader history exceeds 72000 tokens')
  for request in example.part_requests:validate_part_capacity(study.ac_model,request,65536)
  if nested_parts(item):raise ValueError('Panel item nests an AC part (depth-1 rule)')
  item.info.update(target_tokens=len(example.target_ids),reader_tokens=len(example.student_ids))
  examples.append(compact_example(example))
 dump(out/'panel.pkl',{'items':panel,'examples':examples})
 groups=Counter(item.info['panel_group'] for item in panel)
 write(out/'panel.json',{'sha256':file_sha(out/'panel.pkl'),'held_sha256':file_sha(out/'held.json'),'items':len(panel),'groups':dict(sorted(groups.items())),
  'target_tokens':sum(item.info['target_tokens'] for item in panel),'reader_tokens':sum(item.info['reader_tokens'] for item in panel),
  'per_group':args.per_group,'snapshots':{f:digest(s) for f,s in snapshots.items()},'initial_pair':pair,'seed':SEED,'ratio':1/16,
  'rule':'held.json rows only, seeded order; each Q/A view shows the same questions; RAG distractors from every usable RAG row'})
 print(json.dumps({'panel':dict(groups)}),flush=True)

def nested_parts(item)->bool:
 """An AC part of the item carries another AC part inside (nested material, or a multi-input group with such a constituent)."""
 from activation.ac_model.ac_model_utils import direct_parts
 return any(direct_parts(part.get('messages') or []) for part in direct_parts(item.ac_messages))

def derive(args):
 """Stratified one-pass subset of a prepared reconstruction continuation packet, sized to a lifetime target-token total.

 Strata are (layout or multi, source); each keeps the same fraction of its target tokens in seeded order. Held items,
 diagnostics, prepared native examples, the optimizer continuation and the schedule shape are carried over unchanged
 except for the shorter horizon (--flat-only also drops nested diagnostic views).
 """
 src=args.source.resolve();out=args.output;out.mkdir(parents=True,exist_ok=True)
 if (out/'prepared.json').exists():raise FileExistsError('Derived packet already committed; use train or a new output')
 ready=json.loads((src/'prepared.json').read_text());recipe=json.loads((src/'recipe.json').read_text())
 if digest(recipe)!=ready['recipe_sha256']:raise ValueError('Source recipe changed')
 if file_sha(src/'prepared_examples.pkl')!=ready['prepared_examples_sha256']:raise ValueError('Source prepared examples changed')
 old_pair=recipe['initial_pair'];pair_root=args.pair_root.resolve()
 if file_sha(pair_root/'ac'/'optimizer.pt')!=old_pair['optimizer_sha256']:raise ValueError('Relocated pair optimizer differs from the packet receipt')
 with (src/'prepared_examples.pkl').open('rb') as stream:prepared=pickle.load(stream)
 train,held,diagnostics=prepared['train'],prepared['held'],prepared['held_diagnostics']
 examples={id(example.item):example for example in prepared['examples']}
 if any(id(item) not in examples for item in train+held+diagnostics):raise ValueError('Source item/example identity differs')
 accounting=recipe['accounting'];presented=accounting['already_presented_flat_target_tokens']
 excluded=None
 if args.flat_only:
  # No nesting (09-26 design: every AC part holds flat text, depth 1): drop nested items and multi-input groups with a nested constituent.
  keep=[item for item in train if not nested_parts(item)];dropped=[item for item in train if nested_parts(item)]
  excluded={'examples':len(dropped),'target_tokens':sum(sum(map(len,item.assistant_token_ids)) for item in dropped),
   'by_layout':dict(Counter('multi' if item.info.get('multi_target_count') else 'flat' for item in dropped))}
  train=keep;excluded['held_diagnostics']=sum(map(nested_parts,diagnostics));diagnostics=[item for item in diagnostics if not nested_parts(item)]
 tokens=[sum(map(len,item.assistant_token_ids)) for item in train];total=sum(tokens);need=args.lifetime_target_tokens-presented
 if not 0<need<=total:raise ValueError(f'Lifetime target {args.lifetime_target_tokens} is outside presented {presented} + packet {total}')
 strata={}
 for index,item in enumerate(train):
  key='multi' if item.info.get('multi_target_count') else 'flat'
  strata.setdefault((key,str(item.info.get('source'))),[]).append(index)
 chosen=[];receipt={}
 for (layout,source),indexes in sorted(strata.items()):
  budget=need*sum(tokens[i] for i in indexes)/total;kept=0;count=0
  for index in sorted(indexes,key=lambda i:digest((SEED,'derive',train[i].item_id))):
   if kept+tokens[index]/2>budget:break
   chosen.append(index);kept+=tokens[index];count+=1
  receipt[f'{layout}/{source}']={'examples':count,'of':len(indexes),'target_tokens':kept,'budget':round(budget)}
 chosen.sort();subset=[train[i] for i in chosen];kept=sum(tokens[i] for i in chosen)
 offset=recipe['updates_offset'];updates=math.ceil(len(subset)/8);horizon=offset+updates
 config=dict(recipe['training_config'],schedule_total_updates=horizon)
 base=recipe['training_config'];earlier=recipe['clock']['original_schedule_total_updates']
 from activation.agent_training.agent_training_utils import lr_factor
 cooldown={k:base[k] for k in ('cooldown_updates','cooldown_shape') if k in base}   # a cooldown source keeps its decay length; replanned, the decay ends at the new horizon
 if base['lr_schedule']=='cooldown':
  if args.continuous_lr:raise SystemExit('--continuous_lr anchors a cosine; a cooldown source is replanned (its decay ends at the new horizon)')
  if (base.get('cooldown_updates') or 0)>horizon-base['warmup_updates']:raise SystemExit(f"cooldown_updates {base.get('cooldown_updates')} does not fit the continuation horizon {horizon} after the {base['warmup_updates']}-update warm-up")
 at_resume=lr_factor(offset,base['warmup_updates'],total_updates=earlier,schedule=base['lr_schedule'],final_fraction=base['lr_final_fraction'],**cooldown)
 replanned=lr_factor(offset,base['warmup_updates'],total_updates=horizon,schedule=base['lr_schedule'],final_fraction=base['lr_final_fraction'],**cooldown)
 if args.continuous_lr:config.update(schedule_anchor_update=offset,schedule_anchor_factor=at_resume)  # the cosine carries on from the paused rate
 lr={'mode':'continuous' if args.continuous_lr else 'replanned','earlier_schedule_total_updates':earlier,'factor_before_resume':at_resume,
  'factor_at_resume':at_resume if args.continuous_lr else replanned,'final_fraction':base['lr_final_fraction'],'schedule_total_updates':horizon}
 pair={**{k:v for k,v in old_pair.items() if k not in ('ac_checkpoint','reader_checkpoint')},
  'ac_checkpoint':str(pair_root/'ac'),'reader_checkpoint':str(pair_root/'reader'),'relocated_from':{k:old_pair[k] for k in ('ac_checkpoint','reader_checkpoint')}}
 layouts=Counter('multi' if item.info.get('multi_target_count') else 'flat' for item in subset)
 by_layout=Counter()
 for index in chosen:by_layout['multi' if train[index].info.get('multi_target_count') else 'flat']+=tokens[index]
 sources=sum(item.info.get('multi_target_count',1) for item in subset)
 derived={'lifetime_target_tokens_requested':args.lifetime_target_tokens,'lifetime_target_tokens':presented+kept,
  'phase_target_tokens':kept,'phase_target_tokens_requested':need,'phase_examples':len(subset),'phase_source_items':sources,
  'packet_examples':len(train),'packet_target_tokens':total,'kept_fraction':kept/total,'examples_by_layout':dict(layouts),
  'target_tokens_by_layout':dict(by_layout),'learning_rate':lr,'flat_only':bool(args.flat_only),'excluded_nested':excluded,'strata':receipt,'rule':'per (layout|multi, source): seeded order, stop before the stratum budget is exceeded by more than half an item'}
 recipe_data={**recipe,'phase':f'reconstruction continuation to {args.lifetime_target_tokens/1e6:g}M lifetime targets','initial_pair':pair,'training_config':config,
  'counts':{'reconstruction':{'selected_train':len(subset),'selected_held':len(held)}},'actual_train':{'reconstruction':len(subset)},
  'unique_sources':{'reconstruction':sources},'target_tokens':{'reconstruction':kept},
  'multi_targets':{'train':{'reconstruction':{'training_examples':len(subset),'multi_target_examples':layouts['multi']}}},
  'clock':{**recipe['clock'],'phase_updates':updates,'schedule_total_updates':horizon},
  'accounting':{**accounting,'derived':derived},
  'derived_from':{'packet':str(src),'recipe_sha256':ready['recipe_sha256'],'prepared_examples_sha256':ready['prepared_examples_sha256']}}
 identity={**prepared['identity'],'derived':derived,'derived_from':recipe_data['derived_from']}
 write(out/'selection.json',{'source_packet':str(src),'ids':[item.item_id for item in subset]})
 write(out/'recipe.json',recipe_data)
 dump(out/'items.pkl',{'train':subset,'held':held,'held_diagnostics':diagnostics})
 dump(out/'prepared_examples.pkl',{'identity':identity,'train':subset,'held':held,'held_diagnostics':diagnostics,
  'examples':[examples[id(item)] for item in subset+held+diagnostics]})
 write(out/'prepared.json',{'recipe_sha256':digest(recipe_data),'items_sha256':file_sha(out/'items.pkl'),'items_bytes':(out/'items.pkl').stat().st_size,
  'prepared_examples_sha256':file_sha(out/'prepared_examples.pkl'),'validated_capacity':{'reader_tokens':72000,'encoder_tokens':65536},
  'training_examples':len(subset),'held_examples':len(held),'one_pass':True,'requires_continuation_clock':recipe_data['clock'],
  'quality_filter':'Stratified subset of the continuation packet; no truncation or repetition.'})
 print(json.dumps({'output':str(out),'derived':{k:v for k,v in derived.items() if k!='strata'},'clock':recipe_data['clock']}),flush=True)


def snapshot_percent(step,steps,every):
 """The epoch percentage (a multiple of `every`) a report at `step` of `steps` has crossed, or None: before the first
 multiple, at the last step (the epoch checkpoint covers it) or when disabled. Repeats collapse into the one folder."""
 if not every or step>=steps or 100*step<steps*every:return None
 return every*(100*step//(steps*every))

def train(args):
 import torch
 from activation.ac_model import ActivationContextTrainer,ActivationContextTrainingConfig,ActivationContextStudyGenerator,ActivationContextTrainingReporter
 from activation.ac_model.ac_model_utils import DistributedTraining,training_storage_preflight,bucket_training_order
 from run_qa import harness,AC,resume_checkpoint
 attempt_started=time.monotonic()
 os.environ['ACTIVATION_SYNC_ROOT']=str(args.output.resolve())
 torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
 with DistributedTraining(int(os.environ.get('WORLD_SIZE','1'))>1) as parallel:
  locks=[]
  def preflight():
   lock=(args.output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
   receipt=training_storage_preflight(args.output,quota_root=args.quota_root,quota_limit_bytes=int(args.quota_gib*2**30))
   write(args.output/'storage_preflight.json',receipt)
   if (args.output/'PAUSE').exists():raise RuntimeError('PAUSE is present; run remains recoverable and paused')
   allocation=int(os.environ.get('SLURM_MEM_PER_NODE','0'))*2**20
   if allocation:
    data_bytes=max(p.stat().st_size for p in [args.output/'items.pkl',args.output/'prepared_examples.pkl'] if p.exists())
    needed=parallel.world_size*(32*2**30+3*data_bytes)
    if needed>allocation:raise MemoryError(f'Request at least {math.ceil(needed/2**30)}GiB CPU RAM for this frozen data and GPU count')
  parallel.main(preflight)
  ready=json.loads((args.output/'prepared.json').read_text());recipe=json.loads((args.output/'recipe.json').read_text())
  if digest(recipe)!=ready['recipe_sha256']:raise ValueError('Frozen recipe changed')
  with (args.output/'items.pkl').open('rb') as f:
   if hashlib.file_digest(f,'sha256').hexdigest()!=ready['items_sha256']:raise ValueError('Prepared training data changed')
   if 'prepared_examples_sha256' not in ready:f.seek(0);items=pickle.load(f)
  cached={};diagnostic_ids=set()
  if 'prepared_examples_sha256' in ready:
   with (args.output/'prepared_examples.pkl').open('rb') as f:
    if hashlib.file_digest(f,'sha256').hexdigest()!=ready['prepared_examples_sha256']:raise ValueError('Prepared native examples changed')
    f.seek(0);prepared=pickle.load(f)
   items={name:prepared[name] for name in ('train','held')}
   diagnostics=prepared.get('held_diagnostics',[])
   if recipe.get('accounting',{}).get('derived',{}).get('flat_only'):diagnostics=[i for i in diagnostics if not nested_parts(i)]   # nothing nested is trained
   items['held']=items['held']+diagnostics;diagnostic_ids={i.item_id for i in diagnostics}
   cached={id(e.item):e for e in prepared['examples']}
   if any(id(i) not in cached for i in items['train']+items['held']):raise ValueError('Prepared example/item identity differs')
  panel_items=[];panel_sha=None
  excluded_panel=recipe.get('held') if isinstance(recipe.get('held'),dict) else {}
  if not items['held'] and args.task_panel is None and excluded_panel.get('panel'):args.task_panel=Path(excluded_panel['panel'])
  if not items['held'] and args.task_panel is None:raise ValueError('The packet has no held items: train needs its task panel (--task-panel)')
  if args.task_panel:
   # Evaluation-only task panel: appended to held, never trained; one file keeps each group's curve continuous across runs.
   receipt=json.loads((args.task_panel/'panel.json').read_text())
   panel_sha=file_sha(args.task_panel/'panel.pkl')
   if panel_sha!=receipt['sha256']:raise ValueError('Task panel changed')
   if excluded_panel.get('panel_sha256') and (receipt['sha256'],receipt.get('held_sha256'))!=(excluded_panel['panel_sha256'],excluded_panel['held_sha256']):
    raise ValueError('Task panel differs from the one prepare excluded from training (recipe.held)')
   with (args.task_panel/'panel.pkl').open('rb') as f:panel_data=pickle.load(f)
   panel_items=panel_data['items'];cached.update({id(e.item):e for e in panel_data['examples']})
   if {i.item_id for i in panel_items}&{i.item_id for i in items['held']}:raise ValueError('Task panel item ids collide with held items')
   items['held']=items['held']+panel_items
   parallel.main(write,args.output/'task_panel.json',{'path':str(args.task_panel.resolve()),**{k:receipt[k] for k in ('sha256','items','groups')}})
  # Held groups: the run's own held items as held/<group>, plus the task panel's groups (comparable across runs).
  # Headline held NLL: the run's own held items (no diagnostics; by-kind losses add them). A packet without held items
  # (the mixed run: the panel holds its only held-out rows) takes its headline over the task panel.
  panel_ids={i.item_id for i in panel_items};overlap={i.item_id for i in panel_items if i.info.get('reconstruction_overlap')}
  headline_ids={i.item_id for i in items['held'] if i.item_id not in panel_ids} or panel_ids
  def headline_rows(rows):return [r for r in rows if base_id(r['item_id']) in headline_ids]
  held_groups={i.item_id:'held/'+panel_group(i) for i in items['held'] if i.item_id not in panel_ids}
  held_groups.update({i.item_id:i.info['panel_group'] for i in panel_items})
  train_groups={i.item_id:panel_group(i) for i in items['train']}
  class PreparedTrainer(ActivationContextTrainer):
   def build_example(self,model,item,reference=False):
    if reference or id(item) not in cached:return super().build_example(model,item,reference)
    example=cached[id(item)]
    # (or the full-context match term, full_context_match_weight > 0: the items that carry it need the same view; hasattr: an older package has no such option)
    if (self.config.loss_kind=='kl' or (hasattr(self,'_carries_match') and self._carries_match(item))) and not len(example.teacher_ids):
     # Prepared examples are built SFT-style (no teacher view, one shared cache key): build the item the trainer's way and graft the
     # teacher view on, keeping the validated student history (the reader's history does not depend on the loss kind).
     from array import array
     fresh=super().build_example(model,item)
     if (list(fresh.student_ids),list(fresh.student_positions),list(fresh.spans))!=(list(example.student_ids),list(example.student_positions),list(example.spans)):
      raise ValueError(f'{item.item_id}: the prepared student history differs from a fresh build; the packet does not match this trainer')
     example.teacher_ids=array('i',fresh.teacher_ids);example.teacher_positions=array('i',fresh.teacher_positions);example.teacher_cache_key=fresh.teacher_cache_key
    return example
   def _check_capacity(self,model,examples):
    bounds=ready.get('validated_capacity')
    def capacity(loaded):
     config=loaded.model.config
     config=config.get_text_config() if hasattr(config,'get_text_config') else config
     return getattr(config,'max_position_embeddings',None)
    target,side=capacity(model.target),capacity(model.side)
    if bounds and target and side and target>=bounds['reader_tokens'] and side>=bounds['encoder_tokens']:
     examples=[example for example in examples if cached.get(id(example.item)) is not example]
    return super()._check_capacity(model,examples)
  config=ActivationContextTrainingConfig(**recipe['training_config']);config.distributed=True
  teacher='current' if config.loss_kind=='kl' else 'stored'   # KL evaluations of items train() never stored (the initial-control views, the learning sample) take the reader as it is then
  if getattr(args,'deterministic',None) is not None:config.deterministic=args.deterministic
  if args.execution_policy:config.execution_policy=json.loads(args.execution_policy.read_text())
  config.max_duration_seconds=args.seconds
  resume=None
  if (args.output/'AC_MODELS'/AC/'partial_epoch.json').exists() or (args.output/'AC_MODELS'/AC/'completed_epoch.json').exists():
   def resume_metadata():
    value=resume_checkpoint(args.output,math.ceil(len(items['train'])/8),recipe['epochs'],updates_offset=recipe.get('updates_offset',0))
    value.pop('optimizer_state',None)
    return value
   resume=parallel.main(resume_metadata)
   if not resume['remaining_epochs']:
    parallel.main(write,args.output/'run_status.json',{'stop_reason':'complete','examples':len(items['train']),
     'checkpoint':resume['ac_path'],'recovered_from_completed_marker':True})
    def finish_report():
     path=args.output/'report_data.json'
     if path.exists():
      row=json.loads(path.read_text());row.update(phase='complete',epoch=recipe['epochs'],examples_completed=len(items['train']),sources_completed=sum(recipe.get('unique_sources',recipe['actual_train']).values()));write(path,row)
    parallel.main(finish_report);return
   config.resume_optimizer=True;config.resume_skip_steps=resume['skip_steps'];config.num_epochs=resume['remaining_epochs']
  def record_attempt():
   with (args.output/'training_attempts.jsonl').open('a') as stream:
    stream.write(json.dumps({'utc':time.strftime('%FT%TZ',time.gmtime()),'config':asdict(config),
     'world_size':parallel.world_size,'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')})+'\n')
  parallel.main(record_attempt)
  pair=recipe['initial_pair'];ac_shape=recipe.get('ac_model') or {}   # the recipe's AC shape beyond the fixed harness: summary_layout, row_head_linear_skip, bin_density_weight (a fresh pair has no checkpoint to take them from)
  h=harness(resume['ac_path'] if resume else pair['ac_checkpoint'],resume['reader_path'] if resume else pair['reader_checkpoint'],summary_layout=ac_shape.get('summary_layout'),row_head_linear_skip=ac_shape.get('row_head_linear_skip'),
   bin_density_weight=ac_shape.get('bin_density_weight'))
  trainer=PreparedTrainer(h,config);study=ActivationContextStudyGenerator(h,AC)
  if not resume:study.ac_model.epoch_number=0
  report_state={'title':recipe.get('phase','Mixed channel training'),'phase':'initial reader references','epoch':1,'epochs':recipe['epochs'],'examples_completed':0,
   'prior_training':{'source_items':recipe.get('accounting',{}).get('already_presented_passages',0),'target_tokens':recipe.get('accounting',{}).get('already_presented_flat_target_tokens',0)},
   'examples_per_epoch':len(items['train']),'sources_per_epoch':sum(recipe.get('unique_sources',recipe['actual_train']).values()),'counts':recipe.get('unique_sources',recipe['actual_train']),'target_tokens':recipe['target_tokens'],
   'panels':json.loads((args.output/'panels.json').read_text()) if (args.output/'panels.json').exists() else [],'fixed':{},'train_nll':None}
  def exposure(progress):
   if progress.epoch==0:return (resume['first_epoch'],min(len(items['train']),resume['skip_steps']*8)) if resume else (1,0)
   return progress.epoch_number,min(len(items['train']),progress.step*8)
  source_prefixes={};clock={}
  def publish_report(**values):
   report_state.update(values)
   epoch=max(1,report_state['epoch'])
   if report_state['examples_completed']==0:
    report_state['sources_completed']=0
   elif epoch not in source_prefixes:
    order=list(range(len(items['train'])))
    if config.shuffle_examples:random.Random(config.seed+epoch).shuffle(order)
    order=bucket_training_order(order,trainer._training_costs if config.cost_bucket_examples else [],8,config.cost_bucket_examples,config.seed+epoch)
    prefix=[0]
    for index in order:prefix.append(prefix[-1]+items['train'][index].info.get('multi_target_count',1))
    source_prefixes[epoch]=prefix
   if report_state['examples_completed']:
    report_state['sources_completed']=source_prefixes[epoch][report_state['examples_completed']]
   report_state.update(values,updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()))
   write(args.output/'report_data.json',report_state)
  class Reporter(ActivationContextTrainingReporter):
   def report_phase(self,progress,phase):
    super().report_phase(progress,phase)
    # The held (task panel) evaluation starts at the first 'reporting' phase; report_eval repeats the phase after it.
    if phase=='reporting':clock.setdefault('held',time.monotonic())
    publish_report(phase=phase,epoch=exposure(progress)[0],examples_completed=exposure(progress)[1])
   def report_step(self,progress,stats):
    super().report_step(progress,stats)
    recent={}
    for record in reversed(stats.example_records[-4000:]):   # the last <=100 training objectives per group
     group=train_groups.get(record.get('item_id'))
     if group is not None and len(recent.setdefault(group,[]))<100:recent[group].append(record['loss'])
    publish_report(phase='training',epoch=exposure(progress)[0],examples_completed=exposure(progress)[1],
     train_nll=stats.loss[-1],training_nll_by_kind={k:sum(v[-100:])/len(v[-100:]) for k,v in stats.loss_by_kind.items() if v},
     training_nll_by_group={g:{'loss':sum(v)/len(v),'examples':len(v)} for g,v in sorted(recent.items())})
  def init_report():
   import shutil
   template=Path(__file__).with_name('channel_mixture_report.tressoir.html')
   target=args.output/'report.tressoir.html'
   if not target.exists():shutil.copyfile(template,target)
   publish_report()
  parallel.main(init_report)
  reporter=Reporter(str(args.output/'report/diagnostics'),title='Mixed channel training diagnostics',page_rewrite_seconds=math.inf) if parallel.is_main else None
  trainer._training_deadline=attempt_started+args.seconds
  signal.signal(signal.SIGUSR1,lambda *_:setattr(trainer,'_training_deadline',time.monotonic()))
  original_deadline_check=trainer._check_training_deadline
  def deadline_check():
   if any((args.output/name).exists() for name in ('PAUSE','PAUSE_ALLOCATION')):trainer._training_deadline=time.monotonic()
   return original_deadline_check()
  trainer._check_training_deadline=deadline_check
  identity={'recipe_sha256':ready['recipe_sha256'],'panel_sha256':panel_sha}
  def frozen_controls():
   path=args.output/'fixed_references.json'
   if path.exists():
    value=json.loads(path.read_text())
    if value.get('identity',identity)!=identity:raise ValueError('Frozen controls were computed for another recipe/task panel')
    return value
   if resume:raise ValueError('Resumed run has no frozen initial-reader references')
   return None
  def reference(name):
   # One reference per main-rank call: the other ranks wait in a collective, which must stay under the NCCL timeout.
   partial=args.output/'fixed_references.partial.json'
   previous=json.loads(partial.read_text()) if partial.exists() else {}
   if previous and previous.get('identity',{'recipe_sha256':previous.get('recipe_sha256')})!=identity:raise ValueError('Initial controls recipe/task panel changed')
   result=previous.get('references',{})
   if name not in result:
    view=[view for item in items['held'] for view in ([item] if name=='full_context' and item.in_context_messages is None else study.transform_for_eval_reference([item],name))]
    summary=trainer.eval(AC,view,teacher=teacher)
    headline,by_kind=headline_summary(headline_rows(summary['per_item']),diagnostic_ids)
    result[name]={'loss':headline['loss'],'by_kind':by_kind,'by_group':group_summary(summary['per_item'],held_groups,overlap)}
    write(partial,{'identity':identity,'references':result})
   return result
  def finish_controls(result):
   value={**result,'identity':identity};write(args.output/'fixed_references.json',value);return value
  from activation.ac_model.ac_model_training import _TrainingTimeLimit
  try:
   fixed=parallel.main(frozen_controls)
   if fixed is None:
    for name in ['no_context','full_context']:result=parallel.main(reference,name)
    fixed=parallel.main(finish_controls,result)
  except _TrainingTimeLimit:
   parallel.main(write,args.output/'run_status.json',{'stop_reason':'duration limit; no updates','phase':'initial controls','examples':len(items['train'])})
   parallel.main(publish_report,phase='paused during initial controls');return
  remaining=parallel.main(lambda:0 if any((args.output/name).exists() for name in ('PAUSE','PAUSE_ALLOCATION')) else trainer._training_deadline-time.monotonic())
  if remaining<=0:
   parallel.main(write,args.output/'run_status.json',{'stop_reason':'duration limit; no updates','phase':'initial controls','examples':len(items['train'])})
   parallel.main(publish_report,phase='paused before training');return
  config.max_duration_seconds=remaining
  parallel.main(publish_report,fixed=fixed)
  learning=[];by_group={}
  for item in items['train']:by_group.setdefault(train_groups[item.item_id],[]).append(item)
  for name,group in sorted(by_group.items()):   # seeded sample: packets are length-sorted
   learning.extend(random.Random(f'{SEED}:learning:{name}').sample(group,min(len(group),max(1,args.learning_items//len(by_group)))))
  def snapshot(progress):
   """Evaluation snapshot at each --snapshot-percent of the epoch: AC modules + reader LoRA, no optimizer, loadable with
   run_qa.harness(ac, reader). Kept apart from the resumable partials (which prune each other); never used by training."""
   from activation.ac_model.ac_model_utils import check_checkpoint_space
   # A resumed attempt's baseline panel stands at the resume step (its epoch fields are the baseline's): a stop exactly on
   # a boundary skipped that boundary's report, and this is where its snapshot is taken instead.
   epoch=exposure(progress)[0];step=progress.step if progress.epoch else (resume['skip_steps'] if resume else 0)
   percent=snapshot_percent(step,progress.steps,args.snapshot_percent)
   if percent is None:return
   steps=progress.steps
   folder=args.output/'SNAPSHOTS'/f'epoch_{epoch:03d}_{percent:03d}pct';receipt=folder/'snapshot.json'
   if receipt.exists():return                          # taken before a resume
   import shutil
   if folder.exists():shutil.rmtree(folder)            # an interrupted save
   manager=h.module_manager;model=manager.get_ac_model(AC);lora=model.config.target_model_lora_name
   started=time.monotonic()
   try:
    folder.mkdir(parents=True)
    check_checkpoint_space((folder/'ac',folder/'reader'),model.trainable_parameters()+manager.lora_parameters(lora),None)
    model.save(str(folder/'ac'),epoch_number=epoch-1)
    manager.save_lora(model.target.model_config.model_name,lora,str(folder/'reader'))
   except BaseException:
    shutil.rmtree(folder,ignore_errors=True);raise
   write(receipt,{'percent':percent,'epoch':epoch,'step':step,'steps':steps,'global_update':progress.global_step,'resume_baseline':not progress.epoch,
    'examples_presented':exposure(progress)[1],'ac_checkpoint':str(folder/'ac'),'reader_checkpoint':str(folder/'reader'),
    'save_seconds':round(time.monotonic()-started,1),'utc':time.strftime('%FT%TZ',time.gmtime()),'recipe_sha256':ready['recipe_sha256']})
  def panel(progress,held):
   phase=time.monotonic();held_seconds=phase-clock.pop('held',phase)
   learned=trainer.eval(AC,learning,release=False,teacher=teacher)
   learning_seconds=time.monotonic()-phase
   snapshot_error=None
   try:snapshot(progress)
   except _TrainingTimeLimit:raise
   except Exception as error:                         # a snapshot is never worth the run: log it and train on
    snapshot_error=f'{type(error).__name__}: {error}'[:600]
    try:
     with (args.output/'snapshot_failures.jsonl').open('a') as stream:
      stream.write(json.dumps({'utc':time.strftime('%FT%TZ',time.gmtime()),'step':progress.step,'steps':progress.steps,'error':snapshot_error})+'\n')
    except OSError:pass
   headline,by_kind=headline_summary(headline_rows(held['per_item']),diagnostic_ids)
   row={'examples presented':exposure(progress)[1],'epoch':exposure(progress)[0],
    'held NLL':headline['loss'],'learning NLL':learned['loss'],'held by kind':by_kind,'learning by kind':learned['by_kind'],
    'held by group':group_summary(held['per_item'],held_groups,overlap),'learning by group':group_summary(learned['per_item'],train_groups),
    'held eval seconds':round(held_seconds,1),**({'snapshot error':snapshot_error} if snapshot_error else {}),'learning eval seconds':round(learning_seconds,1),'callback seconds':round(time.monotonic()-phase,1),'utc':time.strftime('%FT%TZ',time.gmtime())}
   path=args.output/'panels.json';rows=json.loads(path.read_text()) if path.exists() else []
   rows=[r for r in rows if (r['epoch'],r['examples presented'])<(row['epoch'],row['examples presented'])];rows.append(row);write(path,rows);publish_report(panels=rows)
  stats=trainer.train(AC,items['train'],items['held'],reporter=reporter,reporting_callback=panel)
  summary=stats.summarize()
  parallel.main(write,args.output/'result.json',summary)
  def record_result():   # result.json holds the latest attempt only; keep every attempt's peak memory and phase seconds
   with (args.output/'attempt_results.jsonl').open('a') as stream:
    stream.write(json.dumps({'utc':time.strftime('%FT%TZ',time.gmtime()),'resume_skip_steps':config.resume_skip_steps,
     **{k:summary.get(k) for k in ('stop_reason','steps','peak_memory_gb','duration_s','placement_seconds','baseline_seconds')},
     'epochs':[{k:e.get(k) for k in ('training_seconds','reporting_seconds','checkpoint_seconds')} for e in summary.get('epochs',[])]})+'\n')
  parallel.main(record_result)
  parallel.main(publish_report,phase=stats.stop_reason)
  parallel.main(write,args.output/'run_status.json',{'stop_reason':stats.stop_reason,'examples':stats.examples,'checkpoint':stats.checkpoint_path,'fixed_references':fixed})


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','panel','derive','train']);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--data',type=Path);p.add_argument('--initial-pair',type=Path);p.add_argument('--execution-policy',type=Path)
 # Mixed-run defaults (MIXED_RUN_PLAN.md): whole RAG/trajectory/compaction banks and every regular segment, 10k evidence QA, no token caps.
 p.add_argument('--qa-questions',type=int,default=10**9);p.add_argument('--summary-histories',type=int,default=10**9)
 p.add_argument('--teacher-histories',type=int,default=10000);p.add_argument('--teacher-tokens',type=int)
 p.add_argument('--regular-segments',type=int,default=10**9);p.add_argument('--regular-tokens',type=int)
 p.add_argument('--learning-items',type=int,default=100)
 p.add_argument('--rag-evidence-tokens',type=int,default=8192);p.add_argument('--epochs',type=int,default=1)
 p.add_argument('--multi-target-fraction',type=float,default=.1)
 p.add_argument('--deterministic',action=argparse.BooleanOptionalAction,default=None,help='Strict kernels; CUDA requires CUBLAS_WORKSPACE_CONFIG before process startup. Omit to preserve the prepared recipe.')
 p.add_argument('--seconds',type=float,default=20*3600)
 p.add_argument('--quota-root',type=Path);p.add_argument('--quota-gib',type=float)
 p.add_argument('--task-panel',type=Path,help='train: a committed task-panel folder (panel.json + panel.pkl), evaluated with held')
 p.add_argument('--per-group',type=int,default=64,help='panel: held questions per Q/A family (each shown under both views)')
 p.add_argument('--checkpoint-every',type=int,default=250,help='prepare: resumable checkpoint (with optimizer) every this many updates')
 p.add_argument('--reporting-interval',type=float,default=.05,help='prepare: task-panel evaluation every this fraction of the epoch')
 p.add_argument('--snapshot-percent',type=int,default=10,help='train: evaluation snapshot (no optimizer) every this percent of the epoch; 0 disables')
 p.add_argument('--scan',action='store_true',help='prepare: build and measure everything, record failures, commit no pickles')
 p.add_argument('--subset',type=Path,help='prepare: {"ids": {family: [...]}} rows to keep (probe packets)')
 p.add_argument('--exclude',type=Path,help='prepare: {"ids": {family: [...]}} rows to drop, with a recorded count')
 p.add_argument('--probe',action='store_true',help='prepare: heaviest examples first, no shuffling (M4 probes)')
 p.add_argument('--workers',type=int,default=0,help='prepare: fork workers for the item/example builds (0: every CPU of the allocation; 1: one process)')
 p.add_argument('--probe-heaviest',type=int,default=32);p.add_argument('--probe-sample',type=int,default=60)
 p.add_argument('--source',type=Path,help='derive: the prepared reconstruction continuation packet')
 p.add_argument('--pair-root',type=Path,help='derive: the relocated 6459 pair (ac/, reader/)')
 p.add_argument('--lifetime-target-tokens',type=int,default=100_000_000)
 p.add_argument('--flat-only',action='store_true',help='derive: drop items whose AC parts nest another part (no recursion)')
 p.add_argument('--continuous-lr',action='store_true',help='derive: continue the cosine from the paused rate (no jump at the resume)')
 a=p.parse_args()
 if a.phase in ('prepare','panel','derive'):
  if a.phase in ('prepare','panel') and (a.data is None or a.initial_pair is None):p.error(a.phase+' needs --data and --initial-pair')
  if a.phase=='derive' and (a.source is None or a.pair_root is None):p.error('derive needs --source and --pair-root')
  a.output.mkdir(parents=True,exist_ok=True)
  with (a.output/'run.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);{'prepare':prepare,'panel':build_panel,'derive':derive}[a.phase](a)
 else:
  if a.quota_root is None or a.quota_gib is None:p.error('train needs --quota-root and --quota-gib from the launch host')
  train(a)
if __name__=='__main__':main()
