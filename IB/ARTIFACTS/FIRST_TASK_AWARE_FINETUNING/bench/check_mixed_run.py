"""CPU checks of the mixed run's selection rules (MIXED_RUN_PLAN.md), no models or banks needed.

The held draw and training selection over synthetic banks (reserved RAG questions skipped, one row per held question text,
held questions never train through RAG or evidence QA, failed regular roots dropped, duplicate identities rejected); the
held.json receipt check; the 7:2 contextual/uncontextual view draw; the 10% snapshot boundaries against the trainer's 5%
reporting boundaries; the two text-only guards (the runner's regular-agent items, the exporter's segments); and the parallel
preparation (the fork pool's order, inherited context and errors, the per-row failure record, the speculative packing walk
against the one-process walk, with its measurement waste bounded when most groups are rejected).

  cd IB/ARTIFACTS/FIRST_TASK_AWARE_FINETUNING/bench && PYTHONPATH=/workspace /workspace/.venv/bin/python -m pytest check_mixed_run.py -q -p no:cacheprovider
"""
import json,random
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import pytest

import run_channel_mixture as mix
from activation.ac_model.ac_model_training import reporting_boundaries
from activation.common.ac_parts import AC_PART_TYPE

RESERVED = 'What is reserved?'

def rag(i,question=None):
 return {'id':f'rag{i}','question':question or f'Question {i}?','answer':'a','source_family':'s','domain':'d',
  'source':{'units':[{'doc_id':'u','text':'t'}]},'provenance':{'source_request_id':'r'}}

def trajectory(i):return {'id':f'traj{i}','question':'q','answer':'a','history':{'messages':[]}}

def teacher(i,question,finish='submitted'):
 return {'task':{'id':f'teach{i}','question':question},'quality':{'submitted':finish=='submitted'},'run':{'finish_reason':finish}}

def regular(i,finish):return {'segment_id':f'seg{i}','root':{'finish_reason':finish}}

@pytest.fixture
def banks(monkeypatch):
 """Synthetic frozen banks keyed by folder; the snapshot only carries the reserved RAG questions."""
 rows={'rag_qa':[rag(i) for i in range(200)]+[rag(900,RESERVED),rag(901,'question 0?  '),rag(902,'QUESTION 1?')],
  'trajectory_qa':[trajectory(i) for i in range(150)],
  'compaction':[{'id':f'comp{i}'} for i in range(20)],
  'agent_teacher':[teacher(i,f'Question {i}?') for i in range(200)]+[teacher(300,RESERVED),teacher(301,'Unrelated?','max_turns')],
  'agent_teacher_regular':[regular(i,f) for i,f in enumerate(['submitted','max_tool_errors','no_tool_call','max_duration','submitted'])]}
 import activation.dataset.dataset_caching as caching
 monkeypatch.setattr(caching,'frozen_text_rows',lambda root,snapshot:((row,{'ordinal':i}) for i,row in enumerate(rows[Path(root).name])))
 snapshots={f:{'files':[]} for f in mix.TRAIN_FAMILIES}
 snapshots['rag_qa']['controls']={'reserved_questions':{'data':{'questions':[RESERVED]}}}
 return rows,snapshots

def selection_args(**values):
 return Namespace(data=Path('/banks'),per_group=16,**{**dict(qa_questions=10**9,summary_histories=10**9,teacher_histories=10**9,regular_segments=10**9),**values})

def test_held_draw_and_training_selection(banks):
 rows,snapshots=banks;args=selection_args()
 reserved=mix.reserved_questions(snapshots)
 held_rows=mix.draw_held(args,snapshots,reserved)
 questions=[mix.normalized(r['question']) for r in held_rows['rag_qa']]
 assert len(questions)==len(set(questions))==16 and mix.normalized(RESERVED) not in questions
 assert len(held_rows['trajectory_qa'])==16
 assert mix.draw_held(args,snapshots,reserved)==held_rows                  # seeded: the same draw every time
 held={'rag_qa':set(questions),'trajectory_qa':{r['id'] for r in held_rows['trajectory_qa']}}
 selected,counts,excluded=mix.select_rows(args,snapshots,held)
 trained=lambda family:[row for _,row,_ in selected[family]]
 # Every row sharing a held question text (case/space variants included) is held out of RAG and of evidence QA.
 assert not {mix.normalized(r['question']) for r in trained('rag_qa')}&held['rag_qa']
 assert not {mix.normalized(r['task']['question']) for r in trained('agent_teacher')}&held['rag_qa']
 assert counts['rag_qa']['held_rows']>=16 and counts['rag_qa']['held_rows']+counts['rag_qa']['available_train']+1==len(rows['rag_qa'])
 assert not {r['id'] for r in trained('trajectory_qa')}&held['trajectory_qa'] and counts['trajectory_qa']['selected_train']==150-16
 assert excluded['rag_reserved_qa']==1 and excluded['teacher_reserved_qa']==1 and excluded['teacher_unfinished']==1
 assert excluded['teacher_held_qa']==sum(mix.normalized(f'Question {i}?') in held['rag_qa'] for i in range(200))
 assert excluded['regular_failed_root']==2 and {r['segment_id'] for r in trained('agent_regular')}=={'seg0','seg3','seg4'}
 assert counts['compaction']['selected_train']==20
 # The limit keeps a seeded subset; a subset packet keeps only the listed rows.
 limited,_,_=mix.select_rows(selection_args(teacher_histories=10),snapshots,held)
 assert len(limited['agent_teacher'])==10
 subset,_,_=mix.select_rows(args,snapshots,held,{'compaction':{'comp3'},'rag_qa':set()})
 assert [r['id'] for _,r,_ in subset['compaction']]==['comp3'] and subset['rag_qa']==[]

def test_duplicate_identity_and_missing_held_rows_fail(banks):
 rows,snapshots=banks
 held={'rag_qa':{'question 0?'},'trajectory_qa':{'traj0','not-in-bank'}}
 with pytest.raises(ValueError,match='drawn held rows'):mix.select_rows(selection_args(),snapshots,held)
 rows['compaction'].append({'id':'comp0'})
 with pytest.raises(ValueError,match='Duplicate compaction identity'):mix.select_rows(selection_args(),snapshots,{'rag_qa':set(),'trajectory_qa':set()})

def test_held_check_counts_questions_not_rows(banks):
 rows,snapshots=banks                               # 'question 0?' has two rows (rag0, rag901); the second question is absent
 with pytest.raises(ValueError,match='1 of the 2 drawn held rows are missing'):
  mix.select_rows(selection_args(),snapshots,{'rag_qa':{'question 0?','absent?'},'trajectory_qa':{'traj0'}})

def test_held_receipt_is_checked(tmp_path):
 mix.write(tmp_path/'held.json',{'rag_qa':{'ids':['r'],'questions':['q?']},'trajectory_qa':{'ids':['t']}})
 mix.write(tmp_path/'panel.json',{'held_sha256':mix.file_sha(tmp_path/'held.json')})
 held,_=mix.load_held(tmp_path)
 assert held=={'rag_qa':{'q?'},'trajectory_qa':{'t'}}
 (tmp_path/'held.json').write_text(json.dumps({'rag_qa':{'ids':[],'questions':[]},'trajectory_qa':{'ids':[]}}))
 with pytest.raises(ValueError,match='held.json changed'):mix.load_held(tmp_path)

def test_views_are_seven_to_two():
 views=[mix.conditioning(f'id{i}') for i in range(90000)]
 assert set(views)=={'question','summary'}
 assert abs(views.count('question')/len(views)-7/9)<.005

@pytest.mark.parametrize('steps',[34000,33987,1000,40,13])
def test_snapshots_every_ten_percent_of_the_reports(steps):
 taken={}
 for step in reporting_boundaries(steps,.05):
  percent=mix.snapshot_percent(step,steps,10)
  if percent is not None:taken.setdefault(percent,step)
 assert sorted(taken)==list(range(10,100,10))
 for percent,step in taken.items():assert 100*step>=steps*percent            # never labeled ahead of its progress
 assert mix.snapshot_percent(steps,steps,10) is None and mix.snapshot_percent(steps//2,steps,0) is None

def test_regular_agent_item_rejects_ac_parts():
 row={'segment_id':'seg','tools':[],'target_start':1,'root':{},'segment':{},'path':'main',
  'messages':[{'role':'system','content':'s'},{'role':'user','content':[{'type':AC_PART_TYPE,'messages':[]},{'type':'text','text':'go'}]}]}
 with pytest.raises(ValueError,match='carries an AC part'):mix.make_item(Namespace(),'agent_regular',row,'train')

def test_exporter_rejects_ac_parts():
 from activation.bench.canonical_training.export_agent_teacher_segments import segment_rows
 def run(content):
  return {'config_key':'k','agent_config':{'metadata':{'benchmark':'b'},'dataset_task':{'task_id':'t'}},'finish_reason':'submitted',
   'prompt_messages':[{'role':'user','content':content}],'prompt_token_ids':[],'subagent_results':[],
   'trajectory':[{'role':'assistant','messages':[{'role':'assistant','content':'done'}]}]}
 assert len(segment_rows(run('plain text'),lambda ids:[],'release','model'))==1
 with pytest.raises(ValueError,match='AC part in what the agent saw'):
  segment_rows(run([{'type':'text','text':'x'},{'type':AC_PART_TYPE,'messages':[]}]),lambda ids:[],'release','model')

def test_rag_answers_are_nfc():
 # The reader tokenizer normalizes NFC; a decomposed stored answer ('p' + U+0307) must render as the tokens decode.
 decomposed='ṗ = (RṘᵀ)p '
 assert mix.rag_answer({'answer':decomposed})=='ṗ = (RṘᵀ)p'
 assert mix.rag_material([rag(1,'Q?')|{'answer':decomposed}])[0]['example']['gold_answers']==['ṗ = (RṘᵀ)p']

# Parallel preparation.

def context_square(task):return (mix.BUILD_CONTEXT['offset']+task)**2

def context_fail(task):
 if task==5:raise ValueError('row 5')
 return task

def test_fork_pool_keeps_order_context_and_errors(monkeypatch):
 import gc
 monkeypatch.setattr(mix,'BUILD_CONTEXT',{'offset':3})
 serial=list(mix.ForkPool(1).map(context_square,list(range(40))))
 assert serial==[(3+i)**2 for i in range(40)]
 with mix.ForkPool(4) as pool:
  assert list(pool.map(context_square,list(range(40))))==serial
  assert list(pool.map(context_square,[7]))==[100]                                 # one task: in this process
  first=pool.map(context_square,list(range(40)));assert next(first)==9;first.close()   # abandoned early: no hang
  assert list(pool.map(context_square,list(range(5))))==serial[:5]                 # the same workers serve later maps
 assert gc.get_freeze_count()==0
 with pytest.raises(ValueError,match='row 5'),mix.ForkPool(3) as pool:list(pool.map(context_fail,list(range(20))))
 assert gc.get_freeze_count()==0

def die_in_worker(task):
 """A hard crash (as an OOM kill) on task 13, only in a worker: every time, or once (a flag file)."""
 import os
 once=mix.BUILD_CONTEXT.get('once')
 if task==13 and os.getpid()!=mix.BUILD_CONTEXT['parent'] and not (once and once.exists()):
  if once:once.touch()
  os._exit(3)
 return task*2

@pytest.mark.parametrize('delay',[0,.02])   # a slow consumer meets the broken pool at the next submit, not at a result
def test_fork_pool_survives_worker_deaths(monkeypatch,tmp_path,delay):
 import gc,os,time
 def consume(pool,tasks):
  out=[]
  for result in pool.map(die_in_worker,tasks):out.append(result);time.sleep(delay)
  return out
 # dies once: the pool forks again from the first task not yet yielded and stays parallel
 monkeypatch.setattr(mix,'BUILD_CONTEXT',{'parent':os.getpid(),'once':tmp_path/'died'})
 with mix.ForkPool(3) as pool:
  assert consume(pool,list(range(40)))==[2*i for i in range(40)] and pool.deaths==1 and pool.parallel
 # dies every time: after the third death the block finishes in this process, later maps too
 monkeypatch.setattr(mix,'BUILD_CONTEXT',{'parent':os.getpid()})
 with mix.ForkPool(3) as pool:
  assert consume(pool,list(range(40)))==[2*i for i in range(40)] and pool.deaths==3 and not pool.parallel
  assert list(pool.map(die_in_worker,[13,14]))==[26,28]
 assert gc.get_freeze_count()==0

def test_attempt_records_or_raises():
 def boom():raise KeyError('x')
 assert mix.attempt(True,'rag_qa','r1',lambda:4)==(4,None)
 assert mix.attempt(True,'rag_qa','r1',boom,['a','b'])==(None,{'family':'rag_qa','id':'r1','error':"KeyError: 'x'",'constituent_ids':['a','b']})
 with pytest.raises(KeyError):mix.attempt(False,'rag_qa','r1',boom)

@dataclass
class PackItem:
 item_id:str;info:dict;ac_messages:list;ac_start:int=1;in_context_messages:list|None=None;in_context_start:int=1

class PackStudy:
 """Multi-input items whose reader length is their constituents' plus a fixed header."""
 harness=ac_model=None
 def multi_target_item(self,chosen,*,item_id):
  return PackItem(item_id,{'family':chosen[0].info['family'],'multi_target_count':len(chosen),'constituent_ids':[c.item_id for c in chosen],
   'reader':sum(c.info['reader'] for c in chosen)+50},[{'role':'user','content':'m'}])

class PackBuilder:
 calls=0
 def __init__(self,*args):pass
 def build_example(self,ac,item):
  PackBuilder.calls+=1
  return Namespace(student_ids=range(item.info['reader']))

class InlinePool:
 """ForkPool's contract, flagged parallel, in this process: the speculative walk (and its measurement count) without forking."""
 def __init__(self,workers):self.parallel=workers>1
 def __enter__(self):return self
 def __exit__(self,*exc):pass
 def map(self,function,tasks):return map(function,tasks)

def pack_items(n,heavy):
 """n sources per family; `heavy` in 10 read 23,000 tokens, so any group holding one exceeds the 24,576 budget."""
 rng=random.Random(heavy);items=[]
 for family in ('rag_qa','compaction','agent_teacher'):
  for i in range(n):
   items.append(PackItem(f'{family}{i}',{'family':family,'reader':23_000 if rng.randrange(10)<heavy else rng.randint(500,3000)},
    [{'role':'user','content':'x'*rng.randint(1,400)}]))
 return items

def packed(monkeypatch,items,workers,pool=None,study=PackStudy,fraction=.5):
 import activation.ac_model as ac_model
 monkeypatch.setattr(ac_model,'ActivationContextTrainer',PackBuilder)
 monkeypatch.setattr(mix,'ForkPool',pool or mix.ForkPool)
 PackBuilder.calls=0
 output,receipt=mix.pack_multi_targets(study(),items,fraction=fraction,split='train',workers=workers)
 return [(i.item_id,i.info.get('constituent_ids')) for i in output],receipt,PackBuilder.calls

@pytest.mark.parametrize('heavy',[0,3,10])
def test_speculative_packing_is_the_serial_walk(monkeypatch,heavy):
 items=pack_items(300,heavy)
 serial,receipt,serial_calls=packed(monkeypatch,items,1)
 assert (receipt['rag_qa']['capacity_rejections_kept_single']>0)==(heavy>0) and 'agent_teacher' not in receipt
 workers=8;speculative,receipt8,calls=packed(monkeypatch,items,workers,InlinePool)
 assert (speculative,receipt8)==(serial,receipt)
 # Every deviation from the no-rejection path restarts the look-ahead at one key per worker; between deviations the
 # look-ahead doubles, so it measures at most ~3x the walk's keys plus about one round per deviation (never a long plan
 # re-measured per rejection: the all-heavy case would otherwise measure ~n²).
 steps=sum(r['multi_target_examples']+r['capacity_rejections_kept_single'] for r in receipt.values())
 assert calls<=3*serial_calls+(workers+2)*(serial_calls-steps+sum(r['capacity_rejections_kept_single'] for r in receipt.values())+len(receipt))

def test_forked_packing_matches_serial(monkeypatch):
 items=pack_items(120,3)
 assert packed(monkeypatch,items,4)[:2]==packed(monkeypatch,items,1)[:2]

class RaisingStudy(PackStudy):
 def multi_target_item(self,chosen,*,item_id):
  if any(c.info.get('bad') for c in chosen):raise ValueError('constituent without an AC part')
  return super().multi_target_item(chosen,item_id=item_id)

@pytest.mark.parametrize('bad',[6,5])
def test_speculative_packing_fails_only_where_the_walk_does(monkeypatch,bad):
 # r4 is heavy, so the walk's 3-group (r2,r3,r4) falls back to (r2,r3) and its third group is (r4,r5): the look-ahead,
 # planned without rejections, also measured (r5,r6). A group that cannot combine only matters if the walk takes it.
 items=[PackItem(f'r{i}',{'family':'rag_qa','reader':23_000 if i==4 else 1000,**({'bad':True} if i==bad else {})},[{'role':'user','content':'x'*(i+1)}]) for i in range(7)]
 items+=[PackItem(f'long{i}',{'family':'rag_qa','reader':1000},[{'role':'user','content':'y'*(100+i)}]) for i in range(30)]
 run=lambda workers,pool=None:packed(monkeypatch,items,workers,pool,RaisingStudy,fraction=.083)[:2]
 if bad==6:
  serial=run(1)
  assert [ids for _,ids in serial[0] if ids]==[['r0','r1'],['r2','r3'],['r4','r5']]
  assert run(8,InlinePool)==serial
 else:
  for workers,pool in ((1,None),(8,InlinePool)):
   with pytest.raises(ValueError,match='without an AC part'):run(workers,pool)

def test_item_chunk_returns_failures_without_scan(monkeypatch):
 def make(study,family,row,split):
  if row['segment_id']=='seg2':raise ValueError('bad segment')
  return Namespace(item_id=row['segment_id'])
 monkeypatch.setattr(mix,'make_item',make)
 monkeypatch.setattr(mix,'BUILD_CONTEXT',{'study':None,'rows':{'agent_regular':[(None,regular(i,'submitted'),{}) for i in range(4)]}})
 out=mix.item_chunk(('agent_regular',1,4))
 assert [item and item.item_id for item,_ in out]==['seg1',None,'seg3'] and 'bad segment' in out[1][1]['error']

def test_budget_stop_ignores_failures_past_it():
 entries=[(None,{'id':i},{'ordinal':i}) for i in range(10)]
 failure={'family':'agent_regular','id':'r7','error':'ValueError: bad'}
 results=[(Namespace(item_id=f'r{i}',assistant_token_ids=[[1]*100]),None) if i!=7 else (None,failure) for i in range(10)]
 def rebuild(row):raise ValueError(f'bad row {row["id"]}')
 def kept(budget,scan,failures,rebuild=rebuild):return [item.item_id for item,_ in mix.budgeted_items(iter(results),entries,budget,scan,failures,rebuild)]
 failures=[]
 # 500 tokens stop after 5 rows: row 7, built only because its worker chunk ran past the stop, never counts
 assert kept(500,False,failures)==kept(500,True,failures)==[f'r{i}' for i in range(5)] and failures==[]
 # no budget: without --scan the failing row raises its own error (rebuilt here); under --scan it is recorded and dropped
 with pytest.raises(ValueError,match='bad row 7'):kept(None,False,failures)
 assert kept(None,True,failures)==[f'r{i}' for i in range(10) if i!=7] and failures==[failure]
 with pytest.raises(RuntimeError,match='failed in a worker only'):kept(None,False,[],lambda row:None)
 def stops_at_five():
  yield from results[:5]
  raise AssertionError('waited on a result past the budget stop')
 assert [item.item_id for item,_ in mix.budgeted_items(stops_at_five(),entries,500,False,[],rebuild)]==[f'r{i}' for i in range(5)]

def test_rag_chunk_drops_only_the_failing_row(monkeypatch):
 def build(material,extra):
  assert extra==7                                                                  # every chunk at the whole batch's depth
  if any(m['example']['example_id']=='rag3' for m in material):raise ValueError('unrenderable')
  return [Namespace(item_id=m['example']['example_id']) for m in material]
 monkeypatch.setattr(mix,'condition_rag',lambda item,row,split:item)
 entries=[(None,rag(i),{'ordinal':i}) for i in range(8)]
 monkeypatch.setattr(mix,'BUILD_CONTEXT',{'rag_build':build,'rag_extra':7,'scan':True,'entries':entries})
 out=mix.rag_chunk((0,6))
 assert [item and item.item_id for item,_ in out]==['rag0','rag1','rag2',None,'rag4','rag5']
 assert [failure is None for _,failure in out]==[True,True,True,False,True,True]
 assert out[3][1]['id']==mix.row_identity('rag_qa',rag(3)) and 'unrenderable' in out[3][1]['error']
 assert [item.item_id for item,_ in mix.rag_chunk((6,8))]==['rag6','rag7']
 monkeypatch.setattr(mix,'BUILD_CONTEXT',{'rag_build':build,'rag_extra':7,'scan':False,'entries':entries})
 with pytest.raises(ValueError,match='unrenderable'):mix.rag_chunk((0,6))
