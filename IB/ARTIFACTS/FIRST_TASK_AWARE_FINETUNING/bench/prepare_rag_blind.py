"""Question-blind RAG run: one seeded pass of RAG Q/A whose encoder never sees the question, every item at one fixed compression
ratio, from the mixed run's starting pair (the 75M flat reconstruction pair, fresh optimizer), plus a passage-exposure arm.
CPU only: writes a packet (recipe, items, examples) and its task panel; `run_channel_mixture.py train` runs it unchanged.

Held questions (never trained), all question-blind at the fixed ratio:
  rag_qa        the mixed run's 64 (its task panel), with the mixed panel's own evidence (group rag_qa/uncontextual_1_10);
                also exactly as the mixed panel built them at 1/16, under the mixed panel's own group name and item ids
                (rag_qa/uncontextual), so that curve continues across the two runs;
  rag_exposed / rag_unexposed   --extra-held more, 1:1, drawn with gold passages disjoint from each other, from every
                passage of the 64's evidence and from the reconstruction warm-up corpus the starting pair trained on
                (--seen-gold), split alternately in (source family, seeded) order.
No training row shares a gold passage with a drawn candidate (the few that do are dropped, which also takes them out of the BM25
distractor pool), so an unexposed question's gold passage never reaches training. Every passage of an exposed question's
evidence, except those also in an unexposed or mixed-panel item, trains --repeats times as flat reconstruction at the same ratio.
The reconstruction/exposed and reconstruction/unexposed panel groups re-score each held question's gold passage.
exposure.json records, per held item and group, how many of its passages occur in the training Q/A evidence or were
reconstructed; `reconstruction_overlap` (the train phase's clean_* split) marks items whose gold passage reached training, this
run's or the starting pair's (the reconstruction corpus). The panel rows' headline held NLL averages every panel group (both
ratios and the reconstruction groups): read the groups, not the headline.

  python prepare_rag_blind.py --data DATA --initial-pair PAIR.json --mixed-panel MIXED_TASK_PANEL --seen-gold SEEN.json --output RUN
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import random
import re
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import run_channel_mixture as mix
from run_channel_mixture import (CHUNK,SEED,ForkPool,StageClock,digest,dump,file_sha,load_held,nested_parts,normalized,rag_material,
 rag_renders,reader_tokens,reserved_questions,select_rows,write)

VIEW='summary'   # the question-blind encoder view (qa_encoder_messages' summarize-useful-facts instruction)
HEADER=re.compile(r'(?:^|\n\n)Passage (\d+):\n')


def text_key(text):return hashlib.sha256(text.encode()).hexdigest()


def gold_text(row):
 units=row['source']['units']
 if len(units)!=1:raise ValueError(f'RAG row {row["id"]} has {len(units)} gold units (the bank has exactly one per row)')
 return units[0]['text']


def split_passages(item):
 """The item's evidence passages exactly as material_items joined them, or None when a passage itself contains a
 'Passage k:' header line (the split would be ambiguous)."""
 text=item.info['passages_text'];count=item.info['passages']
 headers=list(HEADER.finditer(text))
 if len(headers)!=count or [int(m.group(1)) for m in headers]!=list(range(1,count+1)) or headers[0].start()!=0:return None
 out=[text[m.end():(headers[i+1].start() if i+1<count else len(text))] for i,m in enumerate(headers)]
 if '\n\n'.join(f'Passage {i+1}:\n{p}' for i,p in enumerate(out))!=text:return None
 return out


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data',type=Path,required=True);p.add_argument('--initial-pair',type=Path,required=True)
 p.add_argument('--mixed-panel',type=Path,required=True,help='the mixed run\'s committed task panel (held.json, panel.json, panel.pkl)')
 p.add_argument('--seen-gold',type=Path,required=True,help='JSON with seen_gold_sha256: bank gold passages occurring in the reconstruction warm-up corpus')
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--total-examples',type=int,default=32_000,help='training examples: Q/A questions plus exposure reconstructions')
 p.add_argument('--extra-held',type=int,default=128);p.add_argument('--repeats',type=int,default=3)
 p.add_argument('--ratio',type=float,default=1/10);p.add_argument('--rag-evidence-tokens',type=int,default=8192)
 p.add_argument('--checkpoint-every',type=int,default=250);p.add_argument('--reporting-interval',type=float,default=.05)
 p.add_argument('--workers',type=int,default=0,help='fork workers (0: every CPU of the allocation; 1: one process)')
 p.add_argument('--compare-recipe',type=Path,help='a recipe.json whose training_config should match except the schedule length')
 a=p.parse_args()
 if a.extra_held%2:p.error('--extra-held must be even (split 1:1)')
 out=a.output;out.mkdir(parents=True,exist_ok=True);panel_dir=out/'task_panel'
 if (out/'prepared.json').exists() or (panel_dir/'panel.json').exists():raise FileExistsError('Packet or panel already committed; use a new output')
 panel_dir.mkdir(exist_ok=True)
 from activation.dataset.dataset_caching import freeze_text_collection,frozen_text_rows
 from activation.ac_model import ActivationContextStudyGenerator,ActivationContextTrainer,ActivationContextTrainingConfig
 from activation.ac_model.ac_model_utils import validate_part_capacity
 from run_qa import harness,AC,material_items
 from run_trajectory_tasks import compact_example
 mix.RAG_VIEW,mix.RATIO=VIEW,a.ratio          # every training and held Q/A and reconstruction item: this view, this ratio
 workers=a.workers or len(os.sched_getaffinity(0))
 if workers>1:
  import torch
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('Parallel preparation needs CUDA_VISIBLE_DEVICES= (a CPU-only job); use --workers 1 otherwise')
  os.environ['TOKENIZERS_PARALLELISM']='false'
 # The bank, frozen, must be the one the mixed panel was drawn from.
 path=out/'rag_qa_snapshot.json'
 snapshot=json.loads(path.read_text()) if path.exists() else freeze_text_collection(a.data/'rag_qa',path)
 snapshots={'rag_qa':snapshot}
 mixed_held,mixed_receipt=load_held(a.mixed_panel)
 if mixed_receipt['snapshots'].get('rag_qa')!=digest(snapshot):raise ValueError('The mixed task panel was drawn from a different RAG bank')
 mixed_ids=json.loads((a.mixed_panel/'held.json').read_text())['rag_qa']['ids']
 with (a.mixed_panel/'panel.pkl').open('rb') as f:mixed_items=[i for i in pickle.load(f)['items'] if i.info['panel_group']=='rag_qa/uncontextual']
 reserved=reserved_questions(snapshots)
 rows={row['id']:row for row,_ in frozen_text_rows(a.data/'rag_qa',snapshot)}
 mixed_rows=[rows[i] for i in mixed_ids]
 by_base={i.item_id.split(':',2)[2]:i for i in mixed_items}
 if sorted(by_base)!=sorted(mixed_ids) or any(normalized(by_base[r['id']].info['question'])!=normalized(r['question']) for r in mixed_rows):
  raise ValueError('Mixed panel items differ from its held.json rows')
 mixed_passages={text_key(x) for i in mixed_items for x in split_passages(i) or ()}
 mixed_unsplit=[i.info['passages_text'] for i in mixed_items if split_passages(i) is None]   # checked by substring instead
 def shown_by_mixed(text):return text_key(text) in mixed_passages or any(text in t for t in mixed_unsplit)
 mixed_gold={text_key(gold_text(r)) for r in mixed_rows}
 pretrained=set(json.loads(a.seen_gold.read_text())['seen_gold_sha256'])   # the starting pair may have reconstructed these
 # Candidates for the extra held questions: seeded order; usable, unreserved, a new question text, a gold passage no other
 # candidate has, that no mixed-panel item shows and that the reconstruction corpus does not contain. A margin covers
 # candidates that fail to render or split.
 ranked=sorted(rows.values(),key=lambda r:(digest((SEED,'rag_blind_held',r['id'])),r['id']))
 candidates=[];taken_questions=set(mixed_held['rag_qa']);taken_gold=set(mixed_gold)
 for row in ranked:
  question=normalized(row['question']);gold=text_key(gold_text(row))
  if question in reserved or question in taken_questions or gold in taken_gold or gold in pretrained or shown_by_mixed(gold_text(row)):continue
  candidates.append(row);taken_questions.add(question);taken_gold.add(gold)
  if len(candidates)==a.extra_held+a.extra_held//4:break
 candidate_gold={text_key(gold_text(r)) for r in candidates}
 # Training rows: every usable row outside the held questions (mixed 64 + candidates), less rows sharing a candidate's gold.
 held={'rag_qa':set(mixed_held['rag_qa'])|{normalized(r['question']) for r in candidates},'trajectory_qa':set()}
 selection=SimpleNamespace(data=a.data,qa_questions=10**9,summary_histories=0,teacher_histories=0,regular_segments=0)
 selected,counts,excluded=select_rows(selection,snapshots,held,families=('rag_qa',))
 eligible=[e for e in selected['rag_qa'] if text_key(gold_text(e[1])) not in candidate_gold]
 excluded['shares_held_gold_passage']=len(selected['rag_qa'])-len(eligible)
 pair=json.loads(a.initial_pair.read_text());h=harness(pair['ac_checkpoint'],pair['reader_checkpoint']);study=ActivationContextStudyGenerator(h,AC)
 if workers>1:torch.set_num_threads(1)   # after harness(), which sets its own count
 kept_candidates,unrenderable=rag_renders(study,[(0,r,None) for r in candidates],True)
 # One BM25 index over every training-eligible row (the mixed run's pool rule: before the render filter) serves the held
 # extras and training.
 retrieval_rows=[r for _,r,_ in eligible]
 eligible,dropped=rag_renders(study,eligible,True)
 excluded['rag_answer_render']=len(dropped)
 _,build=material_items(h,{'train':[],'held':[],'retrieval':rag_material(retrieval_rows)},evidence_tokens=a.rag_evidence_tokens,max_passages=8,builder_only=True)
 material=rag_material([r for _,r,_ in kept_candidates])
 extras=[]
 for item,(_,row,_) in zip(build(material,build.extra_for(material)),kept_candidates,strict=True):
  parts=split_passages(item)
  if parts is None or parts.count(gold_text(row))!=1:continue
  extras.append((item,row,parts))
  if len(extras)==a.extra_held:break
 if len(extras)<a.extra_held:raise ValueError(f'only {len(extras)} usable extra held questions')
 order=sorted(extras,key=lambda e:(e[1]['source_family'],digest((SEED,'rag_blind_split',e[1]['id']))))
 exposed,unexposed=order[0::2],order[1::2]
 # Exposure passages: every passage of an exposed item's evidence not shown by an unexposed or mixed-panel item.
 blocked={text_key(x) for _,_,parts in unexposed for x in parts}
 exposure=[];queued=set()
 for _,_,parts in exposed:
  for x in parts:
   if text_key(x) not in blocked and text_key(x) not in queued and not shown_by_mixed(x):queued.add(text_key(x));exposure.append(x)
 if any(text_key(gold_text(row)) not in queued for _,row,_ in exposed):raise ValueError('an exposed gold passage is blocked')
 recon_rows=[{'text':x,'doc_id':f'rag_exposure:{text_key(x)[:16]}:r{k}','source':'rag_exposure'} for k in range(a.repeats) for x in exposure]
 qa_target=a.total_examples-len(recon_rows)
 entries=eligible[:qa_target+max(64,qa_target//100)]   # a margin for rows whose example fails; cut back to qa_target below
 failures=[];stages={}
 clock=StageClock('rag_qa items',len(entries));done=0;output=[]
 mix.BUILD_CONTEXT={'rag_build':build,'rag_extra':build.extra_for(rag_material([r for _,r,_ in entries])),'scan':True,'entries':entries}
 spans=[(i,min(i+CHUNK,len(entries))) for i in range(0,len(entries),CHUNK)]
 with ForkPool(workers) as pool:
  for span,results in zip(spans,pool.map(mix.rag_chunk,spans)):
   for item,failure in results:
    if failure:failures.append(failure)
    if item is not None:output.append(item)
   done+=span[1]-span[0];clock(done)
 stages['rag_qa_items']=clock.seconds()
 for row in recon_rows:output.append(mix.make_item(study,'reconstruction',row,'train'))
 if any(i.info.get('encoder_conditioning')!=VIEW for i in output if i.info['family']=='rag_qa'):raise ValueError('a training RAG item sees the question')
 config=ActivationContextTrainingConfig(loss_kind='sft',learning_rate_ac=5e-5,learning_rate_target_lora=2e-5,examples_per_update=8,num_epochs=1,completion_samples=0,
  teacher_cache_block_examples=32,logits_chunk_tokens=64,gradient_checkpointing_min_tokens=0,seed=SEED,deterministic=False,
  schedule_total_updates=math.ceil(a.total_examples/8),checkpoint_every_updates=a.checkpoint_every,reporting_interval=a.reporting_interval,shuffle_examples=True)
 trainer=ActivationContextTrainer(h,config)
 clock=StageClock('examples',len(output));done=0;kept=[];examples=[]
 mix.BUILD_CONTEXT={'trainer':trainer,'study':study,'scan':True,'output':output}
 chunks=[range(i,min(i+CHUNK,len(output))) for i in range(0,len(output),CHUNK)]
 with ForkPool(workers) as pool:
  for results in pool.map(mix.example_chunk,chunks):
   for value,failure in results:
    if failure:failures.append(failure);continue
    item,example,_=value;kept.append((item,example))
   done+=len(results);clock(done)
 mix.BUILD_CONTEXT=None;stages['examples']=clock.seconds()
 qa=[pair_ for pair_ in kept if pair_[0].info['family']=='rag_qa'];recon=[pair_ for pair_ in kept if pair_[0].info['family']=='reconstruction']
 if len(recon)!=len(recon_rows):raise ValueError(f'{len(recon_rows)-len(recon)} exposure reconstructions failed: {[f for f in failures if f["family"]=="reconstruction"][:3]}')
 if len(qa)<qa_target:raise ValueError(f'only {len(qa)} Q/A examples built for {qa_target}')
 kept=qa[:qa_target]+recon;train_items=[i for i,_ in kept];examples=[e for _,e in kept]
 total=len(train_items)
 if total!=a.total_examples:raise ValueError(f'{total} training examples, expected {a.total_examples}')
 # Passage overlap of every held item with what trains.
 qa_passages=Counter();ambiguous=0
 for item in train_items:
  if item.info['family']!='rag_qa':continue
  parts=split_passages(item)
  if parts is None:ambiguous+=1;continue
  qa_passages.update({text_key(x) for x in parts})
 qa_gold=Counter(text_key(gold_text(rows[i.item_id])) for i in train_items if i.info['family']=='rag_qa')
 reconstructed={text_key(x) for x in exposure}
 builder=ActivationContextTrainer(h,ActivationContextTrainingConfig(loss_kind='sft',max_example_tokens=1_000_000))
 panel=[];overlap={}
 def add(group,item,row,parts):
  """One panel item: `parts` are its source passages (the evidence, or the gold passage it reconstructs; None if unsplit)."""
  item.item_id=f'panel:{group}:{row["id"]}';item.info.update(panel_group=group,panel=True)
  gold=text_key(gold_text(row));keys=[text_key(x) for x in parts] if parts is not None else None
  stats={'passages':len(keys) if keys is not None else item.info.get('passages'),'split':keys is not None,
   'in_training_qa_evidence':sum(k in qa_passages for k in keys) if keys else None,'reconstructed':sum(k in reconstructed for k in keys) if keys else None,
   'gold_in_training_qa_evidence':gold in qa_passages,'gold_is_training_gold':qa_gold[gold],'gold_reconstructed':gold in reconstructed,
   'gold_in_reconstruction_corpus':gold in pretrained}
  seen=stats['gold_in_training_qa_evidence'] or stats['gold_reconstructed'] or stats['gold_is_training_gold']>0 or stats['gold_in_reconstruction_corpus']
  item.info.update(exposure=stats,reconstruction_overlap=bool(seen))
  overlap[item.item_id]={'group':group,**stats}
  panel.append(item)
 for row in mixed_rows:
  base=by_base[row['id']]
  add('rag_qa/uncontextual',deepcopy(base),row,split_passages(base))   # the mixed panel's own item, id and group (1/16)
  add('rag_qa/uncontextual_1_10',mix.condition_rag(deepcopy(base),row,'held',VIEW),row,split_passages(base))
 for group,chosen in (('rag_exposed/uncontextual',exposed),('rag_unexposed/uncontextual',unexposed)):
  for item,row,parts in chosen:add(group,mix.condition_rag(item,row,'held',VIEW),row,parts)
 for group,chosen in (('reconstruction/exposed',exposed),('reconstruction/unexposed',unexposed)):
  for _,row,_ in chosen:
   add(group,mix.make_item(study,'reconstruction',{'text':gold_text(row),'doc_id':f'rag_held_gold:{row["id"][:16]}','source':'rag_held_gold'},'held'),row,[gold_text(row)])
 # The arm's guarantees, checked on what was built.
 for item_id,stats in overlap.items():
  arm=stats['group'].split('/')[0] if stats['group'].startswith('rag_') else stats['group'].split('/')[1]
  if arm in ('rag_unexposed','unexposed') and (stats['gold_in_training_qa_evidence'] or stats['gold_is_training_gold'] or stats['reconstructed'] or stats['gold_in_reconstruction_corpus']):
   raise ValueError(f'unexposed held item {item_id} reached training: {stats}')
  if arm=='rag_qa' and stats['reconstructed']:raise ValueError(f'mixed-panel item {item_id} has a reconstructed passage')
  if arm in ('rag_exposed','exposed') and not stats['gold_reconstructed']:raise ValueError(f'exposed item {item_id} gold not reconstructed')
 if any(i.info.get('ratio')!=(1/16 if i.info['panel_group']=='rag_qa/uncontextual' else a.ratio) for i in panel if i.info['family']=='rag_qa'):raise ValueError('panel ratio mismatch')
 if len({i.item_id for i in panel})!=len(panel):raise ValueError('Duplicate task-panel item ids')
 held_questions={normalized(i.info['question']) for i in panel if 'question' in i.info}
 if any(normalized(i.info['question']) in held_questions for i in train_items if i.info['family']=='rag_qa'):raise ValueError('a held question trains')
 panel_examples=[]
 for item in panel:
  example=builder.build_example(study.ac_model,item)
  if reader_tokens(builder,study,item)>72000:raise ValueError('Panel reader history exceeds 72000 tokens')
  for request in example.part_requests:validate_part_capacity(study.ac_model,request,65536)
  if nested_parts(item):raise ValueError('Panel item nests an AC part (depth-1 rule)')
  item.info.update(target_tokens=len(example.target_ids),reader_tokens=len(example.student_ids))
  panel_examples.append(compact_example(example))
 write(panel_dir/'held.json',{'rule':'mixed panel\'s 64 RAG questions plus extras drawn in seeded order with gold passages disjoint from each other and from the 64\'s evidence; no training row shares an extra\'s gold passage',
  'rag_qa':{'ids':mixed_ids,'questions':[normalized(r['question']) for r in mixed_rows]},
  'rag_exposed':{'ids':[r['id'] for _,r,_ in exposed],'questions':[normalized(r['question']) for _,r,_ in exposed]},
  'rag_unexposed':{'ids':[r['id'] for _,r,_ in unexposed],'questions':[normalized(r['question']) for _,r,_ in unexposed]},
  'candidates_withheld':[r['id'] for r in candidates]})
 dump(panel_dir/'panel.pkl',{'items':panel,'examples':panel_examples})
 groups=Counter(i.info['panel_group'] for i in panel)
 summary={}
 for stats in overlap.values():
  s=summary.setdefault(stats['group'],Counter());s['items']+=1
  for k in ('passages','in_training_qa_evidence','reconstructed'):s[k]+=stats[k] or 0
  for k in ('gold_in_training_qa_evidence','gold_reconstructed','gold_in_reconstruction_corpus'):s[k]+=bool(stats[k])
  s['gold_is_training_gold']+=stats['gold_is_training_gold']>0;s['unsplit']+=not stats['split']
 summary={g:dict(v) for g,v in sorted(summary.items())}
 write(panel_dir/'panel.json',{'sha256':file_sha(panel_dir/'panel.pkl'),'held_sha256':file_sha(panel_dir/'held.json'),'items':len(panel),'groups':dict(sorted(groups.items())),
  'target_tokens':sum(i.info['target_tokens'] for i in panel),'reader_tokens':sum(i.info['reader_tokens'] for i in panel),
  'snapshots':{'rag_qa':digest(snapshot)},'initial_pair':pair,'seed':SEED,'ratio':a.ratio,'exposure_by_group':summary,
  'mixed_panel':{'path':str(a.mixed_panel.resolve()),'sha256':mixed_receipt['sha256'],'held_sha256':mixed_receipt['held_sha256']}})
 write(out/'exposure.json',{'by_group':summary,'items':overlap,'exposure_passages':len(exposure),'repeats':a.repeats,
  'training_qa_items_unsplit':ambiguous,'training_qa_distinct_passages':len(qa_passages),'mixed_items_unsplit':len(mixed_unsplit),
  'extra_candidates_unrenderable':unrenderable})
 actual=Counter(i.info['family'] for i in train_items)
 recipe={'seed':SEED,'epochs':1,'phase':'Question-blind RAG at 1/%d with passage exposure'%round(1/a.ratio),'ratio':a.ratio,'encoder_view':'uncontextual (the encoder never sees the question)',
  'counts':counts,'actual_train':dict(actual),'unique_sources':dict(actual),'excluded':excluded,
  'target_tokens':dict(Counter({f:sum(i.info['target_tokens'] for i in train_items if i.info['family']==f) for f in actual})),
  'loss_shares':'natural: every example weighs 1','global_batch':8,'exposure':{'passages':len(exposure),'repeats':a.repeats,'examples':len(recon_rows),'by_group':summary},
  'held':{'rule':'task_panel/held.json: the mixed panel\'s 64 plus the extra exposed/unexposed questions; reserved RAG questions never train',
   'panel':str(panel_dir.resolve()),'panel_sha256':file_sha(panel_dir/'panel.pkl'),'held_sha256':file_sha(panel_dir/'held.json')},
  'clean_split':'clean_* in panel rows excludes held items whose gold passage reached training: Q/A evidence or gold, exposure reconstruction, or the starting pair\'s reconstruction corpus',
  'headline_note':'the headline held NLL averages every panel group (both ratios, reconstruction groups): read the groups',
  'rag_evidence_tokens':a.rag_evidence_tokens,'rag_retrieval_rows':len(retrieval_rows),'rag_answer_render_excluded':dropped,'failures':failures,
  'initial_pair':pair,'snapshots':{'rag_qa':digest(snapshot)},'training_config':asdict(config)}
 if a.compare_recipe:
  other=json.loads(a.compare_recipe.read_text())['training_config']
  recipe['training_config_differs_from']={'recipe':str(a.compare_recipe),'keys':{k:[other.get(k),v] for k,v in asdict(config).items() if other.get(k)!=v}}
 for item in train_items:item.weight=1.0
 write(out/'recipe.json',recipe)
 bundle={'train':train_items,'held':[]}
 dump(out/'items.pkl',bundle);dump(out/'prepared_examples.pkl',{**bundle,'examples':examples})
 write(out/'prepared.json',{'recipe_sha256':digest(recipe),'items_sha256':file_sha(out/'items.pkl'),'items_bytes':(out/'items.pkl').stat().st_size,
  'prepared_examples_sha256':file_sha(out/'prepared_examples.pkl'),'validated_capacity':{'reader_tokens':72000,'encoder_tokens':65536},
  'training_examples':total,'held_examples':0,'one_pass':True,'preparation':{'workers':workers,'stage_seconds':stages}})
 print(json.dumps({'actual_train':dict(actual),'panel':dict(groups),'exposure_by_group':summary,'excluded':excluded,'failures':len(failures),
  'differs':recipe.get('training_config_differs_from',{}).get('keys')}),flush=True)

if __name__=='__main__':main()
