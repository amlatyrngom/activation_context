"""Question-blind RAG hillclimb arms (Sept 29): question-blind RAG Q/A kept in company of reconstruction, with the SAME encoder
input for both tasks, from the reconstruction warm-up pair (fresh optimizer), every item at one fixed ratio. CPU only: writes a
packet (recipe, items, examples) and its task panel; `run_channel_mixture.py train` runs it unchanged.

Layout (--layout): `bundle` (default) encodes the joined evidence as one part (~820 rows at 1/10 for 8,192 tokens, far longer than
any warm-up passage); `passages` encodes each passage as its own bare part (exactly a warm-up input, one per "Passage i:" header
in the reader's text). --lr-ac 0 freezes the encoder (the reader LoRA alone learns to read the warm-up's rows).

Encoder input (--encoder-input): `bare` (default) gives every question-blind RAG encoder exactly the warm-up's input, one user
message holding the evidence text ("Passage 1:\\n...\\n\\nPassage 2:\\n..."), and the bundle reconstructions below read the same
message; `summary` keeps the question-blind run's evidence + summarize-useful-facts instruction.

Training examples (--total-examples): Q/A questions, plus --recon-share of the total as reconstruction: exposure reconstructions
(every passage of an exposed held question's evidence, --repeats times, flat, as in the question-blind run) and, for the rest,
bundle reconstructions of training rows not used for Q/A: the reader sees the bundle's rows and reproduces one passage k, drawn
seeded and uniform ("Reproduce Passage k ..."), so the encoder cannot know which passage (or whether a question) will be asked.
Arm options on the Q/A targets (training only; held items keep the gold):
  --teacher-answers FILE  {row id: answer}: the frozen model's own full-context answer replaces the gold (sequence distillation);
                          rows without one keep the gold (counted in the recipe).
  --target-weights FILE   {item id: [w per target token]}: per-token loss weights (evidence weighting) on those Q/A items; every
                          Q/A training item must have weights of its exact target length.
  --exclude-rows-of PKT.. continuation packet for a pair trained on those packets: their training rows (Q/A and bundle) are left
                          out of training here, and the held draw skips any bank row whose gold passage or question they trained
                          on, so every held question is unexposed to that pair too (the first held blocks, held there as well,
                          are unchanged; --blind-held still checks them). Needs each packet's items.pkl (this bank's rows).

Held questions (never trained): the question-blind run's draw (same seeds, same inputs: the mixed panel's 64, then 64 exposed +
64 unexposed extras), plus --extra-unexposed more unexposed questions drawn next in the same order (rag_unexposed2). Panel groups:
  rag_qa/uncontextual_1_10, rag_exposed/uncontextual, rag_unexposed/uncontextual, rag_unexposed2/uncontextual   Q/A, this encoder input
  bundle_recon/original|exposed|unexposed     the held question's own bundle; reproduce its gold passage (information flow)
  bundle_recon_summary/original               the same with the question-blind run's summary encoder input (diagnostic)
  bundle_recon_<other layout>/original         the same in the layout not trained (diagnostic)
  rag_unexposed/swapped                        each unexposed question with the next one's evidence: swapped - correct answer
                                               NLL is how much the reader reads the rows (0 = the shortcut)
  lookup/start|mid|mid_swapped                 cued continuation of the unexposed gold passage from its start / mid-passage /
                                               mid with swapped rows (content-addressed lookup; mid_swapped - mid = its value)

--cite: every Q/A target (training and panel) starts with "Evidence: Passage k." (the gold passage index; chance is 1/passages,
so only reading the rows with the question pays), the index token weighing --cite-weight in training.
  reconstruction/exposed|unexposed            each held gold passage alone, flat (as in the question-blind run)

  python prepare_rag_arms.py --data DATA --initial-pair PAIR.json --mixed-panel MIXED_TASK_PANEL --seen-gold SEEN.json --output RUN
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
from prepare_rag_blind import gold_text,split_passages,text_key

VIEW='summary'   # the question-blind encoder view; --encoder-input bare drops its instruction (run_channel_mixture.BARE_BLIND)
BUNDLE_INSTRUCTION='Reproduce Passage {k} from the passages above exactly, word for word, and stop at its end.'
CITE='Evidence: Passage {k}.'
LOOKUP_INSTRUCTION='Find where the following text occurs in the passages above and continue it exactly, word for word.\nText: {cue}'
LOOKUP_CUE,LOOKUP_TARGET=24,48   # tokens


def cite_target(study,item,k):
 """The Q/A target becomes CITE (the 1-based gold passage index, which only reading the rows with the question can predict)
 followed by the usual submit_answer call (in place): (item, token positions of the index in the target)."""
 from activation.ac_model.ac_model_utils import assistant_target_ids
 from run_qa import answer_positions
 answer=item.info['answer'];cite=CITE.format(k=k)
 output=study.dialect.rendering.assistant_message(cite,[{'id':'answer','name':'submit_answer','arguments':{'answer':answer}}])
 ids=assistant_target_ids(study.tokenizer,output,item.tools,item.info.get('chat_template_kwargs'))
 text=study.tokenizer.decode(ids,skip_special_tokens=False);encoded=study.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
 if encoded['input_ids']!=ids:raise ValueError(f'{item.item_id}: cite target does not round-trip')
 left=text.index(cite)+len(cite)-len(f'{k}.');right=left+len(str(k))
 positions=[i for i,(x,y) in enumerate(encoded['offset_mapping']) if x<right and y>left]
 if not positions:raise ValueError(f'{item.item_id}: no cite token')
 item.in_context_messages[-1]=output;item.ac_messages[-1]=deepcopy(output);item.assistant_token_ids=[ids]
 item.info.update(answer_positions=answer_positions(study.tokenizer,ids,answer),cite=k,cite_positions=positions)
 return item,positions


def lookup_item(study,item,parts,gold_index,start,layout,evidence=None,*,end_cue_tokens=0,mask_stop=False):
 """Cued continuation (content-addressed lookup, never trained): the reader sees the rows (its own evidence or, swapped,
 `evidence`) and LOOKUP_CUE tokens of the gold passage from token `start`, and continues it for LOOKUP_TARGET tokens.
 `end_cue_tokens` > 0 makes it a span: the end cue (the tokens right after the target) is given too and the target's end is
 predictable (diagnostic 'span'); `mask_stop` drops the end-of-turn token from a continuation's target (its stop position is
 unpredictable). None when the gold passage is too short."""
 from activation.ac_model import ActivationContextTrainingItem
 from activation.ac_model.ac_model_study import RECONSTRUCTION_SYSTEM_PROMPT,SPAN_INSTRUCTION
 from activation.ac_model.ac_model_utils import assistant_target_ids,without_stop_token
 if mask_stop and end_cue_tokens:raise ValueError('mask_stop applies to continuations (a span\'s end is predictable)')
 gold=parts[gold_index];offsets=study.tokenizer(gold,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
 if start+LOOKUP_CUE+LOOKUP_TARGET+end_cue_tokens>len(offsets):return None
 cut=lambda t:offsets[t][0] if t<len(offsets) else len(gold)   # character boundaries at token starts (no split characters)
 cue=gold[cut(start):cut(start+LOOKUP_CUE)];target=gold[cut(start+LOOKUP_CUE):cut(start+LOOKUP_CUE+LOOKUP_TARGET)]
 end_cue=gold[cut(start+LOOKUP_CUE+LOOKUP_TARGET):cut(start+LOOKUP_CUE+LOOKUP_TARGET+end_cue_tokens)] if end_cue_tokens else None
 shown=evidence if evidence is not None else parts;ratio=item.info['ratio']
 text='\n\n'.join(f'Passage {i+1}:\n{p}' for i,p in enumerate(shown))
 instruction=LOOKUP_INSTRUCTION.format(cue=cue) if end_cue is None else SPAN_INSTRUCTION.format(cue=cue,end_cue=end_cue)
 system=[{'role':'system','content':RECONSTRUCTION_SYSTEM_PROMPT}];output={'role':'assistant','content':target}
 ids=assistant_target_ids(study.tokenizer,output)
 if mask_stop:ids=without_stop_token(study.tokenizer,ids)
 return ActivationContextTrainingItem(item_id='',kind='reconstruct',
  in_context_messages=system+[{'role':'user','content':f'Passages:\n{text}\n\n{instruction}'},deepcopy(output)],
  ac_messages=system+[{'role':'user','content':rows_content('Passages:\n',shown,ratio,layout,f'\n\n{instruction}')},output],
  in_context_start=2,ac_start=2,assistant_token_ids=[ids],tools=None,dataset_id='rag_lookup' if end_cue is None else 'rag_span',
  info={'family':'reconstruction','diagnostic':'lookup' if end_cue is None else 'span','ratio':ratio,'layout':layout,'passage_index':gold_index,'cue_start':start,
   'swapped':evidence is not None,'source':item.info.get('source'),'encoder_input':'bare','passages_text':text,
   **({'end_cue_tokens':end_cue_tokens} if end_cue else {}),**({'stop_masked':True} if mask_stop else {})})


def rows_content(prefix,passages,ratio,layout,suffix):
 """The reader's user content around the evidence rows. `bundle`: one bare part holding the joined evidence ("Passage 1:\n...");
 `passages`: one bare part per passage (each exactly a warm-up input) with the plain-text "Passage i:" headers between them.
 Either way the text view (parts replaced by their text) is prefix + the joined evidence + suffix."""
 from activation.common.ac_parts import ac_part
 from run_qa import AC
 if layout=='bundle':
  text='\n\n'.join(f'Passage {i+1}:\n{p}' for i,p in enumerate(passages))
  return [{'type':'text','text':prefix},ac_part([{'role':'user','content':text}],AC,ratio),{'type':'text','text':suffix}]
 out=[]
 for i,p in enumerate(passages):
  out+=[{'type':'text','text':(prefix if i==0 else '\n\n')+f'Passage {i+1}:\n'},ac_part([{'role':'user','content':p}],AC,ratio)]
 return out+[{'type':'text','text':suffix}]


def around_rows(item):
 """(prefix, suffix) of a single-part RAG item's user content."""
 from activation.common.ac_parts import is_ac_part
 content=item.ac_messages[item.ac_start-1]['content']
 if len(content)!=3 or not is_ac_part(content[1]) or content[0].get('type')!='text' or content[2].get('type')!='text':raise ValueError(f'{item.item_id}: expected text, one part, text')
 return content[0]['text'],content[2]['text']


def relayout(item,parts,layout,evidence=None):
 """A conditioned bare single-part question-blind RAG item in `layout` (in place; never condition_rag it again: that rewrites
 every part with the whole bundle). `evidence` (passages) replaces its own in both views for a swapped control."""
 prefix,suffix=around_rows(item)
 item.ac_messages[item.ac_start-1]['content']=rows_content(prefix,evidence if evidence is not None else parts,item.info['ratio'],layout,suffix)
 if evidence is not None:
  text='\n\n'.join(f'Passage {i+1}:\n{p}' for i,p in enumerate(evidence))
  item.in_context_messages[item.in_context_start-1]['content']=prefix+text+suffix;item.info.update(passages_text=text,passages=len(evidence))
 item.info['layout']=layout
 return item


def bundle_recon_item(study,item,parts,k,item_id,*,summary=False,layout='bundle'):
 """Reconstruction of passage k (0-based) of a built RAG item's evidence: the encoder reads the evidence in `layout` (bare; or,
 with `summary`, one part holding evidence + the summary instruction), the reader reproduces the one passage. Ratio: the item's."""
 from activation.ac_model import ActivationContextTrainingItem
 from activation.ac_model.ac_model_study import RECONSTRUCTION_SYSTEM_PROMPT
 from activation.ac_model.ac_model_utils import assistant_target_ids,qa_encoder_messages
 from activation.common.ac_parts import ac_part
 from run_qa import AC
 text=item.info['passages_text'];ratio=item.info['ratio']
 evidence=[{'role':'user','content':text}]
 instruction=BUNDLE_INSTRUCTION.format(k=k+1)
 content=([{'type':'text','text':'Passages:\n'},ac_part(qa_encoder_messages(evidence,'',AC,ratio,encoder_sees_question=False),AC,ratio),{'type':'text','text':f'\n\n{instruction}'}]
  if summary else rows_content('Passages:\n',parts,ratio,layout,f'\n\n{instruction}'))
 system=[{'role':'system','content':RECONSTRUCTION_SYSTEM_PROMPT}];output={'role':'assistant','content':parts[k]}
 return ActivationContextTrainingItem(item_id=item_id,kind='reconstruct',
  in_context_messages=system+[{'role':'user','content':f'Passages:\n{text}\n\n{instruction}'},deepcopy(output)],
  ac_messages=system+[{'role':'user','content':content},output],
  in_context_start=2,ac_start=2,assistant_token_ids=[assistant_target_ids(study.tokenizer,output)],tools=None,dataset_id='rag_bundle',
  info={'family':'reconstruction','diagnostic':'bundle_summary' if summary else 'bundle','ratio':ratio,'passage_index':k,'passages':len(parts),'layout':'bundle' if summary else layout,
   'chars':len(parts[k]),'source':item.info.get('source'),'encoder_input':'summary' if summary else 'bare','passages_text':text})


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data',type=Path,required=True);p.add_argument('--initial-pair',type=Path,required=True)
 p.add_argument('--mixed-panel',type=Path,required=True,help='the mixed run\'s committed task panel (held.json, panel.json, panel.pkl)')
 p.add_argument('--seen-gold',type=Path,required=True,help='JSON with seen_gold_sha256: bank gold passages occurring in the reconstruction warm-up corpus')
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--total-examples',type=int,default=6_400);p.add_argument('--recon-share',type=float,default=1/3)
 p.add_argument('--encoder-input',choices=('bare','summary'),default='bare')
 p.add_argument('--layout',choices=('bundle','passages'),default='bundle',help='bundle: one part for the joined evidence; passages: one bare part per passage')
 p.add_argument('--lr-ac',type=float,default=5e-5,help='AC modules + side LoRA learning rate (0 freezes the encoder)')
 p.add_argument('--cite',action='store_true',help='Q/A targets (training and panel) start with "Evidence: Passage k." (the gold passage index)')
 p.add_argument('--cite-weight',type=float,default=5.,help='with --cite: loss weight of the index token in training targets (others 1)')
 p.add_argument('--extra-held',type=int,default=128,help='the question-blind run\'s exposed + unexposed extras (keep 128 to keep its draw)')
 p.add_argument('--extra-unexposed',type=int,default=64);p.add_argument('--repeats',type=int,default=1,help='exposure reconstructions per exposed-evidence passage (0: no exposure arm; smoke tests)')
 p.add_argument('--teacher-answers',type=Path);p.add_argument('--target-weights',type=Path)
 p.add_argument('--blind-held',type=Path,help="the question-blind run's task_panel/held.json: the held draw must reproduce its ids")
 p.add_argument('--exclude-rows-of',type=Path,nargs='+',default=[],metavar='PKT',help='packets whose training rows are excluded from training and whose trained golds/questions the held draw skips (a continuation packet)')
 p.add_argument('--ratio',type=float,default=1/10);p.add_argument('--rag-evidence-tokens',type=int,default=8192)
 p.add_argument('--max-passages',type=int,default=8,help='gold + at most this many minus one BM25 distractors within the evidence budget (1: gold only)')
 p.add_argument('--checkpoint-every',type=int,default=100);p.add_argument('--reporting-interval',type=float,default=.125)
 p.add_argument('--workers',type=int,default=0,help='fork workers (0: every CPU of the allocation; 1: one process)')
 a=p.parse_args()
 if a.extra_held%2:p.error('--extra-held must be even (split 1:1)')
 if not 0<=a.recon_share<1:p.error('--recon-share must be in [0, 1)')
 if a.layout=='passages' and a.encoder_input!='bare':p.error('--layout passages needs --encoder-input bare')
 if a.cite and (a.teacher_answers or a.target_weights):p.error('--cite does not combine with --teacher-answers or --target-weights')
 if a.encoder_input!='bare':p.error('--encoder-input summary is not wired through the bundles, the swapped control and the lookup probe (they read bare evidence)')
 out=a.output;out.mkdir(parents=True,exist_ok=True);panel_dir=out/'task_panel'
 if (out/'prepared.json').exists() or (panel_dir/'panel.json').exists():raise FileExistsError('Packet or panel already committed; use a new output')
 panel_dir.mkdir(exist_ok=True)
 from activation.dataset.dataset_caching import freeze_text_collection,frozen_text_rows
 from activation.ac_model import ActivationContextStudyGenerator,ActivationContextTrainer,ActivationContextTrainingConfig
 from activation.ac_model.ac_model_utils import validate_part_capacity
 from run_qa import harness,AC,material_items
 from run_trajectory_tasks import compact_example
 mix.RAG_VIEW,mix.RATIO,mix.BARE_BLIND=VIEW,a.ratio,a.encoder_input=='bare'   # every training and held item: this view, input and ratio
 workers=a.workers or len(os.sched_getaffinity(0))
 if workers>1:
  import torch
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('Parallel preparation needs CUDA_VISIBLE_DEVICES= (a CPU-only job); use --workers 1 otherwise')
  os.environ['TOKENIZERS_PARALLELISM']='false'
 teacher=json.loads(a.teacher_answers.read_text()) if a.teacher_answers else None
 weights=target_sha=None
 if a.target_weights:
  with a.target_weights.open('rb') as f:signal=pickle.load(f)
  weights,target_sha,weights_hash=signal['weights'],signal['targets'],signal.get('hash','target_ids')
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
 mixed_unsplit=[i.info['passages_text'] for i in mixed_items if split_passages(i) is None]
 def shown_by_mixed(text):return text_key(text) in mixed_passages or any(text in t for t in mixed_unsplit)
 mixed_gold={text_key(gold_text(r)) for r in mixed_rows}
 pretrained=set(json.loads(a.seen_gold.read_text())['seen_gold_sha256'])
 # Rows a previous packet trained on (Q/A item id = row id; a bundle item is bundle:<row id>:p<k>): out of training here, and their
 # golds and questions out of the held draw, so a continuation's held questions are unexposed to the pair trained on those packets.
 # Every passage those items showed (evidence, distractors included) or reproduced (bundle and exposure targets) is skipped too.
 prior_ids=set();prior_gold=set();prior_questions=set();prior_shown=set();prior_unsplit=[];prior_held_ids=set()
 for pkt in a.exclude_rows_of:
  with (pkt/'items.pkl').open('rb') as f:prior_train=pickle.load(f)['train']
  prior_held=json.loads((pkt/'task_panel'/'held.json').read_text())   # its held rows stay eligible for the draw: their golds were reproduced (exposure) or shown there as held, not trained
  prior_held_ids.update(prior_held.get('candidates_withheld',[]),*(v['ids'] for k,v in prior_held.items() if isinstance(v,dict) and 'ids' in v))
  for item in prior_train:
   family=item.info.get('family');diagnostic=item.info.get('diagnostic')
   if family=='rag_qa':rid=item.item_id
   elif diagnostic=='bundle':rid=re.fullmatch(r'bundle:(.+):p\d+',item.item_id).group(1)
   else:rid=None   # exposure reconstructions carry a passage, not a row
   if rid is not None:
    if rid not in rows:raise ValueError(f'{pkt}: training row {rid} is not in this bank')
    prior_ids.add(rid);prior_gold.add(text_key(gold_text(rows[rid])));prior_questions.add(normalized(rows[rid]['question']))
   if item.info.get('passages_text') is not None and item.info.get('passages') is not None:
    parts=split_passages(item)
    if parts is None:prior_unsplit.append(item.info['passages_text'])
    else:prior_shown.update(text_key(x) for x in parts)
   if family!='rag_qa' and isinstance(item.ac_messages[-1]['content'],str):prior_unsplit.append(item.ac_messages[-1]['content'])   # a reconstruction's target text
  print(f'{pkt}: {len(prior_train)} training items; excluding {len(prior_ids)} rows, {len(prior_shown)} shown passages, {len(prior_unsplit)} texts so far',flush=True)
 def shown_by_prior(text):return text_key(text) in prior_shown or any(text in t for t in prior_unsplit)
 # Held candidates exactly as the question-blind run drew them (its first extra_held + extra_held//4 candidates), then the
 # additional unexposed ones, continuing the same seeded order under the same rules.
 ranked=sorted(rows.values(),key=lambda r:(digest((SEED,'rag_blind_held',r['id'])),r['id']))
 candidates=[];taken_questions=set(mixed_held['rag_qa']);taken_gold=set(mixed_gold)
 first_block=a.extra_held+a.extra_held//4;second_block=a.extra_unexposed+a.extra_unexposed//4
 for row in ranked:
  question=normalized(row['question']);gold=text_key(gold_text(row))
  if question in reserved or question in taken_questions or gold in taken_gold or gold in pretrained or shown_by_mixed(gold_text(row)):continue
  if gold in prior_gold or question in prior_questions or row['id'] not in prior_held_ids and (prior_unsplit or prior_shown) and shown_by_prior(gold_text(row)):continue   # trained on, shown or reproduced by the pair this packet continues (its own held rows excepted: reproduced as exposure, never trained)
  candidates.append(row);taken_questions.add(question);taken_gold.add(gold)
  if len(candidates)==first_block+second_block:break
 candidate_gold={text_key(gold_text(r)) for r in candidates}
 held={'rag_qa':set(mixed_held['rag_qa'])|{normalized(r['question']) for r in candidates},'trajectory_qa':set()}
 selection=SimpleNamespace(data=a.data,qa_questions=10**9,summary_histories=0,teacher_histories=0,regular_segments=0)
 selected,counts,excluded=select_rows(selection,snapshots,held,families=('rag_qa',))
 eligible=[e for e in selected['rag_qa'] if text_key(gold_text(e[1])) not in candidate_gold]
 excluded['shares_held_gold_passage']=len(selected['rag_qa'])-len(eligible)
 pair=json.loads(a.initial_pair.read_text());h=harness(pair['ac_checkpoint'],pair['reader_checkpoint']);study=ActivationContextStudyGenerator(h,AC)
 if workers>1:torch.set_num_threads(1)
 kept_first,unrenderable=rag_renders(study,[(0,r,None) for r in candidates[:first_block]],True)
 kept_second,unrenderable_second=rag_renders(study,[(0,r,None) for r in candidates[first_block:]],True)
 retrieval_rows=[r for _,r,_ in eligible]   # the BM25 pool: every training-eligible row, before the render filter (the gold never swaps)
 _,build=material_items(h,{'train':[],'held':[],'retrieval':rag_material(retrieval_rows)},evidence_tokens=a.rag_evidence_tokens,max_passages=a.max_passages,builder_only=True)
 def usable(kept,limit):
  material=rag_material([r for _,r,_ in kept]);found=[]
  for item,(_,row,_) in zip(build(material,build.extra_for(material)),kept,strict=True):
   parts=split_passages(item)
   if parts is None or parts.count(gold_text(row))!=1:continue
   found.append((item,row,parts))
   if len(found)==limit:break
  if len(found)<limit:raise ValueError(f'only {len(found)} usable extra held questions of {limit}')
  return found
 extras=usable(kept_first,a.extra_held)
 order=sorted(extras,key=lambda e:(e[1]['source_family'],digest((SEED,'rag_blind_split',e[1]['id']))))
 exposed,unexposed=order[0::2],order[1::2]
 unexposed2=usable(kept_second,a.extra_unexposed) if a.extra_unexposed else []
 same_evidence=None
 if a.blind_held:
  blind=json.loads(a.blind_held.read_text())
  for name,chosen in (('rag_exposed',exposed),('rag_unexposed',unexposed)):
   if [r['id'] for _,r,_ in chosen]!=blind[name]['ids']:raise ValueError(f"{name}: the held draw differs from the question-blind run's")
  if mixed_ids!=blind['rag_qa']['ids']:raise ValueError("the mixed held ids differ from the question-blind run's")
  # The retrieval pool also leaves out the rag_unexposed2 block, so distractors may differ from the blind run's: recorded.
  with (a.blind_held.parent/'panel.pkl').open('rb') as f:blind_items=pickle.load(f)['items']
  blind_text={(i.info['panel_group'].split('/')[0],i.item_id.split(':',2)[2]):text_key(i.info['passages_text']) for i in blind_items if i.info['panel_group'].split('/')[0] in ('rag_exposed','rag_unexposed')}
  same_evidence={name:sum(blind_text.get((name,r['id']))==text_key(item.info['passages_text']) for item,r,_ in chosen) for name,chosen in (('rag_exposed',exposed),('rag_unexposed',unexposed))}
  print('held evidence identical to the blind run\'s:',same_evidence,flush=True)
 blocked={text_key(x) for _,_,parts in unexposed+unexposed2 for x in parts}
 exposure=[];queued=set()
 for _,_,parts in exposed:
  for x in parts:
   if text_key(x) not in blocked and text_key(x) not in queued and not shown_by_mixed(x):queued.add(text_key(x));exposure.append(x)
 if a.repeats and any(text_key(gold_text(row)) not in queued for _,row,_ in exposed):raise ValueError('an exposed gold passage is blocked')
 recon_rows=[{'text':x,'doc_id':f'rag_exposure:{text_key(x)[:16]}:r{k}','source':'rag_exposure'} for k in range(a.repeats) for x in exposure]
 recon_target=round(a.total_examples*a.recon_share);bundle_target=recon_target-len(recon_rows);qa_target=a.total_examples-recon_target
 if bundle_target<0:raise ValueError(f'{len(recon_rows)} exposure reconstructions exceed the reconstruction share ({recon_target})')
 # Teacher answers replace the gold of training rows before the render check (an answer that cannot render drops its row).
 replaced=0
 if teacher is not None:
  swapped=[]
  for index,row,extra in eligible:
   if row['id'] in teacher and teacher[row['id']]:row=dict(row,answer=teacher[row['id']],gold_answer=row['answer']);replaced+=1
   swapped.append((index,row,extra))
  eligible=swapped
 if prior_ids:before=len(eligible);eligible=[e for e in eligible if e[1]['id'] not in prior_ids];excluded['rows_of_excluded_packets']=before-len(eligible)
 eligible,dropped=rag_renders(study,eligible,True)
 excluded['rag_answer_render']=len(dropped)
 margin=max(64,(qa_target+bundle_target)//50)
 entries=eligible[:qa_target+bundle_target+margin]
 failures=[];stages={}
 clock=StageClock('rag items',len(entries));done=0;built=[]
 mix.BUILD_CONTEXT={'rag_build':build,'rag_extra':build.extra_for(rag_material([r for _,r,_ in entries])),'scan':True,'entries':entries}
 spans=[(i,min(i+CHUNK,len(entries))) for i in range(0,len(entries),CHUNK)]
 with ForkPool(workers) as pool:
  for span,results in zip(spans,pool.map(mix.rag_chunk,spans)):
   for item,failure in results:
    if failure:failures.append(failure)
    if item is not None:built.append(item)
   done+=span[1]-span[0];clock(done)
 stages['rag_items']=clock.seconds()
 # Q/A from the first rows, bundle reconstructions from the next (disjoint rows; a bundle's passage k is drawn seeded).
 qa_items=[];bundle_items=[];unsplit_bundles=0;qa_candidates=0
 for item in built:
  parts=split_passages(item)
  if qa_candidates<qa_target+margin//2:
   # The same filters as the unweighted packet of this layout, so a weighted arm's candidates are that packet's: its frozen
   # signals cover the trained ones (a weightless candidate, a spare, is left out below).
   if (a.layout=='passages' and parts is None) or (a.cite and (parts is None or parts.count(gold_text(rows[item.item_id]))!=1)):
    excluded['qa_unsplit_or_uncitable']=excluded.get('qa_unsplit_or_uncitable',0)+1;continue
   qa_candidates+=1;qa_items.append(item);continue
  if parts is None:unsplit_bundles+=1;continue
  k=random.Random(f'{SEED}:bundle_k:{item.item_id}').randrange(len(parts))
  bundle_items.append(bundle_recon_item(study,item,parts,k,f'bundle:{item.item_id}:p{k+1}',layout=a.layout))
 if len(bundle_items)<bundle_target:raise ValueError(f'only {len(bundle_items)} bundle reconstructions for {bundle_target}')
 bundle_items=bundle_items[:bundle_target+margin//4]   # a spare: the first bundle_target that build are kept
 if weights is not None:
  weighted=[i for i in qa_items if i.item_id in weights];excluded['qa_without_weights']=len(qa_items)-len(weighted);qa_items=weighted
  for item in qa_items:
   # Files from before Sept 29 ~07:40Z hash the target ids alone (recorded in the recipe as weights_hash).
   hashed=[item.assistant_token_ids[0],item.info['passages_text']] if weights_hash=='target_ids+passages_text' else item.assistant_token_ids[0]
   if hashlib.sha256(json.dumps(hashed).encode()).hexdigest()!=target_sha[item.item_id]:
    raise ValueError(f'{item.item_id}: target or evidence differs from the ones the weights were computed on')
   item.info['target_weights']=list(weights[item.item_id])
 if a.layout=='passages':
  if weights is not None and any(split_passages(i) is None for i in qa_items):raise ValueError('a weighted Q/A item cannot take the passages layout')
  qa_items=[relayout(i,split_passages(i),a.layout) for i in qa_items]
 if a.cite:
  for item in qa_items:
   parts=split_passages(item)
   item,positions=cite_target(study,item,parts.index(gold_text(rows[item.item_id]))+1)
   if a.cite_weight!=1:item.info['target_weights']=[a.cite_weight if i in positions else 1. for i in range(len(item.assistant_token_ids[0]))]
 output=qa_items+bundle_items+[mix.make_item(study,'reconstruction',row,'train') for row in recon_rows]
 if any(i.info.get('encoder_conditioning')!=VIEW for i in output if i.info['family']=='rag_qa'):raise ValueError('a training RAG item sees the question')
 if a.encoder_input=='bare' and any(i.info.get('encoder_input')!='bare' for i in output if i.info['family']=='rag_qa'):raise ValueError('a training RAG item is not bare')
 config=ActivationContextTrainingConfig(loss_kind='sft',learning_rate_ac=a.lr_ac,learning_rate_target_lora=2e-5,examples_per_update=8,num_epochs=1,completion_samples=0,
  teacher_cache_block_examples=32,logits_chunk_tokens=64,gradient_checkpointing_min_tokens=0,seed=SEED,deterministic=False,
  schedule_total_updates=math.ceil(a.total_examples/8),checkpoint_every_updates=a.checkpoint_every,reporting_interval=a.reporting_interval,shuffle_examples=True)
 trainer=ActivationContextTrainer(h,config)
 clock=StageClock('examples',len(output));done=0;kept=[]
 mix.BUILD_CONTEXT={'trainer':trainer,'study':study,'scan':True,'output':output}
 chunks=[range(i,min(i+CHUNK,len(output))) for i in range(0,len(output),CHUNK)]
 with ForkPool(workers) as pool:
  for results in pool.map(mix.example_chunk,chunks):
   for value,failure in results:
    if failure:failures.append(failure);continue
    item,example,_=value;kept.append((item,example))
   done+=len(results);clock(done)
 mix.BUILD_CONTEXT=None;stages['examples']=clock.seconds()
 qa=[x for x in kept if x[0].info['family']=='rag_qa']
 bundles=[x for x in kept if x[0].info.get('diagnostic')=='bundle'];flat=[x for x in kept if x[0].info['family']=='reconstruction' and x[0].info.get('diagnostic')!='bundle']
 if len(flat)!=len(recon_rows):raise ValueError(f'{len(recon_rows)-len(flat)} exposure reconstructions failed: {[f for f in failures if f["family"]=="reconstruction"][:3]}')
 if len(bundles)<bundle_target:raise ValueError(f'{bundle_target-len(bundles)} bundle reconstructions short: {failures[:3]}')
 bundles=bundles[:bundle_target]
 if len(qa)<qa_target:raise ValueError(f'only {len(qa)} Q/A examples built for {qa_target}')
 kept=qa[:qa_target]+bundles+flat;train_items=[i for i,_ in kept];examples=[e for _,e in kept]
 if any('target_weights' in i.info and len(i.info['target_weights'])!=len(e.target_ids) for i,e in kept):
  raise ValueError('target weights do not match the Q/A target lengths')
 total=len(train_items)
 if total!=a.total_examples:raise ValueError(f'{total} training examples, expected {a.total_examples}')
 # What trains, by passage: Q/A and bundle evidence (encoder input), reproduced passages (bundle k, exposure), Q/A golds.
 evidence=Counter();ambiguous=0
 for item in train_items:
  if item.info['family']!='rag_qa' and item.info.get('diagnostic')!='bundle':continue
  parts=split_passages(item)
  if parts is None:ambiguous+=1;continue
  evidence.update({text_key(x) for x in parts})
 qa_gold=Counter(text_key(gold_text(rows[i.item_id])) for i in train_items if i.info['family']=='rag_qa')
 reproduced={text_key(x) for x in exposure}|{text_key(i.in_context_messages[-1]['content']) for i in train_items if i.info.get('diagnostic')=='bundle'}
 builder=ActivationContextTrainer(h,ActivationContextTrainingConfig(loss_kind='sft',max_example_tokens=1_000_000))
 panel=[];overlap={}
 def add(group,item,row,parts):
  item.item_id=f'panel:{group}:{row["id"]}';item.info.update(panel_group=group,panel=True)
  gold=text_key(gold_text(row));keys=[text_key(x) for x in parts] if parts is not None else None
  stats={'passages':len(keys) if keys is not None else item.info.get('passages'),'split':keys is not None,
   'in_training_evidence':sum(k in evidence for k in keys) if keys else None,'reproduced':sum(k in reproduced for k in keys) if keys else None,
   'gold_in_training_evidence':gold in evidence,'gold_is_training_gold':qa_gold[gold],'gold_reproduced':gold in reproduced,
   'gold_in_reconstruction_corpus':gold in pretrained}
  seen=stats['gold_in_training_evidence'] or stats['gold_reproduced'] or stats['gold_is_training_gold']>0 or stats['gold_in_reconstruction_corpus']
  item.info.update(exposure=stats,reconstruction_overlap=bool(seen))
  overlap[item.item_id]={'group':group,**stats}
  panel.append(item)
 other='passages' if a.layout=='bundle' else 'bundle'
 def bundle(item,row,parts,summary=False,layout=a.layout):
  return bundle_recon_item(study,item,parts,parts.index(gold_text(row)),'',summary=summary,layout=layout)
 def qa(item,row,parts):
  mix.condition_rag(item,row,'held',VIEW)   # in place: the bundle groups below read its ratio
  out=relayout(deepcopy(item),parts,a.layout) if a.layout!='bundle' else item
  return cite_target(study,out,parts.index(gold_text(row))+1)[0] if a.cite else out
 originals=[]
 for row in mixed_rows:
  base=deepcopy(by_base[row['id']]);parts=split_passages(base)
  if (parts is None and a.layout!='bundle') or (a.cite and (parts is None or parts.count(gold_text(row))!=1)):continue
  add('rag_qa/uncontextual_1_10',qa(base,row,parts),row,parts)
  if parts is not None and parts.count(gold_text(row))==1:originals.append((base,row,parts))
 for group,chosen in (('rag_exposed/uncontextual',exposed),('rag_unexposed/uncontextual',unexposed),('rag_unexposed2/uncontextual',unexposed2)):
  for item,row,parts in chosen:add(group,qa(item,row,parts),row,parts)
 # Swapped control: each unexposed question with the next unexposed question's evidence (same layout): the answer NLL gap
 # swapped - correct is how much the reader uses the rows.
 for index,(item,row,parts) in enumerate(unexposed):
  _,_,evidence=unexposed[(index+1)%len(unexposed)]
  swapped=relayout(mix.condition_rag(deepcopy(item),row,'held',VIEW),parts,a.layout,evidence=evidence)
  if a.cite:cite_target(study,swapped,parts.index(gold_text(row))+1)
  swapped.info.update(swapped_evidence_of=unexposed[(index+1)%len(unexposed)][1]['id'])
  add('rag_unexposed/swapped',swapped,row,evidence)
 for group,chosen in (('bundle_recon/original',originals),('bundle_recon/exposed',exposed),('bundle_recon/unexposed',unexposed)):
  for item,row,parts in chosen:add(group,bundle(item,row,parts),row,parts)
 for item,row,parts in originals:add('bundle_recon_summary/original',bundle(item,row,parts,summary=True),row,parts)
 for item,row,parts in originals:add(f'bundle_recon_{other}/original',bundle(item,row,parts,layout=other),row,parts)
 # Lookup probe on the unexposed golds: cue at the passage start, mid-passage (seeded, from a quarter in), mid with the next
 # unexposed question's rows. Items too short for a mid cue are left out of all three (paired groups).
 for index,(item,row,parts) in enumerate(unexposed):
  k=parts.index(gold_text(row));length=len(study.tokenizer(parts[k],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'])
  if length<2*(LOOKUP_CUE+LOOKUP_TARGET):continue
  mid=random.Random(f'{SEED}:lookup:{row["id"]}').randrange(length//4,length-LOOKUP_CUE-LOOKUP_TARGET+1)
  _,_,evidence=unexposed[(index+1)%len(unexposed)]
  for group,start,shown in (('lookup/start',0,None),('lookup/mid',mid,None),('lookup/mid_swapped',mid,evidence)):
   add(group,lookup_item(study,item,parts,k,start,a.layout,evidence=shown),row,shown if shown is not None else parts)
 for group,chosen in (('reconstruction/exposed',exposed),('reconstruction/unexposed',unexposed)):
  for _,row,_ in chosen:
   add(group,mix.make_item(study,'reconstruction',{'text':gold_text(row),'doc_id':f'rag_held_gold:{row["id"][:16]}','source':'rag_held_gold'},'held'),row,[gold_text(row)])
 for item_id,stats in overlap.items():
  arm=stats['group'].split('/')[1] if stats['group'].startswith(('bundle_recon','reconstruction')) else stats['group'].split('/')[0]
  if arm in ('rag_unexposed','rag_unexposed2','unexposed') and (stats['gold_in_training_evidence'] or stats['gold_is_training_gold'] or stats['gold_reproduced'] or stats['gold_in_reconstruction_corpus']):
   raise ValueError(f'unexposed held item {item_id} reached training: {stats}')
  if a.repeats and arm in ('rag_exposed','exposed') and not stats['gold_reproduced']:raise ValueError(f'exposed item {item_id} gold not reproduced')
 if any(i.info.get('ratio')!=a.ratio for i in panel+train_items):raise ValueError('panel or training ratio mismatch')
 if a.encoder_input=='bare' and any(i.info.get('encoder_input')!='bare' for i in panel if i.info['family']=='rag_qa'):raise ValueError('a panel RAG item is not bare')
 if any(i.info.get('layout',a.layout)!=a.layout for i in panel+train_items if i.info['family']=='rag_qa'):raise ValueError('a RAG item is not in the chosen layout')
 if len({i.item_id for i in panel})!=len(panel):raise ValueError('Duplicate task-panel item ids')
 if any('target_weights' in i.info for i in panel):raise ValueError('a panel item carries training loss weights')
 if a.cite and any('cite' not in i.info for i in panel if i.info['family']=='rag_qa'):raise ValueError('a panel Q/A item has no cite target')
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
 write(panel_dir/'held.json',{'rule':'the question-blind run\'s held draw (mixed panel 64 + exposed/unexposed extras) plus rag_unexposed2 drawn next in the same seeded order; no training row shares a held gold passage',
  'rag_qa':{'ids':mixed_ids,'questions':[normalized(r['question']) for r in mixed_rows]},
  **{name:{'ids':[r['id'] for _,r,_ in chosen],'questions':[normalized(r['question']) for _,r,_ in chosen]} for name,chosen in (('rag_exposed',exposed),('rag_unexposed',unexposed),('rag_unexposed2',unexposed2))},
  'candidates_withheld':[r['id'] for r in candidates]})
 dump(panel_dir/'panel.pkl',{'items':panel,'examples':panel_examples})
 groups=Counter(i.info['panel_group'] for i in panel)
 summary={}
 for stats in overlap.values():
  s=summary.setdefault(stats['group'],Counter());s['items']+=1
  for k in ('passages','in_training_evidence','reproduced'):s[k]+=stats[k] or 0
  for k in ('gold_in_training_evidence','gold_reproduced','gold_in_reconstruction_corpus'):s[k]+=bool(stats[k])
  s['gold_is_training_gold']+=stats['gold_is_training_gold']>0;s['unsplit']+=not stats['split']
 summary={g:dict(v) for g,v in sorted(summary.items())}
 write(panel_dir/'panel.json',{'sha256':file_sha(panel_dir/'panel.pkl'),'held_sha256':file_sha(panel_dir/'held.json'),'items':len(panel),'groups':dict(sorted(groups.items())),
  'target_tokens':sum(i.info['target_tokens'] for i in panel),'reader_tokens':sum(i.info['reader_tokens'] for i in panel),
  'snapshots':{'rag_qa':digest(snapshot)},'initial_pair':pair,'seed':SEED,'ratio':a.ratio,'encoder_input':a.encoder_input,'exposure_by_group':summary,
  'mixed_panel':{'path':str(a.mixed_panel.resolve()),'sha256':mixed_receipt['sha256'],'held_sha256':mixed_receipt['held_sha256']}})
 write(out/'exposure.json',{'by_group':summary,'items':overlap,'exposure_passages':len(exposure),'repeats':a.repeats,
  'training_items_unsplit':ambiguous,'training_distinct_evidence_passages':len(evidence),'mixed_items_unsplit':len(mixed_unsplit),
  'extra_candidates_unrenderable':unrenderable+unrenderable_second,'training_bundles_unsplit':unsplit_bundles,'held_evidence_same_as_blind':same_evidence})
 actual=Counter(mix.panel_group(i) for i in train_items)
 arm={'encoder_input':a.encoder_input,'layout':a.layout,'lr_ac':a.lr_ac,'cite':a.cite_weight if a.cite else None,'recon_share':a.recon_share,'qa_targets':'teacher' if teacher is not None else 'gold',
  'teacher_answers':{'path':str(a.teacher_answers),'sha256':file_sha(a.teacher_answers),'eligible_rows_replaced':replaced,
   'trained_qa_with_teacher':sum(bool(teacher.get(i.item_id)) for i in train_items if i.info['family']=='rag_qa')} if teacher is not None else None,
  'target_weights':{'path':str(a.target_weights),'sha256':file_sha(a.target_weights),'weights_hash':weights_hash} if weights is not None else None}
 recipe={'seed':SEED,'epochs':1,'phase':'Question-blind RAG arm at 1/%d: %s'%(round(1/a.ratio),json.dumps(arm,sort_keys=True)),'ratio':a.ratio,'arm':arm,
  'encoder_view':'uncontextual (the encoder never sees the question); '+('bare evidence, as in reconstruction' if a.encoder_input=='bare' else 'evidence + summary instruction'),
  'counts':counts,'actual_train':dict(actual),'unique_sources':dict(actual),'excluded':excluded,
  'target_tokens':dict(Counter({g:sum(i.info['target_tokens'] for i in train_items if mix.panel_group(i)==g) for g in actual})),
  'loss_shares':'natural: every example weighs 1'+('; Q/A tokens weighted by target_weights' if weights is not None else '')+(f'; the cite index token weighs {a.cite_weight}' if a.cite and a.cite_weight!=1 else ''),'global_batch':8,
  'reconstruction':{'share':a.recon_share,'bundle':len(bundles),'exposure':{'passages':len(exposure),'repeats':a.repeats,'examples':len(recon_rows)}},
  'held':{'rule':'task_panel/held.json','panel':str(panel_dir.resolve()),'panel_sha256':file_sha(panel_dir/'panel.pkl'),'held_sha256':file_sha(panel_dir/'held.json')},
  'clean_split':'clean_* in panel rows excludes held items whose gold passage reached training (evidence, gold, reproduced) or the starting pair\'s reconstruction corpus',
  'headline_note':'the headline held NLL averages every panel group (Q/A and reconstruction): read the groups',
  'rag_evidence_tokens':a.rag_evidence_tokens,'max_passages':a.max_passages,'rag_retrieval_rows':len(retrieval_rows),'rag_answer_render_excluded':dropped,'failures':failures,
  **({'exclude_rows_of':{'packets':[str(x.resolve()) for x in a.exclude_rows_of],'rows':len(prior_ids),'golds':len(prior_gold),'questions':len(prior_questions),'shown_passages':len(prior_shown),'texts':len(prior_unsplit)}} if a.exclude_rows_of else {}),
  'initial_pair':pair,'snapshots':{'rag_qa':digest(snapshot)},'training_config':asdict(config)}
 for item in train_items:item.weight=1.0
 write(out/'recipe.json',recipe)
 bundle_out={'train':train_items,'held':[]}
 dump(out/'items.pkl',bundle_out);dump(out/'prepared_examples.pkl',{**bundle_out,'examples':examples})
 write(out/'prepared.json',{'recipe_sha256':digest(recipe),'items_sha256':file_sha(out/'items.pkl'),'items_bytes':(out/'items.pkl').stat().st_size,
  'prepared_examples_sha256':file_sha(out/'prepared_examples.pkl'),'validated_capacity':{'reader_tokens':72000,'encoder_tokens':65536},
  'training_examples':total,'held_examples':0,'one_pass':True,'preparation':{'workers':workers,'stage_seconds':stages}})
 print(json.dumps({'actual_train':dict(actual),'panel':dict(groups),'exposure_by_group':summary,'excluded':excluded,'failures':len(failures),'arm':arm}),flush=True)

if __name__=='__main__':main()
