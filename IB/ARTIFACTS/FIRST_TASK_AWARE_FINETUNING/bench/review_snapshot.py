"""Review readout of a mixed-run snapshot against the pure frozen model (MIXED_RUN_PLAN.md, 25% review). One GPU, two modes:

  trained  the snapshot (AC + reader LoRA): task-panel and learning-sample NLL by group, each compared with the run's own panel
           row at the same update (reload fidelity), and decoded answers on the first --answers items of every panel group.
  frozen   the pure frozen reader (a fresh LoRA, checked to be exactly zero): the learning sample as is for text-only items and
           under full-context / no-context views for items with AC parts; the task panel under both views; decoded answers under
           both views on the first --answers questions of each family (the contextual and uncontextual groups share questions).

Views are the run's own reference views (`transform_for_eval_reference`, as for its fixed references, which use the starting
reader instead of the frozen one): full context puts the source text inline in the reader's prompt, no context removes the AC
parts. Answers are scored by normalized exact match and token F1.

  python review_snapshot.py --mode trained --snapshot RUN/SNAPSHOTS/epoch_001_010pct --task-panel PANEL --learning LEARN.pkl \
    --run-panels RUN/panels.json --recipe RUN/recipe.json --output out_trained.json
  python review_snapshot.py --mode frozen --task-panel PANEL --learning LEARN.pkl --recipe RUN/recipe.json --output out_frozen.json

A stop checkpoint has no snapshot receipt: pass --ac RUN/AC_MODELS/<ac>/epoch_001_step_N --reader RUN/LORAS/<reader>/epoch_001_step_N
--update N instead of --snapshot. Fidelity then compares with the nearest panel row (its examples-presented count is recorded).

Options for long decodes (all off by default; "recite, then answer": the reader writes a whole article out of its rows):
  --decode-groups G [G ...]  decode exactly these panel groups, also groups without question/answer fields
                             (reconstruction/train_docs); such items are not parsed or scored, their text is kept as written
  --full-completions         keep every completion whole (default: the first 1,500 characters)
  --shard i/n                decode every n-th entry of the decode list, starting at the i-th: n processes split one decode
  --nll-items N              score only the first N panel items (default: all), e.g. 1 in every shard but the first
  --progress-file F          decode one entry per call and append it to F as soon as it is written; a restart with the same
                             F takes the entries already there and decodes the rest
  --decode-batch N           decode N entries per generate call (left-padded batch; default 1: one entry per call, as before)
  --row-cache K              keep the encoder output of the last K distinct articles (part lists) instead of encoding the article
                             again for each of its questions (default 0: off). Either option switches to the batched decode
                             (ActivationContextTrainer._sample_completions_batched): entries with the same article are decoded
                             next to each other, and the progress file gets each finished batch at once
  --compare-sequential N     with --decode-batch / --row-cache: decode the first N entries twice on the loaded models, one at a
                             time and batched, print how many completions and answers differ and the seconds of each path,
                             write the differing entries to --output and stop (the progress file is neither read nor written)
"""
import argparse
import json
import pickle
import re
import time
import unicodedata
from dataclasses import fields
from pathlib import Path

from run_channel_mixture import group_summary,write


def has_parts(item):
 from activation.common.ac_parts import direct_parts
 return bool(direct_parts(item.ac_messages))


def nll_table(summary):return {g:{'items':v['items'],'nll_token':v['nll_token']} for g,v in summary.items()}


def is_question(item):return 'answer' in item.info and 'question' in item.info


def groups_to_decode(by_group,names,answers):
 """{group: its first `answers` items}. No names: the Q/A groups (a reconstruction item has no question or short answer to
 judge). With names: exactly those groups, whatever their items hold."""
 if names:
  missing=[g for g in names if g not in by_group]
  if missing:raise ValueError(f'--decode-groups {missing}: the panel holds {sorted(by_group)}')
  kept={g:by_group[g] for g in names}
 else:kept={g:v for g,v in by_group.items() if all(is_question(i) for i in v)}
 return {g:v[:answers] for g,v in sorted(kept.items())}


def parse_shard(text):
 """'i/n' -> (i, n) with 0 <= i < n."""
 m=re.fullmatch(r'(\d+)/(\d+)',text or '')
 if not m or not int(m.group(1))<int(m.group(2)):raise ValueError(f'--shard {text!r}: write i/n with 0 <= i < n, e.g. 0/2')
 return int(m.group(1)),int(m.group(2))


def take_shard(entries,shard):
 """Every n-th entry starting at the i-th; all of them without a shard."""
 return entries if shard is None else [e for k,e in enumerate(entries) if k%shard[1]==shard[0]]


def progress_key(item,group):return f'{group}\t{item.item_id}'


def read_progress(path):
 """{entry key: completion} of a progress file; a last line cut off by a stop is skipped."""
 done={}
 if path.exists():
  for line in path.read_text().splitlines():
   try:row=json.loads(line)
   except ValueError:continue
   done[row['key']]=row['completion']
 return done


def submitted_answer(parse,completion):
 """The answer of the completion's submit_answer call ('' without one)."""
 return next((str(c['arguments'].get('answer','')) for c in parse(completion)[1] if c['name']=='submit_answer'),'')


def decode_entries(trainer,ac,decode,progress,progress_file,batch,row_cache,stats=None):
 """{entry key: completion} of the decode list through the batched decode. Each finished batch goes to the progress file at
 once (one line per entry, in the order decoded: entries of one article together); entries already there are not decoded."""
 done=read_progress(progress_file) if progress_file else {}
 pending,seen=[],set(done)
 for item,group in decode:
  key=progress_key(item,group)
  if key not in seen:seen.add(key);pending.append((key,item))
 if progress_file:
  progress_file.parent.mkdir(parents=True,exist_ok=True)
  print(f'progress file {progress_file}: {sum(progress_key(i,g) in done for i,g in decode)} of {len(decode)} entries already there',flush=True)
  if progress_file.exists() and progress_file.stat().st_size and not progress_file.read_bytes().endswith(b'\n'):
   with progress_file.open('a') as f:f.write('\n')   # a cut-off last line would swallow the next entry
 clock=[time.monotonic(),0]
 def keep(indexes,samples):
  for k,sample in zip(indexes,samples):done[pending[k][0]]=sample['student']
  if progress_file:
   with progress_file.open('a') as f:
    for k in indexes:f.write(json.dumps({'key':pending[k][0],'completion':done[pending[k][0]]},ensure_ascii=False)+'\n')
  clock[1]+=len(indexes)
  print(f'decoded {clock[1]}/{len(pending)}: {len(indexes)} entries from {pending[indexes[0]][0]!r}, {sum(len(s["student"]) for s in samples)} characters in {time.monotonic()-clock[0]:.0f} s',flush=True)
  clock[0]=time.monotonic()
 trainer._sample_completions_batched(ac,[item for _,item in pending],progress,batch_size=batch,row_cache=row_cache,on_batch=keep,stats=stats)
 return done


def compare_decodes(trainer,ac,parse,entries,progress,batch,row_cache,exact_match):
 """The entries decoded twice on the loaded models: one per call (the path of a run with --progress-file) and through the
 batched decode. One untimed entry / batch first, so neither clock pays for kernel compilation. Returns the counts, the
 seconds of each path and the entries whose completions differ."""
 import torch
 items=[item for item,_ in entries];cuda=torch.cuda.is_available()
 trainer._sample_completions(ac,items[:1],progress);trainer._sample_completions_batched(ac,items[:batch],progress,batch_size=batch,row_cache=row_cache)
 started=time.monotonic();sequential=[trainer._sample_completions(ac,[item],progress)[0]['student'] for item in items];sequential_seconds=time.monotonic()-started
 if cuda:torch.cuda.reset_peak_memory_stats()
 counts={};started=time.monotonic()
 batched=[s['student'] for s in trainer._sample_completions_batched(ac,items,progress,batch_size=batch,row_cache=row_cache,stats=counts)];batched_seconds=time.monotonic()-started
 rows=[]
 for (item,group),one,many in zip(entries,sequential,batched,strict=True):
  answers=[unicodedata.normalize('NFC',submitted_answer(parse,x)) for x in (one,many)];gold=unicodedata.normalize('NFC',item.info['answer']) if is_question(item) else None
  rows.append({'key':progress_key(item,group),'identical':one==many,'same_answer':answers[0]==answers[1],'gold':gold,'sequential_answer':answers[0],'batched_answer':answers[1],
   'exact_match':[exact_match(x,[gold]) for x in answers] if gold is not None else None,'sequential':one,'batched':many})
 judged=[r for r in rows if r['exact_match'] is not None]
 summary={'entries':len(rows),'identical_completions':sum(r['identical'] for r in rows),'differing_completions':sum(not r['identical'] for r in rows),
  'differing_answers':sum(not r['same_answer'] for r in rows),'decode_batch':batch,'row_cache':row_cache,
  'seconds':{'sequential':round(sequential_seconds,1),'batched':round(batched_seconds,1)},'speedup':round(sequential_seconds/max(batched_seconds,1e-9),2),'batched_counts':counts}
 if judged:summary['exact_match']={'sequential':sum(r['exact_match'][0] for r in judged)/len(judged),'batched':sum(r['exact_match'][1] for r in judged)/len(judged)}
 if cuda:summary['batched_gpu_peak_gib']=round(torch.cuda.max_memory_allocated()/2**30,1)
 return {'compare':summary,'differences':[r for r in rows if not r['identical']]}


def group_scores(rows):
 """Scores of the group's Q/A rows; a group of recited texts reports its mean length instead."""
 judged=[x for x in rows if x['exact_match'] is not None]
 if not judged:return {'items':len(rows),'completion_characters':sum(len(x['completion']) for x in rows)/len(rows)}
 return {'items':len(judged),'exact_match':sum(x['exact_match'] for x in judged)/len(judged),'token_f1':sum(x['token_f1'] for x in judged)/len(judged),
  'submitted':sum(x['submitted'] for x in judged)/len(judged)}


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--mode',choices=('trained','frozen'),required=True);p.add_argument('--snapshot',type=Path)
 p.add_argument('--ac',type=Path);p.add_argument('--reader',type=Path);p.add_argument('--update',type=int,help='global update of --ac/--reader')
 p.add_argument('--task-panel',type=Path,required=True);p.add_argument('--learning',type=Path,help='the run\'s learning sample (export_review_items.py); omitted: panel only')
 p.add_argument('--recipe',type=Path,required=True,help='the run\'s recipe.json (training config)');p.add_argument('--run-panels',type=Path)
 p.add_argument('--answers',type=int,default=64);p.add_argument('--max-new-tokens',type=int,default=256);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--decode-groups',nargs='+',help='decode exactly these panel groups, also groups without question/answer fields');p.add_argument('--full-completions',action='store_true',help='keep completions whole')
 p.add_argument('--shard',help='i/n: decode every n-th entry of the decode list, starting at the i-th');p.add_argument('--nll-items',type=int,help='score only the first N panel items')
 p.add_argument('--progress-file',type=Path,help='append each decoded entry here at once; a restart continues from it')
 p.add_argument('--decode-batch',type=int,default=1,help='entries per generate call (default 1: one entry per call, as before)');p.add_argument('--row-cache',type=int,default=0,help='keep the encoder output of the last K distinct articles (default 0: off)')
 p.add_argument('--compare-sequential',type=int,help='decode the first N entries one at a time and batched, report the differences and the seconds, and stop')
 a=p.parse_args()
 try:shard=parse_shard(a.shard) if a.shard else None
 except ValueError as e:p.error(str(e))
 if a.nll_items is not None and a.nll_items<1:p.error('--nll-items is at least 1')
 if a.decode_batch<1 or a.row_cache<0:p.error('--decode-batch is at least 1 and --row-cache at least 0')
 batched=a.decode_batch>1 or a.row_cache>0
 if a.compare_sequential is not None and (a.compare_sequential<1 or not batched):p.error('--compare-sequential N (at least 1) compares with the batched decode: give --decode-batch and/or --row-cache')
 checkpoint=a.ac is not None or a.reader is not None or a.update is not None
 if a.mode=='trained' and (a.snapshot is None)==(not checkpoint):p.error('trained needs --snapshot or --ac/--reader/--update')
 if checkpoint and None in (a.ac,a.reader,a.update):p.error('--ac, --reader and --update go together')
 if a.mode=='frozen' and (a.snapshot or checkpoint):p.error('frozen reads no snapshot or checkpoint')
 if checkpoint and {int(m.group(1)) for m in (re.search(r'step_(\d+)$',str(x)) for x in (a.ac,a.reader)) if m}!={a.update}:
  p.error('--update must match the step_N of both --ac and --reader')
 from activation.ac_model import ActivationContextStudyGenerator,ActivationContextTrainer,ActivationContextTrainingConfig
 from activation.ac_model.ac_model_training import ActivationContextTrainingProgress
 from activation.autotuners import configure_fla_runtime
 from activation.dataset.scoring import exact_match,token_f1
 from run_qa import harness,AC
 started=time.monotonic()
 names={f.name for f in fields(ActivationContextTrainingConfig)}
 config={k:v for k,v in json.loads(a.recipe.read_text())['training_config'].items() if k in names}
 config.update(distributed=False,completion_max_new_tokens=a.max_new_tokens)
 config.update(loss_kind='sft',micro_batch_examples=1)   # the review scores gold NLL and decodes: a KL recipe's stored teacher targets are neither available nor needed here
 recipe_config=json.loads(a.recipe.read_text())['training_config']
 receipt=json.loads((a.snapshot/'snapshot.json').read_text()) if a.snapshot else None
 if a.mode=='trained' and checkpoint:
  receipt={'epoch':1,'step':a.update,'steps':recipe_config['schedule_total_updates'],'global_update':a.update,
   'examples_presented':a.update*config.get('examples_per_update',8),'ac_checkpoint':str(a.ac),'reader_checkpoint':str(a.reader)}
 ac_path,reader_path=((a.snapshot/'ac',a.snapshot/'reader') if a.snapshot else (a.ac,a.reader)) if a.mode=='trained' else (None,None)
 h=harness(ac_path,reader_path)
 study=ActivationContextStudyGenerator(h,AC);ac=study.ac_model
 trainer=ActivationContextTrainer(h,ActivationContextTrainingConfig(**config))
 if a.mode=='frozen':
  # A fresh LoRA adds B·A with B initialized to zero: the reader is then exactly the frozen base.
  lora=h.module_manager.ensure_lora(ac.config.target_model_lora_name)
  lora_b=[(n,t) for n,t in lora.named_parameters() if 'lora_B' in n]
  nonzero=[n for n,t in lora_b if bool(t.detach().abs().max()>0)]
  if not lora_b or nonzero:raise ValueError(f'fresh reader LoRA is not an exact zero: {len(lora_b)} lora_B tensors, nonzero {nonzero[:3]}')
 with (a.task_panel/'panel.pkl').open('rb') as f:panel=pickle.load(f)['items']
 learned={'items':[],'groups':{}}
 if a.learning:
  with a.learning.open('rb') as f:learned=pickle.load(f)
 learning,learning_groups=learned['items'],learned['groups']
 panel_groups={i.item_id:i.info['panel_group'] for i in panel}
 first={}
 for item in panel:first.setdefault(panel_groups[item.item_id],[]).append(item)
 # Only Q/A groups decode (a reconstruction item has no question or short answer to judge) unless --decode-groups names the groups.
 first=groups_to_decode(first,a.decode_groups,a.answers)
 out={'mode':a.mode,'snapshot':receipt,'answers_per_group':a.answers}
 out.update({k:v for k,v in (('decode_groups',a.decode_groups),('shard',a.shard),('nll_items',a.nll_items)) if v is not None})
 if a.mode=='trained':
  panel_eval=[(panel,panel_groups)];learning_eval=[(learning,learning_groups)]
  decode=[(item,panel_groups[item.item_id]) for g in first for item in first[g]]
 else:
  # Reference views keep the item id plus a suffix that group_summary strips: one evaluation per view, groups by base id.
  views=('full_context','no_context');view=lambda items,kind:study.transform_for_eval_reference(items,kind)
  panel_eval=[(view(panel,v),{i.item_id:f'{panel_groups[i.item_id]}|{v}' for i in panel}) for v in views]
  plain=[i for i in learning if not has_parts(i)];parted=[i for i in learning if has_parts(i)]
  learning_eval=[(plain,{i.item_id:learning_groups[i.item_id] for i in plain})]
  # A training item without an in-context history (compaction) has no full-context view.
  learning_eval+=[(view([i for i in parted if v=='no_context' or i.in_context_messages is not None],v),
   {i.item_id:f'{learning_groups[i.item_id]}|{v}' for i in parted}) for v in views]
  # Contextual and uncontextual panel items share their questions: the frozen reader answers each question once per view.
  families={}
  for g,items in first.items():
   family=g.split('/')[0]
   if g.endswith('/swapped'):continue   # a control with another question's evidence (prepare_rag_arms), not the question's own
   if family not in families or g.endswith('/contextual'):families[family]=items
  decode=[(reference,f'{family}|{v}') for family,items in sorted(families.items()) for v in views for reference in view(items,v)]
 decode=take_shard(decode,shard)
 if a.nll_items is not None:panel_eval=[(items[:a.nll_items],groups) for items,groups in panel_eval]
 nll={}
 for label,evaluations in (('panel',panel_eval),('learning',learning_eval)):
  for items,groups in evaluations:
   if items:nll.setdefault(label,{}).update(nll_table(group_summary(trainer.eval(AC,items,release=False)['per_item'],groups)))
 out['nll']=nll;out['eval_seconds']=round(time.monotonic()-started,1)
 if receipt and a.run_panels and a.run_panels.exists():
  rows=json.loads(a.run_panels.read_text())
  row=min(rows,key=lambda r:abs(r['examples presented']-receipt['examples_presented'])) if rows else None
  # Only an exact row checks reload fidelity; a stop a few updates past its panel row compares with training drift.
  if row:
   out['fidelity' if row['examples presented']==receipt['examples_presented'] else 'nearest_row']={
    'row_examples_presented':row['examples presented'],'update_gap':(receipt['examples_presented']-row['examples presented'])//8,'panel':{g:{'review':v['nll_token'],'in_training':row['held by group'].get(g,{}).get('nll_token')} for g,v in nll['panel'].items()},
    'learning':{g:{'review':v['nll_token'],'in_training':row['learning by group'].get(g,{}).get('nll_token')} for g,v in nll.get('learning',{}).items()}}
 chosen=[item for item,_ in decode]
 progress=ActivationContextTrainingProgress(1,1,*((receipt['epoch'],receipt['step'],receipt['steps'],receipt['global_update']) if receipt else (1,0,1,0)),0.,0.)
 decode_started=time.monotonic()
 with configure_fla_runtime(trainer._autotuning_variables(ac)['fla_config']):
  h.module_manager.ensure_lora(ac.config.target_model_lora_name);ac.prepare()
  if a.compare_sequential:
   report=compare_decodes(trainer,ac,study.dialect.parse,decode[:a.compare_sequential],progress,a.decode_batch,a.row_cache,exact_match)
   write(a.output,{**out,**report});print(json.dumps(report['compare']),flush=True);return
  if batched:
   out['decode']={'batch':a.decode_batch,'row_cache':a.row_cache}
   done=decode_entries(trainer,ac,decode,progress,a.progress_file,a.decode_batch,a.row_cache,out['decode']);samples=[{'student':done[progress_key(i,g)]} for i,g in decode]
  elif a.progress_file is None:samples=trainer._sample_completions(ac,chosen,progress)
  else:
   # One entry per call, kept on disk at once: a stopped long decode loses at most the entry it was writing.
   done=read_progress(a.progress_file);a.progress_file.parent.mkdir(parents=True,exist_ok=True);samples=[]
   print(f'progress file {a.progress_file}: {sum(progress_key(i,g) in done for i,g in decode)} of {len(decode)} entries already there',flush=True)
   for number,(item,group) in enumerate(decode,1):
    key=progress_key(item,group)
    if key not in done:
     entry_started=time.monotonic();done[key]=trainer._sample_completions(ac,[item],progress)[0]['student']
     with a.progress_file.open('a') as f:f.write(json.dumps({'key':key,'completion':done[key]},ensure_ascii=False)+'\n')
     print(f'decoded {number}/{len(decode)} {key!r}: {len(done[key])} characters in {time.monotonic()-entry_started:.0f} s',flush=True)
    samples.append({'student':done[key]})
 answers={}
 for sample,(item,group) in zip(samples,decode,strict=True):
  completion=sample['student'] if a.full_completions else sample['student'][:1500]
  if not is_question(item):   # a recited text: kept as written, nothing to parse or judge
   answers.setdefault(group,[]).append({'item_id':item.item_id,'question':'','gold':'','answer':'','submitted':False,'exact_match':None,'token_f1':None,'completion':completion})
   continue
  _,calls=study.dialect.parse(sample['student'])
  prediction=next((str(c['arguments'].get('answer','')) for c in calls if c['name']=='submit_answer'),'')
  gold=unicodedata.normalize('NFC',item.info['answer']);prediction=unicodedata.normalize('NFC',prediction)
  answers.setdefault(group,[]).append({'item_id':item.item_id,'question':item.info['question'],'gold':gold,'answer':prediction,
   'submitted':any(c['name']=='submit_answer' for c in calls),'exact_match':exact_match(prediction,[gold]),'token_f1':token_f1(prediction,[gold]),
   'completion':completion})
 out['answer_scores']={g:group_scores(v) for g,v in sorted(answers.items())}
 out['answers']=answers;out['seconds']={'eval':out.pop('eval_seconds'),'decode':round(time.monotonic()-decode_started,1)}
 write(a.output,out)
 print(json.dumps({k:out[k] for k in ('nll','answer_scores','seconds')}|{k:out[k] for k in ('fidelity','nearest_row') if k in out}),flush=True)

if __name__=='__main__':main()
