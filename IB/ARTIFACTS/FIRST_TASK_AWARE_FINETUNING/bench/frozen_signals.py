"""Frozen-model signals for the question-blind RAG arms (prepare_rag_arms.py), computed on a prepared packet's Q/A training items with
the pure frozen reader (a fresh LoRA, checked to be exactly zero). One GPU; one mode per call:

  weights  per-token evidence weights: the frozen reader scores every Q/A item's target tokens with the full context (the
           evidence as plain text) and with no context; w = FLOOR + max(0, nll_no_context - nll_full_context) per token, so tokens
           the evidence decides dominate the loss and tokens the question alone predicts (format, phrasing) keep a small weight;
           then scaled so the mean weight over all tokens is 1 (the trainer divides by the token count, so an item the question
           alone answers weighs little). Output: pickle {'weights': {item id: [w, ...]}, 'targets': {item id: sha256 of the
           target ids}} plus a JSON summary (the per-token log-probs are kept beside it).
  teacher  the frozen reader's own full-context answer to every Q/A item's question (decoded, the submit_answer argument):
           JSON {row id: answer}; unsubmitted or empty answers are left out (prepare keeps the gold for those rows). Decoded
           in blocks, each appended to teacher_answers_<I>of<N>.partial.jsonl (a restart skips what is there); --shard I/N
           takes every N-th item (sorted by id) from I; --merge N joins N finished shards into teacher_answers.json.

Several --packet folders: the union of their Q/A items by id (the frozen views ignore the encoder layout; an id must carry the
same text view in every packet).

  python frozen_signals.py --mode weights --packet RUN_A0 [RUN_P0 ...] --recipe RUN_A0/recipe.json --output OUT_DIR
  python frozen_signals.py --mode teacher --packet RUN_A0 [RUN_P0 ...] --recipe RUN_A0/recipe.json --output OUT_DIR [--shard 0/2]
  python frozen_signals.py --mode teacher --merge 2 --output OUT_DIR
"""
import argparse
import hashlib
import json
import pickle
import time
import unicodedata
from dataclasses import fields
from pathlib import Path

FLOOR=0.1


def merge(a):
 answers={};summaries=[]
 for i in range(a.merge):
  summaries.append(json.loads((a.output/f'teacher_summary_{i}of{a.merge}.json').read_text()))
  answers.update(json.loads((a.output/f'teacher_answers_{i}of{a.merge}.json').read_text()))
 (a.output/'teacher_answers.json').write_text(json.dumps(answers,ensure_ascii=False,indent=1))
 summary={'mode':'teacher','merged':a.merge,'items':sum(x['items'] for x in summaries),'answers':len(answers),
  'unsubmitted_or_empty':sum(x['unsubmitted_or_empty'] for x in summaries),'max_new_tokens':summaries[0]['max_new_tokens']}
 (a.output/'teacher_summary.json').write_text(json.dumps(summary,indent=1));print(json.dumps(summary),flush=True)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--mode',choices=('weights','teacher'),required=True);p.add_argument('--packet',type=Path,nargs='+')
 p.add_argument('--recipe',type=Path);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--max-new-tokens',type=int,default=1024,help='teacher: the rag_blind frozen review needed ~1024 (a preamble before the call)')
 p.add_argument('--limit',type=int,help='first N Q/A items only (smoke test)')
 p.add_argument('--shard',default='0/1',help='teacher: I/N');p.add_argument('--merge',type=int,help='teacher: join N finished shards')
 p.add_argument('--block',type=int,default=32,help='teacher: items decoded per saved block')
 a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True)
 if a.merge:
  if a.mode!='teacher':p.error('--merge is for teacher shards')
  return merge(a)
 if not a.packet or not a.recipe:p.error('--packet and --recipe are required unless --merge')
 shard,shards=map(int,a.shard.split('/'))
 if not 0<=shard<shards:p.error('--shard I/N needs 0 <= I < N')
 from activation.ac_model import ActivationContextStudyGenerator,ActivationContextTrainer,ActivationContextTrainingConfig
 from activation.ac_model.ac_model_training import ActivationContextTrainingProgress
 from activation.autotuners import configure_fla_runtime
 from run_qa import harness,AC
 started=time.monotonic()
 names={f.name for f in fields(ActivationContextTrainingConfig)}
 config={k:v for k,v in json.loads(a.recipe.read_text())['training_config'].items() if k in names}
 config.update(distributed=False,completion_max_new_tokens=a.max_new_tokens)
 h=harness(None,None);study=ActivationContextStudyGenerator(h,AC);ac=study.ac_model
 trainer=ActivationContextTrainer(h,ActivationContextTrainingConfig(**config))
 lora=h.module_manager.ensure_lora(ac.config.target_model_lora_name)
 lora_b=[(n,t) for n,t in lora.named_parameters() if 'lora_B' in n]
 if not lora_b or any(bool(t.detach().abs().max()>0) for _,t in lora_b):raise ValueError('fresh reader LoRA is not an exact zero')
 items={}
 for packet in a.packet:
  with (packet/'items.pkl').open('rb') as f:
   for item in pickle.load(f)['train']:
    if item.info['family']!='rag_qa':continue
    if item.item_id in items and items[item.item_id].in_context_messages!=item.in_context_messages:raise ValueError(f'{item.item_id}: text views differ between packets')
    if item.item_id not in items and item.info.get('layout','bundle')!='bundle' and packet==a.packet[0]:
     raise ValueError('list a bundle-layout packet first: its no-context view (no per-passage headers) is the one scored')
    items.setdefault(item.item_id,item)
 items=list(items.values())
 if a.limit:items=items[:a.limit]
 if any(i.in_context_messages is None for i in items):raise ValueError('a Q/A item has no full-context view')
 view=lambda chosen,kind='full_context':study.transform_for_eval_reference(chosen,kind)
 if a.mode=='weights':
  logps={}
  for kind in ('full_context','no_context'):
   rows=trainer.eval(AC,view(items,kind),release=False)['per_item']
   by_id={row['item_id'].rsplit(':',1)[0] if row['item_id'].endswith(':'+kind) else row['item_id']:row['target_logp'] for row in rows}
   if len(by_id)!=len(items) or set(by_id)!={i.item_id for i in items}:raise ValueError(f'{kind}: evaluated ids do not match the items')
   logps[kind]=by_id
  weights={};gaps=[];targets={}
  for item in items:
   full,none=logps['full_context'][item.item_id],logps['no_context'][item.item_id]
   if len(full)!=len(none) or len(full)!=len(item.assistant_token_ids[0]):raise ValueError(f'{item.item_id}: target lengths differ between views or from the item')
   gap=[f-n for f,n in zip(full,none)];gaps.extend(gap)
   weights[item.item_id]=[FLOOR+max(0.,g) for g in gap]
   targets[item.item_id]=hashlib.sha256(json.dumps([item.assistant_token_ids[0],item.info['passages_text']]).encode()).hexdigest()
  tokens=sum(len(w) for w in weights.values());scale=tokens/sum(sum(w) for w in weights.values())
  weights={k:[x*scale for x in w] for k,w in weights.items()}
  with (a.output/'target_weights.pkl').open('wb') as f:pickle.dump({'weights':weights,'targets':targets,'hash':'target_ids+passages_text','floor':FLOOR,'scale':scale},f)
  with (a.output/'target_logp.pkl').open('wb') as f:pickle.dump(logps,f)
  total=sum(sum(w) for w in weights.values())
  item_means=sorted(sum(w)/len(w) for w in weights.values())
  summary={'mode':'weights','floor':FLOOR,'scale':scale,'items':len(weights),'tokens':tokens,'mean_weight':total/tokens,
   'item_mean_weight_quantiles':{f:round(item_means[int(f*(len(item_means)-1))],3) for f in (0,.1,.25,.5,.75,.9,1)},
   'share_of_weight_on_top_10pct_tokens':sum(sorted((x for w in weights.values() for x in w),reverse=True)[:tokens//10])/total,
   'mean_gap':sum(gaps)/len(gaps),'positive_gap_tokens':sum(g>0 for g in gaps),
   'nll_token':{k:-sum(sum(v) for v in logps[k].values())/tokens for k in logps},'seconds':round(time.monotonic()-started,1)}
  name='weights_summary.json'
 else:
  items=sorted(items,key=lambda i:i.item_id)[shard::shards];tag=f'_{shard}of{shards}'
  partial=a.output/f'teacher_answers{tag}.partial.jsonl';done={}
  if partial.exists():
   for line in partial.read_text(encoding='utf-8',errors='replace').splitlines():
    try:row=json.loads(line)
    except json.JSONDecodeError:continue   # a block cut by preemption mid-write
    done[row['id']]=row['answer']
  todo=[i for i in items if i.item_id not in done];decoded=0
  progress=ActivationContextTrainingProgress(1,1,1,0,1,0,0.,0.)
  with configure_fla_runtime(trainer._autotuning_variables(ac)['fla_config']):
   ac.prepare()
   for start in range(0,len(todo),a.block):
    block=todo[start:start+a.block];samples=trainer._sample_completions(ac,view(block),progress)
    rows=[]
    for sample,item in zip(samples,block,strict=True):
     _,calls=study.dialect.parse(sample['student'])
     answer=next((str(c['arguments'].get('answer','')) for c in calls if c['name']=='submit_answer'),'')
     rows.append({'id':item.item_id,'answer':unicodedata.normalize('NFC',answer).strip()})
    with partial.open('a',encoding='utf-8') as f:f.write('\n' if partial.exists() and partial.stat().st_size and not partial.read_bytes().endswith(b'\n') else '');f.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    done.update((r['id'],r['answer']) for r in rows);decoded+=len(block)
    print(f'teacher{tag}: {len(done)}/{len(items)} ({decoded} this attempt, {time.monotonic()-started:.0f}s)',flush=True)
  answers={i.item_id:done[i.item_id] for i in items if done.get(i.item_id)}
  (a.output/f'teacher_answers{tag}.json').write_text(json.dumps(answers,ensure_ascii=False,indent=1))
  summary={'mode':'teacher','shard':a.shard,'items':len(items),'answers':len(answers),'unsubmitted_or_empty':len(items)-len(answers),
   'max_new_tokens':a.max_new_tokens,'seconds':round(time.monotonic()-started,1),'decoded_this_attempt':decoded}
  name=f'teacher_summary{tag}.json'
 (a.output/name).write_text(json.dumps(summary,indent=1))
 print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
